#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量 Markdown 文档处理：清洗 → 分割 → 过滤 → GLiNER2分类 → 正则补充 → 输出 JSON
- 输入：包含多个 .md 文件的文件夹
- 输出：每个 .md 文件生成一个同名子文件夹，内含 chunks/ 和 result.json
- 自动从 chips_map.json 读取芯片列表并填入 result.json
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "C:/Users/m/.cache/huggingface"

import json
import re
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from gliner2 import GLiNER2

# ======================== 导入本地模块 ========================
from tools.md_clean import clean_content
from tools.splitter import split_by_headers


# ======================== 配置区域 ========================
MODEL_NAME = "fastino/gliner2-multi-v1"
MAX_CHARS = 2000
THRESHOLD = 0.3
PATTERN_FILE = "category_patterns.json"


# ======================== 类别描述（提升模型理解） ========================
CATEGORY_DESCRIPTIONS = {
    "ADC": "模拟数字转换器，将模拟信号（如电压）转换为数字信号，用于传感器读取",
    "DAC": "数模转换器，将数字值转换为模拟电压或电流输出，用于波形生成、音频输出或控制模拟器件，有的时候也写作'DAK'",
    "BLE": "蓝牙低功耗通信模块，用于短距离无线数据传输",
    "CAN": "控制器局域网络，一种健壮的串行通信协议，专用于汽车、工业控制等需要高可靠性和实时性的环境",
    "CPU": "中央处理器，芯片的核心计算和控制单元，执行指令",
    "DMA": "直接存储器访问控制器，无需 CPU 干预即可在存储器和外设之间高速传输数据，大幅提升数据传输效率",
    "LCD": "液晶显示驱动器，用于控制外部 LCD 屏幕的显示内容",
    "Low_power_mode": "芯片的低功耗运行模式（如 IDLE、STOP），用于降低系统能耗,TMC相关内容也应归类为低功耗运行模式",
    "Memory": "芯片存储器，包括 Flash 程序存储器和 RAM 数据存储器",
    "Overview": "芯片的整体介绍、主要特性和功能概述",
    "PWM": "脉冲宽度调制，用于电机控制、LED 调光等模拟量输出应用",
    "Power": "电源监测模块，包括 LVD（低电压检测）LDO、POR（上电复位）、BOR（欠压复位）等",
    "RTC": "实时时钟，提供高精度的时间和日历功能，可在低功耗下运行",
    "Touch": "触摸按键检测模块，用于电容触摸传感",
    "UART": "通用异步收发传输器，用于串行通信",
    "SPI": "串行外设接口，用于全双工同步通信",
    "I2C": "两线式串行总线，用于低速外设通信",
    "WDT": "看门狗，以及其他复位系统",
    "WIFI": "Wi-Fi 无线网络通信模块，用于连接互联网或局域网，实现远程数据传输",
    "GPIO": "通用输入输出端口，可配置为输入或输出模式",
    "Timer": "定时器，用于定时计数、PWM生成、输入捕获等功能，值得注意的是TMC也属于定时器",
    "USB": "通用串行总线，用于连接计算机或 USB 设备，支持高速数据传输、设备充电和固件升级",
    "Electrical_parameters": "芯片的电气特性参数，如工作电压、电流等",
    "System_clock": "系统时钟源选择与配置",
    "Interrupt_system": "中断控制系统，管理外设中断优先级和响应",
    "Package_type": "芯片封装类型信息",
    "Routine": "例程或应用示例代码",
    "Other": "其他未分类的内容，一般包括LED,程序下载，在线仿真等内容",
}


# ======================== 工具函数 ========================
def load_patterns(pattern_file):
    """加载正则模式文件，返回编译后的正则字典（按 JSON 键顺序）"""
    if not os.path.exists(pattern_file):
        print(f"⚠️ 正则模式文件 {pattern_file} 不存在，将跳过正则补充")
        return {}, []

    with open(pattern_file, 'r', encoding='utf-8') as f:
        patterns = json.load(f)

    pattern_order = list(patterns.keys())
    compiled = {}
    for cat, pat_list in patterns.items():
        if isinstance(pat_list, str):
            pat_list = [pat_list]
        compiled[cat] = [re.compile(p, re.IGNORECASE) for p in pat_list]
    return compiled, pattern_order


def load_chips_map(chips_file):
    """加载芯片映射 JSON，返回 {文件名: [芯片列表]}"""
    if not chips_file or not os.path.exists(chips_file):
        print("⚠️ 芯片映射文件未提供或不存在，chips 字段将留空")
        return {}
    with open(chips_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_regex_matches(text, compiled_patterns):
    """统计每个类别的正则匹配总次数，仅返回在 CATEGORY_DESCRIPTIONS 中存在的类别"""
    counts = {}
    for cat, regexes in compiled_patterns.items():
        if cat not in CATEGORY_DESCRIPTIONS:
            continue
        total = 0
        for regex in regexes:
            matches = regex.findall(text)
            total += len(matches)
        if total > 0:
            counts[cat] = total
    return counts


def supplement_by_regex(pred_cats, regex_counts, pattern_order):
    """根据正则匹配结果补充一个类别到 pred_cats 中"""
    if not regex_counts:
        return pred_cats

    sorted_cats = sorted(
        regex_counts.items(),
        key=lambda item: (-item[1], pattern_order.index(item[0]) if item[0] in pattern_order else 999)
    )

    for cat, _ in sorted_cats:
        if cat not in pred_cats:
            pred_cats.append(cat)
            break
    return pred_cats


def classify_chunk(content, model, compiled_patterns, pattern_order):
    """对单个分片进行分类：GLiNER2 预测 → 正则补充"""
    # 1. GLiNER2 预测
    truncated = content[:MAX_CHARS] if len(content) > MAX_CHARS else content

    schema = {
        "categories": {
            "labels": CATEGORY_DESCRIPTIONS,
            "multi_label": True,
            "cls_threshold": THRESHOLD
        }
    }

    try:
        result = model.classify_text(truncated, schema)
        pred_cats = result.get('categories', [])
    except Exception as e:
        print(f"⚠️ 模型预测出错: {e}")
        pred_cats = []

    # 2. 正则补充
    if compiled_patterns:
        regex_counts = count_regex_matches(content, compiled_patterns)
        pred_cats = supplement_by_regex(pred_cats, regex_counts, pattern_order)

    # 3. 兜底：如果结果为空，尝试仅用正则
    if not pred_cats and compiled_patterns:
        regex_counts = count_regex_matches(content, compiled_patterns)
        if regex_counts:
            sorted_cats = sorted(regex_counts.items(), key=lambda x: -x[1])
            pred_cats = [sorted_cats[0][0]]

    return sorted(set(pred_cats))


# ======================== 处理单个文件 ========================
def process_single_file(md_path, output_base_dir, model, compiled_patterns, pattern_order,
                        chips_map, cleaning_rules=None):
    """
    处理单个 .md 文件，返回生成的 result 字典
    """
    # 读取原始内容
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 清洗
    cleaned = clean_content(content, cleaning_rules)

    # 2. 分割
    splits = split_by_headers(cleaned)

    # 3. 过滤并分类
    files = []
    valid_idx = 1
    header_pattern = re.compile(r'^(#|##)\s+.*$', re.MULTILINE)

    for split in splits:
        h2 = split["metadata"].get("二级标题", "")
        if not h2 or "目录" in h2:
            continue

        actual_content = header_pattern.sub('', split["content"]).strip()
        if not actual_content:
            continue

        # 分类
        if model is not None:
            categories = classify_chunk(split["content"], model, compiled_patterns, pattern_order)
        else:
            # 仅正则模式
            regex_counts = count_regex_matches(split["content"], compiled_patterns)
            if regex_counts:
                sorted_cats = sorted(regex_counts.items(), key=lambda x: -x[1])
                categories = [cat for cat, _ in sorted_cats]
            else:
                categories = ["Other"]

        categories = sorted(set(categories))

        fname = f"chunk_{valid_idx:03d}.md"
        files.append({
            "file": fname,
            "chips": [],          # 稍后统一填充
            "categories": categories
        })
        valid_idx += 1

    # 如果没有有效分片，生成一个默认的 Other 分片？
    if not files:
        print(f"⚠️ {md_path.name} 无有效分片，生成一个占位分片")
        fname = "chunk_001.md"
        files.append({
            "file": fname,
            "chips": [],
            "categories": ["Other"]
        })
        # 将原始内容写入占位分片（或空内容？为了不丢信息，写入清洗后的内容）
        with open(output_base_dir / md_path.stem / "chunks" / fname, 'w', encoding='utf-8') as f:
            f.write(cleaned)

    # 4. 写入分片文件
    chunks_dir = output_base_dir / md_path.stem / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for entry in files:
        chunk_file = chunks_dir / entry["file"]
        # 需要根据 chunk 编号找到对应的原始 split 内容
        # 由于之前没有保存 split 与文件名的对应关系，这里重新遍历一次（或保存映射）
        # 更稳健：在循环中直接写入，但为了结构清晰，我们重新匹配
        # 简便方法：在循环时同时写入，但需要保存内容，这里重构一下
    # 由于上面的循环没有写入，我们重新处理一次并写入
    # 为了减少重复，重构 process_single_file 内部逻辑，在分类循环中直接写入
    
    # 重构：重新执行一次循环并写入文件
    valid_idx = 1
    for split in splits:
        h2 = split["metadata"].get("二级标题", "")
        if not h2 or "目录" in h2:
            continue
        actual_content = header_pattern.sub('', split["content"]).strip()
        if not actual_content:
            continue

        fname = f"chunk_{valid_idx:03d}.md"
        file_path = chunks_dir / fname
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(split["content"])
        valid_idx += 1

    # 如果没有有效分片，写入占位
    if valid_idx == 1:
        fname = "chunk_001.md"
        file_path = chunks_dir / fname
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)

    # 5. 填充 chips
    file_stem = md_path.stem
    chips_list = chips_map.get(file_stem, [])
    for entry in files:
        entry["chips"] = chips_list

    # 6. 构造 corpus_info
    corpus_info = {
        "name": file_stem,
        "description": f"{file_stem} 中文文档",
        "address": f"/RAG/{file_stem}",
        "version": "1.0",
        "operator": "auto"
    }

    result = {
        "corpus_info": corpus_info,
        "files": files
    }

    return result


# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser(
        description="批量处理 Markdown 文档：清洗 → 分割 → GLiNER2分类 → 正则补充 → 输出 JSON"
    )
    parser.add_argument("input_dir", help="包含 .md 文件的文件夹路径")
    parser.add_argument("--output-dir", default="./output", help="输出父目录（默认 ./output）")
    parser.add_argument("--category-patterns", default="category_patterns.json", help="正则分类规则 JSON 文件")
    parser.add_argument("--cleaning-rules", default="rules.json", help="清洗替换规则 JSON 文件")
    parser.add_argument("--chips-map", default="chips_map.json", help="芯片映射 JSON 文件（键为文件名，值为芯片列表）")
    parser.add_argument("--no-model", action="store_true", help="不使用 GLiNER2 模型，仅使用正则分类")
    parser.add_argument("--model-path", default=MODEL_NAME, help="GLiNER2 模型名称或本地模型目录")
    parser.add_argument("--max-chars", type=int, default=2000, help="模型输入最大字符数")
    parser.add_argument("--threshold", type=float, default=0.3, help="模型分类阈值")
    args = parser.parse_args()

    # 设置全局变量
    global MAX_CHARS, THRESHOLD
    MAX_CHARS = args.max_chars
    THRESHOLD = args.threshold

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"❌ 错误：{input_dir} 不存在或不是文件夹")
        return

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # 加载规则
    cleaning_rules = args.cleaning_rules if os.path.exists(args.cleaning_rules) else None
    compiled_patterns, pattern_order = load_patterns(args.category_patterns)
    chips_map = load_chips_map(args.chips_map)

    # 初始化模型
    model = None
    if not args.no_model:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🚀 使用设备: {device}")
            model = GLiNER2.from_pretrained(args.model_path, device=device)
            print("✅ GLiNER2 模型加载成功")
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}，将仅使用正则分类")
            model = None

    if model is None and not compiled_patterns:
        print("❌ 错误：模型不可用且未提供有效的正则模式文件，无法进行分类。")
        return

    # 获取所有 .md 文件
    md_files = list(input_dir.glob("*.md"))
    if not md_files:
        print(f"❌ 在 {input_dir} 中未找到任何 .md 文件")
        return

    print(f"📁 找到 {len(md_files)} 个 .md 文件，开始批量处理...")

    # 批量处理
    for md_path in tqdm(md_files, desc="总体进度"):
        print(f"\n📄 处理: {md_path.name}")
        try:
            result = process_single_file(
                md_path, output_base, model, compiled_patterns, pattern_order,
                chips_map, cleaning_rules
            )

            # 写入 result.json 到该文件的子目录
            file_output_dir = output_base / md_path.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)
            json_path = file_output_dir / "result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"   ✅ 完成，结果保存在: {json_path}")
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")

    print("\n🎉 所有文件处理完成！")


if __name__ == "__main__":
    main()
