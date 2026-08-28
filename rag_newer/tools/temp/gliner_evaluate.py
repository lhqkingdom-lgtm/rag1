# -*- coding: utf-8 -*-
"""
完整版：使用 GLiNER2 进行多标签分类 + 正则补充机制
- 模型预测后，根据正则匹配结果补充一个可能遗漏的类别
- 自动 GPU 加速（如有）
- 进度条显示
- 输出详细评估报告
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"           # 强制离线，完全禁止网络请求
os.environ["HF_HOME"] = "C:/Users/m/.cache/huggingface" 


import json
import os
import re
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from gliner2 import GLiNER2
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
)

# ======================== 配置区域 ========================
JSON_PATH = "classification_result.json"      # 人工标注文件
MD_DIR = "md_files"                           # .md 文件目录
MODEL_NAME = "fastino/gliner2-multi-v1"      # 模型名称
MAX_CHARS = 2000                              # 模型输入截断长度
THRESHOLD = 0.3                               # 模型预测阈值
PATTERN_FILE = "category_patterns.json"      # 正则模式文件

# ---------- 类别描述（提升模型理解）----------
CATEGORY_DESCRIPTIONS = {
    "ADC": "模拟数字转换器，将模拟信号（如电压）转换为数字信号，用于传感器读取",
    "DAC": "数模转换器，将数字值转换为模拟电压或电流输出，用于波形生成、音频输出或控制模拟器件，有的时候也写作‘DAK’",
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
    """加载正则模式文件，返回有序字典（保持 JSON 键顺序）"""
    with open(pattern_file, 'r', encoding='utf-8') as f:
        patterns = json.load(f)
    # Python 3.7+ 字典保持插入顺序，无需额外处理
    # 编译每个类别的所有正则表达式
    compiled = {}
    for cat, pat_list in patterns.items():
        compiled[cat] = [re.compile(p) for p in pat_list]
    return compiled

def count_regex_matches(text, compiled_patterns):
    """
    对给定文本，统计每个类别的正则匹配总次数
    返回字典 {类别: 匹配次数}
    """
    counts = {}
    for cat, regexes in compiled_patterns.items():
        total = 0
        for regex in regexes:
            # 使用 findall 获取所有匹配
            matches = regex.findall(text)
            total += len(matches)
        if total > 0:
            counts[cat] = total
    return counts

def supplement_by_regex(pred_cats, regex_counts, pattern_order):
    """
    根据正则匹配结果补充一个类别到 pred_cats 中
    pred_cats: 模型预测的类别列表
    regex_counts: {类别: 匹配次数}
    pattern_order: 类别列表，按 JSON 键顺序
    返回更新后的 pred_cats（列表）
    """
    if not regex_counts:
        return pred_cats  # 没有匹配，不补充

    # 按匹配次数降序排序，相同次数按 pattern_order 顺序
    sorted_cats = sorted(
        regex_counts.items(),
        key=lambda item: (-item[1], pattern_order.index(item[0]))
    )
    # 遍历，找第一个不在 pred_cats 中的类别
    for cat, _ in sorted_cats:
        if cat not in pred_cats:
            pred_cats.append(cat)
            break
    return pred_cats

# ======================== 主程序 ========================
def main():
    print("=" * 60)
    print("GLiNER2 分类 + 正则补充评估脚本")
    print("=" * 60)

    # 1. 加载正则模式
    print("加载正则模式...")
    compiled_patterns = load_patterns(PATTERN_FILE)
    pattern_order = list(compiled_patterns.keys())
    print(f"已加载 {len(pattern_order)} 个类别的正则表达式")

    # 2. 加载模型（自动选择设备）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    # 强制离线模式（如果已下载，则不会联网）
    os.environ["HF_HUB_OFFLINE"] = "1"
    model = GLiNER2.from_pretrained(MODEL_NAME, device=device)

    # 3. 读取标注 JSON
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        annotation_data = json.load(f)

    file_to_gt = {}          # 文件名 -> 真实类别列表
    all_categories = set()
    for file_info in annotation_data['files']:
        fname = file_info['file']
        cats = file_info['categories']
        file_to_gt[fname] = cats
        all_categories.update(cats)

    category_list = sorted(list(all_categories))
    print(f"共发现 {len(category_list)} 个类别")

    # 检查描述是否覆盖所有类别
    missing_descs = set(category_list) - set(CATEGORY_DESCRIPTIONS.keys())
    if missing_descs:
        print(f"⚠️ 以下类别缺少描述，将使用类别名本身: {missing_descs}")
        for cat in missing_descs:
            CATEGORY_DESCRIPTIONS[cat] = cat

    # 4. 读取 .md 文件（保留完整内容用于正则）
    file_contents = {}   # 文件名 -> 完整内容
    file_contents_truncated = {}  # 文件名 -> 截断后内容（用于模型）
    for fname in file_to_gt.keys():
        md_path = Path(MD_DIR) / fname
        if not md_path.exists():
            print(f"⚠️ 文件不存在，跳过: {md_path}")
            continue
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                content = "（空文档）"
            file_contents[fname] = content
            # 截断用于模型
            if len(content) > MAX_CHARS:
                truncated = content[:MAX_CHARS]
            else:
                truncated = content
            file_contents_truncated[fname] = truncated

    print(f"成功读取 {len(file_contents)} 个文件\n")

    # 5. 构建分类 schema（带描述）
    schema = {
        "categories": {
            "labels": CATEGORY_DESCRIPTIONS,
            "multi_label": True,
            "cls_threshold": THRESHOLD
        }
    }

    # 6. 预测 + 正则补充
    predictions = {}   # 文件名 -> 最终预测类别列表
    print("开始预测与补充...")
    for fname, full_text in tqdm(file_contents.items(), desc="处理进度", unit="file"):
        # 6.1 模型预测
        truncated_text = file_contents_truncated[fname]
        try:
            result = model.classify_text(truncated_text, schema)
            pred_cats = result.get('categories', [])
        except Exception as e:
            print(f"\n预测 {fname} 出错: {e}")
            pred_cats = []

        # 6.2 正则匹配补充
        regex_counts = count_regex_matches(full_text, compiled_patterns)
        # 只考虑 category_list 中的类别（过滤掉可能不在列表中的）
        regex_counts = {k: v for k, v in regex_counts.items() if k in category_list}
        pred_cats = supplement_by_regex(pred_cats, regex_counts, pattern_order)

        predictions[fname] = pred_cats

    # 7. 评估
    print("\n评估中...")
    y_true, y_pred = [], []
    for fname in file_to_gt.keys():
        if fname not in predictions:
            continue
        gt = file_to_gt[fname]
        pred = predictions[fname]
        true_vec = [1 if cat in gt else 0 for cat in category_list]
        pred_vec = [1 if cat in pred else 0 for cat in category_list]
        y_true.append(true_vec)
        y_pred.append(pred_vec)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0:
        print("错误：无评估数据")
        return

    # 详细报告
    print("\n" + "=" * 60)
    print("分类评估报告 (每一类)")
    print("=" * 60)
    report = classification_report(
        y_true, y_pred,
        target_names=category_list,
        zero_division=0,
        digits=4
    )
    print(report)

    # 微平均和宏平均
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    print("\n" + "=" * 60)
    print("整体指标")
    print("=" * 60)
    print(f"微平均 (Micro)  - Precision: {p_micro:.4f}, Recall: {r_micro:.4f}, F1: {f1_micro:.4f}")
    print(f"宏平均 (Macro)  - Precision: {p_macro:.4f}, Recall: {r_macro:.4f}, F1: {f1_macro:.4f}")

    # 8. 保存结果
    output_data = {
        "model": MODEL_NAME,
        "threshold": THRESHOLD,
        "max_chars": MAX_CHARS,
        "category_descriptions": CATEGORY_DESCRIPTIONS,
        "per_file": [
            {
                "file": fname,
                "ground_truth": file_to_gt[fname],
                "predicted": predictions.get(fname, [])
            }
            for fname in file_to_gt.keys() if fname in predictions
        ],
        "metrics": {
            "micro": {"precision": float(p_micro), "recall": float(r_micro), "f1": float(f1_micro)},
            "macro": {"precision": float(p_macro), "recall": float(r_macro), "f1": float(f1_macro)}
        },
        "classification_report_text": report
    }

    output_path = "evaluation_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 结果已保存至: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()