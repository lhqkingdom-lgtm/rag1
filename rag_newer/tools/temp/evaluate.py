#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
整合 v1/v2/v3 召回评估的图形界面工具
- v1/v2：数据库抽样 + API 调用，评估召回率
- v3：批量发送 HTTP 请求（输入 JSON 请求列表，输出响应记录）
- v3 输出可选择文件夹并自定义文件名，默认 output.json
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import random
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Any
from datetime import datetime
from urllib.parse import urljoin

# ======================== 默认配置（v1/v2） ========================

DEFAULT_V1_CONFIG = {
    "db_host": "127.168.1.112",
    "db_port": 5432,
    "db_name": "test",
    "db_user": "postgres",
    "db_password": "88888888",
    "table": "chip_chunks",
    "chunk_id_field": "chunk_id",
    "content_field": "content",
    "index_field": "index",
    "api_base": "http://127.168.1.112:8000",
    "api_endpoint": "/search",
    "k": 100,
    "chunks_per_query": 1,
    "top_k": 10,
}

DEFAULT_V2_CONFIG = {
    "db_host": "192.168.1.115",
    "db_port": 5432,
    "db_name": "cpdb",
    "db_user": "admin",
    "db_password": "miji#123",
    "table": "chunks",
    "chunk_id_field": "vector_id",
    "content_field": "content",
    "api_base": "http://192.168.1.115:8765",
    "api_endpoint": "/kb/retrieve_docs",
    "k": 1000,
    "chunks_per_query": 5,
    "top_k": 10,
}

# v1 的固定分类列表
ALL_CATEGORIES = [
    "pwm", "adc", "dac", "i2c", "uart", "spi", "can", "usb",
    "lcd", "dma", "rtc", "ble", "wifi", "memory", "wdt",
    "gpio", "timer", "interrupt_system", "system_clock",
    "power", "low_power_mode", "touch", "electrical_parameters",
    "package_type", "cpu", "routine", "overview", "other"
]

# ======================== 核心评估函数（v1/v2/v3） ========================

def parse_index(index_val):
    """解析 v1 的 index 字段"""
    if isinstance(index_val, list):
        items = index_val
        if len(items) == 0:
            return [], None
        if len(items) >= 2:
            chips = items[:-1]
            topic = items[-1]
        else:
            chips = items
            topic = None
        return chips, topic
    elif isinstance(index_val, str):
        s = index_val.strip()
        if not s:
            return [], None
        if s.startswith('{') and s.endswith('}'):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(',') if p.strip()]
        if len(parts) >= 2:
            chips = parts[:-1]
            topic = parts[-1]
        elif len(parts) == 1:
            chips = parts
            topic = None
        else:
            chips, topic = [], None
        return chips, topic
    else:
        return [], None

def run_v1_evaluation(params: Dict[str, Any], log_callback, stop_check=None):
    """执行 v1 版评估（拼接查询，按芯片分组）"""
    db_config = {
        "host": params["db_host"],
        "port": int(params["db_port"]),
        "dbname": params["db_name"],
        "user": params["db_user"],
        "password": params["db_password"],
    }
    table = params["table"]
    chunk_id_field = params["chunk_id_field"]
    content_field = params["content_field"]
    index_field = params["index_field"]
    K = int(params["k"])
    chunks_per_query = int(params["chunks_per_query"])
    evaluate_top_k = int(params["top_k"])
    api_base = params["api_base"]
    api_endpoint = params["api_endpoint"]

    log_callback(f"从数据库中随机抽取 {K} 个切片...\n")
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    query = f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT %s"
    cur.execute(query, (K,))
    chunks = cur.fetchall()
    cur.close()
    conn.close()
    log_callback(f"实际抽取到 {len(chunks)} 条记录\n")

    for chunk in chunks:
        index_val = chunk.get(index_field, [])
        chips, topic = parse_index(index_val)
        chunk['chips'] = chips
        chunk['topic'] = topic
        chunk['primary_chip'] = chips[0] if chips else "unknown_chip"
        chunk['query'] = chunk.get(content_field, "")

    groups = defaultdict(list)
    for chunk in chunks:
        groups[chunk['primary_chip']].append(chunk)

    log_callback(f"\n共分为 {len(groups)} 个芯片组：{list(groups.keys())}\n")

    total_hits = 0
    total_chunks = 0

    for chip, chunk_list in groups.items():
        if stop_check and stop_check():
            log_callback("\n用户中断评估。\n")
            return
        log_callback(f"\n========== 芯片组: {chip} (共 {len(chunk_list)} 个切片) ==========\n")
        sub_groups = [chunk_list[i:i+chunks_per_query] for i in range(0, len(chunk_list), chunks_per_query)]
        log_callback(f"划分为 {len(sub_groups)} 个子组（每组最多 {chunks_per_query} 个切片）\n")

        chip_hits = 0
        chip_total = 0

        for g_idx, sub_group in enumerate(sub_groups, 1):
            if stop_check and stop_check():
                log_callback("\n用户中断评估。\n")
                return
            query_text = "\n".join([chunk.get(content_field, "") for chunk in sub_group])
            if not query_text.strip():
                log_callback(f"  子组 {g_idx}: 查询文本为空，跳过\n")
                continue

            expected_ids = {chunk.get(chunk_id_field) for chunk in sub_group if chunk.get(chunk_id_field)}
            if not expected_ids:
                continue

            topics = list({chunk.get('topic') for chunk in sub_group if chunk.get('topic')})
            topics = [t.lower() for t in topics if t.lower() in ALL_CATEGORIES]
            if not topics:
                topics = ["overview"]

            top_k = max(evaluate_top_k, len(sub_group))
            url = f"{api_base}{api_endpoint}"
            payload = {
                "query": query_text,
                "top_k": top_k,
                "chips": [chip],
                "topics": topics
            }
            try:
                resp = requests.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                retrieved_ids = [item.get("chunk_id") for item in results if "chunk_id" in item]
            except Exception as e:
                log_callback(f"  子组 {g_idx}: API 调用失败: {e}\n")
                retrieved_ids = []

            hit_ids = set(retrieved_ids) & expected_ids
            hit_count = len(hit_ids)
            total_in_group = len(expected_ids)

            chip_hits += hit_count
            chip_total += total_in_group
            log_callback(f"  子组 {g_idx}: 共 {total_in_group} 个切片，命中 {hit_count} 个（命中率 {hit_count/total_in_group:.2%}）\n")

        group_recall = chip_hits / chip_total if chip_total > 0 else 0
        log_callback(f"芯片 {chip} 汇总: 命中 {chip_hits}/{chip_total} = {group_recall:.2%}\n")
        total_hits += chip_hits
        total_chunks += chip_total

    overall_recall = total_hits / total_chunks if total_chunks > 0 else 0
    log_callback("\n========== 总体召回评估 ==========\n")
    log_callback(f"总体召回率: {overall_recall:.2%} ({total_hits}/{total_chunks})\n")
    log_callback("评估完成。\n")

def run_v2_evaluation(params: Dict[str, Any], log_callback, stop_check=None):
    """执行 v2 版评估（拼接查询，不区分芯片，直接顺序分组）"""
    db_config = {
        "host": params["db_host"],
        "port": int(params["db_port"]),
        "dbname": params["db_name"],
        "user": params["db_user"],
        "password": params["db_password"],
    }
    table = params["table"]
    chunk_id_field = params["chunk_id_field"]
    content_field = params["content_field"]
    K = int(params["k"])
    chunks_per_query = int(params["chunks_per_query"])
    evaluate_top_k = int(params["top_k"])
    api_base = params["api_base"]
    api_endpoint = params["api_endpoint"]

    log_callback(f"从数据库中随机抽取 {K} 个切片...\n")
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    query = f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT %s"
    cur.execute(query, (K,))
    chunks = cur.fetchall()
    cur.close()
    conn.close()
    log_callback(f"实际抽取到 {len(chunks)} 条记录\n")

    sub_groups = [chunks[i:i+chunks_per_query] for i in range(0, len(chunks), chunks_per_query)]
    log_callback(f"共划分为 {len(sub_groups)} 个查询组（每组最多 {chunks_per_query} 个切片）\n\n")

    total_hits = 0
    total_chunks = 0

    for g_idx, group in enumerate(sub_groups, 1):
        if stop_check and stop_check():
            log_callback("\n用户中断评估。\n")
            return
        query_text = "\n".join([chunk.get(content_field, "") for chunk in group])
        if not query_text.strip():
            log_callback(f"组 {g_idx}: 查询文本为空，跳过\n")
            continue

        expected_ids = {chunk.get(chunk_id_field) for chunk in group if chunk.get(chunk_id_field)}
        if not expected_ids:
            continue

        top_k = max(evaluate_top_k, len(group))
        url = f"{api_base}{api_endpoint}"
        payload = {
            "query": query_text,
            "top_k": top_k,
            "use_rerank": True
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("docs", [])
            retrieved_ids = [doc.get("doc_id") for doc in docs if "doc_id" in doc]
        except Exception as e:
            log_callback(f"组 {g_idx}: API 调用失败: {e}\n")
            retrieved_ids = []

        hit_ids = set(retrieved_ids) & expected_ids
        hit_count = len(hit_ids)
        group_total = len(expected_ids)

        total_hits += hit_count
        total_chunks += group_total
        log_callback(f"组 {g_idx}: 共 {group_total} 个切片，命中 {hit_count} 个（命中率 {hit_count/group_total:.2%}）\n")

    overall_recall = total_hits / total_chunks if total_chunks > 0 else 0
    log_callback("\n========== 总体召回评估 ==========\n")
    log_callback(f"总体召回率: {overall_recall:.2%} ({total_hits}/{total_chunks})\n")
    log_callback("评估完成。\n")

def run_v3_evaluation(params: Dict[str, Any], log_callback, stop_check=None):
    """
    v3：通用 HTTP 请求发送器
    参数：input_file, output_dir, output_name, base_url, endpoint
    """
    input_file = params["input_file"]
    output_dir = params["output_dir"]
    output_name = params["output_name"]
    base_url = params.get("base_url", "").strip()
    endpoint = params.get("endpoint", "").strip()
    # 拼接完整 URL（base_url + endpoint）
    full_base = base_url if base_url else ""
    if endpoint:
        full_url = urljoin(full_base, endpoint) if full_base else endpoint
    else:
        full_url = full_base

    output_path = os.path.join(output_dir, output_name)

    # 读取输入
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            requests_list = json.load(f)
    except Exception as e:
        log_callback(f"读取输入文件失败: {e}\n")
        return

    if not isinstance(requests_list, list):
        log_callback("输入文件必须包含一个请求对象列表\n")
        return

    total = len(requests_list)
    log_callback(f"共 {total} 个请求待发送\n")
    if full_base:
        log_callback(f"Base URL: {full_base}\n")
    if endpoint:
        log_callback(f"Endpoint: {endpoint}\n")

    results = []
    for idx, req in enumerate(requests_list, 1):
        if stop_check and stop_check():
            log_callback("\n用户中断评估。\n")
            break

        log_callback(f"正在发送请求 [{idx}/{total}] ... ")
        # 如果请求中 url 是相对路径，且提供了 full_base，则拼接；否则使用原 url
        # 注意：send_raw_request 内部会用 urljoin(base_url, req_url)
        # 这里我们把 full_base 作为 base_url 传入，但请求中的 url 可能已经是完整路径，若 base_url 非空则拼接
        record = send_raw_request(req, base_url=full_base if full_base else None)
        results.append(record)

        if record['success']:
            status = record['response'].get('status_code')
            log_callback(f"状态码: {status}\n")
        else:
            err = record['response'].get('error', '未知错误')
            log_callback(f"失败: {err[:50]}...\n")

    # 保存结果
    try:
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log_callback(f"\n✅ 记录已保存至: {output_path}\n")
    except Exception as e:
        log_callback(f"写入输出文件失败: {e}\n")

    log_callback("v3 评估完成。\n")

def send_raw_request(item, base_url=None):
    """v3 的核心请求发送函数"""
    method = item.get('method', 'POST').upper()
    url = item.get('url', '')
    if base_url:
        url = urljoin(base_url, url)

    headers = item.get('headers', {})
    if 'Content-Type' not in headers and 'content-type' not in headers:
        headers['Content-Type'] = 'application/json'

    body = item.get('body')

    request_record = {
        'method': method,
        'url': url,
        'headers': headers,
        'body': body
    }

    start_time = datetime.now()
    try:
        if body is not None and isinstance(body, dict):
            resp = requests.request(method, url, headers=headers, json=body, timeout=30)
        else:
            resp = requests.request(method, url, headers=headers, data=body, timeout=30)

        elapsed = resp.elapsed.total_seconds()
        raw_text = resp.text
        try:
            json_body = resp.json()
        except json.JSONDecodeError:
            json_body = None

        response_record = {
            'status_code': resp.status_code,
            'headers': dict(resp.headers),
            'body_raw': raw_text,
            'body_json': json_body,
            'elapsed': elapsed
        }
        success = True
    except requests.RequestException as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        response_record = {
            'status_code': None,
            'headers': {},
            'body_raw': None,
            'body_json': None,
            'elapsed': elapsed,
            'error': str(e)
        }
        success = False

    return {
        'timestamp': start_time.isoformat(),
        'success': success,
        'request': request_record,
        'response': response_record
    }

# ======================== 图形界面 ========================

class EvaluateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("召回评估工具 (v1/v2/v3)")
        self.root.geometry("950x750")

        self.running = False
        self.thread = None

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.v1_frame = ttk.Frame(self.notebook)
        self.v2_frame = ttk.Frame(self.notebook)
        self.v3_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.v1_frame, text="v1 (芯片分组)")
        self.notebook.add(self.v2_frame, text="v2 (顺序分组)")
        self.notebook.add(self.v3_frame, text="v3 (通用HTTP)")

        self.build_v1_page()
        self.build_v2_page()
        self.build_v3_page()

        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=10)
        self.start_btn = ttk.Button(control_frame, text="开始评估", command=self.start_evaluation)
        self.start_btn.grid(row=0, column=0, padx=5)
        self.stop_btn = ttk.Button(control_frame, text="停止", command=self.stop_evaluation, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5)

        log_frame = ttk.LabelFrame(main_frame, text="评估日志", padding="5")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=100, height=25)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

    # ---------- v1 页面 ----------
    def build_v1_page(self):
        frame = self.v1_frame
        entries = {}
        row = 0
        param_defs = [
            ("数据库主机", "db_host"), ("端口", "db_port"), ("数据库名", "db_name"),
            ("用户名", "db_user"), ("密码", "db_password"), ("表名", "table"),
            ("切片ID字段", "chunk_id_field"), ("内容字段", "content_field"),
            ("索引字段 (v1专用)", "index_field"), ("API基础URL", "api_base"),
            ("API端点", "api_endpoint"), ("随机抽取切片数 (K)", "k"),
            ("每组拼接切片数", "chunks_per_query"), ("返回最大结果数 (top_k)", "top_k")
        ]
        for label, key in param_defs:
            ttk.Label(frame, text=label+":").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(value=str(DEFAULT_V1_CONFIG.get(key, "")))
            entry = ttk.Entry(frame, textvariable=var, width=40)
            entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            entries[key] = var
            row += 1
        self.v1_entries = entries

    # ---------- v2 页面 ----------
    def build_v2_page(self):
        frame = self.v2_frame
        entries = {}
        row = 0
        param_defs = [
            ("数据库主机", "db_host"), ("端口", "db_port"), ("数据库名", "db_name"),
            ("用户名", "db_user"), ("密码", "db_password"), ("表名", "table"),
            ("切片ID字段", "chunk_id_field"), ("内容字段", "content_field"),
            ("API基础URL", "api_base"), ("API端点", "api_endpoint"),
            ("随机抽取切片数 (K)", "k"), ("每组拼接切片数", "chunks_per_query"),
            ("返回最大结果数 (top_k)", "top_k")
        ]
        for label, key in param_defs:
            ttk.Label(frame, text=label+":").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(value=str(DEFAULT_V2_CONFIG.get(key, "")))
            entry = ttk.Entry(frame, textvariable=var, width=40)
            entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            entries[key] = var
            row += 1
        self.v2_entries = entries

    # ---------- v3 页面 ----------
    def build_v3_page(self):
        frame = self.v3_frame
        entries = {}
        row = 0

        # 输入文件（选择文件）
        ttk.Label(frame, text="输入 JSON 文件:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        input_var = tk.StringVar()
        entry_input = ttk.Entry(frame, textvariable=input_var, width=30)
        entry_input.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        btn_input = ttk.Button(frame, text="浏览...", command=lambda: self.browse_input(input_var))
        btn_input.grid(row=row, column=2, padx=5)
        entries["input_file"] = input_var
        row += 1

        # 输出文件夹（选择目录）
        ttk.Label(frame, text="输出文件夹:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        output_dir_var = tk.StringVar(value=".")
        entry_output_dir = ttk.Entry(frame, textvariable=output_dir_var, width=30)
        entry_output_dir.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        btn_output_dir = ttk.Button(frame, text="浏览...", command=lambda: self.browse_folder(output_dir_var))
        btn_output_dir.grid(row=row, column=2, padx=5)
        entries["output_dir"] = output_dir_var
        row += 1

        # 输出文件名
        ttk.Label(frame, text="输出文件名:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        output_name_var = tk.StringVar(value="output.json")
        entry_output_name = ttk.Entry(frame, textvariable=output_name_var, width=30)
        entry_output_name.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        entries["output_name"] = output_name_var
        row += 1

        # API Base URL
        ttk.Label(frame, text="API Base URL:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        base_var = tk.StringVar(value="http://192.168.1.115:8765")
        entry_base = ttk.Entry(frame, textvariable=base_var, width=40)
        entry_base.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        entries["base_url"] = base_var
        row += 1

        # API 端点
        ttk.Label(frame, text="API 端点 (可选):").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        endpoint_var = tk.StringVar(value="/kb/retrieve_docs")
        entry_endpoint = ttk.Entry(frame, textvariable=endpoint_var, width=40)
        entry_endpoint.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
        entries["endpoint"] = endpoint_var
        row += 1

        # 预览按钮
        self.preview_btn = ttk.Button(frame, text="预览输入文件", command=self.preview_input)
        self.preview_btn.grid(row=row, column=0, columnspan=2, pady=10)

        self.v3_entries = entries

    def browse_input(self, var):
        path = filedialog.askopenfilename(title="选择输入 JSON 文件", filetypes=[("JSON files", "*.json")])
        if path:
            var.set(path)

    def browse_folder(self, var):
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            var.set(path)

    def preview_input(self):
        input_path = self.v3_entries["input_file"].get().strip()
        if not input_path:
            messagebox.showwarning("预览", "请先选择输入文件")
            return
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                messagebox.showerror("预览", "输入文件必须包含一个请求列表")
                return
            preview = f"共 {len(data)} 个请求，前5个摘要：\n"
            for i, req in enumerate(data[:5], 1):
                method = req.get('method', 'POST')
                url = req.get('url', '')
                preview += f"{i}. {method} {url}\n"
            messagebox.showinfo("预览", preview)
        except Exception as e:
            messagebox.showerror("预览错误", str(e))

    # ---------- 日志与启动/停止 ----------
    def log(self, message):
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def start_evaluation(self):
        tab_id = self.notebook.index(self.notebook.select())
        if tab_id == 0:
            self.run_v1()
        elif tab_id == 1:
            self.run_v2()
        else:
            self.run_v3()

    def run_v1(self):
        params = {}
        for key, var in self.v1_entries.items():
            params[key] = var.get().strip()
        required = ["db_host", "db_port", "db_name", "db_user", "db_password", "table",
                    "chunk_id_field", "content_field", "index_field", "api_base",
                    "api_endpoint", "k", "chunks_per_query", "top_k"]
        for key in required:
            if not params.get(key):
                messagebox.showerror("参数错误", f"请填写 {key}")
                return
        self._start_thread(run_v1_evaluation, params)

    def run_v2(self):
        params = {}
        for key, var in self.v2_entries.items():
            params[key] = var.get().strip()
        required = ["db_host", "db_port", "db_name", "db_user", "db_password", "table",
                    "chunk_id_field", "content_field", "api_base", "api_endpoint",
                    "k", "chunks_per_query", "top_k"]
        for key in required:
            if not params.get(key):
                messagebox.showerror("参数错误", f"请填写 {key}")
                return
        self._start_thread(run_v2_evaluation, params)

    def run_v3(self):
        params = {}
        for key, var in self.v3_entries.items():
            params[key] = var.get().strip()
        if not params.get("input_file"):
            messagebox.showerror("参数错误", "请选择输入文件")
            return
        if not params.get("output_dir"):
            params["output_dir"] = "."
        if not params.get("output_name"):
            params["output_name"] = "output.json"
        # base_url 和 endpoint 可空
        self._start_thread(run_v3_evaluation, params)

    def _start_thread(self, target_func, params):
        self.log_text.delete(1.0, tk.END)
        self.log("========== 开始评估 ==========\n")

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.running = True

        def stop_check():
            return not self.running

        def thread_target():
            try:
                target_func(params, self.log, stop_check)
            except Exception as e:
                self.log(f"\n评估过程发生异常: {e}\n")
                import traceback
                self.log(traceback.format_exc())
            finally:
                self.root.after(0, self.evaluation_finished)

        self.thread = threading.Thread(target=thread_target)
        self.thread.daemon = True
        self.thread.start()

    def evaluation_finished(self):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.running = False

    def stop_evaluation(self):
        self.running = False
        self.log("\n用户请求停止...\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = EvaluateApp(root)
    root.mainloop()