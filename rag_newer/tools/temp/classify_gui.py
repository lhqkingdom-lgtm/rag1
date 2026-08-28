#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from pathlib import Path

# ========== 分类规则 ==========
CATEGORY_PATTERNS = {
    "PWM": [r'\bPWM\b', r'脉宽调制'],
    "ADC": [r'\bADC\b', r'模数转换'],
    "DAC": [r'\bDAC\b', r'数模转换'],
    "I2C": [r'\bI2C\b', r'IIC', r"I²C"],
    "UART": [r'\bUART\b', r'引脚'],
    "SPI": [r'\bSPI\b'],
    "CAN": [r'\bCAN\b'],
    "USB": [r'\bUSB\b'],
    "LCD": [r'\bLCD\b', r'液晶'],
    "DMA": [r'\bDMA\b'],
    "RTC": [r'\bRTC\b', r'实时时钟'],
    "BLE": [r'\bBLE\b', r'蓝牙'],
    "WIFI": [r'\bWiFi\b', r'Wi-Fi', r'无线网络'],
    "Memory": [r'\bMemory\b', r'存储', r'Flash', r'RAM', r'EEPROM'],
    "WDT": [r'\bWDT\b', r'看门狗'],
    "GPIO": [r'\bGPIO\b', r'通用输入输出'],
    "Timer": [r'\bTimer\b', r'定时器'],
    "Interrupt_system": [r'\bInterrupt\b', r'中断系统', r'中断'],
    "System_clock": [r'\bSystem clock\b', r'系统时钟', r'时钟'],
    "Power": [r'\bPower\b', r'LVD', r'电源', r'供电'],
    "Low_power_mode": [r'\bLow power\b', r'低功耗', r'省电'],
    "Touch": [r'\bTouch\b', r'触摸'],
    "Electrical_parameters": [r'\bElectrical\b', r'电气特性', r'参数'],
    "Package_type": [r'\bPackage\b', r'封装'],
    "CPU": [r'\bCPU\b', r'中央处理器', r'核'],
    "Routine": [r'\bRoutine\b', r'示例', r'例程'],
    "Overview": [r'\bOverview\b', r'概述', r'介绍'],
}
ALL_CATEGORIES = list(CATEGORY_PATTERNS.keys())

def classify_all(text):
    """返回文本中匹配到的所有分类（去重），若无匹配则返回 ['Other']"""
    matched = set()
    for category, patterns in CATEGORY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                matched.add(category)
                break
    if not matched:
        return ["Other"]
    return sorted(matched)

class ClassificationApp:
    def __init__(self, root, chunks_dir="./output_chunks"):
        self.root = root
        self.root.title("分片分类校验工具")
        self.chunks_dir = Path(chunks_dir)
        if not self.chunks_dir.exists():
            messagebox.showerror("错误", f"目录 '{chunks_dir}' 不存在！")
            self.root.destroy()
            return

        self.file_data = {}  # {filename: {"content": str, "chips": [], "categories": []}}
        self.current_file = None

        self.create_widgets()
        self.load_files()
        self.update_file_list()
        self.set_default_corpus_info()

        # 绑定快捷键
        self.root.bind('<Delete>', self.delete_file_hotkey)
        self.root.bind('<Control-m>', self.merge_files_hotkey)
        self.root.bind('<Control-M>', self.merge_files_hotkey)
        self.root.bind('<Up>', self.select_prev_file)
        self.root.bind('<Down>', self.select_next_file)
        self.root.bind('<Left>', self.add_category_hotkey)
        self.root.bind('<Right>', self.remove_category_hotkey)
        self.root.bind('<Control-s>', self.save_current_chunk_hotkey)
        self.root.bind('<Control-S>', self.save_current_chunk_hotkey)

        # 存储所有可编辑输入框（用于焦点判断）
        self.edit_widgets = {
            self.chip_entry,
            self.corpus_name_entry,
            self.corpus_desc_entry,
            self.corpus_ver_entry,
            self.corpus_addr_entry,
            self.operator_entry,
        }

    def create_widgets(self):
        # ----- 顶部：全局芯片设置 -----
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(top_frame, text="芯片型号（多个用逗号/空格分隔）：").pack(side=tk.LEFT, padx=5)
        self.chip_entry = ttk.Entry(top_frame, width=80)
        self.chip_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="应用到所有文件", command=self.apply_chips_to_all).pack(side=tk.LEFT, padx=10)

        # ===== 语料信息输入区域 =====
        corpus_frame = ttk.LabelFrame(self.root, text="语料信息")
        corpus_frame.pack(fill=tk.X, padx=10, pady=5)

        corpus_frame.grid_columnconfigure(1, weight=1)
        corpus_frame.grid_columnconfigure(3, weight=1)
        corpus_frame.grid_columnconfigure(5, weight=1)

        ttk.Label(corpus_frame, text="名称：").grid(row=0, column=0, sticky='e', padx=2, pady=2)
        self.corpus_name_entry = ttk.Entry(corpus_frame, width=35)
        self.corpus_name_entry.grid(row=0, column=1, padx=2, pady=2, sticky='w')

        ttk.Label(corpus_frame, text="描述：").grid(row=0, column=2, sticky='e', padx=2, pady=2)
        self.corpus_desc_entry = ttk.Entry(corpus_frame, width=80)
        self.corpus_desc_entry.grid(row=0, column=3, columnspan=2, padx=2, pady=2, sticky='w')

        ttk.Label(corpus_frame, text="版本：").grid(row=1, column=0, sticky='e', padx=2, pady=2)
        self.corpus_ver_entry = ttk.Entry(corpus_frame, width=8)
        self.corpus_ver_entry.grid(row=1, column=1, padx=2, pady=2, sticky='w')

        ttk.Label(corpus_frame, text="操作员：").grid(row=1, column=2, sticky='e', padx=2, pady=2)
        self.operator_entry = ttk.Entry(corpus_frame, width=8)
        self.operator_entry.grid(row=1, column=3, padx=2, pady=2, sticky='w')

        ttk.Label(corpus_frame, text="地址：").grid(row=1, column=4, sticky='e', padx=2, pady=2)
        self.corpus_addr_entry = ttk.Entry(corpus_frame, width=40)
        self.corpus_addr_entry.grid(row=1, column=5, padx=2, pady=2, sticky='w')

        # ----- 主面板 -----
        main_panel = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 左侧文件列表
        left_frame = ttk.Frame(main_panel)
        main_panel.add(left_frame, weight=1)
        ttk.Label(left_frame, text="分片文件列表").pack(pady=5)
        self.file_listbox = tk.Listbox(left_frame, selectmode=tk.SINGLE, exportselection=False)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # 右侧面板
        right_frame = ttk.Frame(main_panel)
        main_panel.add(right_frame, weight=3)

        # 内容显示
        content_frame = ttk.LabelFrame(right_frame, text="文件内容")
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.content_text = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, height=15)
        self.content_text.pack(fill=tk.BOTH, expand=True)

        # 分类操作区域
        bottom_frame = ttk.Frame(right_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        current_cat_frame = ttk.LabelFrame(bottom_frame, text="当前分类")
        current_cat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.current_cat_listbox = tk.Listbox(current_cat_frame, selectmode=tk.MULTIPLE)
        self.current_cat_listbox.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(bottom_frame)
        btn_frame.pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="← 添加", command=self.add_category).pack(pady=5)
        ttk.Button(btn_frame, text="移除 →", command=self.remove_category).pack(pady=5)

        pool_cat_frame = ttk.LabelFrame(bottom_frame, text="分类池")
        pool_cat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.pool_cat_listbox = tk.Listbox(pool_cat_frame, selectmode=tk.MULTIPLE)
        self.pool_cat_listbox.pack(fill=tk.BOTH, expand=True)
        for cat in sorted(ALL_CATEGORIES):
            self.pool_cat_listbox.insert(tk.END, cat)
        self.pool_cat_listbox.insert(tk.END, "Other")

        # ----- 底部操作按钮 -----
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill=tk.X, pady=10, padx=10)

        ttk.Button(action_frame, text="保存当前切片", command=self.save_current_chunk).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="删除当前分片", command=self.delete_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="合并分片", command=self.merge_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="刷新列表", command=self.refresh_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="加载 JSON", command=self.load_json_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="保存结果 (JSON)", command=self.save_results).pack(side=tk.RIGHT, padx=5)

        # 快捷键提示
        tip = ttk.Label(self.root, text="快捷键: Delete=删除, Ctrl+M=合并, ↑/↓=切换文件, ←=添加, →=移除, Ctrl+S=保存切片", foreground="gray")
        tip.pack(side=tk.BOTTOM, pady=2)

    # ========== 设置默认语料信息 ==========
    def set_default_corpus_info(self):
        self.corpus_ver_entry.delete(0, tk.END)
        self.corpus_ver_entry.insert(0, "1.0")
        self.operator_entry.delete(0, tk.END)
        if self.file_data:
            first_file = sorted(self.file_data.keys())[0]
            content = self.file_data[first_file]["content"]
            match = re.search(r'^#\s+(.*?)$', content, re.MULTILINE)
            if match:
                title = match.group(1).strip()
                self.corpus_name_entry.delete(0, tk.END)
                self.corpus_name_entry.insert(0, title)

    # ========== 焦点判断 ==========
    def _focus_on_editable(self):
        focus = self.root.focus_get()
        if focus is None:
            return False
        if focus in self.edit_widgets:
            return True
        if hasattr(focus, 'master') and focus.master == self.content_text:
            return True
        return False

    # ========== 文件加载 ==========
    def load_files(self):
        md_files = list(self.chunks_dir.glob("*.md"))
        if not md_files:
            messagebox.showinfo("提示", f"在 '{self.chunks_dir}' 中没有找到 .md 文件。")
            return

        for md_path in md_files:
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"读取 {md_path.name} 出错: {e}")
                continue
            cat = classify_all(content)
            self.file_data[md_path.name] = {
                "content": content,
                "chips": [],
                "categories": cat
            }

    def update_file_list(self):
        self.file_listbox.delete(0, tk.END)
        for fname in sorted(self.file_data.keys()):
            self.file_listbox.insert(tk.END, fname)

    def on_file_select(self, event):
        selection = self.file_listbox.curselection()
        if not selection:
            return
        fname = self.file_listbox.get(selection[0])
        self.current_file = fname
        data = self.file_data.get(fname)
        if not data:
            return
        self.content_text.delete(1.0, tk.END)
        self.content_text.insert(tk.END, data["content"])
        self.current_cat_listbox.delete(0, tk.END)
        for cat in data["categories"]:
            self.current_cat_listbox.insert(tk.END, cat)

    def refresh_files(self):
        self.file_data.clear()
        self.load_files()
        self.update_file_list()
        self.current_file = None
        self.content_text.delete(1.0, tk.END)
        self.current_cat_listbox.delete(0, tk.END)

    # ========== 批量芯片设置 ==========
    def apply_chips_to_all(self):
        chip_str = self.chip_entry.get().strip()
        if chip_str:
            chips = re.split(r'[,\s;]+', chip_str)
            chips = [c.strip() for c in chips if c.strip()]
        else:
            chips = []
        for fname, data in self.file_data.items():
            data["chips"] = chips.copy()
        messagebox.showinfo("完成", f"已为所有文件设置芯片型号：{', '.join(chips) if chips else '无'}")

    # ========== 分类操作 ==========
    def add_category(self):
        if not self.current_file:
            messagebox.showwarning("警告", "请先选择一个文件。")
            return
        selected = self.pool_cat_listbox.curselection()
        if not selected:
            messagebox.showinfo("提示", "请从分类池中选择要添加的分类。")
            return
        current_cats = set(self.file_data[self.current_file]["categories"])
        added = []
        for idx in selected:
            cat = self.pool_cat_listbox.get(idx)
            if cat not in current_cats:
                current_cats.add(cat)
                added.append(cat)
        if added:
            self.file_data[self.current_file]["categories"] = sorted(list(current_cats))
            self.update_current_cat_list()
        else:
            messagebox.showinfo("提示", "所选分类已存在，无需重复添加。")

    def remove_category(self):
        if not self.current_file:
            messagebox.showwarning("警告", "请先选择一个文件。")
            return
        selected = self.current_cat_listbox.curselection()
        if not selected:
            messagebox.showinfo("提示", "请从当前分类中选择要移除的分类。")
            return
        current_cats = set(self.file_data[self.current_file]["categories"])
        to_remove = [self.current_cat_listbox.get(idx) for idx in selected]
        removed = []
        for cat in to_remove:
            if cat in current_cats:
                current_cats.remove(cat)
                removed.append(cat)
        if removed:
            self.file_data[self.current_file]["categories"] = sorted(list(current_cats))
            self.update_current_cat_list()
            # 无弹窗

    def update_current_cat_list(self):
        self.current_cat_listbox.delete(0, tk.END)
        if self.current_file:
            for cat in self.file_data[self.current_file]["categories"]:
                self.current_cat_listbox.insert(tk.END, cat)

    # ========== 快捷键处理 ==========
    def add_category_hotkey(self, event):
        if self._focus_on_editable():
            return
        self.add_category()
        return "break"

    def remove_category_hotkey(self, event):
        if self._focus_on_editable():
            return
        self.remove_category()
        return "break"

    # ========== 保存当前切片 ==========
    def save_current_chunk(self):
        if not self.current_file:
            messagebox.showwarning("警告", "请先选择一个文件。")
            return
        if not messagebox.askyesno("确认保存", f"确定要保存当前内容到文件 '{self.current_file}' 吗？\n此操作将覆盖原有内容！"):
            return
        content = self.content_text.get(1.0, tk.END).rstrip('\n')
        file_path = self.chunks_dir / self.current_file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            # 更新内存中的内容
            self.file_data[self.current_file]["content"] = content
            messagebox.showinfo("成功", f"文件 '{self.current_file}' 已保存。")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")

    def save_current_chunk_hotkey(self, event):
        if self._focus_on_editable():
            return
        self.save_current_chunk()
        return "break"

    # ========== 文件操作 ==========
    def delete_file(self):
        if not self.current_file:
            messagebox.showwarning("警告", "请先选择一个文件。")
            return
        if messagebox.askyesno("确认删除", f"确定要删除文件 '{self.current_file}' 吗？\n此操作不可恢复！"):
            self._perform_delete()

    def delete_file_hotkey(self, event):
        self.delete_file()

    def _perform_delete(self):
        cur_idx = self.file_listbox.curselection()
        if cur_idx:
            cur_idx = cur_idx[0]
        else:
            cur_idx = 0

        file_path = self.chunks_dir / self.current_file
        try:
            os.remove(file_path)
        except Exception as e:
            messagebox.showerror("错误", f"删除失败: {e}")
            return
        del self.file_data[self.current_file]
        self.current_file = None
        self.update_file_list()

        if self.file_data:
            new_idx = cur_idx
            if new_idx >= len(self.file_data):
                new_idx = len(self.file_data) - 1
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(new_idx)
            self.file_listbox.activate(new_idx)
            self.on_file_select(None)
        else:
            self.content_text.delete(1.0, tk.END)
            self.current_cat_listbox.delete(0, tk.END)

        messagebox.showinfo("完成", "文件已删除。")

    def merge_files(self):
        if not self.current_file:
            messagebox.showwarning("警告", "请先选择一个文件作为目标文件（第一个）。")
            return
        self._open_merge_dialog()

    def merge_files_hotkey(self, event):
        self.merge_files()

    def _open_merge_dialog(self):
        fnames = sorted(self.file_data.keys())
        merge_win = tk.Toplevel(self.root)
        merge_win.title("选择合并的第二个文件")
        ttk.Label(merge_win, text="请选择第二个文件（将追加到当前文件之后）").pack(pady=5)
        listbox = tk.Listbox(merge_win, selectmode=tk.SINGLE)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        for name in fnames:
            listbox.insert(tk.END, name)

        def do_merge():
            sel = listbox.curselection()
            if not sel:
                messagebox.showwarning("警告", "请选择一个文件。")
                return
            second_file = listbox.get(sel[0])
            if second_file == self.current_file:
                messagebox.showwarning("警告", "不能合并同一个文件。")
                return
            first_path = self.chunks_dir / self.current_file
            second_path = self.chunks_dir / second_file
            try:
                with open(first_path, 'a', encoding='utf-8') as f1:
                    with open(second_path, 'r', encoding='utf-8') as f2:
                        f1.write("\n\n")
                        f1.write(f2.read())
                os.remove(second_path)
            except Exception as e:
                messagebox.showerror("错误", f"合并失败: {e}")
                merge_win.destroy()
                return
            # 更新数据
            with open(first_path, 'r', encoding='utf-8') as f:
                merged_content = f.read()
            self.file_data[self.current_file]["content"] = merged_content
            del self.file_data[second_file]
            self.update_file_list()
            idx_new = list(self.file_data.keys()).index(self.current_file)
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(idx_new)
            self.file_listbox.activate(idx_new)
            self.on_file_select(None)
            messagebox.showinfo("完成", f"已将 '{second_file}' 合并到 '{self.current_file}'，并删除 '{second_file}'。")
            merge_win.destroy()

        ttk.Button(merge_win, text="合并", command=do_merge).pack(pady=10)

    # ========== 方向键切换 ==========
    def select_prev_file(self, event):
        if self._focus_on_editable():
            return
        self._select_relative(-1)
        return "break"

    def select_next_file(self, event):
        if self._focus_on_editable():
            return
        self._select_relative(1)
        return "break"

    def _select_relative(self, delta):
        if not self.file_data:
            return
        cur = self.file_listbox.curselection()
        if cur:
            idx = cur[0] + delta
        else:
            idx = 0
        idx = max(0, min(idx, len(self.file_data) - 1))
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(idx)
        self.file_listbox.activate(idx)
        self.on_file_select(None)

    # ========== 反向加载 JSON ==========
    def load_json_config(self):
        json_path = filedialog.askopenfilename(
            title="选择 JSON 配置文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not json_path:
            return

        chunks_dir = filedialog.askdirectory(
            title="选择分片文件所在的目录（chunks）",
            initialdir=str(self.chunks_dir)
        )
        if not chunks_dir:
            return
        chunks_dir = Path(chunks_dir)
        if not chunks_dir.exists():
            messagebox.showerror("错误", "所选目录不存在！")
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("错误", f"读取 JSON 失败: {e}")
            return

        if "files" not in data or not isinstance(data["files"], list):
            messagebox.showerror("错误", "JSON 格式无效：缺少 'files' 列表。")
            return

        self.file_data.clear()
        self.current_file = None

        missing_files = []
        for item in data["files"]:
            fname = item.get("file")
            if not fname:
                continue
            file_path = chunks_dir / fname
            if not file_path.exists():
                missing_files.append(fname)
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"读取 {fname} 失败: {e}")
                missing_files.append(fname)
                continue
            self.file_data[fname] = {
                "content": content,
                "chips": item.get("chips", []),
                "categories": item.get("categories", [])
            }

        self.update_file_list()
        if self.file_data:
            self.file_listbox.selection_set(0)
            self.file_listbox.activate(0)
            self.on_file_select(None)

        corpus = data.get("corpus_info", {})
        self.corpus_name_entry.delete(0, tk.END)
        self.corpus_name_entry.insert(0, corpus.get("name", ""))
        self.corpus_desc_entry.delete(0, tk.END)
        self.corpus_desc_entry.insert(0, corpus.get("description", ""))
        self.corpus_ver_entry.delete(0, tk.END)
        self.corpus_ver_entry.insert(0, corpus.get("version", "1.0"))
        self.corpus_addr_entry.delete(0, tk.END)
        self.corpus_addr_entry.insert(0, corpus.get("address", ""))
        self.operator_entry.delete(0, tk.END)
        self.operator_entry.insert(0, corpus.get("operator", ""))

        if missing_files:
            messagebox.showwarning("警告", f"以下文件在所选目录中不存在，已跳过：\n" + "\n".join(missing_files))
        else:
            messagebox.showinfo("完成", f"成功加载 {len(self.file_data)} 个文件。")

    # ========== 保存结果 ==========
    def save_results(self):
        name = self.corpus_name_entry.get().strip()
        description = self.corpus_desc_entry.get().strip()
        address = self.corpus_addr_entry.get().strip()
        version = self.corpus_ver_entry.get().strip()
        operator = self.operator_entry.get().strip()

        corpus_info = {
            "name": name,
            "description": description,
            "address": address,
            "version": version,
            "operator": operator
        }

        files = []
        for fname, data in self.file_data.items():
            files.append({
                "file": fname,
                "chips": data["chips"],
                "categories": data["categories"]
            })

        output = {
            "corpus_info": corpus_info,
            "files": files
        }

        save_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="classification_result.json"
        )
        if not save_path:
            return
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("成功", f"结果已保存到 {save_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")

def main():
    root = tk.Tk()
    root.geometry("1000x720")
    app = ClassificationApp(root, "./output_chunks")
    root.mainloop()

if __name__ == "__main__":
    main()