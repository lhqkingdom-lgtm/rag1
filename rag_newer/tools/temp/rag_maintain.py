"""
RAG 数据库维护工具
功能：
1. 按 chunk_id 替换 content 或删除 chunk（同时从 RAG_chips 中移除引用）
2. 上传 Markdown 文件，使用 BGE-M3 编码并与库中稠密/稀疏向量比对，返回 Top5 候选，选择后显示 content，可执行替换或删除
依赖：psycopg2-binary, FlagEmbedding, torch, transformers, tkinter (内置)
"""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import psycopg2
import psycopg2.extras
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import os
import threading
import sys

# 模型路径（请确保该路径下存在 bge-m3 模型文件）
MODEL_PATH = r"E:\moshang\RAG_new-master\rag_new\modelscope_models\bge-m3"
config_file = "db_config2.json"

class RAGDatabaseTool:
    # 保持不变，省略（与之前完全一致）
    def __init__(self, config_path, model_path):
        self.config_path = config_path
        self.model_path = model_path
        self.conn = None
        self.model = None
        self.chunk_id_to_content = {}
        
    def connect_db(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        try:
            self.conn = psycopg2.connect(
                host=cfg['host'],
                port=cfg['port'],
                dbname=cfg['dbname'],
                user=cfg['user'],
                password=cfg['password']
            )
            self.conn.autocommit = False
            return True
        except Exception as e:
            messagebox.showerror("数据库连接错误", str(e))
            return False

    def load_model(self):
        try:
            self.model = BGEM3FlagModel(self.model_path, use_fp16=False)
            return True
        except Exception as e:
            messagebox.showerror("模型加载错误", str(e))
            return False

    def format_sparse_vec(self, lexical_weights: dict, dim: int = 250002) -> str:
        if not lexical_weights:
            return f"{{}}/{dim}"
        pairs = []
        for k, v in lexical_weights.items():
            try:
                idx = int(k) + 1
                val = float(v)
                if not np.isfinite(val) or abs(val) < 1e-8:
                    continue
                pairs.append(f"{idx}:{val:.8f}")
            except (ValueError, TypeError):
                continue
        if not pairs:
            return f"{{}}/{dim}"
        pairs.sort(key=lambda x: int(x.split(":")[0]))
        return "{" + ",".join(pairs) + "}" + f"/{dim}"

    def encode_text(self, text):
        if not self.model:
            raise RuntimeError("模型未加载")
        if len(text) > 10000:
            text = text[:10000]
        output = self.model.encode(text, return_dense=True, return_sparse=True)

        dense_vecs = output['dense_vecs']
        if isinstance(dense_vecs, np.ndarray):
            dense_vec = dense_vecs[0] if dense_vecs.ndim == 2 else dense_vecs
        else:
            dense_vec = np.array(dense_vecs)
            dense_vec = dense_vec[0] if dense_vec.ndim == 2 else dense_vec
        norm = np.linalg.norm(dense_vec)
        if norm > 0:
            dense_vec = dense_vec / norm
        dense_list = dense_vec.tolist()

        lexical_weights = output['lexical_weights']
        if isinstance(lexical_weights, list) and len(lexical_weights) > 0:
            lexical_weights = lexical_weights[0]
        elif not isinstance(lexical_weights, dict):
            lexical_weights = {}
        sparse_clean = {int(k): float(v) for k, v in lexical_weights.items() if v != 0}
        return dense_list, sparse_clean

    def update_chunk_content(self, chunk_id, new_content):
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT content FROM chip_chunks WHERE chunk_id = %s", (chunk_id,))
            old_row = cur.fetchone()
            old_content = old_row[0] if old_row else None

            cur.execute("UPDATE chip_chunks SET content = %s WHERE chunk_id = %s", (new_content, chunk_id))
            if cur.rowcount == 0:
                self.conn.rollback()
                return False, "未找到该 chunk_id"

            self.conn.commit()

            cur.execute("SELECT content FROM chip_chunks WHERE chunk_id = %s", (chunk_id,))
            new_row = cur.fetchone()
            updated_content = new_row[0] if new_row else None

            return True, f"更新成功！\n旧内容长度: {len(old_content) if old_content else 0}\n新内容长度: {len(updated_content) if updated_content else 0}"
        except Exception as e:
            self.conn.rollback()
            return False, str(e)
        finally:
            cur.close()

    def delete_chunk(self, chunk_id):
        cur = self.conn.cursor()
        try:
            cur.execute("DELETE FROM chip_chunks WHERE chunk_id = %s", (chunk_id,))
            if cur.rowcount == 0:
                self.conn.rollback()
                return False, "未找到该 chunk_id"
            col_query = """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name='RAG_chips' 
                  AND data_type='ARRAY' 
                  AND column_name != 'chip_name'
            """
            cur.execute(col_query)
            cols = [row[0] for row in cur.fetchall()]
            if cols:
                for col in cols:
                    update_sql = f"UPDATE RAG_chips SET \"{col}\" = array_remove(\"{col}\", %s) WHERE %s = ANY(\"{col}\")"
                    cur.execute(update_sql, (chunk_id, chunk_id))
            self.conn.commit()
            return True, "删除成功（已从所有 RAG_chips 列移除引用）"
        except Exception as e:
            self.conn.rollback()
            return False, str(e)
        finally:
            cur.close()

    def search_similar_chunks(self, text, top_k=5):
        dense_list, sparse_dict = self.encode_text(text)
        dense_str = "[" + ",".join(map(str, dense_list)) + "]"
        sparse_str = self.format_sparse_vec(sparse_dict, dim=250002)

        query = """
            SELECT chunk_id, content,
                1 - (vector <=> %s) AS dense_score,
                1 - (sparse_embedding <=> %s) AS sparse_score,
                (1 - (vector <=> %s) + 1 - (sparse_embedding <=> %s)) / 2 AS avg_score
            FROM chip_chunks
            WHERE vector IS NOT NULL AND sparse_embedding IS NOT NULL
            ORDER BY avg_score DESC
            LIMIT %s
        """
        cur = self.conn.cursor()
        try:
            cur.execute(query, (dense_str, sparse_str, dense_str, sparse_str, top_k))
            rows = cur.fetchall()
            return [(row[0], row[1], float(row[4])) for row in rows]
        except Exception as e:
            messagebox.showerror("检索错误", str(e))
            print(cur.mogrify(query, (dense_str, sparse_str, dense_str, sparse_str, top_k)).decode())
            return []
        finally:
            cur.close()

    def get_chunk_content(self, chunk_id):
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT content FROM chip_chunks WHERE chunk_id = %s", (chunk_id,))
            row = cur.fetchone()
            return row[0] if row else None
        except Exception as e:
            messagebox.showerror("查询错误", str(e))
            return None
        finally:
            cur.close()


# ==================== GUI 部分（重构） ====================
class Application(tk.Tk):
    def __init__(self, tool):
        super().__init__()
        self.tool = tool
        self.title("RAG 数据库维护工具")
        self.geometry("950x750")
        self.resizable(True, True)
        self.create_widgets()
        if not self.tool.connect_db():
            self.destroy()
            return
        if not self.tool.load_model():
            self.destroy()
            return

    def create_widgets(self):
        # 主框架使用 PanedWindow 可调整大小
        main_pane = ttk.PanedWindow(self, orient=tk.VERTICAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ---------- 上半部分：操作区 ----------
        top_frame = ttk.Frame(main_pane)
        main_pane.add(top_frame, weight=1)

        # 第一行：按 chunk_id 操作
        group1 = ttk.LabelFrame(top_frame, text="按 chunk_id 操作", padding=5)
        group1.pack(fill=tk.X, pady=5)

        f1 = ttk.Frame(group1)
        f1.pack(fill=tk.X, pady=2)
        ttk.Label(f1, text="Chunk ID:").pack(side=tk.LEFT, padx=5)
        self.entry_id = ttk.Entry(f1, width=25)
        self.entry_id.pack(side=tk.LEFT, padx=5)
        ttk.Label(f1, text="操作:").pack(side=tk.LEFT, padx=5)
        self.op_var = tk.StringVar(value="替换")
        op_menu = ttk.Combobox(f1, textvariable=self.op_var, values=["替换", "删除"], state="readonly", width=8)
        op_menu.pack(side=tk.LEFT, padx=5)

        # 新内容文件路径和浏览
        ttk.Label(f1, text="新内容文件:").pack(side=tk.LEFT, padx=5)
        self.file_path_var = tk.StringVar()
        ttk.Entry(f1, textvariable=self.file_path_var, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Button(f1, text="浏览...", command=self.browse_replace_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(f1, text="执行", command=self.execute_direct).pack(side=tk.LEFT, padx=20)

        # 替换文件内容编辑框
        ttk.Label(group1, text="替换内容（可编辑）：").pack(anchor=tk.W, padx=5, pady=(5,0))
        self.replace_text = scrolledtext.ScrolledText(group1, wrap=tk.WORD, height=6)
        self.replace_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 第二行：相似检索
        group2 = ttk.LabelFrame(top_frame, text="通过文件内容相似匹配", padding=5)
        group2.pack(fill=tk.X, pady=5)

        f2 = ttk.Frame(group2)
        f2.pack(fill=tk.X, pady=2)
        ttk.Label(f2, text="检索文件:").pack(side=tk.LEFT, padx=5)
        self.sim_file_var = tk.StringVar()
        ttk.Entry(f2, textvariable=self.sim_file_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(f2, text="浏览...", command=self.browse_sim_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(f2, text="开始检索", command=self.start_search).pack(side=tk.LEFT, padx=10)

        # 检索文本编辑框
        ttk.Label(group2, text="检索文本（可编辑）：").pack(anchor=tk.W, padx=5, pady=(5,0))
        self.search_text = scrolledtext.ScrolledText(group2, wrap=tk.WORD, height=6)
        self.search_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ---------- 下半部分：结果展示和操作 ----------
        bottom_frame = ttk.Frame(main_pane)
        main_pane.add(bottom_frame, weight=2)

        # 结果列表
        group3 = ttk.LabelFrame(bottom_frame, text="检索结果（Top5）", padding=5)
        group3.pack(fill=tk.BOTH, expand=True, pady=5)

        self.tree = ttk.Treeview(group3, columns=("score", "content_preview"), show="tree", height=5)
        self.tree.heading("#0", text="Chunk ID")
        self.tree.heading("score", text="相似度")
        self.tree.heading("content_preview", text="内容预览 (前100字符)")
        self.tree.column("#0", width=200)
        self.tree.column("score", width=80)
        self.tree.column("content_preview", width=400)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=5)
        self.tree.bind('<<TreeviewSelect>>', self.on_select_chunk)

        # 选中内容显示
        group4 = ttk.LabelFrame(bottom_frame, text="选中 chunk 的完整内容", padding=5)
        group4.pack(fill=tk.BOTH, expand=True, pady=5)

        self.content_text = scrolledtext.ScrolledText(group4, wrap=tk.WORD, height=8)
        self.content_text.pack(fill=tk.BOTH, expand=True)

        # 操作按钮
        btn_frame = ttk.Frame(bottom_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="用此内容替换选中 chunk", command=self.replace_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="删除选中 chunk", command=self.delete_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="刷新列表", command=self.clear_results).pack(side=tk.RIGHT, padx=5)

        self.selected_chunk_id = None
        self.results_list = []

    # ---------- 文件浏览方法 ----------
    def browse_replace_file(self):
        f = filedialog.askopenfilename(filetypes=[("Markdown", "*.md"), ("All files", "*.*")])
        if f:
            self.file_path_var.set(f)
            # 自动加载内容到编辑框
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    content = file.read()
                self.replace_text.delete(1.0, tk.END)
                self.replace_text.insert(tk.END, content)
            except Exception as e:
                messagebox.showerror("文件读取错误", str(e))

    def browse_sim_file(self):
        f = filedialog.askopenfilename(filetypes=[("Markdown", "*.md"), ("All files", "*.*")])
        if f:
            self.sim_file_var.set(f)
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    content = file.read()
                self.search_text.delete(1.0, tk.END)
                self.search_text.insert(tk.END, content)
            except Exception as e:
                messagebox.showerror("文件读取错误", str(e))

    # ---------- 直接操作（替换/删除） ----------
    def execute_direct(self):
        chunk_id = self.entry_id.get().strip()
        if not chunk_id:
            messagebox.showwarning("提示", "请输入 chunk_id")
            return
        op = self.op_var.get()
        if op == "替换":
            new_content = self.replace_text.get(1.0, tk.END).strip()
            if not new_content:
                messagebox.showwarning("提示", "替换内容不能为空")
                return
            ok, msg = self.tool.update_chunk_content(chunk_id, new_content)
            messagebox.showinfo("结果", msg)
        else:  # 删除
            ok, msg = self.tool.delete_chunk(chunk_id)
            messagebox.showinfo("结果", msg)

    # ---------- 相似检索 ----------
    def start_search(self):
        # 从编辑框获取文本（用户可能已经编辑过）
        text = self.search_text.get(1.0, tk.END).strip()
        if not text:
            # 如果编辑框为空，尝试从文件加载（但已自动加载，可以提示）
            messagebox.showwarning("提示", "检索文本为空，请先浏览文件或手动输入")
            return

        def search_thread():
            self.status("正在检索，请稍候...")
            results = self.tool.search_similar_chunks(text, top_k=5)
            self.after(0, lambda: self.display_results(results))
        threading.Thread(target=search_thread, daemon=True).start()

    # ---------- 结果展示和操作 ----------
    def display_results(self, results):
        self.clear_results()
        self.results_list = results
        for chunk_id, content, score in results:
            preview = content[:100] + "..." if len(content) > 100 else content
            self.tree.insert("", tk.END, text=chunk_id, values=(f"{score:.4f}", preview))
        self.status("检索完成")

    def clear_results(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.content_text.delete(1.0, tk.END)
        self.selected_chunk_id = None
        self.results_list = []

    def on_select_chunk(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        item = selected[0]
        chunk_id = self.tree.item(item, "text")
        content = None
        for cid, cont, _ in self.results_list:
            if cid == chunk_id:
                content = cont
                break
        if content is None:
            content = self.tool.get_chunk_content(chunk_id)
        if content:
            self.content_text.delete(1.0, tk.END)
            self.content_text.insert(tk.END, content)
            self.selected_chunk_id = chunk_id
        else:
            messagebox.showwarning("提示", "未找到该 chunk 的内容")

    def replace_selected(self):
        if not self.selected_chunk_id:
            messagebox.showwarning("提示", "请先在结果列表中选择一个 chunk")
            return
        new_content = self.replace_text.get(1.0, tk.END).strip()
        if not new_content:
            messagebox.showwarning("提示", "内容不能为空")
            return
        ok, msg = self.tool.update_chunk_content(self.selected_chunk_id, new_content)
        messagebox.showinfo("结果", msg)
        if ok:
            # 可刷新列表
            pass

    def delete_selected(self):
        if not self.selected_chunk_id:
            messagebox.showwarning("提示", "请先在结果列表中选择一个 chunk")
            return
        if not messagebox.askyesno("确认删除", f"确定要删除 chunk '{self.selected_chunk_id}' 吗？\n该操作会同时从所有 RAG_chips 中移除引用。"):
            return
        ok, msg = self.tool.delete_chunk(self.selected_chunk_id)
        messagebox.showinfo("结果", msg)
        if ok:
            self.clear_results()

    def status(self, msg):
        self.title(f"RAG 数据库维护工具 - {msg}")


if __name__ == "__main__":
    
    if not os.path.exists(config_file):
        print(f"错误：配置文件 {config_file} 不存在")
        sys.exit(1)
    tool = RAGDatabaseTool(config_file, MODEL_PATH)
    app = Application(tool)
    app.mainloop()