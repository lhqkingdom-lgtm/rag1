import gradio as gr
import psycopg2
import psycopg2.extras
from FlagEmbedding import BGEM3FlagModel
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
# ==================== 全局配置 ====================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("RAG_MODEL_PATH", str(BASE_DIR.parent / ".cache" / "models" / "bge-m3"))
CONFIG_PATH = Path(os.getenv("RAG_DB_CONFIG", BASE_DIR / "config" / "db_config.local.json"))

def load_db_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"配置文件 {CONFIG_PATH} 不存在")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    # 确保必要字段存在
    required = ['host', 'port', 'dbname', 'user']
    for key in required:
        if key not in config:
            raise ValueError(f"配置文件缺少必要的键: {key}")
    return config

DB_CONFIG = load_db_config()

RAG_CHIP_COLUMNS = [
    'PWM', 'ADC', 'DAC', 'I2C', 'UART', 'SPI', 'CAN', 'USB', 'LCD', 'DMA',
    'RTC', 'BLE', 'WIFI', 'Memory', 'WDT', 'GPIO', 'Timer', 'Interrupt_system',
    'System_clock', 'Power', 'Low_power_mode', 'Touch', 'Electrical_parameters',
    'Package_type', 'CPU', 'Routine', 'Overview', 'Other'
]

# ==================== 加载模型 ====================
print("正在加载 BGE-M3 模型...")
model = BGEM3FlagModel(MODEL_PATH, use_fp16=False)
print("模型加载完成。")

def generate_embeddings(content: str):
    # 截断过长文本（可选）
    if len(content) > 10000:
        content = content[:10000]
    
    # 编码
    output = model.encode(content, return_dense=True, return_sparse=True)
    
    # 处理稠密向量：归一化
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
    dense_str = '[' + ','.join(map(str, dense_list)) + ']'
    
    # 处理稀疏向量：转换为 pgvector 的 sparsevec 格式
    lexical_weights = output['lexical_weights']
    if isinstance(lexical_weights, list) and len(lexical_weights) > 0:
        lexical_weights = lexical_weights[0]
    elif not isinstance(lexical_weights, dict):
        lexical_weights = {}
    # 过滤零值
    sparse_clean = {int(k): float(v) for k, v in lexical_weights.items() if v != 0}
    
    dim = 250002
    if not sparse_clean:
        sparse_str = f"{{}}/{dim}"
    else:
        pairs = []
        for k, v in sparse_clean.items():
            idx = k + 1   # 注意：pgvector 的索引从 1 开始
            val = v
            if not np.isfinite(val) or abs(val) < 1e-8:
                continue
            pairs.append(f"{idx}:{val:.8f}")
        if not pairs:
            sparse_str = f"{{}}/{dim}"
        else:
            pairs.sort(key=lambda x: int(x.split(":")[0]))
            sparse_str = "{" + ",".join(pairs) + "}" + f"/{dim}"
    
    return dense_str, sparse_str

# ==================== 数据库辅助函数 ====================
def get_connection(password):
    return psycopg2.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        dbname=DB_CONFIG['dbname'],
        user=DB_CONFIG['user'],
        password=password,
        client_encoding='GB18030'
    )
def list_corpora(password):
    conn = get_connection(password)
    cur = conn.cursor()
    cur.execute("""
        SELECT corpus_id, name, description, category, version, is_review, reviewer
        FROM corpus
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    df = pd.DataFrame(rows, columns=['corpus_id', 'name', 'description', 'category', 'version', 'is_review', 'reviewer'])
    df['is_review'] = df['is_review'].apply(lambda x: '已审核' if x else '未审核')
    return df

def get_chunks(password, corpus_id):
    conn = get_connection(password)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT chunk_id, content, index, is_review, update_time
        FROM chip_chunks
        WHERE corpus_id = %s
        ORDER BY chunk_id
    """, (corpus_id,))
    chunks = cur.fetchall()
    cur.close()
    conn.close()
    df = pd.DataFrame(chunks)
    if not df.empty:
        df['is_review'] = df['is_review'].apply(lambda x: '已审核' if x else '未审核')
        df['index'] = df['index'].apply(lambda x: ', '.join(x) if x else '')
        df['content_preview'] = df['content'].str.slice(0, 100)
        df = df[['chunk_id', 'content_preview', 'index', 'is_review', 'update_time']]
    return df

def find_chip_names_for_chunk(cur, chunk_id):
    where_parts = [f'"{col}" @> ARRAY[%s]' for col in RAG_CHIP_COLUMNS]
    where_clause = ' OR '.join(where_parts)
    sql = f'SELECT chip_name FROM rag_chips WHERE {where_clause}'
    cur.execute(sql, [chunk_id] * len(RAG_CHIP_COLUMNS))
    rows = cur.fetchall()
    return [row[0] for row in rows] 

def update_chunk_content(cur, chunk_id, new_content):
    dense_str, sparse_str = generate_embeddings(new_content)
    cur.execute("""
        UPDATE chip_chunks
        SET content = %s,
            vector = %s::vector,
            sparse_embedding = %s::sparsevec,
            is_review = true,
            update_time = CURRENT_TIMESTAMP
        WHERE chunk_id = %s
    """, (new_content, dense_str, sparse_str, chunk_id))

def update_chunk_index(cur, chunk_id, old_index, new_index):
    # 更新 chip_chunks 表
    cur.execute("""
        UPDATE chip_chunks
        SET index = %s,
            is_review = true,
            update_time = CURRENT_TIMESTAMP
        WHERE chunk_id = %s
    """, (new_index, chunk_id))

    # 获取所有关联的芯片名
    chip_names = find_chip_names_for_chunk(cur, chunk_id)
    if not chip_names:
        raise Exception(f"无法在 rag_chips 中找到 chunk_id: {chunk_id}")
    
    old_set = set(old_index)
    new_set = set(new_index)
    cols_to_remove = old_set - new_set
    cols_to_add = new_set - old_set
    
    for chip_name in chip_names:   # 遍历每个芯片
        for col in cols_to_remove:
            if col not in RAG_CHIP_COLUMNS:
                continue
            cur.execute(
                f'UPDATE rag_chips SET "{col}" = array_remove("{col}", %s) WHERE chip_name = %s',
                (chunk_id, chip_name)
            )
        for col in cols_to_add:
            if col not in RAG_CHIP_COLUMNS:
                continue
            cur.execute(
                f'UPDATE rag_chips SET "{col}" = array_append("{col}", %s) '
                f'WHERE chip_name = %s AND NOT ("{col}" @> ARRAY[%s])',
                (chunk_id, chip_name, chunk_id)
            )

def delete_chunk(cur, chunk_id):
    chip_names = find_chip_names_for_chunk(cur, chunk_id)
    if chip_names:
        for chip_name in chip_names:
            set_parts = [f'"{col}" = array_remove("{col}", %s)' for col in RAG_CHIP_COLUMNS]
            set_clause = ', '.join(set_parts)
            cur.execute(
                f'UPDATE rag_chips SET {set_clause} WHERE chip_name = %s',
                [chunk_id] * len(RAG_CHIP_COLUMNS) + [chip_name]
            )
    cur.execute("DELETE FROM chip_chunks WHERE chunk_id = %s", (chunk_id,))

def confirm_chunk(cur, chunk_id):
    cur.execute("""
        UPDATE chip_chunks
        SET is_review = true,
            update_time = CURRENT_TIMESTAMP
        WHERE chunk_id = %s
    """, (chunk_id,))

# ==================== Gradio 界面 ====================
with gr.Blocks(title="数据库审核工具") as demo:
    gr.Markdown("# 数据库审核工具")
    gr.Markdown("请输入数据库密码并连接，然后选择语料进行审核。")

    password_state = gr.State("")
    current_corpus_id_state = gr.State("")  # 新增：记录当前选中的语料 ID

    with gr.Row():
        password_input = gr.Textbox(label="数据库密码", type="password", placeholder="请输入 postgres 密码")
        connect_btn = gr.Button("连接数据库")
        connect_status = gr.Markdown("未连接")

    corpus_df = gr.Dataframe(label="语料列表", interactive=False, wrap=True)
    chunk_df = gr.Dataframe(label="该语料下的 Chunks", interactive=False, wrap=True)

    with gr.Row():
        chunk_id_text = gr.Textbox(label="当前 chunk_id", interactive=False)
        chunk_review_status = gr.Textbox(label="审核状态", interactive=False)
    with gr.Row():
        content_display = gr.Textbox(label="原始 content", interactive=False, lines=10, max_lines=30)
        content_edit = gr.Textbox(label="修改 content（可编辑）", lines=10, max_lines=30)
    with gr.Row():
        index_display = gr.Textbox(label="原始 index（逗号分隔）", interactive=False)
        index_edit = gr.Textbox(label="修改 index（逗号分隔）")
    with gr.Row():
        confirm_btn = gr.Button("确认（无修改）")
        update_content_btn = gr.Button("更新 content")
        update_index_btn = gr.Button("更新 index")
        delete_btn = gr.Button("删除该 chunk")
        refresh_btn = gr.Button("刷新列表")
    operation_status = gr.Markdown("")

    # ==================== 事件处理 ====================
    def connect_db(password):
        try:
            df = list_corpora(password)
            return df, "✅ 连接成功", password
        except Exception as e:
            return pd.DataFrame(), f"❌ 连接失败: {e}", ""

    connect_btn.click(
        connect_db,
        inputs=[password_input],
        outputs=[corpus_df, connect_status, password_state]
    )

    def load_chunks(password, corpus_df_value, evt: gr.SelectData):
        """选中语料时加载 chunks"""
        if not password:
            return pd.DataFrame(), "", "请先连接数据库"
        try:
            row_index = evt.index[0]
            corpus_id = corpus_df_value.iloc[row_index]['corpus_id']
            df = get_chunks(password, corpus_id)
            return df, corpus_id, f"已加载语料 {corpus_id} 的 chunks"
        except Exception as e:
            return pd.DataFrame(), "", f"加载失败: {e}"

    corpus_df.select(
        load_chunks,
        inputs=[password_state, corpus_df],
        outputs=[chunk_df, current_corpus_id_state, operation_status]
    )

    def load_chunk_detail(password, chunk_df_value, evt: gr.SelectData):
        """选中 chunk 行时加载详情"""
        if not password:
            return "", "", "", "", "", "请先连接数据库"
        try:
            row_index = evt.index[0]
            chunk_id = chunk_df_value.iloc[row_index]['chunk_id']
            conn = get_connection(password)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT chunk_id, content, index, is_review FROM chip_chunks WHERE chunk_id = %s", (chunk_id,))
            chunk = cur.fetchone()
            cur.close()
            conn.close()
            content = chunk['content']
            index = chunk['index']
            is_review = '已审核' if chunk['is_review'] else '未审核'
            index_str = ', '.join(index) if index else ''
            return chunk_id, is_review, content, content, index_str, index_str
        except Exception as e:
            return "", "", "", "", "", f"加载详情失败: {e}"

    chunk_df.select(
        load_chunk_detail,
        inputs=[password_state, chunk_df],
        outputs=[chunk_id_text, chunk_review_status, content_display, content_edit, index_display, index_edit]
    )

    def refresh_chunks_after_operation(password, corpus_id):
        """操作完成后重新加载 chunks"""
        if not password or not corpus_id:
            return pd.DataFrame()
        try:
            return get_chunks(password, corpus_id)
        except Exception as e:
            print(f"刷新 chunks 失败: {e}")
            return pd.DataFrame()

    def on_confirm(password, chunk_id, corpus_id):
        if not password or not chunk_id:
            return "密码或 chunk_id 为空", pd.DataFrame()
        try:
            conn = get_connection(password)
            cur = conn.cursor()
            confirm_chunk(cur, chunk_id)
            conn.commit()
            cur.close()
            conn.close()
            new_df = refresh_chunks_after_operation(password, corpus_id)
            return "✅ 已确认", new_df
        except Exception as e:
            return f"❌ 操作失败: {e}", pd.DataFrame()

    confirm_btn.click(
        on_confirm,
        inputs=[password_state, chunk_id_text, current_corpus_id_state],
        outputs=[operation_status, chunk_df]
    )

    def on_update_content(password, chunk_id, corpus_id, new_content):
        if not password or not chunk_id:
            return "密码或 chunk_id 为空", pd.DataFrame()
        if not new_content.strip():
            return "content 不能为空", pd.DataFrame()
        try:
            conn = get_connection(password)
            cur = conn.cursor()
            update_chunk_content(cur, chunk_id, new_content)
            conn.commit()
            cur.close()
            conn.close()
            new_df = refresh_chunks_after_operation(password, corpus_id)
            return "✅ content 已更新，向量已重新生成", new_df
        except Exception as e:
            return f"❌ 操作失败: {e}", pd.DataFrame()

    update_content_btn.click(
        on_update_content,
        inputs=[password_state, chunk_id_text, current_corpus_id_state, content_edit],
        outputs=[operation_status, chunk_df]
    )

    def on_update_index(password, chunk_id, corpus_id, new_index_str):
        if not password or not chunk_id:
            return "密码或 chunk_id 为空", pd.DataFrame()
        new_index = [x.strip() for x in new_index_str.split(',') if x.strip()]
        if not new_index:
            return "index 不能为空", pd.DataFrame()
        try:
            conn = get_connection(password)
            cur = conn.cursor()
            # 从数据库获取当前 index
            cur.execute("SELECT index FROM chip_chunks WHERE chunk_id = %s", (chunk_id,))
            row = cur.fetchone()
            if not row:
                return "chunk_id 不存在", pd.DataFrame()
            old_index = row[0] if row[0] else []
            # 执行更新
            update_chunk_index(cur, chunk_id, old_index, new_index)
            conn.commit()
            cur.close()
            conn.close()
            new_df = refresh_chunks_after_operation(password, corpus_id)
            return "✅ index 已更新，rag_chips 已同步", new_df
        except Exception as e:
            # 日志记录
            import traceback
            traceback.print_exc()
            return f"❌ 操作失败: {e}", pd.DataFrame()

    update_index_btn.click(
        on_update_index,
        inputs=[password_state, chunk_id_text, current_corpus_id_state, index_edit],
        outputs=[operation_status, chunk_df]
    )

    def on_delete(password, chunk_id, corpus_id):
        if not password or not chunk_id:
            return "密码或 chunk_id 为空", pd.DataFrame()
        try:
            conn = get_connection(password)
            cur = conn.cursor()
            delete_chunk(cur, chunk_id)
            conn.commit()
            cur.close()
            conn.close()
            new_df = refresh_chunks_after_operation(password, corpus_id)
            return "✅ 已删除", new_df
        except Exception as e:
            return f"❌ 操作失败: {e}", pd.DataFrame()

    delete_btn.click(
        on_delete,
        inputs=[password_state, chunk_id_text, current_corpus_id_state],
        outputs=[operation_status, chunk_df]
    )

    def refresh_all(password):
        if not password:
            return pd.DataFrame(), pd.DataFrame(), "请先连接数据库"
        try:
            df_corpora = list_corpora(password)
            return df_corpora, pd.DataFrame(), "已刷新"
        except Exception as e:
            return pd.DataFrame(), pd.DataFrame(), f"刷新失败: {e}"

    refresh_btn.click(
        refresh_all,
        inputs=[password_state],
        outputs=[corpus_df, chunk_df, operation_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
