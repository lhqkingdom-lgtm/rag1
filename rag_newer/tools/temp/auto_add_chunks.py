import os
import re
import json
import uuid
import psycopg2
from psycopg2 import sql, extras
import numpy as np
from FlagEmbedding import BGEM3FlagModel

# ========== 配置 ==========
CONFIG_FILE = 'db_config.json'
MODEL_PATH = r'E:\moshang\RAG_new-master\rag_new\modelscope_models\bge-m3'
CHUNK_DIR = './output_chunks'
CLASSIFICATION_FILE = 'classification_result.json'
PROGRESS_FILE = "import_progress.json"

ALL_CATEGORIES = [
    "pwm", "adc", "dac", "i2c", "uart", "spi", "can", "usb",
    "lcd", "dma", "rtc", "ble", "wifi", "memory", "wdt",
    "gpio", "timer", "interrupt_system", "system_clock",
    "power", "low_power_mode", "touch", "electrical_parameters",
    "package_type", "cpu", "routine", "overview", "other"
]

# ========== 辅助函数 ==========
def load_db_config():
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def get_db_connection():
    config = load_db_config()
    conn = psycopg2.connect(**config)
    conn.autocommit = False
    return conn

def extract_title_and_images(content):
    lines = content.splitlines()
    title = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            title = stripped.lstrip('#').strip()
            break
    if not title:
        title = content[:60].strip() + ('...' if len(content) > 60 else '')
    md_images = re.findall(r'!\[.*?\]\((.*?)\)', content)
    html_images = re.findall(r'<img.*?src=["\'](.*?)["\']', content, re.IGNORECASE)
    images = md_images + html_images
    seen = set()
    unique_images = []
    for img in images:
        if img not in seen:
            seen.add(img)
            unique_images.append(img)
    return title, unique_images

def format_sparse_vec(lexical_weights: dict, dim: int = 250002) -> str:
    if not lexical_weights:
        return f'{{}}/{dim}'
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
        return f'{{}}/{dim}'
    pairs.sort(key=lambda x: int(x.split(':')[0]))
    return '{' + ','.join(pairs) + '}' + f'/{dim}'

def ensure_chip_exists(conn, chip_name):
    with conn.cursor() as cur:
        cur.execute('SELECT 1 FROM rag_chips WHERE chip_name = %s', (chip_name,))
        if not cur.fetchone():
            cur.execute('INSERT INTO rag_chips (chip_name) VALUES (%s)', (chip_name,))
            print(f"创建新芯片: {chip_name}")

def update_chip_category(conn, chip_name, category, chunk_id):
    col_lower = category.lower()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'rag_chips' AND LOWER(column_name) = %s
        """, (col_lower,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"列 '{category}' 在 rag_chips 表中不存在，请先创建该列。")
        actual_col = row[0]
    with conn.cursor() as cur:
        update_sql = sql.SQL("""
            UPDATE rag_chips
            SET {col} = array_append({col}, %s)
            WHERE chip_name = %s
        """).format(col=sql.Identifier(actual_col))
        cur.execute(update_sql, (chunk_id, chip_name))
        print(f"为芯片 '{chip_name}' 的列 '{actual_col}' 追加 chunk_id: {chunk_id}")

def process_corpus(conn, corpus_info):
    name = corpus_info.get('name')
    version = corpus_info.get('version')
    description = corpus_info.get('description', '')
    file_path = corpus_info.get('address', '')
    operator = corpus_info.get('operator', '')
    category = '芯片'

    if not name or not version:
        raise ValueError("corpus_info 中必须包含 name 和 version")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT corpus_id, is_active FROM corpus WHERE name = %s AND version = %s",
            (name, version)
        )
        row = cur.fetchone()
        if row:
            corpus_id, is_active = row
            if not is_active:
                cur.execute("UPDATE corpus SET is_active = true WHERE corpus_id = %s", (corpus_id,))
                conn.commit()
                print(f"重新激活已有 corpus: {corpus_id}")
            else:
                print(f"复用已有 corpus: {corpus_id}")
            return corpus_id

        cur.execute("SELECT corpus_id FROM corpus WHERE name = %s AND version != %s", (name, version))
        for (old_id,) in cur.fetchall():
            cur.execute("UPDATE corpus SET is_active = false WHERE corpus_id = %s", (old_id,))
            print(f"旧版本 corpus_id={old_id} 已置为 inactive")

        corpus_id = str(uuid.uuid4())
        insert_sql = """
            INSERT INTO corpus (corpus_id, name, description, category, file_path, is_active, version, operator)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(insert_sql, (corpus_id, name, description, category, file_path, True, version, operator))
        conn.commit()
        print(f"插入新 corpus 记录: corpus_id={corpus_id}, name='{name}', version='{version}'")
        return corpus_id

# ========== 进度文件管理 ==========
def load_progress(expected_name=None, expected_version=None):
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            progress = json.load(f)

        if expected_name and expected_version:
            saved_name = progress.get("corpus_name")
            saved_version = progress.get("corpus_version")
            if saved_name == expected_name and saved_version == expected_version:
                print(f"进度文件匹配: {saved_name} v{saved_version}，继续断点续传。")
                return progress
            else:
                print(f"进度文件不匹配 (当前语料: {expected_name} v{expected_version}, "
                      f"文件记录: {saved_name} v{saved_version})，将从头开始。")
                backup = PROGRESS_FILE + ".bak"
                # 使用 os.replace 替代 os.rename，避免 FileExistsError
                os.replace(PROGRESS_FILE, backup)
                print(f"旧进度文件已备份为: {backup}")
                return {"corpus_id": None, "completed_files": []}
        else:
            return progress
    return {"corpus_id": None, "completed_files": []}

def save_progress(progress, corpus_name=None, corpus_version=None):
    if corpus_name:
        progress["corpus_name"] = corpus_name
    if corpus_version:
        progress["corpus_version"] = corpus_version
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    os.replace(tmp, PROGRESS_FILE)

# ========== 主流程 ==========
def main():
    print(f"加载模型: {MODEL_PATH}")
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
    except ImportError:
        device = "cpu"
    model = BGEM3FlagModel(
        model_name_or_path=MODEL_PATH,
        use_fp16=False,
        trust_remote_code=True,
        devices=device,
    )
    print("模型加载完成。")

    if not os.path.exists(CLASSIFICATION_FILE):
        print(f"错误: 分类结果文件 '{CLASSIFICATION_FILE}' 不存在。请先运行 GUI 生成。")
        return
    with open(CLASSIFICATION_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    corpus_info = data.get('corpus_info')
    files = data.get('files', [])
    if not corpus_info:
        print("错误: JSON 缺少 'corpus_info' 字段。")
        return
    if not files:
        print("警告: JSON 中 'files' 为空。")

    corpus_name = corpus_info.get('name')
    corpus_version = corpus_info.get('version')
    if not corpus_name or not corpus_version:
        print("错误: corpus_info 必须包含 name 和 version")
        return

    progress = load_progress(expected_name=corpus_name, expected_version=corpus_version)
    completed_set = set(progress.get("completed_files", []))

    conn = get_db_connection()
    print("数据库连接成功。")

    try:
        if progress.get("corpus_id"):
            corpus_id = progress["corpus_id"]
            print(f"从进度文件恢复 corpus_id: {corpus_id}")
        else:
            corpus_id = process_corpus(conn, corpus_info)
            progress["corpus_id"] = corpus_id
            save_progress(progress, corpus_name=corpus_name, corpus_version=corpus_version)

        for entry in files:
            filename = entry.get('file')
            chips = entry.get('chips', [])
            categories = entry.get('categories', [])
            if not filename:
                print("警告: 条目缺少 'file' 字段，跳过。")
                continue

            if filename in completed_set:
                print(f"⏭ 跳过已完成: {filename}")
                continue

            file_path = os.path.join(CHUNK_DIR, filename)
            if not os.path.exists(file_path):
                print(f"警告: 文件 '{file_path}' 不存在，跳过。")
                continue

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if not content.strip():
                print(f"警告: {file_path} 内容为空，跳过。")
                continue

            title, images = extract_title_and_images(content)
            images_json = extras.Json(images) if images else None

            try:
                result = model.encode(
                    content,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
                dense_vecs = result['dense_vecs']
                lexical_weights = result['lexical_weights']

                if isinstance(dense_vecs, np.ndarray):
                    if dense_vecs.ndim == 2:
                        dense_vec = dense_vecs[0]
                    else:
                        dense_vec = dense_vecs
                else:
                    dense_vec = np.array(dense_vecs)
                    if dense_vec.ndim == 2:
                        dense_vec = dense_vec[0]

                if isinstance(lexical_weights, list) and len(lexical_weights) > 0:
                    lexical_weights = lexical_weights[0]
                elif not isinstance(lexical_weights, dict):
                    lexical_weights = {}

                norm = np.linalg.norm(dense_vec)
                if norm > 0:
                    dense_vec = dense_vec / norm

                dense_vec_str = '[' + ','.join(map(str, dense_vec.tolist())) + ']'
                sparse_vec_str = format_sparse_vec(lexical_weights, dim=250002)
                print(f"✓ 生成向量成功: {filename}")
            except Exception as e:
                print(f"编码失败: {e}，跳过此文件。")
                continue

            index_list = chips + categories
            chunk_uuid = str(uuid.uuid4())

            try:
                with conn:
                    with conn.cursor() as cur:
                        insert_sql = sql.SQL("""
                            INSERT INTO chip_chunks (chunk_id, content, vector, images, title, sparse_embedding, index, corpus_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """)
                        cur.execute(insert_sql, (
                            chunk_uuid, content, dense_vec_str, images_json,
                            title, sparse_vec_str, index_list, corpus_id
                        ))
                        print(f"✓ 插入 chunk: {chunk_uuid} (来自 {filename})")

                    if chips:
                        for chip in chips:
                            ensure_chip_exists(conn, chip)
                            for cat in categories:
                                cat_lower = cat.lower()
                                if cat_lower in ALL_CATEGORIES:
                                    try:
                                        update_chip_category(conn, chip, cat_lower, chunk_uuid)
                                    except Exception as e:
                                        print(f"更新分类失败: {e}")
                                        raise
                                else:
                                    print(f"警告: 分类 '{cat}' 不在预定义列表中，跳过。")
                    else:
                        print(f"注意: 文件 '{filename}' 没有指定芯片型号，未进行关联。")

                completed_set.add(filename)
                progress["completed_files"] = list(completed_set)
                save_progress(progress, corpus_name=corpus_name, corpus_version=corpus_version)
                print(f"✓ 完成并记录进度: {filename}")

            except Exception as e:
                print(f"处理文件 {filename} 失败，事务已回滚: {e}")
                continue

        print("\n所有分片处理完成！")
    except Exception as e:
        print(f"发生严重错误: {e}")
        raise
    finally:
        conn.close()
        print("数据库连接已关闭。")

if __name__ == "__main__":
    main()