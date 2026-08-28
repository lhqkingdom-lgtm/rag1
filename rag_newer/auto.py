import os
import re
import json
import uuid
import argparse
from pathlib import Path
import psycopg2
from psycopg2 import sql, extras
import numpy as np
from FlagEmbedding import BGEM3FlagModel

# ========== 配置 ==========
CONFIG_FILE = 'config/db_config.json'
MODEL_PATH = r'E:\moshang\RAG_new-master\rag_new\modelscope_models\bge-m3'
PROGRESS_FILE = "import_progress.json"

ALL_CATEGORIES = [
    "pwm", "adc", "dac", "i2c", "uart", "spi", "can", "usb",
    "lcd", "dma", "rtc", "ble", "wifi", "memory", "wdt",
    "gpio", "timer", "interrupt_system", "system_clock",
    "power", "low_power_mode", "touch", "electrical_parameters",
    "package_type", "cpu", "routine", "overview", "other"
]

# ========== 辅助函数 ==========
def load_db_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def get_db_connection(config_file):
    config = load_db_config(config_file)
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
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"progress": {}}

def save_progress(progress_dict):
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(progress_dict, f, indent=2, ensure_ascii=False)
    os.replace(tmp, PROGRESS_FILE)

# ========== 核心处理：处理一个 result.json + chunks 目录 ==========
def process_corpus_data(result_file, chunk_dir, model, conn, progress_dict):
    """
    result_file: Path 对象，指向 result.json
    chunk_dir: Path 对象，指向 chunks 目录
    返回 (成功处理的文件数, 总文件数)
    """
    if not result_file.exists():
        print(f"⚠️ {result_file} 不存在")
        return 0, 0
    if not chunk_dir.exists():
        print(f"⚠️ {chunk_dir} 不存在")
        return 0, 0

    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    corpus_info = data.get('corpus_info')
    files = data.get('files', [])
    if not corpus_info:
        print(f"⚠️ {result_file} 缺少 corpus_info，跳过")
        return 0, 0

    corpus_name = corpus_info.get('name')
    corpus_version = corpus_info.get('version')
    if not corpus_name or not corpus_version:
        print(f"⚠️ {result_file} corpus_info 缺少 name 或 version，跳过")
        return 0, 0

    key = f"{corpus_name}#{corpus_version}"
    # 从进度字典中获取该语料已完成的文件列表
    completed_set = set(progress_dict.get(key, {}).get("completed_files", []))
    saved_corpus_id = progress_dict.get(key, {}).get("corpus_id")

    if saved_corpus_id:
        corpus_id = saved_corpus_id
        print(f"从进度恢复 corpus_id: {corpus_id} (语料: {key})")
    else:
        corpus_id = process_corpus(conn, corpus_info)
        if key not in progress_dict:
            progress_dict[key] = {}
        progress_dict[key]["corpus_id"] = corpus_id
        save_progress(progress_dict)

    total = len(files)
    processed = 0

    for entry in files:
        filename = entry.get('file')
        chips = entry.get('chips', [])
        categories = entry.get('categories', [])
        if not filename:
            print(f"  警告: 条目缺少 'file' 字段，跳过。")
            continue

        if filename in completed_set:
            print(f"  ⏭ 跳过已完成: {filename}")
            processed += 1
            continue

        file_path = chunk_dir / filename
        if not file_path.exists():
            print(f"  ⚠️ 文件 '{file_path}' 不存在，跳过。")
            continue

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        if not content.strip():
            print(f"  ⚠️ {file_path} 内容为空，跳过。")
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
            print(f"  ✓ 生成向量成功: {filename}")
        except Exception as e:
            print(f"  ❌ 编码失败: {e}，跳过此文件。")
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
                    print(f"  ✓ 插入 chunk: {chunk_uuid} (来自 {filename})")

                if chips:
                    for chip in chips:
                        ensure_chip_exists(conn, chip)
                        for cat in categories:
                            cat_lower = cat.lower()
                            if cat_lower in ALL_CATEGORIES:
                                try:
                                    update_chip_category(conn, chip, cat_lower, chunk_uuid)
                                except Exception as e:
                                    print(f"  ❌ 更新分类失败: {e}")
                                    raise
                            else:
                                print(f"  ⚠️ 分类 '{cat}' 不在预定义列表中，跳过。")
                else:
                    print(f"  注意: 文件 '{filename}' 没有指定芯片型号，未进行关联。")

            # 更新进度（以分片文件为单位）
            completed_set.add(filename)
            progress_dict[key]["completed_files"] = list(completed_set)
            save_progress(progress_dict)
            processed += 1
            print(f"  ✅ 完成并记录进度: {filename}")

        except Exception as e:
            print(f"  ❌ 处理文件 {filename} 失败，事务已回滚: {e}")
            continue

    return processed, total

# ========== 主入口 ==========
def main():
    global PROGRESS_FILE  # 必须放在函数开头，以便后续修改全局变量

    parser = argparse.ArgumentParser(description="批量入库脚本：支持单语料或根目录批量处理")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--root-dir", help="批量模式：根目录，每个子文件夹为一个语料（包含 result.json 和 chunks）")
    group.add_argument("--classification-file", help="单文件模式：result.json 路径")
    parser.add_argument("--chunk-dir", help="单文件模式：chunks 目录路径（默认 ./output_chunks）")
    parser.add_argument("--model-path", default=MODEL_PATH, help="模型路径")
    parser.add_argument("--db-config", default=CONFIG_FILE, help="数据库配置文件")
    parser.add_argument("--progress-file", default=PROGRESS_FILE, help="进度文件路径")
    args = parser.parse_args()

    # 更新全局进度文件路径
    PROGRESS_FILE = args.progress_file

    # 加载模型
    print(f"加载模型: {args.model_path}")
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
    except ImportError:
        device = "cpu"
    model = BGEM3FlagModel(
        model_name_or_path=args.model_path,
        use_fp16=False,
        trust_remote_code=True,
        devices=device,
    )
    print("模型加载完成。")

    # 连接数据库
    conn = get_db_connection(args.db_config)
    print("数据库连接成功。")

    # 加载全局进度
    progress_dict = load_progress()
    print(f"当前进度记录数: {len(progress_dict.get('progress', {}))}")

    try:
        if args.root_dir:
            # 批量模式
            root = Path(args.root_dir)
            if not root.exists() or not root.is_dir():
                print(f"错误: 根目录 {root} 不存在或不是文件夹")
                return
            # 查找所有包含 result.json 的子目录
            subdirs = [d for d in root.iterdir() if d.is_dir() and (d / "result.json").exists()]
            if not subdirs:
                print("未找到任何包含 result.json 的子目录，退出。")
                return
            print(f"找到 {len(subdirs)} 个语料目录，开始批量处理...")
            total_processed = 0
            total_files = 0
            for sub in subdirs:
                print(f"\n--- 处理语料: {sub.name} ---")
                result_file = sub / "result.json"
                chunk_dir = sub / "chunks"
                p, t = process_corpus_data(result_file, chunk_dir, model, conn, progress_dict["progress"])
                total_processed += p
                total_files += t
            print(f"\n✅ 批量处理完成。总进度: {total_processed}/{total_files} 个分片已入库。")
        else:
            # 单文件模式
            result_file = Path(args.classification_file)
            chunk_dir = Path(args.chunk_dir) if args.chunk_dir else Path("./output_chunks")
            if not result_file.exists():
                print(f"错误: {result_file} 不存在")
                return
            if not chunk_dir.exists():
                print(f"错误: {chunk_dir} 不存在")
                return
            print(f"单文件模式: result={result_file}, chunks={chunk_dir}")
            p, t = process_corpus_data(result_file, chunk_dir, model, conn, progress_dict["progress"])
            print(f"\n✅ 单语料处理完成，成功入库 {p}/{t} 个分片。")
    finally:
        conn.close()
        print("数据库连接已关闭。")

if __name__ == "__main__":
    main()