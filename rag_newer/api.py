import os
import re
import json
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import psycopg2
from psycopg2 import sql
from FlagEmbedding import BGEM3FlagModel

# ================== 配置 ==================
BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = Path(os.getenv("RAG_DB_CONFIG", BASE_DIR / "config" / "db_config.json"))
KEYWORDS_FILE = Path(os.getenv("RAG_KEYWORDS_FILE", BASE_DIR / "config" / "keywords.json"))
MODEL_PATH = os.getenv("RAG_MODEL_PATH", "BAAI/bge-m3")

# 所有可用的分类主题（小写，用于校验）
ALL_CATEGORIES = [
    "pwm", "adc", "dac", "i2c", "uart", "spi", "can", "usb",
    "lcd", "dma", "rtc", "ble", "wifi", "memory", "wdt",
    "gpio", "timer", "interrupt_system", "system_clock",
    "power", "low_power_mode", "touch", "electrical_parameters",
    "package_type", "cpu", "routine", "overview", "other"
]

# ================== Pydantic 模型 ==================
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    chips: List[str] = []                # 改为空列表默认值，便于判空
    topics: List[str] = []               # 改为空列表默认值

class SearchResultItem(BaseModel):
    chunk_id: str
    content: str
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResultItem]

# ================== FastAPI 应用 ==================
app = FastAPI(title="Chip RAG Search API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== 全局变量 ==================
model = None
db_config = None
keywords_config = None          # 存储 keywords.json 内容

# ================== 辅助函数 ==================
def load_db_config():
    config = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        config = raw.get("postgresql", raw)

    # 环境变量优先，避免把生产数据库密码写进仓库。
    env_mapping = {
        "host": "RAG_DB_HOST",
        "port": "RAG_DB_PORT",
        "dbname": "RAG_DB_NAME",
        "user": "RAG_DB_USER",
        "password": "RAG_DB_PASSWORD",
    }
    if "database" in config and "dbname" not in config:
        config["dbname"] = config.pop("database")
    for key, env_name in env_mapping.items():
        value = os.getenv(env_name)
        if value:
            config[key] = int(value) if key == "port" else value

    missing = [key for key in ("host", "port", "dbname", "user", "password") if not config.get(key)]
    if missing:
        raise RuntimeError(f"数据库配置缺少字段: {', '.join(missing)}")
    return config

def load_keywords_config():
    """加载关键词正则配置文件"""
    try:
        with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告: {KEYWORDS_FILE} 未找到，将无法自动提取芯片/主题。")
        return {"chips": {}, "topics": {}}

def get_db_connection():
    conn = psycopg2.connect(**db_config, connect_timeout=8)
    # 在线检索服务只允许查询，防止误执行任何写入语句。
    conn.set_session(readonly=True, autocommit=True)
    return conn

def format_sparse_vec(lexical_weights: dict, dim: int = 250002) -> str:
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

def get_actual_column(conn, topic_lower: str) -> str:
    """根据小写主题名获取 rag_chips 表中的实际列名"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'rag_chips' AND LOWER(column_name) = %s
            """,
            (topic_lower,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"主题 '{topic_lower}' 在 rag_chips 表中不存在")
        return row[0]

def extract_entities(query: str):
    if not keywords_config:
        return [], []

    found_chips = set()
    found_topics = set()

    # 1. 先尝试精确匹配
    exact = keywords_config.get("chips", {}).get("exact", {})
    for chip_name, patterns in exact.items():
        if isinstance(patterns, str):
            patterns = [patterns]
        for pat in patterns:
            if re.search(pat, query, re.IGNORECASE):
                found_chips.add(chip_name)
                break
    if found_chips:
        # 提取主题（不受芯片影响）
        for topic_name, patterns in keywords_config.get("topics", {}).items():
            if isinstance(patterns, str):
                patterns = [patterns]
            for pat in patterns:
                if re.search(pat, query, re.IGNORECASE):
                    found_topics.add(topic_name)
                    break
        return list(found_chips), list(found_topics)

    # 2. 精确未命中，尝试宽松匹配（前缀）
    fuzzy = keywords_config.get("chips", {}).get("fuzzy", {})
    for group_key, group_info in fuzzy.items():
        pattern = group_info.get("pattern", "")
        if not pattern:
            continue
        if re.search(pattern, query, re.IGNORECASE):
            # 将该组所有芯片型号加入结果
            for chip in group_info.get("chips", []):
                found_chips.add(chip)
            break  # 只取第一个匹配的前缀（可调整，但一般只有一个）

    # 提取主题
    for topic_name, patterns in keywords_config.get("topics", {}).items():
        if isinstance(patterns, str):
            patterns = [patterns]
        for pat in patterns:
            if re.search(pat, query, re.IGNORECASE):
                found_topics.add(topic_name)
                break

    return list(found_chips), list(found_topics)

# ================== 启动事件 ==================
@app.on_event("startup")
def startup():
    global model, db_config, keywords_config
    db_config = load_db_config()
    keywords_config = load_keywords_config()

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"使用设备: {device}，加载模型 {MODEL_PATH} ...")
    model = BGEM3FlagModel(
        model_name_or_path=MODEL_PATH,
        use_fp16=False,
        trust_remote_code=True,
        devices=device,
    )
    print("模型加载完成。")

# ================== 健康检查 ==================
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# ================== 核心检索接口 ==================
@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    if not model:
        raise HTTPException(status_code=503, detail="模型尚未加载")

    # ----- 步骤0：自动提取缺失的 chips / topics -----
    extracted_chips, extracted_topics = extract_entities(request.query)

    # 确定最终使用的芯片列表和主题列表
    effective_chips = request.chips if request.chips else extracted_chips
    effective_topics = request.topics if request.topics else extracted_topics

    # 如果两者都提取不到（且请求中也没给），则报错
    if not effective_chips and not effective_topics:
        raise HTTPException(
            status_code=400,
            detail="无法从 query 中自动提取 chips 或 topics，且请求中也未提供，请手动指定。"
        )

    # 对最终主题做合法性校验（若有）
    for t in effective_topics:
        if t.lower() not in ALL_CATEGORIES:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的 topic: {t}，可选: {ALL_CATEGORIES}"
            )

    conn = None
    try:
        conn = get_db_connection()

        # ----- 步骤1：根据 effective_chips 和 effective_topics 收集所有候选 chunk_id -----
        # 若 effective_topics 为空，则查询所有主题（即整行）
        target_topics = effective_topics if effective_topics else ALL_CATEGORIES
        chunk_ids = set()

        with conn.cursor() as cur:
            for topic in target_topics:
                actual_col = get_actual_column(conn, topic.lower())

                # 若 effective_chips 为空，则查询所有芯片（即整列）
                if effective_chips:
                    select_sql = sql.SQL(
                        "SELECT {col} FROM rag_chips WHERE chip_name = ANY(%s)"
                    ).format(col=sql.Identifier(actual_col))
                    cur.execute(select_sql, (effective_chips,))
                else:
                    select_sql = sql.SQL(
                        "SELECT {col} FROM rag_chips"
                    ).format(col=sql.Identifier(actual_col))
                    cur.execute(select_sql)

                rows = cur.fetchall()
                for row in rows:
                    if row[0] is not None:
                        chunk_ids.update(row[0])   # row[0] 是 TEXT[]，转为 Python list

        if not chunk_ids:
            return SearchResponse(results=[])

        # ----- 步骤2：编码 query -----
        enc_result = model.encode(
            request.query,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vecs = enc_result["dense_vecs"]
        if isinstance(dense_vecs, np.ndarray):
            dense_vec = dense_vecs[0] if dense_vecs.ndim == 2 else dense_vecs
        else:
            dense_vec = np.array(dense_vecs)
            dense_vec = dense_vec[0] if dense_vec.ndim == 2 else dense_vec
        norm = np.linalg.norm(dense_vec)
        if norm > 0:
            dense_vec = dense_vec / norm
        dense_vec_str = "[" + ",".join(map(str, dense_vec.tolist())) + "]"

        lexical_weights = enc_result["lexical_weights"]
        if isinstance(lexical_weights, list) and len(lexical_weights) > 0:
            lexical_weights = lexical_weights[0]
        elif not isinstance(lexical_weights, dict):
            lexical_weights = {}
        sparse_vec_str = format_sparse_vec(lexical_weights, dim=250002)

        # ----- 步骤3：混合检索排序 -----
        with conn.cursor() as cur:
            query_sql = sql.SQL("""
                SELECT
                    chunk_id,
                    content,
                    1 - (vector <=> %s) AS dense_sim,
                    1 - (sparse_embedding <=> %s) AS sparse_sim,
                    (1 - (vector <=> %s)) * 0.5 + (1 - (sparse_embedding <=> %s)) * 0.5 AS combined
                FROM chip_chunks
                WHERE chunk_id = ANY(%s)
                ORDER BY combined DESC
                LIMIT %s
            """)
            cur.execute(
                query_sql,
                (
                    dense_vec_str,
                    sparse_vec_str,
                    dense_vec_str,
                    sparse_vec_str,
                    list(chunk_ids),
                    request.top_k,
                ),
            )
            rows = cur.fetchall()
            results = [
                SearchResultItem(
                    chunk_id=row[0],
                    content=row[1],
                    score=row[4]
                )
                for row in rows
            ]
            return SearchResponse(results=results)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")
    finally:
        if conn:
            conn.close()

# ================== 主程序 ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
