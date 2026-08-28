-- 1. 创建数据库（请替换 your_database_name）
CREATE DATABASE test
    WITH 
    OWNER = postgres
    TEMPLATE = template0   
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

-- 2. 连接到新数据库
\c test

-- 3. 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 4. 创建 category 表（不变）
CREATE TABLE category (
    "id"          SERIAL PRIMARY KEY,
    "name"        TEXT,
    "description" TEXT
);

-- 5. 创建 chip_chunks 表（存储详细参数、向量等）
-- 注意：chunk_id 是文本型唯一标识，建议添加唯一约束以便外键引用
CREATE TABLE chip_chunks (
    "title"     TEXT,
    "content"   TEXT,
    "images"    TEXT,
    "chunk_id"  TEXT PRIMARY KEY,   -- 添加唯一约束，方便 chip 表引用
    "corpus_id" TEXT,
    "vector"    vector(1024),
    "sparse_embedding" sparsevec(250002),
    "index"     TEXT[]
);

-- 6. 创建 chip 表，所有参数列存储 chip_chunks.chunk_id（TEXT 类型）
CREATE TABLE RAG_chips (
    "chip_name" TEXT PRIMARY KEY,
    "PWM" TEXT[],
    "ADC" TEXT[],
    "DAC" TEXT[],
    "I2C" TEXT[],
    "UART" TEXT[],
    "SPI" TEXT[],
    "CAN" TEXT[],
    "USB" TEXT[],
    "LCD" TEXT[],
    "DMA" TEXT[],
    "RTC" TEXT[],
    "BLE" TEXT[],
    "WIFI" TEXT[],
    "Memory" TEXT[],
    "WDT" TEXT[],
    "GPIO" TEXT[],
    "Timer" TEXT[],
    "Interrupt_system" TEXT[],
    "System_clock" TEXT[],
    "Power" TEXT[],
    "Low_power_mode" TEXT[],
    "Touch" TEXT[],
    "Electrical_parameters" TEXT[],
    "Package_type" TEXT[],
    "CPU" TEXT[],
    "Routine" TEXT[],
    "Overview" TEXT[],
    "Other" TEXT[]
);

CREATE TABLE corpus (
    "corpus_id"    TEXT PRIMARY KEY,           
    "name"         TEXT,
    "description"  TEXT,
    "category"     TEXT,
    "file_path"    TEXT,
    "is_active"    BOOLEAN,                   
    "version"      TEXT,
    "created_at"   TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP NOT NULL,
    "operator"   TEXT
);