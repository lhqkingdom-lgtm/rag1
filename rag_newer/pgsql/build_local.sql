CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS category (
    id SERIAL PRIMARY KEY,
    name TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS corpus (
    corpus_id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    category TEXT,
    file_path TEXT,
    is_active BOOLEAN,
    version TEXT,
    created_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP NOT NULL,
    operator TEXT,
    is_review BOOLEAN DEFAULT FALSE,
    reviewer TEXT
);

CREATE TABLE IF NOT EXISTS chip_chunks (
    title TEXT,
    content TEXT,
    images TEXT,
    chunk_id TEXT PRIMARY KEY,
    corpus_id TEXT,
    vector vector(1024),
    sparse_embedding sparsevec(250002),
    "index" TEXT[],
    is_review BOOLEAN DEFAULT FALSE,
    update_time TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rag_chips (
    chip_name TEXT PRIMARY KEY,
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

