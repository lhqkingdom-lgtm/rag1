-- 芯片基础信息
CREATE TABLE IF NOT EXISTS ssot_mcu_chips (
    chip_id VARCHAR(50) PRIMARY KEY,
    series VARCHAR(100),
    core VARCHAR(50),
    flash_kb INTEGER,
    ram_kb INTEGER,
    max_freq_mhz INTEGER,
    package_types JSONB,
    voltage_min NUMERIC(8,3),
    voltage_max NUMERIC(8,3),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 引脚复用
CREATE TABLE IF NOT EXISTS ssot_mcu_pin_mux (
    chip_id VARCHAR(50) REFERENCES ssot_mcu_chips(chip_id) ON DELETE CASCADE,
    pin_name VARCHAR(20),
    pin_number INTEGER,
    function_name VARCHAR(50),
    is_supported BOOLEAN DEFAULT TRUE,
    af_code INTEGER,
    default_function VARCHAR(50),
    constraints JSONB,
    PRIMARY KEY (chip_id, pin_name, function_name)
);

-- 外设
CREATE TABLE IF NOT EXISTS ssot_mcu_peripherals (
    chip_id VARCHAR(50) REFERENCES ssot_mcu_chips(chip_id) ON DELETE CASCADE,
    peripheral_type VARCHAR(50),
    instance_count INTEGER,
    max_resolution INTEGER,
    max_frequency_mhz INTEGER,
    features JSONB,
    PRIMARY KEY (chip_id, peripheral_type)
);

-- 电气参数
CREATE TABLE IF NOT EXISTS ssot_mcu_electrical (
    chip_id VARCHAR(50) REFERENCES ssot_mcu_chips(chip_id) ON DELETE CASCADE,
    param_name VARCHAR(100),
    min_value NUMERIC(12,6),
    max_value NUMERIC(12,6),
    unit VARCHAR(20),
    conditions JSONB,
    PRIMARY KEY (chip_id, param_name)
);

-- 存储器配置
CREATE TABLE IF NOT EXISTS ssot_mcu_memory (
    chip_id VARCHAR(50) REFERENCES ssot_mcu_chips(chip_id) ON DELETE CASCADE,
    memory_type VARCHAR(50),
    size_kb NUMERIC(10,2),
    organization VARCHAR(100),
    features JSONB,
    PRIMARY KEY (chip_id, memory_type)
);

-- 可选：为常用查询添加索引
CREATE INDEX IF NOT EXISTS idx_pin_mux_chip_pin ON ssot_mcu_pin_mux(chip_id, pin_name);
CREATE INDEX IF NOT EXISTS idx_peripherals_chip_type ON ssot_mcu_peripherals(chip_id, peripheral_type);
CREATE INDEX IF NOT EXISTS idx_electrical_chip ON ssot_mcu_electrical(chip_id);
CREATE INDEX IF NOT EXISTS idx_memory_chip ON ssot_mcu_memory(chip_id);