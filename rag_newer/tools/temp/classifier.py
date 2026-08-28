#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re

# 默认内容分类规则（与 classify_gui 保持一致）
DEFAULT_PATTERNS = {
    "PWM": [r'\bPWM\b', r'脉宽调制'],
    "ADC": [r'\bADC\b', r'模数转换'],
    "DAC": [r'\bDAC\b', r'数模转换'],
    "I2C": [r'\bI2C\b', r'IIC', r"I²C"],
    "UART": [r'\bUART\b', r'引脚', r'UART'],
    "SPI": [r'\bSPI\b'],
    "CAN": [r'\bCAN\b'],
    "USB": [r'\bUSB\b'],
    "LCD": [r'\bLCD\b', r'液晶'],
    "DMA": [r'\bDMA\b'],
    "RTC": [r'\bRTC\b', r'实时时钟'],
    "BLE": [r'\bBLE\b', r'蓝牙'],
    "WIFI": [r'\bWiFi\b', r'Wi-Fi', r'无线网络'],
    "Memory": [r'\bMemory\b', r'存储', r'Flash', r'RAM', r'EEPROM'],
    "WDT": [r'\bWDT\b', r'看门狗', '复位'],
    "GPIO": [r'\bGPIO\b', r'通用输入输出'],
    "Timer": [r'\bTimer\b', r'定时器'],
    "Interrupt_system": [r'\bInterrupt\b', r'中断系统', r'中断'],
    "System_clock": [r'\bSystem clock\b', r'系统时钟', r'时钟'],
    "Power": [r'\bPower\b', r'LVD', r'电源', r'供电'],
    "Low_power_mode": [r'\bLow power\b', r'低功耗', r'省电', r'TMC'],
    "Touch": [r'\bTouch\b', r'触摸'],
    "Electrical_parameters": [r'\bElectrical\b', r'电气特性', r'参数'],
    "Package_type": [r'\bPackage\b', r'封装'],
    "CPU": [r'\bCPU\b', r'中央处理器', r'核'],
    "Routine": [r'\bRoutine\b', r'示例', r'例程'],
    "Overview": [r'\bOverview\b', r'概述', r'介绍'],
}

CONFIG_JSON = "category_patterns.json"

def load_category_patterns():
    """加载内容分类规则（优先从 category_patterns.json 读取）"""
    if os.path.exists(CONFIG_JSON):
        try:
            with open(CONFIG_JSON, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            # 确保值是列表
            for k, v in loaded.items():
                if not isinstance(v, list):
                    loaded[k] = [v] if isinstance(v, str) else []
            return loaded
        except Exception as e:
            print(f"读取 {CONFIG_JSON} 失败: {e}，使用默认规则")
    # 不存在或读取失败，写默认文件并返回
    try:
        with open(CONFIG_JSON, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_PATTERNS, f, ensure_ascii=False, indent=2)
        print(f"已生成默认分类规则文件：{CONFIG_JSON}，您可以编辑它。")
    except Exception as e:
        print(f"无法写入 {CONFIG_JSON}: {e}")
    return DEFAULT_PATTERNS.copy()

def classify_all(text, patterns=None):
    """返回文本匹配到的所有分类（去重），若无匹配则返回 ['Other']"""
    if patterns is None:
        patterns = load_category_patterns()
    matched = set()
    for category, pats in patterns.items():
        for pat in pats:
            if re.search(pat, text, re.IGNORECASE):
                matched.add(category)
                break
    if not matched:
        return ["Other"]
    return sorted(matched)