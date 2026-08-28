#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from bs4 import BeautifulSoup
import json

def html_table_to_markdown(table_tag):
    """将 BeautifulSoup <table> 转换为 Markdown 表格字符串"""
    rows = table_tag.find_all('tr')
    if not rows:
        return ""

    table_data = []
    for row in rows:
        cells = row.find_all(['td', 'th'])
        row_data = [cell.get_text(strip=True) for cell in cells]
        table_data.append(row_data)

    if not table_data:
        return ""

    col_count = max(len(row) for row in table_data)
    for row in table_data:
        while len(row) < col_count:
            row.append("")

    header = table_data[0]
    separator = ["---"] * col_count

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(separator) + " |")
    for row in table_data[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

def convert_html_tables_to_markdown(content):
    """将内容中所有 HTML 表格替换为 Markdown 表格"""
    soup = BeautifulSoup(content, 'html.parser')
    tables = soup.find_all('table')
    if not tables:
        return content
    for table in tables:
        md_table = html_table_to_markdown(table)
        table.replace_with(BeautifulSoup(md_table, 'html.parser'))
    return str(soup)

def clean_content(content, rules=None):
    """
    清洗 Markdown 内容：
    - 将 HTML 表格转为 Markdown 表格
    - 若 rules 为 dict，则进行字符串替换（old → new）
    """
    content = convert_html_tables_to_markdown(content)
    if rules:
        with open(rules, 'r', encoding='utf-8') as f:
            rules = json.load(f) 
        for old, new in rules.items():
            content = content.replace(old, new)
    return content