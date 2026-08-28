#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from langchain_text_splitters import MarkdownHeaderTextSplitter

def split_by_headers(content, headers_to_split_on=None):
    """
    按一级和二级标题分割 Markdown 内容。
    返回列表，每个元素为 {"content": str, "metadata": dict}。
    自动过滤掉“只有标题、没有实际正文”的切片。
    """
    if headers_to_split_on is None:
        headers_to_split_on = [("#", "一级标题"), ("##", "二级标题")]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False        # 保留标题行在正文中
    )
    raw_splits = splitter.split_text(content)

    # 过滤无正文的切片（去掉所有标题行后为空）
    header_pattern = re.compile(r'^(#|##)\s+.*$', re.MULTILINE)
    def has_actual_content(text):
        return header_pattern.sub('', text).strip() != ""

    valid_splits = [doc for doc in raw_splits if has_actual_content(doc.page_content)]

    return [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in valid_splits
    ]