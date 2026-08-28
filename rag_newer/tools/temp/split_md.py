#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split a Markdown file into chunks:
- First split by first-level headings (#).
- For each # block, if it contains second-level headings (##), split further by ##.
- If no ## inside, keep the whole # block as one chunk.
Each output file starts with the top-level heading for context.
"""

import os
import re
import argparse
import unicodedata

def slugify(text):
    """Convert text to a safe filename."""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\-_\. ]', '_', text)
    text = re.sub(r'[ \-]+', '_', text)
    return text.strip('_')

def split_md(input_file, output_dir, prefix=''):
    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Regular expressions for headings at line start (allow spaces)
    h1_pattern = re.compile(r'^#\s+(.*)')    # one # at start
    h2_pattern = re.compile(r'^##\s+(.*)')   # two ## at start

    # Step 1: Split by first-level headings (#)
    h1_indices = []
    for i, line in enumerate(lines):
        if h1_pattern.match(line):
            h1_indices.append(i)

    # If no H1, treat whole file as a single block (no top heading)
    if not h1_indices:
        h1_indices = [0]  # dummy start

    blocks = []  # list of (h1_title, start_line_idx, end_line_idx_exclusive)
    for idx, start in enumerate(h1_indices):
        end = h1_indices[idx+1] if idx+1 < len(h1_indices) else len(lines)
        title_match = h1_pattern.match(lines[start]) if lines[start].strip() else None
        title = title_match.group(1).strip() if title_match else None
        blocks.append((title, start, end))

    # Step 2: For each H1 block, find H2 headings inside and split if any
    chunks = []  # each chunk: (heading_text, content_lines)
    for h1_title, start, end in blocks:
        # Extract lines of this block
        block_lines = lines[start:end]
        # Find H2 positions within this block (relative to block start)
        h2_positions = []
        for rel_i, line in enumerate(block_lines):
            if h2_pattern.match(line):
                h2_positions.append(rel_i)

        if not h2_positions:
            # No H2, keep whole block as one chunk
            # Include H1 title as heading (if present)
            heading = h1_title if h1_title else "intro"
            chunks.append((heading, block_lines))
        else:
            # Split by H2, each chunk gets H1 title at top for context
            for idx, rel_start in enumerate(h2_positions):
                rel_end = h2_positions[idx+1] if idx+1 < len(h2_positions) else len(block_lines)
                # Content: H1 title line + H2 heading and its content
                # We'll include the H1 line at the top, then the H2 sub-block
                h1_line = f"# {h1_title}\n" if h1_title else ""
                h2_sub_block = block_lines[rel_start:rel_end]
                # Build chunk content
                content = []
                if h1_line:
                    content.append(h1_line)
                    # Optionally add a blank line to separate
                    # content.append("\n")
                content.extend(h2_sub_block)
                # Heading for filename: use H2 title
                h2_match = h2_pattern.match(block_lines[rel_start])
                h2_title = h2_match.group(1).strip() if h2_match else f"h2_{idx+1:02d}"
                chunks.append((h2_title, content))

    # Write chunks to files
    for idx, (heading, content) in enumerate(chunks, start=1):
        safe_name = slugify(heading) if heading else f"chunk_{idx:02d}"
        if prefix:
            filename = f"{prefix}_{idx:02d}_{safe_name}.md"
        else:
            filename = f"{idx:02d}_{safe_name}.md"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(content)
        print(f"Written: {filepath}")

    print(f"Total chunks: {len(chunks)}")

def main():
    parser = argparse.ArgumentParser(description="Split Markdown by # and further by ## if present.")
    parser.add_argument('input', help="Input Markdown file path")
    parser.add_argument('-o', '--output', default='output_chunks', help="Output directory (default: output_chunks)")
    parser.add_argument('-p', '--prefix', default='', help="Optional prefix for output filenames")
    args = parser.parse_args()
    split_md(args.input, args.output, args.prefix)

if __name__ == '__main__':
    main()