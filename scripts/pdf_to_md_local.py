# scripts/pdf_to_md_local.py
"""
本地 PDF → Markdown 转换器（MinerU 排队时的替代通道）
====================================================================
为什么需要它：
    MinerU 公共云今天对这三个任务一页都没开始解析（网页版与 API 同时卡住），
    而知识库为空会让 RAGAS 评估整条链路跑不通。实测这三本设备手册都是
    「带文本层的数字 PDF」（非扫描件），文本可以直接抽出来，标题层级可以本地还原。
    本机 i5-13500H 无独显，本地部署 MinerU 也不划算 —— 这条纯 CPU 通道是首选。

标题怎么认（两路信号，都是实测出来的）：
    1. 字号：比正文字号（字符量众数）更大的字号，按从大到小映射成 ##/###/####
       —— H1 留给文件名，避免一级标题被封面大字占用
    2. 加粗：与正文同号但字体名含 Bold 的短行（如「直流电压测量」「规格」）
       注意 PyMuPDF 的 flags 位对中文 YaHei-Bold 不置位，只能读字体名
    3. 排除：含冒号或句末标点的行（「警告: 为防触电…」是正文警示，不是标题）

它做不到什么（和 MinerU 的差异，必须知道）：
    - 表格不会还原成 Markdown 表格，会退化成逐行文本（参数表这类内容召回会受影响）
    - 复杂多栏版面按坐标排序，可能有轻微乱序
    - 扫描件/纯图片 PDF 抽不出内容（结尾会打印平均每页文本量告警）

用法（pymupdf 不在项目依赖里，用 --with 临时注入）：
    uv run --with pymupdf python scripts/pdf_to_md_local.py ^
        --pdf "D:\\path\\to\\x.pdf" --out-dir temp-files/local_md
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    print("缺少 PyMuPDF，请用: uv run --with pymupdf python scripts/pdf_to_md_local.py ...")
    raise SystemExit(1)

# 标题最多还原到几级（H1 保留给文件名，所以实际用 2~4）
MAX_HEADING_LEVEL = 4
# 字号判定：整行文本不超过这个长度才允许当标题
MAX_HEADING_CHARS = 40
# 加粗判定：与正文同号时，只有这么短的行才敢当标题
MAX_BOLD_HEADING_CHARS = 20
# 字号候选的最小字符总量，防止个别装饰性大字/水印成为一级标题
MIN_HEADING_SIZE_CHARS = 10
# 出现这些字符的行一定是正文句子，不是标题
NOT_HEADING_CHARS = ":：。！？；!?"
# 页眉页脚判定：某短行在超过该比例的页面上重复出现
HEADER_FOOTER_RATIO = 0.5
# 目录点线行：出现 4 个以上连续点（1.1 提示定义......1）说明是目录条目，不是正文
TOC_LEADER = re.compile(r"\.{4,}\s*\d*$")
# 纯章节编号标题（如「#### 1.1」），需把下一段首词拼上来当章节名
_SECTION_NUM = re.compile(r"^(#+) (\d+(?:\.\d+)*)$")
# 章节名长度上限，超过说明下一段首词不是标题而是正文
MAX_SECTION_NAME_CHARS = 20
# 单页文本少于此字数视为图片页/空白页（用于统计告警）
SPARSE_PAGE_CHARS = 20


def _iter_lines(doc):
    """按页、按块、按行遍历，产出 (页号, 行字典)"""
    for page_no, page in enumerate(doc):
        blocks = sorted(
            page.get_text("dict").get("blocks", []),
            key=lambda b: (round(b["bbox"][1]), round(b["bbox"][0])),
        )
        for block in blocks:
            for line in block.get("lines", []):
                yield page_no, line


def _line_info(line: dict) -> Tuple[str, int, bool]:
    """
    取一行的文本、行内最大字号、是否加粗
    加粗读字体名：PyMuPDF 的 flags 位对中文字体不置位，flags 不可靠
    """
    texts, max_size, bold = [], 0.0, False
    for span in line.get("spans", []):
        text = span.get("text") or ""
        if text.strip():
            texts.append(text)
            max_size = max(max_size, float(span.get("size") or 0))
            if "bold" in (span.get("font") or "").lower():
                bold = True
    return "".join(texts).strip(), round(max_size), bold


def _collect_size_stats(doc) -> Tuple[int, Dict[int, int]]:
    """
    统计字号 → 字符总量
    :return: (正文字号=字符量众数, {字号: 字符数})
    """
    counter: collections.Counter = collections.Counter()
    for _, line in _iter_lines(doc):
        text, size, _ = _line_info(line)
        if text:
            counter[size] += len(text)
    if not counter:
        return 10, {}
    return counter.most_common(1)[0][0], dict(counter)


def _build_heading_map(body_size: int, size_chars: Dict[int, int]) -> Dict[int, int]:
    """
    比正文大的字号按从大到小映射为 ## / ### / ####
    :return: {字号: 标题层级}
    """
    bigger = sorted(
        [size for size, chars in size_chars.items()
         if size > body_size and chars >= MIN_HEADING_SIZE_CHARS],
        reverse=True,
    )
    return {size: level for level, size in enumerate(bigger, start=2)
            if level <= MAX_HEADING_LEVEL}


def _hf_key(text: str) -> str:
    """
    页眉页脚比对用的行指纹
    纯数字行统一折叠成一个键，这样每页不同的页码也能被识别为「跨页重复」
    （目录页那批 #### 1 / #### 2 就是这么漏进来的）
    """
    if text.isdigit():
        return "#PAGE_NUMBER#"
    return re.sub(r"\d+", "", text).strip()


def _find_header_footer_lines(doc, page_count: int) -> set:
    """
    找出跨页重复的短行（页眉、页脚、页码），这些不该进正文
    判定：行指纹出现在 ≥50% 的页面上
    """
    if page_count < 3:
        return set()

    freq: collections.Counter = collections.Counter()
    for page_no in range(page_count):
        seen = set()
        for pno, line in _iter_lines(doc):
            if pno != page_no:
                continue
            text, _, _ = _line_info(line)
            key = _hf_key(text)
            if key and len(key) <= 40:
                seen.add(key)
        freq.update(seen)

    threshold = max(2, int(page_count * HEADER_FOOTER_RATIO))
    return {key for key, cnt in freq.items() if cnt >= threshold}


def _heading_level(text: str, size: int, bold: bool, body_size: int,
                   heading_map: Dict[int, int], bold_level: int) -> Optional[int]:
    """
    这一行是不是标题？是则返回层级，否则 None
    先排除正文句子，再走字号通道，最后走「同号加粗短行」通道
    """
    if not text or len(text) < 2 or text.isdigit():
        return None
    if len(text) > MAX_HEADING_CHARS:
        return None
    if any(ch in text for ch in NOT_HEADING_CHARS):
        return None
    if size in heading_map:
        return heading_map[size]
    if bold and size >= body_size and len(text) <= MAX_BOLD_HEADING_CHARS:
        return bold_level
    return None


def _extract_images(doc, page_no: int, images_dir: Path) -> List[str]:
    """导出该页内嵌图片，返回 markdown 引用链接列表（供 node_md_img 上传 MinIO）"""
    links: List[str] = []
    for idx, info in enumerate(doc[page_no].get_images(full=True), start=1):
        try:
            extracted = doc.extract_image(info[0])
        except Exception:
            continue
        ext = (extracted.get("ext") or "png").lower()
        name = f"page{page_no + 1}_{idx}.{ext}"
        try:
            (images_dir / name).write_bytes(extracted.get("image") or b"")
        except OSError:
            continue
        links.append(f"![{name[:-4]}](images/{name})")
    return links


def _merge_numbered_headings(lines: List[str]) -> Tuple[List[str], int]:
    """
    把「纯编号标题 + 下一段首词」合并成完整章节标题

    Brother 这类手册的排版是：1.1（大字加粗）单独占行，章节名「提示定义」在下一行
    正文里，结果标题只拿到光秃秃的 1.1，检索时参不到章节名。这里把下一段的首词
    提上来拼成「1.1 提示定义」，并从正文里去掉这个词。
    :return: (修正后的行列表, 合并次数)
    """
    merged = 0
    out: List[str] = []
    idx = 0
    while idx < len(lines):
        match = _SECTION_NUM.match(lines[idx])
        if not match:
            out.append(lines[idx])
            idx += 1
            continue

        cursor = idx + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1

        if cursor < len(lines) and not lines[cursor].startswith("#"):
            head, _, tail = lines[cursor].partition(" ")
            # 章节名有时会独占一段（如「2.1」+「设备」），此时 tail 为空也要合并
            if 1 < len(head) <= MAX_SECTION_NAME_CHARS:
                out.append(f"{match.group(1)} {match.group(2)} {head}")
                lines[cursor] = tail.strip()
                merged += 1
                idx += 1
                continue

        out.append(lines[idx])
        idx += 1
    return out, merged


def convert(pdf_path: Path, out_dir: Path) -> Dict[str, object]:
    """
    单个 PDF 转 md，落到 out_dir/<文件名>.md（图片在 out_dir/images/）
    :return: 统计信息，用于判断这次转换质量够不够用
    """
    doc = fitz.open(pdf_path)
    page_count = doc.page_count

    body_size, size_chars = _collect_size_stats(doc)
    heading_map = _build_heading_map(body_size, size_chars)
    # 加粗短行排在所有字号层级之下，且不超过最深一级
    bold_level = min(max(heading_map.values(), default=1) + 1, MAX_HEADING_LEVEL)
    junk_lines = _find_header_footer_lines(doc, page_count)

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    lines_out: List[str] = [f"# {pdf_path.stem}", ""]
    paragraph: List[str] = []
    stats = collections.Counter()

    def flush_paragraph():
        if paragraph:
            text = " ".join(paragraph).strip()
            paragraph.clear()
            if text:
                lines_out.append(text)
                lines_out.append("")

    for page_no, line in _iter_lines(doc):
        text, size, bold = _line_info(line)
        if not text:
            continue
        if _hf_key(text) in junk_lines:
            stats["dropped_header_footer"] += 1
            continue
        if TOC_LEADER.search(text):
            stats["dropped_toc_lines"] += 1
            continue

        level = _heading_level(text, size, bold, body_size, heading_map, bold_level)
        if level:
            flush_paragraph()
            lines_out.append(f"{'#' * level} {text}")
            lines_out.append("")
            stats["headings"] += 1
            stats[f"h{level}"] += 1
        else:
            paragraph.append(text)
        stats["chars_written"] += len(text)

    # 图片按页插入（放在正文之后，位置精度够用即可）
    for page_no in range(page_count):
        refs = _extract_images(doc, page_no, images_dir)
        if refs:
            flush_paragraph()
            lines_out.extend(sum(([r, ""] for r in refs), []))
            stats["images"] += len(refs)
        if len((doc[page_no].get_text() or "").strip()) < SPARSE_PAGE_CHARS:
            stats["sparse_pages"] += 1

    flush_paragraph()

    lines_out, merged = _merge_numbered_headings(lines_out)
    stats["merged_number_headings"] = merged

    md = re.sub(r"\n{3,}", "\n\n", "\n".join(lines_out)).strip() + "\n"
    md_path = out_dir / f"{pdf_path.stem}.md"
    md_path.write_text(md, encoding="utf-8")
    doc.close()

    result = {
        "pdf": str(pdf_path),
        "md": str(md_path),
        "pages": page_count,
        "body_font_size": body_size,
        "heading_map": {str(k): v for k, v in heading_map.items()},
        "bold_heading_level": bold_level,
        "chars_written": stats["chars_written"],
        "headings": stats["headings"],
        "heading_by_level": {f"h{lvl}": stats[f"h{lvl}"] for lvl in range(2, MAX_HEADING_LEVEL + 1)},
        "images": stats["images"],
        "sparse_pages": stats["sparse_pages"],
        "dropped_header_footer": stats["dropped_header_footer"],
        "dropped_toc_lines": stats["dropped_toc_lines"],
        "merged_number_headings": stats["merged_number_headings"],
    }
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # Windows 控制台 GBK 兜底
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="本地 PDF → Markdown（不依赖 MinerU）")
    parser.add_argument("--pdf", required=True, help="PDF 路径")
    parser.add_argument("--out-dir", default="", help="输出目录，默认 temp-files/local_md/<文件名>")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"文件不存在：{pdf_path}")
        raise SystemExit(2)

    out_dir = Path(args.out_dir) if args.out_dir else Path("temp-files/local_md") / pdf_path.stem
    stats = convert(pdf_path, out_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # 质量红线：平均每页文本太少说明是扫描件，本通道不适用，必须回头用 MinerU
    if stats["chars_written"] < max(1, stats["pages"]) * 40:
        print("\n警告：平均每页文本量过低，这份 PDF 可能是扫描件/图片型，本地抽取不可用。")
    print(f"\n下一步入库：uv run python -m scripts.mineru_probe --import-md \"{stats['md']}\" "
          f"--file-title \"{pdf_path.stem}\"")


if __name__ == "__main__":
    main()
