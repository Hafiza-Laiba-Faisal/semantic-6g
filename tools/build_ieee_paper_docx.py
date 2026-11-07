"""Build the IEEE conference-paper DOCX from fyp_md_files/IEEE_PAPER_DRAFT.md.

Two-column IEEE conference layout: single-column title/abstract block, then a
continuous section break into a two-column body. The Markdown subset is the
same one used by the paper source: level-2 headings are IEEE section headings,
level-3 headings are lettered subsections, pipe tables, bullet lists, images
with a caption in the alt text, and a numbered reference list.

Usage:
    python tools/build_ieee_paper_docx.py \
        --md fyp_md_files/IEEE_PAPER_DRAFT.md \
        --out IEEE_PAPER_DRAFT.docx
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FONT = "Times New Roman"
BODY_PT = 10.0
SMALL_PT = 8.0
ABSTRACT_PT = 9.0
COLUMN_WIDTH = 3.35  # inches; US Letter, 0.625" side margins, 0.17" gutter


def set_columns(section, count: int) -> None:
    sect_pr = section._sectPr
    cols = sect_pr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect_pr.append(cols)
    cols.set(qn("w:num"), str(count))
    cols.set(qn("w:space"), str(int(0.17 * 1440)))
    cols.set(qn("w:sep"), "0")
    cols.set(qn("w:equalWidth"), "1")


def set_run_font(run, size: float = BODY_PT, *, bold: bool = False,
                 italic: bool = False) -> None:
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), FONT)


TOKEN_RE = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*)")


def add_inline(paragraph, text: str, size: float = BODY_PT,
               *, bold: bool = False, italic: bool = False) -> None:
    for token in TOKEN_RE.split(text):
        if not token:
            continue
        run_bold, run_italic = bold, italic
        if token.startswith("**") and token.endswith("**"):
            token = token[2:-2]
            run_bold = True
        elif token.startswith("*") and token.endswith("*"):
            token = token[1:-1]
            run_italic = True
        run = paragraph.add_run(token)
        set_run_font(run, size, bold=run_bold, italic=run_italic)


def add_body_paragraph(doc, text: str, *, indent: bool = True,
                       size: float = BODY_PT) -> None:
    p = doc.add_paragraph()
    fmt = p.paragraph_format
    fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(0)
    fmt.line_spacing = 1.0
    if indent:
        fmt.first_line_indent = Inches(0.2)
    add_inline(p, text, size)


def add_bullet(doc, text: str) -> None:
    p = doc.add_paragraph()
    fmt = p.paragraph_format
    fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt.left_indent = Inches(0.22)
    fmt.first_line_indent = Inches(-0.12)
    fmt.space_before = Pt(2)
    fmt.space_after = Pt(2)
    fmt.line_spacing = 1.0
    add_inline(p, "\u2022  " + text, BODY_PT)


def add_section_heading(doc, text: str) -> None:
    p = doc.add_paragraph()
    fmt = p.paragraph_format
    fmt.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt.space_before = Pt(11)
    fmt.space_after = Pt(5)
    fmt.keep_with_next = True
    run = p.add_run(text.upper())
    set_run_font(run, BODY_PT, bold=False)
    run.font.small_caps = True


def add_subsection_heading(doc, text: str) -> None:
    p = doc.add_paragraph()
    fmt = p.paragraph_format
    fmt.alignment = WD_ALIGN_PARAGRAPH.LEFT
    fmt.space_before = Pt(7)
    fmt.space_after = Pt(3)
    fmt.keep_with_next = True
    add_inline(p, text, BODY_PT, italic=True)


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells)


def add_table(doc, rows: list[list[str]], caption: str) -> None:
    cap = doc.add_paragraph()
    cap.paragraph_format.space_before = Pt(8)
    cap.paragraph_format.space_after = Pt(3)
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.keep_with_next = True
    add_inline(cap, caption, SMALL_PT)

    ncols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = True
    for r_idx, row in enumerate(rows):
        for c_idx in range(ncols):
            cell = table.cell(r_idx, c_idx)
            text = row[c_idx] if c_idx < len(row) else ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0
                           else WD_ALIGN_PARAGRAPH.CENTER)
            add_inline(p, text, SMALL_PT, bold=(r_idx == 0))
            for run in p.runs:
                run.font.size = Pt(SMALL_PT)


def add_figure(doc, caption: str, src: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(src, width=Inches(COLUMN_WIDTH))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)
    add_inline(cap, caption, SMALL_PT)


def add_reference(doc, text: str) -> None:
    p = doc.add_paragraph()
    fmt = p.paragraph_format
    fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt.left_indent = Inches(0.18)
    fmt.first_line_indent = Inches(-0.18)
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(1)
    fmt.line_spacing = 1.0
    add_inline(p, text, SMALL_PT)


def parse_and_build(md_path: Path, out_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    doc = Document()

    # Base style
    style = doc.styles["Normal"]
    style.font.name = FONT
    style.font.size = Pt(BODY_PT)
    style.paragraph_format.space_after = Pt(0)
    style.paragraph_format.line_spacing = 1.0

    # Page geometry (US Letter, IEEE conference margins)
    first = doc.sections[0]
    first.page_width = Inches(8.5)
    first.page_height = Inches(11)
    first.top_margin = Inches(0.75)
    first.bottom_margin = Inches(1.0)
    first.left_margin = Inches(0.625)
    first.right_margin = Inches(0.625)
    set_columns(first, 1)

    # Front matter
    title = ""
    authors = affiliation = abstract = index_terms = ""
    body_start = 0
    for i, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        elif line.startswith("**AUTHORS:**"):
            authors = line.split("**AUTHORS:**", 1)[1].strip()
        elif line.startswith("**AFFILIATION:**"):
            affiliation = line.split("**AFFILIATION:**", 1)[1].strip()
        elif line.startswith("**ABSTRACT:**"):
            abstract = line.split("**ABSTRACT:**", 1)[1].strip()
        elif line.startswith("**INDEX TERMS:**"):
            index_terms = line.split("**INDEX TERMS:**", 1)[1].strip()
        elif line.startswith("## "):
            body_start = i
            break

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run(title)
    set_run_font(r, 22.0)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(authors)
    set_run_font(r, 11.0)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(affiliation)
    set_run_font(r, 9.0, italic=True)

    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run("Abstract\u2014")
    set_run_font(r, ABSTRACT_PT, bold=True, italic=True)
    add_inline(p, abstract, ABSTRACT_PT, bold=True)

    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run("Index Terms\u2014")
    set_run_font(r, ABSTRACT_PT, bold=True, italic=True)
    add_inline(p, index_terms, ABSTRACT_PT, bold=True)

    # Continuous section break into two columns
    body = doc.add_section(WD_SECTION.CONTINUOUS)
    body.top_margin = Inches(0.75)
    body.bottom_margin = Inches(1.0)
    body.left_margin = Inches(0.625)
    body.right_margin = Inches(0.625)
    set_columns(body, 2)

    # Body
    i = body_start
    in_references = False
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        if not line:
            i += 1
            continue

        if line.startswith("## "):
            heading = line[3:].strip()
            in_references = heading.upper() == "REFERENCES"
            add_section_heading(doc, heading)
            i += 1
            continue

        if line.startswith("### "):
            add_subsection_heading(doc, line[4:].strip())
            i += 1
            continue

        if line.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if m:
                add_figure(doc, m.group(1), m.group(2))
            i += 1
            continue

        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = []
            caption = ""
            for entry in block:
                cells = _split_row(entry)
                if _is_separator(cells):
                    continue
                rows.append(cells)
            # The caption is the bold line immediately before the table.
            j = i - len(block) - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j >= 0 and lines[j].strip().startswith("**Table"):
                caption = re.sub(r"^\*\*(.+?)\*\*", r"\1", lines[j].strip()).strip()
            if rows:
                add_table(doc, rows, caption)
            continue

        if line.startswith("- "):
            add_bullet(doc, line[2:].strip())
            i += 1
            continue

        if line.startswith("**Table"):
            # Caption consumed by add_table; skip standalone caption lines.
            i += 1
            continue

        if in_references and line.startswith("["):
            add_reference(doc, line)
            i += 1
            continue

        add_body_paragraph(doc, line)
        i += 1

    doc.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    parse_and_build(Path(args.md), Path(args.out))
    print(f"built {args.out}")


if __name__ == "__main__":
    main()
