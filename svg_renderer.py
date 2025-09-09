from __future__ import annotations

from dataclasses import dataclass, field

from typing import Optional, List, Tuple

# Import shapes for type hints only (no Qt)
from puzzle_engine import PuzzleResult
import re
# from PyQt6.QtGui import QFont, QFontMetrics
# from PyQt6.QtGui import QImage, QPainter, QTextDocument, QColor
# from PyQt6.QtCore import QBuffer, QIODevice
import base64




# -----------------------------------------------------------------------------
# Simple logger hook (optional; mirrors puzzle_engine)
# -----------------------------------------------------------------------------
_LOGGER = None  # type: Optional[callable]


def set_logger(fn) -> None:
    """Allow the UI to inject a logger callback: fn(text: str)."""
    global _LOGGER
    _LOGGER = fn


def _log(msg: str) -> None:
    if _LOGGER:
        try:
            _LOGGER(msg)
            return
        except Exception:
            pass
    print(msg)


@dataclass
class Appearance:
    """
    Visual settings used by the SVG renderer.
    Keep this in sync with your UI fields.
    """
    # Grid
    cell_bg_color: str = "#FFFFFF"
    cell_line_color: str = "#000000"
    cell_line_thickness: float = 1.0

    # Letters
    grid_font_family: str = "Arial"
    grid_font_size: int = 24
    grid_font_bold: bool = False
    grid_font_color: str = "#000000"

    # Legend
    list_font_family: str = "Arial"
    list_font_size: int = 14
    list_font_color: str = "#000000"
    list_align: str = "Left"  # "Left", "Center", "Right"
    list_bold: bool = False
    list_underline_words: bool = False
    sentence_width_factor: float = 0.58  # tuning knob for wrap width

    legend_columns: int = 2
    show_legend: bool = True
    # Solutions: include legend (True) or grid-only (False)
    solution_show_legend: bool = False
    # --- Solution marking options ---
    solution_mark_style: str = "highlight"     # "highlight" | "circle"
    solution_circle_stroke: str = "#D94242"
    solution_circle_width: float = 2.0
    solution_circle_band_frac: float = 0.55    # fraction of cell height
    solution_circle_pad_len: float = 2.0       # extra length at each end (px)
     # Unified color for BOTH highlight fill and circle stroke
    solution_mark_color: str = "#D94242"

    # Border
    add_border: bool = False
    border_thickness: float = 2.0
    border_color: str = "#000000"
    # Distance of border rectangle to the grid (px)
    border_distance: float = 2.0
        # --- Sentence rendering (Sentence mode) ---
    show_sentence: bool = False
    # For SOLUTION SVGs only: append the sentence block if True.
    solution_show_sentence: bool = False                # draw sentence block?
    sentence_text: str = ""                    # full sentence (with punctuation/case)
    sentence_targets: list[str] = field(default_factory=list)  # display-case target words
    sentence_font_family: str = "Arial"
    sentence_font_size: int = 16
    sentence_font_bold: bool = False           # overall sentence weight (targets still bold/underline)
    sentence_font_color: str = "#000000"
    sentence_align: str = "Left"               # "Left" | "Center" | "Right"
    sentence_line_spacing: float = 1.25        # line height multiplier

        # --- Secret highlighting (solution view) ---
    # Font color of secret letters (leftovers used for hidden text)
    secret_font_color: Optional[str] = None
    # Background fill color for secret cells
    secret_bg_color: Optional[str] = None
        # Secret placement in solution view: "start" (default) or "center"
    secret_placement: str = "start"
    # Mark endpoints of placed words in puzzle view
    puzzle_mark_endpoints: bool = False
    puzzle_endpoint_marker: str = "*"
    puzzle_endpoint_color: Optional[str] = None  # falls back to grid_font_color
    puzzle_endpoint_scale: float = 0.60          # fraction of grid_font_size

    




# -----------------------------------------------------------------------------
# Tiny helper to build safe SVG text (no external lib, very basic)
# -----------------------------------------------------------------------------
def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _text_anchor(align: str) -> str:
    if align.lower().startswith("c"):
        return "middle"
    if align.lower().startswith("r"):
        return "end"
    return "start"

def _html_escape(s: str) -> str:
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))

def _html_sentence_with_targets(sentence: str, targets: list[str]) -> str:
    """
    Build HTML for the sentence.
    - Bold+underline only the *first occurrence* of each target word (alnum-only match).
    - Punctuation is not styled.
    - Spaces remain normal so wrapping is natural.
    """
    if not sentence:
        return ""

    # tokens preserve whitespace as separate items, e.g. ["Love", " ", "is", " ", "patient,", ...]
    toks = re.findall(r'\s+|[^\s]+', sentence)

    def norm_token(s: str) -> str:
        return "".join(ch for ch in s.upper() if ch.isalnum())

    # targets to normalize and track; emphasize only first time we see each
    tgt_set = {norm_token(t) for t in (targets or []) if t}
    used = set()  # normalized tokens we've already emphasized

    parts = []
    for tok in toks:
        if tok.isspace():
            parts.append(tok)  # keep real spaces; allows natural wrapping
            continue

        n = norm_token(tok)
        is_first_hit = (n in tgt_set) and (n not in used) and bool(n)

        if not is_first_hit:
            # Not a first occurrence → plain text (escape only)
            parts.append(_html_escape(tok))
            continue

        # First time we see this target: emphasize alnum segments only
        used.add(n)
        segments = re.findall(r'[A-Za-z0-9]+|[^A-Za-z0-9]+', tok)
        seg_out = []
        for seg in segments:
            if re.fullmatch(r'[A-Za-z0-9]+', seg):
                seg_out.append(
                    f'<span style="font-weight:bold; text-decoration: underline">{_html_escape(seg)}</span>'
                )
            else:
                seg_out.append(_html_escape(seg))
        parts.append("".join(seg_out))

    return "".join(parts)




# -----------------------------------------------------------------------------
# Core renderers
# -----------------------------------------------------------------------------
def render_puzzle_svg(result: PuzzleResult, appearance: Appearance) -> str:
    """
    Draw a simple grid with letters and a legend under it.
    The goal is clarity, not fancy style (that can come later).
    """
    letters = result.letters
    rows = len(letters)
    cols = len(letters[0]) if rows else 0

    # Size math: cell becomes font_size * 1.6, with some padding
    cell = max(12, int(appearance.grid_font_size * 1.6))
    pad = int(cell * 0.4)
    grid_w = cols * cell
    grid_h = rows * cell

    # Legend layout
    legend = result.legend or []
    if not getattr(appearance, "show_legend", True):
        legend = []

    col_count = max(1, int(appearance.legend_columns))
    # Rough legend cell size
    legend_line_h = max(12, int(appearance.list_font_size * 1.4))
    legend_rows = (len(legend) + col_count - 1) // col_count
    legend_h = legend_rows * legend_line_h + (pad if legend else 0)

    total_w = grid_w + pad * 2
    total_h = grid_h + legend_h + pad * 2

    # Begin SVG
    out = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{total_w}" height="{total_h}" '
        f'viewBox="0 0 {total_w} {total_h}">'
    )



    # Optional border (around GRID, offset by distanceBorder)
    if appearance.add_border:
        d = float(getattr(appearance, "border_distance", 0.0) or 0.0)
        bx = pad - d
        by = pad - d
        bw = grid_w + 2 * d
        bh = grid_h + 2 * d
        out.append(
            f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" '
            f'stroke="{appearance.border_color}" stroke-width="{appearance.border_thickness}" fill="none" />'
        )

    # Grid background
    out.append(
        f'<rect x="{pad}" y="{pad}" width="{grid_w}" height="{grid_h}" '
        f'fill="{appearance.cell_bg_color}" stroke="none" />'
    )

    # Cell lines
    stroke = appearance.cell_line_color
    sw = appearance.cell_line_thickness
    # Vertical lines
    for c in range(cols + 1):
        x = pad + c * cell
        out.append(f'<line x1="{x}" y1="{pad}" x2="{x}" y2="{pad + grid_h}" stroke="{stroke}" stroke-width="{sw}" />')
    # Horizontal lines
    for r in range(rows + 1):
        y = pad + r * cell
        out.append(f'<line x1="{pad}" y1="{y}" x2="{pad + grid_w}" y2="{y}" stroke="{stroke}" stroke-width="{sw}" />')

    # Letters
    font_weight = "bold" if appearance.grid_font_bold else "normal"
    out.append(
        f'<g font-family="{_esc(appearance.grid_font_family)}" font-size="{appearance.grid_font_size}" '
        f'font-weight="{font_weight}" fill="{appearance.grid_font_color}">'
    )
    # Center letters in cells
    txt_dy = int(appearance.grid_font_size * 0.35)
    for r in range(rows):
        for c in range(cols):
            ch = letters[r][c]
            x = pad + c * cell + cell // 2
            y = pad + r * cell + cell // 2 + txt_dy
            out.append(f'<text x="{x}" y="{y}" text-anchor="middle">{_esc(ch)}</text>')
    out.append('</g>')

    # Legend
    if legend:
        lx = pad
        ly = pad + grid_h + pad
        col_w = grid_w // col_count if col_count else grid_w
        anchor = _text_anchor(appearance.list_align)
        font_weight = "bold" if appearance.list_bold else "normal"
        text_decoration = "underline" if appearance.list_underline_words else "none"

        out.append(
            f'<g font-family="{_esc(appearance.list_font_family)}" font-size="{appearance.list_font_size}" '
            f'font-weight="{font_weight}" fill="{appearance.list_font_color}" '
            f'text-decoration="{text_decoration}">'
        )

        # Column-major layout
        per_col = (len(legend) + col_count - 1) // col_count
        for i, word in enumerate(legend):
            col_idx = i // per_col
            row_idx = i % per_col
            tx = lx + col_idx * col_w
            if anchor == "middle":
                tx += col_w // 2
            elif anchor == "end":
                tx += col_w - 4
            else:
                tx += 4
            ty = ly + (row_idx + 1) * legend_line_h
            out.append(f'<text x="{tx}" y="{ty}" text-anchor="{anchor}">{_esc(word)}</text>')
        out.append('</g>')

    # Endpoint markers (*) for SECRET only: first and last leftover cell mapped to hidden_text (optional)
    if getattr(appearance, "puzzle_mark_endpoints", False) and getattr(result, "hidden_text", None):
        used = getattr(result, "used_mask", None)
        if used is not None:
            # leftover cells in row-major order
            spots: list[tuple[int, int]] = []
            for rr in range(rows):
                for cc in range(cols):
                    if not used[rr][cc]:
                        spots.append((rr, cc))
            # letters-only length
            text_len = sum(1 for ch in result.hidden_text if ch.isalnum())
            count = min(text_len, len(spots))
            if count > 0:
                placement = (getattr(appearance, "secret_placement", "start") or "start").lower()
                start = 0
                if placement.startswith("c"):
                    start = max(0, (len(spots) - count) // 2)
                first_rc = spots[start]
                last_rc  = spots[start + count - 1]
                endpoints = (first_rc, last_rc)
                star = getattr(appearance, "puzzle_endpoint_marker", "*") or "*"
                star_fs = max(8, int(appearance.grid_font_size * float(getattr(appearance, "puzzle_endpoint_scale", 0.60) or 0.60)))
                star_color = getattr(appearance, "puzzle_endpoint_color", None) or appearance.grid_font_color
                out.append(
                    f'<g font-family="{_esc(appearance.grid_font_family)}" font-size="{star_fs}" '
                    f'font-weight="bold" fill="{star_color}">'
                )
                # place small star near top-left corner of the cell
                offset_x = max(2, int(0.08 * cell))
                for (rr, cc) in endpoints:
                    sx = pad + cc * cell + offset_x
                    # Put the TEXT BASELINE ~0.9*star_fs below the top edge to keep glyph fully inside the box,
                    # independent of renderer support for dominant-baseline.
                    sy = pad + rr * cell + int(0.90 * star_fs)
                    out.append(f'<text x="{sx}" y="{sy}" text-anchor="start">{_esc(star)}</text>')
                out.append('</g>')
    # # --- add hidden_text breadcrumb for debugging/regression ---
    # if getattr(result, "hidden_text", None):
    #     out.append(f'<!-- hidden_text:{_esc(result.hidden_text)} -->')

    out.append('</svg>')
    return "\n".join(out)



def render_solution_svg(result: PuzzleResult, appearance: Appearance) -> str:
    """
    Solution SVG:
      - Draw grid + letters like the puzzle.
      - Mark answers with either:
          * "highlight": per-cell soft yellow rects behind letters
          * "circle": rotated pill per placed word with semicircular endcaps,
            extended to fully include the first & last letters (including diagonals).
    """
    letters = result.letters
    used = result.used_mask
    rows = len(letters)
    cols = len(letters[0]) if rows else 0

    # --- Sizing (mirror puzzle) ---
    cell = max(12, int(appearance.grid_font_size * 1.6))
    pad = int(cell * 0.4)
    grid_w = cols * cell
    grid_h = rows * cell

    # Legend (if any)
    legend = result.legend or []
    col_count = max(1, int(getattr(appearance, "legend_columns", 1) or 1))
    legend_line_h = max(12, int(appearance.list_font_size * 1.4))
    legend_rows = (len(legend) + col_count - 1) // col_count
    legend_h = legend_rows * legend_line_h + (pad if legend else 0)

    total_w = grid_w + pad * 2
    total_h = grid_h + legend_h + pad * 2

    out: list[str] = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">')

    # Optional border (around GRID, offset by distanceBorder)
    if getattr(appearance, "add_border", False):
        d = float(getattr(appearance, "border_distance", 0.0) or 0.0)
        bx = pad - d
        by = pad - d
        bw = grid_w + 2 * d
        bh = grid_h + 2 * d
        out.append(
            f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" '
            f'fill="none" stroke="{appearance.border_color}" stroke-width="{appearance.border_thickness}" />'
        )

    # Grid background
    out.append(
        f'<rect x="{pad}" y="{pad}" width="{grid_w}" height="{grid_h}" '
        f'fill="{appearance.cell_bg_color}" stroke="none" />'
    )

    # --- Highlights behind letters (if style == "highlight") ---
    mark_style = (getattr(appearance, "solution_mark_style", "highlight") or "highlight").lower()
    if mark_style == "highlight":
        hi_fill = getattr(appearance, "solution_mark_color", "#fff2a8") or "#fff2a8"
        hi_opacity = 0.8
        for r in range(rows):
            for c in range(cols):
                if used[r][c]:
                    x = pad + c * cell + 1
                    y = pad + r * cell + 1
                    out.append(
                        f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" '
                        f'fill="{hi_fill}" fill-opacity="{hi_opacity}" stroke="none" />'
                    )

    # --- Grid lines ---
    stroke_grid = appearance.cell_line_color
    sw_grid = appearance.cell_line_thickness
    for c in range(cols + 1):
        x = pad + c * cell
        out.append(f'<line x1="{x}" y1="{pad}" x2="{x}" y2="{pad + grid_h}" stroke="{stroke_grid}" stroke-width="{sw_grid}" />')
    for r in range(rows + 1):
        y = pad + r * cell
        out.append(f'<line x1="{pad}" y1="{y}" x2="{pad + grid_w}" y2="{y}" stroke="{stroke_grid}" stroke-width="{sw_grid}" />')

    # --- Pill bands below letters (if style == "circle") ---
    if mark_style == "circle":
        import math
         # Use unified mark color for circle stroke
        stroke = getattr(appearance, "solution_mark_color", None) or getattr(appearance, "solution_circle_stroke", "#D94242") or "#D94242"
       
        sw = float(getattr(appearance, "solution_circle_width", 2.0) or 2.0)
        band_frac = float(getattr(appearance, "solution_circle_band_frac", 0.55) or 0.55)  # thickness vs cell
        pad_len = float(getattr(appearance, "solution_circle_pad_len", 2.0) or 2.0)        # extra along axis
        rect_h = max(1.0, band_frac * cell)
        rx = ry = rect_h * 0.5  # true half-circle endcaps

        for pw in getattr(result, "placed_words", []) or []:
            cells = getattr(pw, "cells", None)
            if not cells:
                continue

            # Centers of first/last letters
            (r0, c0) = cells[0]
            (r1, c1) = cells[-1]
            x0 = pad + c0 * cell + 0.5 * cell
            y0 = pad + r0 * cell + 0.5 * cell
            x1 = pad + c1 * cell + 0.5 * cell
            y1 = pad + r1 * cell + 0.5 * cell

            dx = x1 - x0
            dy = y1 - y0
            D = math.hypot(dx, dy)

            # Axis unit vector; if single-letter word, pick horizontal
            if D > 1e-6:
                ux, uy = dx / D, dy / D
            else:
                ux, uy = 1.0, 0.0

            # End extension to cover the furthest edge of a cell along the axis:
            # ext_each = 0.5*cell*(|ux|+|uy|) + pad_len
            # (0.5*cell for axis-aligned; ~0.707*cell for 45°)
            ext_each = 0.5 * cell * (abs(ux) + abs(uy)) + pad_len

            rect_w = D + 2.0 * ext_each
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5
            tl_x = cx - rect_w * 0.5
            tl_y = cy - rect_h * 0.5
            ang = math.degrees(math.atan2(dy, dx)) if D > 1e-6 else 0.0

            out.append(
                f'<rect x="{tl_x:.2f}" y="{tl_y:.2f}" width="{rect_w:.2f}" height="{rect_h:.2f}" '
                f'fill="none" stroke="{stroke}" stroke-width="{sw:.2f}" '
                f'rx="{rx:.2f}" ry="{ry:.2f}" transform="rotate({ang:.2f} {cx:.2f} {cy:.2f})" />'
            )
    # --- Secret cells (leftovers with hidden_text), background fills ---
    secret_bg = getattr(appearance, "secret_bg_color", None)
    secret_font = getattr(appearance, "secret_font_color", None)
    is_secret = [[False] * cols for _ in range(rows)]
    if result.hidden_text:
        # Collect leftover cells (== not used), row-major
        spots: list[tuple[int, int]] = []
        for rr in range(rows):
            for cc in range(cols):
                if not used[rr][cc]:
                    spots.append((rr, cc))
        # center by **letters only** (ignore spaces/punct)
        text_len = sum(1 for ch in result.hidden_text if ch.isalnum())
        count = min(text_len, len(spots))
        if count > 0:
            placement = (getattr(appearance, "secret_placement", "start") or "start").lower()
            start = 0
            if placement.startswith("c"):
                # center block of 'count' cells within 'spots'
                start = max(0, (len(spots) - count) // 2)
            for i in range(count):
                r, c = spots[start + i]
                is_secret[r][c] = True
                if secret_bg:
                    x = pad + c * cell + 1
                    y = pad + r * cell + 1
                    out.append(
                        f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" '
                        f'fill="{secret_bg}" stroke="none" />'
                    )


    # --- Letters on top ---
    font_weight = "bold" if appearance.grid_font_bold else "normal"
    out.append(
        f'<g font-family="{_esc(appearance.grid_font_family)}" font-size="{appearance.grid_font_size}" '
        f'font-weight="{font_weight}" fill="{appearance.grid_font_color}">'
    )
    txt_dy = int(appearance.grid_font_size * 0.35)
    for r in range(rows):
        for c in range(cols):
            ch = letters[r][c]
            x = pad + c * cell + cell // 2
            y = pad + r * cell + cell // 2 + txt_dy
            if is_secret[r][c] and secret_font:
                out.append(f'<text x="{x}" y="{y}" text-anchor="middle" fill="{_esc(secret_font)}">{_esc(ch)}</text>')
            else:
                out.append(f'<text x="{x}" y="{y}" text-anchor="middle">{_esc(ch)}</text>')
    out.append('</g>')

    # --- Legend (if any) ---
    if legend:
        lx = pad
        ly = pad + grid_h + pad
        col_w = grid_w // col_count if col_count else grid_w
        anchor = _text_anchor(appearance.list_align)
        fw = "bold" if appearance.list_bold else "normal"
        deco = "underline" if appearance.list_underline_words else "none"

        out.append(
            f'<g font-family="{_esc(appearance.list_font_family)}" font-size="{appearance.list_font_size}" '
            f'font-weight="{fw}" fill="{appearance.list_font_color}" text-decoration="{deco}">'
        )

        per_col = (len(legend) + col_count - 1) // col_count
        for i, word in enumerate(legend):
            col_idx = i // per_col
            row_idx = i % per_col
            tx = lx + col_idx * col_w
            if anchor == "middle":
                tx += col_w // 2
            elif anchor == "end":
                tx += col_w - 4
            else:
                tx += 4
            ty = ly + (row_idx + 1) * legend_line_h
            out.append(f'<text x="{tx}" y="{ty}" text-anchor="{anchor}">{_esc(word)}</text>')
        out.append('</g>')

    # if getattr(result, "hidden_text", None):
    #     out.append(f'<!-- hidden_text:{_esc(result.hidden_text)} -->')

    out.append('</svg>')
    return "\n".join(out)

# ---------------- Sentence block (append to bottom of SVG) ----------------

def _norm_token_for_match(s: str) -> str:
    """Uppercase and keep only letters/digits for matching targets."""
    if not s:
        return ""
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


# def _build_sentence_block_svg(app: Appearance, avail_width_px: float, left_margin_px: float = 20.0):
#     """
#     Build an SVG <g> for the sentence using precise widths from QFontMetrics.
#     We draw each token as its own <text> at an exact x, and draw a real <line>
#     for underlines of target tokens. This fixes spacing and underline length.
#     Returns (svg_snippet:str, block_height_px:float).
#     """
#     text = app.sentence_text or ""
#     if not text.strip():
#         return "", 0.0

#     # Local esc so we don't depend on any other helper
#     def esc(s: str) -> str:
#         return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

#     # Build base font and metrics
#     fs = float(app.sentence_font_size or 16)
#     family = app.sentence_font_family or "Arial"

#     base_font = QFont(family)
#     base_font.setPointSizeF(fs)
#     base_font.setBold(False)
#     fm_base = QFontMetrics(base_font)

#     bold_font = QFont(base_font)
#     bold_font.setBold(True)
#     fm_bold = QFontMetrics(bold_font)

#     # Layout box
#     inner_w = max(10.0, (avail_width_px - 2 * left_margin_px))

#     # Split into tokens that preserve spaces
#     # e.g., ["Do", " ", "not", " ", "focus", ...]
#     tokens = re.findall(r'\s+|[^\s]+', text)

#     # normalize target set for matching
#     tgt_norm = {_norm_token_for_match(t) for t in (app.sentence_targets or []) if t}

#     # Helpers to measure tokens with correct weight
#     def tok_is_target(tok: str) -> bool:
#         return bool(_norm_token_for_match(tok) and _norm_token_for_match(tok) in tgt_norm)

#     def tok_width(tok: str, is_tgt: bool) -> float:
#         if not tok:
#             return 0.0
#         return (fm_bold if is_tgt else fm_base).horizontalAdvance(tok)

#     # Wrap into lines using precise widths
#     lines = []   # list[list[(tok, is_tgt, width)]]
#     cur_line = []
#     cur_w = 0.0

#     for tok in tokens:
#         is_tgt = tok_is_target(tok) and not tok.isspace()
#         w_tok = tok_width(tok, is_tgt)
#         if cur_line and (cur_w + w_tok) > inner_w:
#             lines.append(cur_line)
#             cur_line = []
#             cur_w = 0.0
#         cur_line.append((tok, is_tgt, w_tok))
#         cur_w += w_tok
#     if cur_line:
#         lines.append(cur_line)

#     # Vertical metrics
#     line_h = fm_base.height() * float(app.sentence_line_spacing or 1.25)
#     y0 = fm_base.ascent()  # baseline for first line
#     underline_offset = max(1.0, fs * 0.10)
#     underline_thickness = max(1.0, fs * 0.08)

#     # Alignment per line using measured line width
#     align = (app.sentence_align or "Left").lower()
#     def base_x_for(line_items) -> float:
#         lw = sum(w for (_tok, _tgt, w) in line_items)
#         if align.startswith("c"):
#             return left_margin_px + (inner_w - lw) / 2.0
#         elif align.startswith("r"):
#             return left_margin_px + (inner_w - lw)
#         return left_margin_px  # left

#     # Build SVG
#     out = []
#     out.append('<g class="sentence-block">')

#     # Common style pieces
#     fill = app.sentence_font_color or "#000000"

#     for i, items in enumerate(lines):
#         base_x = base_x_for(items)
#         base_y = y0 + i * line_h
#         x = base_x

#         for tok, is_tgt, w_tok in items:
#             if not tok:
#                 continue
#             # Draw token
#             fw = "bold" if is_tgt else "normal"
#             # xml:space="preserve" ensures spaces are rendered when present
#             out.append(
#                 f'<text xml:space="preserve" x="{x:.2f}" y="{base_y:.2f}" '
#                 f'text-anchor="start" font-family="{esc(family)}" font-size="{fs:.2f}px" '
#                 f'font-weight="{fw}" fill="{fill}" dominant-baseline="alphabetic">{esc(tok)}</text>'
#             )
#             # Underline only non-space target tokens
#             if is_tgt:
#                 uy = base_y + underline_offset
#                 out.append(
#                     f'<line x1="{x:.2f}" y1="{uy:.2f}" x2="{(x + w_tok):.2f}" y2="{uy:.2f}" '
#                     f'stroke="{fill}" stroke-width="{underline_thickness:.2f}" />'
#                 )
#             x += w_tok

#     out.append('</g>')
#     block_h = y0 + (len(lines) - 1) * line_h + fm_base.descent() + fs * 0.2
#     return "\n".join(out), block_h


def inject_sentence_block(svg_text: str, app: Appearance) -> str:
    """
    Append the sentence as plain SVG <text>/<tspan> (no foreignObject, no Qt).
    Lines are wrapped approximately by character count; targets rendered bold+underline.
    Safe for CairoSVG.
    """
    if not getattr(app, "show_sentence", False):
        return svg_text
    sentence = getattr(app, "sentence_text", "")
    if not sentence or not sentence.strip():
        return svg_text

    # --- read current canvas size
    m = re.search(r'viewBox="0\s+0\s+([\d.]+)\s+([\d.]+)"', svg_text)
    if m:
        W = float(m.group(1)); H = float(m.group(2))
        has_viewbox = True
    else:
        mw = re.search(r'width="([\d.]+)"', svg_text)
        mh = re.search(r'height="([\d.]+)"', svg_text)
        if not (mw and mh):
            return svg_text
        W = float(mw.group(1)); H = float(mh.group(1))
        has_viewbox = False

    # --- tokenize (preserve spaces) and mark first-hit targets
    toks = re.findall(r'\s+|[^\s]+', sentence)

    def norm(s: str) -> str:
        return "".join(ch for ch in s.upper() if ch.isalnum())

    targets_norm = []
    seen = set()
    # preserve order of targets; underline bold only first occurrence
    for t in (app.sentence_targets or []):
        n = norm(t)
        if n and n not in seen:
            seen.add(n); targets_norm.append(n)
    first_hits = set()  # which targets we've styled already

    # --- crude width model for wrapping: ~0.6 * font_size per char
    fs = float(getattr(app, "sentence_font_size", 16) or 16)

    # Make sentence width match the grid width
    cell = max(12, int(getattr(app, "grid_font_size", 24) * 1.6))   # same as render_puzzle_svg
    pad = int(cell * 0.4)                                            # same as render_puzzle_svg

    left_margin = float(pad)
    right_margin = float(pad)
    inner_w = max(40.0, W - (left_margin + right_margin))            # == grid_w
    factor = float(getattr(app, "sentence_width_factor", 0.58) or 0.58)
    approx_char_w = max(5.0, factor * fs)


    max_chars = max(10, int(inner_w / approx_char_w))
    line_h = fs * float(getattr(app, "sentence_line_spacing", 1.25) or 1.25)

    # --- build lines of (token, is_target_first_hit)
    lines: list[list[tuple[str, bool]]] = []
    cur: list[tuple[str, bool]] = []
    cur_len = 0

    def add_line():
        nonlocal cur, cur_len, lines
        if cur:
            # strip trailing spaces at end of line
            while cur and cur[-1][0].isspace():
                cur.pop()
            if cur:
                lines.append(cur)
        cur = []
        cur_len = 0

    for tok in toks:
        is_space = tok.isspace()
        n = norm(tok)
        is_target = (not is_space) and n and (n in targets_norm) and (n not in first_hits)
        if is_target:
            first_hits.add(n)

        add_len = len(tok)

        # would overflow current line?
        if cur and (cur_len + add_len) > max_chars:
            add_line()
            # never start a new line with a space
            if is_space:
                continue

        # never start any line with a space
        if (not cur) and is_space:
            continue

        cur.append((tok, is_target))
        cur_len += add_len

    add_line()

    # --- compute x start by alignment
    align = (getattr(app, "sentence_align", "Left") or "Left").lower()
    def x_for(line_text: str) -> float:
        est_w = len(line_text) * approx_char_w
        if align.startswith("c"):  # center within grid band
            return left_margin + max(0.0, (inner_w - est_w) / 2.0)
        if align.startswith("r"):  # right-align to grid’s right edge
            return left_margin + max(0.0, inner_w - est_w)
        return left_margin  # left


    # --- build SVG block
    fill = getattr(app, "sentence_font_color", "#000000") or "#000000"
    weight_all = "bold" if getattr(app, "sentence_font_bold", False) else "normal"

    # increase canvas height
    total_h = H + (len(lines) * line_h) + 10.0
    if has_viewbox:
        svg_text = re.sub(r'(<svg\b[^>]*\bviewBox="0\s+0\s+[\d.]+\s+)[\d.]+(")',
                        rf'\g<1>{total_h:.2f}\g<2>', svg_text, count=1)
    svg_text = re.sub(r'(<svg\b[^>]*\bheight=")[^"]+(")',
                    rf'\g<1>{total_h:.2f}\g<2>', svg_text, count=1)



    # compose lines
    y0 = H + fs + 5.0
    block = []
    block.append(f'<g class="sentence" font-family="{_esc(getattr(app, "sentence_font_family", "Arial") or "Arial")}" '
                 f'font-size="{fs:.2f}" fill="{_esc(fill)}" font-weight="{weight_all}">')
    for i, items in enumerate(lines):
        line_text = "".join(t for (t, _b) in items)
        x = x_for(line_text)
        y = y0 + i * line_h
        parts = [f'<text x="{x:.2f}" y="{y:.2f}" xml:space="preserve">']
        for (tok, is_tgt) in items:
            if not tok:
                continue
            if is_tgt and not tok.isspace():
                parts.append(f'<tspan style="font-weight:bold; text-decoration: underline">{_esc(tok)}</tspan>')
            else:
                parts.append(_esc(tok))
        parts.append('</text>')
        block.append("".join(parts))
    block.append("</g>")

    return svg_text.replace("</svg>", "\n".join(block) + "\n</svg>")





# def inject_sentence_block(svg_text: str, app: Appearance) -> str:
#     """
#     Append the sentence as a PNG rendered by Qt's text engine.
#     This guarantees correct spacing/bold/underline in the preview.
#     """
#     if not getattr(app, "show_sentence", False):
#         return svg_text
#     sentence = getattr(app, "sentence_text", "")
#     if not sentence or not sentence.strip():
#         return svg_text

#     # Extract canvas size from viewBox (fallback to width/height attrs)
#     m = re.search(r'viewBox="0\s+0\s+([\d.]+)\s+([\d.]+)"', svg_text)
#     if m:
#         W = float(m.group(1)); H = float(m.group(2))
#     else:
#         mw = re.search(r'width="([\d.]+)"', svg_text)
#         mh = re.search(r'height="([\d.]+)"', svg_text)
#         if not (mw and mh):
#             return svg_text
#         W = float(mw.group(1)); H = float(mh.group(1))

#     # Build HTML with highlights
#     html = _html_sentence_with_targets(sentence, getattr(app, "sentence_targets", []))
#     if not html:
#         return svg_text

#     # Layout using QTextDocument at the SAME width as the SVG (with small margins)
#     left_margin = 20
#     right_margin = 20
#     text_width = max(10.0, W - (left_margin + right_margin))

#     doc = QTextDocument()
#     # Default font
#     from PyQt6.QtGui import QFont
#     f = QFont(app.sentence_font_family or "Arial")
#     f.setPointSizeF(float(app.sentence_font_size or 16))
#     doc.setDefaultFont(f)

#     # Align: simulate by wrapping HTML in a div
#     align = (app.sentence_align or "Left").lower()
#     if align.startswith("c"):
#         html = f'<div style="text-align:center">{html}</div>'
#     elif align.startswith("r"):
#         html = f'<div style="text-align:right">{html}</div>'

#     doc.setHtml(html)
#     doc.setTextWidth(text_width)
#     doc_size = doc.size().toSize()
#     out_w = int(text_width)
#     out_h = int(doc_size.height()) + 2  # tiny padding

#     # Render at 2x scale for crispness, then downscale in SVG
#     scale = 2
#     img = QImage(out_w * scale, out_h * scale, QImage.Format.Format_ARGB32_Premultiplied)
#     img.fill(QColor(255, 255, 255, 0))  # transparent background
#     p = QPainter(img)
#     p.scale(scale, scale)
#     p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
#     # color is controlled by HTML; if you want to force color, add a wrapper style here
#     doc.drawContents(p)
#     p.end()

#     # PNG -> base64
#     buf = QBuffer()
#     buf.open(QIODevice.OpenModeFlag.WriteOnly)
#     img.save(buf, "PNG")
#     b64 = base64.b64encode(bytes(buf.data())).decode("ascii")
#     data_url = f"data:image/png;base64,{b64}"

#     # Increase SVG height and append the image block
#     new_H = H + out_h + 10  # small gap
#     out = re.sub(r'(viewBox="0\s+0\s+[\d.]+\s+)[\d.]+(")', rf'\g<1>{new_H:.2f}\2', svg_text, count=1)
#     out = re.sub(r'(height=")[\d.]+(")', rf'\g<1>{new_H:.2f}\2', out, count=1)

#     # Position the image at the bottom; center horizontally inside [left_margin, W-right_margin]
#     x_img = left_margin
#     y_img = H + 5
#     out += ""  # no-op to keep string type

#     # Insert <image> before </svg>
#     image_tag = (f'\n<image x="{x_img:.2f}" y="{y_img:.2f}" '
#                 f'width="{text_width:.2f}" height="{out_h:.2f}" '
#                 f'xlink:href="{data_url}" />\n')

#     out = out.replace("</svg>", image_tag + "</svg>")
#     return out



def save_svg(svg_text: str, path: str) -> None:
    """Write an SVG string to disk."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_text)
