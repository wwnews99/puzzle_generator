from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Literal, Sequence

# 8 compass directions for placement (row delta, col delta)
DIR_VECTORS = {
    "E":  (0, 1),
    "W":  (0, -1),
    "N":  (-1, 0),
    "S":  (1, 0),
    "NE": (-1, 1),
    "SE": (1, 1),
    "SW": (1, -1),
    "NW": (-1, -1),
}


# -----------------------------------------------------------------------------
# Simple logger hook
# -----------------------------------------------------------------------------
# main.py can call set_logger(my_ui_logger). If you do nothing, we print().
_LOGGER = None  # type: Optional[callable]


def set_logger(fn) -> None:
    """Allow the UI to inject a logger callback: fn(text: str)."""
    global _LOGGER
    _LOGGER = fn


def _log(msg: str) -> None:
    """Log to UI if available; otherwise print. Keep messages simple."""
    if _LOGGER:
        try:
            _LOGGER(msg)
            return
        except Exception:
            pass
    print(msg)


# -----------------------------------------------------------------------------
# Data shapes used across the app
# -----------------------------------------------------------------------------
Mode = Literal["Classic", "Sentence", "Hidden", "Double-word", "Story"]
OrderPolicy = Literal["shuffled", "source", "alphabetic"]
RunOutPolicy = Literal["stop", "loop", "random"]
ReadingOrder = Literal["row-major"]
Direction = Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


@dataclass
class PlacedWord:
    """One placed word with its path in the grid."""
    text: str
    start: Tuple[int, int]  # (row, col)
    direction: Direction
    cells: List[Tuple[int, int]]  # all grid coordinates used


@dataclass
class PuzzleSpec:
    """
    Everything needed to generate a single puzzle.
    Keep this explicit and simple so it is easy to build in main.py.
    """
    mode: Mode
    grid_width: int
    grid_height: int
    num_words: int
    min_len: int
    max_len: int
    order_policy: OrderPolicy
    runout_policy: RunOutPolicy
    directions: List[Direction]
    seed: Optional[str] = None

    # Inputs (choose one or more depending on mode)
    words: List[str] = field(default_factory=list)                  # Classic
    sentences: List[str] = field(default_factory=list)              # Sentence
    pairs: List[Tuple[str, str]] = field(default_factory=list)      # Double-word (clue, answer)
    story_rows: List[Tuple[str, str, str]] = field(default_factory=list)  # (sentence, hidden, title)

    # Hidden text (optional for Classic/Sentence, required for Story)
    hidden_text: Optional[str] = None
    hidden_reading_order: ReadingOrder = "row-major"
    hidden_add_markers: bool = True

    # Where to begin embedding hidden text into leftover cells
    # "start" (default) or "center"
    hidden_start_mode: Literal["start", "center"] = "start"

    # Legend and display options that the renderer may need
    legend_columns: int = 2
    sentence_first_row_header: bool = True
    allow_sentence_reuse: bool = False
    target_word_policy: Literal["random", "longest-first"] = "random"


@dataclass
class PuzzleResult:
    """
    The outcome of the generator. This is what the renderer needs.
    """
    letters: List[List[str]]                 # final grid of letters
    used_mask: List[List[bool]]              # True where a placed word letter sits
    placed_words: List[PlacedWord]           # coordinates for solution highlighting
    legend: List[str]                        # words shown under the grid (or target words)
    sentence_text: Optional[str] = None      # for Sentence/Story modes
    hidden_text: Optional[str] = None        # normalized hidden phrase actually embedded
    hidden_start_end: Optional[Tuple[int, int]] = None  # index range in reading order (for markers)
    # --- Secret truncation metadata (for strict validation) ---
    hidden_truncated: bool = False
    hidden_written_len: int = 0
    hidden_intended_len: int = 0


# -----------------------------------------------------------------------------
# Helpers: normalization and randomness
# -----------------------------------------------------------------------------
_NORMALIZE_RE = re.compile(r"[A-Za-z0-9]+")


def _normalize_for_grid(text: str) -> str:
    """
    Remove spaces and punctuation so only letters/numbers remain.
    Uppercase so it looks consistent in the grid.
    """
    if not text:
        return ""
    parts = _NORMALIZE_RE.findall(text.upper())
    return "".join(parts)


def seed_everything(seed: Optional[str]) -> None:
    """
    Initialize deterministic randomness for the engine.

    We only use random from the stdlib. If seed is None/empty, we keep random
    in its default non-deterministic state.
    """
    if seed is None or str(seed).strip() == "":
        #_log("seed: none (non-deterministic)")
        return
    # A stable seed from the string, so the same seed gives same results.
    s = str(seed)
    random.seed(s)
    _log(f"seed: {s}")


# -----------------------------------------------------------------------------
# CSV / text loading (UI calls these)
# -----------------------------------------------------------------------------
def _read_rows(path: str) -> List[List[str]]:
    """Read CSV rows as lists of strings. Strip whitespace in each cell."""
    rows: List[List[str]] = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for r in reader:
                rows.append([c.strip() for c in r])
        _log(f"csv: loaded {len(rows)} rows from {path}")
    except Exception as e:
        _log(f"csv error: cannot read {path}: {e}")
        raise
    return rows


def load_wordlists_csv(path: str, first_row_header: bool = True) -> Dict[str, List[str]]:
    """
    Load a multi-column wordlist CSV.
    Each column becomes a named list: {header: [words...]}.
    Blank cells are ignored.
    """
    rows = _read_rows(path)
    if not rows:
        return {}
    headers: List[str]
    data_rows: List[List[str]]
    if first_row_header:
        headers = rows[0]
        data_rows = rows[1:]
    else:
        # Create generic headers Col1, Col2, ...
        max_cols = max(len(r) for r in rows)
        headers = [f"Col{i+1}" for i in range(max_cols)]
        data_rows = rows

    # Build columns
    cols: Dict[str, List[str]] = {h: [] for h in headers}
    for r in data_rows:
        for i, h in enumerate(headers):
            if i < len(r):
                val = r[i].strip()
                if val:
                    cols[h].append(val)
    _log(f"wordlists: {len(cols)} columns")
    return cols


def load_sentences_csv_multi(path: str, first_row_header: bool = True) -> Dict[str, List[str]]:
    """
    Load a sentences CSV where EACH COLUMN is a category (like Wordlists).
    Returns: {header: [sentences...]}.
    """
    rows = _read_rows(path)
    if not rows:
        return {}

    if first_row_header:
        headers = rows[0]
        data_rows = rows[1:]
    else:
        max_cols = max(len(r) for r in rows)
        headers = [f"Col{i+1}" for i in range(max_cols)]
        data_rows = rows

    cols: Dict[str, List[str]] = {h: [] for h in headers}
    for r in data_rows:
        for i, h in enumerate(headers):
            if i < len(r):
                val = r[i].strip()
                if val:
                    cols[h].append(val)
    _log(f"sentences: {len(cols)} categories")
    return cols


def load_sentences_csv(path: str, first_row_header: bool = True) -> List[str]:
    """Load a sentences CSV (use the first column)."""
    rows = _read_rows(path)
    if not rows:
        return []
    if first_row_header and len(rows) > 1:
        rows = rows[1:]
    sentences = [r[0].strip() for r in rows if r and r[0].strip()]
    _log(f"sentences: {len(sentences)} items")
    return sentences


def load_hidden_csv(path: str, first_row_header: bool = True) -> Dict[str, List[str]]:
    """
    Load hidden targets CSV (multiple themed columns).
    Each column name is a theme; each cell a hidden phrase for that theme.
    """
    rows = _read_rows(path)
    if not rows:
        return {}
    headers: List[str]
    data_rows: List[List[str]]
    if first_row_header:
        headers = rows[0]
        data_rows = rows[1:]
    else:
        max_cols = max(len(r) for r in rows)
        headers = [f"Theme{i+1}" for i in range(max_cols)]
        data_rows = rows

    cols: Dict[str, List[str]] = {h: [] for h in headers}
    for r in data_rows:
        for i, h in enumerate(headers):
            if i < len(r):
                val = r[i].strip()
                if val:
                    cols[h].append(val)
    _log(f"hidden: {len(cols)} themes")
    return cols


_PAIR_RE = re.compile(r"^(.*)\((.*)\)$")


def _parse_pair_cell(cell: str) -> Optional[Tuple[str, str]]:
    """
    Parse a cell like 'Clue (Answer)' into ('Clue', 'Answer').
    Returns None if it doesn't match.
    """
    c = cell.strip()
    if not c:
        return None
    m = _PAIR_RE.match(c)
    if not m:
        return None
    clue = m.group(1).strip()
    answer = m.group(2).strip()
    if not clue or not answer:
        return None
    return (clue, answer)


def load_double_word_csv(path: str, first_row_header: bool = True) -> Dict[str, List[Tuple[str, str]]]:
    """
    Load double-word pairs CSV. Cells look like 'Clue (Answer)'.
    """
    rows = _read_rows(path)
    if not rows:
        return {}
    headers: List[str]
    data_rows: List[List[str]]
    if first_row_header:
        headers = rows[0]
        data_rows = rows[1:]
    else:
        max_cols = max(len(r) for r in rows)
        headers = [f"Col{i+1}" for i in range(max_cols)]
        data_rows = rows

    cols: Dict[str, List[Tuple[str, str]]] = {h: [] for h in headers}
    for r in data_rows:
        for i, h in enumerate(headers):
            if i < len(r):
                pair = _parse_pair_cell(r[i])
                if pair:
                    cols[h].append(pair)
    _log(f"double-word: {len(cols)} columns")
    return cols


def load_story_csv(path: str, first_row_header: bool = True) -> List[Tuple[str, str, str]]:
    """
    Load combined Story CSV where each row already contains (Sentence | Hidden | Title).
    Uses the first 3 columns in that order.
    """
    rows = _read_rows(path)
    if not rows:
        return []
    if first_row_header and len(rows) > 1:
        rows = rows[1:]
    out: List[Tuple[str, str, str]] = []
    for r in rows:
        if len(r) < 2:
            continue
        sentence = r[0].strip()
        hidden = r[1].strip()
        title = r[2].strip() if len(r) > 2 else ""
        if sentence and hidden:
            out.append((sentence, hidden, title))
    _log(f"story rows: {len(out)}")
    return out


# -----------------------------------------------------------------------------
# Word picking / preparation
# -----------------------------------------------------------------------------
def filter_by_length(words: Sequence[str], min_len: int, max_len: int) -> List[str]:
    """
    Keep only words whose normalized length is within [min_len, max_len].
    """
    out: List[str] = []
    for w in words:
        norm = _normalize_for_grid(w)
        if min_len <= len(norm) <= max_len:
            out.append(w)
    return out


def apply_ordering(words: Sequence[str], policy: OrderPolicy, seed: Optional[str] = None) -> List[str]:
    """
    Order words before placement.
    """
    lst = list(words)
    if policy == "source":
        return lst
    if policy == "alphabetic":
        return sorted(lst, key=lambda s: s.lower())
    # shuffled
    r = random.Random(seed if seed else None)
    r.shuffle(lst)
    return lst


def _repeat_sequence(items: Sequence, total: int, policy: RunOutPolicy, seed: Optional[str]) -> List:
    """
    Return a list of length 'total' based on 'items' and policy.
    - stop: up to len(items), then truncate
    - loop: repeat from start until total reached
    - random: sample with replacement until total reached
    """
    src = list(items)
    if total <= 0 or not src:
        return []
    if policy == "stop":
        return src[:total]
    if policy == "loop":
        out = []
        i = 0
        while len(out) < total:
            out.append(src[i % len(src)])
            i += 1
        return out[:total]
    # random
    r = random.Random(seed if seed else None)
    return [r.choice(src) for _ in range(total)]


def when_inputs_run_out(items: Sequence, policy: RunOutPolicy, seed: Optional[str] = None) -> Sequence:
    """
    Return the original sequence. The "repeating" is handled in generate_one_puzzle,
    where we know the 'total' we need.
    """
    return list(items)


# -----------------------------------------------------------------------------
# Grid building (basic: horizontal placement only, simple back-off)
# -----------------------------------------------------------------------------
def _empty_grid(h: int, w: int):
    return [[None for _ in range(w)] for _ in range(h)]


def _fits_horizontal(grid, row: int, col: int, word: str) -> bool:
    w = len(grid[0])
    if col + len(word) > w:
        return False
    # Allow overlapping only if letters match
    for i, ch in enumerate(word):
        cell = grid[row][col + i]
        if cell is not None and cell != ch:
            return False
    return True


def _place_horizontal(grid, row: int, col: int, word: str) -> List[Tuple[int, int]]:
    coords = []
    for i, ch in enumerate(word):
        grid[row][col + i] = ch
        coords.append((row, col + i))
    return coords

def _can_place_word(grid, r, c, dr, dc, word_norm):
    """Check bounds and compatibility (allow crossing on identical letters)."""
    H = len(grid)
    W = len(grid[0]) if H else 0
    nr = r + dr * (len(word_norm) - 1)
    nc = c + dc * (len(word_norm) - 1)
    if nr < 0 or nr >= H or nc < 0 or nc >= W:
        return False

    rr, cc = r, c
    for ch in word_norm:
        cell = grid[rr][cc]
        if cell is not None and cell != ch:
            return False

        rr += dr
        cc += dc
    return True


def _place_one_word(grid, used_mask, r, c, dr, dc, word_norm):
    """Write the word on the grid; return list of (r,c)."""
    cells = []
    rr, cc = r, c
    for ch in word_norm:
        grid[rr][cc] = ch
        used_mask[rr][cc] = True
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _place_words_multi(spec, rng):
    """
    Greedy multi-direction placer (systematic scan):
    - Deduplicate by normalized form (A–Z/0–9, upper).
    - Longest-first ordering (better packing).
    - For each word, try all directions (shuffled), and for each, try ALL valid starts (shuffled).
      If any legal slot exists, we will find it (no backtracking required).
    - Returns (grid_letters, placed_words). 'placed_words' is sorted A→Z for a stable legend.
    """
    H = spec.grid_height
    W = spec.grid_width
    grid = [[None for _ in range(W)] for _ in range(H)]
    used_mask = [[False for _ in range(W)] for _ in range(H)]

    # Normalize directions from spec, keep only known ones
    dirs = []
    for d in (spec.directions or []):
        d = (d or "").upper()
        if d in DIR_VECTORS:
            dirs.append(d)
    if not dirs:
        dirs = ["E"]  # fallback

    # Helpers
    def _norm(s: str) -> str:
        return "".join(ch for ch in str(s).upper() if ch.isalnum())

    # Deduplicate by normalized form, keep first occurrence
    seen = set()
    uniq_norm_words = []
    for w in (spec.words or []):
        n = _norm(w)
        if not n:
            continue
        if n in seen:
            continue
        seen.add(n)
        uniq_norm_words.append(n)

    # Respect requested count if provided
    if spec.num_words and len(uniq_norm_words) > spec.num_words:
        uniq_norm_words = uniq_norm_words[:spec.num_words]

    # Order longest-first (best for packing); shuffle equal-length runs to avoid identical layouts
    uniq_norm_words.sort(key=len, reverse=True)
    i = 0
    while i < len(uniq_norm_words):
        L = len(uniq_norm_words[i])
        j = i + 1
        while j < len(uniq_norm_words) and len(uniq_norm_words[j]) == L:
            j += 1
        if j - i > 1:
            rng.shuffle(uniq_norm_words[i:j])
        i = j

    placed = []
    PlacedWordCls = globals().get("PlacedWord", None)

    for word_norm in uniq_norm_words:
        # Shuffle direction order for variety
        dir_list = list(dirs)
        rng.shuffle(dir_list)

        placed_this = False

        for d in dir_list:
            dr, dc = DIR_VECTORS[d]
            L = len(word_norm)

            # Compute all valid start positions that keep the word fully in-bounds
            if dr < 0:
                r_min, r_max = L - 1, H - 1
            elif dr > 0:
                r_min, r_max = 0, H - L
            else:
                r_min, r_max = 0, H - 1

            if dc < 0:
                c_min, c_max = L - 1, W - 1
            elif dc > 0:
                c_min, c_max = 0, W - L
            else:
                c_min, c_max = 0, W - 1

            if r_min > r_max or c_min > c_max:
                continue  # direction cannot fit this word length in this grid

            starts = [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
            rng.shuffle(starts)  # variety across runs

            for (r, c) in starts:
                if _can_place_word(grid, r, c, dr, dc, word_norm):
                    cells = _place_one_word(grid, used_mask, r, c, dr, dc, word_norm)

                    if PlacedWordCls:
                        placed.append(PlacedWordCls(text=word_norm, start=(r, c), direction=d, cells=cells))
                    else:
                        tmp = type("PW", (), {})()
                        tmp.text = word_norm
                        tmp.start = (r, c)
                        tmp.direction = d
                        tmp.cells = cells
                        placed.append(tmp)

                    placed_this = True
                    break  # next word
            if placed_this:
                break

        if not placed_this:
            try:
                _log(f"place: could not place '{word_norm}' in {W}x{H} with dirs={dir_list}, skipping it")
            except Exception:
                pass

    # Deterministic legend order (A→Z); grid remains as placed
    placed.sort(key=lambda pw: pw.text.upper())
    return grid, placed


def _estimate_capacity_greedy(spec: PuzzleSpec, rng: random.Random, per_word_tries: int = 60) -> int:
    """
    Dry-run estimator: tries to place the requested words on a throwaway grid
    using the SAME rules as the real placer, but with limited tries per word.
    Returns how many it managed to place. This makes the preflight number
    track real behavior much more closely than the old algebraic guess.
    """
    # Clone essential bits (don’t mutate spec)
    H, W = spec.grid_height, spec.grid_width
    dirs = []
    for d in (spec.directions or []):
        d = (d or "").upper()
        if d in DIR_VECTORS:
            dirs.append(d)
    if not dirs:
        dirs = ["E"]

    # Normalize + filter by min/max length; take only up to num_words if set
    words = []
    for w in (spec.words or []):
        n = _normalize_for_grid(w)
        if n and (spec.min_len <= len(n) <= spec.max_len):
            words.append(n)
    if spec.num_words and len(words) > spec.num_words:
        words = words[: spec.num_words]

    # Place longer words first (usually improves fit)
    words.sort(key=len, reverse=True)

    # Throwaway grid (None = empty)
    grid: List[List[Optional[str]]] = _empty_grid(H, W)
    used_mask: List[List[bool]] = [[False for _ in range(W)] for _ in range(H)]


    placed_count = 0

    for nword in words:
        success = False
        # Limit attempts per word to keep it fast
        tries = max(10, per_word_tries)
        for _ in range(tries):
            d = rng.choice(dirs)
            dr, dc = DIR_VECTORS[d]
            r = rng.randrange(0, H)
            c = rng.randrange(0, W)
            if not _can_place_word(grid, r, c, dr, dc, nword):
                continue
            _place_one_word(grid, used_mask, r, c, dr, dc, nword)

            placed_count += 1
            success = True
            break
        # If it didn’t fit within tries, skip and move on
    return placed_count



def place_words(spec: PuzzleSpec) -> Tuple[List[List[Optional[str]]], List[PlacedWord]]:
    """
    Place words into a grid. For now we only place left-to-right (E).
    If the spec allows other directions, we log that we are ignoring them for now.
    """
    h, w = spec.grid_height, spec.grid_width
    grid = _empty_grid(h, w)

    # Normalize chosen words for placement (spaces/punct removed, uppercase)
    chosen = [ _normalize_for_grid(wd) for wd in spec.words if _normalize_for_grid(wd) ]

    # Warn about directions we do not support yet
    allowed = set(spec.directions)
    if allowed - {"E"}:
        _log("note: only 'E' (left-to-right) placement is implemented in this step.")

    r = random.Random(spec.seed if spec.seed else None)
    placed: List[PlacedWord] = []

    for word in chosen:
        # Try a handful of random positions
        success = False
        for _ in range(200):
            row = r.randrange(0, h)
            col = r.randrange(0, w)
            if _fits_horizontal(grid, row, col, word):
                coords = _place_horizontal(grid, row, col, word)
                placed.append(PlacedWord(text=word, start=(row, col), direction="E", cells=coords))
                success = True
                break
        if not success:
            _log(f"place: could not place '{word}' in {h}x{w}, skipping it")

        if len(placed) >= spec.num_words:
            break

    return grid, placed


def fill_grid(grid: List[List[Optional[str]]], seed: Optional[str] = None) -> List[List[str]]:
    """
    Fill None cells with random uppercase letters.
    """
    r = random.Random(seed if seed else None)
    out: List[List[str]] = []
    for row in grid:
        new_row = []
        for cell in row:
            if cell is None or cell == "":

                new_row.append(chr(r.randrange(ord('A'), ord('Z') + 1)))
            else:
                new_row.append(cell)
        out.append(new_row)
    return out


# -----------------------------------------------------------------------------
# Hidden reveal (write into leftover cells in reading order)
# -----------------------------------------------------------------------------
def _leftover_indices(used_mask: List[List[bool]], order: ReadingOrder) -> List[Tuple[int, int]]:
    """Return a list of (row, col) pairs for cells not used by words."""
    indices: List[Tuple[int, int]] = []
    if order == "row-major":
        for r in range(len(used_mask)):
            for c in range(len(used_mask[0])):
                if not used_mask[r][c]:
                    indices.append((r, c))
    else:
        # Only one order supported for now
        for r in range(len(used_mask)):
            for c in range(len(used_mask[0])):
                if not used_mask[r][c]:
                    indices.append((r, c))
    return indices


def embed_hidden_via_leftovers(
    letters: List[List[str]],
    used_mask: List[List[bool]],
    hidden_text: str,
    reading_order: ReadingOrder = "row-major",
    add_markers: bool = True,
    start_mode: str = "start",
) -> Tuple[str, Optional[Tuple[int, int]]]:
    """
    Write the hidden phrase into leftover cells. We replace filler letters only;
    we never overwrite placed words.
    """
    normalized = _normalize_for_grid(hidden_text)
    if not normalized:
        _log("hidden: empty after normalization; nothing to embed")
        return "", None

    spots = _leftover_indices(used_mask, reading_order)
    if not spots:
        _log("hidden: no leftover cells available")
        return "", None

    # If not enough space, we embed what we can
    count = min(len(normalized), len(spots))
    # if count < len(normalized):
    #     try:
    #         _log(f"hidden: truncated, wrote {count} of {len(normalized)} chars")
    #     except Exception:
    #         pass

    # Compute starting offset in leftovers
    start_idx = 0
    if str(start_mode).lower().startswith("c"):
        start_idx = max(0, (len(spots) - count) // 2)

    end_idx = start_idx + count - 1 if count > 0 else 0

    for i in range(count):
        r, c = spots[start_idx + i]
        letters[r][c] = normalized[i]

    if count < len(normalized):
        _log(f"WARNING-hidden: truncated, wrote {count} of {len(normalized)} chars")
   

    # Renderer can use (start_idx, end_idx) to mark secret endpoints if desired.
    return normalized[:count], (start_idx, end_idx)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------
def generate_one_puzzle(spec: PuzzleSpec, rng: Optional[random.Random] = None) -> PuzzleResult:
    """
    Orchestrator:
      - choose words according to mode (Classic/Sentence/Story use spec.words from UI)
      - normalize/dedupe + TOP-UP from pool to hit requested count when possible
      - place multi-direction (uses provided rng to stay in the same sequence across a batch)
      - fill empty cells (also consumes provided rng)
      - optional hidden embed into leftovers
      - build legend from actually placed words (mapped to original case)
    """
    # IMPORTANT:
    # - Do NOT reseed inside; if rng is provided we must keep consuming it across the batch.
    # - If rng is None (single build), fall back to a local rng seeded from spec.seed.
    _rng = rng if rng is not None else random.Random(spec.seed if spec.seed else None)
    try:
        if spec.seed is None or (isinstance(spec.seed, str) and not spec.seed.strip()):
            _log("seed: none (non-deterministic)")
        else:
            _log(f"seed: {spec.seed}")
    except Exception:
        pass

    # 1) source words (as provided by UI per mode)
    source_words = list(spec.words or [])

    # Sentence/Story: log unique usable targets in the sentence
    try:
        if str(getattr(spec, "mode", "")).lower() in ("sentence", "story"):
            uniq = set()
            for w in source_words:
                n = _normalize_for_grid(w)
                if n and spec.min_len <= len(n) <= spec.max_len:
                    uniq.add(n)
            if uniq:
                _log(f"[sentence] unique usable targets in sentence: {len(uniq)}")
    except Exception:
        pass

    # 2) length filter & ordering of the FULL pool (not yet capped).
    # NOTE: ordering/runout use spec.seed so that, for a given seed, the pool order is stable.
    filtered_full = filter_by_length(source_words, spec.min_len, spec.max_len)
    ordered_full  = apply_ordering(filtered_full, spec.order_policy, spec.seed)

    # 3) Initial pick according to runout policy
    initial = _repeat_sequence(ordered_full, spec.num_words, spec.runout_policy, spec.seed)

    # 4) Normalize + dedupe the initial pick, then TOP-UP from the rest of the pool
    def _norm(s: str) -> str:
        return _normalize_for_grid(s)

    want = int(spec.num_words or 0)
    seen_norm: set[str] = set()
    chosen: list[str] = []

    # 4a) take from initial (preserves the policy ordering)
    for w in (initial or []):
        n = _norm(w)
        if not n or n in seen_norm:
            continue
        seen_norm.add(n)
        chosen.append(w)
        if want and len(chosen) >= want:
            break

    # 4b) top-up from remaining ordered_full pool if we still need more
    if want and len(chosen) < want:
        for w in ordered_full:
            n = _norm(w)
            if not n or n in seen_norm:
                continue
            seen_norm.add(n)
            chosen.append(w)
            if len(chosen) >= want:
                break

    # For placement we use the chosen list; legend will show display strings later
    spec.words = chosen
    spec.num_words = len(chosen)

    # ---- preflight capacity estimate (log-only, deterministic from spec.seed) ----
    try:
        req_seen = set()
        req = []
        for w in (spec.words or []):
            n = _norm(w)
            if n and (spec.min_len <= len(n) <= spec.max_len) and (n not in req_seen):
                req_seen.add(n)
                req.append(n)
        if spec.num_words and len(req) > spec.num_words:
            req = req[: spec.num_words]
        N = len(req)

        est_rng = random.Random(spec.seed if spec.seed is not None else None)
        for _ in range(7):
            est_rng.random()
        est_fit = _estimate_capacity_greedy(spec, est_rng, per_word_tries=50)
        dirs_count = len([d for d in (spec.directions or []) if str(d).upper() in DIR_VECTORS]) or 1
        status = "WARN" if N > est_fit else "OK"
        _log(
            f"[preflight/{status}] requesting {N} words in {spec.grid_width}x{spec.grid_height} "
            f"with {dirs_count} dir(s): estimated fit ~ {est_fit}"
        )
    except Exception as _e:
        _log(f"[preflight][WARN] estimate failed: {_e}")
    # ---- end preflight ----

    # 5) place (multi-direction) — using backoff-capable placer
    # >>> DO NOT re-create _rng here; use the one from above <<<
    raw_grid, placed = _place_words_with_backoff(spec, _rng)

    # 6) build used mask
    h, w = spec.grid_height, spec.grid_width
    used = [[False] * w for _ in range(h)]
    for pw in placed:
        for (r, c) in pw.cells:
            used[r][c] = True

    # 7) fill the rest of the cells with random letters (consume the SAME _rng)
    letters = [["" for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            ch = raw_grid[r][c]
            letters[r][c] = ch if ch else _rand_letter(_rng)

    # 8) optional hidden embed via leftovers
    hidden_text_norm, hidden_span = "", None
    _hidden_intended_norm = ""
    _hidden_written_len = 0
    _hidden_intended_len = 0
    _hidden_truncated = False
    # compute intended normalized first to detect truncation
    _hidden_intended_norm = _normalize_for_grid(spec.hidden_text)
    _hidden_intended_len = len(_hidden_intended_norm)
    hidden_text_norm, hidden_span = embed_hidden_via_leftovers(
            letters, used, spec.hidden_text, spec.hidden_reading_order, spec.hidden_add_markers,
            start_mode=getattr(spec, "hidden_start_mode", "start")
        )
    _hidden_written_len = len(hidden_text_norm or "")
    _hidden_truncated = (_hidden_intended_len > _hidden_written_len)

    # 9) legend from actually placed words (use originals if available; fallback to normalized)
    # Map normalized -> first original in 'chosen'
    disp_map: dict[str, str] = {}
    for w0 in chosen:
        n0 = _norm(w0)
        if n0 and n0 not in disp_map:
            disp_map[n0] = w0

    legend: list[str] = []
    for pw in placed:
        legend.append(disp_map.get(pw.text, pw.text))
    # keep A–Z order stable for the legend (placed is already sorted for drawing)
    legend = sorted(list(dict.fromkeys(legend)), key=lambda s: _normalize_for_grid(s))

    return PuzzleResult(
        letters=letters,
        used_mask=used,
        placed_words=placed,
        legend=legend,
        sentence_text=None,
        hidden_text=hidden_text_norm or (spec.hidden_text or ""),  # << ensure echo
        hidden_start_end=hidden_span,
        hidden_truncated=_hidden_truncated,
        hidden_written_len=_hidden_written_len,
        hidden_intended_len=_hidden_intended_len,
    )

def _rand_letter(rng: random.Random) -> str:
    # Uppercase A–Z
    return chr(ord('A') + rng.randrange(26))


def render_preview_ascii(result: PuzzleResult) -> str:
    """
    Simple ASCII for quick debugging.
    """
    lines = []
    for row in result.letters:
        lines.append(" ".join(ch if ch else "." for ch in row))
    return "\n".join(lines)

def _place_words_with_backoff(
    spec: "PuzzleSpec",
    rng: random.Random,
    *,
    max_attempts_per_word: int = 200,
    backoff_remove: int = 3,
    backoff_retries: int = 3,
) -> tuple[list[list[Optional[str]]], list[PlacedWord]]:
    """
    Backoff-aware multi-direction placer.
    Signature and return shape are IDENTICAL to _place_words_multi(spec, rng):
        -> returns (grid_letters, placed_words)

    Strategy:
      - Normalize & dedupe words by grid form (A–Z/0–9, upper).
      - Order longest-first; shuffle equal-length runs via rng.
      - For each word:
          try all dirs + in-bounds starts (rng-shuffled).
          If it doesn't fit, temporarily remove a few shortest placed words,
          rebuild, retry the long word, then try to reinsert removed ones.

    Notes:
      - This function *does not* build mask or legend; generate_one_puzzle()
        already does that later. Keep return as (grid, placed).
    """
    H = int(spec.grid_height)
    W = int(spec.grid_width)

    # Throwaway grid (None = empty for compatibility with the rest of the file)
    grid: list[list[Optional[str]]] = _empty_grid(H, W)  # reuses engine helper
    used_mask: list[list[bool]] = [[False for _ in range(W)] for _ in range(H)]

    # Normalize directions from spec, keep only known ones (fallback to E)
    dirs: list[str] = []
    for d in (spec.directions or []):
        d = (d or "").upper()
        if d in DIR_VECTORS:
            dirs.append(d)
    if not dirs:
        dirs = ["E"]

    # Helpers
    def _norm(s: str) -> str:
        return "".join(ch for ch in str(s).upper() if ch.isalnum())

    # Deduplicate by normalized form, length filter, then respect num_words
    seen = set()
    uniq_norm_words: list[str] = []
    for w in (spec.words or []):
        n = _norm(w)
        if not n:
            continue
        if not (spec.min_len <= len(n) <= spec.max_len):
            continue
        if n in seen:
            continue
        seen.add(n)
        uniq_norm_words.append(n)

    if spec.num_words and len(uniq_norm_words) > spec.num_words:
        uniq_norm_words = uniq_norm_words[: spec.num_words]

    # Longest-first; shuffle equal-length runs for variety (deterministic with rng)
    uniq_norm_words.sort(key=len, reverse=True)
    i = 0
    while i < len(uniq_norm_words):
        L = len(uniq_norm_words[i])
        j = i + 1
        while j < len(uniq_norm_words) and len(uniq_norm_words[j]) == L:
            j += 1
        if j - i > 1:
            rng.shuffle(uniq_norm_words[i:j])
        i = j

    placed: list[PlacedWord] = []
    PlacedWordCls = globals().get("PlacedWord", None)

    def _rebuild_grid_from_current() -> tuple[list[list[Optional[str]]], list[list[bool]]]:
        g = _empty_grid(H, W)
        m = [[False for _ in range(W)] for _ in range(H)]
        for pw in placed:
            for (r, c), ch in zip(pw.cells, pw.text):
                g[r][c] = ch
                m[r][c] = True
        return g, m

    def _try_place_word(word_norm: str) -> Optional[PlacedWord]:
        # Shuffle directions (variety per attempt)
        dir_list = list(dirs)
        rng.shuffle(dir_list)

        for d in dir_list:
            dr, dc = DIR_VECTORS[d]
            L = len(word_norm)

            # compute in-bounds start ranges for this direction
            if dr < 0:
                r_min, r_max = L - 1, H - 1
            elif dr > 0:
                r_min, r_max = 0, H - L
            else:
                r_min, r_max = 0, H - 1

            if dc < 0:
                c_min, c_max = L - 1, W - 1
            elif dc > 0:
                c_min, c_max = 0, W - L
            else:
                c_min, c_max = 0, W - 1

            if r_min > r_max or c_min > c_max:
                continue  # direction cannot fit this length

            starts = [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
            rng.shuffle(starts)

            attempts = 0
            for (r, c) in starts:
                attempts += 1
                if attempts > max_attempts_per_word:
                    break
                if _can_place_word(grid, r, c, dr, dc, word_norm):  # engine helper
                    cells = _place_one_word(grid, used_mask, r, c, dr, dc, word_norm)  # engine helper
                    if PlacedWordCls:
                        return PlacedWordCls(text=word_norm, start=(r, c), direction=d, cells=cells)
                    else:
                        tmp = type("PW", (), {})()
                        tmp.text = word_norm
                        tmp.start = (r, c)
                        tmp.direction = d
                        tmp.cells = cells
                        return tmp
        return None

    for word_norm in uniq_norm_words:
        # fast path
        pw = _try_place_word(word_norm)
        if pw:
            placed.append(pw)
            continue

        # backoff path: remove a few shortest placed words, retry the long one
        removed_stack: list[PlacedWord] = []
        success = False

        for _ in range(max(1, int(backoff_retries))):
            if not placed:
                break

            # choose up to backoff_remove shortest (prefer strictly shorter than target)
            placed_sorted = sorted(placed, key=lambda p: len(p.text))
            to_remove = [p for p in placed_sorted if len(p.text) < len(word_norm)][:backoff_remove]
            if not to_remove:
                to_remove = placed_sorted[:backoff_remove]  # fallback to absolute shortest few

            # remove from 'placed'
            ids = {id(p) for p in to_remove}
            kept = [p for p in placed if id(p) not in ids]
            removed_stack.extend(to_remove)
            placed = kept

            # rebuild grid without removed ones
            grid, used_mask = _rebuild_grid_from_current()

            # retry the target
            pw = _try_place_word(word_norm)
            if pw:
                placed.append(pw)
                success = True
                # try to reinsert removed ones, smallest-first
                readded = 0
                for p_old in sorted(removed_stack, key=lambda p: len(p.text)):
                    p2 = _try_place_word(p_old.text)
                    if p2:
                        placed.append(p2)
                        readded += 1
                grid, used_mask = _rebuild_grid_from_current()
                try:
                    _log(f"[backoff] placed '{word_norm}' after removing {len(removed_stack)}; reinserted {readded}")
                except Exception:
                    pass
                break

        if not success:
            # best-effort reinsert what we removed (longer-first fairness)
            readded = 0
            for p_old in sorted(removed_stack, key=lambda p: len(p.text), reverse=True):
                p2 = _try_place_word(p_old.text)
                if p2:
                    placed.append(p2)
                    readded += 1
            grid, used_mask = _rebuild_grid_from_current()
            try:
                _log(f"[backoff] gave up on '{word_norm}' ({len(word_norm)}); reinserted {readded} of {len(removed_stack)}")
            except Exception:
                pass

    # Stable legend order later — but keep placed A→Z now for consistency with _place_words_multi
    placed.sort(key=lambda pw: pw.text.upper())
    return grid, placed

