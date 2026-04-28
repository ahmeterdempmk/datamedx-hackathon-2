import re
from typing import List, Optional

BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
NUM_RE = re.compile(r"-?\d+[.,]?\d*")


def parse_bracket_list(value) -> List[str]:
    if value is None:
        return []
    s = str(value)
    if not s or s.lower() == "nan":
        return []
    items = BRACKET_RE.findall(s)
    return [it.strip() for it in items if it.strip()]


def extract_floats(value) -> List[float]:
    items = parse_bracket_list(value)
    out = []
    for it in items:
        m = NUM_RE.search(it.replace(",", "."))
        if m:
            try:
                out.append(float(m.group()))
            except ValueError:
                pass
    return out


def latest_float(value) -> Optional[float]:
    fs = extract_floats(value)
    return fs[-1] if fs else None


def mean_float(value) -> Optional[float]:
    fs = extract_floats(value)
    return sum(fs) / len(fs) if fs else None


def clean_epikriz_text(value) -> str:
    if value is None:
        return ""
    s = str(value)
    s = s.replace("_x000D_", " ").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def strip_brackets(value) -> str:
    items = parse_bracket_list(value)
    return " | ".join(items)
