"""
Nested list parser for regulatory documents.

Regulatory compliance documents use deeply nested enumerations::

    1. The MCP shall maintain all of the following records:
       a. Personnel files for each employee, including:
          i.  Training certifications (dated within the past year)
          ii. Background check results
       b. Credentialing documentation for contracted providers

The current PDF parser treats these as flat body text, causing the LLM
to miss the parent–child relationship between sub-items and their enclosing
obligation.  This module parses such text into a tree and exposes the result
via a :class:`smolagents.Tool` so the extraction agent can walk the hierarchy
when formulating compliance requirements (improving Step 6: hierarchy assembly).

Public API
----------
- :func:`parse_nested_list` — pure-Python entry point: text → list[dict]
- :class:`ParseNestedListTool` — smolagents wrapper
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from smolagents import Tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Marker type classification
# ---------------------------------------------------------------------------

# Known lowercase roman numeral strings (up to xxv covers 25 sub-items, which
# is more than sufficient for any regulatory document list in practice).
_LOWER_ROMAN_MARKERS: frozenset[str] = frozenset({
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
    "xxi", "xxii", "xxiii", "xxiv", "xxv",
})

# Canonical nesting order in DHCS/CMS regulatory documents:
#   decimal (1, 2, 3) → lower_alpha (a, b, c) → lower_roman (i, ii, iii) → upper_alpha (A, B, C)
_TYPE_PRIORITY: dict[str, int] = {
    "decimal":     1,
    "lower_alpha": 2,
    "lower_roman": 3,
    "upper_alpha": 4,
    "unknown":     0,
}


def _classify_marker(marker: str) -> str:
    """
    Classify a list-item marker string into a canonical type.

    Returns one of: ``"decimal"``, ``"lower_alpha"``, ``"lower_roman"``,
    ``"upper_alpha"``, or ``"unknown"``.

    Classification rules:

    - Pure digit string → ``"decimal"``
    - Lowercase string that is a known roman numeral (see
      :data:`_LOWER_ROMAN_MARKERS`) → ``"lower_roman"``
    - Other single lowercase letter → ``"lower_alpha"``
    - Single uppercase letter → ``"upper_alpha"``

    .. note::
        Single ``"i"`` is classified as ``"lower_roman"`` because it is
        almost universally used as a level-3 roman numeral in regulatory
        documents.  When ``i`` appears in an alphabetical run
        (a, b, c, …, i), indentation is the primary disambiguation
        mechanism and will place it at the correct level regardless.
    """
    if re.fullmatch(r"\d+", marker):
        return "decimal"
    if marker.islower() and marker in _LOWER_ROMAN_MARKERS:
        return "lower_roman"
    if re.fullmatch(r"[a-z]", marker):
        return "lower_alpha"
    if re.fullmatch(r"[A-Z]", marker):
        return "upper_alpha"
    return "unknown"


# ---------------------------------------------------------------------------
# Line scanning
# ---------------------------------------------------------------------------

# Known multi-char roman numerals listed longest-first so the regex
# alternation does not greedily consume a prefix (e.g. "xx" before "xxi").
_ROMAN_MULTI = (
    "xxv|xxiv|xxiii|xxii|xxi|xx|"
    "xix|xviii|xvii|xvi|xv|xiv|xiii|xii|xi|"
    "ix|viii|vii|vi|iv|iii|ii"
)

# A valid list marker is one of:
#   • 1–3 decimal digits (covers lists up to item 999)
#   • a single uppercase or lowercase letter
#   • a known multi-char roman numeral (lowercase)
#
# Using the single-letter branch for "i", "v", etc. is intentional: the
# _classify_marker function handles the roman/alpha ambiguity based on the
# full marker string after extraction.
_MARKER_PAT = rf"(?:\d{{1,3}}|[A-Za-z]|{_ROMAN_MULTI})"

# Matches a list item at the start of a line.
#
# Named groups
# ~~~~~~~~~~~~
# indent        — leading whitespace (used to infer nesting depth)
# paren_marker  — marker inside parentheses: "(a)" "(iii)" "(1)"
# bare_marker   — marker before "." or ")": "1." "a)" "iii."
# text          — item body text (everything after the marker+delimiter)
_LINE_ITEM_RE = re.compile(
    rf"^(?P<indent>[ \t]*)"
    rf"(?:"
    rf"  \((?P<paren_marker>{_MARKER_PAT})\)[ \t]+"   # parenthetical: (marker)
    rf"  |"
    rf"  (?P<bare_marker>{_MARKER_PAT})[.)][ \t]+"    # bare: marker. or marker)
    rf")"
    rf"(?P<text>\S.*)$",
    re.VERBOSE,
)


@dataclass
class _RawItem:
    """A list item detected during line scanning, before tree assembly."""

    indent: int            # leading whitespace depth (tabs expanded to 4 spaces)
    marker: str            # original-case marker, e.g. "1", "a", "iii", "A"
    marker_type: str       # "decimal" | "lower_alpha" | "lower_roman" | "upper_alpha" | "unknown"
    marker_display: str    # as it appears in text, e.g. "1.", "(a)", "iii."
    text: str              # body text (continuation lines are appended here)


def _scan_items(text: str) -> list[_RawItem]:
    """
    Scan *text* line by line and return a flat list of raw list items.

    Lines that do not start a new list item but follow one (continuation
    lines — non-blank, non-marker) are appended to the preceding item's
    ``text`` field so that multi-line items are captured intact.
    """
    items: list[_RawItem] = []

    for line in text.splitlines():
        m = _LINE_ITEM_RE.match(line)
        if m:
            indent_str = m.group("indent") or ""
            indent = len(indent_str.expandtabs(4))

            paren = m.group("paren_marker")
            bare = m.group("bare_marker")
            actual = paren if paren is not None else bare  # preserve original case

            # Determine the display form from what was actually in the source
            if paren is not None:
                display = f"({actual})"
            else:
                # Look at the character immediately after the marker to get . vs )
                stripped = line.lstrip()
                sep_idx = len(actual)
                sep = stripped[sep_idx] if sep_idx < len(stripped) else "."
                display = f"{actual}{sep}"

            items.append(_RawItem(
                indent=indent,
                marker=actual,
                marker_type=_classify_marker(actual),
                marker_display=display,
                text=m.group("text").strip(),
            ))

        elif items and line.strip():
            # Non-blank, non-item line → continuation of the last item
            items[-1].text += " " + line.strip()

    return items


# ---------------------------------------------------------------------------
# Level assignment
# ---------------------------------------------------------------------------


def _assign_levels(items: list[_RawItem]) -> list[tuple[_RawItem, int]]:
    """
    Assign a 1-indexed nesting level to each raw item.

    Two strategies (tried in order):

    1. **Indentation-first** — when items have more than one distinct
       indentation value, map each unique indent to a level (smallest = 1).
       This is reliable for most PDF-extracted regulatory text where
       different nesting depths land at different x-positions.

    2. **Type-hierarchy fallback** — when all items share the same
       indentation (plain-text extraction without spatial info), use the
       canonical ``decimal → lower_alpha → lower_roman → upper_alpha``
       order together with a growing type stack that tracks the current
       nesting path.  Encountering a type already in the stack means the
       document is returning to a shallower level.
    """
    if not items:
        return []

    indent_values = sorted(set(it.indent for it in items))
    if len(indent_values) > 1:
        # Strategy 1: indentation is informative
        indent_to_level = {v: i + 1 for i, v in enumerate(indent_values)}
        return [(item, indent_to_level[item.indent]) for item in items]

    # Strategy 2: type-stack fallback
    type_stack: list[str] = []   # marker types in the current nesting path
    result: list[tuple[_RawItem, int]] = []

    for item in items:
        mt = item.marker_type
        if mt == "unknown":
            # Keep at the current depth; default to 1 if stack is empty
            level = max(len(type_stack), 1)
        elif mt in type_stack:
            # Pop back to the level where this type was first established
            idx = type_stack.index(mt)
            type_stack = type_stack[: idx + 1]
            level = idx + 1
        else:
            # New marker type → push (deeper nesting level)
            type_stack.append(mt)
            level = len(type_stack)

        result.append((item, level))

    return result


# ---------------------------------------------------------------------------
# Tree building
# ---------------------------------------------------------------------------


@dataclass
class ListNode:
    """One node in the parsed enumeration tree."""

    marker: str                        # display form, e.g. "1.", "(a)", "iii."
    text: str                          # item body text
    level: int                         # 1-indexed nesting depth
    path: str                          # dotted breadcrumb, e.g. "1.a.iii"
    children: list[ListNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "text": self.text,
            "level": self.level,
            "path": self.path,
            "children": [c.to_dict() for c in self.children],
        }


def _build_tree(items: list[_RawItem]) -> list[ListNode]:
    """
    Assemble a :class:`ListNode` tree from raw items with assigned levels.

    Uses a node stack where ``stack[L-1]`` is the most recent node at
    nesting level *L*.  Adding a node at level *L*:

    1. Truncates the stack to length *L-1* (discards deeper levels).
    2. Appends the node to ``stack[L-2].children`` (parent at level *L-1*).
    3. Pushes the node onto the stack.
    """
    levelled = _assign_levels(items)
    if not levelled:
        return []

    roots: list[ListNode] = []
    stack: list[ListNode] = []  # stack[i] = most recent node at level i+1

    for item, level in levelled:
        # Build a dotted breadcrumb path using the lowercase marker value
        marker_key = item.marker.lower()
        if level == 1 or not stack:
            path = marker_key
        elif len(stack) >= level - 1:
            path = f"{stack[level - 2].path}.{marker_key}"
        else:
            path = marker_key  # orphan node; shouldn't occur in well-formed input

        node = ListNode(
            marker=item.marker_display,
            text=item.text,
            level=level,
            path=path,
        )

        # Trim the stack to the parent level
        del stack[level - 1:]

        if level == 1:
            roots.append(node)
        elif stack:
            stack[-1].children.append(node)

        stack.append(node)

    return roots


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_nested_list(text: str) -> list[dict[str, Any]]:
    """
    Parse regulatory text with nested enumerations into a JSON-serialisable tree.

    Recognised enumeration styles
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * Decimal:         ``1.``  ``2.``  ``(1)``  ``1)``
    * Lowercase alpha: ``a.``  ``b.``  ``(a)``  ``a)``
    * Lowercase roman: ``i.``  ``ii.`` ``(i)``  ``i)``
    * Uppercase alpha: ``A.``  ``B.``  ``(A)``  ``A)``

    Multi-line items (text that wraps onto the next line without starting a
    new marker) are joined into a single ``text`` value.

    When indentation is present (typical for PDF-extracted text where each
    nesting level has a different x-position) it is used as the primary
    depth signal.  When all items share the same indentation, the canonical
    ``decimal → lower_alpha → lower_roman → upper_alpha`` type hierarchy is
    used as a fallback.

    Args:
        text: Section body text that may contain nested list items.

    Returns:
        A list of top-level node dicts.  Each dict has keys:

        * ``marker`` (str) — display form, e.g. ``"1."`` or ``"(a)"``
        * ``text`` (str) — item body text
        * ``level`` (int) — 1-indexed nesting depth
        * ``path`` (str) — dotted breadcrumb, e.g. ``"1.a.i"``
        * ``children`` (list) — nested sub-items (same structure)

        Returns an empty list when no list structure is detected in *text*.
    """
    items = _scan_items(text)
    if not items:
        return []
    tree = _build_tree(items)
    return [node.to_dict() for node in tree]


# ---------------------------------------------------------------------------
# Smolagents tool
# ---------------------------------------------------------------------------


class ParseNestedListTool(Tool):
    """
    Smolagents tool that parses regulatory text containing nested enumerations
    into a JSON tree structure so the LLM can correctly associate sub-items
    with their parent obligations.

    Regulatory documents use deeply nested list patterns such as::

        1. The MCP shall maintain all of the following records:
           a. Personnel files for each employee, including:
              i.  Training certifications (within the past year)
              ii. Background check results
           b. Credentialing documentation for all contracted providers

    This tool returns a JSON tree where each node carries a ``marker``,
    ``text``, ``level``, ``path``, and ``children`` list.  Walking the tree
    lets you extract each sub-obligation together with the full breadcrumb
    context that shows which higher-level requirement it belongs to.

    Typical usage: after calling ``strip_headers_footers``, check the cleaned
    text for nested list structures.  If present, call this tool and traverse
    the resulting tree level-by-level instead of scanning the raw text
    linearly — this prevents incorrectly treating sub-items as independent
    top-level obligations.

    Returns ``"[]"`` (JSON empty array) when no nested list structure is
    detected in the input, so it is safe to call on any section text.
    """

    name = "parse_nested_list"
    description = (
        "Parse regulatory text containing nested enumerations "
        "(e.g. 1 \u2192 a \u2192 i \u2192 A) into a JSON tree. "
        "Each node in the output has 'marker', 'text', 'level', 'path', "
        "and 'children' keys. Use this when the cleaned section text "
        "contains deeply nested numbered or lettered sub-clauses so you "
        "can correctly attribute each sub-item to its parent obligation. "
        "Returns '[]' when no list structure is found."
    )
    inputs = {
        "text": {
            "type": "string",
            "description": (
                "The section text (or a relevant subset) containing nested "
                "list items to parse. Typically the cleaned text returned "
                "by strip_headers_footers."
            ),
        }
    }
    output_type = "string"

    def forward(self, text: str) -> str:  # type: ignore[override]
        if not text or not text.strip():
            return "[]"
        try:
            tree = parse_nested_list(text)
            if not tree:
                return "[]"
            return json.dumps(tree, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("ParseNestedListTool: parsing failed: %s", exc)
            return json.dumps({"error": str(exc)})
