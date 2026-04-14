"""
Tests for backend/tools/nested_list_parser.py.

Covers all parsing stages:
  _classify_marker  — marker string → type string
  _scan_items       — raw line scanning + continuation-line handling
  _assign_levels    — indent-first strategy, type-hierarchy fallback
  parse_nested_list — end-to-end: text → list[dict] tree
  ParseNestedListTool — smolagents tool wrapper

Test fixture text tries to represent real DHCS/CMS regulatory language to
ensure the tool works correctly on the patterns it will actually encounter.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from backend.tools.nested_list_parser import (
    ListNode,
    ParseNestedListTool,
    _RawItem,
    _assign_levels,
    _build_tree,
    _classify_marker,
    _scan_items,
    parse_nested_list,
)


# ===========================================================================
# _classify_marker
# ===========================================================================


class TestClassifyMarker:
    def test_single_digit_decimal(self):
        assert _classify_marker("1") == "decimal"

    def test_multi_digit_decimal(self):
        assert _classify_marker("42") == "decimal"

    def test_three_digit_decimal(self):
        assert _classify_marker("100") == "decimal"

    def test_lowercase_alpha_common(self):
        for letter in ("a", "b", "c", "d", "e", "f", "g", "h"):
            assert _classify_marker(letter) == "lower_alpha", f"failed for '{letter}'"

    def test_lowercase_alpha_later_alphabet(self):
        # Letters that are NOT roman numerals
        for letter in ("j", "k", "n", "o", "p", "q", "r", "s", "t", "u", "w", "y", "z"):
            assert _classify_marker(letter) == "lower_alpha", f"failed for '{letter}'"

    def test_single_i_is_lower_roman(self):
        # 'i' is treated as roman numeral (the canonical level-3 case)
        assert _classify_marker("i") == "lower_roman"

    def test_multi_char_lower_roman(self):
        for marker in ("ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"):
            assert _classify_marker(marker) == "lower_roman", f"failed for '{marker}'"

    def test_upper_alpha(self):
        for letter in ("A", "B", "C", "Z"):
            assert _classify_marker(letter) == "upper_alpha", f"failed for '{letter}'"

    def test_unknown_for_arbitrary_string(self):
        assert _classify_marker("foo") == "unknown"
        assert _classify_marker("??") == "unknown"

    def test_uppercase_multi_char_is_unknown(self):
        # Uppercase multi-char roman (II, III) are not list items in our scope
        assert _classify_marker("II") == "unknown"


# ===========================================================================
# _scan_items
# ===========================================================================


class TestScanItems:
    # --- Basic detection ---

    def test_empty_text_returns_empty(self):
        assert _scan_items("") == []

    def test_plain_prose_returns_empty(self):
        assert _scan_items("The MCP shall ensure compliance at all times.") == []

    def test_single_decimal_item(self):
        items = _scan_items("1. The MCP shall maintain records.")
        assert len(items) == 1
        assert items[0].marker == "1"
        assert items[0].marker_type == "decimal"
        assert items[0].text == "The MCP shall maintain records."
        assert items[0].marker_display == "1."

    def test_multiple_decimal_items(self):
        items = _scan_items("1. First\n2. Second\n3. Third")
        assert len(items) == 3
        assert [it.marker for it in items] == ["1", "2", "3"]

    def test_lowercase_alpha_items(self):
        items = _scan_items("a. Alpha\nb. Beta\nc. Gamma")
        assert len(items) == 3
        assert all(it.marker_type == "lower_alpha" for it in items)

    def test_single_i_classified_as_lower_roman(self):
        items = _scan_items("i. First roman")
        assert len(items) == 1
        assert items[0].marker_type == "lower_roman"

    def test_multi_char_roman_items(self):
        items = _scan_items("i. First\nii. Second\niii. Third")
        assert all(it.marker_type == "lower_roman" for it in items)

    def test_uppercase_alpha_items(self):
        items = _scan_items("A. Apple\nB. Banana")
        assert len(items) == 2
        assert all(it.marker_type == "upper_alpha" for it in items)

    # --- Delimiter variants ---

    def test_parenthetical_form_detected(self):
        items = _scan_items("(1) First\n(2) Second")
        assert len(items) == 2
        assert items[0].marker_display == "(1)"
        assert items[0].marker == "1"

    def test_half_paren_form_detected(self):
        items = _scan_items("1) First\n2) Second")
        assert len(items) == 2
        assert items[0].marker_display == "1)"

    def test_parenthetical_alpha(self):
        items = _scan_items("(a) Item\n(b) Item")
        assert len(items) == 2
        assert items[0].marker_display == "(a)"
        assert items[0].marker_type == "lower_alpha"

    def test_parenthetical_roman(self):
        items = _scan_items("(i) Item\n(ii) Item")
        assert len(items) == 2
        assert items[0].marker_type == "lower_roman"

    # --- Indentation ---

    def test_indented_items_record_indent(self):
        items = _scan_items("   a. Sub-item")
        assert len(items) == 1
        assert items[0].indent == 3

    def test_tab_indented_items(self):
        items = _scan_items("\ta. Tab-indented sub-item")
        assert len(items) == 1
        assert items[0].indent == 4  # tab expanded to 4 spaces

    # --- Continuation lines ---

    def test_continuation_line_appended(self):
        text = "1. First item that wraps\n   onto the next line\n2. Second item"
        items = _scan_items(text)
        assert len(items) == 2
        assert "onto the next line" in items[0].text

    def test_blank_line_does_not_continue(self):
        text = "1. First item\n\n2. Second item"
        items = _scan_items(text)
        assert len(items) == 2
        # Blank line is not appended as continuation
        assert items[0].text == "First item"

    # --- Mixed indent (nested) ---

    def test_mixed_indent_correctly_scanned(self):
        text = "1. Top\n   a. Sub\n      i. Deep"
        items = _scan_items(text)
        assert len(items) == 3
        assert items[0].indent == 0
        assert items[1].indent == 3
        assert items[2].indent == 6


# ===========================================================================
# _assign_levels
# ===========================================================================


class TestAssignLevels:
    def test_empty_returns_empty(self):
        assert _assign_levels([]) == []

    def test_indent_based_two_levels(self):
        text = "1. Top\n   a. Sub"
        items = _scan_items(text)
        levelled = _assign_levels(items)
        levels = [lv for _, lv in levelled]
        assert levels == [1, 2]

    def test_indent_based_three_levels(self):
        text = "1. Top\n   a. Sub\n      i. Deep"
        items = _scan_items(text)
        levels = [lv for _, lv in _assign_levels(items)]
        assert levels == [1, 2, 3]

    def test_indent_based_four_levels(self):
        text = "1. Top\n   a. Sub\n      i. Deep\n         A. Deepest"
        items = _scan_items(text)
        levels = [lv for _, lv in _assign_levels(items)]
        assert levels == [1, 2, 3, 4]

    def test_type_hierarchy_fallback_no_indent(self):
        # All items at indent=0 → use type-hierarchy fallback
        text = "1. Decimal\na. Alpha\ni. Roman\nA. Upper"
        items = _scan_items(text)
        levels = [lv for _, lv in _assign_levels(items)]
        assert levels == [1, 2, 3, 4]

    def test_type_stack_pops_on_return_to_parent(self):
        # Sequence: 1(decimal) → a(alpha, level 2) → i(roman, level 3)
        #           → a(alpha again, level 2) → 2(decimal again, level 1)
        text = "1. First\na. Sub\ni. Deep\na. Back to sub\n2. Back to top"
        items = _scan_items(text)
        levels = [lv for _, lv in _assign_levels(items)]
        assert levels[0] == 1   # "1." → decimal → level 1
        assert levels[1] == 2   # "a." → new type → level 2
        assert levels[2] == 3   # "i." → new type → level 3
        assert levels[3] == 2   # "a." → lower_alpha already in stack → pop to 2
        assert levels[4] == 1   # "2." → decimal already in stack → pop to 1

    def test_single_item_level_one(self):
        items = _scan_items("1. Only item")
        levels = [lv for _, lv in _assign_levels(items)]
        assert levels == [1]


# ===========================================================================
# parse_nested_list (end-to-end)
# ===========================================================================


class TestParseNestedList:
    def test_empty_text_returns_empty_list(self):
        assert parse_nested_list("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert parse_nested_list("   \n\n   ") == []

    def test_no_list_markers_returns_empty_list(self):
        text = "The MCP shall comply with all applicable regulations."
        assert parse_nested_list(text) == []

    def test_single_top_level_item(self):
        tree = parse_nested_list("1. MCPs must maintain records.")
        assert len(tree) == 1
        assert tree[0]["marker"] == "1."
        assert tree[0]["text"] == "MCPs must maintain records."
        assert tree[0]["level"] == 1
        assert tree[0]["path"] == "1"
        assert tree[0]["children"] == []

    def test_two_top_level_items(self):
        tree = parse_nested_list("1. First\n2. Second")
        assert len(tree) == 2
        assert tree[1]["path"] == "2"

    def test_two_level_list_with_indentation(self):
        text = textwrap.dedent("""\
            1. MCPs must maintain all of the following:
               a. Personnel files
               b. Credentialing records
            2. MCPs must submit annual reports
        """)
        tree = parse_nested_list(text)
        assert len(tree) == 2
        assert tree[0]["text"] == "MCPs must maintain all of the following:"
        assert len(tree[0]["children"]) == 2
        assert tree[0]["children"][0]["text"] == "Personnel files"
        assert tree[0]["children"][0]["path"] == "1.a"
        assert tree[0]["children"][1]["path"] == "1.b"

    def test_three_level_list(self):
        text = textwrap.dedent("""\
            1. First top-level requirement:
               a. Sub-requirement A including:
                  i. Detail one
                  ii. Detail two
               b. Sub-requirement B
        """)
        tree = parse_nested_list(text)
        assert len(tree) == 1
        node1 = tree[0]
        assert len(node1["children"]) == 2
        sub_a = node1["children"][0]
        assert len(sub_a["children"]) == 2
        assert sub_a["children"][0]["path"] == "1.a.i"
        assert sub_a["children"][1]["path"] == "1.a.ii"

    def test_four_level_list(self):
        text = textwrap.dedent("""\
            1. Level one:
               a. Level two:
                  i. Level three:
                     A. Level four leaf
        """)
        tree = parse_nested_list(text)
        leaf = tree[0]["children"][0]["children"][0]["children"][0]
        assert leaf["level"] == 4
        assert leaf["path"] == "1.a.i.a"   # uppercase A stored lowercase in path
        assert leaf["marker"] == "A."
        assert leaf["text"] == "Level four leaf"

    def test_flat_text_type_hierarchy_fallback(self):
        # No indentation — relies on the type-stack fallback
        text = "1. Decimal level\na. Alpha level\ni. Roman level"
        tree = parse_nested_list(text)
        assert len(tree) == 1
        assert tree[0]["level"] == 1
        assert len(tree[0]["children"]) == 1
        assert tree[0]["children"][0]["level"] == 2
        assert len(tree[0]["children"][0]["children"]) == 1
        assert tree[0]["children"][0]["children"][0]["level"] == 3

    def test_parenthetical_markers(self):
        text = textwrap.dedent("""\
            (1) Top-level requirement
                (a) Sub-requirement alpha
                (b) Sub-requirement beta
        """)
        tree = parse_nested_list(text)
        assert len(tree) == 1
        assert tree[0]["marker"] == "(1)"
        assert len(tree[0]["children"]) == 2
        assert tree[0]["children"][0]["marker"] == "(a)"

    def test_half_paren_markers(self):
        text = "1) First\n   a) Sub-first\n2) Second"
        tree = parse_nested_list(text)
        assert len(tree) == 2
        assert tree[0]["marker"] == "1)"
        assert tree[0]["children"][0]["marker"] == "a)"

    def test_siblings_return_to_correct_parent(self):
        text = textwrap.dedent("""\
            1. First parent
               a. Child A
               b. Child B
            2. Second parent
               a. Child C
        """)
        tree = parse_nested_list(text)
        assert len(tree) == 2
        assert len(tree[0]["children"]) == 2
        assert len(tree[1]["children"]) == 1
        assert tree[1]["children"][0]["path"] == "2.a"

    def test_multi_line_item_text_joined(self):
        text = "1. First item that spans\n   multiple continuation lines\n2. Second"
        tree = parse_nested_list(text)
        assert "multiple continuation lines" in tree[0]["text"]

    def test_path_breadcrumbs_correct(self):
        text = "1. Top\n   a. Sub\n      i. Deep\n      ii. Also deep\n   b. Other sub"
        tree = parse_nested_list(text)
        assert tree[0]["path"] == "1"
        assert tree[0]["children"][0]["path"] == "1.a"
        assert tree[0]["children"][0]["children"][0]["path"] == "1.a.i"
        assert tree[0]["children"][0]["children"][1]["path"] == "1.a.ii"
        assert tree[0]["children"][1]["path"] == "1.b"

    def test_roman_numeral_sequence(self):
        text = textwrap.dedent("""\
            a. Alpha level
               i. Roman one
               ii. Roman two
               iii. Roman three
               iv. Roman four
        """)
        tree = parse_nested_list(text)
        assert len(tree) == 1
        assert len(tree[0]["children"]) == 4
        assert tree[0]["children"][3]["marker"] == "iv."
        assert tree[0]["children"][3]["path"] == "a.iv"

    def test_realistic_dhcs_style_list(self):
        """Simulate a real DHCS regulatory enumeration (4 levels)."""
        text = textwrap.dedent("""\
            1. The MCP shall ensure all of the following:
               a. Provider credentialing requirements are met, including:
                  i. Primary source verification of:
                     A. Medical license
                     B. Board certification
                  ii. Background check completion
               b. Re-credentialing occurs every two years
            2. The MCP shall maintain documentation of credentialing activities
        """)
        tree = parse_nested_list(text)

        # Two top-level items
        assert len(tree) == 2

        # First top-level: two children (a and b)
        node1 = tree[0]
        assert len(node1["children"]) == 2

        # "a." has two children (i and ii)
        sub_a = node1["children"][0]
        assert sub_a["path"] == "1.a"
        assert len(sub_a["children"]) == 2

        # "i." (Primary source verification) has two children (A and B)
        sub_a_i = sub_a["children"][0]
        assert sub_a_i["path"] == "1.a.i"
        assert len(sub_a_i["children"]) == 2
        assert sub_a_i["children"][0]["path"] == "1.a.i.a"  # A stored as lowercase
        assert sub_a_i["children"][0]["marker"] == "A."
        assert sub_a_i["children"][1]["path"] == "1.a.i.b"
        assert sub_a_i["children"][1]["marker"] == "B."


# ===========================================================================
# ParseNestedListTool (smolagents wrapper)
# ===========================================================================


class TestParseNestedListTool:
    def setup_method(self):
        self.tool = ParseNestedListTool()

    def test_tool_name_and_output_type(self):
        assert self.tool.name == "parse_nested_list"
        assert self.tool.output_type == "string"

    def test_empty_text_returns_empty_json_array(self):
        assert self.tool(text="") == "[]"
        assert self.tool(text="   \n") == "[]"

    def test_no_list_returns_empty_json_array(self):
        result = self.tool(text="Plain regulatory prose with no enumeration.")
        assert result == "[]"

    def test_returns_valid_json_string(self):
        result = self.tool(text="1. Item one\n   a. Sub-item")
        data = json.loads(result)
        assert isinstance(data, list)

    def test_nested_structure_in_output(self):
        result = self.tool(text="1. Top\n   a. Child\n      i. Grandchild")
        data = json.loads(result)
        assert len(data) == 1
        assert len(data[0]["children"]) == 1
        assert len(data[0]["children"][0]["children"]) == 1

    def test_output_has_all_required_keys(self):
        result = self.tool(text="1. Only item\n   a. Sub")
        data = json.loads(result)
        for node in [data[0], data[0]["children"][0]]:
            assert "marker" in node
            assert "text" in node
            assert "level" in node
            assert "path" in node
            assert "children" in node

    def test_tool_does_not_raise_on_edge_cases(self):
        # Non-list text that contains digits and letters
        text = "Section 4.2 requires MCPs to comply with 42 CFR § 438.210."
        result = self.tool(text=text)
        # Should return a string (either "[]" or valid JSON)
        json.loads(result)  # must not raise


# ===========================================================================
# _build_tree unit tests
# ===========================================================================


class TestBuildTree:
    """Lower-level tests for _build_tree to verify tree assembly logic."""

    def _make_item(
        self,
        marker: str,
        marker_type: str,
        text: str,
        indent: int = 0,
    ) -> _RawItem:
        display = f"{marker}."
        return _RawItem(
            indent=indent,
            marker=marker,
            marker_type=marker_type,
            marker_display=display,
            text=text,
        )

    def test_single_root(self):
        items = [self._make_item("1", "decimal", "Only item")]
        roots = _build_tree(items)
        assert len(roots) == 1
        assert roots[0].marker == "1."
        assert roots[0].level == 1
        assert roots[0].children == []

    def test_parent_child_relationship(self):
        items = [
            self._make_item("1", "decimal", "Parent", indent=0),
            self._make_item("a", "lower_alpha", "Child", indent=3),
        ]
        roots = _build_tree(items)
        assert len(roots) == 1
        assert len(roots[0].children) == 1
        assert roots[0].children[0].text == "Child"

    def test_multiple_siblings(self):
        items = [
            self._make_item("1", "decimal", "Parent", indent=0),
            self._make_item("a", "lower_alpha", "Sibling A", indent=3),
            self._make_item("b", "lower_alpha", "Sibling B", indent=3),
        ]
        roots = _build_tree(items)
        assert len(roots[0].children) == 2

    def test_return_to_root_level(self):
        items = [
            self._make_item("1", "decimal", "First root", indent=0),
            self._make_item("a", "lower_alpha", "Child", indent=3),
            self._make_item("2", "decimal", "Second root", indent=0),
        ]
        roots = _build_tree(items)
        assert len(roots) == 2
        assert roots[0].children[0].text == "Child"
        assert roots[1].children == []
