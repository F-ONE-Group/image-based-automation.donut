from pyhere import here
import sys

sys.path.append(str(here().resolve()))

from text_extractor import _is_a_list, reformat_text

# Test functions
def test_is_a_list():
    assert _is_a_list("[example]") == True
    assert _is_a_list("{example}") == False
    assert _is_a_list("example") == False
    assert _is_a_list("[example") == False
    assert _is_a_list("]") == False


def test_reformat_text():
    assert reformat_text("[1, 2, 3]") == [1, 2, 3]
    assert reformat_text('["a", "b", "c"]') == ["a", "b", "c"]
    assert reformat_text("example") == ["example"]
    assert reformat_text("{}") == ["{}"]
    assert reformat_text("") == [""]
