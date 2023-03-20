from pyhere import here
import sys

sys.path.append(str(here().resolve()))
from label_extractor import (
    extract_labels_from_str,
    reformat_labels,
)


def test_extract_labels_from_str():
    input_str = '[{"label": "A", "value": 1}, {"label": "B", "value": 2}]'
    expected_output = [{"label": "A", "value": 1}, {"label": "B", "value": 2}]

    result = extract_labels_from_str(input_str)

    assert result == expected_output
