from pyhere import here
import sys

sys.path.append(str(here().resolve()))
from label_extractor import (
    extract_labels_from_str,
    _create_comma_separated_label,
    reformat_labels,
)


def test_extract_labels_from_str():
    input_str = '[{"label": "A", "value": 1}, {"label": "B", "value": 2}]'
    expected_output = [{"label": "A", "value": 1}, {"label": "B", "value": 2}]

    result = extract_labels_from_str(input_str)

    assert result == expected_output


# Test functions
def test_create_comma_separated_label():
    label = {
        "x": 25.26665253937983,
        "y": 92.15576411262083,
        "width": 73.17934590661864,
        "height": 3.081664098613243,
    }

    expected_output = (
        "25.26665253937983,92.15576411262083,73.17934590661864,3.081664098613243"
    )

    result = _create_comma_separated_label(label)
    assert result == expected_output


def test_reformat_labels():
    labels = [
        {"x": 1, "y": 2, "width": 3, "height": 4},
        {"x": 5, "y": 6, "width": 7, "height": 8},
    ]

    expected_output = ["1,2,3,4", "5,6,7,8"]

    result = reformat_labels(labels)
    assert result == expected_output
