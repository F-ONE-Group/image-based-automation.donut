import json
from typing import List, Dict
from pydantic import BaseModel

class ReformattedLabel(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def extract_labels_from_str(labels: str) -> List[dict]:
    labels = json.loads(labels)
    return labels


def _get_reformatted_label(label: dict) -> ReformattedLabel:
    xmin = label['x'] / 100
    ymin = label['y'] / 100
    xmax = (label['x'] + label['width']) / 100
    ymax = (label['y'] + label['height']) / 100

    reformat_label: ReformattedLabel = ReformattedLabel(
        xmin = float(xmin),
        ymin = float(ymin),
        xmax = float(xmax),
        ymax = float(ymax)
    )

    return reformat_label


def reformat_labels(labels: List[Dict]) -> List[str]:
    reformatted_labels: List = []
    for _label in labels:
        # import pdb
        # pdb.set_trace()
        reformatted_label: ReformattedLabel = _get_reformatted_label(_label)
        reformatted_labels.append(reformatted_label.dict())

    return reformatted_labels
