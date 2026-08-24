"""Instantiation counts per WebArena intent template, and their CSV form."""

import pathlib

import pandas as pd
import pytest

from evaluation.summarise_tasks.summarise import count_instantiations, save_to_csv

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("template_ids", "expected"),
    [
        ([], {}),
        (["A"], {"A": 1}),
        (["A", "B"], {"A": 1, "B": 1}),
        (["A", "A", "B"], {"A": 2, "B": 1}),
    ],
)
def test_count_instantiations(template_ids: list[str], expected: dict[str, int]) -> None:
    data = [{"intent_template_id": tid} for tid in template_ids]
    assert count_instantiations(data) == expected


def test_save_to_csv_round_trip(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out.csv"

    save_to_csv({"A": 3, "B": 1}, str(out))

    df = pd.read_csv(out)
    assert list(df.columns) == ["intent_template_id", "num_instantiations"]
    assert dict(zip(df["intent_template_id"], df["num_instantiations"], strict=True)) == {
        "A": 3,
        "B": 1,
    }
