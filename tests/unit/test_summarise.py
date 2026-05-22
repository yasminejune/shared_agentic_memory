import pathlib

import pytest

from evaluation.summarise_tasks.summarise import count_instantiations, save_to_csv

# --- count_instantiations ---


@pytest.mark.unit
def test_count_single_entry() -> None:
    data = [{"intent_template_id": "A"}]
    assert count_instantiations(data) == {"A": 1}


@pytest.mark.unit
def test_count_multiple_distinct_ids() -> None:
    data = [{"intent_template_id": "A"}, {"intent_template_id": "B"}]
    assert count_instantiations(data) == {"A": 1, "B": 1}


@pytest.mark.unit
def test_count_repeated_ids() -> None:
    data = [
        {"intent_template_id": "A"},
        {"intent_template_id": "A"},
        {"intent_template_id": "B"},
    ]
    assert count_instantiations(data) == {"A": 2, "B": 1}


@pytest.mark.unit
def test_count_empty_data() -> None:
    assert count_instantiations([]) == {}


# --- save_to_csv ---


@pytest.mark.unit
def test_save_to_csv_creates_file(tmp_path: pathlib.Path) -> None:
    counts = {"A": 2, "B": 1}
    out = tmp_path / "out.csv"
    save_to_csv(counts, str(out))
    assert out.exists()


@pytest.mark.unit
def test_save_to_csv_correct_columns(tmp_path: pathlib.Path) -> None:
    import pandas as pd

    counts = {"A": 2, "B": 1}
    out = tmp_path / "out.csv"
    save_to_csv(counts, str(out))
    df = pd.read_csv(out)
    assert list(df.columns) == ["intent_template_id", "num_instantiations"]


@pytest.mark.unit
def test_save_to_csv_correct_values(tmp_path: pathlib.Path) -> None:
    import pandas as pd

    counts = {"A": 3}
    out = tmp_path / "out.csv"
    save_to_csv(counts, str(out))
    df = pd.read_csv(out)
    assert df.loc[0, "intent_template_id"] == "A"
    assert df.loc[0, "num_instantiations"] == 3
