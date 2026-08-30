from src.data.load import COLUMNS, load_train


def test_load_train_has_expected_columns():
    df = load_train("FD001")
    assert list(df.columns) == COLUMNS
    assert len(df.columns) == 26
    assert df["unit"].notna().all()
    assert df["cycle"].notna().all()
