from data import FEATURES, TARGET, make_data


def test_columns_and_size():
    df = make_data(500)
    assert list(df.columns) == FEATURES + [TARGET]
    assert len(df) == 500


def test_same_seed_same_data():
    assert make_data(200, seed=1).equals(make_data(200, seed=1))


def test_failure_rate_is_realistic():
    assert 0.02 < make_data()[TARGET].mean() < 0.15


def test_no_missing_values():
    assert not make_data().isna().any().any()
