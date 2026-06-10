from footyml.utils.money import parse_market_value


def test_millions() -> None:
    assert parse_market_value("€25.5m") == 25_500_000


def test_thousands() -> None:
    assert parse_market_value("€750k") == 750_000


def test_billions() -> None:
    assert parse_market_value("€1.2bn") == 1_200_000_000


def test_dash_returns_none() -> None:
    assert parse_market_value("-") is None


def test_none_input() -> None:
    assert parse_market_value(None) is None


def test_empty_string() -> None:
    assert parse_market_value("") is None


def test_integer_euros() -> None:
    result = parse_market_value("€500000")
    assert result == 500_000


def test_no_currency_symbol() -> None:
    assert parse_market_value("10m") == 10_000_000
