import re


def parse_market_value(value_str: str | None) -> int | None:
    """Convert Transfermarkt value strings to integer euros.

    Examples:
        "€25.5m"  -> 25_500_000
        "€750k"   -> 750_000
        "€1.2bn"  -> 1_200_000_000
        "-"       -> None
    """
    if value_str is None:
        return None
    if isinstance(value_str, (int, float)):
        return int(value_str) if value_str else None
    if not value_str or value_str.strip() in ("-", "N/A", ""):
        return None

    multipliers = {"k": 1_000, "m": 1_000_000, "bn": 1_000_000_000}
    pattern = r"[€$]?\s*([\d.,]+)\s*(bn|m|k)?"
    match = re.match(pattern, value_str.strip(), re.IGNORECASE)
    if not match:
        return None

    number_str = match.group(1).replace(",", "")
    try:
        number = float(number_str)
    except ValueError:
        return None

    suffix = (match.group(2) or "").lower()
    multiplier = multipliers.get(suffix, 1)
    return int(number * multiplier)
