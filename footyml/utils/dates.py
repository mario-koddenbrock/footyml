import re


def season_id_to_year(season_id: str) -> int:
    """Parse Transfermarkt season ID to start year.

    Handles "2024", "24/25", "2024/25", "2024-25".
    """
    season_id = season_id.strip()
    if re.match(r"^\d{4}$", season_id):
        return int(season_id)
    m = re.match(r"^(\d{4})[/-]\d{2}$", season_id)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d{2})[/-]\d{2}$", season_id)
    if m:
        year = int(m.group(1))
        return 2000 + year if year < 50 else 1900 + year
    return int(season_id[:4])


def year_to_season_id(year: int) -> str:
    """Convert start year to Transfermarkt season ID format: 2024 -> '2024'."""
    return str(year)
