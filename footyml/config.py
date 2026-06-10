import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
DUCKDB_PATH = DATA_DIR / "footyml.duckdb"

TRANSFERMARKT_BASE_URL = os.environ.get("TRANSFERMARKT_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 2.0
RATE_LIMIT_RPS = 2

LEAGUE_IDS: dict[str, str] = {
    "bundesliga": "L1",
    "premier_league": "GB1",
    "la_liga": "ES1",
    "serie_a": "IT1",
    "ligue_1": "FR1",
    "bundesliga2": "L2",
}

# TabPFN token — set TABPFN_API_KEY so TabPFNClassifier authenticates at import
_tabpfn_token = os.environ.get("TABPFN_TOKEN", "")
if _tabpfn_token:
    os.environ.setdefault("TABPFN_API_KEY", _tabpfn_token)

# Football-data.org API (https://www.football-data.org/)
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
FOOTBALL_DATA_RATE_LIMIT_RPS = 10 / 60  # free tier: 10 req/min

# Tournament competition IDs (football-data.org codes)
TOURNAMENT_IDS: dict[str, str] = {
    "world_cup": "WC",
    "euro": "EC",
    "copa_america": "CLI",
    "champions_league": "CL",
    "europa_league": "EL",
}

# Venue type constants
VENUE_NEUTRAL = "neutral"
VENUE_HOME_A = "home_advantage_a"
VENUE_HOME_B = "home_advantage_b"

# Venue type encoded as float for ML features
VENUE_TYPE_ENCODING: dict[str, float] = {
    VENUE_NEUTRAL: 0.0,
    VENUE_HOME_A: 1.0,
    VENUE_HOME_B: -1.0,
}

ELO_K_FACTOR = 32.0
ELO_INITIAL = 1500.0
HOME_ADVANTAGE_ELO = 100.0

# Previous international tournaments to ingest for national team history
# (football-data.org competition code → list of seasons)
INTERNATIONAL_HISTORY: dict[str, list[int]] = {
    "WC": [2018, 2022],
    "EC": [2020, 2024],
}

# Transfermarkt club IDs for national teams.
# Key = team name as returned by football-data.org
# Value = Transfermarkt club ID for that national team
# Verify / extend at: https://transfermarkt-api.fly.dev/clubs/{id}/profile
NATIONAL_TEAM_TM_IDS: dict[str, str] = {
    "Germany": "3262",
    "Brazil": "3439",
    "Argentina": "3437",
    "France": "3377",
    "England": "3166",
    "Spain": "3375",
    "Italy": "3376",
    "Portugal": "3930",
    "Netherlands": "3379",
    "Belgium": "3382",
    "Croatia": "3385",
    "Switzerland": "3378",
    "Poland": "3388",
    "Denmark": "3386",
    "Sweden": "3384",
    "Austria": "3380",
    "Czech Republic": "3381",
    "Serbia": "3399",
    "Turkey": "3389",
    "Wales": "3383",
    "Scotland": "3390",
    "Ukraine": "3391",
    "Hungary": "3387",
    "Slovakia": "3396",
    "Slovenia": "3397",
    "Romania": "3392",
    "Albania": "3394",
    "Georgia": "3402",
    "Uruguay": "3440",
    "Colombia": "3442",
    "Ecuador": "3444",
    "Chile": "3441",
    "Peru": "3443",
    "Mexico": "3448",
    "United States": "3438",
    "USA": "3438",
    "Canada": "3456",
    "Panama": "3458",
    "Honduras": "3460",
    "Costa Rica": "3459",
    "Morocco": "5765",
    "Senegal": "5768",
    "Nigeria": "5769",
    "Ivory Coast": "5779",
    "Egypt": "5764",
    "Cameroon": "5770",
    "Algeria": "5763",
    "South Africa": "5785",
    "Tunisia": "5776",
    "Japan": "3469",
    "South Korea": "3445",
    "Iran": "3460",
    "Saudi Arabia": "3463",
    "Australia": "3465",
    "Qatar": "3474",
    "United Arab Emirates": "3466",
}
