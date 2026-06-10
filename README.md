# FootyML

Football match prediction platform — automated data collection, feature engineering, and match outcome prediction using TabPFN.

## Overview

FootyML predicts **Home Win / Draw / Away Win** for football matches using rich pre-match features:

- Team form (rolling 3/5/10 matches)
- Home/away specific form
- Head-to-head history
- Squad market values (from Transfermarkt)
- League standing & Elo ratings
- Transfer window activity

Model: **TabPFN v3** (tabular prior-fitted network, sklearn-compatible)

## Setup

```bash
# Install
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev

# Add API keys to .env (never commit this file)
echo "FOOTBALL_DATA_API_KEY=your_key_here" >> .env  # free at football-data.org
echo "TABPFN_TOKEN=your_token_here" >> .env          # from tabpfn.com
```

## Quick Start — League Mode

```bash
# 1. Ingest multiple leagues (2015–2025 recommended for max training data)
python scripts/ingest.py --league bundesliga --seasons 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
python scripts/ingest.py --league premier_league --seasons 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
python scripts/ingest.py --league la_liga --seasons 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
python scripts/ingest.py --league serie_a --seasons 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
python scripts/ingest.py --league ligue_1 --seasons 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025

# 2. Build features (all leagues at once)
python -c "from footyml.features import FeaturePipeline; FeaturePipeline().build()"

# 3. Train on all leagues combined (best for generalization)
python scripts/train.py --league all --season 2025 --train-from 2015 --eval

# 4. Predict a match
python scripts/predict.py --home "Bayern Munich" --away "Borussia Dortmund"
```

## Quick Start — World Cup Mode

National team predictions require historical international match data and squad values.
The national teams use football-data.org IDs (`fd_XXX`), which are separate from club IDs.

```bash
# 1. Ingest past WC/Euro results → builds Elo + form for national teams
python scripts/ingest_international.py  # WC 2018/2022, Euro 2020/2024

# 2. Ingest national team squad market values from Transfermarkt
python scripts/ingest_national_teams.py --season 2025 --competition WC

# 3. Build features for WC matches
python -c "from footyml.features import FeaturePipeline; FeaturePipeline().build(competition_id='WC')"

# 4. Fetch WC 2026 fixtures and predict upcoming matches
python scripts/predict_tournament.py --competition WC --season 2026

# List all available competition codes
python scripts/predict_tournament.py --list

# Optional: launch the Streamlit dashboard
uv sync --extra dashboard
uv run streamlit run app/dashboard.py
```

## Python API

```python
from footyml.features import FeaturePipeline
from footyml.data import MatchDataset
from footyml.models import TabPFNMatchPredictor

# Build features for all leagues
df = FeaturePipeline().build()

# Load all leagues combined
dataset = MatchDataset.all_leagues(seasons=range(2015, 2026))
X, y = dataset.load()   # y: 0=away win, 1=draw, 2=home win

# Or a single league
dataset = MatchDataset(league="bundesliga", seasons=range(2015, 2026))

# Train
clf = TabPFNMatchPredictor()
clf.fit(X_train, y_train)
clf.predict_proba(X_test)  # shape (n, 3) — [P(away), P(draw), P(home)]
```

## Supported Leagues & Tournaments

| Key | League/Tournament |
|---|---|
| `bundesliga` | Bundesliga (Germany) |
| `premier_league` | Premier League (England) |
| `la_liga` | La Liga (Spain) |
| `serie_a` | Serie A (Italy) |
| `ligue_1` | Ligue 1 (France) |
| `WC` | FIFA World Cup |
| `EC` | UEFA European Championship |
| `CL` | UEFA Champions League |

## World Cup Mode — Key Design

For World Cup and other neutral-venue tournaments:
- **Separate ID namespace**: National teams use `fd_XXX` IDs (football-data.org); club teams use Transfermarkt integer IDs. These namespaces never mix.
- **Elo is built from international history**: Ingest WC 2018/2022, Euro 2020/2024 first so each national team has a meaningful Elo.
- **Squad market values from Transfermarkt**: `ingest_national_teams.py` maps team names → Transfermarkt club IDs and stores squad values under the `fd_` team ID.
- **Season fallback**: If exact-season squad data isn't available, the system falls back to the most recent season on record.
- **No home advantage** for WC matches (`venue_type=neutral`).
- **Group standings** reconstructed per group (no cross-group leakage).
- **Elo fallback prediction** used when feature data is absent.

## Project Structure

```
footyml/
├── footyml/
│   ├── ingestion/     # Transfermarkt API client + fetchers
│   ├── providers/     # football-data.org API provider
│   ├── features/      # Feature pipeline (8 builders + diff features)
│   ├── data/          # MatchDataset + DuckDB store
│   ├── models/        # TabPFNMatchPredictor + evaluation
│   ├── prediction/    # PredictionService + TournamentPredictor
│   ├── visualization/ # Plotly dashboards
│   └── utils/         # money parsing, date helpers
├── app/
│   └── dashboard.py   # Streamlit World Cup dashboard
├── scripts/
│   ├── ingest.py                # League data ingestion
│   ├── ingest_international.py  # WC/Euro history for national team Elo
│   ├── ingest_national_teams.py # National team squad market values
│   ├── train.py                 # Training CLI (--league all for multi-league)
│   ├── predict.py               # Single match prediction
│   ├── predict_upcoming.py      # League fixture predictions
│   └── predict_tournament.py    # World Cup / tournament predictions
└── tests/
```

## Development

```bash
uv run pytest          # run tests
uv run ruff check .    # linting
uv run black .         # formatting
uv run mypy footyml/   # type checking
```
