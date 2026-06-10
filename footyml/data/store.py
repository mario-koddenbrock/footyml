from pathlib import Path

import duckdb
import polars as pl

from footyml.config import DUCKDB_PATH


class DataStore:
    """DuckDB-backed storage for FootyML tables."""

    def __init__(self, db_path: Path = DUCKDB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        schema_sql = (Path(__file__).parent / "schema.sql").read_text()
        for stmt in schema_sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Apply non-destructive schema migrations for new columns."""
        cols = {
            r[0]
            for r in self._conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='matches'"
            ).fetchall()
        }
        migrations = [
            ("venue_type", "VARCHAR DEFAULT 'home_advantage_a'"),
            ("group_id", "VARCHAR"),
            ("stage", "VARCHAR"),
            ("home_club_name", "VARCHAR"),
            ("away_club_name", "VARCHAR"),
        ]
        for col, typedef in migrations:
            if col not in cols:
                self._conn.execute(f"ALTER TABLE matches ADD COLUMN {col} {typedef}")

    def query(self, sql: str) -> pl.DataFrame:
        return self._conn.execute(sql).pl()

    def execute(self, sql: str, params: list | None = None) -> None:
        if params:
            self._conn.execute(sql, params)
        else:
            self._conn.execute(sql)

    # ------------------------------------------------------------------
    # Match upserts
    # ------------------------------------------------------------------

    def upsert_matches(self, df: pl.DataFrame) -> None:
        """Insert or replace matches from a Polars DataFrame.

        Fills missing optional columns (venue_type, group_id, stage) with defaults.
        """
        from footyml.config import VENUE_HOME_A

        if "venue_type" not in df.columns:
            df = df.with_columns(pl.lit(VENUE_HOME_A).alias("venue_type"))
        if "group_id" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.String).alias("group_id"))
        if "stage" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.String).alias("stage"))

        # Keep only columns that exist in the matches table
        table_cols = [
            "game_id", "competition_id", "season", "matchday", "date",
            "home_club_id", "away_club_id", "home_goals", "away_goals",
            "result", "venue_type", "group_id", "stage",
            "home_club_name", "away_club_name",
        ]
        available = [c for c in table_cols if c in df.columns]
        self._conn.execute(
            f"INSERT OR REPLACE INTO matches ({', '.join(available)}) "
            f"SELECT {', '.join(available)} FROM df"
        )

    def get_matches(
        self,
        competition_id: str | None = None,
        seasons: list[int] | None = None,
        completed_only: bool = True,
        group_id: str | None = None,
    ) -> pl.DataFrame:
        where_clauses: list[str] = []
        if competition_id:
            where_clauses.append(f"competition_id = '{competition_id}'")
        if seasons:
            season_list = ", ".join(str(s) for s in seasons)
            where_clauses.append(f"season IN ({season_list})")
        if completed_only:
            where_clauses.append("result IS NOT NULL")
        if group_id:
            where_clauses.append(f"group_id = '{group_id}'")
        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        return self.query(f"SELECT * FROM matches {where} ORDER BY date")

    # ------------------------------------------------------------------
    # Squad / market values
    # ------------------------------------------------------------------

    def upsert_squad(self, df: pl.DataFrame) -> None:
        self._conn.execute("INSERT OR REPLACE INTO player_market_values SELECT * FROM df")

    def get_squad(self, club_id: str, season_id: str) -> pl.DataFrame:
        result = self.query(
            f"SELECT * FROM player_market_values "
            f"WHERE club_id = '{club_id}' AND season_id = '{season_id}'"
        )
        if len(result) > 0:
            return result
        # Fall back to most recent available season for this club
        best = self.query(
            f"SELECT season_id FROM player_market_values "
            f"WHERE club_id = '{club_id}' "
            f"ORDER BY TRY_CAST(season_id AS INTEGER) DESC NULLS LAST LIMIT 1"
        )
        if len(best) == 0:
            return result
        fallback_season = best["season_id"][0]
        return self.query(
            f"SELECT * FROM player_market_values "
            f"WHERE club_id = '{club_id}' AND season_id = '{fallback_season}'"
        )

    # ------------------------------------------------------------------
    # Transfers
    # ------------------------------------------------------------------

    def upsert_transfers(self, df: pl.DataFrame) -> None:
        self._conn.execute("INSERT OR REPLACE INTO transfers SELECT * FROM df")

    def get_transfers(self, club_id: str, season_id: str) -> pl.DataFrame:
        return self.query(
            f"SELECT * FROM transfers WHERE club_id = '{club_id}' AND season_id = '{season_id}'"
        )

    # ------------------------------------------------------------------
    # Elo ratings
    # ------------------------------------------------------------------

    def upsert_elo(self, df: pl.DataFrame) -> None:
        self._conn.execute("INSERT OR REPLACE INTO elo_ratings SELECT * FROM df")

    def get_elo(self, club_id: str | None = None) -> pl.DataFrame:
        where = f"WHERE club_id = '{club_id}'" if club_id else ""
        return self.query(f"SELECT * FROM elo_ratings {where} ORDER BY date")

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def upsert_features(self, df: pl.DataFrame) -> None:
        # Drop any non-numeric non-id columns that shouldn't be in the feature table
        keep_cols = ["game_id"] + [
            c for c in df.columns
            if c != "game_id" and df[c].dtype not in (pl.Date, pl.Datetime, pl.String, pl.Utf8)
        ]
        df = df.select(keep_cols)

        _PL_TO_DUCK = {
            pl.Float32: "FLOAT",
            pl.Float64: "DOUBLE",
            pl.Int8: "TINYINT",
            pl.Int16: "SMALLINT",
            pl.Int32: "INTEGER",
            pl.Int64: "BIGINT",
            pl.UInt8: "UTINYINT",
            pl.UInt16: "USMALLINT",
            pl.UInt32: "UINTEGER",
            pl.UInt64: "UBIGINT",
            pl.Boolean: "BOOLEAN",
        }
        col_defs = ", ".join(
            f'"{c}" {_PL_TO_DUCK.get(df[c].dtype, "DOUBLE")}'
            for c in df.columns
            if c != "game_id"
        )
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS features_new (
                game_id VARCHAR PRIMARY KEY
                {", " + col_defs if col_defs else ""}
            )
        """)
        self._conn.execute("DROP TABLE IF EXISTS features")
        self._conn.execute("ALTER TABLE features_new RENAME TO features")
        self._conn.execute("INSERT OR REPLACE INTO features SELECT * FROM df")

    def get_features(self) -> pl.DataFrame:
        return self.query("SELECT * FROM features")

    def close(self) -> None:
        self._conn.close()
