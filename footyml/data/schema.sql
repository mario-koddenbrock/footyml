CREATE TABLE IF NOT EXISTS matches (
    game_id          VARCHAR PRIMARY KEY,
    competition_id   VARCHAR NOT NULL,
    season           INTEGER NOT NULL,
    matchday         INTEGER,
    date             DATE NOT NULL,
    home_club_id     VARCHAR NOT NULL,
    away_club_id     VARCHAR NOT NULL,
    home_goals       INTEGER,
    away_goals       INTEGER,
    result           INTEGER,  -- 0=away win, 1=draw, 2=home win
    venue_type       VARCHAR DEFAULT 'home_advantage_a',  -- neutral | home_advantage_a | home_advantage_b
    group_id         VARCHAR,  -- tournament group (e.g. "Group A") or NULL for leagues
    stage            VARCHAR,  -- GROUP_STAGE | ROUND_OF_16 | QUARTER_FINALS | etc.
    home_club_name   VARCHAR,
    away_club_name   VARCHAR
);

CREATE TABLE IF NOT EXISTS club_profiles (
    club_id                VARCHAR NOT NULL,
    club_name              VARCHAR,
    squad_size             INTEGER,
    average_age            FLOAT,
    foreigners_number      INTEGER,
    foreigners_percentage  FLOAT,
    total_market_value_eur BIGINT,
    PRIMARY KEY (club_id)
);

CREATE TABLE IF NOT EXISTS player_market_values (
    player_id        VARCHAR NOT NULL,
    club_id          VARCHAR NOT NULL,
    season_id        VARCHAR NOT NULL,
    player_name      VARCHAR,
    market_value_eur BIGINT,
    date_of_birth    VARCHAR,
    nationality      VARCHAR,
    PRIMARY KEY (player_id, season_id)
);

CREATE TABLE IF NOT EXISTS transfers (
    club_id          VARCHAR NOT NULL,
    season_id        VARCHAR NOT NULL,
    direction        VARCHAR NOT NULL,
    player_id        VARCHAR,
    player_name      VARCHAR,
    fee_eur          BIGINT,
    PRIMARY KEY (club_id, season_id, direction, player_id)
);

CREATE TABLE IF NOT EXISTS elo_ratings (
    club_id          VARCHAR NOT NULL,
    date             DATE NOT NULL,
    elo              DOUBLE NOT NULL,
    PRIMARY KEY (club_id, date)
);

CREATE TABLE IF NOT EXISTS features (
    game_id          VARCHAR PRIMARY KEY
);
