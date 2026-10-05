# Football micro-event prediction backend

[![Tests](https://github.com/Pascal-Ahmadu/football-prediction-backend/actions/workflows/ci.yml/badge.svg)](https://github.com/Pascal-Ahmadu/football-prediction-backend/actions/workflows/ci.yml)

Forecasts five football markets (total corners, total cards, total fouls, player
shots on target and player fouls committed) for 17 European leagues, and scores
its own forecasts against results and bookmaker prices afterwards.

Built to the BRD/SRD specification BRD-SRD-FMEP-001. Python 3.12, FastAPI,
PostgreSQL 16, SQLAlchemy 2 (async), Alembic, LightGBM.

## Results

Each market was developed on the 2024/25 season and then evaluated **once** on
2025/26, which was held back during development. Nothing was tuned afterwards.

| Market | Line | Sample | Beats baseline by | Hit rate | Calibration error |
| --- | --- | --- | --- | --- | --- |
| Total fouls | 24.5 | 5,602 matches | +0.0881 | 67.7% | 0.0174 |
| Total cards | 4.5 | 5,624 matches | +0.0240 | 60.6% | 0.0116 |
| Total corners | 9.5 | 5,623 matches | +0.0067 | 55.1% | 0.0161 |
| Player fouls committed | 0.5 | 174,006 appearances | +0.0217 | n/a | 0.0074 |
| Player shots on target | 0.5 | 174,006 appearances | +0.0147 | n/a | 0.0048 |

"Beats baseline by" is the reduction in log loss against that market's own
baseline: for match markets, always predicting the side more common in that
league; for player markets, the player's own rate per 90 minutes times his usual
minutes, shrunk toward his position's average. Calibration error is the mean gap
between stated probability and observed frequency; the specification asks for
under 0.02 and all five markets meet it.

Accuracy is not profit. On the market prices collected so far, bookmaker margins
on these markets average 8.7%, and whether any edge survives that is still being
measured. See [Measuring value](#measuring-value).

## How it works

```
providers ──► core (matches, teams, referees, players, statistics, odds)
                │
                ├─► features (point-in-time snapshots, versioned)
                │
                ├─► models (LightGBM Poisson mean + negative binomial spread)
                │
                └─► API (fixtures, forecasts, distributions, picks, performance)
```

**Point-in-time correctness.** A match's feature snapshot is written *before* that
match is folded into any average, so no forecast can see its own result. A test
recomputes sampled snapshots from strictly earlier matches and fails if a single
number disagrees. It was verified to fail when the decay setting was altered.

**Features.** Exponentially weighted team form (half-life 6 matches), ridge-fitted
attack and defence ratings per competition, referee profiles (half-life 10
matches), and per-player rates with involvement over a team's last five matches.
Feature sets are versioned, so a change can be measured against the live one
before replacing it.

**Models.** LightGBM with a Poisson objective predicts the mean; a negative
binomial turns that into a distribution, because these counts vary more than a
Poisson allows. Cards uses two variants, with and without referee history,
because referees are often named only days before kickoff.

**Walk-forward evaluation.** Models are retrained repeatedly through a season and
only ever predict matches later than everything they have seen, which mirrors
production.

## Data

37,359 matches (34,833 with statistics) and 1,364,653 player appearances across
17 leagues, seasons 2020/21 to 2026/27.

| Source | Provides | Notes |
| --- | --- | --- |
| [apifootball.com](https://apifootball.com) | Matches, statistics, players, referees | Paid; the primary source |
| [football-data.co.uk](https://www.football-data.co.uk) | Results, statistics, fixtures | Free, no key; automatic fallback for 16 of the 17 leagues, no player data |
| [The Odds API](https://the-odds-api.com) | Corners and cards prices | Free tier: 500 credits a month, so fixtures are sampled |

The pipeline detects an unusable primary feed (a lapsed subscription answers
politely with nothing), switches to the free source, and records a notification
rather than reporting success.

## Running it

Requires Python 3.12 and PostgreSQL 16, or Docker.

```bash
cp .env.example .env            # fill in keys; never commit .env
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

With Docker:

```bash
docker compose up -d db
docker compose run --rm api alembic upgrade head
docker compose up -d api
```

The weekly work is one command, and the order within it matters: features must
follow results, forecasts must follow features:

```bash
python -m app.pipeline.weekly                               # full refresh
python -m app.pipeline.weekly --days-back 3 --days-ahead 4  # before a matchweek
```

`scripts/crontab.example` installs the schedule on Linux; `scripts/run_weekly.ps1`
and the Windows Task Scheduler do the same on a laptop. `scripts/backup_db.sh`
(or `.ps1`) keeps seven daily database dumps plus one per month.

## API

Every endpoint under `/api/v1` requires an `X-API-Key` header once `API_KEYS` is
set, and is rate limited per caller. `/health` stays open for monitoring.

| Endpoint | Returns |
| --- | --- |
| `GET /health` | Service and database status |
| `GET /api/v1/fixtures` | Upcoming fixtures |
| `GET /api/v1/fixtures/{id}/predictions` | Every market's forecast for one fixture, with fair odds |
| `GET /api/v1/fixtures/{id}/distributions` | The probability of every exact count |
| `GET /api/v1/picks` | Ranked picks for any market |
| `GET /api/v1/accumulator` | The best combination available, with its true chance |
| `GET /api/v1/performance` | Live accuracy from settled forecasts |
| `GET /api/v1/notifications` | What the platform wants the operator to know |

Picks are ranked by how far a forecast departs from its **base rate**: the league's
own rate for match markets, the player's own for player markets, rather than by
distance from 50/50. That choice matters: on 2024/25 corners, ranking by distance from
50/50 added nothing measurable, while ranking by distance from the league rate
added 8.4 points of hit rate.

## Measuring value

Being accurate is not the same as beating a price. `models.prediction_outcomes`
stores, for every played fixture, the last forecast made before kickoff, the
actual count, and the best price available at the time. `GET /api/v1/performance`
reports hit rate, log loss against the observed rate and, where prices were
collected, profit and return for a flat stake on each side the model prices as
value.

3,180 forecasts have been settled so far. Two findings already recorded:

- **Margins are the hurdle.** 8.7% on corners and cards, roughly double the main
  markets, so an edge must be large to survive.
- **Accumulator correlation is not an edge.** Cards and fouls in the same match
  land together 1.24× more often than multiplying their base rates suggests.
  But conditional on the models' own forecasts the figure is 0.94× [0.90, 0.98],
  slightly *worse* than independent. The models had already extracted it.

## Tests

```bash
pytest                # 134 tests
pytest -m "not db"    # the 114 that need no database; this is what CI runs
```

Every push runs the second set through GitHub Actions. The 20 database tests
need six seasons of real data, so they run locally.

They cover provider quirks that fail silently (card statistics omitted from
otherwise complete payloads, possession reported as 0/0, duplicate fixture ids),
the point-in-time rule, the probability maths, team-name matching across
providers, API authentication and rate limits, and the pick-selection logic.

## Known limitations

- **Profitability is unproven.** Accurate and well calibrated, but not yet shown
  to beat a bookmaker's price over a meaningful sample.
- **Every test season is spent.** All five markets used 2025/26 as their one-off
  evaluation, so future changes can only be judged on 2024/25 plus live results.
- **Player markets depend on the paid source.** The free fallback carries no
  player data, so those two markets go stale without a subscription.
- **Austria** is not covered by the free source (16 of 17 leagues).
- **Fouls differ between providers.** Their counts agree with the primary
  source's on 88% of matches, so sources are kept separate rather than mixed.
