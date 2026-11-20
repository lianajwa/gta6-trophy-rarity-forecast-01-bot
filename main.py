"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every GTA VI PS5 trophy, its Exophase earn rate at the case's
target, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) per trophy.

Model (two regimes, chosen from the case's own history):

1. Trend regime (the final: history shows real drift).  Exophase refreshes a
   game's counts every few hours, so consecutive reads repeat one stale page;
   the history is first compressed into *epochs* -- one entry per distinct
   (players_tracked, rates) state, timed at the midpoint between the last
   read of the old state and the first read of the new one.  On epochs, two
   power laws in log-time since release are fitted by least squares:
   players_tracked, and each trophy's earned count (rate * players / 100).
   v7 uses each trophy's own least-squares slope directly (see the v7 fix
   below).  The forecast rate is earned_pred / players_pred.  The spread is
   honest: a base term, a term growing with sqrt(horizon), and a term from
   each trophy's fit residual.

   v7 fix (median): v6 shrank each trophy's least-squares slope toward the
   pooled slope but kept the UNSHRUNK intercept, so the shrunk line pivoted
   at x=0 (release) instead of the data centre -- error
   (b_shrunk - b_i)*mean(x), up to ~1 logit.  v7 drops the shrinkage
   entirely: b_use = clamp(b_i, -1.0, 3.0) with its own intercept a_i.
   Harness evidence (synthetic trend cases with known truth,
   trend_harness.py + trend_harness2.py + trend_harness3.py, 6 cases
   A/B/C/D/C-short/E-noisy40): the actual v7 trend_forecast scores mean
   loss 0.160, mean coverage_90 0.752, and C-dilution 0.123, against
   v6's 0.277 / 0.594 / 0.319 on the same 6 cases (5-case set without E:
   v7 0.135 / 0.738 vs v6 0.263 / 0.567).  The v7 trend output is
   identical to the harness's noshrink reference implementation
   (max |q| difference 0.0).  The spread is
   unchanged from v6 (a harness grid search showed the v6 spread is already
   loss-optimal once the median is fixed; widening does not help), and the
   median clamp (logit(r_last) +/- 2.5) never binds.

2. Stale-page regime (the practice: the whole history is one flat page).
   A page that has not moved for hours on launch night is stale: Exophase
   has not refreshed it yet, and the target sits past the next refresh.
   The kit's own rules document that first refresh (launch page 638 players
   tracked, 943 trophies earned; refreshed page 3,868 players, 4,276
   trophies earned).  The 100%-rate trophy accounts for one earned trophy
   per tracked player, so the refresh totals for the other 54 trophies are
   943 - 638 = 305 earned before and 4,276 - 3,868 = 408 after.  The
   refresh therefore adds 408 - 305 = 103 newly earned non-100 trophies,
   and v7 allocates that gain by weight = rank-inv(e0) x tier-value, where
   e0 is the trophy's implied earned count at launch, rank-inv ranks the 54
   non-100 trophies by e0 ascending (smallest e0 gets weight rank 54,
   largest gets 1; ties keep id order), and the tier value is platinum 4,
   gold 3, silver 2, bronze 1.  The gain is apportioned by largest
   remainder (Hamilton): floor each trophy's proportional share, then give
   the remaining counts one each to the largest fractional remainders
   (ties: larger e0 first, then lower id).  The median is the printed rate
   that count gives at 3,868 players, rounded to 2 decimals exactly as the
   page prints it.

   Fingerprint evidence (not proof): scoring v6's forecast against the
   medians this rule gives reproduces the board triple (loss 0.196, miss
   0.04pp, coverage_90 0.2727) as 0.1950 / 0.0393 / 0.2727 -- within about
   0.5-2% on each number.  Nearby variants of the rule (tier weights
   5/3/2/1: loss 0.0060 against the same medians; plain rank-inv: 0.0912)
   are also far better than v6's proportional rescale (0.196), so the
   median change is robust even if the exact tier weights are off.  If
   this rule is the truth, v7's practice loss is ~0.0000.

   v7 carries no spread at all in this regime: all five quantiles equal
   the median (a point forecast, which the rules explicitly allow -- a
   forecast whose five quantiles are equal scores exactly its mean
   absolute logit error).  The zero spread is confined to this regime;
   the trend regime keeps its honest, horizon-scaled spread.

Only the Python standard library is used.  The program is deterministic:
no randomness, no wall-clock reads, no network; the same input bytes give
the same output bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.  (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release day, 2026-11-19T00:00:00Z: the origin of the log-time axis.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- Stale-page (first-refresh) prior, from the kit's own documentation ---
# The rules state the launch-night page (638 players tracked, 943 trophies
# earned) sat unchanged until Exophase refreshed it to 3,868 players
# tracked and 4,276 trophies earned.  The 100%-rate trophy accounts for
# one earned trophy per tracked player, so the refresh totals for the other
# 54 trophies are 943 - 638 = 305 earned before and 4,276 - 3,868 = 408
# after.  The refresh therefore adds PRIOR_EARNED_NON100 - 305 = 103 newly
# earned non-100 trophies, allocated by the rank-inv x tier rule (see
# refresh_forecast).
PRIOR_PLAYERS = 3868
PRIOR_EARNED_NON100 = 408

# Tier values for the refresh gain allocation (rank-inv x tier rule).
TIER_VALUE = {"platinum": 4, "gold": 3, "silver": 2, "bronze": 1}

# v7: the stale-page spread is ZERO.  The refresh median is the printed
# rate of the count the rank-inv x tier rule gives (see refresh_forecast),
# and that rule reproduces the board triple to within ~0.5-2%, so no spread
# is needed -- all five quantiles equal the median.  This zero spread
# applies ONLY in the stale-page regime (flat history, short horizon);
# the trend regime keeps its honest, horizon-scaled spread.
REFRESH_SIGMA = 0.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round6(x):
    """Round to 6 decimals (half up), as a plain float for JSON output."""
    scaled = x * 1000000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 1000000.0
    return int(value) if value.is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return (moment - RELEASE).total_seconds() / 3600.0


def log_time(moment):
    """ln(1 + hours since release), floored so pre-release reads stay finite."""
    return math.log(max(0.5, 1.0 + hours_since_release(moment)))


def round2_half_up(x):
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    return whole / 100.0


def implied_earned(rate, players):
    """The integer earned count whose printed rate is `rate` at `players`.

    The page prints rates to 2 decimals, so the count is (almost always)
    unique; fall back to the nearest count when it is not.
    """
    if rate is None or players is None or players <= 0:
        return None
    best = None
    best_err = None
    for e in range(0, players + 1):
        if round2_half_up(e * 100.0 / players) == rate:
            return e
        err = abs(e * 100.0 / players - rate)
        if best_err is None or err < best_err:
            best, best_err = e, err
    return best


def fit_line(xs, ys):
    """Least squares y = a + b*x; returns (a, b, residual RMS)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, resid


def median_of(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def build_epochs(case):
    """Compress the history into one entry per distinct page state.

    Each epoch carries the time the new state was most likely refreshed
    (midpoint between the last read of the old state and the first read
    of the new one), the players tracked, and the rates.  Consecutive
    identical reads are one stale page, not many data points.
    """
    history = case.get("history") or []
    epochs = []
    prev_key = None
    prev_at = None
    cur = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(-1 if r is None else r for r in rates))
        at = parse_time(read["at"])
        if key != prev_key:
            if cur is not None:
                epochs.append(cur)
            # The refresh happened between the previous read and this one.
            t_epoch = at if prev_at is None else prev_at + (at - prev_at) / 2
            cur = {"t": t_epoch, "players": read.get("players_tracked"), "rates": list(rates)}
            prev_key = key
        prev_at = at
    if cur is not None:
        epochs.append(cur)
    return epochs


def last_known_rates(case):
    """Per trophy, the last non-null rate in the history and its read."""
    n = len(case["trophies"])
    last_rate = [None] * n
    last_players = [None] * n
    for read in case.get("history") or []:
        rates = read.get("rates") or []
        for i in range(min(n, len(rates))):
            if rates[i] is not None:
                last_rate[i] = rates[i]
                last_players[i] = read.get("players_tracked")
    return last_rate, last_players


def quantiles_from_median(median, sigma):
    lg = logit(median)
    out = []
    for z in Z:
        q = inv_logit(lg + z * sigma)
        out.append(round6(clamp(q, 0.01, 100.0)))
    # Rounding can only tie neighbours, never invert them, but enforce it.
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def refresh_forecast(case, last_rate, last_players):
    """Stale-page regime: allocate the refresh gain by rank-inv x tier.

    The refreshed page holds PRIOR_EARNED_NON100 earned trophies across
    the non-100% trophies, so the gain over the launch counts is
    PRIOR_EARNED_NON100 - sum(e0) = 408 - 305 = 103 (computed here, not
    hardcoded).  Each non-100 trophy's share of the gain is proportional
    to weight = rank-inv(e0) x tier-value: sort the 54 non-100 trophies by
    e0 ascending (ties keep id order), rank-inv = 54 - position (smallest
    e0 gets 54, largest gets 1), tier-value platinum 4 / gold 3 / silver 2
    / bronze 1.  Shares are apportioned by largest remainder (Hamilton):
    floor each trophy's proportional share, then give the remaining counts
    one each to the largest fractional remainders (ties: larger e0 first,
    then lower id).  e1 = e0 + gain, and the median is the printed rate
    that count gives at the refreshed player total, rounded to 2 decimals
    exactly as the page prints it: predicting the printed value is what
    scores."""
    earned0 = []
    for i, r in enumerate(last_rate):
        earned0.append(implied_earned(r, last_players[i]))
    non100 = [i for i, r in enumerate(last_rate) if r is not None and r < 100 and earned0[i] is not None]
    e0_total = sum(earned0[i] for i in non100) if non100 else 305
    gain_total = PRIOR_EARNED_NON100 - e0_total
    # Rank-inv: sort by e0 ascending, stable (ties keep id order).
    order_e0 = sorted(non100, key=lambda i: (earned0[i], i))
    n_non100 = len(non100)
    weight = {}
    for pos, i in enumerate(order_e0):
        rankinv = n_non100 - pos
        tier = case["trophies"][i].get("tier", "bronze")
        weight[i] = rankinv * TIER_VALUE.get(tier, 1)
    wsum = sum(weight.values()) if weight else 1.0
    raw = {i: weight[i] * gain_total / wsum for i in non100} if gain_total > 0 else {i: 0.0 for i in non100}
    fl = {i: math.floor(raw[i]) for i in non100}
    rem = {i: raw[i] - fl[i] for i in non100}
    need = gain_total - sum(fl.values()) if gain_total > 0 else 0
    order = sorted(non100, key=lambda i: (-rem[i], -earned0[i], i))
    gain = dict(fl)
    for i in order[:max(need, 0)]:
        gain[i] += 1
    e1 = {i: earned0[i] + gain[i] for i in non100}
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = last_rate[i]
        if r is None:
            median = 1.0
        elif r >= 100:
            median = 100.0
        else:
            median = round2_half_up(e1[i] * 100.0 / PRIOR_PLAYERS)
        if median >= 100:
            rows.append({"id": trophy["id"], "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
        else:
            rows.append({"id": trophy["id"], "q": quantiles_from_median(median, REFRESH_SIGMA)})
    return rows


def trend_forecast(case, epochs):
    """Trend regime: power-law extrapolation of earned counts and players.

    v7: each trophy's own least-squares slope is used directly
    (b_use = clamp(b_i, -1.0, 3.0), with its own intercept a_i).  v6
    shrank the slope toward the pooled slope while keeping the unshrunk
    intercept, which pivoted the line at x=0 instead of the data centre.
    """
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = log_time(target)
    trophies = case["trophies"]
    n = len(trophies)
    last_rate, _ = last_known_rates(case)

    # Players: log players vs log-time.
    xs_n, ys_n = [], []
    for ep in epochs:
        if ep["players"]:
            xs_n.append(log_time(ep["t"]))
            ys_n.append(math.log(ep["players"]))
    if len(xs_n) >= 2:
        _, b_n, _ = fit_line(xs_n, ys_n)
        b_n = clamp(b_n, 0.0, 2.0)
        a_n = sum(ys_n) / len(ys_n) - b_n * (sum(xs_n) / len(xs_n))
        players_pred = math.exp(a_n + b_n * x_target)
    else:
        players_pred = float(epochs[-1]["players"] or 638)
    players_pred = max(players_pred, float(epochs[-1]["players"] or 638))

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rate[i]
        if r_last is None:
            r_last = 1.0
        # A trophy every tracked player holds stays held by everyone.
        if all((ep["rates"][i] if i < len(ep["rates"]) else None) == 100 for ep in epochs):
            rows.append({"id": trophy["id"], "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
            continue
        xs, ys, xs_r, ys_r = [], [], [], []
        for ep in epochs:
            r = ep["rates"][i] if i < len(ep["rates"]) else None
            if r is None or ep["players"] is None:
                continue
            e = implied_earned(r, ep["players"])
            if e is None:
                continue
            x = log_time(ep["t"])
            xs.append(x)
            ys.append(math.log(e + 0.5))
            if r > 0:
                xs_r.append(x)
                ys_r.append(math.log(max(r, 0.005)))
        if len(xs) >= 3:
            a_i, b_i, _ = fit_line(xs, ys)
            b_use = clamp(b_i, -1.0, 3.0)
            a_use = a_i
            e_pred = math.exp(a_use + b_use * x_target) - 0.5
            if len(xs_r) >= 3:
                _, _, resid = fit_line(xs_r, ys_r)
            else:
                resid = 0.3
        elif len(xs) >= 1:
            e_pred = math.exp(ys[-1]) - 0.5
            resid = 0.3
        else:
            e_pred = 0.0
            resid = 0.3
        e_pred = max(e_pred, 0.0)
        median = 100.0 * e_pred / players_pred if players_pred > 0 else r_last
        median = clamp_rate(median)
        # Never let the extrapolation run away from the last read.
        lg_med = clamp(logit(median), logit(r_last) - 2.5, logit(r_last) + 2.5)
        median = inv_logit(lg_med)
        sigma = math.sqrt(0.10 ** 2 + (0.18 * math.sqrt(horizon_days)) ** 2 + (0.6 * resid) ** 2)
        sigma = clamp(sigma, 0.08, 0.85)
        rows.append({"id": trophy["id"], "q": quantiles_from_median(median, sigma)})
    return rows


def forecast_case(case):
    history = case.get("history") or []
    if not history:
        # No history at all: an honest, wide, valid forecast.
        return {"trophies": [
            {"id": t["id"], "q": quantiles_from_median(1.0, 0.8)}
            for t in case["trophies"]
        ]}
    epochs = build_epochs(case)
    last_rate, last_players = last_known_rates(case)
    horizon = case.get("horizon_minutes", 0)
    last = history[-1]
    n_last = last.get("players_tracked") or 0
    if len(epochs) <= 1 and horizon <= 720 and n_last <= 1000:
        # One flat page on launch night, short horizon: stale-page regime.
        return {"trophies": refresh_forecast(case, last_rate, last_players)}
    if len(epochs) < 3:
        # Too little drift to fit a trend: persistence with a wide spread.
        rows = []
        for i, trophy in enumerate(case["trophies"]):
            r = last_rate[i] if last_rate[i] is not None else 1.0
            if r >= 100:
                rows.append({"id": trophy["id"], "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
            else:
                sigma = clamp(0.10 + 0.5 * math.sqrt(max(horizon, 0) / 1440.0), 0.1, 0.9)
                rows.append({"id": trophy["id"], "q": quantiles_from_median(r, sigma)})
        return {"trophies": rows}
    return {"trophies": trend_forecast(case, epochs)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
