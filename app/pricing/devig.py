"""Bookmaker margin removal (B14 "de-vigged probability", FR-PRED-03).

A bookmaker's prices imply probabilities that sum to more than one; the excess
is the margin. De-vigging scales them back to a fair set that sums to one.
"""


def implied_probability(decimal_price: float) -> float:
    """The probability a decimal price implies, margin included."""
    if decimal_price <= 1.0:
        raise ValueError(f"decimal price must exceed 1.0, got {decimal_price}")
    return 1.0 / decimal_price


def margin(prices: list[float]) -> float:
    """The bookmaker's overround: how far implied probabilities exceed one."""
    return sum(implied_probability(p) for p in prices) - 1.0


def devig(prices: list[float]) -> list[float]:
    """Fair probabilities for a complete market, by proportional scaling.

    `prices` must cover every outcome of ONE market -- home/draw/away, or
    over/under at one line. Pass half a market and the result is meaningless.
    """
    if len(prices) < 2:
        raise ValueError("a market needs at least two outcomes")
    implied = [implied_probability(p) for p in prices]
    total = sum(implied)
    return [p / total for p in implied]
