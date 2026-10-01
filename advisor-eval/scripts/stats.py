#!/usr/bin/env python3
"""Exact McNemar test on help / harm counts (Agresti, 2012)."""

from __future__ import annotations


def mcnemar_exact(b: int, c: int) -> float:
    """Two-tailed exact McNemar on discordant counts b (help) and c (harm)."""
    from scipy.stats import binomtest

    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
