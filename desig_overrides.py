"""
desig_overrides.py -- terminal hard-override tier for desig assignment.

Runs LAST in the desig pipeline, AFTER the expansion splitter. Whatever
desig was assigned by the portfolio waterfall, expansion, or anything
else is unconditionally overwritten when a bond matches a configured
rule. Name/book/region/desk fields are auto-filled from `book_maps` so
ops only supplies `(when, traderId)` and the rest stays consistent.

Generic matching: `when` is a dict from main-frame column name to value
or list of values. Multiple keys are AND'd; a list value is OR'd within.

    # Every bond with ticker ABC -> John Smith
    {'when': {'ticker': 'ABC'}, 'traderId': 'jsmith123'}

    # All ABC, DEF, GHI tickers -> Mary Jones
    {'when': {'ticker': ['ABC', 'DEF', 'GHI']}, 'traderId': 'mjones'}

    # GBP bank bonds -> Kim Watson
    {'when': {'industryGroup': 'Banks', 'currency': 'GBP'},
     'traderId': 'kwatson'}

Integration:
  * kdb_queries_dev_v3.py -- re-export `desig_hard_override` the same
    way the expansion tier does it.
  * load_sequence_v3.py -- add a DataTask with
    `strict_task_requirements=['desig_expanded_splitter']` so it runs
    after every algorithmic desig step. Example block at the bottom
    of this file.

The `HARD_OVERRIDES` list below is the default source. Swap it for a
config-file loader, KDB lookup, etc. -- the task function accepts an
`overrides=` kwarg for injection from DataTask kwargs.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

import polars as pl

from app.logs.logging import log


# ====================================================================
# Config
# ====================================================================

# Default override list. Empty until ops populates it, so importing
# this module is a no-op until rules exist.
HARD_OVERRIDES: List[Dict[str, Any]] = [
    # {
    #     'when': {'ticker': 'ABC'},
    #     'traderId': 'jsmith123',
    #     'reason': 'Ops manual assignment 2026-04-17',
    # },
]

HARD_OVERRIDE_LABEL = 'HARD_OVERRIDE'
# Score well above anything the algorithmic tiers can produce, so that
# any downstream code that (incorrectly) re-sorts on desigScore still
# puts overridden rows first. Also a clear tell in logs.
HARD_OVERRIDE_SCORE = 1000.0
HARD_OVERRIDE_GAP = 1.0

# Columns we'll write. Only those present on the target frame are kept
# in the returned frame -- the DataTask merge preserves any we didn't
# touch, and avoids schema-mismatch errors if a column was dropped
# upstream.
_OVERRIDE_TARGETS = (
    'desigTraderId', 'desigBookId', 'desigName',
    'desigFirstName', 'desigLastName', 'desigRegion',
    'deskAsset', 'desigConfidence', 'desigScore', 'desigGapRatio',
    # List-form columns the splitters also populate. Overwrite so the
    # scalar desig* fields stay consistent with the list heads.
    'topTradersIds', 'topBooks', 'topNames', 'topRegions', 'topScores',
)


# ====================================================================
# Late imports (avoid circular deps)
# ====================================================================

def _late_imports():
    from app.services.loaders.kdb_queries_dev_v3 import book_maps
    return book_maps


# ====================================================================
# Validation
# ====================================================================

def _validate_rule(rule: Any) -> Optional[str]:
    """Return an error string if the rule is malformed, else None."""
    if not isinstance(rule, dict):
        return f"rule is not a dict: {rule!r}"
    when = rule.get('when')
    if not isinstance(when, dict) or not when:
        return f"rule missing non-empty 'when' dict"
    for col, val in when.items():
        if not isinstance(col, str) or not col:
            return f"rule 'when' key must be a non-empty string: {col!r}"
        if val is None:
            return f"rule 'when' value for {col!r} cannot be None"
        if isinstance(val, (list, tuple, set)) and not val:
            return f"rule 'when' value for {col!r} is an empty collection"
    trader_id = rule.get('traderId')
    if not isinstance(trader_id, str) or not trader_id:
        return f"rule missing non-empty 'traderId' string"
    return None


# ====================================================================
# Matching
# ====================================================================

def _match_expr(when: Dict[str, Any], available_cols: set) -> Optional[pl.Expr]:
    """Build a Polars filter expression from a `when` dict. Returns
    None if any referenced column is missing from the target frame --
    the caller skips the rule and logs a warning.
    """
    exprs: List[pl.Expr] = []
    for col, val in when.items():
        if col not in available_cols:
            return None
        if isinstance(val, (list, tuple, set)):
            exprs.append(pl.col(col).is_in(list(val)))
        else:
            exprs.append(pl.col(col) == val)
    out = exprs[0]
    for e in exprs[1:]:
        out = out & e
    return out


# ====================================================================
# Trader metadata lookup
# ====================================================================

async def _resolve_trader_metadata(
    trader_ids: List[str],
) -> Optional[pl.DataFrame]:
    """For each traderId, return one row with canonical name, main book,
    region, and deskAsset -- sourced from `book_maps` using the same
    main-book resolution as the expansion tier (highest desigCount wins,
    ties broken by bigSize then alphabetical bookId).

    Returns None if book_maps is unavailable or no traderIds resolve.
    """
    if not trader_ids:
        return None
    book_maps = _late_imports()
    bm = await book_maps()
    if bm is None:
        log.warning("desig_hard_override: book_maps unavailable")
        return None
    try:
        if bm.hyper.is_empty():
            return None
    except Exception:
        return None

    trader_ids_lc = [tid.lower() for tid in trader_ids]

    resolved = (
        bm.lazy()
          .select([
              pl.col('traderId').cast(pl.String, strict=False)
                  .str.to_lowercase().alias('traderId'),
              pl.col('bookId'),
              pl.col('traderName'),
              pl.col('traderFirstName'),
              pl.col('traderLastName'),
              pl.col('traderRegion'),
              pl.col('deskAsset'),
              pl.col('desigCount').cast(pl.Float64, strict=False).fill_null(0),
              pl.col('bigSize').cast(pl.Float64, strict=False).fill_null(0),
          ])
          .filter(pl.col('traderId').is_in(trader_ids_lc))
          .sort(
              ['traderId', 'desigCount', 'bigSize', 'bookId'],
              descending=[False, True, True, False],
          )
          .group_by('traderId', maintain_order=True)
          .agg([
              pl.col('bookId').first().alias('desigBookId'),
              pl.col('traderName').first().alias('desigName'),
              pl.col('traderFirstName').first().alias('desigFirstName'),
              pl.col('traderLastName').first().alias('desigLastName'),
              pl.col('traderRegion').first().alias('desigRegion'),
              pl.col('deskAsset').first().alias('deskAsset'),
          ])
          .rename({'traderId': 'desigTraderId'})
    )
    try:
        return await resolved.collect_async()
    except Exception:
        log.warning(
            "desig_hard_override: trader metadata collect failed:\n"
            f"{traceback.format_exc()}"
        )
        return None


# ====================================================================
# DataTask entrypoint
# ====================================================================

async def desig_hard_override(
    my_pt,
    region: str = "US",
    dates=None,
    frames=None,
    overrides: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
):
    """Terminal hard-override.

    For every rule whose `when` clause matches bonds on the main frame,
    overwrite `desigTraderId / desigBookId / desigName / desigFirstName
    / desigLastName / desigRegion / deskAsset / desigConfidence /
    desigScore / desigGapRatio` plus the top* list equivalents.

    Should run LAST in the desig chain so portfolio HIGH, waterfall
    P1/P2, and expansion HIGH are all superseded on conflict.

    Returns a frame keyed by `isin` with the override columns, or None
    if no rules fire. The DataTask merges this onto main by isin.
    """
    if my_pt is None:
        return None

    rules = overrides if overrides is not None else HARD_OVERRIDES
    if not rules:
        return None

    # Validate up-front. Fail loudly on broken rules rather than
    # silently ignoring them.
    valid_rules: List[Dict[str, Any]] = []
    for i, rule in enumerate(rules):
        err = _validate_rule(rule)
        if err:
            log.warning(
                f"desig_hard_override: rule {i} invalid ({err}); skipping"
            )
            continue
        valid_rules.append(rule)
    if not valid_rules:
        return None

    try:
        available_cols = set(my_pt.hyper.fields)
    except Exception:
        return None
    if 'isin' not in available_cols:
        log.warning("desig_hard_override: main frame has no 'isin'; skipping")
        return None

    # Resolve metadata for every referenced trader in ONE book_maps
    # lookup, not per-rule. Book_maps itself is 6h-cached so this is
    # near-free, but still: batch.
    trader_ids = list({rule['traderId'].lower() for rule in valid_rules})
    trader_meta = await _resolve_trader_metadata(trader_ids)
    if trader_meta is None or trader_meta.hyper.is_empty():
        log.warning(
            "desig_hard_override: no book_maps metadata resolved for any "
            f"override trader; skipping all rules. traderIds tried: {trader_ids}"
        )
        return None

    trader_map = {
        row['desigTraderId']: row
        for row in trader_meta.iter_rows(named=True)
    }

    # Apply each rule. Later rules win on the same ISIN (dedup with
    # keep='last' at the end), so rule order in the config is the
    # conflict-resolution policy -- stable and obvious.
    parts: List[pl.DataFrame] = []
    for i, rule in enumerate(valid_rules):
        when = rule['when']
        tid = rule['traderId'].lower()
        reason = rule.get('reason', '')

        info = trader_map.get(tid)
        if info is None:
            log.warning(
                f"desig_hard_override: rule {i} traderId={tid!r} not in "
                f"book_maps; skipping rule"
            )
            continue

        expr = _match_expr(when, available_cols)
        if expr is None:
            missing = [c for c in when if c not in available_cols]
            log.warning(
                f"desig_hard_override: rule {i} references missing columns "
                f"{missing}; skipping rule"
            )
            continue

        try:
            matched = my_pt.filter(expr).select(['isin']).unique()
            if isinstance(matched, pl.LazyFrame):
                matched = await matched.collect_async()
        except Exception:
            log.warning(
                f"desig_hard_override: rule {i} match evaluation failed:\n"
                f"{traceback.format_exc()}"
            )
            continue

        n = matched.hyper.height()
        if n == 0:
            log.info(
                f"desig_hard_override: rule {i} ({when}) matched 0 bonds"
            )
            continue

        log.info(
            f"desig_hard_override: rule {i} ({when}) -> {tid} "
            f"[{n} bond{'s' if n != 1 else ''}]"
            + (f" ({reason})" if reason else "")
        )

        # One row per matched ISIN with all override fields stamped.
        row = pl.DataFrame({
            'isin':            matched.hyper.to_list('isin'),
            'desigTraderId':   [tid] * n,
            'desigBookId':     [info.get('desigBookId')] * n,
            'desigName':       [info.get('desigName')] * n,
            'desigFirstName':  [info.get('desigFirstName')] * n,
            'desigLastName':   [info.get('desigLastName')] * n,
            'desigRegion':     [info.get('desigRegion')] * n,
            'deskAsset':       [info.get('deskAsset')] * n,
            'desigConfidence': [HARD_OVERRIDE_LABEL] * n,
            'desigScore':      [HARD_OVERRIDE_SCORE] * n,
            'desigGapRatio':   [HARD_OVERRIDE_GAP] * n,
            # List-form columns: single-element lists matching the scalars
            # so downstream code that reads `topTradersIds.list.first()`
            # (which is most of it) stays consistent.
            'topTradersIds':   [[tid]] * n,
            'topBooks':        [[info.get('desigBookId')]] * n,
            'topNames':        [[info.get('desigName')]] * n,
            'topRegions':      [[info.get('desigRegion')]] * n,
            'topScores':       [[HARD_OVERRIDE_SCORE]] * n,
        })
        parts.append(row)

    if not parts:
        return None

    result = pl.concat(parts, how='diagonal_relaxed')
    # keep='last' -> later rules win conflicts, as documented above.
    result = result.unique(subset=['isin'], keep='last', maintain_order=True)

    # Guard: only return columns whose names exist in _OVERRIDE_TARGETS
    # (plus `isin`). Prevents accidental leakage of helper columns and
    # keeps the DataTask's expected_col_provides honest.
    keep = ['isin'] + [c for c in result.columns
                       if c in _OVERRIDE_TARGETS and c != 'isin']
    return result.select(keep)


# ====================================================================
# Reference: DataTask block for load_sequence_v3.py
# ====================================================================
#
# Paste AFTER `desig_expanded_splitter` in the desig_tasks list. The
# `overrides=` kwarg is optional -- omit it to use HARD_OVERRIDES from
# this module's top-level config.
#
# DataTask(
#     task_name='desig_hard_override',
#     broadcast_name="Desigs - Hard Override",
#     func=desig_hard_override,
#     merge_key='isin',
#     strict_task_requirements=['desig_expanded_splitter'],
#     fromFrame='main',
#     isOptional=True,
#     use_cached_providers=False,
#     # kwargs={'overrides': [...]},  # or leave out to use HARD_OVERRIDES
#     expected_col_provides={
#         'desigBookId'    : pl.String,
#         'desigTraderId'  : pl.String,
#         'desigName'      : pl.String,
#         'desigFirstName' : pl.String,
#         'desigLastName'  : pl.String,
#         'desigRegion'    : pl.String,
#         'desigConfidence': pl.String,
#         'desigGapRatio'  : pl.Float64,
#         'desigScore'     : pl.Float64,
#         'deskAsset'      : pl.String,
#         'topTradersIds'  : pl.List(pl.String),
#         'topScores'      : pl.List(pl.Float64),
#         'topBooks'       : pl.List(pl.String),
#         'topNames'       : pl.List(pl.String),
#         'topRegions'     : pl.List(pl.String),
#     },
# ),
