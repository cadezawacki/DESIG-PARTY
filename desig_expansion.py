"""
desig_expansion.py -- expanded-universe tier for the desig waterfall.

Adds a second `apply_waterfall()` round that uses firm-wide KDB evidence as
universe -- not just our portfolio. Intended for LOW / VERY_LOW confidence
bonds the portfolio round couldn't resolve.

Query strategy
--------------
PANOPROXY-only (cheaper than GATEWAY), snapshot (no history walkback).

  * `firm_wide_desigs(region)` -- ONE PANOPROXY call per region, filtered
    on `desig=1`. The desig=1 gate keeps the scan bounded at
    ~firm_desig_count rows regardless of portfolio size. Hypercached at
    12h so the cost is amortized across every run in the session.

  * `firm_wide_refdata(isins)` -- ONE GATEWAY call for refdata on ISINs
    we may not hold (needed for the waterfall's bucket-match columns:
    `ticker`, `ratingAssetClass`, `issuerCountry`, `industryGroup`,
    `currency`). Hypercached at 24h.

  * Main-book resolution via `book_maps`: for each traderId, the book
    with the highest firm-wide `desigCount` becomes that trader's
    canonical `desigBookId` on every expansion row. This means
    (isin, traderId) scoring always reports the trader's MAIN book, not
    whatever happens to be `.first()` after a group_by.

  * Region (`bookRegion`) is preserved on every row but NOT used as a
    hard filter -- desk regions are soft signals and the existing
    `region_match` rule already nudges scores by +/-8.

Why firm-wide and not "by ticker only"
--------------------------------------
Ticker-based expansion misses boutique issuers, one-off notes, and
private placements. Pulling all firm desigs once (cheap, ~10-30k rows
typically) and letting the waterfall's bucket passes do the matching
is more robust than building a clever pre-filter -- and cheaper too,
because the cache key is regional (not portfolio-dependent).

The waterfall's bucket passes already cover ticker, rating, industry,
country, currency, curve position. So a LOW bond whose ticker has no
firm-wide siblings still gets a shot via (industryGroup, issuerCountry)
or (ticker, currency) co-occurrence with firm desigs on bonds we don't
hold.

Integration points
------------------
  * `kdb_queries_dev_v3.py` -- the dead stub at line 6170 should be
    replaced with a re-export of `desig_expander` from this module.

  * `load_sequence_v3.py` -- the commented-out DataTask at line 576
    should be replaced by the two new tasks shown at the bottom of
    this file (`desig_expander`, `desig_expanded_splitter`).

No new scoring logic. All of `apply_waterfall`'s safeguards (time
budget, deterministic tie-breakers, dedup, self-exclusion) still apply.
"""

from __future__ import annotations

import asyncio
import traceback
from datetime import timedelta
from typing import Optional

import polars as pl

from app.helpers.common import BAD_BOOKS, BAD_USERNAMES
from app.helpers.date_helpers import latest_biz_date
from app.helpers.generic_helpers import L1_DESK_MAP
from app.helpers.q_helpers import kdb_convert_series_to_sym
from app.logs.logging import log
from app.services.kdb.hosts.connections import PANOPROXY, GATEWAY, fconn
from app.services.kdb.kdb import (
    construct_panoproxy_triplet, kdb_col_select_helper, query_kdb,
)

# Late-import to avoid circular dependencies (kdb_queries_dev_v3 imports
# from desigs_redux which imports from kdb_queries_dev_v3).
def _late_imports():
    from app.services.loaders.kdb_queries_dev_v3 import (
        hypercache, build_pt_query, book_maps, coalesce_left_join,
        RATING_MAP_SIMPLE,
    )
    return hypercache, build_pt_query, book_maps, coalesce_left_join, RATING_MAP_SIMPLE


def _late_waterfall():
    from app.services.loaders.desigs_redux import (
        apply_waterfall, _WATERFALL_MATCH_COLS, _PROMOTABLE_LABELS,
    )
    return apply_waterfall, _WATERFALL_MATCH_COLS, _PROMOTABLE_LABELS


REGIONS = ("US", "EU", "SGP")

# Output columns expected by `apply_waterfall` on its `universe_basket` arg.
_UNIVERSE_OUTPUT_COLS = [
    'isin', 'desigTraderId', 'desigBookId', 'desigName', 'desigRegion',
    'deskAsset',
    'ticker', 'ratingAssetClass', 'issuerCountry',
    'yieldCurvePosition', 'industryGroup', 'currency',
]


# ====================================================================
# Layer 1 -- Cheap PANOPROXY pulls
# ====================================================================

async def _firm_wide_desigs_one_region(region: str, dates):
    """One PANOPROXY call: every (isin, bookId) where the firm flagged
    `desig=1` in the given region. Snapshot only.

    Modeled on `pano_positions` (kdb_queries_dev_v3.py:5409) but simpler
    -- no funges, no historical lookback, no per-portfolio ISIN filter.
    The `desig=1` predicate is what keeps the result tiny.
    """
    _, build_pt_query, _, _, _ = _late_imports()
    triplet = construct_panoproxy_triplet(region, 'bondpositions', dates)
    cols = [
        'isin:securityAltId3',
        'netPosition:position',
        'traderId:lower[traderId]',
        'deskType',
    ]
    cols = kdb_col_select_helper(cols, "last")
    q = build_pt_query(
        triplet, cols, dates,
        date_kwargs={'return_today': False},
        filters={'desig': 1},
        by=['isin:securityAltId3', 'bookId'],
    )
    # Drop known-bad books at the KDB layer to cut the result set.
    q += ',(not bookId in (%s))' % kdb_convert_series_to_sym(BAD_BOOKS)
    pano_region = "US" if region == "SGP" else region
    r = await query_kdb(q, fconn(PANOPROXY, region=pano_region))
    if r is None or r.hyper.is_empty():
        return None
    return r.with_columns([
        pl.lit(region, pl.String).alias('bookRegion'),
        pl.col('netPosition').cast(pl.Float64, strict=False).fill_null(0),
        pl.col('deskType').replace_strict(
            L1_DESK_MAP, default="OTHER", return_dtype=pl.String,
        ).alias("deskAsset"),
    ])


def _firm_wide_desigs_cached():
    """Hypercache wrapper, defined at call-time so we don't eagerly
    initialize the cache decorator at import."""
    hypercache, *_ = _late_imports()

    @hypercache.cached(
        ttl=timedelta(hours=12),
        key_params=['region', 'dates'],
    )
    async def _impl(region: str = "US", dates=None):
        dates = latest_biz_date(dates, True)
        return await _firm_wide_desigs_one_region(region, dates)

    return _impl


async def firm_wide_desigs(dates=None) -> Optional[pl.DataFrame]:
    """Fan out one PANOPROXY query per region in parallel; concat results.

    Returns one row per (isin, bookId) for every firm-wide desig=1
    position. Columns: isin, bookId, traderId, netPosition, deskType,
    deskAsset, bookRegion.
    """
    impl = _firm_wide_desigs_cached()
    tasks = [
        asyncio.create_task(impl(region=r, dates=dates))
        for r in REGIONS
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    frames = []
    for r, region in zip(results, REGIONS):
        if isinstance(r, Exception):
            log.warning(
                f"firm_wide_desigs[{region}] failed:\n"
                f"{''.join(traceback.format_exception(type(r), r, r.__traceback__))}"
            )
            continue
        if r is not None and not r.hyper.is_empty():
            frames.append(r)
    if not frames:
        return None
    return pl.concat(frames, how='diagonal_relaxed')


# ====================================================================
# Layer 2 -- Refdata for ISINs we may not hold
# ====================================================================

async def _firm_wide_refdata_impl(isins, dates):
    """Pull refdata bucket columns for an ISIN list. GATEWAY hop, but
    refdata changes slowly so the 24h cache absorbs the cost.
    """
    _, build_pt_query, _, _, _ = _late_imports()
    if not isins:
        return None
    cols = [
        'ticker:Ticker',
        'ratingCombined:RatingCombined',
        'issuerCountry:IssuerCountry',
        'industryGroup:IndustryGroup',
        'currency:Currency',
        'maturityDate:CurrentMaturityDate',
    ]
    q = build_pt_query(
        '.mt.get[`.credit.refData]',
        kdb_col_select_helper(cols, "last"),
        dates,
        date_kwargs={'return_today': False},
        filters={'isin': isins},
        by=['isin'],
    )
    return await query_kdb(q, fconn(GATEWAY))


def _firm_wide_refdata_cached():
    hypercache, *_ = _late_imports()

    @hypercache.cached(
        ttl=timedelta(hours=24),
        deep={"my_pt": True},
        primary_keys={'my_pt': ['isin']},
        key_params=['my_pt', 'dates'],
    )
    async def _impl(my_pt, dates=None):
        isins = my_pt.hyper.ul('isin')
        _, _, _, _, RATING_MAP_SIMPLE = _late_imports()
        r = await _firm_wide_refdata_impl(isins, dates)
        if r is None or r.hyper.is_empty():
            return None
        return r.with_columns([
            pl.col('ratingCombined')
              .replace(RATING_MAP_SIMPLE)
              .alias('ratingAssetClass'),
        ])

    return _impl


async def firm_wide_refdata(my_pt, dates=None) -> Optional[pl.DataFrame]:
    impl = _firm_wide_refdata_cached()
    return await impl(my_pt=my_pt, dates=dates)


# ====================================================================
# Layer 3 -- Main-book resolution
# ====================================================================

async def _resolve_main_books(firm_desigs: pl.DataFrame) -> pl.DataFrame:
    """For each traderId, pick the book with the highest firm-wide
    desigCount. Returns a DataFrame with one row per traderId:
    (desigTraderId, mainBookId, mainDeskAsset, mainBookRegion).

    Uses `book_maps` (already 6h-cached) for the desigCount/bigSize
    signal -- we don't query KDB here, just join to a frame that's
    already in memory.
    """
    _, _, book_maps, _, _ = _late_imports()
    bm = await book_maps()
    if bm is None or bm.hyper.is_empty():
        # Degrade: pick the book where each trader holds the most rows
        # in our firm-desig pull. Better than nothing. Fill the
        # name/region/desk columns with nulls so the downstream join
        # in `build_expanded_universe` still has the expected schema.
        return (
            firm_desigs
            .group_by(['traderId', 'bookId'])
            .agg(pl.len().alias('_n'))
            .sort(['traderId', '_n'], descending=[False, True])
            .group_by('traderId', maintain_order=True)
            .agg([
                pl.col('bookId').first().alias('mainBookId'),
            ])
            .rename({'traderId': 'desigTraderId'})
            .with_columns([
                pl.lit(None, pl.String).alias('desigName'),
                pl.lit(None, pl.String).alias('desigRegion'),
                pl.lit(None, pl.String).alias('mainDeskAsset'),
            ])
        )

    bm = bm.lazy().select([
        pl.col('bookId'),
        pl.col('traderId').cast(pl.String, strict=False).str.to_lowercase()
            .alias('traderId'),
        pl.col('traderName'),
        pl.col('traderRegion'),
        pl.col('deskAsset'),
        pl.col('desigCount').cast(pl.Float64, strict=False).fill_null(0),
        pl.col('bigSize').cast(pl.Float64, strict=False).fill_null(0),
    ])

    # Restrict to books that actually appear in our firm-desig pull --
    # avoids picking a "main book" the trader hasn't used recently.
    active_books = (
        firm_desigs.lazy()
        .select(['traderId', 'bookId'])
        .unique()
    )
    bm = bm.join(active_books, on=['traderId', 'bookId'], how='inner')

    # Rank: most desigs first, then biggest book, then alphabetical
    # bookId for deterministic tie-break. `.first()` after the sorted
    # group_by picks the trader's main book.
    main = (
        bm
        .sort(
            ['traderId', 'desigCount', 'bigSize', 'bookId'],
            descending=[False, True, True, False],
        )
        .group_by('traderId', maintain_order=True)
        .agg([
            pl.col('bookId').first().alias('mainBookId'),
            pl.col('traderName').first().alias('desigName'),
            pl.col('traderRegion').first().alias('desigRegion'),
            pl.col('deskAsset').first().alias('mainDeskAsset'),
        ])
        .rename({'traderId': 'desigTraderId'})
    )
    return await main.collect_async()


# ====================================================================
# Layer 4 -- Build the universe frame for apply_waterfall
# ====================================================================

async def build_expanded_universe(
    low_basket: pl.DataFrame,
    portfolio_universe: Optional[pl.DataFrame],
    dates=None,
) -> Optional[pl.DataFrame]:
    """Returns a universe-shaped DataFrame ready to pass to
    `apply_waterfall(..., universe_basket=...)`.

    Schema matches `_UNIVERSE_OUTPUT_COLS`. Each row is one
    (isin, desigTraderId) pair with bucket columns attached. The
    `desigBookId` is the trader's MAIN book (per `_resolve_main_books`).
    """
    if low_basket is None or low_basket.hyper.is_empty():
        return None

    firm = await firm_wide_desigs(dates=dates)
    if firm is None or firm.hyper.is_empty():
        log.warning("build_expanded_universe: firm_wide_desigs returned nothing")
        return None

    # Drop known-bad usernames at the polars layer (BAD_BOOKS already
    # filtered in KDB).
    firm = firm.filter(~pl.col('traderId').is_in(list(BAD_USERNAMES)))
    if firm.hyper.is_empty():
        return None

    # Refdata for every firm-desig ISIN we don't already have bucket
    # columns for. The portfolio universe has these for the ISINs we
    # hold; the rest need a one-shot GATEWAY pull (24h cached).
    held_isins = set()
    if portfolio_universe is not None and not portfolio_universe.hyper.is_empty():
        try:
            held_isins = set(portfolio_universe.hyper.to_list('isin'))
        except Exception:
            held_isins = set()

    firm_isins_frame = firm.select(['isin']).unique()
    needs_refdata = firm_isins_frame.filter(~pl.col('isin').is_in(list(held_isins)))
    refdata = None
    if not needs_refdata.hyper.is_empty():
        try:
            refdata = await firm_wide_refdata(needs_refdata, dates=dates)
        except Exception:
            log.warning(
                f"build_expanded_universe: firm_wide_refdata failed; "
                f"continuing without bucket columns for non-held ISINs:\n"
                f"{traceback.format_exc()}"
            )
            refdata = None

    # Held ISINs: pull bucket columns from the portfolio universe so we
    # don't double-fetch refdata.
    held_buckets = None
    if portfolio_universe is not None and not portfolio_universe.hyper.is_empty():
        bucket_cols_present = [
            c for c in ('ticker', 'ratingAssetClass', 'issuerCountry',
                        'industryGroup', 'currency', 'yieldCurvePosition')
            if c in portfolio_universe.hyper.fields
        ]
        if bucket_cols_present:
            held_buckets = (
                portfolio_universe
                .select(['isin'] + bucket_cols_present)
                .unique(subset=['isin'])
            )

    # Stitch refdata + held_buckets together
    if refdata is not None and held_buckets is not None:
        bucket_table = pl.concat([refdata, held_buckets], how='diagonal_relaxed')
    elif refdata is not None:
        bucket_table = refdata
    elif held_buckets is not None:
        bucket_table = held_buckets
    else:
        bucket_table = None

    if bucket_table is not None:
        bucket_table = bucket_table.unique(subset=['isin'], keep='first')
        firm = firm.join(bucket_table, on='isin', how='left')

    # Resolve main book per trader and override `desigBookId`.
    main = await _resolve_main_books(firm)
    firm = firm.rename({'traderId': 'desigTraderId'})
    firm = firm.join(main, on='desigTraderId', how='left')

    # Pick the canonical book + metadata. Fall back to the row's own
    # bookId / region / deskAsset if main-book resolution didn't yield
    # anything (rare; only when book_maps is unavailable AND the trader
    # has zero rows in firm desigs that match book_maps -- shouldn't
    # happen in practice but degrade gracefully).
    firm = firm.with_columns([
        pl.coalesce([pl.col('mainBookId'), pl.col('bookId')]).alias('desigBookId'),
        pl.coalesce([pl.col('desigRegion'), pl.col('bookRegion')]).alias('desigRegion'),
        pl.coalesce([pl.col('mainDeskAsset'), pl.col('deskAsset')]).alias('deskAsset'),
    ])

    # One row per (isin, desigTraderId). Keep the row whose bookId
    # matches the resolved main book if available; otherwise any row.
    firm = (
        firm
        .with_columns([
            (pl.col('bookId') == pl.col('desigBookId')).cast(pl.Int8).alias('_is_main_row'),
        ])
        .sort(
            ['isin', 'desigTraderId', '_is_main_row', 'netPosition'],
            descending=[False, False, True, True],
        )
        .unique(subset=['isin', 'desigTraderId'], keep='first', maintain_order=True)
        .drop('_is_main_row')
    )

    # Ensure all expected output columns exist (fill missing with null
    # so downstream apply_waterfall doesn't error on schema mismatch).
    out = firm.hyper.ensure_columns(_UNIVERSE_OUTPUT_COLS)
    return out.select(_UNIVERSE_OUTPUT_COLS)


# ====================================================================
# Layer 5 -- DataTask entrypoints
# ====================================================================

async def desig_expander(my_pt, region="US", dates=None, frames=None, **kwargs):
    """DataTask: produces the `desig_expanded` frame.

    Reads the LOW basket and the existing portfolio universe (joined
    desigs frame), builds an expanded universe via firm-wide PANOPROXY
    pulls + main-book resolution, then runs a second `apply_waterfall`
    round on the LOW basket against (portfolio_universe + expanded).

    Returns a frame with the same shape as the portfolio waterfall
    output (topTradersIds, topScores, ..., desigConfidence, etc.).
    Empty if the LOW basket is empty or KDB fan-out yielded nothing
    -- the caller (`desig_expanded_splitter`) then no-ops.
    """
    apply_waterfall, _WATERFALL_MATCH_COLS, _PROMOTABLE_LABELS = _late_waterfall()

    frames = frames or {}
    low = frames.get('desig_low')
    joined = frames.get('desig_joined')

    if low is None or low.hyper.is_empty():
        return None

    # Portfolio universe = the post-portfolio-waterfall HIGH/P1/P2 bonds.
    # This is what apply_waterfall's first round produced; we extend it.
    portfolio_universe = None
    if joined is not None and not joined.hyper.is_empty():
        _UNIVERSE_LABELS = (
            {'HIGH_CONFIDENCE', 'MEDIUM_CONFIDENCE'}
            | set(_PROMOTABLE_LABELS)
        )
        portfolio_universe = joined.filter(
            pl.col('desigConfidence').is_in(list(_UNIVERSE_LABELS))
        )

    expanded = await build_expanded_universe(
        low_basket=low, portfolio_universe=portfolio_universe, dates=dates,
    )
    if expanded is None or expanded.hyper.is_empty():
        log.info("desig_expander: no expansion data; skipping second waterfall")
        return None

    # Combined universe = portfolio + expanded. The waterfall is
    # source-agnostic; it just counts bucket co-occurrences.
    if portfolio_universe is not None and not portfolio_universe.hyper.is_empty():
        combined_universe = pl.concat(
            [portfolio_universe, expanded], how='diagonal_relaxed',
        )
    else:
        combined_universe = expanded

    log.info(
        f"desig_expander: low={low.hyper.height()} bonds, "
        f"portfolio_universe={0 if portfolio_universe is None else portfolio_universe.hyper.height()} rows, "
        f"expanded={expanded.hyper.height()} rows"
    )

    try:
        result = await apply_waterfall(low, combined_universe)
    except Exception:
        log.error(
            f"desig_expander: apply_waterfall failed:\n{traceback.format_exc()}"
        )
        return None

    if result is None or result.hyper.is_empty():
        return None

    # Tag the output so downstream merges can tell expansion-derived
    # desigs apart from portfolio-derived ones.
    return result.with_columns(
        pl.lit('expanded', pl.String).alias('_desigSource')
    )


async def desig_expanded_splitter(my_pt, region="US", dates=None, frames=None, **kwargs):
    """DataTask: merges expansion-derived HIGH/P1/P2 desigs onto main.

    Precedence rule: portfolio-round HIGH/P1/P2 ALWAYS wins over an
    expansion-round HIGH/P1/P2 for the same ISIN. Expansion only fills
    holes -- it never overwrites portfolio evidence.
    """
    _, _, _PROMOTABLE_LABELS = _late_waterfall()
    _COMPLETE_LABELS = {'HIGH_CONFIDENCE'} | set(_PROMOTABLE_LABELS)

    frames = frames or {}
    expanded = frames.get('desig_expanded')
    joined = frames.get('desig_joined')

    if expanded is None or expanded.hyper.is_empty():
        return None

    # ISINs already resolved by the portfolio round -- keep them
    # untouched.
    portfolio_resolved_isins = set()
    if joined is not None and not joined.hyper.is_empty():
        portfolio_resolved_isins = set(
            joined.filter(pl.col('desigConfidence').is_in(list(_COMPLETE_LABELS)))
                  .hyper.to_list('isin')
        )

    out = expanded.filter(
        pl.col('desigConfidence').is_in(list(_COMPLETE_LABELS))
        & ~pl.col('isin').is_in(list(portfolio_resolved_isins))
    )
    if out.hyper.is_empty():
        return None

    keep_cols = [
        'isin', 'desigBookId', 'desigTraderId', 'desigName', 'desigRegion',
        'desigConfidence', 'desigGapRatio', 'desigScore', 'deskAsset',
    ]
    keep_cols = [c for c in keep_cols if c in out.hyper.fields]
    return out.select(keep_cols)


# ====================================================================
# Reference: DataTask block to paste into load_sequence_v3.py at :576
# replacing the commented-out `desig_expander` stub.
# ====================================================================
#
# DataTask(
#     task_name='desig_expander',
#     broadcast_name="Desigs - Expanded Universe",
#     func=desig_expander,
#     merge_key='isin',
#     strict_task_requirements=['desig_splitter_low'],
#     fromFrame='main',
#     frameContext=['desig_low', 'desig_joined'],
#     toFrame='desig_expanded',
#     isOptional=True,
#     use_cached_providers=False,
#     expected_col_provides={
#         'desigBookId'    : pl.String,
#         'desigTraderId'  : pl.String,
#         'desigName'      : pl.String,
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
# DataTask(
#     task_name='desig_expanded_splitter',
#     broadcast_name="Desigs - Expanded Splitter",
#     func=desig_expanded_splitter,
#     merge_key='isin',
#     strict_task_requirements=['desig_expander'],
#     fromFrame='main',
#     frameContext=['desig_expanded', 'desig_joined'],
#     isOptional=True,
#     use_cached_providers=False,
#     expected_col_provides={
#         'desigBookId'    : pl.String,
#         'desigTraderId'  : pl.String,
#         'desigName'      : pl.String,
#         'desigRegion'    : pl.String,
#         'desigConfidence': pl.String,
#         'desigGapRatio'  : pl.Float64,
#         'desigScore'     : pl.Float64,
#         'deskAsset'      : pl.String,
#     },
# ),
