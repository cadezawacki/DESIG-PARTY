
import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
import traceback
import math

import orjson
import polars as pl
from aiocache import caches
from app.helpers.string_helpers import clean_camel
from app.helpers.date_helpers import get_today, now_datetime
from app.helpers.generic_helpers import get_asset_class_from_rating_agency
from app.helpers.string_helpers import similarity_score
from app.logs.logging import log
from app.services.kdb.hosts.connections import *
from app.services.kdb.kdb import kdb_where
from app.services.kdb.kdb import query_kdb
from app.helpers.async_timer import async_timer
from app.helpers.type_helpers import ensure_list, ensure_lazy
from app.services.loaders.kdb_queries_dev_v3 import RATING_MAP_SIMPLE

try:
    from rapidfuzz.distance import Levenshtein as _RF_LEV

    _HAVE_RAPIDFUZZ = True
except ImportError:
    _HAVE_RAPIDFUZZ = False

from app.services.loaders.kdb_queries_dev_v3 import (
    junior_traders,
    coalesce_left_join,
    book_maps
)

# -- Tunables -----------------------------
TOP_K = 6

HIGH_SCORE = 20
MEDIUM_SCORE = 10
MEDIUM_CONFIDENCE = 0.4
HIGH_CONFIDENCE = 0.6

WATERFALL_CONFIDENCE = 0.5
WATERFALL_STEP = 0.05
BINARY_EVIDENCE = False

REGIONS = ("US", "EU", "SGP")

# WATERFALL_PASSES: List[Tuple[float, List[str]]] = [
#     (10.0, ["ratingAssetClass", "issuerCountry", "yieldCurvePosition", "industryGroup", "currency", "ticker"]),
#     (8.0, ["ratingAssetClass", "issuerCountry", "industryGroup", "currency", "ticker"]),
#     (6.0, ["ratingAssetClass", "issuerCountry", "currency", "ticker"]),
#     (4.0, ["issuerCountry", "currency", "ticker"]),
#     (2.5, ["ticker", "currency"]),
#     (1.5, ["ticker"]),
# ]

EXLCUDE_DEPARTED = True
EXLCUDE_LOST = False
ENABLE_EXPLAINS = True
INCLUDE_FUNGIBLE = True

# -- Market Hours & Regional Adjustments ----
# SGP market hours (UTC): 09:00-21:00 = working hours for Singapore, penalize; outside = boost
SGP_MARKET_HOURS_START_UTC = 9  # 09:00 UTC = ~17:00 SGT
SGP_MARKET_HOURS_END_UTC = 21  # 21:00 UTC = ~05:00 SGT next day

# -- Desk-Aware Waterfall Pass Configurations ----
# Different desks have different portfolio structures.
# Used in desig_waterfall_portfolio to customize matching criteria per desk.
#
# IG: Split by ticker + yield curve position → include curve in matching
# HY: Split by ticker only → skip curve, use ticker-only matching
# EM: Trust high-scoring bonds → use looser thresholds
#
# Format: {desk_asset: [(min_score, [matching_columns]), ...]}
DESK_WATERFALL_PASSES: Dict[str, List[Tuple[float, List[str]]]] = {
    'IG': [
        (10.0, ['ticker', 'ratingAssetClass', 'issuerCountry', 'yieldCurvePosition', 'industryGroup', 'currency']),
        (8.0, ['ticker', 'ratingAssetClass', 'issuerCountry', 'yieldCurvePosition', 'currency']),
        (6.0, ['ticker', 'ratingAssetClass', 'issuerCountry', 'currency']),
        (4.0, ['ticker', 'issuerCountry', 'currency']),
        (2.5, ['ticker', 'currency']),
        (1.5, ['ticker']),
    ],
    'HY': [
        (8.0, ['ticker', 'ratingAssetClass', 'issuerCountry', 'currency']),
        (6.0, ['ticker', 'ratingAssetClass', 'currency']),
        (4.0, ['ticker', 'issuerCountry', 'currency']),
        (2.5, ['ticker', 'currency']),
        (1.5, ['ticker']),
    ],
    'EM': [
        (8.0, ['ticker', 'ratingAssetClass', 'issuerCountry', 'currency']),
        (5.0, ['ticker', 'ratingAssetClass', 'currency']),
        (3.0, ['ticker', 'issuerCountry', 'currency']),
        (1.5, ['ticker', 'currency']),
        (1.0, ['ticker']),
    ],
}

# Default passes if desk not specified
DEFAULT_WATERFALL_PASSES: List[Tuple[float, List[str]]] = [
    (10.0, ['ticker', 'ratingAssetClass', 'issuerCountry', 'yieldCurvePosition', 'industryGroup', 'currency']),
    (8.0, ['ticker', 'ratingAssetClass', 'issuerCountry', 'industryGroup', 'currency']),
    (6.0, ['ticker', 'ratingAssetClass', 'issuerCountry', 'currency']),
    (4.0, ['ticker', 'issuerCountry', 'currency']),
    (2.5, ['ticker', 'currency']),
    (1.5, ['ticker']),
]

FUZZY_MAX_CANDIDATES_PER_ISIN = 10
FUZZY_THRESHOLD = 0.88
FUZZY_WEIGHT = 10.0
FUZZY_ENABLE = True

WATERFALL_BUDGET_MS = 240_000
WATERFALL_MAX_ROUNDS = 3
EPS = 1e-6

BLOCK_SIZE = 10_000_000

# -- Waterfall coherence scoring --
LOG_NORM = math.log(4)  # ln(4) ≈ 1.386; count=3 maps to boost multiplier 1.0
WATERFALL_HIGH_STEP = 4  # Each pass level adds this to the HIGH_SCORE threshold
INJECTION_SCALE = 2.5    # Multiplier for injected candidates (no base score, pure coherence)
_WATERFALL_MATCH_COLS = ('ticker', 'ratingAssetClass', 'issuerCountry',
                         'yieldCurvePosition', 'industryGroup', 'currency')

# -- Globals -----------------------------

from app.helpers.common import BAD_USERNAMES, BAD_BOOKS

DEPARTED_TRADERS = ['cookmat1', 'liuyin1', 'pridhame', 'cenkevin']

from app.helpers.common import get_algo_books, CRB_STRATEGY_BOOKS
from app.server import get_threads

ALGO_BOOKS = get_threads().submit(get_algo_books()).result()
NON_DESIG_BOOKS = set(BAD_BOOKS) | set(ALGO_BOOKS) | set(CRB_STRATEGY_BOOKS)


# ----------------------------------------
# Rule expressions
# ----------------------------------------

def _is_expr(x):
    if x is None: return False
    return isinstance(x, pl.Expr)


def _weight_to_expr(weight, default_weight):
    if isinstance(weight, dict):
        fallback = default_weight if default_weight is not None else 0.0
        chain = pl.when(pl.lit(False)).then(pl.lit(0.0, pl.Float64))
        for desk, val in weight.items():
            chain = chain.when(pl.col('deskAsset')==desk).then(pl.lit(val, pl.Float64))
        return chain.otherwise(pl.lit(fallback, pl.Float64))
    return pl.lit(weight, pl.Float64)


def _scale_to_expr(desk_scale, default_desk_scale):
    if desk_scale is None: return None
    fallback = default_desk_scale if default_desk_scale is not None else 1.0
    chain = pl.when(pl.lit(False)).then(pl.lit(1.0, pl.Float64))
    for desk, val in desk_scale.items():
        chain = chain.when(pl.col('deskAsset')==desk).then(pl.lit(val, pl.Float64))
    return chain.otherwise(pl.lit(fallback, pl.Float64))


@dataclass
class Rule:
    name: str
    success_weight: Union[float, Dict[str, float]]
    fail_weight: Union[float, Dict[str, float]]
    expr: pl.Expr
    raw_expr: bool = False
    default_success_weight: Optional[float] = None
    default_fail_weight: Optional[float] = None
    desk_scale: Optional[Dict[str, float]] = None
    default_desk_scale: Optional[float] = None
    description: Optional[str] = None
    dynamic_scoring_hint: Optional[str] = None
    scored_expr: pl.Expr = field(init=False)
    col_name: str = field(init=False)

    def __post_init__(self):
        self.col_name = f"_rule_{self.name}"
        scale_expr = _scale_to_expr(self.desk_scale, self.default_desk_scale)

        if not self.raw_expr:
            success_expr = _weight_to_expr(self.success_weight, self.default_success_weight)
            fail_expr = _weight_to_expr(self.fail_weight, self.default_fail_weight)
            base = (pl.when(self.expr.cast(pl.Boolean, strict=False))
                    .then(success_expr)
                    .otherwise(fail_expr)
                    .fill_null(0)
                    )
            if scale_expr is not None:
                base = base * scale_expr
            self.scored_expr = base.alias(self.col_name)
        else:
            base = self.expr.cast(pl.Float64, strict=False).fill_null(0)
            if scale_expr is not None:
                base = base * scale_expr
            self.scored_expr = base.alias(self.col_name)

    def __repr__(self):
        base = self.description if self.description else self.name
        base += ": "

        if self.raw_expr:
            base += "[Dynamic Scoring]" if self.dynamic_scoring_hint is None else self.dynamic_scoring_hint
        elif isinstance(self.success_weight, dict) or isinstance(self.fail_weight, dict):
            base += "{desk-aware}"
        else:
            base += "%+0.0f / %+0.0f" % (self.success_weight, self.fail_weight)
        if self.desk_scale:
            base += " {scaled}"
        return base


def _build_rule_expressions():
    required_cols = {
        "isDesig"                 : (0, pl.Int64),
        "_isTrueDesig"            : (0, pl.Int64),
        "_isFungeDesig"           : (None, pl.Int64),
        "_historicalTrueDesig"    : (0, pl.Int64),
        "_historicalFungeDesig"   : (None, pl.Int64),
        "traderId"                : (None, pl.String),
        "traderName"              : (None, pl.String),
        "runzSenderName"          : (None, pl.String),
        "bookRegion"              : (None, pl.String),
        "houseUsRefreshTime"      : (None, pl.Datetime),
        "houseEuRefreshTime"      : (None, pl.Datetime),
        "houseSgpRefreshTime"     : (None, pl.Datetime),
        "regionBarclaysRegion"    : (None, pl.String),
        "currency"                : (None, pl.String),
        "netPosition"             : (0, pl.Float64),
        "_totalHistoricalPosition": (0, pl.Float64),
        "_maxHistoricalPosition"  : (0, pl.Float64),
        "_numberTrades"           : (0, pl.Int64),
        "_daysSinceLastTrade"     : (None, pl.Int64),
        "isMuni"                  : (0, pl.Int64)
    }

    now = now_datetime()
    default_vals = [
        pl.col('isDesig').fill_null(0).cast(pl.Int64, strict=False).alias('isDesig'),
        pl.col('_isTrueDesig').fill_null(0).cast(pl.Int64, strict=False).alias('_isTrueDesig'),
        pl.col('_historicalTrueDesig').fill_null(0).cast(pl.Float64, strict=False).alias('_historicalTrueDesig'),

        pl.col("traderId").cast(pl.String, strict=False).str.to_lowercase().alias('traderId'),
        pl.col("traderName").cast(pl.String, strict=False).str.to_titlecase().alias('traderName'),
        pl.col("runzSenderName").cast(pl.String, strict=False).str.to_lowercase().alias('runzSenderName'),
        pl.col("traderName").str.split(" ").list.get(-1).str.to_lowercase().alias("_traderLastName"),
        pl.col("runzSenderName").str.split(" ").list.get(-1).str.to_lowercase().alias("_runzLastName"),

        pl.col('bookRegion').cast(pl.String, strict=False).fill_null("NA_B").str.to_uppercase().alias('bookRegion'),
        pl.col('houseUsRefreshTime').cast(pl.Datetime, strict=False).alias('houseUsRefreshTime'),
        pl.col('houseEuRefreshTime').cast(pl.Datetime, strict=False).alias('houseEuRefreshTime'),
        pl.col('houseSgpRefreshTime').cast(pl.Datetime, strict=False).alias('houseSgpRefreshTime'),

        (pl.lit(now, pl.Datetime) - pl.col('houseUsRefreshTime')).dt.total_days().cast(pl.Int64, strict=False).alias(
            '_daysSinceHouseUsRefresh'
        ),

        (pl.lit(now, pl.Datetime) - pl.col('houseEuRefreshTime')).dt.total_days().cast(pl.Int64, strict=False).alias(
            '_daysSinceHouseEuRefresh'
        ),

        (pl.lit(now, pl.Datetime) - pl.col('houseSgpRefreshTime')).dt.total_days().cast(pl.Int64, strict=False).alias(
            '_daysSinceHouseSgpRefresh'
        ),

        pl.col("regionBarclaysRegion").cast(pl.String, strict=False).fill_null("NA_R").str.to_uppercase().alias(
            "regionBarclaysRegion"
            ),

        pl.col('currency').cast(pl.String, strict=False).fill_null("NA_C").str.to_uppercase().alias("currency"),

        pl.col('netPosition').cast(pl.Float64, strict=False).fill_null(0).alias('netPosition'),
        pl.col('_totalHistoricalPosition').cast(pl.Float64, strict=False).fill_null(0).alias(
            '_totalHistoricalPosition'
            ),
        pl.col('_maxHistoricalPosition').cast(pl.Float64, strict=False).fill_null(0).alias('_maxHistoricalPosition'),

        pl.col('_numberTrades').cast(pl.Int64, strict=False).fill_null(0).alias('_numberTrades'),
        pl.col('_daysSinceLastTrade').cast(pl.Int64, strict=False).alias('_daysSinceLastTrade'),
    ]

    additional_values = [

        # Dont penalize bonds without a funge
        pl.when(pl.col('_isFungeDesig').is_null()).then(pl.col('isDesig')).otherwise(pl.col('_isFungeDesig')).fill_null(
            0
            ).cast(pl.Int64, strict=False).alias('_isFungeDesig'),
        pl.when(pl.col('_historicalFungeDesig').is_null()).then(pl.col('_historicalTrueDesig')).otherwise(
            pl.col('_historicalFungeDesig')
            ).fill_null(0).cast(pl.Int64, strict=False).alias(
            '_historicalFungeDesig'
        ),

        pl.when(pl.col("bookRegion")=="US").then(pl.col('houseUsRefreshTime'))
        .when(pl.col("bookRegion")=="EU").then(pl.col('houseEuRefreshTime'))
        .when(pl.col("bookRegion")=="SGP").then(pl.col('houseSgpRefreshTime'))
        .otherwise(pl.lit(None, pl.Datetime)).alias('_houseRefreshTime'),

        pl.when(pl.col("bookRegion")=="US").then(pl.col('_daysSinceHouseUsRefresh'))
        .when(pl.col("bookRegion")=="EU").then(pl.col('_daysSinceHouseEuRefresh'))
        .when(pl.col("bookRegion")=="SGP").then(pl.col('_daysSinceHouseSgpRefresh'))
        .otherwise(pl.lit(None, pl.Int64)).alias('_daysSinceHouseRefresh'),

        pl.col('netPosition').abs().alias('grossPosition'),
    ]

    rules = [
        Rule(
            name="live_desig_true",
            description="Book IS desig",
            success_weight={"EM": 32.0, "MUNI":32.0},
            default_success_weight=16.0,
            fail_weight=-2.0,
            expr=(
                    pl.col("_isTrueDesig")==1
            )
        ),

        Rule(
            name="live_desig_funge",
            description="Book IS desig of funge",
            success_weight={"EM": 16.0, "MUNI": 16.0},
            default_success_weight=8.0,
            fail_weight=0,
            expr=(
                    pl.col("_isFungeDesig")==1
            )
        ),

        Rule(
            name="lost_desig_true",
            description="Book LOST desig title",
            success_weight=-30.0,
            fail_weight=0.0,
            expr=(
                    (pl.col("_historicalTrueDesig") > 0) &
                    (pl.col("isDesig")==0)
            )
        ),

        Rule(
            name="lost_desig_funge",
            description="Book LOST desig title of funge",
            success_weight=-15.0,
            fail_weight=0.0,
            expr=(
                    (pl.col("_historicalFungeDesig") > 0) &
                    (pl.col("isDesig") < 1.0) &
                    (pl.col("isDesig")==0)
            )
        ),

        Rule(
            name="departed_desig",
            description="Trader departed",
            success_weight=-100.0,
            fail_weight=0.0,
            expr=(
                    (pl.col('traderId').is_not_null()) &
                    (pl.col('traderId').is_in(DEPARTED_TRADERS))
            )
        ),

        Rule(
            name="bad_desig",
            description="Non-Desk trader Id",
            success_weight=-50.0,
            fail_weight=0.0,
            expr=(
                    (pl.col('traderId').is_not_null()) &
                    (pl.col('traderId').is_in(BAD_USERNAMES))
            )
        ),

        Rule(
            name="has_runz",
            description="Trader sent RUNZ",
            success_weight=24.0,
            fail_weight=0.0,
            expr=(
                    (pl.col('_traderLastName').is_not_null()) &
                    (pl.col('_runzLastName').is_not_null()) &
                    (pl.col('_traderLastName')!="") &
                    (pl.col('_runzLastName')!="") &
                    (
                            (pl.col('_traderLastName').str.contains(pl.col('_runzLastName'))) |
                            (pl.col('_runzLastName').str.contains(pl.col('_traderLastName')))
                    )
            )
        ),

        Rule(
            name="has_house",
            description="HOUSE Mark exists",
            success_weight=2.0,
            fail_weight=-6.0,
            expr=(
                pl.col('_houseRefreshTime').is_not_null()
            )
        ),

        Rule(
            name="active_house",
            description="Days since mark updated",
            dynamic_scoring_hint="[+10 ... -10]",
            success_weight=0,
            fail_weight=0,
            raw_expr=True,
            expr=(
                pl.when(pl.col('_daysSinceHouseRefresh') < 1).then(pl.lit(10.0, pl.Float64))
                .when(pl.col('_daysSinceHouseRefresh') < 2).then(pl.lit(8.0, pl.Float64))
                .when(pl.col('_daysSinceHouseRefresh') < 5).then(pl.lit(2.0, pl.Float64))
                .when(pl.col('_daysSinceHouseRefresh') < 10).then(pl.lit(-1.0, pl.Float64))
                .when(pl.col('_daysSinceHouseRefresh') < 30).then(pl.lit(-3.0, pl.Float64))
                .when(pl.col('_daysSinceHouseRefresh') < 60).then(pl.lit(-8.0, pl.Float64))
                .when(pl.col('_daysSinceHouseRefresh') >= 60).then(pl.lit(-10.0, pl.Float64))
                .otherwise(pl.lit(0, pl.Float64))
            )
        ),

        Rule(
            name="region_match",
            description="Bond regiion match",
            success_weight=4.0,
            fail_weight=-8.0,
            expr=(
                    pl.col('regionBarclaysRegion')==pl.col('bookRegion')
            )
        ),

        Rule(
            name="local_currency_match",
            description="Local currency match",
            dynamic_scoring_hint="+18 / -6",
            success_weight=0.0,
            fail_weight=0.0,
            raw_expr=True,
            expr=(
                pl.when(pl.col('currency')=="USD").then(pl.lit(0, pl.Float64))  # ambiguous

                # EUR Special case, less of a bonus
                .when((pl.col('currency')=="EUR") & (pl.col('bookRegion')=="US")).then(pl.lit(1.0, pl.Float64))
                .when((pl.col('currency')=="EUR") & (pl.col('bookRegion')=="EU")).then(pl.lit(4.0, pl.Float64))

                .when(
                    (pl.col('bookRegion')=="EU") & (pl.col("currency").is_in(
                        [
                            "GBP", "CHF", "ZAR", "ILS", "NOK", "SEK", "DKK", "PLN", "CZK", "HUF", "RON", "BGN", "ALL",
                            "BAM", "MKD", "ISK", "TRY", "MDL", "UAH", "BYN", "RUB", "SAR", "AED", "QAR", "KWD", "BHD",
                            "OMR", "JOD", "LBP", "IQD", "IRR", "YER", "SYP", "PKR", "AFN", "UZS", "TMT", "KGS", "TJS",
                            "AMD", "GEL", "AZN"
                        ]
                    ))
                    ).then(pl.lit(18.0, pl.Float64))
                .when(
                    (pl.col('bookRegion')=="SGP") & (pl.col("currency").is_in(
                        [
                            "JPY", "CHN", "HKD", "SGD", "INR", "AUD", "CNY", "KRW", "KPW", "MNT", "TWD", "MOP", "MYR",
                            "THB", "IDR", "PHP", "VND", "KHR", "LAK", "MMK", "BND", "BDT", "LKR", "NPR", "BTN", "MVR",
                            "KZT", "NZD", "PGK", "FJD", "SBD", "WST", "TOP", "VUV"
                        ]
                    ))
                    ).then(pl.lit(18.0, pl.Float64))
                .when(
                    (pl.col('bookRegion')=="US") & (pl.col("currency").is_in(
                        [
                            "BRL", "MXN", "AUD", "CAD", "CRC", "PAB", "NIO", "HNL", "GTQ", "BZD", "BSD", "BBD", "CUP",
                            "DOP", "HTG", "JMD", "TTD", "XCD", "ARS", "BOB", "CLP", "COP", "GYD", "PYG", "PEN", "SRD",
                            "UYU", "VES"
                        ]
                    ))
                    ).then(pl.lit(18.0, pl.Float64))
                .otherwise(pl.lit(-6.0, pl.Float64))  # FAIL
            )
        ),

        Rule(
            name="live_position",
            description="Live position in book",
            success_weight=12.0,
            fail_weight=0.0,
            expr=(
                    pl.col('grossPosition').abs() > 0
            )
        ),

        Rule(
            name="hist_position",
            description="Historical position in book",
            success_weight=6.0,
            fail_weight=0.0,
            expr=(
                    pl.col('_totalHistoricalPosition') > 0
            )
        ),

        Rule(
            name="live_block_position",
            description="Live block position in book",
            success_weight=12.0,
            fail_weight=0.0,
            expr=(
                    pl.col('grossPosition') >= BLOCK_SIZE
            )
        ),
        Rule(
            name="hist_block_position",
            description="Historic block position in book",
            success_weight=6.0,
            fail_weight=0.0,
            expr=(
                    pl.col('_maxHistoricalPosition') >= BLOCK_SIZE
            )
        ),

        Rule(
            name="trade_count",
            description="Number of trades",
            dynamic_scoring_hint="[-2 ... +8]",
            success_weight=0.0,
            fail_weight=0.0,
            raw_expr=True,
            expr=(
                pl.when(pl.col('_numberTrades')==0).then(pl.lit(-2.0, pl.Float64))
                .when(pl.col('_numberTrades') < 5).then(pl.lit(2.0, pl.Float64))
                .when(pl.col('_numberTrades') < 10).then(pl.lit(3.0, pl.Float64))
                .when(pl.col('_numberTrades') < 20).then(pl.lit(5.0, pl.Float64))
                .when(pl.col('_numberTrades') >= 20).then(pl.lit(8.0, pl.Float64))
                .otherwise(pl.lit(0.0, pl.Float64))

            )
        ),

        Rule(
            name="trade_recency",
            description="Recency of trades",
            dynamic_scoring_hint="[+20 ... +0]",
            success_weight=0.0,
            fail_weight=0.0,
            raw_expr=True,
            expr=(
                pl.when(pl.col('_daysSinceLastTrade').is_null()).then(pl.lit(0.0, pl.Float64))
                .when(pl.col('_daysSinceLastTrade') < 2).then(pl.lit(20.0, pl.Float64))
                .when(pl.col('_daysSinceLastTrade') < 7).then(pl.lit(14.0, pl.Float64))
                .when(pl.col('_daysSinceLastTrade') < 30).then(pl.lit(8.0, pl.Float64))
                .when(pl.col('_daysSinceLastTrade') < 90).then(pl.lit(2.0, pl.Float64))
                .otherwise(pl.lit(0.0, pl.Float64))
            )
        ),

        Rule(
            name="sgp_market_hours",
            description="SGP trader boost/penalty by market hours",
            dynamic_scoring_hint="%+0.0f / %+0.0f" % (-8, 8),
            success_weight=0.0,
            fail_weight=0.0,
            desk_scale={"IG":1},
            raw_expr=True,
            expr=(
                pl.when(
                    (pl.col('bookRegion')=='SGP') &
                    pl.lit(SGP_MARKET_HOURS_START_UTC <= now_datetime().hour < SGP_MARKET_HOURS_END_UTC)
                ).then(pl.lit(-8.0, pl.Float64))
                .when(
                    (pl.col('bookRegion')=='SGP') &
                    ~pl.lit(SGP_MARKET_HOURS_START_UTC <= now_datetime().hour < SGP_MARKET_HOURS_END_UTC)
                ).then(pl.lit(8.0, pl.Float64))
                .otherwise(pl.lit(0.0, pl.Float64))
            )
        ),

        Rule(
            name="muni_desk_match",
            description="MUNI desk trader on a muni bond",
            success_weight=40.0,
            fail_weight=0.0,
            expr=(
                (pl.col('isMuni')==1) &
                (pl.col('deskAsset')=='MUNI')
            )
        ),
    ]

    return required_cols, default_vals, additional_values, rules


def build_fast_rule_expressions(region="US"):
    now = now_datetime()
    rl = region.lower()
    rt = region.title()
    additional_values = [
        pl.when(
            [
                pl.col('_isFungeDesig').is_null()
            ]
        ).then(pl.col('isDesig')).otherwise(pl.col('_isFungeDesig')).fill_null(0).cast(pl.Int64, strict=False).alias(
            '_isFungeDesig'
            ),

        pl.col('netPosition').abs().alias('grossPosition'),

        pl.col('traderName').str.to_titlecase().alias('traderName'),
        pl.col('traderName').str.to_titlecase().str.split(" ").list.first().alias('_traderFirstName'),
        pl.col('traderName').str.to_titlecase().str.split(" ").list.last().alias('_traderLastName'),

        (pl.lit(now, pl.Datetime) - pl.col(f'house{rt}RefreshTime')).dt.total_days().cast(pl.Int64, strict=False).alias(
            '_daysSinceHouseRefresh'
            ),

        pl.when(pl.col('bvalSubAssetClass')=='Agency').then(pl.lit('EM', pl.String)).otherwise(
            pl.col('bvalSubAssetClass')
            ).alias('bvalSubAssetClass'),

        pl.col('ratingCombined').replace(RATING_MAP_SIMPLE).alias("ratingAssetClass")
    ]

    rules = [
        Rule(
            name="departed_desig",
            description="Trader departed",
            success_weight=-100.0,
            fail_weight=0.0,
            expr=(
                    (pl.col('traderId').is_not_null()) &
                    (pl.col('traderId').is_in(DEPARTED_TRADERS))
            )
        ),

        Rule(
            name="bad_desig",
            description="Non-Desk trader Id",
            success_weight=-50.0,
            fail_weight=0.0,
            expr=(
                    (pl.col('traderId').is_not_null()) &
                    (pl.col('traderId').is_in(BAD_USERNAMES))
            )
        ),

        Rule(
            name="has_house",
            description="HOUSE Mark exists",
            success_weight=2.0,
            fail_weight=-6.0,
            expr=(
                pl.col('_daysSinceHouseRefresh').is_not_null()
            )
        ),

        Rule(
            name="live_desig_true",
            description="Book is the listed desig",
            success_weight=0.0,
            fail_weight=0.0,
            desk_scale={"EM": 2.0, "MUNI":2.0},
            raw_expr=True,
            expr=(
                pl.when((pl.col("_isTrueDesig")==1) & (pl.col('_isFungeDesig')==1))
                .then(pl.lit(34.0, pl.Float64))
                .when(pl.col("_isTrueDesig")==1)
                .then(pl.lit(16.0, pl.Float64))
                .when(pl.col("_isFungeDesig")==1)
                .then(pl.lit(8.0, pl.Float64))
                .when(pl.col('isDesig')==1)
                .then(pl.lit(8.0, pl.Float64))  # redundant, but safety catch
                .otherwise(pl.lit(-2.0, pl.Float64))
            )
        ),

        Rule(
            name="has_runz",
            description="Trader sent RUNZ",
            success_weight=8.0,
            fail_weight=0.0,
            expr=(
                    (pl.col('_traderLastName').is_not_null()) &
                    (pl.col('_runzSenderLastName').is_not_null()) &
                    (pl.col('_traderLastName')!="") &
                    (
                        pl.col('_runzSenderLastName').cast(pl.List, strict=False)
                        .list.eval(pl.element().str.to_lowercase())
                        .list.contains(pl.col('_traderLastName').str.to_lowercase())
                    )

            )
        ),

        Rule(
            name="active_house",
            description="House mark exists and has been recently updated",
            dynamic_scoring_hint="[+10 ... -10]",
            success_weight=0,
            fail_weight=0,
            raw_expr=True,
            expr=(
                pl.when(pl.col("_daysSinceHouseRefresh").is_not_null()).then(
                    pl.when(pl.col('_daysSinceHouseRefresh') <= 1).then(pl.lit(10.0, pl.Float64))
                    .when(pl.col('_daysSinceHouseRefresh') <= 2).then(pl.lit(8.0, pl.Float64))
                    .when(pl.col('_daysSinceHouseRefresh') <= 5).then(pl.lit(3.0, pl.Float64))
                    .when(pl.col('_daysSinceHouseRefresh') <= 14).then(pl.lit(0, pl.Float64))
                    .when(pl.col('_daysSinceHouseRefresh') <= 30).then(pl.lit(-3.0, pl.Float64))
                    .when(pl.col('_daysSinceHouseRefresh') <= 60).then(pl.lit(-6.0, pl.Float64))
                    .otherwise(pl.lit(-10.0, pl.Float64))
                ).otherwise(pl.lit(-10.0, pl.Float64))
            )
        ),

        Rule(
            name="region_match",
            description="Bond region match",
            success_weight=4.0,
            fail_weight=-8.0,
            expr=(
                    pl.col('regionBarclaysRegion')==pl.col('bookRegion')
            )
        ),

        Rule(
            name="asset_match",
            description="BVAL Asset Class match",
            success_weight=0.0,
            fail_weight=0.0,
            raw_expr=True,
            expr=(
                pl.when(pl.col('bvalAssetClass').is_not_null() & pl.col('ratingAssetClass').is_not_null()).then(

                    pl.when(
                        (pl.col('bvalAssetClass')==pl.col('ratingAssetClass')) | (pl.col('bvalAssetClass')=='EM')
                        ).then(
                        pl.when(pl.col("deskAsset")==pl.col('bvalAssetClass')).then(
                            pl.lit(10.0, pl.Float64)  # Strong Success
                        ).otherwise(
                            pl.lit(-8.0, pl.Float64)  # Strong Failure
                        )
                    )
                    .when(pl.col('bvalAssetClass')==pl.col("deskAsset")).then(
                        pl.lit(5.0, pl.Float64)  # Mild Success
                    )
                    .when(pl.col('ratingAssetClass')==pl.col("deskAsset")).then(
                        pl.lit(2.0, pl.Float64)  # Weaker Success
                    )
                    .otherwise(pl.lit(-2.0, pl.Float64))  # Mild Failure
                )
                .when(pl.col('bvalAssetClass').is_not_null()).then(
                    pl.when(pl.col('bvalAssetClass')==pl.col("deskAsset"))
                    .then(pl.lit(5.0, pl.Float64))
                    .otherwise(pl.lit(-2.0, pl.Float64))
                )
                .when(pl.col('ratingAssetClass').is_not_null()).then(
                    pl.when(pl.col('ratingAssetClass')==pl.col("deskAsset"))
                    .then(pl.lit(2.0, pl.Float64))
                    .otherwise(pl.lit(-2.0, pl.Float64))
                )
                .otherwise(pl.lit(0.0, pl.Float64))  # ambiguous
            )
        ),

        Rule(
            name="local_currency_match",
            description="Local currency match",
            dynamic_scoring_hint="+18 / -6",
            success_weight=0.0,
            fail_weight=0.0,
            raw_expr=True,
            expr=(
                pl.when(pl.col('currency')=="USD").then(pl.lit(0, pl.Float64))  # ambiguous

                # EUR Special case, less of a bonus
                .when((pl.col('currency')=="EUR") & (pl.col('bookRegion')=="US")).then(pl.lit(1.0, pl.Float64))
                .when((pl.col('currency')=="EUR") & (pl.col('bookRegion')=="EU")).then(pl.lit(4.0, pl.Float64))

                .when(
                    (pl.col('bookRegion')=="EU") & (pl.col("currency").is_in(
                        [
                            "GBP", "CHF", "ZAR", "ILS", "NOK", "SEK", "DKK", "PLN", "CZK", "HUF", "RON", "BGN", "ALL",
                            "BAM", "MKD", "ISK", "TRY", "MDL", "UAH", "BYN", "RUB", "SAR", "AED", "QAR", "KWD", "BHD",
                            "OMR", "JOD", "LBP", "IQD", "IRR", "YER", "SYP", "PKR", "AFN", "UZS", "TMT", "KGS", "TJS",
                            "AMD", "GEL", "AZN"
                        ]
                    ))
                    ).then(pl.lit(18.0, pl.Float64))
                .when(
                    (pl.col('bookRegion')=="SGP") & (pl.col("currency").is_in(
                        [
                            "JPY", "CHN", "HKD", "SGD", "INR", "AUD", "CNY", "KRW", "KPW", "MNT", "TWD", "MOP", "MYR",
                            "THB", "IDR", "PHP", "VND", "KHR", "LAK", "MMK", "BND", "BDT", "LKR", "NPR", "BTN", "MVR",
                            "KZT", "NZD", "PGK", "FJD", "SBD", "WST", "TOP", "VUV"
                        ]
                    ))
                    ).then(pl.lit(18.0, pl.Float64))
                .when(
                    (pl.col('bookRegion')=="US") & (pl.col("currency").is_in(
                        [
                            "BRL", "MXN", "AUD", "CAD", "CRC", "PAB", "NIO", "HNL", "GTQ", "BZD", "BSD", "BBD", "CUP",
                            "DOP", "HTG", "JMD", "TTD", "XCD", "ARS", "BOB", "CLP", "COP", "GYD", "PYG", "PEN", "SRD",
                            "UYU", "VES"
                        ]
                    ))
                    ).then(pl.lit(18.0, pl.Float64))
                .otherwise(pl.lit(-6.0, pl.Float64))  # FAIL
            )
        ),

        Rule(
            name="live_position",
            description="Live position in book",
            success_weight=0.0,
            fail_weight=0.0,
            raw_expr=True,
            desk_scale={"MUNI":2.0},
            expr=(
                pl.when(pl.col('grossPosition').abs() > 0).then(
                    pl.when(pl.col('grossPosition').abs() >= BLOCK_SIZE)
                    .then(pl.lit(16.0, pl.Float64))
                    .otherwise(pl.lit(12.0, pl.Float64))
                ).otherwise(pl.lit(0.0, pl.Float64))
            )
        ),

        Rule(
            name="sgp_market_hours",
            description="SGP trader boost/penalty by market hours",
            dynamic_scoring_hint="%+0.0f / %+0.0f" % (8, -8),
            success_weight=0.0,
            fail_weight=0.0,
            raw_expr=True,
            expr=(
                pl.when(
                    (pl.col('bookRegion')=='SGP') &
                    pl.lit(SGP_MARKET_HOURS_START_UTC <= now_datetime().hour < SGP_MARKET_HOURS_END_UTC)
                ).then(pl.lit(-8, pl.Float64))
                .when(
                    (pl.col('bookRegion')=='SGP') &
                    ~pl.lit(SGP_MARKET_HOURS_START_UTC <= now_datetime().hour < SGP_MARKET_HOURS_END_UTC)
                ).then(pl.lit(8, pl.Float64))
                .otherwise(pl.lit(0.0, pl.Float64))
            )
        ),

        Rule(
            name="muni_desk_match",
            description="MUNI desk trader on a muni bond",
            success_weight=40.0,
            fail_weight=0.0,
            expr=(
                (pl.col('isMuni')==1) &
                (pl.col('deskAsset')=='MUNI')
            )
        ),
    ]

    return additional_values, rules


# FAST_MODIFIERS, FAST_RULES = build_fast_rule_expressions()


# ----------------------------------------------------------------
# -- Task Helpers
# ----------------------------------------------------------------

async def _cancel_task_safely(task: Optional[List | asyncio.Task]):
    if task is None: return False
    tasks = ensure_list(task)

    for t in tasks:
        if t.done():
            try:
                _ = t.result()
            except (Exception, asyncio.CancelledError, asyncio.TimeoutError):
                pass
        else:
            t.cancel()
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 0.1)
    except (Exception, asyncio.CancelledError, asyncio.TimeoutError):
        pass


# ----------------------------------------------------------------
# -- Data Helpers
# ----------------------------------------------------------------

async def desig_frame_enhancer(r, region, modifiers):
    # These books cannot win desig
    s = r.hyper.schema()
    if 'bookRegion' not in s:
        r = r.with_columns(
            [
                pl.lit(str(region).upper(), pl.String).alias('bookRegion')
            ]
        )

    rl, rt = region.lower(), region.title()
    r = r.filter(
        ~(
                pl.col('bookId').is_null() |
                pl.col('bookId').is_in(list(NON_DESIG_BOOKS)) |
                pl.col('bookId').str.ends_with("DTC")  # tsy books
        )
    ).with_columns(
        [
            pl.col('bookId').cast(pl.String, strict=False).str.to_uppercase().alias('bookId'),
            pl.col('traderId').cast(pl.String, strict=False).str.to_lowercase().alias('traderId'),
            pl.col('bookRegion').cast(pl.String, strict=False).str.to_uppercase().alias('bookRegion'),
            pl.col(f'_{rl}RunzSenderLastName').alias('_runzSenderLastName')
        ]
    )

    r = await coalesce_left_join(r, await book_maps(), on="bookId")
    r = r.filter(~pl.col('traderId').is_in(BAD_USERNAMES))

    if modifiers:
        r = r.with_columns(modifiers)

    return await r.hyper.compress_plan_async()


# ----------------------------------------------------------------
# -- Scoring
# ----------------------------------------------------------------

def _confidence_expr() -> pl.Expr:
    s1 = pl.col("topScores").list.get(0, null_on_oob=True)
    s2 = pl.col("topScores").list.get(1, null_on_oob=True)

    eps = pl.lit(EPS)

    # Case 1 — At least two candidates
    conf_two = (
            (s1 - s2).abs()
            / (s1.abs() + s2.abs() + eps)
    ).clip(lower_bound=0.0, upper_bound=1.0)

    # Case 2 — Exactly one candidate

    conf_one = (
        pl.when(s1 > 0)
        .then(pl.lit(MEDIUM_CONFIDENCE, pl.Float64))
        .otherwise(pl.lit(0.0, pl.Float64))
    )

    # Case 3 — No candidates → 0
    return (
        pl.when(pl.col("topScores").list.len() >= 2)
        .then(conf_two)
        .when(pl.col("topScores").list.len()==1)
        .then(conf_one)
        .otherwise(pl.lit(0.0))
        .alias('desigGapRatio')
    )


async def apply_rules_to_frame(res, rules):
    rule_cols = [r.col_name for r in rules if (r is not None) and hasattr(r, "col_name")]
    if not rule_cols: raise ValueError("Missing Rules")

    # Apply base rule expressions
    exprs = [r.scored_expr for r in rules if (r is not None) and hasattr(r, "scored_expr")]
    with_base_rules = res.with_columns(exprs).hyper.fill_missing(
        rule_cols, defaults={r: 0 for r in rule_cols}
    )

    with_region = with_base_rules.with_columns(
        [
            pl.coalesce([pl.col('traderRegion'), pl.col('bookRegion')]).alias('desigRegion')
        ]
    )

    return await with_region.with_columns(
        [
            pl.sum_horizontal(rule_cols, ignore_nulls=True)
            .alias('desigScore')
        ]
    ).select(
        [
            pl.col('desigScore'),
            pl.col('isin'),
            pl.col('grossPosition'),
            pl.col('bookId').alias('desigBookId'),
            pl.col('traderId').alias('desigTraderId'),
            pl.col('traderName').alias('desigName'),
            pl.col('isDesig'),
            pl.col('desigRegion')
        ]
    ).hyper.compress_plan_async()


async def rank_scored_frame(r):
    half_life = 2
    decay = math.log(2) / half_life
    score_func = pl.lit(1, pl.Float64)  # (-decay * pl.col('_pos')).exp()

    return r.filter(
        [
            pl.col('desigScore').is_not_null()
        ]
    ).with_columns(
        [
            pl.col("desigScore").round(1)
        ]
    ).sort(
        ["isin", "desigScore", "grossPosition"],
        descending=[False, True, True]
    ).group_by(
        [
            pl.col(['isin', 'desigTraderId'])
        ], maintain_order=True
    ).agg(
        [
            pl.all().sort_by(["desigScore", "grossPosition"], descending=[False, False]).last()
        ]
    ).group_by(
        [
            pl.col("isin"),  # pl.col('desigTraderId')
        ], maintain_order=True
    ).agg(
        [
            pl.col("desigBookId").implode().list.head(TOP_K).alias("topBooks"),
            pl.col("desigTraderId").implode().list.head(TOP_K).alias("topTradersIds"),
            pl.col("desigScore").implode().list.head(TOP_K).alias("topScores"),
            pl.col("desigRegion").implode().list.head(TOP_K).alias("topRegions"),
            pl.col("desigName").implode().list.head(TOP_K).alias("topNames"),
        ]
    ).with_columns(_confidence_expr()).with_columns(
        [
            pl.col('topBooks').list.first().alias('desigBookId'),
            pl.col('topTradersIds').list.first().alias('desigTraderId'),
            pl.col('topRegions').list.first().alias('desigRegion'),
            pl.col('topNames').list.first().alias('desigName'),
            pl.col('topScores').list.first().fill_null(0).alias('desigScore'),
        ]
    ).with_columns(
        [
            pl.when(
                (pl.col("desigGapRatio") >= HIGH_CONFIDENCE) &
                (pl.col('desigScore') >= HIGH_SCORE)
            ).then(pl.lit("HIGH_CONFIDENCE"))
            .when(
                (pl.col("desigGapRatio") >= MEDIUM_CONFIDENCE) &
                (pl.col('desigScore') >= MEDIUM_SCORE)
            ).then(pl.lit("MEDIUM_CONFIDENCE"))
            .when(
                (pl.col("desigGapRatio") >= HIGH_CONFIDENCE)
                # & (pl.col('desigScore') >= MEDIUM_SCORE) always true here
            ).then(
                pl.lit("MEDIUM_CONFIDENCE")
            ).when(
                (pl.col("desigScore") <= 0)
            ).then(
                pl.lit("VERY_LOW_CONFIDENCE", pl.String)
            ).otherwise(
                pl.lit("LOW_CONFIDENCE", pl.String)
            )
            .alias("desigConfidence")
        ]
    )


# ----------------------------------------------------------------
# -- Waterfall Superscoring
# ----------------------------------------------------------------

def _label_waterfall_confidence(ranked: pl.DataFrame) -> pl.DataFrame:
    """Apply confidence labels to waterfall-scored bonds (pass-dependent thresholds)."""
    boosted = pl.col('_best_boost').fill_null(0.0) > 0
    pidx = pl.col('_best_pidx').fill_null(0)

    # Looser passes require higher scores to reach HIGH/MEDIUM
    waterfall_high_thresh = (
        pl.lit(HIGH_SCORE, pl.Float64)
        + pidx.cast(pl.Float64) * pl.lit(WATERFALL_HIGH_STEP, pl.Float64)
    )

    return ranked.with_columns(
        pl.when(
            (pl.col('desigGapRatio') >= HIGH_CONFIDENCE)
            & (pl.col('desigScore') >= waterfall_high_thresh)
            & boosted
        ).then(
            pl.concat_str([
                pl.lit("WATERFALL_P"),
                (pidx + 1).cast(pl.String)
            ])
        )
        .when(
            (pl.col('desigGapRatio') >= HIGH_CONFIDENCE)
            & (pl.col('desigScore') >= HIGH_SCORE)
        ).then(pl.lit('HIGH_CONFIDENCE'))
        .when(
            boosted
            & (pl.col('desigGapRatio') >= MEDIUM_CONFIDENCE)
            & (pl.col('desigScore') >= MEDIUM_SCORE)
        ).then(pl.lit('BOOSTED_MEDIUM'))
        .when(
            boosted & (pl.col('desigScore') > 0)
        ).then(pl.lit('BOOSTED_LOW'))
        .when(
            pl.col('desigScore') <= 0
        ).then(pl.lit('VERY_LOW_CONFIDENCE'))
        .otherwise(pl.lit('LOW_CONFIDENCE'))
        .alias('desigConfidence')
    )


_PROMOTABLE_LABELS = frozenset({
    'HIGH_CONFIDENCE', 'WATERFALL_P1', 'WATERFALL_P2',
})


def _find_promoted(scored: pl.DataFrame) -> pl.DataFrame:
    """Extract bonds from a scored waterfall round that are confident enough to
    grow the universe for subsequent rounds.

    Gate requirements (all must be true):
      - desigGapRatio >= HIGH_CONFIDENCE (0.6) -- real separation, not the
        single-candidate default of 0.4
      - desigScore >= HIGH_SCORE (20) -- meaningfully positive
      - label in _PROMOTABLE_LABELS -- excludes BOOSTED_MEDIUM/LOW which are
        too marginal to serve as universe evidence
    """
    return scored.filter(
        (pl.col('desigScore') >= HIGH_SCORE)
        & (pl.col('desigGapRatio') >= HIGH_CONFIDENCE)
        & pl.col('desigConfidence').is_in(list(_PROMOTABLE_LABELS))
    )


def _run_waterfall_round(
    basket_to_score: pl.DataFrame,
    universe_basket: pl.DataFrame,
    shared_match: list[str],
) -> pl.DataFrame:
    """
    Single waterfall round: boost candidate scores using universe evidence.

    For each uncertain bond, boosts candidate trader scores based on how many
    bonds in the universe the trader owns in the same bucket.
    Each candidate is scored through their desk's waterfall passes
    (DESK_WATERFALL_PASSES). Best pass wins (highest boost).

    Self-exclusion: a bond never counts itself as evidence.

    Returns the basket with boosted scores, re-ranked candidates, and
    confidence labels -- or the original basket if no boosts were computed.
    """
    b_cols = set(basket_to_score.hyper.fields)
    u_cols = set(universe_basket.hyper.fields)

    # ---- Trader -> desk mapping (from universe) ----
    has_desk = 'deskAsset' in u_cols
    trader_desk = None
    if has_desk:
        trader_desk = (
            universe_basket
            .filter(pl.col('deskAsset').is_not_null())
            .group_by('desigTraderId')
            .agg(pl.col('deskAsset').first())
        )

    # ---- Explode candidate lists into per-(isin, trader) rows ----
    list_cols = [c for c in ('topTradersIds', 'topScores', 'topBooks',
                             'topNames', 'topRegions')
                 if c in b_cols]
    if 'topTradersIds' not in list_cols or 'topScores' not in list_cols:
        return basket_to_score

    exploded = basket_to_score.explode(list_cols)
    if exploded.hyper.is_empty():
        return basket_to_score

    # Attach desk to each candidate trader
    if trader_desk is not None:
        exploded = exploded.join(
            trader_desk.rename({
                'desigTraderId': 'topTradersIds',
                'deskAsset': '_cand_desk'
            }),
            on='topTradersIds', how='left'
        )
    else:
        exploded = exploded.with_columns(
            pl.lit(None, pl.String).alias('_cand_desk')
        )

    # ---- Trader info for injection candidates ----
    trader_info = (
        universe_basket
        .filter(pl.col('desigTraderId').is_not_null())
        .group_by('desigTraderId')
        .agg([
            pl.col('desigBookId').first(),
            pl.col('desigName').first(),
            pl.col('desigRegion').first(),
        ])
    )
    injection_parts: list[pl.DataFrame] = []
    basket_inj_cols = ['isin', 'topTradersIds'] + [c for c in shared_match if c in b_cols]
    basket_for_injection = basket_to_score.select(basket_inj_cols)

    # ---- Walk desk/pass combinations, collect boosts + injections ----
    boost_parts: list[pl.DataFrame] = []
    known_desks = list(DESK_WATERFALL_PASSES.keys())
    desk_configs = (
        list(DESK_WATERFALL_PASSES.items())
        + [('_DEFAULT', DEFAULT_WATERFALL_PASSES)]
    )

    for desk_key, passes in desk_configs:
        try:
            # Filter candidates belonging to this desk
            if desk_key == '_DEFAULT':
                mask = (
                    pl.col('_cand_desk').is_null()
                    | ~pl.col('_cand_desk').is_in(known_desks)
                )
            else:
                mask = pl.col('_cand_desk') == desk_key

            desk_rows = exploded.filter(mask)
            if desk_rows.hyper.is_empty():
                continue

            for pidx, (weight, pcols) in enumerate(passes):
                try:
                    usable = [c for c in pcols if c in shared_match]
                    if not usable: continue

                    # Universe bond counts per (bucket, trader)
                    uc = (
                        universe_basket
                        .filter(pl.col('desigTraderId').is_not_null())
                        .group_by(usable + ['desigTraderId'])
                        .agg(pl.len().alias('_uc'))
                    )

                    # Self-presence markers for exclusion
                    us = (
                        universe_basket
                        .filter(pl.col('desigTraderId').is_not_null())
                        .select(['isin', 'desigTraderId'] + usable)
                        .unique()
                        .with_columns(pl.lit(1, pl.Int8).alias('_sf'))
                    )

                    # Join: candidates <- counts <- self markers
                    j = (
                        desk_rows
                        .join(uc,
                              left_on=usable + ['topTradersIds'],
                              right_on=usable + ['desigTraderId'],
                              how='left')
                        .join(us,
                              left_on=['isin', 'topTradersIds'] + usable,
                              right_on=['isin', 'desigTraderId'] + usable,
                              how='left')
                    )

                    # Self-excluded count -> capped-log boost
                    j = (
                        j
                        .with_columns(
                            (pl.col('_uc').fill_null(0)
                             - pl.col('_sf').fill_null(0))
                            .clip(lower_bound=0)
                            .alias('_adj')
                        )
                        .filter(pl.col('_adj') > 0)
                        .with_columns([
                            (
                                pl.lit(weight, pl.Float64)
                                * (
                                    (pl.col('_adj').cast(pl.Float64) + 1.0).log()
                                    / pl.lit(LOG_NORM, pl.Float64)
                                ).clip(upper_bound=1.0)
                            ).alias('_boost'),
                            pl.lit(pidx, pl.Int32).alias('_pidx'),
                        ])
                        .select(['isin', 'topTradersIds', '_boost', '_pidx'])
                    )

                    if not j.hyper.is_empty():
                        boost_parts.append(j)

                    # ---- INJECTION: find universe traders NOT in existing candidates ----
                    inj = (
                        basket_for_injection
                        .join(uc, on=usable, how='inner')
                        .filter(
                            ~pl.col('topTradersIds')
                            .list.contains(pl.col('desigTraderId'))
                        )
                        .join(us,
                              left_on=['isin', 'desigTraderId'] + usable,
                              right_on=['isin', 'desigTraderId'] + usable,
                              how='left')
                        .with_columns(
                            (pl.col('_uc').fill_null(0)
                             - pl.col('_sf').fill_null(0))
                            .clip(lower_bound=0)
                            .alias('_adj')
                        )
                        .filter(pl.col('_adj') > 0)
                        .with_columns([
                            (
                                pl.lit(weight * INJECTION_SCALE, pl.Float64)
                                * (
                                    (pl.col('_adj').cast(pl.Float64) + 1.0).log()
                                    / pl.lit(LOG_NORM, pl.Float64)
                                ).clip(upper_bound=1.0)
                            ).alias('_inject_score'),
                            pl.lit(pidx, pl.Int32).alias('_pidx'),
                        ])
                        .select(['isin', 'desigTraderId', '_inject_score', '_pidx'])
                    )
                    if not inj.hyper.is_empty():
                        injection_parts.append(inj)

                except Exception as e:
                    log.warning(f"Waterfall {desk_key}/P{pidx + 1}: {e}")
                    continue

        except Exception as e:
            log.warning(f"Waterfall desk '{desk_key}': {e}")
            continue

    # ---- Nothing to do: return unchanged ----
    has_boosts = bool(boost_parts)
    has_injections = bool(injection_parts)
    if not has_boosts and not has_injections:
        log.info("Waterfall: no coherence boosts or injections computed")
        return basket_to_score

    # ---- Setup per-pass column names ----
    _MAX_PASSES = max(len(p) for p in DESK_WATERFALL_PASSES.values())
    _MAX_PASSES = max(_MAX_PASSES, len(DEFAULT_WATERFALL_PASSES))
    _pass_cols = [f'_p{p + 1}_boost' for p in range(_MAX_PASSES)]

    # ---- Aggregate boosts per (isin, trader) ----
    if has_boosts:
        all_boosts = pl.concat(boost_parts, how='diagonal_relaxed')

        # Spread per-pass boosts into columns BEFORE grouping
        for p in range(_MAX_PASSES):
            all_boosts = all_boosts.with_columns(
                pl.when(pl.col('_pidx') == p)
                .then(pl.col('_boost'))
                .otherwise(pl.lit(None, pl.Float64))
                .alias(f'_p{p + 1}_boost')
            )

        best = (
            all_boosts
            .group_by(['isin', 'topTradersIds'], maintain_order=True)
            .agg(
                [
                    pl.col('_boost').max().alias('_boost'),
                    pl.col('_pidx').sort_by('_boost', descending=True).first().alias('_pidx'),
                ] + [pl.col(c).max() for c in _pass_cols]
            )
        )

        merged = (
            exploded
            .join(best, on=['isin', 'topTradersIds'], how='left')
            .with_columns([
                pl.col('_boost').fill_null(0.0).alias('_boost'),
                (pl.col('topScores') + pl.col('_boost').fill_null(0.0))
                .round(1).alias('topScores'),
                pl.lit(False, pl.Boolean).alias('_injected'),
            ])
        )
    else:
        merged = exploded.with_columns(
            [
                pl.lit(0.0, pl.Float64).alias('_boost'),
                pl.lit(None, pl.Int32).alias('_pidx'),
                pl.lit(False, pl.Boolean).alias('_injected'),
            ] + [pl.lit(None, pl.Float64).alias(c) for c in _pass_cols]
        )

    # ---- Process and append injection candidates ----
    if has_injections:
        all_inj = pl.concat(injection_parts, how='diagonal_relaxed')

        # Spread per-pass injection scores into columns
        for p in range(_MAX_PASSES):
            all_inj = all_inj.with_columns(
                pl.when(pl.col('_pidx') == p)
                .then(pl.col('_inject_score'))
                .otherwise(pl.lit(None, pl.Float64))
                .alias(f'_p{p + 1}_boost')
            )

        best_inj = (
            all_inj
            .group_by(['isin', 'desigTraderId'], maintain_order=True)
            .agg(
                [
                    pl.col('_inject_score').max().alias('topScores'),
                    pl.col('_pidx').sort_by('_inject_score', descending=True).first().alias('_pidx'),
                ] + [pl.col(c).max() for c in _pass_cols]
            )
        )

        # Add trader metadata and format to match merged schema
        best_inj = (
            best_inj
            .join(trader_info, on='desigTraderId', how='left')
            .rename({
                'desigTraderId': 'topTradersIds',
                'desigBookId': 'topBooks',
                'desigName': 'topNames',
                'desigRegion': 'topRegions',
            })
            .with_columns([
                pl.col('topScores').round(1),
                pl.col('topScores').alias('_boost'),
                pl.lit(True, pl.Boolean).alias('_injected'),
            ])
        )

        n_injected = best_inj.hyper.height()
        log.info(f"Waterfall: {n_injected} injection candidates computed")
        merged = pl.concat([merged, best_inj], how='diagonal_relaxed')

    # ---- Re-rank per ISIN (greedy: highest total score wins) ----
    ranked = (
        merged
        .sort(['isin', 'topScores'], descending=[False, True])
        .group_by('isin', maintain_order=True)
        .agg(
            [
                pl.col('topBooks').implode().list.head(TOP_K),
                pl.col('topTradersIds').implode().list.head(TOP_K),
                pl.col('topScores').implode().list.head(TOP_K),
                pl.col('topRegions').implode().list.head(TOP_K),
                pl.col('topNames').implode().list.head(TOP_K),
                pl.col('_boost').first().alias('_best_boost'),
                pl.col('_pidx').first().alias('_best_pidx'),
                pl.col('_injected').first().alias('_injected'),
            ] + [
                pl.col(c).first() for c in _pass_cols
            ]
        )
    )

    # ---- Recompute confidence on boosted scores ----
    ranked = ranked.with_columns(_confidence_expr())

    # ---- Extract top candidate (greedy: highest scorer wins) ----
    ranked = ranked.with_columns([
        pl.col('topBooks').list.first().alias('desigBookId'),
        pl.col('topTradersIds').list.first().alias('desigTraderId'),
        pl.col('topRegions').list.first().alias('desigRegion'),
        pl.col('topNames').list.first().alias('desigName'),
        pl.col('topScores').list.first().fill_null(0.0).alias('desigScore'),
    ])

    # ---- Confidence labels ----
    ranked = _label_waterfall_confidence(ranked)

    # ---- Re-attach original non-list columns (matching cols, etc.) ----
    reattach = [c for c in basket_to_score.columns
                if c not in ranked.columns and c != 'isin']
    if reattach:
        ranked = ranked.join(
            basket_to_score.select(['isin'] + reattach).unique(subset=['isin']),
            on='isin', how='left'
        )

    # ---- Preserve bonds that had no candidates (empty lists) ----
    scored_isins = ranked.hyper.to_list('isin')
    missing = basket_to_score.filter(~pl.col('isin').is_in(scored_isins))
    if not missing.hyper.is_empty():
        ranked = pl.concat([ranked, missing], how='diagonal_relaxed')

    # ---- Drop internal columns (keep _best_boost, _best_pidx, _p*_boost for debugging) ----
    _internal = {'_cand_desk', '_boost', '_pidx', '_uc', '_sf', '_adj'}
    ranked = ranked.drop([c for c in ranked.columns if c in _internal])

    return ranked


async def apply_waterfall(
    basket_to_score: pl.DataFrame,
    universe_basket: pl.DataFrame,
) -> pl.DataFrame:
    """
    Portfolio coherence superscoring via desk-aware waterfall with iterative
    cascading.

    Runs up to WATERFALL_MAX_ROUNDS rounds. Each round:
      1. Scores the current basket against the current universe.
      2. Identifies newly promoted bonds (score >= MEDIUM_SCORE,
         gap >= MEDIUM_CONFIDENCE, and label in _PROMOTABLE_LABELS).
      3. Moves promoted bonds from the basket into the universe.
      4. Repeats until no new promotions or max rounds reached.

    This cascading allows confidence to propagate: a bond promoted in round 1
    becomes evidence for uncertain bonds in round 2 that share a broader
    bucket (e.g., same ticker but different curve position).

    Self-exclusion: a bond never counts itself as evidence.

    Args:
        basket_to_score: Bonds to superscore (output of rank_scored_frame).
            Required: isin, topTradersIds, topScores, topBooks,
                      topNames, topRegions.
            Recommended: ticker, ratingAssetClass, issuerCountry,
                         yieldCurvePosition, industryGroup, currency.
        universe_basket: HIGH + MEDIUM confidence reference bonds.
            Required: isin, desigTraderId.
            Recommended: same matching columns + deskAsset.

    Returns:
        Same schema with boosted scores, re-ranked candidates, and labels:
        WATERFALL_P{N}, BOOSTED_MEDIUM, BOOSTED_LOW, HIGH_CONFIDENCE,
        LOW_CONFIDENCE, VERY_LOW_CONFIDENCE.
    """
    try:
        if basket_to_score.hyper.is_empty() or universe_basket.hyper.is_empty():
            return basket_to_score

        b_cols = set(basket_to_score.hyper.fields)
        u_cols = set(universe_basket.hyper.fields)

        # Matching columns available in both inputs
        shared_match = [c for c in _WATERFALL_MATCH_COLS
                        if c in b_cols and c in u_cols]
        if not shared_match:
            log.warning("Waterfall: no shared matching columns, returning unchanged")
            return basket_to_score

        # ---- Iterative cascading ----
        universe = universe_basket
        basket = basket_to_score
        all_promoted: list[pl.DataFrame] = []
        last_scored = None

        for round_idx in range(WATERFALL_MAX_ROUNDS):
            await log.debug(f'WATERFALL ROUND: {round_idx}')
            scored = _run_waterfall_round(basket, universe, shared_match)

            # Tag which round scored these bonds
            scored = scored.with_columns(
                pl.lit(round_idx + 1, pl.Int32).alias('_waterfall_round')
            )

            # Identify newly promoted bonds
            promoted = _find_promoted(scored)
            if promoted.hyper.is_empty():
                log.info(f"Waterfall round {round_idx + 1}: converged (no new promotions)")
                last_scored = scored
                break

            promoted_isins = set(promoted.hyper.to_list('isin'))
            n_promoted = len(promoted_isins)
            log.info(f"Waterfall round {round_idx + 1}: {n_promoted} bonds promoted to universe")

            all_promoted.append(promoted)

            # Grow universe with promoted bonds
            universe = pl.concat([universe, promoted], how='diagonal_relaxed')

            # Shrink basket: remove promoted ISINs
            remaining = scored.filter(~pl.col('isin').is_in(promoted_isins))

            if remaining.hyper.is_empty():
                log.info(f"Waterfall round {round_idx + 1}: basket fully resolved")
                last_scored = remaining
                break

            basket = remaining
        else:
            # Hit max rounds without convergence - do a final pass with the
            # fully grown universe so the last basket benefits from all promotions
            last_scored = _run_waterfall_round(basket, universe, shared_match)
            last_scored = last_scored.with_columns(
                pl.lit(WATERFALL_MAX_ROUNDS + 1, pl.Int32).alias('_waterfall_round')
            )
            log.info(f"Waterfall: max rounds ({WATERFALL_MAX_ROUNDS}) reached, final rescore done")

        # Combine all promoted bonds with the final scored remainder
        if all_promoted:
            parts = all_promoted + ([last_scored] if last_scored is not None
                                    and not last_scored.hyper.is_empty() else [])
            return pl.concat(parts, how='diagonal_relaxed')

        # No promotions ever happened - return the single-round result
        return last_scored if last_scored is not None else scored

    except Exception as e:
        log.error(f"Waterfall failed: {e}\n{traceback.format_exc()}")
        return basket_to_score

