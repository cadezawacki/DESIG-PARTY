
from __future__ import annotations

import asyncio
import os

import httpx
import polars as pl

try:
    import re2 as re
except ImportError:
    import re

try:
    from rapidfuzz.distance import Levenshtein as _RF_LEV

    _HAVE_RAPIDFUZZ = True
except ImportError:
    _HAVE_RAPIDFUZZ = False

from functools import lru_cache, partial, wraps
from itertools import product
from async_lru import alru_cache

import country_converter
import dateutil.parser
import numpy as np
from gics import GICS

from typing import Callable
from app.helpers.async_timer import async_timer
from app.helpers.common import BAD_DESKS
from app.helpers.common import PRICE_TYPES, SPREAD_TYPES, MMY_TYPES, DM_TYPES, ETF_TICKERS, ALGO_PROFILES, DM_ALGO_BOOKS_PROFILES, SIDE_TYPES, SIZE_TYPES, BENCH_TYPES, QT_TYPES, SETTLE_TYPES, \
    get_algo_books
from app.helpers.date_helpers import BVALSnapshotTimes
from app.helpers.date_helpers import get_today
from app.helpers.date_helpers import get_bval_snap
from app.helpers.date_helpers import parse_single_date, is_today, next_settle_date_from_today, next_biz_date, to_kdb_date, get_utc_bval_mappings, now_datetime, parse_date, add_business_days, isonow, \
    latest_biz_date, date_to_datetime
from app.helpers.generic_helpers import BVAL_ASSET_MAP, BVAL_SUB_ASSET_MAP, L0_DESK_MAP
from app.helpers.generic_helpers import QUOTE_EVENT_MARKET_MAP, MOODY_TO_SP, SP_RATINGS, MOODYS_RATINGS, FITCH_RATINGS, SP_TO_MNEMONIC
from app.helpers.polars_hyper_plugin import *
from app.helpers.type_helpers import ensure_list, ensure_lazy
from app.helpers.generic_helpers import market_id_maps, MARKET_MAPS
from app.helpers.generic_helpers import convert_rating_to_sp, get_asset_class_from_rating_agency, get_rating_mnemonic
from app.helpers.hash import hash_as_hex
from app.helpers.hash import md5_string as generate_portfolio_key
from app.helpers.provides import provides
from app.helpers.q_helpers import kdb_date_query, kdb_convert_polars_to_sym, kdb_convert_polars_to_str, kdb_convert_series_to_str
from app.helpers.string_helpers import clean_camel
from app.helpers.string_helpers import similarity_score
from app.helpers.timedCache import TimedCache
from app.helpers.q_helpers import kdb_convert_series_to_sym, kdb_convert_series_to_sym_strings
from app.logs.logging import log
from app.services.kdb.hosts.connections import *
from app.services.kdb.kdb import kdb_col_select_helper, construct_gateway_triplet, kdb_where, kdb_fby, kdb_by, construct_panoproxy_triplet, region_to_gateway, region_to_panoproxy, query_kdb
from app.helpers.generic_helpers import L1_DESK_MAP
from datetime import timedelta
from pathlib import Path
from app.services.cache.hyperCache import HyperCache
from app.data.united_states_abbr import NAME_TO_STATE_ABBR
from app.data.muni_abbreviations import MUNI_ABBREVIATIONS
from app.helpers.common import get_algo_businesses, get_algo_map, SIGNAL_BOOK_PRIORITY
from app.helpers.common import BAD_BOOKS, BAD_USERNAMES, ALGO_BOOKS, CRB_STRATEGY_BOOKS

# Tunables ===========================================================
MAX_LOOKBACK = 180
BENCH_REGEX = '(^(US)*912)|(^(DE|FR|IT|ES|NL|BE|AT|IE|PT|FI|GR|SI|SK|CZ|PL|HU|RO|BG|HR|LT|LV|EE|LU|CY|MT|DK|SE|NO|IS|GB|CH|EU){1}(000)|(001)|(400))'
BENCHMARK_PRIORITY = ["House", "UsHouse", "bval", "EuHouse", "SgpHouse"]
WHITESPACE_TO_NULL = True
SLOW_TTL = 12 * 60 * 60
SIDE_PAT = re.compile('(Bid|Mid|Ask|bid|mid|ask)')

# ICE Data Services R+


'''
KDB WHERE CONDITIONS:
1. Attribute-backed lookup first
2. Simple comparisons
3. fby 

"last by" is generally the better approach than i=(last;i) fby 
'''



# ====================================================================
## Helpers
# ====================================================================

from app.server import HYPER_VERSION
cache_dir = Path(f"./app/data/{HYPER_VERSION}")
hypercache = HyperCache(
    cache_dir=cache_dir,
    cleanup_enabled=True,
    cleanup_interval=timedelta(minutes=1),
    cleanup_on_start=True,
    max_total_size=None,
)


def deep_cached(deep_param="my_pt", pk_cols=("isin",), **preset_kwargs):
    defaults = dict(
        ttl=timedelta(hours=6),
        deep={deep_param: True},
        primary_keys={deep_param: list(pk_cols)},
        # add any other common defaults here:
        # cache_none=False,
        # heat="cold",
        # verbose=True,
    )
    defaults.update(preset_kwargs)

    def decorator(func):
        return hypercache.cached(**defaults)(func)

    return decorator


FRAME_DTYPE = (pl.DataFrame, pl.LazyFrame)
FLAG_YES = pl.lit(1, pl.Int8)
FLAG_NO = pl.lit(0, pl.Int8)

def write_to_cache(func_name: str, result: FrameLike, my_pt: FrameLike, force=True, **kwargs):
    from app.server import get_ctx
    return get_ctx().spawn(hypercache.write(func_name, result, my_pt, **kwargs), f'write-to-cache-{func_name}')


# debug
async def write_to_cache_async(func_name: str, result: FrameLike, my_pt: FrameLike, force=False, **kwargs):
    kwargs.setdefault('__verbose', False)
    kwargs.setdefault('__loop', asyncio.get_event_loop())
    await hypercache.write(func_name, result, my_pt, force=force, **kwargs)


def _as_lazy(df):
    if df is None: return pl.LazyFrame()
    return df.lazy() if isinstance(df, pl.DataFrame) else df


def _typed_utf8_null():
    return pl.lit(None, dtype=pl.Utf8)


def market_columns(my_pt, markets=None, qt_list=None, exclusions=None, case_sensitive=False, just_columns=True, group=0):
    qt_list = "|".join(list(qt_list or ['Px', 'Spd', 'Yld', 'Mmy', 'Dm']))
    exclusions = "|".join(list(exclusions or ['bench']) + ['eod', 'signal', 'bix'])
    markets = "|".join(list(markets or []))
    return my_pt.hyper.cols_like(rf'(?i)\b({markets})(?!\w*({exclusions}))\w*(?:{qt_list})\b', case_sensitive=case_sensitive, just_columns=just_columns, group=group)


async def hash_df_from_column(df, col) -> str:
    if not col: return ''
    d = await df.select(col).filter(pl.col(col).is_not_null()).collect_async()
    return hash_as_hex(sorted((d.get_column(col).to_list())))


async def lazy_to_list(my_pt, col, unique=True, omit_nulls=True, splitter=None):
    s = my_pt.select(col)
    s = s.filter(pl.col(col).is_not_null()) if omit_nulls else s
    s = await s.collect_async() if isinstance(s, pl.LazyFrame) else s
    s = s[col].unique().to_list() if unique else s[col].to_list()
    if splitter:
        s = [si for sublist in s for si in sublist.split(splitter) if sublist if si not in ['', ' ']]
        s = list(np.unique(s)) if unique else s
    return s


def kdb_time_ms_to_time(iv):
    h, r = divmod(iv, 3_600_000)
    m, r = divmod(r, 60_000)
    s, ms = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def get_schema(my_pt):
    return my_pt.schema if isinstance(my_pt, pl.DataFrame) else my_pt.collect_schema()


# ====================================================================
## Build PT query
# ====================================================================

def build_pt_query(table, cols: Union[List, str] = "", dates=None, date_kwargs=None, filters=None, by=None, firstby=None, lastby=None, raw_filter=False):
    date_kwargs = date_kwargs or {}
    final_filters = ""
    if (cols is None) or (cols==''):
        cols = []
    if isinstance(cols, list):
        cols = kdb_col_select_helper(cols, method='last')

    dates = [dates] if not isinstance(dates, list) else dates
    dates = [parse_date(dt, biz=True) for dt in dates]

    try:
        deep = (len(dates) > 1) or (not is_today(dates[0], True))
    except:
        deep = False

    if filters or deep or (firstby or lastby):
        date_query = kdb_date_query(dates, **date_kwargs)
        f_filters = (kdb_where(filters) if filters else None) if not raw_filter else filters
        f_firstby = kdb_fby(firstby, 'first', 'i') if firstby else None
        f_lastby = kdb_fby(lastby, 'last', 'i') if lastby else None
        fs = [x for x in [date_query, f_filters, f_firstby, f_lastby] if not x is None]
        final_filters = " where " + ",".join(fs)
    f_by = kdb_by(by) if by else ''
    return f"select {cols}{f_by} from {table}{final_filters}"


def _remove_date(cols: str):
    for f in ["date", "date,", ",date"]:
        cols = cols.replace(f, "")
    return cols


async def collect_lazy_ids(my_pt, id_col, concat_name=None):
    if isinstance(id_col, (list, tuple)):
        result = []
        concat_name = id_col[0] if concat_name is None else concat_name
        for col in id_col:
            t = (await collect_lazy_ids(my_pt, col)).select(pl.col(col).alias(concat_name))
            result.append(t)
        return pl.concat(result)
    if id_col in my_pt.collect_schema():
        if isinstance(my_pt, pl.LazyFrame):
            return await my_pt.select(id_col).collect_async()
        return my_pt.select(id_col)


def validate_portfolio_schema(my_pt: pl.LazyFrame, required_cols: List[str]) -> bool:
    schema = my_pt.collect_schema().names()
    missing = set(required_cols) - set(schema)
    if missing:
        return False
    return True


async def _regional_frame_aggregator(my_pt, frames, *, frame_prefix="longterm_positions", regions=('us', 'eu', 'sgp'), join_key='isin', lazy=True):
    name_template = f'{frame_prefix}_%s'
    regionals = {
        reg.lower(): frames.get(name_template % reg.lower(), None)
        for reg in regions
        if (name_template % reg.lower()) in frames
    }
    regions = [ensure_lazy(r) for r in regionals.values() if r is not None]
    if not regions: return None
    r = await pl.concat(regions, how='diagonal_relaxed', strict=False).hyper.compress_plan_async()
    res = await coalesce_left_join(r, my_pt, on=join_key)
    return res if lazy else (await res.hyper.collect_async())

async def coalesce_left_join(df1, df2, on):
    df1 = df1.lazy()
    df2 = df2.lazy()

    # Need a unique index on the joining frame
    df2 = df2.unique(subset=on)

    # Perform the left join
    joined_df = df1.join(df2, on=on, how="left", suffix="_right")

    on_set = set(on) if isinstance(on, list) else {on}
    # Find shared columns, excluding join keys
    schema1 = df1.collect_schema() if isinstance(df1, pl.LazyFrame) else df1.schema
    schema2 = df2.collect_schema() if isinstance(df2, pl.LazyFrame) else df2.schema
    shared_columns = set(schema1.keys()) & set(schema2.keys()) - on_set

    # Coalesce shared columns
    coalesce_cols = []
    for col in shared_columns:
        dtype = schema1[col]
        dtype_new = schema2[col]
        if (dtype==pl.Null) and (dtype_new!=pl.Null):
            coalesce_cols.append(pl.coalesce([pl.col(col).cast(dtype_new, strict=False), pl.col(f"{col}_right")]).alias(col))
        else:
            coalesce_cols.append(pl.coalesce([pl.col(col), pl.col(f"{col}_right").cast(dtype, strict=False)]).alias(col))
    joined_df = joined_df.with_columns(coalesce_cols)

    # Drop the temporary "_right" columns
    columns_to_drop = [f"{col}_right" for col in shared_columns]
    final = joined_df.drop(columns_to_drop)
    return final


def coerce_maybe_columns(my_pt, columns, new_alias, dtype, default=None):
    s = my_pt.hyper.schema().keys()
    norm_cols = {clean_camel(c) for c in columns}
    missing = [x for x in norm_cols if x not in s]
    return my_pt.with_columns([
        pl.lit(default, dtype).alias(x) for x in missing
    ]).with_columns([
        pl.coalesce([pl.col(c) for c in norm_cols]).alias(clean_camel(new_alias))
    ]).drop(missing, strict=False)


def generate_fake_rfq_list_id(venue="MANUAL", date=None, region="US", version=1):
    _venue = str(venue).upper() if venue else "MANUAL"
    _date = parse_single_date(date).strftime('%Y%m%d')
    _region = str(region).upper() if region else ""
    _rand = os.urandom(4).hex().upper()
    _version = f".{int(version or 1)}"
    _code = _region + _rand
    return "_".join([_venue, _date, _code]).upper() + _version


def naive_isin_to_cusip(isin: str) -> str:
    isin = isin.strip().upper()
    return isin[2:11]  # 9-char CUSIP portion

# ====================================================================
## Triplets
# ====================================================================
def _construct_gateway_triplet(region=None, schema=None, table=None, stripe=None, default_region="US"):
    if not (schema and table): raise ValueError
    region = region or default_region
    return construct_gateway_triplet(schema, region, table, stripe=stripe)


portfolio_triplet = partial(_construct_gateway_triplet, schema="portfoliorfqs", table="portfolioToolRfqs")
creditext_triplet = partial(_construct_gateway_triplet, schema="creditext")
bval_triplet = partial(creditext_triplet, table="bval")
am_triplet = partial(creditext_triplet, table="catspricefeed")
runz_triplet = partial(creditext_triplet, table="bbgRunzData")
idc_triplet = partial(creditext_triplet, table="externalBondQuote")
evb_triplet = partial(creditext_triplet, table="marketDataEnhanced")
quoteevent_triplet = partial(_construct_gateway_triplet, schema="marketdata", table="quoteevent")


# ====================================================================
## PORTFOLIOS
# ====================================================================

async def _kdb_portfolio(region, cols="", dates=None, filters=None, lazy=True, *, date_kwargs=None):
    try:
        q = build_pt_query(portfolio_triplet(region), cols=cols, dates=dates, filters=filters, date_kwargs=date_kwargs)
        pt = await query_kdb(q, config=fconn(GATEWAY, dbtype="prod", weave=False), name=f"portfolioToolRfqs", lazy=lazy)
    except Exception as e:
        timeline = "realtime" if is_today(dates, utc=True) else "historical"
        config = fconn(PORTFOLIO, region=region, weave=False, timeline=timeline, strict=True)
        q = build_pt_query('portfolioToolRfqs', _remove_date(cols), dates, date_kwargs={"return_today": False}, filters=filters)
        pt = await query_kdb(q, config=config, name=f"portfolioToolRfqs", lazy=lazy)
    return pt


async def raw_kdb_portfolio(rfq_list_id, region="US", dates=None):
    cols = """date,time,sym,rfqMsgType,rfqActionStr,rfqAon,rfqB0DealerValueType,rfqB0DealerQty,rfqB0Desc,rfqB0InstrumentId,rfqB0InstrumentIdType,rfqB0Price,rfqB1DealerValueType,rfqB1Desc,rfqB1InstrumentId,rfqB1InstrumentIdType,rfqB1Price,rfqBenchmarkSD,rfqBondSD,rfqClosed,rfqConsolidatedState,normalizedState,rfqCoverPrice,rfqCreateDate,rfqCreateTime,rfqCrossType,rfqCustFirm,rfqCustTierInfo,rfqCustUserInfo,rfqCustUserName,rfqDealComment,rfqDealerActions,rfqDealerTrader,rfqDealerTraderName,rfqDealGoodForTime,rfqEnquiryType,rfqId,rfqInfo,rfqIsCompetitive,rfqL0B0DealerSpread,rfqL0B0OffDays,rfqL0B0SpreadAgainst,rfqL0CompositePrice,rfqL0Coupon,rfqL0Currency,rfqL0CustValue,rfqL0DateSettl,rfqL0DealQty,rfqL0DealValue,rfqL0DealValueTypeStr,rfqL0Desc,rfqL0InstrumentInternalId,rfqL0InstrumentId,rfqL0InstrumentIdType,rfqL0Qty,rfqL0Ticker,rfqL0VerbString,rfqL0Yield,rfqL0Book,rfqL1Currency,rfqL1CustValue,rfqL1DateSettl,rfqL1DealQty,rfqL1DealValue,rfqL1DealValueTypeStr,rfqL1Desc,rfqL1InstrumentId,rfqL1InstrumentIdType,rfqL1Qty,rfqL1VerbString,rfqL1Yield,rfqL1CompositePrice,rfqLastResponder,rfqListDue,rfqListId,rfqListItemCount,rfqListSize,rfqMarketId,rfqMaturityDate,rfqMktCover,rfqNLeg,rfqNumOfDealers,rfqOnTheWire,rfqRecId,rfqSDSId,rfqStatusPrevStr,marketEvent,rfqStatusStr,rfqTimer1,rfqTimer1Label,rfqTimer2,rfqTimer2Label,rfqTimeToAuction,rfqTradeId,rfqType,rfqValidityTime,statusString,action,rfqRespAutoHedgeFlag,rfqRespAutoRespondFlag,rfqRespCalculatedByUser,rfqRespErrorMessage,rfqRespHydrTrader,rfqRespQuoteReqID,rfqRespReplyTo,rfqRespMessageId,rfqRespStandardHeaderECN,rfqRespStandardHeaderMsgID,rfqRespStandardHeaderMsgType,rfqRespStandardHeaderProtocolVersion,rfqRespStandardHeaderSendingTime,rfqRespTraderAction,rfqRespType,rfqRespWarnMessage,algoId,autoExComments,deskType,isPact,isPactSize,sefExecuted,mtfExecuted,autoExOriginator,isBBI,quotingAlgoId,hasAdvisorResponded,partyID,partyName,partyRole,tradingPlatform,hasTradingPermissions,clientRating,rfqInitiator,quotePriceType,refPriceSrc,refPriceTime,refPriceSide,refPriceType"""

    rfq_list_id = ensure_list(rfq_list_id)
    rfq_list_id = [str(x) for x in rfq_list_id]
    pt = await _kdb_portfolio(region, cols, dates, filters={"rfqListId": rfq_list_id}, date_kwargs={'biz_days': dates is None})
    states = pt.group_by(['rfqConsolidatedState']).agg([
        pl.col("rfqCreateDate").first(),
        pl.col("rfqCreateTime").first(),
        pl.col("time").first()
    ])

    _s = await (states.select('rfqConsolidatedState', 'time').sort('time').limit(1).collect_async())
    if (_s is None) or _s.is_empty(): return
    original_state = _s.item(0, 0)

    fill_date = parse_single_date(dates)

    pt = pt.with_columns([
        pl.lit(region).alias("kdbRegion"),
        pl.when(pl.col("rfqCreateTime").is_null()).then(pl.col("time")).otherwise(pl.col("rfqCreateTime")).alias("rfqCreateTime"),
        pl.when(pl.col("rfqCreateDate").is_null()).then(pl.lit(fill_date)).otherwise(pl.col("rfqCreateDate")).alias("rfqCreateDate")
    ])

    constituents = pt.filter([
        pl.col("rfqConsolidatedState")==original_state
    ])

    return constituents, pt, states


async def query_pt_constituents_from_kdb(rfq_list_id, region="US", dates=None):
    '''Retrieve the constituents of a portfolio by rfqListId'''
    r = await raw_kdb_portfolio(rfq_list_id, region, dates)
    if r is not None:
        constituents, pt, states = r
        if constituents is not None:
            return constituents

async def init_rfq_leg(my_pt, dates=None, region="US", clean=True, **kwargs):
    default_today = parse_date(dates)
    description_patterns = {
        " (UREGS)": " (RegS)",
        " (REGS)": " (RegS)",
        "(UREGS)": " (RegS)",
        "(REGS)": " (RegS)",
        "(FLAT)": "",
        "FLAT": "",
        "(BABS)": "",
        "BABS": "",
        "(DEFAULT)": "",
        "DEFAULT": "",
        "CO-CO": "",
        "COCO": "",
        " COCO": "",
        "(CO-CO)": "",
        "(COCO)": "",
        " (COCO)": "",
        " USD": "",
        " EUR": "",
        " GBP": "",
        "()": "",
        "(144A)": "(144a)",
        "144A": "144a",
        "  ": " ",
        " C ": " "
    }

    list_id = my_pt.hyper.to_list('rfqListId', unique=True, drop_nulls=True)
    if len(list_id) > 1:
        await log.warning(f"More than 1 unique rfqListId detected! Seen: {len(list_id)}", rfqListId=list_id[:5])
        list_expr = pl.col('rfqListId')
    elif not list_id:
        region = my_pt.hyper.peek('kdbRegion', default=None)
        date = my_pt.hyper.peek('rfqCreateDate', default=None)
        venue = market_id_maps(my_pt.hyper.peek('rfqMarketId', default="MANUAL"))
        fake_list_id = generate_fake_rfq_list_id(venue=venue, region=region, date=date)
        list_expr = pl.lit(fake_list_id, dtype=pl.String)
    else:
        list_expr = pl.lit(str(list_id[0]), dtype=pl.String)

    _pt = my_pt.lazy().with_row_index(name='_idx').with_columns([
        pl.when(
            pl.col('rfqB0InstrumentId').str.contains(BENCH_REGEX)
        ).then(pl.lit("rfqB0InstrumentId"))
        .when(
            pl.col('rfqB1InstrumentId').str.contains(BENCH_REGEX)
        ).then(pl.lit("rfqB1InstrumentId"))
        .when(
            pl.col('rfqL0InstrumentId').str.contains(BENCH_REGEX)
        ).then(pl.lit("rfqL0InstrumentId"))
        .when(
            pl.col('rfqL1InstrumentId').str.contains(BENCH_REGEX)
        ).then(pl.lit("rfqL1InstrumentId"))
        .otherwise(pl.lit(None, dtype=pl.String)).alias('_benchmarkColumn'),
        pl.col("quotePriceType").str.to_uppercase().alias('quotePriceType')
    ]).with_columns([
        pl.when(
            pl.col('rfqL0InstrumentId').is_not_null() &
            (pl.col('_benchmarkColumn').is_null() | (pl.col('_benchmarkColumn')!='rfqL0InstrumentId'))
        ).then(pl.lit('rfqL0InstrumentId'))
        .when(
            pl.col('rfqL0InstrumentId').is_null() &
            pl.col('rfqL1InstrumentId').is_not_null() &
            (pl.col('_benchmarkColumn').is_null() | (pl.col('_benchmarkColumn')!='rfqL1InstrumentId'))
        ).then(pl.lit('rfqL1InstrumentId'))
        .when(
            pl.col('rfqL0InstrumentId').is_null() &
            pl.col('rfqL1InstrumentId').is_null() &
            pl.col('sym').is_not_null()
        ).then(pl.lit('sym'))
        .when(
            pl.col('rfqL0InstrumentId').is_not_null() &
            pl.col('rfqL1InstrumentId').is_null() &
            (pl.col('_benchmarkColumn').is_not_null() | (pl.col('_benchmarkColumn')=='rfqL0InstrumentId'))
        ).then(pl.lit('rfqL0InstrumentId'))
        .when(
            pl.col('rfqL0InstrumentId').is_null() &
            pl.col('rfqL1InstrumentId').is_not_null() &
            (pl.col('_benchmarkColumn').is_not_null() | (pl.col('_benchmarkColumn')=='rfqL1InstrumentId'))
        ).then(pl.lit('rfqL1InstrumentId'))
        .when(
            pl.col('sym').is_not_null()
        ).then(pl.lit('sym'))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias('_cashColumn')
    ]).with_columns([
        pl.when(pl.col('_benchmarkColumn')==pl.col('_cashColumn'))
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col('_benchmarkColumn'))
        .alias('_benchmarkColumn'),
        pl.when(pl.col('_cashColumn').is_null()).then(pl.lit(0))
        .when(pl.col('_cashColumn')=='rfqL0InstrumentId').then(pl.lit(0))
        .when(pl.col('_cashColumn')=='rfqL1InstrumentId').then(pl.lit(1))
        .otherwise(pl.lit(0)).alias('rfqLeg')
    ]).hyper.from_columns(['_cashColumn', '_benchmarkColumn'], ['id', 'rfqBenchmark']).with_columns([
        pl.col('id').cast(pl.String, strict=False),
        pl.col('rfqBenchmark').cast(pl.String, strict=False)
    ]).with_columns([
        pl.when(pl.col('rfqLeg')==0).then(pl.col('rfqL0VerbString')).otherwise(pl.col('rfqL1VerbString')).alias('side'),
        pl.when(pl.col('rfqLeg')==0).then(pl.col('rfqL0DealQty')).otherwise(pl.col('rfqL1DealQty')).abs().alias('size'),
        pl.when(
            pl.col("quotePriceType").is_in(SPREAD_TYPES)).then(pl.lit("SPD"))
        .when(pl.col("quotePriceType").is_in(PRICE_TYPES)).then(pl.lit("PX"))
        .when(pl.col("quotePriceType").is_in(MMY_TYPES)).then(pl.lit("MMY"))
        .when(pl.col("quotePriceType").is_in(DM_TYPES)).then(pl.lit("DM"))
        .when(pl.col("quotePriceType").str.contains('PRICE')).then(pl.lit('PX'))
        .when(pl.col("quotePriceType").str.contains('PX')).then(pl.lit('PX'))
        .when(pl.col("quotePriceType").str.contains('SP')).then(pl.lit('SPREAD'))
        .when(pl.col("quotePriceType").str.contains('Y')).then(pl.lit('MMY'))
        .otherwise(pl.lit("PX")).alias('quoteType'),
        pl.when(pl.col('rfqLeg')==0).then(pl.col('rfqL0DateSettl')).otherwise(pl.col('rfqL1DateSettl')).alias('settleDate'),
        pl.when(
            pl.col('rfqId').is_null() | (pl.col('rfqId')=='')
        ).then(
            pl.concat_str(pl.lit('u', dtype=pl.String), pl.col('_idx').cast(pl.String, strict=False))
        ).otherwise(
            pl.col("rfqId").cast(pl.Utf8).str.split(by=".").list.get(-2, null_on_oob=True).str.split(by="_").list.get(-1, null_on_oob=True).fill_null(pl.col('rfqId'))
        ).alias('tnum'),
        pl.when(pl.col('rfqLeg')==0).then(pl.col('rfqL0Desc')).otherwise(pl.col('rfqL1Desc')).cast(pl.String, strict=False).str.to_uppercase().alias('description'),
        pl.when(pl.col('rfqLeg')==0).then(pl.col('rfqL0Ticker')).otherwise(pl.lit(None, dtype=pl.String)).str.to_uppercase().alias('ticker'),
        pl.when(pl.col('rfqLeg')==0).then(pl.col('rfqL0Coupon').cast(pl.Float64, strict=False)).otherwise(pl.lit(None, dtype=pl.Float64)).alias('coupon'),
        pl.when(pl.col('rfqLeg')==0).then(pl.col('rfqL0Currency')).otherwise(pl.col('rfqL1Currency')).cast(pl.String, strict=False).alias('currency'),
        pl.col('rfqMaturityDate').cast(pl.Date, strict=False).alias('maturityDate'),
        pl.when(pl.col('id').cast(pl.Utf8).str.len_chars()==12).then(pl.lit('ISIN')).otherwise(pl.lit("CUSIP")).alias('idType'),
        pl.when(pl.col('rfqBenchmark').is_null()).then(pl.lit(None))
        .when(pl.col('rfqBenchmark').cast(pl.Utf8).str.len_chars()==12)
        .then(pl.lit('ISIN'))
        .otherwise(pl.lit("CUSIP"))
        .alias('rfqBenchmarkIdType'),
        list_expr.alias('rfqListId'),
        pl.when(pl.col('rfqCreateDate').is_not_null()).then(pl.col('rfqCreateDate').cast(pl.Date, strict=False)).otherwise(
            pl.lit(default_today, dtype=pl.Date)
        ).alias('rfqCreateDate')
    ]).with_columns([
        pl.when(pl.col('idType')=='ISIN')
        .then(pl.col('id'))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias('isin'),
        pl.when(pl.col('idType')=='CUSIP')
        .then(pl.col('id'))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias('cusip'),
        pl.when(pl.col('rfqBenchmarkIdType')=='ISIN').then(pl.col('rfqBenchmark')).otherwise(pl.lit(None, dtype=pl.String)).alias('rfqBenchmarkIsin'),
        pl.coalesce([
            pl.when(pl.col('rfqBenchmarkIdType')=='CUSIP').then(pl.col('rfqBenchmark')),
            pl.when(pl.col('rfqBenchmarkIdType')=='ISIN').then(pl.col('rfqBenchmark').str.slice(2, 9))
        ]).alias('rfqBenchmarkCusip'),
        pl.when(
            pl.col('description').is_null() &
            pl.col('ticker').is_not_null() &
            pl.col('coupon').is_not_null() &
            pl.col('maturityDate').is_not_null()
        ).then(pl.concat_str([
            pl.col('ticker'),
            pl.col('coupon').round(3),
            pl.col('maturityDate').dt.strftime("%m/%d/%Y")
        ], separator=" ")).otherwise(pl.col('description')).alias('description'),
        pl.when(
            pl.col('ticker').is_null() &
            pl.col('description').is_not_null()
        ).then(pl.col('description').str.extract(r"^([a-zA-Z]+)", 1)).otherwise(pl.col('ticker')).alias('ticker'),
        pl.col('description').str.extract(r" ([cC]{1} ?([0-9]{1,4}|[ ]))", 0).alias('_descriptionStub')
    ]).with_columns([

        pl.when(pl.col('_descriptionStub').is_not_null()).then(
            pl.col('description').str.split(pl.col('_descriptionStub')).list.join("")
        ).otherwise(pl.col('description')).alias('description'),

        pl.when(pl.col('description').is_not_null() & pl.col('description').str.contains("FLAT", literal=True, strict=False))
        .then(FLAG_YES).otherwise(FLAG_NO).alias('isFlat'),

        pl.when(pl.col('description').is_not_null() & (
                pl.col('description').str.contains("COCO", literal=True, strict=False) |
                pl.col('description').str.contains("CO-CO", literal=True, strict=False)
        )).then(FLAG_YES).otherwise(FLAG_NO).alias('isCoco'),

        pl.when(pl.col('description').is_not_null() & (
            pl.col('description').str.contains("BABS", literal=True, strict=False)
        )).then(FLAG_YES).otherwise(FLAG_NO).alias('isBabs')

    ]).with_columns([
        pl.col('description').cast(pl.String, strict=False).str.replace_many(
            description_patterns,
            ascii_case_insensitive=True
        ).str.strip_chars().alias('description'),
        pl.when(pl.col('maturityDate').is_not_null() & pl.col('maturityDate').dt.year()==2099).then(pl.lit(None, dtype=pl.Date)).otherwise(pl.col('maturityDate')).alias('maturityDate')
    ]).drop(['_idx', '_cashColumn', '_benchmarkColumn', '_descriptionStub'], strict=False)

    return _pt.select(
        [
            "tnum", "id", 'isin', 'cusip', 'idType',
            "side", "size", "quoteType", "settleDate",
            "rfqLeg", "rfqListId", "rfqId", "rfqBenchmark",
            "rfqBenchmarkIsin", "rfqBenchmarkCusip", "description",
            "ticker", "coupon", "maturityDate", "currency",
            "rfqCreateDate",
            "isFlat", "isCoco", "isBabs"
        ]) if clean else _pt


# ====================================================================
## DEBUG
# ====================================================================

def _ensure_column(my_pt, col, fill=None):
    s = my_pt.collect_schema().keys()
    if col not in s:
        return my_pt.with_columns([
            pl.lit(fill).alias(col)
        ])
    return my_pt.with_columns(pl.col(col).fill_null(fill))


async def random_rfq_list_id(region="US", dates=None, lookback=5):
    '''Select a random rfqListId from today - lookback days'''
    pt = await _kdb_portfolio(region, cols='distinct rfqListId, date', dates=dates, filters={'not rfqListId': ''}, lazy=False)
    if pt.is_empty() and (not is_today(dates, True)):
        pt = await _kdb_portfolio(region, cols='distinct rfqListId, date', dates=[f"T-{lookback}", "T-1"], filters={'not rfqListId': ''}, lazy=False)
    return tuple(pt.sample(n=1, shuffle=True).to_dicts()[0].values())


async def random_pt_and_date(region="US", dates=None, lookback=5, auto_format=True):
    rfqListId, dates = await random_rfq_list_id(region, dates, lookback)
    constituents = await query_pt_constituents_from_kdb(rfqListId, region=region, dates=dates)
    if auto_format: constituents = await init_rfq_leg(constituents)
    return constituents, dates


async def random_pt(region="US", dates=None, lookback=5, auto_format=True):
    constituents, dates = await random_pt_and_date(region, dates, lookback, auto_format)
    return constituents


# ====================================================================
## MISC / LITERALS / FORMATTING
# ====================================================================

async def init_values(my_pt, **kwargs):
    my_pt = ensure_lazy(my_pt)
    schema = my_pt.hyper.schema()
    force_fake_key = kwargs.pop('force_fake_key', False)

    ## Primary Key setup
    rfq_list_id = None
    if (not force_fake_key) and ("rfqListId" in schema):
        rfq_list_id = str(my_pt.hyper.peek("rfqListId", default=None))
    if rfq_list_id is None:
        rfq_list_id = str(generate_fake_rfq_list_id())

    portfolio_key = None
    if 'portfolioKey' in schema:
        portfolio_key = my_pt.hyper.peek(col='portfolioKey', default=None)
    if portfolio_key is None:
        portfolio_key = generate_portfolio_key(rfq_list_id)
    portfolio_key = str(portfolio_key).lower()

    # Ensure unique tnum
    if "tnum" not in schema:
        my_pt = await my_pt.with_row_index(name="tnum").with_columns([
            pl.col('tnum').cast(pl.String, strict=False).alias('tnum')
        ])
    else:
        my_pt = my_pt.with_columns([
            pl.col('tnum').cast(pl.String, strict=False).alias('tnum')
        ])

    tnum = my_pt.hyper.to_list('tnum', unique=True, drop_nulls=True)
    if len(tnum)!=my_pt.hyper.height():
        await log.warning("TNUMS are not unique. Appending index.")
        my_pt = await my_pt.with_row_index(name="index").with_columns([
            pl.col('tnum').cast(pl.String, strict=False).fill_null("").alias('tnum')
        ]).with_columns([
            pl.concat_str([pl.col("tnum"), pl.col("index")], separator="_").alias("tnum")
        ]).drop("index")

    if not "rfqId" in schema:
        my_pt = my_pt.with_columns([pl.col("tnum").alias("rfqId")])

    return my_pt.select([
        pl.lit(portfolio_key, pl.String).alias('portfolioKey'),
        pl.col("tnum").cast(pl.String, strict=False).alias('tnum'),
        pl.col("rfqId").cast(pl.String, strict=False).alias('rfqId'),
        pl.col("id").cast(pl.String, strict=False).alias('originalId'),
        pl.col("side").cast(pl.String, strict=False).alias('originalSide'),
        pl.col("size").cast(pl.Float64, strict=False).alias('originalSize'),
        pl.col("quoteType").cast(pl.String, strict=False).alias('originalQuoteType'),
        pl.col("size").cast(pl.Float64, strict=False).abs().alias('grossSize'),
        (pl.col("size").abs() * (pl.when(pl.col("side")=='SELL').then(-1).otherwise(1))).alias('netSize'),
        (pl.col("size").abs() * (pl.when(pl.col("side")=='BUY').then(1).otherwise(0))).alias('bidSize'),
        (pl.col("size").abs() * (pl.when(pl.col("side")=='SELL').then(1).otherwise(0))).alias('askSize'),
        pl.lit(None, pl.String).alias('comment'),

        # Store the manual level - and results from S3
        pl.lit(None, pl.Float64).alias('manualBidPx'),
        pl.lit(None, pl.Float64).alias('manualMidPx'),
        pl.lit(None, pl.Float64).alias('manualAskPx'),
        pl.lit(None, pl.Float64).alias('manualBidSpd'),
        pl.lit(None, pl.Float64).alias('manualMidSpd'),
        pl.lit(None, pl.Float64).alias('manualAskSpd'),
        pl.lit(None, pl.Float64).alias('manualBidMmy'),
        pl.lit(None, pl.Float64).alias('manualMidMmy'),
        pl.lit(None, pl.Float64).alias('manualAskMmy'),
        pl.lit(None, pl.Float64).alias('manualBidDm'),
        pl.lit(None, pl.Float64).alias('manualMidDm'),
        pl.lit(None, pl.Float64).alias('manualAskDm'),

        pl.lit(0, pl.Int8).alias('manualRefMktOverride'),  # Are we overriding or simplying filling in a gap?
        pl.lit(None, pl.String).alias('manualRefMktUser'),
        pl.lit(None, pl.Datetime).alias('manualRefreshTime'),

        pl.lit(0, pl.Int8).alias('isMarked'),
        pl.lit(0, pl.Int8).alias('isPendingHedges'),
        pl.lit(0, pl.Int8).alias('isLocked'),
        pl.lit(1, pl.Int8).alias('isReal'),
    ])


async def init_pricing(my_pt, **kwargs):
    return my_pt.select(

        tnum=pl.col("tnum"),
        skewType=pl.lit(0, pl.Int8),  # 0 = fixed, 1 = relative
        relativeSkewValue=pl.lit(0, pl.Float64),  # relative skew value
        relativeSkewTargetMkt=pl.lit(None, pl.String),  # Market name
        relativeSkewTargetSide=pl.lit(None, pl.String),  # Side of market - bid/mid/ask
        relativeSkewTargetQuoteType=pl.lit(None, pl.String),  # Base QT

        newLevel=pl.lit(None, pl.Float64),  # FINAL, client requested, submission
        newLevelPx=pl.lit(None, pl.Float64),  # $px
        newLevelSpd=pl.lit(None, pl.Float64),  # Spread
        newLevelYld=pl.lit(None, pl.Float64),  # Yield (to convention)
        newLevelMmy=pl.lit(None, pl.Float64),  # MMY
        newLevelDm=pl.lit(None, pl.Float64),  # DM

        # Truly the last edit
        lastEditUser=pl.lit(None, pl.String),  # Username of editor
        lastEditTime=pl.lit(None, pl.String),  # timestamp
        lastEditSource=pl.lit(None, pl.String),  # Direct, bulk, etc
        lastEditQuoteType=pl.lit(None, pl.String),  # QT of entry for audit
        lastComputeTimestamp=pl.lit(None, pl.String),  # Recalc for audit
        lastEditTraceId=pl.lit(None, pl.String),  # Actual message trace id

        # Last Admin touch
        lastAdminEditUser=pl.lit(None, pl.String),
        lastAdminEditTimestamp=pl.lit(None, pl.String),
        lastAdminEditNewLevel=pl.lit(None, pl.String),

        # Last Non-Admin touch
        lastTraderEditUser=pl.lit(None, pl.String),
        lastTraderEditTimestamp=pl.lit(None, pl.String),
        lastTraderEditNewLevel=pl.lit(None, pl.String),

    )


@hypercache.cached(ttl=timedelta(hours=2), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'dates'])
async def settle_dater(my_pt, region="US", dates=None, **kwargs):
    t1 = next_biz_date(dates, 1)
    t2 = next_biz_date(dates, 2)
    tn = partial(next_biz_date, dates)
    s = my_pt.hyper.schema()
    exprs = []
    if 'daysToSettle' not in s:
        exprs.append(pl.lit(None, pl.Int64).alias('daysToSettle'))
    return my_pt.with_columns(exprs).select([
        pl.col('isin'),
        pl.when(pl.col('daysToSettle').is_null())
        .then(
            pl.when(pl.col('isRegS').is_null() & pl.col('isRule144A').is_null())
            .then(pl.lit(t1, pl.Date))
            .when(pl.col('isRegS').is_not_null() & (pl.col('isRegS').cast(pl.Int8, strict=False)==1))
            .then(pl.lit(t2, pl.Date))
            .otherwise(pl.lit(t1, pl.Date))
        ).otherwise(
            pl.when(pl.col('daysToSettle')==1)
            .then(pl.lit(t1, pl.Date))
            .when(pl.col('daysToSettle')==2)
            .then(pl.lit(t2, pl.Date))
            .otherwise(pl.col('daysToSettle').cast(pl.Int64, strict=False).abs().map_elements(tn, return_dtype=pl.Date))
        )
        .alias('settleDate')
    ]).with_columns(pl.col('settleDate').alias('standardSettleDate'))


RATING_MAP = {**{None: None, 'NR': None, 'NA': None}, **SP_RATINGS, **MOODYS_RATINGS, **FITCH_RATINGS}
RATING_MAP_SIMPLE = {k: ('HY' if v=='Distressed' else v) for k, v in RATING_MAP.items()}


async def ref_data_transforms(my_pt, region="US", dates=None, **kwargs):
    _today = pl.lit(get_today(utc=True), pl.Date)
    flag_cols = my_pt.hyper.cols_like("^is[A-Z]+")
    flags = [pl.col(col).cast(pl.Int8, strict=False).fill_null(FLAG_NO).alias(col) for col in flag_cols]

    return my_pt.with_columns(flags + [
        pl.col('bondType').cast(pl.String, strict=False).str.to_uppercase().alias('bondType'),
        pl.col("ratingCombined").cast(pl.String, strict=False).str.strip_chars(),
        pl.col("ratingSandP").cast(pl.String, strict=False).str.strip_chars(),
        pl.col("ratingMoody").cast(pl.String, strict=False).str.strip_chars(),
        pl.col("ratingFitch").cast(pl.String, strict=False).str.strip_chars(),
        pl.col("ticker").cast(pl.String, strict=False).str.strip_chars(),
        pl.col("description").cast(pl.String, strict=False).str.strip_chars(),
        pl.col("shortDescription").cast(pl.String, strict=False).str.strip_chars(),
        pl.col("issuerName").cast(pl.String, strict=False).str.strip_chars(),
        pl.col("industrySector").cast(pl.String, strict=False).str.strip_chars(),
        pl.when(pl.len().over("isin") > 1).then(FLAG_YES).otherwise(FLAG_NO).alias('isDuplicated'),
    ]).with_columns([

        # Ratings
        pl.when(pl.col("ratingCombined")=="").then(pl.lit(None, pl.String)).otherwise(pl.col("ratingCombined")).alias("ratingCombined"),
        pl.when(pl.col("ratingSandP")=="").then(pl.lit(None, pl.String)).otherwise(pl.col("ratingSandP")).alias("ratingSandP"),
        pl.when(pl.col("ratingMoody")=="").then(pl.lit(None, pl.String)).otherwise(pl.col("ratingMoody")).alias("ratingMoody"),
        pl.when(pl.col("ratingFitch")=="").then(pl.lit(None, pl.String)).otherwise(pl.col("ratingFitch")).alias("ratingFitch"),

        # Tickers
        pl.when(pl.col("ticker")=="").then(pl.lit(None, pl.String)).otherwise(pl.col("ticker")).alias("ticker"),
        pl.when(pl.col("description")=="").then(pl.lit(None, pl.String)).otherwise(pl.col("description")).alias("description"),
        pl.when(pl.col("shortDescription")=="").then(pl.lit(None, pl.String)).otherwise(pl.col("shortDescription")).alias("shortDescription"),
        pl.when(pl.col("issuerName")=="").then(pl.lit(None, pl.String)).otherwise(pl.col("issuerName")).alias("issuerName"),
    ]).select([
        pl.col('tnum'),

        # Flags
        pl.col('isDuplicated'),
        pl.when(pl.col('bondType')=='HYBRID').then(FLAG_YES).otherwise(FLAG_NO).alias('isHybrid'),
        pl.when(pl.col('bondType')=='CONVERTIBLE').then(FLAG_YES).otherwise(FLAG_NO).alias('isConvertible'),
        pl.when(pl.col('bondType')=='YANKEE').then(FLAG_YES).otherwise(FLAG_NO).alias('isYankee'),
        pl.when(pl.col('bondType')=='CATASTROPHE').then(FLAG_YES).otherwise(FLAG_NO).alias('isCatastrophe'),
        pl.when(pl.col('bondType')=='ASSET BACKED').then(FLAG_YES).otherwise(FLAG_NO).alias('isAssetBacked'),
        pl.when((pl.col('maturityType')=='PERP') | (pl.col('description').is_not_null() & (
                pl.col('description').str.to_uppercase().str.contains('PERP', strict=False, literal=True) | pl.col('description').str.ends_with(" P")))).then(FLAG_YES).otherwise(
            pl.col('isPerpetual')).alias('isPerpetual'),
        pl.when((pl.col('couponType')=='VAR') | (pl.col('isInflationLinked')==1)).then(FLAG_YES).otherwise(FLAG_NO).alias('isVariable'),
        pl.when(pl.col('description').is_not_null() & pl.col('description').str.to_uppercase().str.contains('FLOAT', strict=False, literal=True)).then(FLAG_YES).otherwise(pl.col('isFloater')).alias(
            'isFloater'),

        pl.when(pl.col("ticker").is_not_null()).then(pl.col("ticker")).otherwise(
            pl.coalesce([pl.col("description"), pl.col('shortDescription'), pl.col('issuerName')]).str.split(" ").list.get(0)
        ).alias("ticker"),

        pl.when(pl.len().over("description") > 1).then(FLAG_YES).otherwise(FLAG_NO).alias('isDuplicatedDescription'),

        # Products
        pl.when(pl.col("industrySector")=="GOVN").then(pl.lit("SOV", pl.String))
        .when(pl.col("industrySector")=="MUNI").then(pl.lit("MUNI", pl.String))
        .when(pl.col("issuerSector")=="GOVERNMENT_RELATED").then(pl.lit('QUASI', pl.String))
        .otherwise(pl.lit('CORP')).alias('emProductType'),

        ((pl.when(pl.col("isPerpetual")==1)
          .then(pl.coalesce([pl.col("pseudoWorkoutDate"), pl.col("maturityDate")]))
          .otherwise(pl.col("maturityDate"))
          .cast(pl.Date, strict=False) - _today
          ).dt.total_days()).cast(pl.Float64, strict=False).alias("daysToMaturity"),

        ((_today - pl.col('issueDate').cast(pl.Date, strict=False)).dt.total_days() / 365.25).alias("yrsSinceIssuance"),

        pl.coalesce([
            pl.col("ratingCombined"),
            pl.col("ratingSandP"),
            pl.col("ratingMoody"),
            pl.col("ratingFitch")
        ]).replace(MOODY_TO_SP).alias("ratingCombined"),

    ]).with_columns([
        (pl.col('daysToMaturity') / 365.25).cast(pl.Float64, strict=False).alias("yrsToMaturity"),
    ]).with_columns([

        pl.col('yrsToMaturity').floor().cast(pl.Int64, strict=False).cast(pl.String, strict=False).replace_strict(YIELD_CURVE_MAP, default="Long-end", return_dtype=pl.String).alias(
            "yieldCurvePosition"),

        pl.when(pl.col('yrsSinceIssuance') < 0.03).then(FLAG_YES).otherwise(FLAG_NO).alias("isNewIssue"),
        pl.col('ratingCombined').replace(RATING_MAP).alias("ratingAssetClass"),
        pl.col('ratingCombined').replace(SP_TO_MNEMONIC).alias("ratingMnemonic"),

        pl.when(pl.col("yrsToMaturity") <= 2.5).then(pl.lit("2YR", pl.String))
        .when(pl.col("yrsToMaturity") <= 4.0).then(pl.lit("3YR", pl.String))
        .when(pl.col("yrsToMaturity") <= 6.0).then(pl.lit("5YR", pl.String))
        .when(pl.col("yrsToMaturity") <= 8.5).then(pl.lit("7YR", pl.String))
        .when(pl.col("yrsToMaturity") <= 15.0).then(pl.lit("10YR", pl.String))
        .when(pl.col("yrsToMaturity") <= 25.0).then(pl.lit("20YR", pl.String))
        .otherwise(pl.lit("30YR")).alias("maturityBucket"),

    ])


# ====================================================================
## Maps & Helpers
# ====================================================================

def yield_curve_position(maturity):
    if maturity <= 4:
        return 'Front-end'
    elif maturity <= 10:
        return 'Intermediate'
    else:
        return 'Long-end'


YIELD_CURVE_MAP = {
    None: None,
    '0' : 'Front-end',
    '1' : 'Front-end',
    '2' : 'Front-end',
    '3' : 'Front-end',
    '4' : 'Front-end',
    '5' : 'Intermediate',
    '6' : 'Intermediate',
    '7' : 'Intermediate',
    '8' : 'Intermediate',
    '9' : 'Intermediate',
    '10': 'Intermediate',
    # 'Long-end'
}

# ====================================================================
## RISK
# ====================================================================

YIELD_PRIORITY = (
    "bvalMid",
    "houseMid",
    "usHouseMid",
    "euHouseMid",
    "macpMid",
    "mlcrMid",
    "markitMid",
    "cbbtMid",
    "idcMid",
    "bvalNy4PmMid",
    "bvalNy3PmMid",
    "bvalLo415PmMid",
    "bvalLo3PmMid",
    "bvalLo12PmMid",
    "bvalTo5PmMid",
    "bvalSh5PmMid",
    "bvalTo4PmMid",
    "bvalTo3PmMid",
    "sgpHouseMid",
)

YIELD_PRIORITY_DICT = {k: i for i, k in enumerate(YIELD_PRIORITY)}
_CUMU_DAYS_NORM = np.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.int32)
_CUMU_DAYS_LEAP = np.array([0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335], dtype=np.int32)
_EPOCH_ORD = 719_163
_INF_ORD = np.iinfo(np.int64).max


def _normalize_day_count(day_count) -> tuple[str, bool]:
    if day_count is None: return "", False
    s = str(day_count).strip().upper()
    if s in ("", "NONE", "NULL"): return "", False
    s = s.replace("_", " ").replace("-", " ").replace(":", " ").replace("\\", "/")
    s = " ".join(s.split())
    n_eom = ("N EOM" in s) or ("N-EOM" in s) or ("N/EOM" in s)
    s = s.replace("N EOM", "").replace("N-EOM", "").replace("N/EOM", "")
    s = " ".join(s.split())
    if s in ("ACT 360", "ACTUAL 360", "ACTUAL/360"):
        s = "ACT/360"
    elif s in ("ACT 365", "ACTUAL 365", "ACTUAL/365", "ACT/365F", "ACT 365F", "ACTUAL/365F"):
        s = "ACT/365"
    elif s in ("ACT ACT", "ACTUAL ACTUAL", "ACTUAL/ACTUAL", "ACTUAL/ACT"):
        s = "ACT/ACT"
    elif s in ("30 360", "30U/360", "BOND BASIS", "US 30/360", "NASD 30/360"):
        s = "30/360"
    elif s in ("30E 360", "30E/360", "EURO 30/360", "30/360E", "30/360 EURO"):
        s = "30E/360"
    if "GERMAN" in s and "30/360" in s:
        s = "GERMAN 30/360"
    elif "ISDA" in s and "ACT/ACT" in s:
        s = "ISDA ACT/ACT"
    elif "ISDA" in s and "30" in s and "360" in s:
        if "30E" in s or "30E/360" in s:
            s = "ISDA 30E/360"
        else:
            s = "ISDA SWAPS 30/360"
    elif "ISMA" in s and "30/360" in s:
        s = "ISMA 30/360"
    return s, n_eom


def _np_is_leap(y): return ((y % 4==0) & (y % 100!=0)) | (y % 400==0)


def _np_days_in_year(y): return np.where(_np_is_leap(y), 366, 365)


def _np_is_eom(y, m, d): return d==_np_days_in_month(y, m)


_DAYS_IN_MONTH_BASE = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int32)


def _np_days_in_month(y, m):
    d = _DAYS_IN_MONTH_BASE[m]
    return np.where((m==2) & _np_is_leap(y), 29, d)


def _jan1_ordinal(y):
    y1 = y - 1
    return 365 * y1 + y1 // 4 - y1 // 100 + y1 // 400 + 1


def _ordinal_to_ymd(o):
    a = o - 1
    y_raw = (10000 * a + 14780) // 3652425
    doy = a - (365 * y_raw + y_raw // 4 - y_raw // 100 + y_raw // 400)
    adj = doy < 0
    y_raw = np.where(adj, y_raw - 1, y_raw)
    doy = np.where(adj, a - (365 * y_raw + y_raw // 4 - y_raw // 100 + y_raw // 400), doy)
    mi = (100 * doy + 52) // 3060
    month = np.where(mi < 10, mi + 3, mi - 9).astype(np.int32)
    year = (y_raw + np.where(month <= 2, 1, 0)).astype(np.int32)
    day = (doy - (mi * 306 + 5) // 10 + 1).astype(np.int32)
    return year, month, day


def _ymd_to_ordinal_vec(y, m, d):
    y = y.astype(np.int32)
    m = m.astype(np.int32)
    d = d.astype(np.int32)
    leap = _np_is_leap(y)
    dim = _np_days_in_month(y, m).astype(np.int32)
    d_clamped = np.minimum(np.maximum(d, 1), dim)
    cum = np.where(leap, _CUMU_DAYS_LEAP[m], _CUMU_DAYS_NORM[m]).astype(np.int32)
    doy = (cum + d_clamped).astype(np.int32)
    return (_jan1_ordinal(y).astype(np.int64) + (doy.astype(np.int64) - 1)).astype(np.int64)


def _estimate_next_coupon_ord_vec(cur_ord, cur_y, cur_m, cur_d, mat_ord, freq_arr):
    n = mat_ord.shape[0]
    cur_ord_arr = np.full(n, cur_ord, dtype=np.int64)
    mat_y, mat_m, mat_d = _ordinal_to_ymd(mat_ord)
    freq = freq_arr.astype(np.int32)
    ok_freq = (freq > 0) & ((12 % freq)==0)
    step = np.where(ok_freq, (12 // freq), 0).astype(np.int32)
    max_k = 12
    best = np.full(n, _INF_ORD, dtype=np.int64)
    cy = np.full(n, cur_y, dtype=np.int32)
    cm = np.full(n, cur_m, dtype=np.int32)
    cd = np.full(n, cur_d, dtype=np.int32)
    cur_ord_vec = cur_ord_arr
    for j in range(max_k):
        use_j = ok_freq & (j < freq)
        if not use_j.any():
            continue
        mj = ((mat_m - (j * step) - 1) % 12) + 1  # 1..12
        y0 = cy
        y1 = cy + 1
        ord0 = _ymd_to_ordinal_vec(y0, mj, mat_d)
        ord1 = _ymd_to_ordinal_vec(y1, mj, mat_d)
        cand = np.where(ord0 > cur_ord_vec, ord0, ord1)
        cand = np.where(use_j, cand, _INF_ORD)
        best = np.minimum(best, cand)

    return best


def _yf_act_act(diff, sy, ey, s_ord, e_ord):
    diy = _np_days_in_year(sy).astype(np.float64)
    same = sy==ey
    result = np.where(same, diff / diy, 0.0)
    cross = ~same
    if cross.any():
        cs_y, ce_y = sy[cross], ey[cross]
        cs_o, ce_o = s_ord[cross], e_ord[cross]
        j1n = _jan1_ordinal(cs_y + 1)
        j1e = _jan1_ordinal(ce_y)
        pf = (j1n - cs_o).astype(np.float64) / _np_days_in_year(cs_y).astype(np.float64)
        wh = np.maximum((ce_y - cs_y - 1).astype(np.float64), 0.0)
        pl_ = (ce_o - j1e).astype(np.float64) / _np_days_in_year(ce_y).astype(np.float64)
        result[cross] = pf + wh + pl_
    return result


def _dc_30_360_us(sy, sm, sd, ey, em, ed, suppress_eom):
    dd1 = sd.astype(np.int32).copy()
    dd2 = ed.astype(np.int32).copy()
    if not suppress_eom:
        m1 = (sm==2) & _np_is_eom(sy, sm, sd)
        dd1 = np.where(m1, 30, dd1)
        m2 = (em==2) & _np_is_eom(ey, em, ed) & ((dd1==30) | (dd1==31))
        dd2 = np.where(m2, 30, dd2)
    dd2 = np.where((dd2==31) & ((dd1==30) | (dd1==31)), 30, dd2)
    dd1 = np.where(dd1==31, 30, dd1)
    return (360 * (ey.astype(np.int32) - sy.astype(np.int32)) + 30 * (em.astype(np.int32) - sm.astype(np.int32)) + (dd2 - dd1)).astype(np.float64)


def _dc_30e_360(sy, sm, sd, ey, em, ed, adjust_eom):
    dd1 = sd.astype(np.int32).copy()
    dd2 = ed.astype(np.int32).copy()
    if adjust_eom:
        dd1 = np.where(_np_is_eom(sy, sm, sd), 30, dd1)
        dd2 = np.where(_np_is_eom(ey, em, ed), 30, dd2)
    else:
        dd1 = np.where(dd1==31, 30, dd1)
        dd2 = np.where(dd2==31, 30, dd2)
    return (360 * (ey.astype(np.int32) - sy.astype(np.int32)) + 30 * (em.astype(np.int32) - sm.astype(np.int32)) + (dd2 - dd1)).astype(np.float64)


def _dc_german_30_360(sy, sm, sd, ey, em, ed, suppress_eom):
    dd1 = sd.astype(np.int32).copy()
    dd2 = ed.astype(np.int32).copy()
    if not suppress_eom:
        dd1 = np.where((sm==2) & _np_is_eom(sy, sm, sd), 30, dd1)
        dd2 = np.where((em==2) & _np_is_eom(ey, em, ed), 30, dd2)
    dd1 = np.where(dd1==31, 30, dd1)
    dd2 = np.where(dd2==31, 30, dd2)
    return (360 * (ey.astype(np.int32) - sy.astype(np.int32)) + 30 * (em.astype(np.int32) - sm.astype(np.int32)) + (dd2 - dd1)).astype(np.float64)


def _year_fraction_vec(s_ord, e_ord, sy, sm, sd, ey, em, ed, code, n_eom):
    diff = (e_ord - s_ord).astype(np.float64)
    if code=="": return diff / 365.25
    if code=="ACT/360": return diff / 360.0
    if code=="ACT/365": return diff / 365.0
    if code in ("ACT/ACT", "ISDA ACT/ACT"): return _yf_act_act(diff, sy, ey, s_ord, e_ord)
    if code in ("30/360", "ISMA 30/360", "ISDA SWAPS 30/360"): return _dc_30_360_us(sy, sm, sd, ey, em, ed, n_eom) / 360.0
    if code in ("30E/360", "ISDA 30E/360"): return _dc_30e_360(sy, sm, sd, ey, em, ed, not n_eom) / 360.0
    if code=="GERMAN 30/360": return _dc_german_30_360(sy, sm, sd, ey, em, ed, n_eom) / 360.0
    return diff / 365.25


def _modified_duration_vec(cr, ytm, years, freq):
    m = freq
    y = ytm / m
    c = cr / m
    N = np.ceil(years * m - 1e-12).astype(np.int64)
    N = np.maximum(N, 1)
    N_f = N.astype(np.float64)

    eps = 1e-10
    opy = 1.0 + y
    pow_n = np.power(opy, N_f)
    denom = c * (pow_n - 1.0) + y

    # Main closed-form Macaulay
    safe_y = np.where(np.abs(y) < eps, 1.0, y)
    safe_d = np.where(np.abs(denom) < 1e-18, 1.0, denom)
    dur_main = ((opy / safe_y) - ((opy + N_f * (c - safe_y)) / safe_d)) / m

    # Zero-yield fallback
    sum_cf = c * N_f + 1.0
    sum_t_cf = c * N_f * (N_f + 1.0) / 2.0 + N_f
    safe_sc = np.where(np.abs(sum_cf) < 1e-18, 1.0, sum_cf)
    dur_zero = (sum_t_cf / safe_sc) / m

    mac = np.where(
        years <= 0.0, 0.0,
        np.where(np.abs(y) < eps, dur_zero,
                 np.where(opy <= 0.0, np.nan,
                          np.where(np.abs(denom) < 1e-18, dur_zero,
                                   dur_main))))

    # Modified duration: D_mac / (1 + y/m)
    y_per = ytm / m
    good = (1.0 + y_per) > 0.0
    safe_den = np.where(good, 1.0 + y_per, 1.0)
    mod = mac / safe_den
    mod = np.where(good, mod, np.nan)
    return np.maximum(mod, 0.0)


def _modified_duration_vec_fractional(cr, ytm, freq, a_periods, N):
    m = freq.astype(np.float64)
    y = ytm / m
    c = cr / m

    # Validity: (1 + y) must be > 0 (per-period discount factor must exist)
    good = (1.0 + y) > 0.0
    safe_m = np.where(m > 0.0, m, 1.0)

    # Clamp inputs
    a = np.maximum(a_periods, 0.0)
    # Most bonds have 0 < a <= 1, but stubs happen; allow >1 rather than exploding.
    N = np.maximum(N.astype(np.int64), 1)

    v = np.where(good, 1.0 / (1.0 + y), 1.0)  # per-period discount factor
    vN = np.power(v, N.astype(np.float64))

    eps = 1e-12
    one_minus_v = 1.0 - v
    small_den = np.abs(one_minus_v) < eps

    # sum_{j=0}^{N-1} v^j
    sum_v = np.where(
        small_den,
        N.astype(np.float64),
        (1.0 - vN) / one_minus_v
    )

    # sum_{j=0}^{N-1} j v^j
    # closed form: v*(1 - N*v^(N-1) + (N-1)*v^N)/(1-v)^2
    vNm1 = np.power(v, (N - 1).astype(np.float64))
    sum_jv = np.where(
        small_den,
        (N.astype(np.float64) - 1.0) * N.astype(np.float64) / 2.0,
        (v * (1.0 - N.astype(np.float64) * vNm1 + (N.astype(np.float64) - 1.0) * vN)) / (one_minus_v * one_minus_v)
    )

    # sum_{j=0}^{N-1} (a + j) v^j = a * sum_v + sum_jv
    sum_av = a * sum_v + sum_jv

    # PV scaling for fractional first period
    va = np.power(v, a)

    # Price per par=1: v^a * [ c * sum_v + v^(N-1) ]
    # (principal arrives with the last coupon at time a + (N-1) periods)
    price = va * (c * sum_v + vNm1)

    # Numerator for Macaulay duration (in periods):
    # v^a * [ c * sum_{j=0}^{N-1} (a+j) v^j + (a + N - 1) v^(N-1) ]
    num = va * (c * sum_av + (a + (N.astype(np.float64) - 1.0)) * vNm1)

    # Avoid division by zero
    safe_price = np.where(np.abs(price) < 1e-18, 1.0, price)
    mac_periods = num / safe_price

    # Convert periods -> years
    mac_years = mac_periods / safe_m

    # Modified duration: D_mac / (1 + y)
    safe_one_plus_y = np.where(good, 1.0 + y, 1.0)
    mod = mac_years / safe_one_plus_y

    # For invalid yields (1+y<=0) return NaN
    mod = np.where(good, mod, np.nan)

    # Duration should not be negative in normal cases
    return np.maximum(mod, 0.0)


def _compute_duration_numpy(
        cur_ord, cur_y, cur_m, cur_d,
        mat_i32, mat_null,
        pseudo_i32, pseudo_null,
        called_i32, called_null,
        next_call_i32, next_call_null,
        next_i32, next_null,
        coupon, is_perp, is_float, is_hybrid, is_variable, freq_arr,
        accrual_codes, code_table,
        ytm, max_maturity_years
):
    n = len(coupon)

    def _to_ord(arr_i32, null_mask):
        if arr_i32 is None:
            return np.full(n, _INF_ORD, dtype=np.int64)
        o = arr_i32.astype(np.int64) + _EPOCH_ORD
        if null_mask is not None:
            o = o.copy()
            o[null_mask] = _INF_ORD
        return o

    cur_ord_arr = np.full(n, cur_ord, dtype=np.int64)

    # Enforce consistency: derive settle Y/M/D from cur_ord (ignore caller-provided Y/M/D)
    cy0, cm0, cd0 = _ordinal_to_ymd(np.array([np.int64(cur_ord)], dtype=np.int64))
    cur_y = int(cy0[0])
    cur_m = int(cm0[0])
    cur_d = int(cd0[0])

    mat_ord = _to_ord(mat_i32, mat_null)
    pseudo_ord = _to_ord(pseudo_i32, pseudo_null)
    called_ord = _to_ord(called_i32, called_null)
    next_call_ord = _to_ord(next_call_i32, next_call_null)
    next_coupon_in = _to_ord(next_i32, next_null)

    freq_f = freq_arr.astype(np.float64)
    freq_f = np.where(freq_f > 0.0, freq_f, 2.0)

    call_ord = np.minimum(called_ord, np.minimum(next_call_ord, pseudo_ord))

    wo_non_perp = np.minimum(mat_ord, call_ord)
    wo_perp = call_ord
    wo = np.where(is_perp, wo_perp, wo_non_perp)

    no_workout = wo==_INF_ORD
    expired = (~no_workout) & (wo <= cur_ord_arr)
    skip = no_workout | expired

    wo_safe = np.where(no_workout, cur_ord_arr, wo)
    wo_y, wo_m, wo_d = _ordinal_to_ymd(wo_safe)

    next_ok = (next_coupon_in!=_INF_ORD) & (next_coupon_in >= cur_ord_arr)
    next_est = _estimate_next_coupon_ord_vec(cur_ord, cur_y, cur_m, cur_d, mat_ord, freq_arr)
    next_coupon = np.where(next_ok, next_coupon_in, next_est)

    next_valid_fixed = (next_coupon!=_INF_ORD) & (next_coupon > cur_ord_arr) & (next_coupon <= wo_safe) & (~skip)

    next_valid_reprice = (next_coupon!=_INF_ORD) & (next_coupon >= cur_ord_arr) & (~skip)
    next_eff_reprice = np.where(next_valid_reprice, np.minimum(next_coupon, wo_safe), wo_safe)

    diff_wo_days = (wo_safe - cur_ord_arr).astype(np.float64)
    yf_wo = np.maximum(diff_wo_days, 0.0) / 365.25

    diff_next_fixed_days = (np.where(next_valid_fixed, next_coupon, cur_ord_arr) - cur_ord_arr).astype(np.float64)
    yf_next_fixed = np.maximum(diff_next_fixed_days, 0.0) / 365.25

    diff_next_to_wo_days = (wo_safe - np.where(next_valid_fixed, next_coupon, wo_safe)).astype(np.float64)
    yf_next_to_wo = np.maximum(diff_next_to_wo_days, 0.0) / 365.25

    diff_next_rep_days = (next_eff_reprice - cur_ord_arr).astype(np.float64)
    yf_next_reprice = np.maximum(diff_next_rep_days, 0.0) / 365.25

    c_y = np.full(n, cur_y, dtype=np.int32)
    c_m = np.full(n, cur_m, dtype=np.int32)
    c_d = np.full(n, cur_d, dtype=np.int32)

    next_safe_fixed = np.where(next_valid_fixed, next_coupon, cur_ord_arr)
    next_y_fixed, next_m_fixed, next_d_fixed = _ordinal_to_ymd(next_safe_fixed)

    next_safe_rep = np.where(skip, cur_ord_arr, next_eff_reprice)
    next_y_rep, next_m_rep, next_d_rep = _ordinal_to_ymd(next_safe_rep)

    for code_idx, (code, n_eom) in enumerate(code_table):
        mask = accrual_codes==code_idx
        if not mask.any():
            continue

        yf_wo[mask] = _year_fraction_vec(
            cur_ord_arr[mask], wo_safe[mask],
            c_y[mask], c_m[mask], c_d[mask],
            wo_y[mask], wo_m[mask], wo_d[mask],
            code, n_eom
        )

        yf_next_fixed[mask] = _year_fraction_vec(
            cur_ord_arr[mask], next_safe_fixed[mask],
            c_y[mask], c_m[mask], c_d[mask],
            next_y_fixed[mask], next_m_fixed[mask], next_d_fixed[mask],
            code, n_eom
        )

        yf_next_to_wo[mask] = _year_fraction_vec(
            next_safe_fixed[mask], wo_safe[mask],
            next_y_fixed[mask], next_m_fixed[mask], next_d_fixed[mask],
            wo_y[mask], wo_m[mask], wo_d[mask],
            code, n_eom
        )

        yf_next_reprice[mask] = _year_fraction_vec(
            cur_ord_arr[mask], next_safe_rep[mask],
            c_y[mask], c_m[mask], c_d[mask],
            next_y_rep[mask], next_m_rep[mask], next_d_rep[mask],
            code, n_eom
        )

    if max_maturity_years is not None:
        lim = float(max_maturity_years)
        np.minimum(yf_wo, lim, out=yf_wo)
        np.minimum(yf_next_to_wo, lim, out=yf_next_to_wo)

    yf_wo[skip] = 0.0
    yf_next_fixed[~next_valid_fixed] = 0.0
    yf_next_to_wo[~next_valid_fixed] = 0.0
    yf_next_reprice[skip] = 0.0

    np.maximum(yf_wo, 0.0, out=yf_wo)
    np.maximum(yf_next_fixed, 0.0, out=yf_next_fixed)
    np.maximum(yf_next_to_wo, 0.0, out=yf_next_to_wo)
    np.maximum(yf_next_reprice, 0.0, out=yf_next_reprice)

    cr = np.where(np.abs(coupon) > 1.0, coupon / 100.0, coupon)

    a_periods = yf_next_fixed * freq_f
    rem_periods = yf_next_to_wo * freq_f
    N = np.floor(rem_periods + 1e-12).astype(np.int64) + 1
    N = np.maximum(N, 1)

    mod_frac = _modified_duration_vec_fractional(cr, ytm, freq_f, a_periods, N)
    mod_rough = _modified_duration_vec(cr, ytm, yf_wo, freq_f)
    fixed_mod = np.where(next_valid_fixed, mod_frac, mod_rough)

    reprice_dur = np.where(next_valid_reprice, yf_next_reprice, 0.0)
    reprice_dur = np.maximum(reprice_dur, 0.0)

    safe_ytm_p = np.where(ytm <= 0.0, 1.0, ytm)
    perp_dur = np.where(ytm <= 0.0, np.inf, 1.0 / safe_ytm_p)

    has_call_horizon = (call_ord!=_INF_ORD) & (call_ord > cur_ord_arr)

    out = np.where(
        is_float,
        reprice_dur,
        np.where(
            is_perp & has_call_horizon,
            fixed_mod,
            np.where(is_perp, perp_dur, fixed_mod)
        )
    )

    out = np.where(skip, 0.0, out)
    return np.maximum(out, 0.0)


@hypercache.cached(ttl=timedelta(hours=6), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'only_missing', 'output_name'])
async def estimate_duration_from_dv01(my_pt, region="US", dates=None, only_missing=True, output_name="duration", **kwargs):
    my_pt = ensure_lazy(my_pt)

    def _nn(c):
        return pl.col(c).is_not_null() | (pl.col(c).cast(pl.Float64, strict=False)!=0)

    def _n(c):
        return pl.col(c).is_null() | (pl.col(c).cast(pl.Float64, strict=False)==0)

    if only_missing:
        missing = my_pt.filter(_nn('unitDv01') & _nn('bvalMidPx') & _n('duration'))
    else:
        missing = my_pt
    if missing.hyper.is_empty(): return
    return missing.select([
        pl.col('isin'),
        (pl.col('unitDv01') / (pl.col('bvalMidPx') * 0.01)).alias(output_name)
    ])


@hypercache.cached(ttl=timedelta(hours=6), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'only_missing', 'output_col'])
async def rough_maturity_to_duration_polars(my_pt, region="US", dates=None, *,
                                            max_maturity_years=None,
                                            default_coupon=None,
                                            default_couponFrequency=2,
                                            output_col="duration",
                                            only_missing=True,
                                            **kwargs
                                            ):
    my_pt = ensure_lazy(my_pt)
    if only_missing:
        my_pt = my_pt.with_columns([
            pl.col('duration').cast(pl.Float64, strict=False).alias('duration')
        ]).filter(pl.col('duration').is_null() | (pl.col('duration')==0))
    if (my_pt is None) or (my_pt.hyper.is_empty()): return
    n = my_pt.hyper.height()
    cols = my_pt.hyper.schema()
    my_date = parse_single_date(dates, biz_days=False, utc=True)
    cur_ord = my_date.toordinal()

    def _date_arrays(col: str):
        from datetime import date as _date
        if col not in cols:
            null = pl.Series([True] * n).cast(pl.Boolean, strict=True).to_numpy()
            arr = pl.Series([_date(1970, 1, 1)] * n).cast(pl.Int32, strict=True).to_numpy()
        else:
            s = my_pt.hyper.to_series(col)
            s = s.cast(pl.Date, strict=False)
            null = s.is_null().to_numpy()
            arr = s.fill_null(_date(1970, 1, 1)).cast(pl.Int32, strict=True).to_numpy()
        return arr, null

    mat_i32, mat_null = _date_arrays("maturityDate")
    pseudo_i32, pseudo_null = _date_arrays("pseudoWorkoutDate")
    called_i32, called_null = _date_arrays("calledDate")
    next_i32, next_null = _date_arrays("nextCouponDate")
    next_call_i32, next_call_null = _date_arrays("nextCallDate")
    if mat_i32 is None: return

    default_coupon = float(default_coupon) if default_coupon is not None else None
    if "coupon" in cols:
        coupon = pl.Series(my_pt.select(pl.col('coupon').cast(pl.Float64, strict=False)).hyper.to_list('coupon'))
        if default_coupon is not None:
            coupon = coupon.fill_null(value=default_coupon)
        coupon = coupon.to_numpy()
    else:
        coupon = np.full(n, default_coupon)

    if "isPerpetual" in cols:
        is_perp = pl.Series(my_pt.hyper.to_list('isPerpetual')).fill_null(0).to_numpy().astype(bool)
    else:
        is_perp = np.zeros(n, dtype=bool)

    if "isFloater" in cols:
        is_float = pl.Series(my_pt.hyper.to_list('isFloater')).fill_null(0).to_numpy().astype(bool)
    else:
        is_float = np.zeros(n, dtype=bool)

    if "isHybrid" in cols:
        is_hybrid = pl.Series(my_pt.hyper.to_list('isHybrid')).fill_null(0).to_numpy().astype(bool)
    else:
        is_hybrid = np.zeros(n, dtype=bool)

    if "isVariable" in cols:
        is_variable = pl.Series(my_pt.hyper.to_list('isVariable')).fill_null(0).to_numpy().astype(bool)
    else:
        is_variable = np.zeros(n, dtype=bool)

    # is_variable = is_hybrid | is_variable

    if "couponFrequency" in cols:
        freq_arr = pl.Series(my_pt.hyper.to_list("couponFrequency")).fill_null(default_couponFrequency).cast(pl.Int64, strict=False).to_numpy()
        freq_arr = np.where(freq_arr <= 0, default_couponFrequency, freq_arr)
    else:
        freq_arr = np.full(n, 2, dtype=np.int64)

    if "accrualMethod" in cols:
        raw_series = pl.Series(my_pt.hyper.to_list("accrualMethod")).fill_null("")
        unique_methods = raw_series.unique().to_list()
        code_table = [_normalize_day_count(m) for m in unique_methods]
        method_to_idx = {m: np.int8(i) for i, m in enumerate(unique_methods)}
        accrual_codes = raw_series.replace_strict(method_to_idx, return_dtype=pl.Int8).to_numpy()
    else:
        code_table = [("", False)]
        accrual_codes = np.zeros(n, dtype=np.int8)

    def _sort_yld(l):
        return sorted(l, key=lambda x: YIELD_PRIORITY_DICT.get(x[:-3], 99))

    yields = (
            _sort_yld(market_columns(my_pt, qt_list=["MidYtw$"])) +
            _sort_yld(market_columns(my_pt, qt_list=["MidYld$"])) +
            _sort_yld(market_columns(my_pt, qt_list=["MidYtm$"])) +
            ['coupon']
    )

    yield_expr = pl.coalesce(yields).cast(pl.Float64, strict=False).alias('_yld_raw')
    yld_raw = pl.Series(my_pt.select(yield_expr).hyper.to_list('_yld_raw')).to_numpy()
    yld_arr = np.where(np.abs(yld_raw) > 1.0, yld_raw / 100.0, yld_raw)

    from app.server import get_threads
    fut = get_threads().submit(
        _compute_duration_numpy,
        cur_ord, my_date.year, my_date.month, my_date.day,
        mat_i32, mat_null,
        pseudo_i32, pseudo_null,
        called_i32, called_null,
        next_call_i32, next_call_null,
        next_i32, next_null,
        coupon, is_perp, is_float, is_hybrid, is_variable, freq_arr,
        accrual_codes, code_table,
        yld_arr, max_maturity_years
    )
    result = await asyncio.wrap_future(fut)
    return my_pt.select([pl.col("isin"), pl.Series(output_col, result, dtype=pl.Float64)]).with_columns([
        pl.when(pl.col(output_col)==0).then(pl.lit(None, pl.Float64)).otherwise(pl.col(output_col)).alias(output_col)
    ])


@hypercache.cached(ttl=timedelta(hours=6), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'missing_only', 'output_name', 'filter_floaters'])
async def interpolated_dv01(my_pt, region="US", dates=None, frames=None, missing_only=True, output_name='unitDv01', filter_floaters=True, **kwargs):
    if missing_only:
        missing = my_pt.filter(pl.col('unitDv01').is_null() & pl.col('duration').is_not_null() & (pl.col('duration')!=0))
    else:
        missing = my_pt

    s = missing.hyper.schema()
    if ('isFloater' in s) and filter_floaters:
        missing = missing.filter(pl.col('isFloater')==0)

    if missing.hyper.is_empty(): return

    bench_frame = frames.get('benchmarks', None) if isinstance(frames, dict) else frames
    if bench_frame is None: return
    bench_frame = bench_frame.filter(pl.col('benchmarkDuration').is_not_null() & pl.col(
        'benchmarkUnitDv01').is_not_null())
    if bench_frame.hyper.is_empty(): return
    return (
        missing.hyper.interpolate('benchmarkDuration', 'benchmarkUnitDv01', 'duration', bench_frame,
                                  out_col=output_name)
        .select([pl.col('isin'), pl.col(output_name)])
    )


@hypercache.cached(ttl=timedelta(hours=6), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt'])
async def approx_dv01(my_pt, region="US", dates=None, missing_only=True, output_name='unitDv01', **kwargs):
    def missing_cond(c):
        return pl.col(c).is_null() | (pl.col(c)==0)

    if missing_only:
        missing_pt = my_pt.filter(missing_cond('unitDv01'))
        if missing_pt.hyper.is_empty(): return
    else:
        missing_pt = my_pt

    prices = missing_pt.hyper.cols_like("MidPx$")
    if not prices: return

    def _sort_px(l):
        return sorted(l, key=lambda x: YIELD_PRIORITY_DICT.get(x[:-2], 99))

    return missing_pt.select([
        pl.col('isin'),
        (pl.col('duration').cast(pl.Float64, strict=False).fill_null(0) * (
                pl.coalesce(_sort_px(prices)).cast(pl.Float64, strict=False).fill_null(0) +
                (pl.col('unitAccrued').cast(pl.Float64, strict=False).fill_null(0) / 10_000)
        ) * 0.01).alias(output_name)
    ]).filter(~missing_cond(output_name))


@hypercache.cached(ttl=timedelta(hours=6), deep={"my_pt": True}, primary_keys={'my_pt': ["sym"]}, key_params=['my_pt', 'estimate'])
async def cds_basis(my_pt, region="US", dates=None, estimate=True, **kwargs):
    dates = latest_biz_date(dates, True)
    my_pt = ensure_lazy(my_pt)
    syms = await lazy_to_list(my_pt, 'sym')
    cols = [
        'cdsCurve:curveName',
        'cdsBasisToWorst:basisW',
        # 'cdsBasisToMat:basisM',
        # 'cdsParSpdM:cdsParSpdM',
        'cdsParSpdW:cdsParSpdW',
    ]
    cols = kdb_col_select_helper(cols, method="last")
    triplet = construct_gateway_triplet('eodust', "US", 'latestCdsBondtable')
    q = build_pt_query(
        triplet,
        dates=dates,
        cols=cols,
        by="sym",
        lastby="sym",
        date_kwargs={'return_today': False},
        filters={'sym': syms, 'not basisW': None}
    )
    res = await query_kdb(q, fconn(GATEWAY))
    res = res.join(my_pt.select('sym', 'ticker'), on='sym', how='left').with_columns([
        pl.col('cdsCurve').str.split(by=".", inclusive=False).list.get(0).alias('cdsTicker')
    ]).with_columns([
        pl.coalesce([pl.col('ticker'), pl.col('cdsTicker')]).alias('ticker')
    ]).with_columns([
        pl.col('sym').cast(pl.String, strict=False).alias('sym')
    ])
    if (not estimate) or ('bvalBidZspd' not in my_pt.hyper.schema()):
        return res.select(['sym', 'cdsCurve', 'cdsTicker', 'ticker', 'cdsBasisToWorst', 'cdsParSpdW'])

    missing = my_pt.select('ticker', 'sym', 'bvalBidZspd').filter([
        pl.col('bvalBidZspd').is_not_null()
    ]).join(res, on='sym', how='anti').join(res.unique(subset=['ticker'], keep="any").select([
        'ticker', 'cdsCurve', 'cdsTicker', 'cdsParSpdW'
    ]), on='ticker', how='inner').with_columns([
        (pl.col('cdsParSpdW') - pl.col('bvalBidZspd')).alias('cdsBasisToWorst')
    ]).select(['sym', 'cdsCurve', 'cdsTicker', 'ticker', 'cdsBasisToWorst', 'cdsParSpdW'])

    return pl.concat([
        res.select(['sym', 'cdsCurve', 'cdsTicker', 'ticker', 'cdsBasisToWorst', 'cdsParSpdW']),
        missing
    ], how='vertical')


@hypercache.cached(ttl=timedelta(hours=6), deep={"my_pt": True}, primary_keys={'my_pt': ["sym"]}, key_params=['my_pt', 'dates'])
async def cds_percentiles(my_pt, region="US", dates=None, **kwargs):
    dates = latest_biz_date(dates, True)
    syms = await lazy_to_list(my_pt, 'sym')
    await log.debug('running cds percentile for date:', dates)
    cols = [
        'minRangeCdsBasis:minPrice',
        'maxRangeCdsBasis:maxPrice',
        'rangeCdsBasisPercentile:percentile',
        'rangeCdsBasisPercentileShiftOver1D:T0minusT1',
        'rangeCdsBasisPercentileShiftOver7D:T0minusT7',
        'rangeCdsBasisPercentileShiftOver30D:T0minusT30',
        'rangeCdsBasisPercentileShiftOver90D:T0minusT90',
        'rangeCdsBasisPercentileShiftOver365D:T0minusT365',
    ]
    triplet = construct_gateway_triplet('eodust', 'US', 'minMaxPercentileCdsBond')
    q = build_pt_query(
        triplet,
        dates=dates,
        cols=cols,
        by="sym",
        date_kwargs={'return_today': False},
        filters={'sym': syms, 'not latestPrice': None}
    )
    return await query_kdb(q, fconn(GATEWAY))


@hypercache.cached(ttl=timedelta(hours=6), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'dates'])
async def bk2_risk(my_pt, region="US", dates=None, **kwargs):
    isins = await lazy_to_list(my_pt, 'isin')
    cols = [
        'unitCs01:(cs01 * 10000)',
        'unitCs01Pct:(cs01pct * 10000)',
        'unitDv01:(ir01 * 10000)'
    ]
    triplet = construct_panoproxy_triplet("EU", 'bk2', None)
    q = build_pt_query(
        triplet,
        cols=kdb_col_select_helper(cols, method='last'),
        dates=None,
        by="isin:sym",
        date_kwargs={'return_today': False},
        filters={'sym': isins}
    )
    return await query_kdb(q, config=fconn(PANOPROXY, region="EU"))


@hypercache.cached(ttl=timedelta(hours=6), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'dates'])
async def bk2_deep(my_pt, region="US", dates=None, **kwargs):
    isins = await lazy_to_list(my_pt, 'isin')
    cols = [
        'avgLife:bk2AverageLife',
        'bk2Cds:bk2Bcds',
        'unitCs01:(cs01 * 10000)',
        'unitCs01Pct:(cs01pct * 10000)',
        'unitDv01:(ir01 * 10000)'
    ]
    triplet = construct_panoproxy_triplet("EU", 'bk2', dates)
    q = build_pt_query(
        triplet,
        cols=kdb_col_select_helper(cols, method='last'),
        dates=None,
        by="isin:sym",
        date_kwargs={'return_today': False},
        filters={'sym': isins}
    )
    return await query_kdb(q, config=fconn(PANOPROXY, region="EU"))


async def diversification_metrics(my_pt, region="US", dates=None, **kwargs):
    s = my_pt.hyper.schema()
    exprs = [
        pl.col('country').fill_null("NA").alias('country') if 'country' in s else pl.lit("NA", pl.String).alias('country'),
        pl.col('industrySector').fill_null("NA").alias('industrySector') if 'industrySector' in s else pl.lit("NA", pl.String).alias('industrySector')
    ]
    return my_pt.select([pl.col("tnum"), pl.concat_str(exprs, separator="_").alias("divSector")])


def _replace_if_null(col, default=0, dtype=pl.Float64):
    return pl.when(pl.col(col).is_null()).then(pl.lit(default, pl.Float64)).otherwise(pl.col(col)).alias(col)


# CS01 = DV01 * (1+(SPREAD * mat in yrs)/10k)^-1
@hypercache.cached(ttl=timedelta(hours=6), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt'])
async def cs01_estimate(my_pt, region="US", dates=None, **kwargs):
    s = my_pt.hyper.schema()

    spd_cols = [
        pl.col(col) for col in
        ['bvalMidSpd', 'macpMidSpd', 'houseMidSpd', 'amMidSpd', 'allqMidSpd']
        if col in s
    ] or my_pt.hyper.cols_like('MidSpd')

    if not spd_cols:
        spd_cols = [pl.lit(0, pl.Float64)]

    return my_pt.with_columns([
        pl.coalesce(spd_cols).cast(pl.Float64, strict=False).alias('_mid_spd'),
    ]).with_columns([
        _replace_if_null('_mid_spd'),
        _replace_if_null('yrsToMaturity'),
        _replace_if_null('unitDv01'),
    ]).with_columns([
        pl.when(pl.col('yrsToMaturity') < 0).then(pl.lit(0, pl.Float64)).otherwise(pl.col('yrsToMaturity')).alias('yrsToMaturity')
    ]).select([
        pl.col('isin'),
        (
            (pl.col('unitDv01') * (1 + (pl.col('_mid_spd') * pl.col('yrsToMaturity')) / 10_000) ** -1).alias('unitCs01')
        )
    ])


# DV01 SOURCES: riskAnalytics, bondquote, s3 -> bk2,
@hypercache.cached(ttl=timedelta(hours=6), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'dates'])
async def pano_dv01(my_pt, region="US", dates=None, **kwargs):
    ids = my_pt.hyper.to_kdb_sym('isin', unique=True, drop_nulls=True)
    if not ids: return

    region = "US" if region=="SGP" else region
    pano_region = region_to_panoproxy(region)

    yest = next_biz_date(dates, -1)
    date_query = kdb_date_query(yest, return_today=True, as_min=True)
    cols = kdb_col_select_helper(['unitDv01:dv01', 'unitCs01Pct:cs01pct'], method="last")
    q = build_pt_query(
        table=f'.mt.get[`.credit.{pano_region}.riskanalytics.historical]',
        cols=cols,
        filters=f"{date_query}, sym in ({ids})",
        raw_filter=True,
        by=["isin:sym"]
    )
    return await query_kdb(q, config=fconn(PANOPROXY, region=region))



async def risk_transforms(my_pt, **kwargs):
    s = my_pt.hyper.schema()
    my_pt = ensure_lazy(my_pt)

    def zero_to_null(c):
        return pl.when(pl.col(c).is_null() | (pl.col(c) == 0)).then(pl.lit(None, pl.Float64)).otherwise(pl.col(c).cast(pl.Float64, strict=False)).alias(c)

    # Legacy column
    if 'accrued' in s:
        my_pt = my_pt.hyper.ensure_columns(['unitAccrued', 'accrued'], default=None).with_columns([
            zero_to_null('unitAccrued'),
            zero_to_null('accrued')
        ]).with_columns([
            pl.coalesce([pl.col('unitAccrued'), pl.col('accrued')]).alias('unitAccrued')
        ]).drop(['accrued'], strict=False)

    gross_pos_cols = {'grossSize', 'unitAccrued', 'unitDv01', 'unitCs01', 'unitCs01Pct','axeFullBidSize', 'axeFullAskSize',}
    signed_pos_cols = {"netFirmPosition", "netAlgoPosition", "netStrategyPosition", "netDeskPosition"}
    cols = gross_pos_cols|signed_pos_cols
    my_pt = my_pt.hyper.ensure_columns(
        list(cols), default=0.0, dtypes={k:pl.Float64 for k in cols}
    ).with_columns(
        [pl.col(c).abs().cast(pl.Float64, strict=False).fill_null(0.0).alias(c) for c in gross_pos_cols] +
        [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0).alias(c) for c in signed_pos_cols]
    )

    bs_group_keys = ['isin']
    lit0 = pl.lit(0.0, pl.Float64)

    def _pos_part(expr):
        return pl.max_horizontal([expr, lit0])

    def _alloc_bsr_bsi(pos_col: str, alias_mask=None):
        gross = pl.col("grossSize").cast(pl.Float64, strict=False)
        side = pl.col("side")
        pos = pl.col(pos_col).cast(pl.Float64, strict=False).fill_null(0.0)
        abs_pos = pos.abs()

        nonzero_pos = abs_pos > 0
        is_long = pos > 0
        is_short = pos < 0

        # Reducing direction depends on sign of starting position
        is_reducing = nonzero_pos & (
                (is_long & (side=="SELL")) |
                (is_short & (side=="BUY"))
        )

        # Opposite direction is strict BSI (and if pos==0, everything is BSI)
        is_opposite = nonzero_pos & (
                (is_long & (side=="BUY")) |
                (is_short & (side=="SELL"))
        )

        red_amt = pl.when(is_reducing).then(gross).otherwise(lit0)
        opp_amt = pl.when(is_opposite).then(gross).otherwise(lit0)

        v_red = red_amt.sum().over(bs_group_keys)
        v_opp = opp_amt.sum().over(bs_group_keys)

        net_red = _pos_part(v_red - v_opp)  # max(0, V_red - V_opp)
        cap_red = pl.min_horizontal([abs_pos, net_red])  # min(|P|, net_red)
        overshoot = _pos_part(net_red - abs_pos)  # max(0, net_red - |P|)

        # Allocate BSR across reducing-side rows: largest gross first
        reduce_first = pl.when(is_reducing).then(pl.lit(0, pl.Int32)).otherwise(pl.lit(1, pl.Int32))
        red_cum = red_amt.cum_sum().over(bs_group_keys).sort_by(
            [reduce_first, gross],
            descending=[False, True],
        )
        red_prev = red_cum - red_amt
        bsr_alloc = pl.when(is_reducing).then(
            pl.min_horizontal([red_amt, _pos_part(cap_red - red_prev)])
        ).otherwise(lit0)

        # Leftover on reducing rows after BSR allocation
        leftover = pl.when(is_reducing).then(red_amt - bsr_alloc).otherwise(lit0)
        ov_cum = leftover.cum_sum().over(bs_group_keys).sort_by(
            [reduce_first, gross],
            descending=[False, False],
        )
        ov_prev = ov_cum - leftover
        ov_alloc = pl.when(is_reducing).then(
            pl.min_horizontal([leftover, _pos_part(overshoot - ov_prev)])
        ).otherwise(lit0)

        bsi_alloc = (
            pl.when(abs_pos==0).then(gross)
            .when(is_opposite).then(gross)
            .when(is_reducing).then(ov_alloc)
            .otherwise(lit0)
        )

        bsr_mask = alias_mask.replace("_","Bsr") if alias_mask is not None else "_bsr"
        bsi_mask = alias_mask.replace("_", "Bsi") if alias_mask is not None else "_bsi"

        return bsr_alloc.alias(bsr_mask), bsi_alloc.alias(bsi_mask)

    return await (my_pt.select(["side", "isin", "signalFlag", 'tnum'] + list(gross_pos_cols) + list(signed_pos_cols))
        .with_columns([
            (pl.col("grossSize") * (pl.when(pl.col("side")=="SELL").then(-1).otherwise(1))).alias("netSize"),
            (pl.col("grossSize") * (pl.when(pl.col("side")=="BUY").then(1).otherwise(0))).alias("bidSize"),
            (pl.col("grossSize") * (pl.when(pl.col("side")=="SELL").then(1).otherwise(0))).alias("askSize"),

            pl.min_horizontal([pl.col("grossSize"), pl.col("axeFullBidSize")]).alias("axeBidSize"),
            pl.min_horizontal([pl.col("grossSize"), pl.col("axeFullAskSize")]).alias("axeAskSize"),
        ])
        .with_columns([
            pl.struct(_alloc_bsr_bsi("netFirmPosition", 'firm_Size')).alias("_firmSize"),
            pl.struct(_alloc_bsr_bsi("netAlgoPosition", 'algo_Size')).alias("_algoSize"),
            pl.struct(_alloc_bsr_bsi("netStrategyPosition", 'strategy_Size')).alias("_strategySize"),
            pl.struct(_alloc_bsr_bsi("netDeskPosition", 'desk_Size')).alias("_deskSize"),
        ]).unnest(['_firmSize', '_algoSize', '_strategySize', '_deskSize']).with_columns([
            pl.when(pl.col("side")=="BUY").then(pl.col("axeBidSize")).otherwise(pl.col("axeAskSize")).alias("axeSize"),
            pl.when(pl.col("side")=="BUY").then(
                pl.when(pl.col("axeBidSize")==0).then(pl.col("axeAskSize")).otherwise(pl.lit(0.0, pl.Float64))
            ).otherwise(
                pl.when(pl.col("axeAskSize")==0).then(pl.col("axeBidSize")).otherwise(pl.lit(0.0, pl.Float64))
            ).alias("antiSize"),

            (pl.col("grossSize") - pl.col("firmBsrSize") - pl.col("firmBsiSize")).alias("firmBsnSize"),
            (pl.col("grossSize") - pl.col("algoBsrSize") - pl.col("algoBsiSize")).alias("algoBsnSize"),
            (pl.col("grossSize") - pl.col("strategyBsrSize") - pl.col("strategyBsiSize")).alias("strategyBsnSize"),
            (pl.col("grossSize") - pl.col("deskBsrSize") - pl.col("deskBsiSize")).alias("deskBsnSize"),

            (pl.col("unitDv01") / 10_000 * pl.col("grossSize")).alias("grossDv01"),
            (pl.col("unitDv01") / 10_000 * pl.col("netSize")).alias("netDv01"),
            (pl.col("unitCs01") / 10_000 * pl.col("grossSize")).alias("cs01"),
            (pl.col("unitCs01Pct") / 10_000 * pl.col("grossSize")).alias("cs01Pct"),
            (pl.col("unitAccrued") / 100 * pl.col("grossSize")).alias("accruedInterest"),

            (pl.col("unitDv01") / 10_000 * pl.col("axeBidSize")).alias("axeBidDv01"),
            (pl.col("unitDv01") / 10_000 * pl.col("axeAskSize")).alias("axeAskDv01"),
            (pl.col("unitCs01") / 10_000 * pl.col("axeBidSize")).alias("axeBidCs01"),
            (pl.col("unitCs01") / 10_000 * pl.col("axeAskSize")).alias("axeAskCs01"),
            (pl.col("unitCs01Pct") / 10_000 * pl.col("axeBidSize")).alias("axeBidCs01Pct"),
            (pl.col("unitCs01Pct") / 10_000 * pl.col("axeAskSize")).alias("axeAskCs01Pct"),
        ])
        .with_columns([
            (pl.col("firmBsiSize") + pl.col("firmBsnSize")).alias("firmBsinSize"),
            (pl.col("algoBsiSize") + pl.col("algoBsnSize")).alias("algoBsinSize"),
            (pl.col("strategyBsiSize") + pl.col("strategyBsnSize")).alias("strategyBsinSize"),
            (pl.col("deskBsiSize") + pl.col("deskBsnSize")).alias("deskBsinSize"),
        ])
    ).with_columns([

        (pl.col("unitDv01") / 10_000 * pl.col("axeSize")).alias("axeDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("axeSize")).alias("axeCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("axeSize")).alias("axeCs01Pct"),

        (pl.col("unitDv01") / 10_000 * pl.col("antiSize")).alias("antiDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("antiSize")).alias("antiCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("antiSize")).alias("antiCs01Pct"),

        # Balance Sheet
        (pl.col("unitDv01") / 10_000 * pl.col("firmBsrSize")).alias("firmBsrDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("firmBsrSize")).alias("firmBsrCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("firmBsrSize")).alias("firmBsrCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("firmBsiSize")).alias("firmBsiDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("firmBsiSize")).alias("firmBsiCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("firmBsiSize")).alias("firmBsiCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("firmBsnSize")).alias("firmBsnDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("firmBsnSize")).alias("firmBsnCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("firmBsnSize")).alias("firmBsnCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("firmBsinSize")).alias("firmBsinDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("firmBsinSize")).alias("firmBsinCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("firmBsinSize")).alias("firmBsinCs01Pct"),

        (pl.col("unitDv01") / 10_000 * pl.col("algoBsrSize")).alias("algoBsrDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("algoBsrSize")).alias("algoBsrCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("algoBsrSize")).alias("algoBsrCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("algoBsiSize")).alias("algoBsiDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("algoBsiSize")).alias("algoBsiCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("algoBsiSize")).alias("algoBsiCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("algoBsnSize")).alias("algoBsnDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("algoBsnSize")).alias("algoBsnCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("algoBsnSize")).alias("algoBsnCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("algoBsinSize")).alias("algoBsinDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("algoBsinSize")).alias("algoBsinCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("algoBsinSize")).alias("algoBsinCs01Pct"),

        (pl.col("unitDv01") / 10_000 * pl.col("strategyBsrSize")).alias("strategyBsrDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("strategyBsrSize")).alias("strategyBsrCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("strategyBsrSize")).alias("strategyBsrCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("strategyBsiSize")).alias("strategyBsiDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("strategyBsiSize")).alias("strategyBsiCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("strategyBsiSize")).alias("strategyBsiCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("strategyBsnSize")).alias("strategyBsnDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("strategyBsnSize")).alias("strategyBsnCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("strategyBsnSize")).alias("strategyBsnCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("strategyBsinSize")).alias("strategyBsinDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("strategyBsinSize")).alias("strategyBsinCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("strategyBsinSize")).alias("strategyBsinCs01Pct"),

        (pl.col("unitDv01") / 10_000 * pl.col("deskBsrSize")).alias("deskBsrDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("deskBsrSize")).alias("deskBsrCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("deskBsrSize")).alias("deskBsrCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("deskBsiSize")).alias("deskBsiDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("deskBsiSize")).alias("deskBsiCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("deskBsiSize")).alias("deskBsiCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("deskBsnSize")).alias("deskBsnDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("deskBsnSize")).alias("deskBsnCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("deskBsnSize")).alias("deskBsnCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("deskBsinSize")).alias("deskBsinDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("deskBsinSize")).alias("deskBsinCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("deskBsinSize")).alias("deskBsinCs01Pct"),

        pl.min_horizontal([pl.col('axeSize'), pl.col('firmBsrSize')]).alias('firmAxeBsrSize'),
        pl.min_horizontal([pl.col('axeSize'), pl.col('firmBsiSize')]).alias('firmAxeBsiSize'),
        pl.min_horizontal([pl.col('axeSize'), pl.col('firmBsnSize')]).alias('firmAxeBsnSize'),
        pl.min_horizontal([pl.col('axeSize'), pl.col('firmBsinSize')]).alias('firmAxeBsinSize'),

        pl.min_horizontal([pl.col('antiSize'), pl.col('firmBsrSize')]).alias('firmAntiBsrSize'),
        pl.min_horizontal([pl.col('antiSize'), pl.col('firmBsiSize')]).alias('firmAntiBsiSize'),
        pl.min_horizontal([pl.col('antiSize'), pl.col('firmBsnSize')]).alias('firmAntiBsnSize'),
        pl.min_horizontal([pl.col('antiSize'), pl.col('firmBsinSize')]).alias('firmAntiBsinSize'),

    ]).with_columns([

        (pl.col("unitDv01") / 10_000 * pl.col("firmAxeBsrSize")).alias("firmAxeBsrDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("firmAxeBsrSize")).alias("firmAxeBsrCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("firmAxeBsrSize")).alias("firmAxeBsrCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("firmAxeBsiSize")).alias("firmAxeBsiDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("firmAxeBsiSize")).alias("firmAxeBsiCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("firmAxeBsiSize")).alias("firmAxeBsiCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("firmAxeBsnSize")).alias("firmAxeBsnDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("firmAxeBsnSize")).alias("firmAxeBsnCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("firmAxeBsnSize")).alias("firmAxeBsnCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("firmAxeBsinSize")).alias("firmAxeBsinDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("firmAxeBsinSize")).alias("firmAxeBsinCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("firmAxeBsinSize")).alias("firmAxeBsinCs01Pct"),

        (pl.col("unitDv01") / 10_000 * pl.col("firmAntiBsrSize")).alias("firmAntiBsrDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("firmAntiBsrSize")).alias("firmAntiBsrCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("firmAntiBsrSize")).alias("firmAntiBsrCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("firmAntiBsiSize")).alias("firmAntiBsiDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("firmAntiBsiSize")).alias("firmAntiBsiCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("firmAntiBsiSize")).alias("firmAntiBsiCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("firmAntiBsnSize")).alias("firmAntiBsnDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("firmAntiBsnSize")).alias("firmAntiBsnCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("firmAntiBsnSize")).alias("firmAntiBsnCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("firmAntiBsinSize")).alias("firmAntiBsinDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("firmAntiBsinSize")).alias("firmAntiBsinCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("firmAntiBsinSize")).alias("firmAntiBsinCs01Pct"),

        pl.col('signalFlag').is_in(['SSA', 'SA']).alias('_alignedMask'),
        pl.col('signalFlag').is_in(['SSU', 'SU']).alias('_unalignedMask'),

    ]).with_columns([

        pl.when(pl.col('_alignedMask')).then(pl.col('grossSize')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedSize'),
        pl.when(pl.col('_unalignedMask')).then(pl.col('grossSize')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedSize'),
        pl.when(pl.col('_alignedMask')).then(pl.col('grossDv01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedDv01'),
        pl.when(pl.col('_unalignedMask')).then(pl.col('grossDv01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedDv01'),
        pl.when(pl.col('_alignedMask')).then(pl.col('cs01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedCs01'),
        pl.when(pl.col('_unalignedMask')).then(pl.col('cs01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedCs01'),
        pl.when(pl.col('_alignedMask')).then(pl.col('cs01Pct')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedCs01Pct'),
        pl.when(pl.col('_unalignedMask')).then(pl.col('cs01Pct')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedCs01Pct'),

        pl.when(pl.col('_alignedMask')).then(pl.col('axeSize')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedAxeSize'),
        pl.when(pl.col('_alignedMask')).then(pl.col('axeDv01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedAxeDv01'),
        pl.when(pl.col('_alignedMask')).then(pl.col('axeCs01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedAxeCs01'),
        pl.when(pl.col('_alignedMask')).then(pl.col('axeCs01Pct')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedAxeCs01Pct'),

        pl.when(pl.col('_unalignedMask')).then(pl.col('antiSize')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedAntiSize'),
        pl.when(pl.col('_unalignedMask')).then(pl.col('antiDv01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedAntiDv01'),
        pl.when(pl.col('_unalignedMask')).then(pl.col('antiCs01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedAntiCs01'),
        pl.when(pl.col('_unalignedMask')).then(pl.col('antiCs01Pct')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedAntiCs01Pct'),

        pl.when(pl.col('_alignedMask')).then(pl.col('firmBsrSize')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedBsrSize'),
        pl.when(pl.col('_alignedMask')).then(pl.col('firmBsrDv01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedBsrDv01'),
        pl.when(pl.col('_alignedMask')).then(pl.col('firmBsrCs01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedBsrCs01'),
        pl.when(pl.col('_alignedMask')).then(pl.col('firmBsrCs01Pct')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedBsrCs01Pct'),

        pl.when(pl.col('_unalignedMask')).then(pl.col('firmBsinSize')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedBsinSize'),
        pl.when(pl.col('_unalignedMask')).then(pl.col('firmBsinDv01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedBsinDv01'),
        pl.when(pl.col('_unalignedMask')).then(pl.col('firmBsinCs01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedBsinCs01'),
        pl.when(pl.col('_unalignedMask')).then(pl.col('firmBsinCs01Pct')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedBsinCs01Pct'),

        pl.when(pl.col('_alignedMask')).then(pl.col('firmAxeBsrSize')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedAxeBsrSize'),
        pl.when(pl.col('_alignedMask')).then(pl.col('firmAxeBsrDv01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedAxeBsrDv01'),
        pl.when(pl.col('_alignedMask')).then(pl.col('firmAxeBsrCs01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedAxeBsrCs01'),
        pl.when(pl.col('_alignedMask')).then(pl.col('firmAxeBsrCs01Pct')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalAlignedAxeBsrCs01Pct'),

        pl.when(pl.col('_unalignedMask')).then(pl.col('firmAntiBsinSize')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedAntiBsinSize'),
        pl.when(pl.col('_unalignedMask')).then(pl.col('firmAntiBsinDv01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedAntiBsinDv01'),
        pl.when(pl.col('_unalignedMask')).then(pl.col('firmAntiBsinCs01')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedAntiBsinCs01'),
        pl.when(pl.col('_unalignedMask')).then(pl.col('firmAntiBsinCs01Pct')).otherwise(pl.lit(0.0, pl.Float64)).alias('signalUnalignedAntiBsinCs01Pct'),

    ]).select([
        'tnum', 'grossSize', 'netSize',
        'bidSize', 'askSize',
        'grossDv01', 'netDv01', 'cs01', 'cs01Pct',
        'accruedInterest',
        'axeSize', 'antiSize',
        'axeBidSize', 'axeBidDv01', 'axeBidCs01', 'axeBidCs01Pct',
        'axeAskSize', 'axeAskDv01', 'axeAskCs01', 'axeAskCs01Pct',

        'firmBsrSize', 'firmBsiSize', 'firmBsnSize', 'firmBsinSize',
        'firmBsrDv01', 'firmBsiDv01', 'firmBsnDv01', 'firmBsinDv01',
        'firmBsrCs01', 'firmBsiCs01', 'firmBsnCs01', 'firmBsinCs01',
        'firmBsrCs01Pct', 'firmBsiCs01Pct', 'firmBsnCs01Pct', 'firmBsinCs01Pct',

        'algoBsrSize', 'algoBsiSize', 'algoBsnSize', 'algoBsinSize',
        'algoBsrDv01', 'algoBsiDv01', 'algoBsnDv01', 'algoBsinDv01',
        'algoBsrCs01', 'algoBsiCs01', 'algoBsnCs01', 'algoBsinCs01',
        'algoBsrCs01Pct', 'algoBsiCs01Pct', 'algoBsnCs01Pct', 'algoBsinCs01Pct',

        'strategyBsrSize', 'strategyBsiSize', 'strategyBsnSize', 'strategyBsinSize',
        'strategyBsrDv01', 'strategyBsiDv01', 'strategyBsnDv01', 'strategyBsinDv01',
        'strategyBsrCs01', 'strategyBsiCs01', 'strategyBsnCs01', 'strategyBsinCs01',
        'strategyBsrCs01Pct', 'strategyBsiCs01Pct', 'strategyBsnCs01Pct', 'strategyBsinCs01Pct',

        'deskBsrSize', 'deskBsiSize', 'deskBsnSize', 'deskBsinSize',
        'deskBsrDv01', 'deskBsiDv01', 'deskBsnDv01', 'deskBsinDv01',
        'deskBsrCs01', 'deskBsiCs01', 'deskBsnCs01', 'deskBsinCs01',
        'deskBsrCs01Pct', 'deskBsiCs01Pct', 'deskBsnCs01Pct', 'deskBsinCs01Pct',

        'firmAxeBsrSize', 'firmAxeBsiSize', 'firmAxeBsnSize', 'firmAxeBsinSize',
        'firmAntiBsrSize', 'firmAntiBsiSize', 'firmAntiBsnSize', 'firmAntiBsinSize',
        'firmAxeBsrDv01', 'firmAxeBsiDv01', 'firmAxeBsnDv01', 'firmAxeBsinDv01',
        'firmAntiBsrDv01', 'firmAntiBsiDv01', 'firmAntiBsnDv01', 'firmAntiBsinDv01',
        'firmAxeBsrCs01', 'firmAxeBsiCs01', 'firmAxeBsnCs01', 'firmAxeBsinCs01',
        'firmAntiBsrCs01', 'firmAntiBsiCs01', 'firmAntiBsnCs01', 'firmAntiBsinCs01',
        'firmAxeBsrCs01Pct', 'firmAxeBsiCs01Pct', 'firmAxeBsnCs01Pct', 'firmAxeBsinCs01Pct',
        'firmAntiBsrCs01Pct', 'firmAntiBsiCs01Pct', 'firmAntiBsnCs01Pct', 'firmAntiBsinCs01Pct',

        'signalAlignedAxeSize', 'signalAlignedAxeDv01', 'signalAlignedAxeCs01', 'signalAlignedAxeCs01Pct',
        'signalUnalignedAntiSize', 'signalUnalignedAntiDv01', 'signalUnalignedAntiCs01', 'signalUnalignedAntiCs01Pct',
        'signalAlignedBsrSize', 'signalAlignedBsrDv01', 'signalAlignedBsrCs01', 'signalAlignedBsrCs01Pct',
        'signalUnalignedBsinSize', 'signalUnalignedBsinDv01', 'signalUnalignedBsinCs01', 'signalUnalignedBsinCs01Pct',
        'signalAlignedAxeBsrSize', 'signalAlignedAxeBsrDv01', 'signalAlignedAxeBsrCs01', 'signalAlignedAxeBsrCs01Pct',
        'signalUnalignedAntiBsinSize', 'signalUnalignedAntiBsinDv01', 'signalUnalignedAntiBsinCs01', 'signalUnalignedAntiBsinCs01Pct'

    ]).hyper.compress_plan_async()


# dv01 source..
# .credit.common.pricing.utils.getAnalytics[`us]

@hypercache.cached(ttl=timedelta(hours=6), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'dates', 'tickers'])
async def etf_overlap(my_pt, region="US", dates=None, tickers=None, **kwargs):
    tickers = ETF_TICKERS if tickers is None else tickers
    isins = my_pt.hyper.to_kdb_sym('isin', unique=True, drop_nulls=True)
    date_query = kdb_date_query("T-5", as_min=True)
    triplet = creditext_triplet(region, table="etfConstituents")
    ticker_lst = kdb_convert_series_to_sym(ensure_list(tickers))
    q = f'select etfTicker:"," sv string distinct sym by isin from {triplet} where {date_query}, isin in ({isins}), sym in ({ticker_lst})'
    result = await query_kdb(q, config=fconn(GATEWAY), name=f"etfConstituents", lazy=True)
    result = result.with_columns([pl.lit(ticker).is_in(pl.col("etfTicker").str.strip_chars().str.split(",")).cast(pl.Int8, strict=False).alias(clean_camel(f"inEtf{ticker}")) for ticker in tickers])
    result = result.drop("etfTicker", strict=False)
    return result


@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'dates'])
async def fungible_series(my_pt, region="US", dates=None, **kwargs):
    isins = my_pt.hyper.to_list('isin')
    triplet = construct_panoproxy_triplet(region="US", table="fungibleBonds", dates=None)
    my_cols = kdb_col_select_helper(['fungibleIsin:linkedIsin', 'fungibleSeries:linkedIsinType'])
    q = build_pt_query(
        triplet,
        dates=None,
        date_kwargs={'return_today': False},
        by="isin:sym",
        cols=my_cols,
        filters={
            'sym': isins,
        }
    )
    return await query_kdb(q, config=fconn(PANOPROXY_US))

@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["fungibleIsin"]}, key_params=['my_pt', 'dates'])
async def fungible_enhance(my_pt, region="US", dates=None, **kwargs):
    isins = my_pt.hyper.to_list('fungibleIsin', unique=True, drop_nulls=True)
    triplet = ".mt.get[`.credit.refData]"
    my_cols = kdb_col_select_helper(['fungibleDescription:Description', "fungibleSym:sym"])
    q = build_pt_query(
        triplet,
        dates=None,
        date_kwargs={'return_today': False},
        by="fungibleIsin:isin",
        cols=my_cols,
        filters={
            'isin': isins,
        }
    )
    return await query_kdb(q, config=fconn(PANOPROXY, region=region))

async def fungible_join(my_pt, frames, region="US", dates=None, **kwargs):
    if (my_pt is None) or (my_pt.hyper.is_empty()): return
    return my_pt.unique(subset=['isin'])

# ========================================================
## FLAGS
# ========================================================

#
# async def earnings_date(my_pt, region="US", dates=None, **kwargs):
#     if await DAILY_CACHE.exists('EARNINGS'):
#         r = await DAILY_CACHE.get('EARNINGS')
#     else:
#         kdb_ids = kdb_convert_polars_to_sym(await collect_lazy_ids(my_pt, "ticker"))
#         date_query = kdb_date_query(dates)
#         q = r'({formatString:"sssd";earningDates: value ("(\"",formatString,"\";enlist\",\")0:`$\":","Earnings.csv","\"");select Ticker:ticker, bbgEquity:Equity, earningsDate:date from earningDates where date >= .z.D, i=(last;i) fby ticker}[])'
#         r = await query_kdb(q, config=fconn(SMAD_US, region="US", strict=True, dbtype=['UAT']), name=f"uat")
#         await DAILY_CACHE.set('EARNINGS', r, ttl=86_000)
#     if not r is None:
#         return my_pt.select('ticker').join(r.unique(subset=['ticker'], keep="last", maintain_order=False), on='ticker', how='left')

@hypercache.cached(ttl=timedelta(hours=1), deep={"my_pt": True}, primary_keys={'my_pt': ["esmi"]}, key_params=['my_pt'])
async def pano_restricted_list(my_pt, region="US", dates=None, **kwargs):
    esmi = my_pt.hyper.to_kdb_sym('esmi', drop_nulls=True, unique=True)
    q = """select restrictedCode:last string legacyCode, restrictionTier:last tier by esmi:sym from .mt.get[`.credit.us.restrictedList.realtime] where sym in (%s), legacyCode in `R3`R4`R5`R6`R7`R9`RS`Q""" % esmi
    r = await query_kdb(q, config=fconn(PANOPROXY_US))
    if r is not None:
        return r.with_columns(pl.col('esmi').cast(pl.String, strict=False).alias('esmi'))


@hypercache.cached(ttl=timedelta(minutes=1), deep={"my_pt": True}, primary_keys={'my_pt': ["ticker"]}, key_params=['my_pt'])
async def pano_dnt_warnings_by_ticker(my_pt, region="US", dates=None, **kwargs):
    my_pt = ensure_lazy(my_pt)
    tickers = my_pt.hyper.to_kdb_sym('ticker', unique=True, drop_nulls=True)
    isins = my_pt.hyper.to_kdb_sym('isin', unique=True, drop_nulls=True)
    pano_region = region_to_panoproxy(region).lower()
    flags = ['Earnings', 'earningsDNT', 'earningsRR', 'premarketMover', 'restricted', 'tickerOverride']
    flags_str = ";".join([f'"{f}"' for f in flags])
    q1 = '''select dntComment:last comment, dntUpdateTime:last eventTimestamp by sym from .mt.get[`.credit.%s.creditConfigs][`latestQuotingConfig] where active=1b, sym in (%s), comment in (%s)''' % (
        pano_region, isins, flags_str)
    q2 = '''select dntComment:last comment, dntUpdateTime:last eventTimestamp by sym from .mt.get[`.credit.%s.creditConfigs][`latestTraderOverrides] where eventEnd>.z.p, active=1, ((symType=`ticker) & (sym in (%s)))''' % (
        pano_region, tickers)
    q = '{byi:%s;byi: update symType:`isin from byi;byt:%s;byt: update symType:`ticker from byt;byi,byt}[]' % (q1, q2)
    res = await query_kdb(q, fconn(PANOPROXY, region=region))
    if res is None: return
    isin_map = my_pt.select([pl.col('isin').alias('sym'), pl.col('ticker')])

    by_isin = (
        res.filter(pl.col('symType') == 'isin')
        .join(isin_map, on='sym', how='left')
        .drop(['sym', 'symType'], strict=False)
    )

    by_ticker = (
        res.filter(pl.col('symType') == 'ticker')
        .rename({'sym': 'ticker'})
        .drop('symType', strict=False)
    )

    combined = pl.concat([by_isin, by_ticker], how='diagonal_relaxed')
    return combined.unique(subset=['ticker'], keep='last').with_columns([
        pl.col('dntUpdateTime').cast(pl.Datetime, strict=False)
    ])

# ========================================================
## Static Data
# ========================================================

def _get_sym_filter(cusips, isins, csp_col="cusip", isin_col='isin'):
    c = f"{csp_col} in (%s)" % cusips if not cusips in ['""', "`", None] else ''
    i = f"{isin_col} in (%s)" % isins if not isins in ['""', "`", None] else ''
    if c!='' and i!='':
        return 'any (%s; %s)' % (c, i)
    elif c!='':
        return '%s' % c
    elif i!='':
        return '%s' % i
    else:
        return "1b"


async def get_syms(my_pt, csp_col="cusip", isin_col="isin", as_string=False):
    s = my_pt.collect_schema() if isinstance(my_pt, pl.LazyFrame) else my_pt.schema
    isins, cusips = None, None

    if as_string:
        if 'isin' in s:
            isins = kdb_convert_polars_to_str(await collect_lazy_ids(my_pt, "isin"))
            if 'cusip' in s:
                cusips = kdb_convert_polars_to_str(await collect_lazy_ids(my_pt.filter(pl.col("isin").is_null()), "cusip"))
        elif 'cusip' in s:
            cusips = kdb_convert_polars_to_str(await collect_lazy_ids(my_pt, "cusip"))

    else:
        if 'isin' in s:
            isins = kdb_convert_polars_to_sym(await collect_lazy_ids(my_pt, "isin"))
            if 'cusip' in s:
                cusips = kdb_convert_polars_to_sym(await collect_lazy_ids(my_pt.filter(pl.col("isin").is_null()), "cusip"))
        elif 'cusip' in s:
            cusips = kdb_convert_polars_to_sym(await collect_lazy_ids(my_pt, "cusip"))

    return _get_sym_filter(cusips, isins, csp_col=csp_col, isin_col=isin_col)


@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt'])
async def kdb_bond_pano_static(my_pt, region="US", dates=None, **kwargs):
    my_pt = ensure_lazy(my_pt)
    s = my_pt.hyper.schema()
    k = None
    if 'id' in s:
        k = 'id'
        o_ids = my_pt.hyper.ul('id')
    elif 'isin' in s:
        k = 'isin'
        o_ids = my_pt.hyper.ul('isin')
    else:
        k = 'cusip'
        o_ids = my_pt.hyper.ul('cusip')

    if not o_ids: return

    isins = [str(i) for i in o_ids if len(str(i))==12]
    cusips = [str(i) for i in o_ids if len(str(i))==9]
    isins = kdb_convert_polars_to_sym(isins) if isins else None
    cusips = kdb_convert_polars_to_sym(cusips) if cusips else None

    filters = []
    if isins: filters.append('(isin in %s)' % isins)
    if cusips: filters.append('(cusip in %s)' % cusips)
    filters = " | ".join(filters)
    if filters:
        cols = {
            'cusip', 'Description', 'sym', 'ESMP', 'ESMI', 'Currency', 'IssuerCountry', 'IssuerName', 'Ticker', 'maturityDate:CurrentMaturityDate', 'IndustrySector', 'IndustryGroup', 'IssuerIndustrySector', 'IsCallable', 'amountOutstanding:PrincipalAmountOutstanding', 'nextCouponDate:NextPayDate', 'amountIssued:PrincipalAmountIssued', 'RatingCombined', 'IsRule144A', 'IsRegS', 'couponType:CouponDividendType'
        } - {k}
        cols = kdb_col_select_helper(list(cols), method="first")
        q = 'select %s by isin from .mt.get[`.credit.refData] where %s' % (cols, filters)
        r = await query_kdb(q, config=fconn(PANOPROXY, region=region))
        if (r is None) or (r.hyper.is_empty()): return
        lk = [k] if k == 'isin' else [k, 'isin']
        by_cusip = my_pt.select(lk).join(r.drop('isin', strict=False), left_on=k, right_on='cusip', how='inner')
        by_isin  = my_pt.select(lk).join(r.drop('cusip', strict=False), left_on=k, right_on='isin', how='inner')
        return pl.concat([by_cusip, by_isin])

@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt'])
async def kdb_series_static(my_pt, region="EU", dates=None, **kwargs):
    my_pt = ensure_lazy(my_pt)
    isins = my_pt.hyper.ul('isin')

    triplet = construct_gateway_triplet('seriesmodels', 'EU', 'bonds')
    cols = kdb_col_select_helper([
        'esmp', 'cusip', 'description', 'ticker', 'issueDate', 'maturityDate',
        'currency', 'coupon', 'isRule144A', 'isRegS', 'nextCallDate'
    ])
    q = build_pt_query(triplet, cols, dates, filters={'sym': isins}, by=['isin:sym'])
    r = await query_kdb(q, config=fconn(GATEWAY_EU))
    if (r is None) or (r.hyper.is_empty()): return
    return r

@hypercache.cached(ttl=timedelta(hours=12), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt'])
async def house_eu_benchmark(my_pt, region="EU", dates=None, **kwargs):
    my_pt = ensure_lazy(my_pt)
    isins = my_pt.hyper.ul('isin')
    triplet = construct_gateway_triplet('seriesmodels', 'EU', 'benchmarks')
    cols = kdb_col_select_helper(['houseEuBenchmark:benchmark'])
    q = build_pt_query(triplet, cols, dates, filters={'sym': isins, 'not benchmark':None}, by=['isin:sym'])
    r = await query_kdb(q, config=fconn(GATEWAY_EU))
    if (r is None) or (r.hyper.is_empty()): return
    return r

@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt'])
async def kdb_bond_static_data(my_pt, region="US", dates=None, direct=True, **kwargs):
    if direct:
        syms = await get_syms(my_pt, "CSP", "ISN", as_string=True)
        by = 'isin:ISN'
        cols = ['sym',
                'ESMP',
                'ESMI',
                'BBPK',
                'cusip:CSP',
                'Country',
                'Description',
                'IsRule144A',
                'IsWhenIssued',
                'IssueDate',
                'ParValue',
                'IssuerName',
                'IssuerCountry',
                'Seniority',
                'IsPrivatePlacement',
                'AccrualMethod',
                'couponFrequency:CouponDividendFrequency',
                'coupon:CouponDividendRate',
                'couponType:CouponDividendType',
                'Currency',
                'maturityDate:CurrentMaturityDate',
                'IsCallable',
                'IsConvertible',
                'IsPutable',
                'IsSinkable',
                'MaturityType',
                'amountIssued:PrincipalAmountIssued',
                'amountOutstanding:PrincipalAmountOutstanding',
                'ratingSandP:RatingSANDP',
                'RatingMoody',
                'IndustryGroup',
                'IndustrySector',
                'IssuerIndustryGroup',
                'IssuerIndustrySector',
                'IndustrySubGroup',
                'IssuerIndustrySubGroup',
                'IsRegS',
                'isDtcEligible:DTCEligible',
                'Series',
                'IsOriginalIssueDiscount',
                'IsStructuredNote',
                'IsRangeNote',
                'IsRatingSensitive',
                'minDenomination:MinimumPrincipalDenomination',
                'minIncrement:MinimumPrincipalIncrement',
                'IsSubordinated',
                'IsGlobalForm:InGlobalForm',
                'CalledDate',
                'NextCallDate',
                'isCalled:(.z.d+1^CalledDate)<=.z.d',
                'nextCallPx:NextCallPrice',
                'nextCouponDate:NextPayDate'
                ]
    else:
        syms = await get_syms(my_pt, "cusip", "isin")
        by = 'isin'
        cols = ['sym',
                'esmp',
                'esmi',
                'bbpk',
                'cusip',
                'country',
                'description',
                'isRule144A',
                'isWhenIssued',
                'issueDate',
                'parValue',
                'issuerName',
                'issuerCountry',
                'seniority',
                'isPrivatePlacement',
                'accrualMethod',
                'couponFrequency:couponDividendFrequency',
                'coupon:couponDividendRate',
                'couponType:couponDividendType',
                'currency',
                'maturityDate:currentMaturityDate',
                'isCallable',
                'isConvertible',
                'isPutable',
                'isSinkable',
                'maturityType',
                'amountIssued:principalAmountIssued',
                'amountOutstanding:principalAmountOutstanding',
                'ratingSandP:ratingSANDP',
                'ratingMoody',
                'industryGroup',
                'industrySector',
                'issuerIndustryGroup',
                'issuerIndustrySector',
                'industrySubGroup',
                'issuerIndustrySubGroup',
                'isRegS',
                'isDtcEligible:dtcEligible',
                'series',
                'isOriginalIssueDiscount',
                'isStructuredNote',
                'isRangeNote',
                'isRatingSensitive',
                'minDenomination:minimumPrincipalDenomination',
                'minIncrement:minimumPrincipalIncrement',
                'isSubordinated',
                'isGlobalForm:inGlobalForm',
                'calledDate',
                'nextCallDate',
                'isCalled:(.z.d+1^calledDate)<=.z.d',
                'nextCallPx:nextCallPrice',
                'nextCouponDate:nextPayDate'
                ]
    my_cols = kdb_col_select_helper(cols)
    triplet = 'fixedinc' if direct else construct_gateway_triplet(schema="esm", region="US", table="fixedinc")
    q = build_pt_query(triplet, cols=my_cols, by=by, dates=None, filters=syms, raw_filter=True, date_kwargs={"return_today": False})
    q = q.replace("1b,", "")
    r = await query_kdb(q, config=fconn(ESM_DIRECT) if direct else fconn(GATEWAY, region=region), timeout=kwargs.pop('timeout', 20))
    if (r is None) or (r.hyper.is_empty()): return
    my_pt = my_pt.lazy() if isinstance(my_pt, pl.DataFrame) else my_pt
    ref = pl.concat(
        [
            my_pt.select("id").join(r, left_on="id", right_on="cusip", how="inner", coalesce=False),
            my_pt.select("id").join(r, left_on="id", right_on="isin", how="inner", coalesce=False)
        ]
    )
    return ref


@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt'])
async def instrument_static(my_pt, region="US", dates=None, **kwargs):
    syms = await get_syms(my_pt)
    cols = [
        'sym', 'esmp', 'productType', 'cusip', 'description', 'ticker', 'currency', 'series', 'country', 'issueDate',
        'accrualMethod', 'coupon', 'couponFrequency', 'couponType', 'isInDefault',
        'isRule144A', 'regsEsmp', 'isOperational:operationalIndicator', 'ratingMoody', 'ratingSandP:ratingSP',
        'ratingFitch', 'ratingCombined', 'industryGroup', 'industrySector',
        'industrySubGroup', 'issuerIndustry', 'seniority', 'isBullet', 'isCovered', 'isFloater', 'isInflationLinked',
        'isPerpetual', 'isPrivatePlacement', 'isSubordinated',
        'issueBenchmarkTreasury', 'maturityDate', 'pseudoWorkoutDate', 'isCallable', 'isConvertible', 'isPutable',
        'isSinkable', 'minDenomination:minimumPrincipalDenomination',
        'minIncrement:minimumPrincipalIncrement', 'amountIssued:principalAmountIssued',
        'amountOutstanding:principalAmountOutstanding', 'calledDate', 'countryOfRisk',
        'ultimateParentCountryOfRisk', 'issuerGicsSector', 'issuerGicsIndustryGroup', 'issuerGicsSubIndustry',
        'issuerIndustryGroup', 'issuerIndustrySector', 'issuerIndustrySubGroup', 'bondType',
        'assetType', 'daysToSettle', 'isRegS', 'isPayInKind', 'isWhenIssued', 'isStructuredNote', 'currentFactor',
        'isProRataSink', 'isProRataCall', 'isMakeWholeCall', 'issuerName', 'parValue',
        'nextCallDate', 'nextCallPx:nextCallPrice', 'maturityType', 'isCalled:(.z.d+1^calledDate)<=.z.d'
    ]
    my_cols = kdb_col_select_helper(cols)
    triplet = construct_gateway_triplet(schema="esm", region="US", table="instrumentStatic")
    q = build_pt_query(triplet, cols=my_cols, by="isin", dates=dates, filters=syms, raw_filter=True, date_kwargs={"return_today": False})
    r = await query_kdb(q, config=fconn(GATEWAY), timeout=20)
    if (r is None) or (r.hyper.is_empty()): return
    my_pt = my_pt.lazy() if isinstance(my_pt, pl.DataFrame) else my_pt
    ref = pl.concat(
        [
            my_pt.select("id").join(r, left_on="id", right_on="cusip", how="inner", coalesce=False),
            my_pt.select("id").join(r, left_on="id", right_on="isin", how="inner", coalesce=False)
        ]
    )
    return ref


@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'region'])
async def kdb_smad_bondStaticData_pano(my_pt, region="US", dates=None, **kwargs):
    my_pt = ensure_lazy(my_pt)
    syms = await get_syms(my_pt)
    p_region = region_to_panoproxy(region)
    triplet = '.mt.get[`.credit.%s.bondStaticData.realtime]' % p_region
    cols = [
        'sym', 'esmp', 'productType', 'cusip', 'description', 'ticker', 'currency', 'series', 'country', 'issueDate',
        'accrualMethod', 'coupon', 'couponFrequency', 'isInDefault', 'isRule144A',
        'ratingMoody', 'ratingSandP:ratingSP', 'ratingFitch', 'ratingCombined', 'industryGroup', 'industrySector',
        'industrySubGroup', 'issuerIndustry', 'paymentRank', 'seniority', 'isBullet',
        'isCovered', 'isFloater', 'isInflationLinked', 'isPerpetual', 'isPrivatePlacement', 'maturityDate',
        'pseudoWorkoutDate', 'isCallable', 'isConvertible', 'isPutable', 'isSinkable',
        'minDenomination:minimumPrincipalDenomination', 'minIncrement:minimumPrincipalIncrement',
        'amountIssued:principalAmountIssued', 'amountOutstanding:principalAmountOutstanding',
        'nextCallDate', 'nextCallPx:nextCallPrice', 'calledDate', 'countryOfRisk', 'ultimateParentCountryOfRisk',
        'issuerGicsSector', 'issuerGicsIndustryGroup', 'issuerGicsIndustry',
        'issuerGicsSubIndustry', 'issuerIndustryGroup', 'issuerIndustrySector', 'issuerIndustrySubGroup',
        'daysToSettle', 'isRegS', 'isWhenIssued', 'couponType', 'maturityType',
        'isCalled:(.z.d+1^calledDate)<=.z.d',
    ]
    cols = kdb_col_select_helper(cols)
    q = build_pt_query(triplet, by="isin", cols=cols, dates=dates, filters=syms, raw_filter=True)
    r = await query_kdb(q, config=fconn(PANOPROXY, region=region), timeout=5)
    if (r is None) or (r.hyper.is_empty()): return
    return pl.concat(
        [
            my_pt.select("id").join(r, left_on="id", right_on="cusip", how="inner", coalesce=False),
            my_pt.select("id").join(r, left_on="id", right_on="isin", how="inner", coalesce=False)
        ]
    )


@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'region'])
async def kdb_smad_bondStaticData(my_pt, region="US", dates=None, **kwargs):
    my_pt = ensure_lazy(my_pt)
    syms = await get_syms(my_pt)
    triplet = construct_gateway_triplet("smad", region, "bondStaticData")
    cols = [
        'sym', 'esmp', 'productType', 'cusip', 'description', 'ticker', 'currency', 'series', 'country', 'issueDate',
        'securityType', 'accrualMethod', 'coupon', 'couponFrequency', 'isInDefault', 'isRule144A',
        'ratingMoody',
        'ratingSandP:ratingSP', 'ratingFitch', 'ratingCombined', 'industryGroup', 'industrySector', 'industrySubGroup',
        'issuerIndustry', 'paymentRank', 'seniority', 'isBullet', 'isCovered',
        'isFloater', 'isInflationLinked', 'isPerpetual', 'isPrivatePlacement', 'maturityDate', 'pseudoWorkoutDate',
        'isCallable', 'isConvertible', 'isPutable', 'isSinkable',
        'minDenomination:minimumPrincipalDenomination', "minIncrement:minimumPrincipalIncrement",
        'amountIssued:principalAmountIssued',
        'amountOutstanding:principalAmountOutstanding', 'nextCallDate', 'nextCallPx:nextCallPrice', 'calledDate',
        'countryOfRisk', 'ultimateParentCountryOfRisk', 'issuerGicsSector',
        'issuerGicsIndustryGroup', 'issuerGicsIndustry', 'issuerGicsSubIndustry', 'issuerIndustryGroup',
        'issuerIndustrySector', 'issuerIndustrySubGroup', 'daysToSettle', 'isRegS', 'isWhenIssued',
        'issuerName', 'maturityType', 'couponType', 'isCalled:(.z.d+1^calledDate)<=.z.d',
    ]

    date_q = next_biz_date(dates, -5)
    cols = kdb_col_select_helper(cols)
    q = build_pt_query(triplet, by="isin", cols=cols, dates=date_q, filters=syms, date_kwargs={"as_min": True}, raw_filter=True)
    r = await query_kdb(q, config=fconn(GATEWAY, region=region), name=f"smad", lazy=True)
    if (r is None) or (r.hyper.is_empty()): return
    ref = pl.concat(
        [
            my_pt.select("id").join(r, left_on="id", right_on="cusip", how="inner", coalesce=False),
            my_pt.select("id").join(r, left_on="id", right_on="isin", how="inner", coalesce=False)
        ]
    )
    return ref


@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'region', 'tbl'])
async def kdb_bbg_bond(my_pt, region="US", dates=None, tbl="internalDomBond", **kwargs):
    my_pt = ensure_lazy(my_pt)
    syms = await get_syms(my_pt)
    triplet = construct_gateway_triplet("bbg", region, tbl)
    my_dates = next_biz_date(dates, -1) if is_today(dates, utc=True) else dates
    cols = [
        'avgLife', 'bbgId:sym', 'capSecTyp', 'convexity', 'country', 'coupon', 'currency', 'cusip8:cusip',
        'duration:durWrsMod', 'figi', 'issrClass', 'issuerIndustry:issrclsl3', 'issuerIndustryGroup:issrclsl2',
        'issuerName:issuer', 'issuerSector:issrclsl1', 'issuerSubIndustry:issrclsl4', 'lbclass', 'lbclsdnd', 'lblevel2',
        'maturityDate:maturdate', 'ratingMoody:qualitye', 'sinktype', 'sovRating:qualsov', 'ticker', 'typebond',
        'unitAccrued:accrinte'
    ]
    my_cols = kdb_col_select_helper(cols)
    q = build_pt_query(triplet, cols=my_cols, by="isin", dates=my_dates, filters=syms, raw_filter=True)
    r = await query_kdb(q, config=fconn(GATEWAY, region=region))
    if (r is None) or (r.hyper.is_empty()): return
    r = r.with_columns([
        pl.col("maturityDate").cast(pl.Utf8, strict=False).str.to_date(format="%Y%m%d", strict=False, exact=False).alias("maturityDate"),
        pl.when(pl.col('cusip8').is_not_null()).then(
            pl.concat_str([pl.col('cusip8'), pl.col('isin').str.slice(10, 1)])
        ).otherwise(pl.lit(None, pl.String)).alias('cusip'),
        pl.when(pl.col('duration')==0).then(pl.lit(None, pl.Float64)).otherwise(pl.col('duration')).alias('duration')
    ])
    ref = pl.concat(
        [
            my_pt.select("id").join(r, left_on="id", right_on="isin", how="inner", coalesce=False),
            my_pt.select("id").join(r, left_on="id", right_on="cusip", how="inner", coalesce=False)
        ]
    )
    return ref


@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt'])
async def kdb_eag_bond(my_pt, region="US", dates=None, **kwargs):
    my_pt = ensure_lazy(my_pt)
    syms = await get_syms(my_pt)
    triplet = construct_gateway_triplet("bbg", "EU", "internalEagBond")
    my_dates = next_biz_date(dates, -1) if is_today(dates, utc=True) else dates
    cols = [
        'cusip8:cusip', 'convexity', 'country', 'currency', 'duration:durAdjmod', 'issrClass', 'issuerSector:issrclsl1',
        'issuerIndustryGroup:issrclsl2', 'issuerIndustry:issrclsl3',
        'issuerSubIndustry:issrclsl4', 'issuerName:issuer', 'sovRating:qualsov', 'ticker', 'ratingMoody:qualitye',
        'maturityDate:maturdate'
    ]
    my_cols = kdb_col_select_helper(cols)

    q = build_pt_query(triplet, cols=my_cols, by="isin", dates=my_dates, filters=syms, raw_filter=True)
    r = await query_kdb(q, config=fconn(GATEWAY, region=region))
    if (r is None) or (r.hyper.is_empty()): return
    r = r.with_columns([
        pl.col("maturityDate").cast(pl.Utf8, strict=False).str.to_date(format="%Y%m%d", strict=False, exact=False).alias("maturityDate"),
        pl.when(pl.col('cusip8').is_not_null()).then(
            pl.concat_str([pl.col('cusip8'), pl.col('isin').str.slice(10, 1)])
        ).otherwise(pl.lit(None, pl.String)).alias('cusip'),
        pl.when(pl.col('duration')==0).then(pl.lit(None, pl.Float64)).otherwise(pl.col('duration')).alias('duration')
    ])
    ref = pl.concat(
        [
            my_pt.select("id").join(r, left_on="id", right_on="isin", how="inner", coalesce=False),
            my_pt.select("id").join(r, left_on="id", right_on="cusip", how="inner", coalesce=False)
        ]
    )
    return ref


async def kdb_bbg_domBond(my_pt, region="US", dates=None, **kwargs):
    return await kdb_bbg_bond(my_pt, region="US", dates=dates, tbl='internalDomBond')


async def kdb_bbg_emgBond(my_pt, region="EU", dates=None, **kwargs):
    return await kdb_bbg_bond(my_pt, region="EU", dates=dates, tbl='internalEmgBond')


@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, cache_none=True, key_params=['my_pt', 'region'])
async def kdb_muni_data(my_pt, region="US", dates=None, **kwargs):
    my_pt = ensure_lazy(my_pt)
    syms = await get_syms(my_pt, as_string=False, csp_col="csp", isin_col="isin")
    triplet = construct_gateway_triplet("esm", "US", "munis")
    cols = [
        'sym', 'esmp', 'esmi', 'bbpk', 'cusip:csp', 'IssuerCountry', 'IssuerName', 'amountIssued:IssueSize', 'Country',
        'Description', 'IsInDefault', 'currency:IssueCurrency',
        'maturityDate:currentMaturityDate', "IsCallable", "IsPutable", "IsSinkable",
        "amountOutstanding:PrincipalAmountOutstanding", 'ratingSandP:SANDPRating', 'ratingMoody:MoodyRating',
        'ratingFitch:FitchRating', 'bbgId:BBGI', "minDenomination:MinimumPrincipalDenomination"
    ]
    my_cols = kdb_col_select_helper(cols)
    q = build_pt_query(triplet, cols=my_cols, by="isin:isin", dates=dates, filters=syms, raw_filter=True)
    r = await query_kdb(q, config=fconn(GATEWAY, region=region), name=f"esm")
    if (r is None) or (r.hyper.is_empty()): return
    r = r.with_columns([
        pl.lit("MUNI").alias("industrySector"),
        pl.lit("MUNI").alias("industryGroup"),
        pl.lit("MUNI").alias("issuerIndustrySector"),
        pl.lit("MUNI").alias("issuerIndustry"),
        pl.lit("MUNI").alias("issuerIndustryGroup"),
        pl.lit("MUNI").alias("issuerIndustrySubGroup"),
        pl.lit("MUNI").alias("industrySubGroup"),
        pl.lit(1, pl.Int8).alias("isMuni"),
    ])
    ref = pl.concat(
        [
            my_pt.select("id").join(r, left_on="id", right_on="cusip", how="inner", coalesce=False),
            my_pt.select("id").join(r, left_on="id", right_on="isin", how="inner", coalesce=False)
        ]
    )

    return ref


@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, cache_none=True, key_params=['my_pt'])
async def muni_direct(my_pt, region="US", dates=None, **kwargs):
    syms = await get_syms(my_pt, as_string=False, csp_col="cusip", isin_col="sym")
    triplet = construct_gateway_triplet("muni", "US", "barclaysMuniPricingLiquidity")
    cols = [
        'coupon', 'maturityDate:maturity', 'duration:bidModifiedDurationn', 'convexity:bidConvexityn',
        'muniLiqScore:liquidityScore', 'currency', 'unitAccrued:accruedInterestn',
        'issuerName:longNameofIssuer', 'usHouseBidPx:bidPricen', 'usHouseBidSpd:bidSpreadToWorstBPS',
        'usHouseMidPx:midPrice', 'usHouseMidSpd:midSpreadToWorstBPS', 'usHouseAskPx:askPrice',
        'usHouseAskSpd:askSpreadToWorstBPS'
    ]
    d = next_biz_date(dates, -1)
    q = build_pt_query(
        triplet,
        cols=kdb_col_select_helper(cols, method="last"),
        by="isin:sym",
        dates=d,
        date_kwargs={"as_min": True},
        filters=syms,
        raw_filter=True
    )
    res = await query_kdb(q, config=fconn(GATEWAY))
    if (res is None) or (res.hyper.is_empty()): return
    return res.with_columns([
        pl.lit(1, pl.Int8).alias('isMuni'),
        pl.lit("US", pl.String).alias('country'),
        pl.lit('MUNI').alias('productType'),
        pl.when(pl.col('duration')==0).then(pl.lit(None, pl.Float64)).otherwise(pl.col('duration')).alias('duration')
    ])


@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["cusip"]}, cache_none=True, key_params=['my_pt'])
async def muni_min_increment(my_pt, region="US", dates=None, **kwargs):
    cusips = my_pt.hyper.to_list('cusip', drop_nulls=True, unique=True)
    triplet = construct_gateway_triplet("muni", "US", "VenueMarketData")
    cols = ['minDenomination:minimumQuantity', 'minIncrement:minimumIncrement']
    d = next_biz_date(dates, -1)
    q = build_pt_query(
        triplet,
        cols=kdb_col_select_helper(cols, method="first"),
        by="cusip:sym",
        dates=d,
        date_kwargs={"as_min": True},
        filters={'sym': cusips, 'not minimumQuantity': 0},
    )
    r = await query_kdb(q, config=fconn(GATEWAY))
    if (r is None) or (r.hyper.is_empty()): return
    return r.with_columns([
        pl.lit(1, pl.Int8).alias('isMuni'),
        pl.lit("US", pl.String).alias('country'),
        pl.lit('MUNI').alias('productType')
    ])


async def muni_filler(my_pt, region="US", dates=None, **kwargs):
    return my_pt.select([
        pl.col('tnum'),
        pl.lit(0, pl.Int8).alias('isMuni')
    ])


@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, cache_none=True, key_params=['my_pt'])
async def muni_ticker(my_pt, region="US", dates=None, **kwargs):
    s = my_pt.hyper.schema()
    name_cols = [n for n in ['ticker', 'description', 'issuerName'] if n in s]
    if not name_cols: return

    abbrs = {**MUNI_ABBREVIATIONS, **NAME_TO_STATE_ABBR}
    abbrs = {k.upper(): v for k, v in abbrs.items()}
    abbrs = {key: abbrs[key] for key in sorted(abbrs, key=lambda x: len(x), reverse=True)}
    x = my_pt.filter(pl.col('isMuni')==1)

    return x.with_columns([
        pl.coalesce([
            pl.col(n).cast(pl.String, strict=False) for n in name_cols
        ]).str.replace_many(abbrs, ascii_case_insensitive=True, leftmost=True).alias('ticker')
    ]).select([
        pl.col('isin'),
        pl.col('ticker').str.replace_all(" ", "_", literal=True).alias('ticker'),
        pl.concat_str([
            pl.col('ticker'),
            pl.when(pl.col('coupon').is_null()).then(pl.lit("")).otherwise(pl.col('coupon').cast(pl.Float64, strict=False).round(3).cast(pl.String, strict=False)),
            pl.when(pl.col('maturityDate').is_null()).then(pl.lit("")).otherwise(pl.col('maturityDate').cast(pl.Date, strict=False).dt.year()),
        ], separator=" ").str.replace_all("  ", " ", literal=True).alias('description')
    ])


@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt'])
async def trade_to_flag(my_pt, region="US", dates=None, **kwargs):
    TO_WORST = "TO_WORST"
    TO_MATURITY = "TO_MATURITY"
    TO_CALL = "TO_CALL"

    b = lambda c: (pl.col(c).is_not_null() & (pl.col(c).cast(pl.Int8, strict=False)==1))

    is_callable = b("isCallable")
    is_called = b("isCalled")
    is_makewhole_only = b("isMakeWholeCall")
    is_hybrid = b("isHybrid")
    is_perpetual = b("isPerpetual")

    x = my_pt.with_columns([
        pl.col('isin'),
        pl.when(~is_callable)  # non-callable
        .then(pl.lit(TO_MATURITY))

        .when(is_called)  # already called / in called state
        .then(pl.lit(TO_MATURITY))

        .when(is_perpetual | is_hybrid)  # perps/hybrids
        .then(pl.lit(TO_CALL))

        .when(is_makewhole_only)  # make-whole treated as non-callable for convention
        .then(pl.lit(TO_MATURITY))

        # default callable convention for “corp/agency/etc”
        .otherwise(pl.lit(TO_WORST)).alias("tradeToConvention")
    ])

    return x.select(['isin', 'tradeToConvention'])


@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'region'])
async def kdb_smad_bondLevelData(my_pt, region="US", dates=None, **kwargs):
    syms = await get_syms(my_pt, isin_col='sym', csp_col='cusip')
    triplet = construct_gateway_triplet("smad", region, "bondLevelData")
    cols = [
        'description', 'esmp', 'cusip', 'isRule144A', 'isRegS', 'issuerName', 'ratingSandP:ratingSANDP', 'ratingMoody',
        'ticker', 'industrySector', 'issueDate', 'currency:issueCurrency',
        'maturityDate:currentMaturityDate', 'nextCallDate', 'amountOutstanding:principalAmountOutstanding', 'country',
        'smadRating:rating', 'idiosyncraticVar', 'smadMaturityBucket:maturity',
        'smadMaturity:maturityRiskLimit', 'smadSector:sector', 'smadRanking:ranking', 'smadRatingSpread:ratingSpread',
        'smadLiqScore:liquidityScore', 'smadLiquidity:liquidity',
        'smadBenchmarkIsin:benchmark', 'benchmarkRoll', 'smadBeta:beta', 'coupon:couponRate',
        'couponType:couponDividendType'
    ]
    my_cols = kdb_col_select_helper(cols)

    my_dates = next_biz_date(dates, -3) if is_today(dates, utc=True) else dates
    q = build_pt_query(triplet, cols=my_cols, by="isin:sym", dates=my_dates, date_kwargs={'as_min': True}, filters=syms, raw_filter=True)
    r = await query_kdb(q, config=fconn(GATEWAY, region=region))
    if r is not None:
        r = r.with_columns([
            pl.when(pl.col("smadMaturity").len() > 0).then(
                pl.col("smadMaturity").str.replace_all("MATURITY_", "")).otherwise(None).alias("smadMaturity"),
            pl.when(pl.col("smadSector").len() > 0).then(
                pl.col("smadSector").str.replace_all("SECTOR_", "")).otherwise(None).alias("smadSector"),
            pl.when(pl.col("smadLiquidity").len() > 0).then(
                pl.col("smadLiquidity").cast(pl.Utf8).str.replace_all("LIQUIDITY_SCORE_", "")).otherwise(
                None).alias("smadLiquidity"),
            pl.when(pl.col("smadRanking").len() > 0).then(
                pl.col("smadRanking").cast(pl.Utf8).str.replace_all("RANKING_", "")).otherwise(
                None).alias("smadRanking"),
            pl.when(pl.col("smadRatingSpread").len() > 0).then(
                pl.col("smadRatingSpread").cast(pl.Utf8).str.replace_all("RATINGSPREAD_", "")).otherwise(
                None).alias("smadRatingSpread"),
            pl.when(pl.col("smadRating").len() > 0).then(
                pl.col("smadRating").cast(pl.Utf8).str.replace_all("RATING_", "")).otherwise(
                None).alias("smadRating"),
        ])
        ref = pl.concat(
            [
                my_pt.lazy().select("id").join(r, left_on="id", right_on="cusip", how="inner", coalesce=False),
                my_pt.lazy().select("id").join(r, left_on="id", right_on="isin", how="inner", coalesce=False)
            ]
        )
        return ref


# ========================================================
## Static Enhancements
# ========================================================

async def isin_countries(my_pt, **kwargs):
    return my_pt.select([
        pl.col('isin'),
        pl.col('isin').str.slice(0, 2).alias('_isinCountry')
    ])


custom_country_data = pl.DataFrame({
    'name_short': ['European Union', 'OTC Derivative', 'Euroclear', 'Etrading'],
    'name_official': ['European Union', 'OTC Derivative', 'Euroclear', 'Etrading'],
    'regex': ['EU', 'EZ', 'XS', 'XT'],
    'ISO2': ['EU', 'EZ', 'XS', 'XT'],
    'ISO3': ['EU', 'EZ', 'XS', 'XT'],
    'EU': ['EU', 'not found', 'EU', 'not found'],
    'APEC': ['not found', 'not found', 'not found', 'not found'],
    'BRIC': ['not found', 'not found', 'not found', 'not found'],
    'continent': ['Europe', 'NA', 'Europe', 'NA'],
    'UNregion': ['Europe', 'NA', 'Europe', 'NA'],
    'MESSAGE': ['WEU', 'not found', 'WEU', 'not found'],
}).to_pandas()
coco = country_converter.CountryConverter(additional_data=custom_country_data)


async def map_regions(my_pt, country_columns=('country',), **kwargs):
    my_pt = ensure_lazy(my_pt)
    # Waterfall
    s = my_pt.hyper.schema()
    valid_cols = [col for col in country_columns if col in s]
    if not valid_cols: return

    country_expr = pl.coalesce(*valid_cols).alias("regionIso")

    iso = (await my_pt.select(country_expr).unique().hyper.collect_async()).to_series()
    iso = iso.filter(iso != 'SNAT').to_pandas()
    temp_pt = pl.LazyFrame([
        pl.Series(iso).alias("regionIso"),
        pl.Series(coco.pandas_convert(iso, to="name_short")).cast(pl.String, strict=False).alias("regionCountry"),
        pl.Series(coco.pandas_convert(iso, to="name_official")).cast(pl.String, strict=False).alias("regionCountryOfficial"),
        pl.Series(coco.pandas_convert(iso, to="ISO2")).cast(pl.String, strict=False).alias("regionIso2"),
        pl.Series(coco.pandas_convert(iso, to="ISO3")).cast(pl.String, strict=False).alias("regionIso3"),
        pl.Series(coco.pandas_convert(iso, to="MESSAGE")).cast(pl.String, strict=False).alias("regionMessage"),
        pl.Series(coco.pandas_convert(iso, to="MESSAGE").map(
            {"AFR": "Sub-Saharan Africa", "CPA": "Centrally planned Asia and China",
             "EEU": "Central and Eastern Europe", "FSU": "Former Soviet Union",
             "LAC": "Latin America and the Caribbean", "MEA": "Middle East and North Africa",
             "NAM": "North America", "PAO": "Pacific OECD", "PAS": "Other Pacific Asia", "SAS": "South Asia",
             "WEU": "Western Europe", "not found": "NA"})).cast(pl.String, strict=False).alias("regionMessageRegion"),
        pl.Series(coco.pandas_convert(iso, to="UNregion")).cast(pl.String, strict=False).alias("regionUN"),
        pl.Series(coco.pandas_convert(iso, to="ccTLD")).cast(pl.String, strict=False).alias("regionTls"),
        pl.Series(coco.pandas_convert(iso, to="continent")).cast(pl.String, strict=False).alias("regionContinent"),
        pl.Series(coco.pandas_convert(iso, to="MESSAGE").map(
            {"AFR": "eu", "CPA": "sgp", "EEU": "eu", "FSU": "eu", "LAC": "nyk", "MEA": "eu", "NAM": "nyk",
             "PAO": "sgp", "PAS": "sgp", "SAS": "sgp", "WEU": "eu", "not found": "NA"})).alias("regionBarclaysDesk")
    ]).with_columns([
        pl.when(pl.col("regionIso2").is_in(["PK", "AF"]))
        .then(pl.lit("eu"))
        .when(pl.col("regionIso2").is_in(["AU"]))
        .then(pl.lit('us'))
        .otherwise(pl.col("regionBarclaysDesk")).alias('regionBarclaysDesk')
    ]).with_columns([
        pl.when(pl.col("regionBarclaysDesk").eq("nyk")).then(pl.lit("US"))
        .when(pl.col("regionBarclaysDesk").eq("eu")).then(pl.lit("EU"))
        .when(pl.col("regionBarclaysDesk").eq("ldn")).then(pl.lit("EU"))
        .when(pl.col("regionBarclaysDesk").eq("sgp")).then(pl.lit("SGP"))
        .when(pl.col("regionBarclaysDesk").eq("asia")).then(pl.lit("SGP"))
        .otherwise(pl.lit("NA")).alias('regionBarclaysRegion')
    ])

    ts = temp_pt.hyper.schema()
    str_cols = [k for k, v in ts.items() if v == pl.String]
    temp_pt = temp_pt.with_columns([
        pl.when(pl.col(col).eq('not found')).then(pl.lit(None, pl.String)).otherwise(pl.col(col)).alias(col) for col in str_cols
    ])

    return my_pt.with_columns(country_expr).select(["isin", "regionIso"]).join(temp_pt, on="regionIso", how="left")


def convert_gics(x):
    result = {
        'gicsSector': None,
        'gicsIndustryGroup': None,
        'gicsIndustry': None,
        'gicsSubIndustry': None
    }
    if x is not None and str(x)!='':
        g = GICS(str(x))
        if hasattr(g, '_levels') and g._levels:
            level1 = g.level(1)
            result["gicsSector"] = level1.get('name', None) if isinstance(level1, dict) else None
            level2 = g.level(2)
            result["gicsIndustryGroup"] = level2.get('name', None) if isinstance(level2, dict) else None
            level3 = g.level(3)
            result["gicsIndustry"] = level3.get('name', None) if isinstance(level3, dict) else None
            level4 = g.level(4)
            result["gicsSubIndustry"] = level4.get('name', None) if isinstance(level4, dict) else None
        for key in result:
            try:
                if result[key] is not None:
                    result[key] = str(result[key])
            except Exception:
                continue
        return result


@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt'])
async def map_gic_sector(my_pt, region="US", dates=None, **kwargs):
    s = my_pt.hyper.schema()

    gics_cols = [
        col for col in
        ['issuerGicsSubIndustry', 'issuerGicsIndustry', 'issuerGicsIndustryGroup', 'issuerGicsSector']
        if col in s
    ]

    if not gics_cols: return
    gics_struct_schema = pl.Struct([
        pl.Field("gicsSector", pl.String),
        pl.Field("gicsIndustryGroup", pl.String),
        pl.Field("gicsIndustry", pl.String),
        pl.Field("gicsSubIndustry", pl.String)
    ])

    return my_pt.select([
        pl.col("isin"),
        pl.coalesce(gics_cols).alias('_gicsMap')
    ]).filter(pl.col("_gicsMap").is_not_null()).select([
        pl.col('isin'),
        pl.when(pl.col('_gicsMap').is_not_null())
        .then(
            pl.col("_gicsMap").map_elements(convert_gics, return_dtype=gics_struct_schema)
        ).otherwise(pl.lit(None, gics_struct_schema))
        .alias("gicsStruct")
    ]).unnest('gicsStruct').drop(['_gicsMap'], strict=False)


# ========================================================
## ALGO
# ========================================================


SIGNAL_MAP = {
    "Credit Momentum": "signalCreditMom",
    "Credit Reversal": "signalCreditReversal",
    "Equity Momentum": "signalEqMom",
    "Factor Score": "signalFactor",
    "Front end Signal": "signalFrontEnd",
    "FullSignal": "signalTotal",
    "High Frequency": "signalHf",
    "RFQ Imbalance": "signalRfqImbalance",
    "RelVal": "signalRelVal",
    "Total Signal Rank": "signalTotalRank",
    "Trace Imbalance": "signalTraceImbalance",
    "traceImbalance": "signalTraceImbalance",
    "indexRebalance": "signalIndexRebalance",
    "relVal": "signalRelVal",
    "creditMomentum": "signalCreditMom",
    "equityMomentum": "signalEqMom",
    "totalSignalRank": "aggSignal",
    "signalTotalRank": "aggSignal",
    "signalBpsTotalRank": "aggSignal",
    "signalPxTotalRank": "aggSignal",
    "creditReversal": "signalCreditReversal",
    "rfqImbalance": "signalRfqImbalance",
    "highFrequency": "signalHf",
    "fullSignal": "signalTotal"
}

KDB_SIGNAL_MAP = {
    'intradayRfqImbalanceSignal': 'rfqImbalance',
    'intradayTraceImbalanceSignal': 'traceImbalance',
    'intradayRelValSignal': 'relVal',
    'intradayEquityMomSignal': 'eqMom',
    'intradayCreditMomSignal': 'creditMom',
    'intradayCreditReversalSignal': 'creditReversal',
    'rtRawTgtDataSourceSignal': 'hf',
    'rebalanceSignal': 'indexRebalance',
    'eodStatsSignal': 'stats',
    'intradayStatsSignal': 'liveStatsRaw',
    'intradayFrontEndSignal': 'liveFrontEnd',
    'rtTotalSignalRank': 'aggSignal'
}

AGG_SIGNAL_MAP = {
    '1': 'STRONG SELL',
    '2': 'SELL',
    '3': 'NEUTRAL',
    '4': 'BUY',
    '5': 'STRONG BUY',
    '0': None
}

async def stats_signals(my_pt, region="US", dates=None, *, books='priority', **kwargs):
    biz = await get_algo_map(region="US", values='algoAsset')
    books = SIGNAL_BOOK_PRIORITY if books=='priority' else books
    books = books if not books is None else (
        await get_algo_books(region=region) if biz is None else list(biz.keys())
    )
    isins = my_pt.hyper.to_kdb_sym('isin')
    q = """{
        oneBook:{[book]
            getForBookSignal:{[book;x] update signalId:x from .credit.common.pricing.utils.getBondTargets[book;x]};
            sc:.credit.common.pricing.parameters.signalComponents;
            sigIds: exec distinct signalId from .credit.common.pricing.parameters.signalComponents;
            t: raze (getForBookSignal[book;] peach sigIds);
            select isin:sym, signalName, book, signal: targetPositionNotional %% scale from ((select from t where sym in (%s)) lj (select first signalName, first scale by signalId from sc))
        };
        signals_base:raze oneBook peach %s;
        if[0=count signals_base;:()];
        signals_base:0^.tutil.pivot2[signals_base; `isin`book; `signalName; enlist `signal];
        missing_signals:(`$"signal_",/: string exec signalName from .credit.common.pricing.parameters.signalComponents) except cols signals_base;
        add0col:{[t;c] ![t; (); 0b; (enlist c)!enlist ((count t)#0)]};
        signals_base:add0col/[signals_base; missing_signals];
        signals_base: update signal_liveStats: signal_liveStatsRaw + signal_indexRebalance + signal_liveFrontEnd from signals_base;
        signals_base: update signal_eodStats: signal_stats+signal_indexRebalance from signals_base;
        signals_base: update signal_total:signal_liveStats+signal_hf from signals_base;
        signals_base
        }[]""" % (isins, "`" + "`".join(books))
    r = await query_kdb(q, fconn(PANOPROXY, region=region))
    if (r is None) or (r.hyper.is_empty()): return
    signal_names = r.hyper.cols_like('^signal')
    signal_names_typed = [s.replace('signal', f'signal{unit}') for unit in ['Px', 'Bps'] for s in signal_names]
    priorities = {v: i for i, v in enumerate(SIGNAL_BOOK_PRIORITY)}
    r = r.hyper.ensure_columns(['book'], default=None, dtypes={'book': pl.String})
    r_full = (await r.with_columns([
        pl.col('book').replace(biz).alias('algoAsset'),
        pl.col('book').replace_strict(priorities, default=99, return_dtype=pl.Int64).alias('_priority'),
        pl.col('book').alias('signalSource'),
    ]).sort('_priority').unique(subset=['isin'], keep='first').drop('_priority').hyper.compress_plan_async()).with_columns([
        pl.when(
            (pl.col('algoAsset').is_not_null() & (pl.col('algoAsset')=='HY'))
        ).then(pl.lit('px', pl.String))
        .otherwise(pl.lit('bps', pl.String))
        .alias('signalUnit')
    ]).join(my_pt.lazy().select([pl.col('isin'), pl.col('unitCs01').abs()]), on='isin', how='left').with_columns([
        pl.when(pl.col('unitCs01').is_not_null() & (pl.col('unitCs01')!=0)).then(
            pl.when(pl.col('signalUnit')=='px')
            .then(1 / pl.col('unitCs01')).otherwise(pl.col('unitCs01'))
        ).otherwise(pl.lit(None, pl.Float64)).alias('_mult')
    ]).unpivot(
        index=['isin', 'signalUnit', '_mult', 'signalSource'],
        on=signal_names,
        variable_name='signalType',
        value_name='signalValue'
    ).with_columns([
        pl.when(pl.col('signalType') != 'signalAggSignal')
        .then(
            pl.when(pl.col('_mult').is_not_null())
            .then((pl.col('_mult') * pl.col('signalValue') * -1))
            .otherwise(pl.lit(None, pl.Float64))
        ).otherwise(pl.col('signalValue'))
        .alias('signalValue_other'),
        pl.when(pl.col('signalUnit')=='px').then(
            pl.col('signalType').str.replace('^signal', 'signalPx')
        ).otherwise(
            pl.col('signalType').str.replace('^signal', 'signalBps')
        ).alias('signalTypedName'),
        pl.when(pl.col('signalUnit')=='px').then(
            pl.col('signalType').str.replace('^signal', 'signalBps')
        ).otherwise(
            pl.col('signalType').str.replace('^signal', 'signalPx')
        ).alias('signalTypedName_other')
    ])

    signals = (pl.concat([
        r_full.select([
            pl.col('isin'),
            pl.col('signalTypedName'),
            pl.col('signalValue')
        ]),
        r_full.select([
            pl.col('isin'),
            pl.col('signalTypedName_other').alias('signalTypedName'),
            pl.col('signalValue_other').alias('signalValue')
        ]),
    ], how="vertical_relaxed").unique(['isin', 'signalTypedName'])
    .pivot(
        index=['isin'],
        on=['signalTypedName'],
        on_columns=signal_names_typed,
        values=['signalValue']
    )).hyper.ensure_columns(['signalPxAggSignal', 'signalBpsAggSignal']).with_columns([
        pl.coalesce([
            pl.col('signalPxAggSignal'),
            pl.col('signalBpsAggSignal')
        ]).cast(pl.Int64, strict=False).cast(pl.String, strict=False).replace(AGG_SIGNAL_MAP).alias('signalMnemonic')
    ])
    return my_pt.select('isin', 'side').join(signals, on='isin', how='left').with_columns([
        pl.when(pl.col('signalMnemonic').is_not_null()).then(
            pl.when(pl.col('signalMnemonic')=='NEUTRAL').then(pl.lit('N', pl.String))
            .when(pl.col('side')=='BUY').then(
                pl.when(pl.col('signalMnemonic')=='STRONG SELL').then(pl.lit('SSU', pl.String))
                .when(pl.col('signalMnemonic')=='SELL').then(pl.lit('SU', pl.String))
                .when(pl.col('signalMnemonic')=='BUY').then(pl.lit('SA', pl.String))
                .when(pl.col('signalMnemonic')=='STRONG BUY').then(pl.lit('SSA', pl.String))
                .otherwise(pl.col('signalMnemonic'))
            ).otherwise(
                pl.when(pl.col('signalMnemonic')=='STRONG BUY').then(pl.lit('SSU', pl.String))
                .when(pl.col('signalMnemonic')=='BUY').then(pl.lit('SU', pl.String))
                .when(pl.col('signalMnemonic')=='SELL').then(pl.lit('SA', pl.String))
                .when(pl.col('signalMnemonic')=='STRONG SELL').then(pl.lit('SSA', pl.String))
                .otherwise(pl.col('signalMnemonic'))
            )
        ).otherwise(pl.lit('NoS', pl.String))
        .alias('signalFlag')
    ]).drop(['side', 'aggSignal'], strict=False)


async def signal_filler(my_pt, region="US", dates=None, **kwargs):
    return my_pt.select([
        pl.col('tnum'),
        pl.lit('NO SIGNAL', pl.String).alias('signalMnemonic'),
        pl.lit('NoS', pl.String).alias('signalFlag'),
    ])


async def us_algo_eligibility(my_pt, region="US", dates=None, **kwargs):
    books = (await get_algo_businesses(region="US")).hyper.to_map('algoBusiness', 'algoAsset')
    isins = my_pt.hyper.to_kdb_sym('isin')
    qs = [f"(select isin, asset:`{asset} from ([] isin:.credit.common.pricing.getListOfIsins[`{business}; 1b]) where isin in (%s))" % isins for business, asset in books.items() if
          'OVERNIGHT' not in business]
    if not qs: return
    res = await query_kdb(",".join(qs), fconn(PANOPROXY_US))
    if res is None: return
    ub = list(set(books.values()))

    return res.with_columns([
        pl.lit(1, pl.Int8).alias('_isInAlgoUniverse')
    ]).pivot(index='isin', on='asset', on_columns=ub, aggregate_function='first').fill_null(pl.lit(0, pl.Int8)).rename({
        "MUNI": 'isMuniAlgoEligible',
        'HG': 'isIgAlgoEligible',
        'HY': 'isHyAlgoEligible',
        'EM': 'isEmAlgoEligible',
        'HYBRID': 'isHybridAlgoEligible',
    }, strict=False).with_columns([
        pl.lit(1, pl.Int8).alias('isInAlgoUniverse'),
    ])


async def realtime_signals_us(my_pt, region="US", dates=None, **kwargs):
    bonds = my_pt.hyper.to_list('isin')
    triplet = construct_gateway_triplet('creditone', "US", 'realtimeAlgoSignals')
    cols = kdb_col_select_helper(['signalValue', 'signalRefreshTime:eventTimestamp', "signalUnit:unit"], method="last")
    q = build_pt_query(triplet, cols, dates=dates,
                       filters={'targetBond': bonds},
                       by=['isin:targetBond', 'signalType:sym']
                       )
    return await query_kdb(q, config=fconn(GATEWAY, region="US"))


async def realtime_signals_eu(my_pt, region="EU", dates=None, **kwargs):
    bonds = my_pt.hyper.to_list('isin')
    triplet = construct_gateway_triplet('creditone', "EU", 'algoSignalsEMEA')
    cols = ['creditMomentum', 'creditReversal', 'indexRebalance', 'traceImbalance:tapeImbalance', 'relVal', 'equityMomentum', 'highFrequency', 'rfqImbalance', 'fullSignal', 'totalSignalRank',
            'signalRefreshTime:sendingTime']
    q = build_pt_query(triplet, dates=dates,
                       cols=cols,
                       filters={'sym': bonds},
                       by=['isin:sym', 'signalUnit:unit']
                       )
    res = await query_kdb(q, config=fconn(GATEWAY, region="EU"))
    if res is None: return
    return res.unpivot(on=['creditMomentum', 'creditReversal', 'indexRebalance', 'traceImbalance', 'relVal', 'equityMomentum', 'highFrequency', 'rfqImbalance', 'fullSignal', 'totalSignalRank'], index=['isin', 'signalRefreshTime', 'signalUnit'], value_name='signalValue', variable_name='signalType')


async def realtime_signals(my_pt, region="US", dates=None, **kwargs):
    my_pt = ensure_lazy(my_pt)
    if region=="EU":
        r = await realtime_signals_eu(my_pt, dates=dates, **kwargs)
    else:
        r = await realtime_signals_us(my_pt, dates=dates, **kwargs)

    if (r is None) or (r.hyper.is_empty()): return

    mini_pt = my_pt.select([
        pl.col('isin'),
        pl.col('unitCs01').abs()
    ]).filter(pl.col('unitCs01').is_not_null() & (pl.col('unitCs01') > 0))

    full = r.join(mini_pt, on='isin', how='left').with_columns([
        pl.when(pl.col('signalUnit')=='bps').then(pl.col('unitCs01')).otherwise(1 / pl.col('unitCs01')).alias('_mult'),
    ]).with_columns(pl.col('signalType').replace(SIGNAL_MAP)).with_columns([
        pl.when(pl.col('signalType')!="aggSignal")
        .then(pl.col('signalValue') * pl.col('_mult'))
        .otherwise(pl.col('signalValue'))
        .alias('signalOther')
    ]).with_columns([
        pl.when(pl.col('signalUnit')=='cents')
        .then(pl.col('signalType').str.replace('signal', 'signalPx'))
        .otherwise(pl.col('signalType').str.replace("signal", "signalBps"))
        .alias('signalType'),
        pl.when(pl.col('signalUnit')=='cents')
        .then(pl.col('signalType').str.replace('signal', 'signalBps'))
        .otherwise(pl.col('signalType').str.replace("signal", "signalPx"))
        .alias('signalOtherType')
    ])

    prep = (await pl.concat([
        full.select([
            pl.col('isin'),
            pl.col('signalType'),
            pl.col('signalValue'),
            pl.col('signalRefreshTime')
        ]),
        full.select([
            pl.col('isin'),
            pl.col('signalOtherType').alias('signalType'),
            pl.col('signalOther').alias('signalValue'),
            pl.col('signalRefreshTime')
        ]),
    ], how='vertical').hyper.collect_async()).with_columns([pl.col('signalType').replace(SIGNAL_MAP)])

    signals = prep.pivot(
        on=['signalType'],
        index=['isin'],
        values=['signalValue'],
        aggregate_function='first',
        separator=""
    ).with_columns([
        pl.col('aggSignal').cast(pl.Int64, strict=False).cast(pl.String, strict=False).replace(AGG_SIGNAL_MAP).alias('signalMnemonic')
    ]).lazy()

    return my_pt.select('isin', 'side').join(signals, on='isin', how='left').with_columns([
        pl.when(pl.col('signalMnemonic').is_not_null()).then(
            pl.when(pl.col('signalMnemonic')=='NEUTRAL').then(pl.lit('N', pl.String))
            .when(pl.col('side')=='BUY').then(
                pl.when(pl.col('signalMnemonic')=='STRONG SELL').then(pl.lit('SSU', pl.String))
                .when(pl.col('signalMnemonic')=='SELL').then(pl.lit('SU', pl.String))
                .when(pl.col('signalMnemonic')=='BUY').then(pl.lit('SA', pl.String))
                .when(pl.col('signalMnemonic')=='STRONG BUY').then(pl.lit('SSA', pl.String))
                .otherwise(pl.col('signalMnemonic'))
            ).otherwise(
                pl.when(pl.col('signalMnemonic')=='STRONG BUY').then(pl.lit('SSU', pl.String))
                .when(pl.col('signalMnemonic')=='BUY').then(pl.lit('SU', pl.String))
                .when(pl.col('signalMnemonic')=='SELL').then(pl.lit('SA', pl.String))
                .when(pl.col('signalMnemonic')=='STRONG SELL').then(pl.lit('SSA', pl.String))
                .otherwise(pl.col('signalMnemonic'))
            )
        ).otherwise(pl.lit('NoS', pl.String))
        .alias('signalFlag')
    ]).drop(['side', 'aggSignal'], strict=False)

    # 'signalRefreshTime'

    # TOTAL SIGNAL RANK [1-5]
    # 1 - STRONG SELL, 2 - SELL, 3 - NEUTRAL, 4 - BUY, 5 - STRONG BUY.


# ========================================================
## Data
# ========================================================

async def dedupe_pt_sides(my_pt, group='isin', key='side', value='netSize'):
    g = ensure_list(group)
    return my_pt.group_by(g).agg(
        pl.col(value).sum()
    ).with_columns([
        pl.when(pl.col(value) >= 0)
            .then(pl.lit("BUY", pl.String))
            .otherwise(pl.lit('SELL', pl.String))
        .alias(key)
    ])

@hypercache.cached(ttl=timedelta(hours=6), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt'])
async def internal_liquidity_score(my_pt, region="US", dates=None, pano=False, **kwargs):
    my_pt = ensure_lazy(my_pt)
    isins = my_pt.hyper.to_list('isin')
    triplet = construct_panoproxy_triplet("US", 'liquidityScores', None)
    cols = ['liqBkt10ScaleSell', 'liqBkt10ScaleBuy']
    q = build_pt_query(
        table=triplet if pano else 'liquidityScores',
        cols=kdb_col_select_helper(cols, method='last'),
        dates=None,
        date_kwargs={'return_today': False},
        filters={'sym': isins},
        by=['isin:sym', 'model']
    )
    r: pl.LazyFrame = await query_kdb(q, fconn(PANOPROXY if pano else SMAD, region='US'))
    if r is None: return
    deduped_pt = await dedupe_pt_sides(my_pt)
    return r.join(deduped_pt.select('isin', 'side'), on='isin', how='left').select([
        pl.col('isin'),
        pl.col('model'),
        pl.when(pl.col('side')=='BUY')
            .then(pl.col('liqBkt10ScaleSell'))
            .otherwise(pl.col('liqBkt10ScaleBuy'))
            .round(3)
            .alias('dkLiqScore')
    ]).pivot(index='isin', on='model', values=['dkLiqScore'], on_columns=['BLSPlus', 'portfolio']).rename({
        'BLSPlus': 'blsLiqScore',
        'portfolio': 'dkLiqScore'
    })

async def funding_rate(my_pt, region="US", dates=None, **kwargs):
    isins = my_pt.hyper.ul('isin')
    cols = [
        'fundingQuantityOnLoan:quantityOnLoan', 'fundingGenAvail:genAvail',
        'fundingOfferedRate:offeredRate', 'fundingOfferedFee:offeredFee',
        'fundingIndicator:indicator'
    ]
    q = build_pt_query(
        '.mt.get[`.credit.us.fundingRate.realtime]',
        cols=kdb_col_select_helper(cols, method='last'),
        by='isin:sym',
        dates=dates,
        date_kwargs={'return_today': False},
        filters={'sym': isins}
    )
    return await query_kdb(q, config=fconn(PANOPROXY_US))


def safe_date_lookback(dates, adj=-1):
    adj = -1 * adj if adj > 0 else adj
    d = [dates] if not isinstance(dates, (list, tuple)) else dates
    t = get_today(utc=True)
    d = sorted([e if e <= t else t for e in [parse_date(e) for e in d]])
    d = np.unique(d).tolist()
    if len(d)==1:
        sd, ed = add_business_days(d[0], adj), d[0]
        d = [sd, ed] if adj < 0 else [ed, sd]
    return d


async def get_books_from_last_name(last_names):
    last_names = last_names if isinstance(last_names, list) else [last_names]
    return (await cached_book_maps()).filter(pl.col('traderLastName').is_in(last_names))


async def get_books_from_full_name(names):
    names = names if isinstance(names, list) else [names]
    onames = names.copy()
    names = [n.title() for n in names]

    traders, algos = set(), set()
    for ai in range(len(names)):
        an = names[ai]
        al = an.lower()
        if 'algo' in al:
            if 'ig' in al:
                algos.add('IG')
                names[ai] = ALGO_PROFILES.get('IG', {}).get('traderName', None)
            elif 'hy' in al:
                algos.add('HY')
                names[ai] = ALGO_PROFILES.get('HY', {}).get('traderName', None)
            elif 'em' in al:
                algos.add('EM')
                names[ai] = ALGO_PROFILES.get('EM', {}).get('traderName', None)
            else:
                algos.add('IG')
                names[ai] = ALGO_PROFILES.get('IG', {}).get('traderName', None)
        else:
            pa = an.title()
            names[ai] = pa
            traders.add(pa)

    algo_packets = [ALGO_PROFILES.get(a, {}) for a in algos]
    algo_results = pl.DataFrame(algo_packets).with_columns(pl.lit(0).alias('livePosition'))

    if traders:
        trader_results = (await cached_book_maps()).filter(pl.col('traderName').is_in(list(traders)))
    else:
        trader_results = pl.DataFrame()

    results = pl.concat([trader_results, algo_results], how="diagonal_relaxed")
    name_map = pl.DataFrame({'traderName': names, '_passedName': onames})
    return results.join(name_map, on='traderName', how='left')


# TODO
async def coop_bonds_from_rfqs(region="US", dates=None, lookback=365, **kwargs):
    lookback = lookback * -1 if lookback > 0 else lookback
    d = safe_date_lookback(dates, lookback)
    triplet = construct_gateway_triplet("credit", region, "bondrfqs")
    cols = ['coopStatus']
    q = build_pt_query(
        triplet,
        by=f'isin:sym',
        # rfqResponder:lower[rfqRespHydrTrader],
        cols=kdb_col_select_helper(cols, method='last'),
        dates=d,
        date_kwargs={'between': True, 'as_sequence': False},
        filters={
            'not coopStatus': None,
        }
    )
    return await query_kdb(q, config=fconn(GATEWAY), name=f"credit")


async def junior_traders(regions=("US", "EU"), lookback=30, **kwargs):
    tasks = [
        asyncio.create_task(
            junior_traders_regional(region=r, dates=None, lookback=lookback)
        ) for r in regions
    ]
    js = await asyncio.gather(*tasks, return_exceptions=True)
    js = [x for x in js if isinstance(x, pl.LazyFrame)]
    return pl.concat(js, how='vertical', strict=False).unique()


@hypercache.cached(ttl=timedelta(hours=12))
async def junior_traders_regional(region="US", dates=None, lookback=30, **kwargs):
    lookback = lookback * -1 if lookback > 0 else lookback
    d = safe_date_lookback(dates, lookback)
    triplet = construct_gateway_triplet("credit", region, "bondrfqs")
    cols = ['lower[rfqDealerTrader]', 'rfqDealerTraderName', 'upper[rfqL0Book]', 'lower[rfqRespHydrTrader]', 'lower[lastSentUser]']
    q = build_pt_query(
        triplet,
        by=cols,
        # rfqResponder:lower[rfqRespHydrTrader],
        cols=kdb_col_select_helper(["date"], method="last"),
        dates=d,
        date_kwargs={'between': True, 'as_sequence': False},
        filters={"not rfqL0Book": ""}
    )
    z_raw = await query_kdb(q, config=fconn(GATEWAY), name=f"credit")
    if (z_raw is None) or (z_raw.hyper.is_empty()): return
    z_raw = z_raw.with_columns([
        pl.col('rfqDealerTraderName').str.to_titlecase().alias('rfqDealerTraderName')
    ])
    traders = (await book_maps()).select([
        pl.col('traderName').str.to_titlecase(),
        pl.col('traderName').str.split(" ").list.last().str.to_titlecase().alias('_traderLastName'),
        pl.col('traderId').str.to_lowercase(),
        pl.col('bookId').str.to_uppercase()
    ])

    tn = traders.select('traderName')
    tid = traders.select('traderId')

    z = z_raw.join(tid, left_on="rfqDealerTrader", right_on="traderId", how="anti")
    z = z.join(tn, left_on="rfqDealerTraderName", right_on="traderName", how="anti")
    z = z.join(tid, left_on="lastSentUser", right_on="traderId", how="anti")
    z = z.join(tid, left_on="rfqRespHydrTrader", right_on="traderId", how="anti")

    zmin = z.unpivot(index='rfqL0Book', on=[pl.col('rfqDealerTrader'), pl.col('lastSentUser'), pl.col('rfqRespHydrTrader')]).select([
        pl.col('rfqL0Book').alias('bookId'),
        pl.col('value').alias('juniorId'),
    ]).drop_nulls().unique().join(traders, on='bookId', how='inner').filter([
        ~(
                pl.col('juniorId').is_in(BAD_USERNAMES) |
                pl.col('traderId').is_in(BAD_USERNAMES) |
                pl.col('bookId').is_in(BAD_BOOKS)
        )
    ])

    from app.server import get_pb
    users = (await get_pb()).select([
        pl.col('username').alias('juniorId'),
        pl.col('lastName').alias('_runzSenderLastName')
    ]).join(zmin, on='juniorId', how='inner')

    return users.select([
        pl.col('_runzSenderLastName'),
        pl.col('_traderLastName'),
        pl.col('traderName').alias('_traderName')
    ]).unique()


@alru_cache(ttl=6 * 12 * 12)
async def _get_book_maps(region, *, allow_bad_books=False, allow_bad_usernames=False, allow_bad_desks=False, allow_all=False):
    triplet = construct_gateway_triplet("credit", region, "bondpositions")
    cols = ['traderFirstName', 'traderLastName', 'traderId', 'deskName', 'livePosition:sum[abs[position]], desigCount:sum[desig]']
    filters = []
    if not allow_all:
        if not allow_bad_books: filters.append('not upper[bookId] in (%s)' % kdb_convert_series_to_sym(BAD_BOOKS))
        if not allow_bad_usernames: filters.append('not lower[traderId] in (%s)' % kdb_convert_series_to_sym(BAD_USERNAMES))

    filter = ", ".join(filters) if filters else None
    q = build_pt_query(
        triplet,
        by=f'bookId',
        cols=kdb_col_select_helper(cols, method="last"),
        dates="T",
        date_kwargs={'between': False},
        filters=filter,
        raw_filter=True
    )
    data = await query_kdb(q, config=fconn(GATEWAY), name=f"credit", lazy=False)
    return data.with_columns([
        pl.lit(region).str.to_uppercase().alias('traderRegion'),
        pl.concat_str([pl.col("traderFirstName"), pl.col("traderLastName")], separator=" ").str.to_titlecase().alias("traderName"),
        pl.col('traderId').str.to_lowercase().alias('traderId'),
        pl.col('bookId').str.to_uppercase().alias('bookId')
    ])


@hypercache.cached(ttl=timedelta(hours=6))
async def book_maps_eu():
    dates = latest_biz_date(now_datetime(False), True)
    triplet = construct_panoproxy_triplet("EU", 'bondpositions', dates)
    _q = 'select traderRegion:`EU, traderId:first lower[traderId], first traderFirstName, first traderLastName, first deskType, first deskName,  bigSize:sum[(abs[position]+abs[openPosition])], usage:sum[(not (position = openPosition))*desig], livePosition:sum[abs[position]], desigCount:sum[desig] by bookId from %s' % triplet
    if not is_today(dates):
        _q += f' where date={dates.strftime("%Y.%m.%d")}, not ((instrumentType=`GOVT) & (ticker=`T))'
    else:
        _q += f'where not ((instrumentType=`GOVT) & (ticker=`T))'
    return await query_kdb(_q, fconn(PANOPROXY_EU))


@hypercache.cached(ttl=timedelta(hours=6))
async def book_maps_us():
    dates = latest_biz_date(now_datetime(False), True)
    triplet = construct_panoproxy_triplet("US", 'bondpositions', dates)
    _q = 'select traderRegion:`US, traderId:first lower[traderId], first traderFirstName, first traderLastName, first deskType, first deskName,  bigSize:sum[(abs[position]+abs[openPosition])], usage:sum[(not (position = openPosition))*desig], livePosition:sum[abs[position]], desigCount:sum[desig] by bookId from %s' % triplet
    if not is_today(dates):
        _q += f' where date={dates.strftime("%Y.%m.%d")}, not ((instrumentType=`GOVT) & (ticker=`T))'
    else:
        _q += f'where not ((instrumentType=`GOVT) & (ticker=`T))'
    return await query_kdb(_q, fconn(PANOPROXY_US))


@hypercache.cached(ttl=timedelta(hours=6))
async def book_maps_sgp():
    dates = latest_biz_date(now_datetime(False), True)
    triplet = construct_panoproxy_triplet("SGP", 'bondpositions', None)
    _q = 'select traderRegion:`SGP, traderId:first lower[traderId], first traderFirstName, first traderLastName, first deskType, first deskName,  bigSize:sum[(abs[position]+abs[openPosition])], usage:sum[(not (position = openPosition))*desig], livePosition:sum[abs[position]], desigCount:sum[desig] by bookId from %s' % triplet
    if not is_today(dates):
        _q += f' where date={dates.strftime("%Y.%m.%d")}, not ((instrumentType=`GOVT) & (ticker=`T))'
    else:
        _q += f'where not ((instrumentType=`GOVT) & (ticker=`T))'
    return await query_kdb(_q, fconn(PANOPROXY_US))


@hypercache.cached(ttl=timedelta(hours=6))
async def book_maps_us_and_sgp():
    dates = latest_biz_date(now_datetime(False), True)
    q_us = 'select traderRegion:`US, traderId:first lower[traderId], first traderFirstName, first traderLastName, first deskType, first deskName, bigSize:sum[(abs[position]+abs[openPosition])], usage:sum[(not (position = openPosition))*desig], livePosition:sum[abs[position]], desigCount:sum[desig] by bookId from %s' % construct_panoproxy_triplet("US", 'bondpositions',dates)
    q_sgp = 'select traderRegion:`SGP, traderId:first lower[traderId], first traderFirstName, first traderLastName, first deskType, first deskName,  bigSize:sum[(abs[position]+abs[openPosition])], usage:sum[(not (position = openPosition))*desig], livePosition:sum[abs[position]], desigCount:sum[desig] by bookId from %s' % construct_panoproxy_triplet("SGP",'bondpositions',dates)
    if not is_today(dates):
        q_us += f' where date={dates.strftime("%Y.%m.%d")}, not ((instrumentType=`GOVT) & (ticker=`T))'
        q_sgp += f' where date={dates.strftime("%Y.%m.%d")}, not ((instrumentType=`GOVT) & (ticker=`T))'
    else:
        q_us += f' where not ((instrumentType=`GOVT) & (ticker=`T))'
        q_sgp += f' where not ((instrumentType=`GOVT) & (ticker=`T))'
    _q = ",".join([f"({x})" for x in (q_us, q_sgp)])
    return await query_kdb(_q, fconn(PANOPROXY_US))


ERROR_LIST = (Exception, asyncio.Timeout, asyncio.CancelledError, BaseException)

def get_books_from_names(names, books: pl.LazyFrame, top_only=True):
    if isinstance(names, str):
        names = [names]

    idx = pl.LazyFrame({
        "_key": [n.lower() if n is not None else None for n in names],
        "_ord": range(len(names)),
    })

    b = books.with_columns(
        pl.col("traderName").str.to_lowercase().alias("_tn"),
        pl.col("traderLastName").str.to_lowercase().alias("_tl"),
        pl.col("traderFirstName").str.to_lowercase().alias("_tf"),
    )

    orig_cols = books.collect_schema().names()

    m1 = idx.join(b, left_on="_key", right_on="_tn", how="left").select("_ord", *orig_cols, pl.lit(1).alias("_p"))
    unmatched1 = m1.filter(pl.col(orig_cols[0]).is_null()).select("_ord", "_key" if False else pl.col("_ord").alias("_ord"))

    m1f = m1.filter(pl.col(orig_cols[0]).is_not_null())
    rest1 = idx.join(m1f.select("_ord"), on="_ord", how="anti")

    m2 = rest1.join(b, left_on="_key", right_on="_tl", how="left").select("_ord", *orig_cols, pl.lit(2).alias("_p"))
    m2f = m2.filter(pl.col(orig_cols[0]).is_not_null())
    rest2 = rest1.join(m2f.select("_ord"), on="_ord", how="anti")

    m3 = rest2.join(b, left_on="_key", right_on="_tf", how="left").select("_ord", *orig_cols, pl.lit(3).alias("_p"))

    if top_only:
        return pl.concat([m1f, m2f, m3], how="vertical").unique(subset='_ord', keep='first').sort("_ord").select(*orig_cols)
    return pl.concat([m1f, m2f, m3], how="vertical").sort("_ord").select(*orig_cols)

@hypercache.cached(ttl=timedelta(hours=6), schema_version=4)
async def book_maps(regions=("US", "EU", "SGP")):
    if not regions: return
    regions = ensure_set(regions)
    tasks = []
    if "EU" in regions:
        tasks.append(asyncio.create_task(book_maps_eu()))

    dual_flag = -1
    if ("US" in regions) and ("SGP" in regions):
        dual_flag = 1 if "EU" in regions else 0
        tasks.append(asyncio.create_task(book_maps_us_and_sgp()))
    elif "US" in regions:
        tasks.append(asyncio.create_task(book_maps_us()))
    elif "SGP" in regions:
        tasks.append(asyncio.create_task(book_maps_sgp()))

    res = await asyncio.gather(*tasks, return_exceptions=True)
    if (dual_flag!=-1) and isinstance(res[dual_flag], ERROR_LIST):
        us_only = await book_maps_us()
        res.append(us_only)
    res = [r for r in res if isinstance(r, (pl.DataFrame, pl.LazyFrame))]
    if not res:
        await log.critical("Book Maps are None. Date is likely off.")
        return
    result = pl.concat(res, how='diagonal_relaxed').with_columns([
        pl.concat_str([
            pl.col('traderFirstName').str.to_titlecase(),
            pl.col('traderLastName').str.to_titlecase()
        ], separator=" ").alias("traderName"),
        pl.col('deskType').replace_strict(L1_DESK_MAP, return_dtype=pl.String, default="OTHER").alias('deskAsset')
    ]).hyper.ensure_columns(['usage', 'bigSize', 'livePosition', 'desigCount']).sort(by=['usage', 'bigSize', 'livePosition', 'desigCount'], descending=True)

    # HARD OVERRIDE
    result.with_columns(
        [
            pl.col('traderName').hyper.case(mappings=[
                ('Cormac Walsh', 'Vikram Kaushik'),
                ('Mohammed Almaliky','Vikram Kaushik')
                ], default=pl.col('traderName')
            ).alias('traderName'),
            pl.col('bookId').hyper.case(
                mappings=[
                    ('Cormac Walsh', 'NYC29136'),
                    ('Mohammed Almaliky', 'NYC29136')
                ], default=pl.col('bookId')
            ).alias('bookId')
        ]
    )

    # SOFT OVERRIDES
    overrides = {
        'bookId': {
            '25313': {
                'traderId': 'x01399318',
                'deskAsset': 'EM',
                'traderRegion': 'EU',
                'traderName': 'Surya Singh'
            },
        },
    }

    exprs = []
    for mask_col, col_overrides in overrides.items():
        for k, v in col_overrides.items():
            mask = (pl.col(mask_col)==k)
            override_exprs = [
                pl.when(mask).then(pl.lit(vv)).otherwise(pl.col(kk)).alias(kk)
                for kk, vv in v.items()
            ]
            exprs.extend(override_exprs)
    return result.with_columns(exprs)

async def _clear_book_maps_hypercache():
    await hypercache.clear('book_maps_eu')
    await hypercache.clear('book_maps_us')
    await hypercache.clear('book_maps_sgp')
    await hypercache.clear('book_maps_us_and_sgp')
    await hypercache.clear('book_maps')






async def cached_book_maps(regions=("US", "EU", "SGP"), force=False, allow_bad_books=False, allow_bad_usernames=False, allow_bad_desks=False, allow_all=False):
    if force:
        _get_book_maps.cache_clear()
    res = []
    for region in regions:
        r = await _get_book_maps(region, allow_bad_books=allow_bad_books, allow_bad_usernames=allow_bad_usernames, allow_bad_desks=True, allow_all=allow_all)
        if r is not None:
            res.append(r)

    my_data = pl.concat(res, how="diagonal_relaxed")

    # MANUAL OVERRIDES
    overrides = {
        'bookId': {
            '25313': {
                'traderId': 'x01399318',
                'deskName': 'EU Illiquids',
                'traderFirstName': 'Surya',
                'traderLastName': 'Singh',
                'traderRegion': 'EU',
                'traderName': 'Surya Singh'
            },
        }
    }

    for mask_col, col_overrides in overrides.items():
        for k, v in col_overrides.items():
            mask = (pl.col(mask_col)==k)
            override_exprs = [pl.when(mask).then(pl.lit(vv)).otherwise(pl.col(kk)).alias(kk) for kk, vv in v.items()]
            my_data = my_data.with_columns(override_exprs)

    return my_data


async def _desig_map_books_to_names(my_pt, region="US", dates=None, book_col='bookId', **kwargs):
    cached = await cached_book_maps()
    if not cached is None and not cached.is_empty():
        return my_pt.lazy().select('bookId').unique('bookId').join(cached.lazy(), on='bookId', how='inner')
    cached = await cached_book_maps(force=True)
    if not cached is None and not cached.is_empty():
        return my_pt.lazy().select('bookId').unique('bookId').join(cached.lazy(), on='bookId', how='inner')


async def _desig_phonebook_raw(trading_only=True, **kwargs):
    from app.server import get_pb
    pb = await get_pb(trading_only=trading_only, lazy=True)
    return pb.select([
        pl.col('username').str.to_lowercase().alias('desigTraderId'),
        pl.col('name').str.to_titlecase().alias('desigName'),
        pl.col('brid').str.to_uppercase().alias('desigBrid'),
        pl.col('asset').str.to_uppercase().alias('desigAsset'),
        pl.col('organisationalUnit').str.to_titlecase().alias('desigOrg'),
        pl.col('email').str.to_lowercase().alias('desigEmail'),
        pl.col('region').str.to_uppercase().alias('desigRegion'),
        pl.col('businessArea3').str.to_titlecase().alias('desigBusinessArea3'),
        pl.col('businessArea4').str.to_titlecase().alias('desigBusinessArea4'),
        pl.col('businessArea5').str.to_titlecase().alias('desigBusinessArea5'),
        pl.col('nickname').str.to_titlecase().alias('desigNickname'),
        pl.col('firstName').str.to_titlecase().alias('desigFirstName'),
        pl.col('lastName').str.to_titlecase().alias('desigLastName'),
        pl.col('role').str.to_titlecase().alias('desigRole')
    ]).with_columns([
        pl.when(pl.col('desigRegion')=='NYK').then(pl.lit('US', pl.String))
        .when(pl.col('desigRegion')=="LDN").then(pl.lit("EU", pl.String))
        .when(pl.col('desigRegion')=="SGP").then(pl.lit("SGP", pl.String))
        .otherwise(pl.col('desigRegion')).alias('desigRegion')
    ])


async def desig_phonebook(my_pt, region="US", dates=None, trading_only=True, **kwargs):
    pbe = await _desig_phonebook_raw(trading_only, **kwargs)
    ids = (await my_pt.collect_async()).get_column('desigTraderId').unique().to_list()
    return pbe.filter(pl.col('desigTraderId').is_in(ids)).unique(subset=['desigTraderId'], keep="any")


def _coalesce_asset_class(my_pt, destination="_desigAsset"):
    s = my_pt.collect_schema() if isinstance(my_pt, pl.LazyFrame) else my_pt.schema
    asset_waterfall = {
        'desigAsset': 1,
        'desigDesk': 1,
        'bvalSubAssetClass': 0.5,
        'markitAssetClass': 0.3
    }
    assetCols = {a: v for a, v in asset_waterfall.items() if a in s}
    t = sum(assetCols.values())
    assetCols = {k: (v / t) for k, v in assetCols.items()}
    _LABELS = ["IG", "HY", "EM"]

    def _classify_expr(col_name: str) -> pl.Expr:
        lc = pl.col(col_name).cast(pl.Utf8).str.to_lowercase()
        ig_pat = r"(ig|investment grade|hg|high grade)"
        hy_pat = r"(hy|high yield|distressed|illiquid)"
        em_pat = r"(em|japan|singapore|asia|emerging|nja|ja)"
        loan_pat = r"(loan)"
        return (
            pl.when(lc.is_null()).then(pl.lit(None))
            .when(lc.str.contains(ig_pat)).then(pl.lit("IG"))
            .when(lc.str.contains(hy_pat)).then(pl.lit("HY"))
            .when(lc.str.contains(em_pat)).then(pl.lit("EM"))
            .when(lc.str.contains(loan_pat)).then(pl.lit("LOAN"))
            .otherwise(pl.lit("OTHER"))
        )

    _cat_aliases = {c: f"{c}__cat" for c in assetCols.keys()}
    return my_pt.select(
        [pl.col("tnum"), *[_classify_expr(c).alias(_cat_aliases[c]) for c in assetCols.keys()]]
    ).with_columns(
        pl.sum_horizontal(
            [
                pl.when(pl.col(_cat_aliases[c])=="IG").then(v).otherwise(0)
                for c, v in assetCols.items()
            ]
        ).alias("ig_score"),
        pl.sum_horizontal(
            [
                pl.when(pl.col(_cat_aliases[c])=="HY").then(v).otherwise(0)
                for c, v in assetCols.items()
            ]
        ).alias("hy_score"),
        pl.sum_horizontal(
            [
                pl.when(pl.col(_cat_aliases[c])=="EM").then(v).otherwise(0)
                for c, v in assetCols.items()
            ]
        ).alias("em_score"),
    ).with_columns(
        pl.concat_list([pl.col("ig_score"), pl.col("hy_score"), pl.col("em_score")]).list.arg_max().alias("_max_idx")
    ).with_columns(pl.lit(_LABELS).list.get(pl.col("_max_idx")).alias(destination)).select(["tnum", destination])


async def assigned_trader(my_pt, region="US", dates=None, include_em=False, **kwargs):
    schema = my_pt.collect_schema()
    if ('whichAlgo' not in schema) or ('desigName' not in schema):
        await log.warn("Missing algo or desig info, cannot assign.")
        return

    em_flag = "~" if include_em else "EM"
    my_pt = my_pt.join(_coalesce_asset_class(my_pt, '_desigAsset'), on='tnum', how='left')

    return my_pt.select([
        pl.col('isin'),
        pl.when(
            pl.col('whichAlgo').is_not_null() & \
            pl.col('whichAlgo').is_in(list(DM_ALGO_BOOKS_PROFILES.keys())) & \
            (pl.col('_desigAsset')!=em_flag) & \
            (pl.col('isHybrid').cast(pl.Int64, strict=False)!=1)
        ).then(pl.concat_str(pl.col('whichAlgo'), pl.lit(" ALGO")))
        .otherwise(pl.col('desigName')).alias('assignedTrader'),

        pl.when(
            pl.col('whichAlgo').is_not_null() & \
            pl.col('whichAlgo').is_in(list(DM_ALGO_BOOKS_PROFILES.keys())) & \
            (pl.col('_desigAsset')!=em_flag) & \
            (pl.col('isHybrid').cast(pl.Int64, strict=False)!=1)
        ).then(pl.col('whichAlgo').replace_strict(DM_ALGO_BOOKS_PROFILES, default=None))
        .otherwise(pl.col('desigBookId')).alias('assignedBookId'),

        pl.when(
            pl.col('whichAlgo').is_not_null() & \
            pl.col('whichAlgo').is_in(list(DM_ALGO_BOOKS_PROFILES.keys())) & \
            (pl.col('_desigAsset')!=em_flag) & \
            (pl.col('isHybrid').cast(pl.Int64, strict=False)!=1)
        ).then(pl.lit(1))
        .otherwise(pl.lit(0)).alias('isAlgoAssigned'),
    ])


def _get_algo_book(asset, region=None):
    return ALGO_PROFILES.get('IG', {}).get('bookId', None)


async def final_cleanup(my_pt):
    s = my_pt.hyper.schema()




# ====================================================================
## S3
# ====================================================================
async def benchmark_quotes(my_pt, region="US", dates=None):
    return

#
# async def s3_market_expansion(my_pt, region="US", dates=None, s3=None, use_cache=True, cache_results=True, include_risk=True, markets=None, include_new_level=False, include_ref_markets=True,
#                               **kwargs):
#     from app.server import get_s3
#     if s3 is None:
#         async def _s3():
#             return get_s3()
#
#         s3 = await _s3()
#     try:
#
#         exclusions = ['cod', 'eod', 'bench', 'Adj', '\d', 'bix']
#         markets = (markets if isinstance(markets, list) else [markets]) if markets else None
#         if not include_new_level:
#             exclusions.append('newLevel')
#         elif (not markets) and (not include_ref_markets):
#             markets = ['newLevel']
#
#         market_cols = market_columns(my_pt, markets=markets, exclusions=exclusions, qt_list=['Px', 'Spd', 'Yld'])
#         payloads = await s3.pt_to_s3_payloads(my_pt, dates, market_cols=market_cols, filter_out_perps=True)
#
#         if use_cache:
#             await s3.cache.clear_expired_entries()
#
#         results = await s3.query_from_payloads_to_dataframe(payloads, use_cache, cache_results)
#         if (results is None) or (results.hyper.is_empty()): return None
#
#         new_levels = results.hyper.cols_like("^(newLevel).*")
#         if new_levels:
#             new_mapping = {n: n.replace("Bid", "").replace("Mid", "").replace("Ask", "") for n in new_levels}
#             results = results.rename(new_mapping, strict=False)
#
#         exprs = []
#         if include_risk:
#             cs01_cols = market_columns(results, exclusions=['bench'], qt_list=['Cs01Pct'])
#             if cs01_cols: exprs.append((pl.mean_horizontal(cs01_cols, ignore_nulls=True) * 100).alias('unitCs01Pct'))
#
#             pv01_cols = market_columns(results, exclusions=['bench'], qt_list=['pv01'])
#             if pv01_cols: exprs.append((pl.mean_horizontal(pv01_cols, ignore_nulls=True)).alias('unitDv01'))
#
#             duration_cols = market_columns(results, exclusions=['MacDuration', 'bench'], qt_list=['duration'])
#             if duration_cols:
#                 exprs.append((pl.mean_horizontal(duration_cols, ignore_nulls=True)).alias('duration'))
#             else:
#                 exprs.append(pl.lit(None, pl.Float64).alias('duration'))
#
#             convex_cols = market_columns(results, exclusions=['bench'], qt_list=['convexity'])
#             if convex_cols: exprs.append((pl.mean_horizontal(convex_cols, ignore_nulls=True)).alias('convexity'))
#
#         market_val_cols = market_columns(results, exclusions=['bench'])
#         exprs.extend([pl.col(c) for c in market_val_cols])
#
#         return results.select([
#                                   pl.col('s3Time').alias('s3RefreshTime'),
#                                   pl.col('isin')
#                               ] + exprs).with_columns([
#             pl.when(pl.col('duration')==0).then(pl.lit(None, pl.Float64)).otherwise(pl.col('duration')).alias('duration')
#         ])
#
#
#     except Exception as e:
#         await log.error(f"Error during S3 expansion: {e}")
#     finally:
#         await s3.close()


def similar(a, b):
    if (not a) or (not b): return 0
    return similarity_score(a, b)


# -----------------------------------
# Consolidation & Utility
# -----------------------------------
async def _book_helper(df, region):
    d = await _desig_map_books_to_names(df, region=region)
    if d is None: return pl.LazyFrame()
    return d.with_columns([
        pl.lit(region).alias('desigRegion')
    ])


async def _map_trader_name_from_book(final_scores):
    regional_tasks = []
    for region in ['US', 'EU', 'SGP']:
        regional_tasks.append(asyncio.create_task(_book_helper(final_scores, region)))
    results = await asyncio.gather(*regional_tasks, return_exceptions=True)
    results = [r for r in results if not isinstance(r, Exception)]
    books = pl.concat(results, how="vertical_relaxed").unique('bookId')
    return books.select([
        pl.col('bookId'),
        pl.col('traderId'),
        pl.col('desigRegion'),
        pl.concat_str([pl.col('traderFirstName'), pl.col('traderLastName')], separator=" ").alias('traderName')
    ])


def _format_rules(rules_list: List[Dict[str, Any]]) -> List[str]:
    active_rules = [r for r in rules_list if r and abs(r.get('contribution', 0.0)) > 1e-6]
    active_rules.sort(key=lambda x: abs(x.get('contribution', 0.0)), reverse=True)
    return [f"{r['name']}: {r['contribution']:.1f}" for r in active_rules[:3]]


def _create_default_explanation(isin: str, assignment_type: str, reason: str = "") -> Dict[str, Any]:
    return {
        "isin": isin,
        "bookId": assignment_type,
        "region": "UNKNOWN",
        "assignmentType": assignment_type,
        "score": 0.0,
        "confidence": 0.0,
    }


#########################################################
## NEW QUOTES
##########################################################
async def fx_rates_realtime(my_pt=None, region="US", dates=None, pairs=('EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF', 'AUDUSD', 'USDCAD'), **kwargs):
    q = 'select last midPrice, last timestamp by sym from .mt.get[`.fx.fox.fxfoxquote.sampledMids.realtime] where sym in (%s)' % kdb_convert_series_to_sym(ensure_list(pairs))
    return await query_kdb(q, fconn(PANOPROXY))


async def quote_cbbt(my_pt, region="US", dates=None, **kwargs):
    dates = parse_date(dates, biz=True)
    if not is_today(dates, True): return
    isins = my_pt.hyper.to_list('isin')
    triplet = construct_gateway_triplet('marketdatacep', region, 'cbbt')
    cols = kdb_col_select_helper(['cbbtBidPx:bid', 'cbbtMidPx:mid', 'cbbtAskPx:ask', 'cbbtRefreshTime:datetime'], 'last')
    q = build_pt_query(triplet, cols, dates=dates, filters={'isin': isins}, by='isin')
    return await query_kdb(q, fconn(GATEWAY))

async def quote_cbbt_pano(my_pt, region="US", dates=None, **kwargs):
    dates = parse_date(dates, biz=True)
    if not is_today(dates, True): return
    isins = my_pt.hyper.ul('isin')
    cols = kdb_col_select_helper([
        'cbbtBidPx:bid', 'cbbtMidPx:mid', 'cbbtAskPx:ask', 'cbbtRefreshTime:datetime'
    ], method='last')
    # All triplets are only defined on US host
    triplet = construct_panoproxy_triplet(region, 'cbbt', None)
    q = build_pt_query(triplet, cols=cols, by="isin:sym", dates=dates, filters={'sym': isins},
                       date_kwargs={"return_today":False}
    )
    return await query_kdb(q, config=fconn(PANOPROXY_US))

async def quote_cbbt_basket(my_pt, region="US", dates=None, **kwargs):
    isins = my_pt.hyper.ul('isin')
    cols = kdb_col_select_helper(['cbbtBidPx:bid', 'cbbtMidPx:(bid+ask)%2', 'cbbtAskPx:ask', 'cbbtRefreshTime:time'], method='last')
    triplet = "lastCBBT"
    q = build_pt_query(triplet, cols=cols, by="isin:sym", dates=None, filters={'sym': isins}, date_kwargs={'incl_param': False, "return_today": False})
    r = await query_kdb(q, config=fconn(BASKET_VIEWER, region="US"), name=f"basket")
    return r

def scale_linear(N, min_key=200, min_value=10, max_key=1000, max_value=20):
    if N <= min_key:
        return min_value
    if N >= max_key:
        return max_value
    t = (N - min_key) / (max_key - min_key)  # t in (0, 1)
    return min_value + t * (max_value - min_value)

def scale_exp(N, k=0.1, min_key=200, min_value=10, max_key=1000, max_value=20):
    import math
    if N <= min_key:
        return min_value
    if N >= max_key:
        return max_value
    t = (N - min_key) / (max_key - min_key)
    ek = math.exp(k)
    eased = (math.exp(k * t))
    return (min_value + min_value * ((eased-1)/(ek-1)))

def get_timeout(my_pt, **kwargs):
    timeout_explicit = kwargs.pop('timeout', None)
    if timeout_explicit: return timeout_explicit
    scale_mode = kwargs.pop('scale', 'exp')
    if scale_mode not in ['linear', 'exp']:
        log.warning(f"unknown scale mode {scale_mode}, switching to exp")
        scale_mode = 'exp'
    min_key = kwargs.pop('min_key', 200)
    min_value = kwargs.pop('min_value', 10)
    max_key = kwargs.pop('max_key', 1000)
    max_value = kwargs.pop('max_value', 20)
    k = kwargs.pop("k", 0.1)
    N = my_pt.hyper.height()
    if scale_mode=="linear":
        return scale_linear(N, min_key, min_value, max_key, max_value)
    return scale_exp(N, k, min_key, min_value, max_key, max_value)

async def quote_cbbt_ext(my_pt, region="US", dates=None, diff_threshold=5, **kwargs):
    _mapp = my_pt.hyper.to_map('bbpk', 'isin', drop_nulls=True, drop_null_keys=True)
    if not _mapp: return
    bb = {k.split(" ")[0]: v for k, v in _mapp.items()}
    triplet = construct_gateway_triplet('externalfeeds', "EU", 'cbbt')
    cols = [
        'cbbtRefreshTime:sendingTime',
        'cbbtBidPx:fills[bid]', 'cbbtMidPx:fills[mid]', 'cbbtAskPx:fills[offer]',
        'cbbtBidSpd:fills[spread_bid]',
        'cbbtMidSpd:fills[spread_mid]',
        'cbbtAskSpd:fills[spread_offer]',
        'cbbtBenchmarkCusip:fills[bMSecurityID]'
    ]
    cols = kdb_col_select_helper(cols, "last")
    dates = latest_biz_date(dates, True)
    q = build_pt_query(triplet, cols, dates=dates, filters={'sym': list(bb.keys())}, by='sym')
    res = await query_kdb(q, fconn(GATEWAY, region=region), timeout=get_timeout(my_pt, **kwargs))
    if res is None: return
    res = res.filter(~((pl.mean_horizontal([pl.col('cbbtBidSpd'), pl.col('cbbtAskSpd')]) - pl.col('cbbtMidSpd')).abs() > diff_threshold))
    res = res.hyper.utc_datetime(time_col='cbbtRefreshTime', date_override=dates, output_name='cbbtRefreshTime')
    return res.with_columns([
        pl.col('sym').replace(bb).alias('isin'),
        ]).drop(['sym', '_bad_mask'], strict=False)

# select from .externalfeeds.ldn.cbbt where date=.z.d, sym=`BH7864169
# NEED BB_IG, NOT THE GLOBAL FIGI BBG ID
# idBbUnique': 'COBH7864169',

async def quote_macp(my_pt, region="US", dates=None, **kwargs):
    dates = parse_date(dates, biz=True)
    if not is_today(dates, True): return
    isins = my_pt.hyper.to_list('isin')
    triplet = construct_gateway_triplet('marketdatacep', region, 'macp')
    cols = kdb_col_select_helper([
        'macpBid:bid', 'macpMid:mid', 'macpAsk:ask', 'macpRefreshTime:datetime', 'macpBenchmarkIsin:benchmark',
        'macpLiqScore:liquidityScore'
    ], 'last')
    dates = latest_biz_date(dates, True)
    q = build_pt_query(triplet, cols, dates=dates, filters={'sym': isins}, by='isin:sym, priceType')
    res = await query_kdb(q, fconn(GATEWAY))
    if res is None: return
    statics = res.filter(pl.col('macpBenchmarkIsin').is_not_null()).select('isin', 'macpBenchmarkIsin', 'macpLiqScore')
    res = res.with_columns([
        pl.col('priceType').replace({"PRICE": "Px", "SPREAD": "Spd"}).alias('priceType')
    ]).pivot(
        index=['isin', 'macpRefreshTime'],
        on='priceType',
        on_columns=['Px', 'Spd'],
        values=['macpBid', 'macpMid', 'macpAsk'], separator=""
    ).join(statics, on='isin', how='left')

    cachable = res.filter([
        pl.col('macpLiqScore').is_not_null(),
        pl.col('macpLiqScore')!=0
    ]).select(['isin', 'macpLiqScore']).unique(subset=['isin'])
    await write_to_cache_async('macp_liqScore', cachable, cachable.select(['isin']), force=False)

    return res


@hypercache.cached(name='macp_liqScore', ttl=timedelta(days=1), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt'])
async def macp_liqScore(my_pt, region="US", dates=None, **kwargs):
    if not is_today(dates, True): return
    isins = my_pt.hyper.to_list('isin')
    triplet = construct_gateway_triplet('marketdatacep', region, 'macp')
    cols = kdb_col_select_helper(['macpLiqScore:liquidityScore'], 'last')
    dates = latest_biz_date(dates, True)
    q = build_pt_query(triplet, cols, dates=None, filters={'sym': isins, 'not liquidityScore': 0}, by='isin:sym')
    return await query_kdb(q, fconn(GATEWAY))

@hypercache.cached(name='macp_liqScore', ttl=timedelta(days=1), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt'])
async def macp_liq_score_eu(my_pt, region="US", dates=None, **kwargs):
    isins = my_pt.hyper.to_list('isin')
    triplet = construct_panoproxy_triplet('EU', 'quoteevent', dates)
    cols = kdb_col_select_helper(['macpLiqScore:liquidityScore'], 'last')
    q = build_pt_query(triplet, cols, dates=None,
                       filters={'sym': isins, 'not liquidityScore': 0},
                       by='isin:sym', date_kwargs={"return_today":False}
    )
    return await query_kdb(q, fconn(PANOPROXY_EU))

async def quote_macp_pano(my_pt, region="US", dates=None, **kwargs):
    isins = my_pt.hyper.to_list('isin')
    triplet = construct_panoproxy_triplet(region, 'quoteevent.macp', dates)
    cols = kdb_col_select_helper(
        [
            'macpBid:bid', 'macpMid:mid', 'macpAsk:ask', 'macpRefreshTime:datetime', 'macpBenchmarkIsin:benchmark',
            'macpLiqScore:liquidityScore'
        ], 'last'
    )
    dates = latest_biz_date(dates, True)
    q = build_pt_query(triplet, cols, dates=dates, filters={'sym': isins}, by='isin:sym, priceType')
    res = await query_kdb(q, fconn(PANOPROXY))
    if res is None: return
    statics = res.filter(pl.col('macpBenchmarkIsin').is_not_null()).select('isin', 'macpBenchmarkIsin', 'macpLiqScore')
    res = res.with_columns(
        [
            pl.col('priceType').replace({"PRICE": "Px", "SPREAD": "Spd"}).alias('priceType')
        ]
    ).pivot(
        index=['isin', 'macpRefreshTime'],
        on='priceType',
        on_columns=['Px', 'Spd'],
        values=['macpBid', 'macpMid', 'macpAsk'], separator=""
    ).join(statics, on='isin', how='left')

    if is_today(dates, True):
        cachable = res.filter(
            [
                pl.col('macpLiqScore').is_not_null(),
                pl.col('macpLiqScore')!=0
            ]
        ).select(['isin', 'macpLiqScore']).unique(subset=['isin'])
        await write_to_cache_async('macp_liqScore', cachable, cachable.select(['isin']), force=False)

    return res

async def macp_ingress(my_pt, region="US", dates=None, **kwargs):
    return my_pt.select([
        pl.col('isin'),
        pl.col('macpBidSpd').alias('macpBidSpdIngress'),
        pl.col('macpBidPx').alias('macpBidPxIngress'),
        pl.col('macpMidSpd').alias('macpMidSpdIngress'),
        pl.col('macpMidPx').alias('macpMidPxIngress'),
        pl.col('macpAskSpd').alias('macpAskSpdIngress'),
        pl.col('macpAskPx').alias('macpAskPxIngress'),
    ])

async def quote_markitRt(my_pt, region="US", dates=None, **kwargs):
    dates = parse_date(dates, biz=True)
    if not is_today(dates, True): return
    isins = my_pt.hyper.to_list('isin')
    triplet = construct_gateway_triplet('marketdatacep', region, 'markit')
    cols = kdb_col_select_helper(['markitRtBid:bid', 'markitRtMid:mid', 'markitRtAsk:ask', 'markitRtRefreshTime:datetime', 'markitRtBenchmarkIsin:benchmark'], 'last')
    q = build_pt_query(triplet, cols, dates=dates, filters={'sym': isins}, by='isin:sym, priceType')
    res = await query_kdb(q, fconn(GATEWAY))
    if res is None: return
    benches = res.filter(pl.col('markitRtBenchmarkIsin').is_not_null()).select('isin', 'markitRtBenchmarkIsin')
    return res.with_columns([
        pl.col('priceType').replace({"PRICE": "Px", "SPREAD": "Spd"}).alias('priceType')
    ]).pivot(
        index=['isin', 'markitRtRefreshTime'],
        on='priceType',
        on_columns=['Px', 'Spd'],
        values=['markitRtBid', 'markitRtMid', 'markitRtAsk'],
        separator=""
    ).join(benches, on='isin', how='left')

async def quote_markit(my_pt, region="US", dates=None, **kwargs):
    isins = my_pt.hyper.ul('isin')
    cols = [
        'markitBidPx:bidPrice', 'markitAskPx:offerPrice', 'markitMidPx:(bidPrice+offerPrice)%2',
        'markitBidSpd:bidSpread', 'markitAskSpd:offerSpread', 'markitMidSpd:(bidSpread+offerSpread)%2',
        'markitBenchmarkIsin:benchmark', 'markitRefreshTime:time', 'date'
    ]
    triplet = construct_panoproxy_triplet(region, 'evb', dates)
    q = build_pt_query(
        triplet,
        cols=kdb_col_select_helper(cols, method='last'),
        dates=dates,
        by="isin:sym",
        filters={'sym': isins, 'market': 'MARKIT_EVB'}
    )
    res = await query_kdb(q, PANOPROXY_US)
    if (res is None) or (res.hyper.is_empty()): return
    return res.hyper.utc_datetime(time_col="markitRefreshTime", date_col="date", output_name="markitRefreshTime").drop(['date'], strict=False)


async def quote_idc(my_pt, region='US', dates=None, **kwargs):
    isins = my_pt.hyper.ul('isin')
    cols = [
        'idcMidSpd:(bidSpread+offerSpread)%2', 'idcAskSpd:offerSpread', 'idcBidSpd:bidSpread',
        'idcBidPx:bidPrice', 'idcMidPx:(bidPrice+offerPrice)%2', 'idcAskPx:offerPrice',
        'idcBenchmarkIsin:benchmark', 'idcRefreshTime:time','date'
    ]
    triplet = construct_panoproxy_triplet(region, 'idc', dates)
    q = build_pt_query(
        triplet,
        cols=kdb_col_select_helper(cols, method='last'),
        dates=dates,
        by="isin:sym",
        date_kwargs={'return_today': False},
        filters={'sym': isins}
    )
    res = await query_kdb(q, PANOPROXY_US)
    if (res is None) or (res.hyper.is_empty()): return
    return res.hyper.utc_datetime(time_col="idcRefreshTime", date_col="date", output_name="idcRefreshTime").drop(['date'], strict=False)

async def quote_mlcr(my_pt, region="US", dates=None, **kwargs):
    dates = parse_date(dates, biz=True)
    if not is_today(dates, True): return
    isins = my_pt.hyper.to_kdb_sym('isin')
    cols = [
        'mlcrBidPx:bidPrice*100', 'mlcrAskPx:offerPrice*100', 'mlcrMidPx:(bidPrice+offerPrice)%2*100',
        'mlcrBidSpd:bidSpread', 'mlcrAskSpd:offerSpread', 'mlcrMidSpd:(bidSpread+offerSpread)%2',
        'mlcrLiqScore:liquidityScore', 'mlcrBenchmarkIsin:benchmark'
    ]
    cols = kdb_col_select_helper(cols)
    q = 'select %s by isin:sym from .credit.common.pricing.marketData.getCreditSmmData[`us] where sym in (%s)' % (cols, isins)
    now = date_to_datetime(dates, time=None, utc=True, biz=True)
    return (await query_kdb(q, PANOPROXY)).with_columns(pl.lit(now, pl.Datetime).alias('mlcrRefreshTime'))

# BACKUP
async def quote_mlcr_px(my_pt, region="US", dates=None, **kwargs):
    isins = my_pt.hyper.to_list('isin')
    triplet = construct_gateway_triplet('marketdatacep', region, 'creditsmm')
    cols = kdb_col_select_helper(['mlcrBid:bid*100', 'mlcrMid:mid*100', 'mlcrAsk:ask*100', 'mlcrRefreshTime:datetime'], 'last')
    dates = latest_biz_date(dates, True)
    q = build_pt_query(triplet, cols, dates=dates, filters={'sym': isins}, by='isin:sym, priceType')
    res = await query_kdb(q, fconn(GATEWAY))
    if (res is None) or (res.hyper.is_empty()): return
    return res.with_columns(pl.col('priceType').replace({"PRICE": "Px", "SPREAD": "Spd"}).alias('priceType')).pivot(
        index=['isin', 'mlcrRefreshTime'], on='priceType', on_columns=['Px', 'Spd'],
        values=['mlcrBid', 'mlcrMid', 'mlcrAsk'], separator=""
        )


@hypercache.cached(ttl=timedelta(hours=6), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt'])
async def lqa_liqScore(my_pt, region="US", dates=None, **kwargs):
    triplet = construct_panoproxy_triplet(region, 'bval', dates)
    cols = ['lqaLiqScore:lqaLiquidityScore']
    isins = my_pt.hyper.to_list('isin', unique=True, drop_nulls=True)
    dates = latest_biz_date(dates, True)
    q = build_pt_query(triplet, cols, dates, {"return_today": False}, {"isin": isins, 'not lqaLiquidityScore': 0}, by=['isin'])
    return await query_kdb(q, config=fconn(PANOPROXY))


@hypercache.cached(ttl=timedelta(hours=6), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt'])
async def bval_duration(my_pt, region="US", dates=None, **kwargs):
    my_pt = ensure_lazy(my_pt)
    triplet = construct_panoproxy_triplet(region, 'bval', dates)
    cols = ['duration:bvalDurBid']
    isins = my_pt.hyper.to_list('isin', unique=True, drop_nulls=True)
    dates = latest_biz_date(dates, True)
    q = build_pt_query(triplet, cols, dates, {"return_today": False}, {"isin": isins, 'not bvalDurBid': 0}, by=['isin'])
    return await query_kdb(q, config=fconn(PANOPROXY))


async def quote_bval_non_dm(my_pt, region="US", dates=None, *, live_only=False, last_snapshot_only=False, snapshot=None, depth=0, **kwargs):
    if depth > 3: return
    depth += 1
    my_pt = ensure_lazy(my_pt)
    triplet = construct_panoproxy_triplet(region, 'bval', dates)
    cols = [
        'bvalBidSpd:bidSpread', 'bvalMidSpd:bvalSprdBenchmarkMid', 'bvalAskSpd:offerSpread',
        'bvalBidYld:bidYield', 'bvalMidYld:bvalMidYield', 'bvalAskYld:offerYield',
        'bvalBidPx:bidPrice', 'bvalMidPx:bvalMidPrice', 'bvalAskPx:offerPrice',
        'lqaLiqScore:lqaLiquidityScore', 'duration:bvalDurBid', 'bvalBidGspd:bvalGspreadBid',
        'bvalBidOas:bvalOasBidSprd', 'bvalMidOas:bvalOasMidSprd', 'bvalAskOas:(2*bvalOasMidSprd-bvalOasBidSprd)',
        'bvalBidZspd:bvalZsprdBid', 'bvalBenchmarkIsin:benchmark', 'cusip', 'bvalRefreshTime:time'
    ]
    isins = my_pt.hyper.to_list('isin', unique=True, drop_nulls=True)
    dates = latest_biz_date(dates, True)
    q = build_pt_query(triplet, cols, dates, {"return_today": False}, {"isin": isins}, by=['bvalSnapshot:snapshot', 'isin'])
    r = await query_kdb(q, config=fconn(PANOPROXY))
    if r is None: return
    dt = parse_single_date(dates, utc=True)
    result = r.with_columns([
        pl.col('bvalSnapshot').str.strip_chars().str.replace_all(" ", "").str.replace(":", "").str.to_uppercase().str.replace("PM", "").alias("bvalSnapshot"),
        pl.when(pl.col('cusip').str.strip_chars()=="")
        .then(pl.lit(None, pl.String))
        .otherwise(pl.col('cusip')).alias('cusip'),
        pl.when(pl.col('duration').is_null() | (pl.col('duration').cast(pl.Float64, strict=False)==0))
        .then(pl.lit(None, pl.Float64)).otherwise(pl.col('duration')).alias('duration'),
        pl.lit(dt, pl.Date).dt.combine(pl.col('bvalRefreshTime')).alias('bvalRefreshTime')
    ])

    # Splice out the most recent
    newest_snapshot = result.sort('bvalRefreshTime', descending=True).hyper.peek("bvalSnapshot")
    if newest_snapshot is None:
        return await quote_bval_non_dm(my_pt, region, next_biz_date(dates, -1), live_only=live_only, last_snapshot_only=last_snapshot_only, snapshot=snapshot, depth=depth)
    newest_data = result.filter(pl.col('bvalSnapshot').is_not_null() & (pl.col('bvalSnapshot')==newest_snapshot))

    cachable = newest_data.filter([
        pl.col('lqaLiqScore').is_not_null(),
        pl.col('lqaLiqScore')!=0
    ]).select(['isin', 'lqaLiqScore']).unique(subset=['isin'])
    await write_to_cache_async('lqa_liqScore', cachable, cachable.select(['isin']), force=False)

    cachable_dur = newest_data.filter([
        pl.col('duration').is_not_null(),
        pl.col('duration')!=0
    ]).select(['isin', 'duration']).unique(subset=['isin'])
    await write_to_cache_async('bval_duration', cachable_dur, cachable_dur.select(['isin']), force=False)

    if live_only:
        return newest_data
    if last_snapshot_only:
        result = newest_data
        monikers = {newest_snapshot.upper()}
    elif snapshot:
        result = result.filter(pl.col('bvalSnapshot')==snapshot)
        monikers = {snapshot.upper()}
    else:
        monikers = BVALSnapshotTimes().monikers - {"LO4", 'SY5'}

    result_schema = result.hyper.schema()
    static_cols = {'lqaLiqScore', 'cusip', 'duration', 'bvalBenchmarkIsin'}
    non_static = {col for col in result_schema if col not in static_cols}

    # Split into static/non-static
    presult = result.select(non_static).pivot(on=['bvalSnapshot'], on_columns=list(monikers), index=['isin'], separator="-")
    pfields = presult.hyper.fields

    rename_map = {}
    for col in pfields:
        if col=="isin": continue
        base, sep, suffix = col.rpartition("-")
        if sep:
            new = base.replace("bval", "bval" + suffix.title())
        else:
            new = col
        rename_map[col] = new
    presult = presult.rename(rename_map, strict=False)
    if snapshot:
        return presult.join(newest_data.select(['isin'] + list(static_cols)), on='isin', how='left')

    return presult.join(newest_data, on='isin', how='left')


async def quote_bval_full(my_pt, region="US", dates=None, *, cod=True, live_only=False, last_snapshot_only=False, snapshot=None, **kwargs):
    triplet = construct_gateway_triplet('creditext', region, 'bval')
    isins = my_pt.hyper.to_list('isin', unique=True, drop_nulls=True)
    my_date = latest_biz_date(dates)
    cols = [
        # Timestamps
        "bvalRefreshTime:time",

        # Ids
        "cusip:idCusipReal", "ticker", "description:securityDes", "bvalAssetClass",

        # Risk
        "unitAccrued:accInterestBid100",
        "duration:bvalDurBid", "lqaLiqScore:lqaLiquidityScore",
        "bvalQuoteConvention:bvalPricingDomain",

        # Benchmarks
        "bvalBenchmarkBidPx:bvalBenchmarkBid", "bvalBenchmarkBidYld:bvalBenchmarkBidYld", "bvalBenchmarkId:bvalSprdBenchmarkId",

        # Quotes
        "bvalBidSpd:bvalSprdBenchmarkBid", "bvalMidSpd:bvalSprdBenchmarkMid", "bvalAskSpd:bvalSprdBenchmarkAsk",
        "bvalBidPx:bvalBidPrice", "bvalMidPx:bvalMidPrice", "bvalAskPx:bvalAskPrice",
        "bvalBidYld:bvalBidYield", "bvalMidYld:bvalMidYield", "bvalAskYld:bvalAskYield",
        "bvalBidDm:bvalDiscMarginBid", "bvalMidDm:bvalDiscMarginMid", "bvalAskDm:bvalDiscMarginAsk",
        "bvalBidOas:bvalOasBidSprd", "bvalMidOas:bvalOasMidSprd", 'bvalAskOas:(2*bvalOasMidSprd-bvalOasBidSprd)'
    ]

    q = build_pt_query(triplet, cols, my_date, filters={"sym": isins}, by=['bvalSnapshot', 'isin:sym'], lastby=['sym'])
    r = await query_kdb(q, config=fconn(GATEWAY))
    if r is None: return
    result = r.with_columns([
        pl.col('bvalBenchmarkId').str.slice(2).alias('bvalBenchmarkCusip'),
        pl.col('bvalQuoteConvention').replace({
            'Price': "PX",
            "Spread": "SPD",
            "Discount Margin": "DM",
            "DiscountMargin": "DM",
            "Money Market Yield": "MMY",
            "MoneyMarketYield": "MMY",
            "Yield": "YLD",
            "Yield To Maturity": "YTM",
            "Yield To Worst": "YTW",
            "Yield To Call": "YTC",
        }).alias('bvalQuoteConvention'),
        pl.col('bvalAssetClass').replace_strict(BVAL_SUB_ASSET_MAP, return_dtype=pl.String).alias("bvalSubAssetClass"),
        pl.col('bvalAssetClass').replace_strict(BVAL_ASSET_MAP, return_dtype=pl.String).alias("bvalAssetClass"),
        pl.col('bvalSnapshot').str.strip_chars().str.replace_all(" ", "").str.replace(":", "").str.to_uppercase().str.replace("PM", "").alias("bvalSnapshot"),
        pl.when(pl.col('cusip').str.strip_chars()=="")
        .then(pl.lit(None, pl.String))
        .otherwise(pl.col('cusip')).alias('cusip'),
        pl.when(pl.col('duration').is_null() | (pl.col('duration').cast(pl.Float64, strict=False)==0))
        .then(pl.lit(None, pl.Float64)).otherwise(pl.col('duration')).alias('duration'),
        pl.lit(my_date, pl.Date).dt.combine(pl.col('bvalRefreshTime')).alias('bvalRefreshTime')
    ]).drop(['bvalBenchmarkId'], strict=False)

    # Splice out the most recent
    newest_snapshot = result.sort('bvalRefreshTime', descending=True).hyper.peek("bvalSnapshot")
    if newest_snapshot is None:
        return await quote_bval_full(my_pt, region, next_biz_date(my_date, -1), cod=cod, live_only=live_only, last_snapshot_only=last_snapshot_only, snapshot=snapshot, __no_cache=True, **kwargs)
    newest_data = result.filter(pl.col('bvalSnapshot').is_not_null() & (pl.col('bvalSnapshot')==newest_snapshot))

    cachable = newest_data.filter([
        pl.col('lqaLiqScore').is_not_null(),
        pl.col('lqaLiqScore')!=0
    ]).select(['isin', 'lqaLiqScore']).unique(subset=['isin'])
    await write_to_cache_async('lqa_liqScore', cachable, cachable.select(['isin']), force=False)

    cachable_dur = newest_data.filter([
        pl.col('duration').is_not_null(),
        pl.col('duration')!=0
    ]).select(['isin', 'duration']).unique(subset=['isin'])
    await write_to_cache_async('bval_duration', cachable_dur, cachable_dur.select(['isin']), force=False)

    if live_only:
        return newest_data
    if last_snapshot_only:
        result = newest_data
        monikers = {newest_snapshot.upper()}
    elif snapshot:
        result = result.filter(pl.col('bvalSnapshot')==snapshot)
        monikers = {snapshot.upper()}
    else:
        monikers = BVALSnapshotTimes().monikers - {"LO4", 'SY5'}

    result_schema = result.hyper.schema()
    dynamic_cols = set(result.hyper.cols_like("(Px|Spd|Yld|Mmy|Oas|sprd|Dm|time|isin|snapshot)$", case_sensitive=False))

    # Split into static/non-static
    presult = result.select(dynamic_cols).pivot(on=['bvalSnapshot'], on_columns=list(monikers), index=['isin'], separator="-")
    pfields = presult.hyper.fields

    rename_map = {}
    for col in pfields:
        if col=="isin": continue
        base, sep, suffix = col.rpartition("-")
        if sep:
            new = base.replace("bval", "bval" + suffix.title())
        else:
            new = col
        rename_map[col] = new
    presult = presult.rename(rename_map, strict=False)

    return presult.join(newest_data, on='isin', how='left')

HOUSE_MAP = {
    'REUTERS_PRICE': 'PX',
    'PRICE': 'PX',
    'SPREAD_TO_BENCHMARK': 'SPD',
    'ASSET_SWAP_SPREAD': 'SPD',
    'FULL_PRICE': 'PX',
    'MM_YIELD': 'MMY',
    'YIELD': 'YLD',
    'DISCOUNT_MARGIN': 'DM'
}

async def quote_house_tm(my_pt, region="US", dates=None, **kwargs):
    my_date = latest_biz_date(dates, utc=True)
    if not is_today(my_date): return
    triplet = construct_panoproxy_triplet(region, 'traderMark', my_date)
    isins = my_pt.hyper.to_list('isin')
    rt = f'house{region.title()}'
    cols = [f'{rt}RefreshTime:lastActiveChange', f'{rt}BenchmarkIsin:tradermarkBenchmarkIsin',
            f'{rt}BidPx:tradermarkBidPrice', f'{rt}MidPx:tradermarkMidPrice', f'{rt}AskPx:tradermarkAskPrice',
            f'{rt}BidSpd:tradermarkBidSpread', f'{rt}MidSpd:tradermarkMidSpread', f'{rt}AskSpd:tradermarkAskSpread',
            f'{rt}BidYld:tradermarkBidYield', f'{rt}MidYld:tradermarkMidYield', f'{rt}AskYld:tradermarkAskYield',
            f'{rt}QuoteConvention:tradermarkQuoteType'
            ]
    cols = kdb_col_select_helper(cols, 'last')
    q = build_pt_query(triplet, cols, dates=my_date, date_kwargs={"return_today": False}, filters={'sym': isins}, by=['isin:sym'])
    res = await query_kdb(q, fconn(PANOPROXY, region=region))
    return res.with_columns(pl.col(f'{rt}QuoteConvention').replace(HOUSE_MAP).alias(f'{rt}QuoteConvention'))


async def quote_house_bq(my_pt, region="US", dates=None, **kwargs):
    my_date = latest_biz_date(dates, utc=True)
    triplet = construct_panoproxy_triplet(region, 'bondquote', my_date)
    isins = my_pt.hyper.to_list('isin')
    rt = f'house{region.title()}'
    cols = [f'{rt}RefreshTime:lastActiveChange',
            f'{rt}BidPx:bid', f'{rt}MidPx:mid', f'{rt}AskPx:ask', f'{rt}BenchmarkEsm:benchmarkEsmId',
            f'{rt}BidSpd:spreadToBenchmarkBid', f'{rt}MidSpd:spreadToBenchmarkMid', f'{rt}AskSpd:spreadToBenchmarkAsk',
            f'{rt}BidYld:(midYield+(spreadToBenchmarkBid-spreadToBenchmarkAsk)%100)',
            f'{rt}MidYld:midYield',
            f'{rt}AskYld:(midYield-(spreadToBenchmarkBid-spreadToBenchmarkAsk)%100)',
            'unitDv01:dv01Mid',
            f'{rt}QuoteConvention:pricingMethod'
            ]
    cols = kdb_col_select_helper(cols, 'last')
    q = build_pt_query(triplet, cols, dates=my_date, date_kwargs={"return_today": False}, filters={'sym': isins}, by=['isin:sym'])
    res = await query_kdb(q, fconn(PANOPROXY, region=region))
    return res.with_columns(pl.col(f'{rt}QuoteConvention').replace(HOUSE_MAP).alias(f'{rt}QuoteConvention'))


def _cap(x):
    return x[0].upper() + x[1:]

    # .credit.us.traderMark.realtime


async def quote_stats(my_pt, region="US", dates=None, **kwargs):
    isins = my_pt.hyper.ul('isin')
    rt = f'stats{region.title()}'
    rtc = _cap(rt)
    my_date = latest_biz_date(dates, utc=True)
    bench = 'benchmark^benchmarkIsin' if region=='US' and is_today(my_date, utc=True) else 'benchmark'
    cols = [
        f'{rt}:price', f'{rt}BenchmarkIsin:{bench}', f'{rt}Size:size', 'time', 'date'
    ]
    triplet = construct_panoproxy_triplet(region, 'axes', my_date)
    q = build_pt_query(
        triplet,
        cols=kdb_col_select_helper(cols, method='last'),
        dates=my_date,
        by="isin:sym, side, priceType",
        date_kwargs={'return_today': False},
        filters={'sym': isins, 'market': 'CHP_STATS'}
    )
    r = await query_kdb(q, config=fconn(PANOPROXY, region=region))
    if (r is None) or (r.hyper.is_empty()): return
    refresh_time = f"{rt}RefreshTime"
    refresh = r.hyper.utc_datetime(time_col="time", date_col='date', output_name=refresh_time)
    return r.with_columns([
        pl.col('priceType').replace_strict({
            'PRICE': 'Px',
            'SPREAD': 'Spd'
        }).alias('quoteType'),
        pl.col('side').replace_strict({
            'BID': 'Bid',
            'OFFER': 'Ask'
        }).alias('side')
    ]).with_columns([
        pl.concat_str([
            pl.col('side').str.to_titlecase(), pl.col('quoteType').str.to_titlecase()
        ], separator="").alias('k')
    ]).pivot(
        index=['isin'],
        on=['k'],
        on_columns=['BidPx', 'AskPx', 'BidSpd', 'AskSpd'],
        values=[f'{rt}Size', f'{rt}'],
        separator="",
        aggregate_function="last",
    ).join(r.select('isin', f'{rt}BenchmarkIsin'), on='isin', how='left').with_columns([
        pl.coalesce([pl.col(f'{rt}SizeBidPx'), pl.col(f'{rt}SizeBidSpd')]).fill_null(0).alias(f'{rt}BidSize'),
        pl.coalesce([pl.col(f'{rt}SizeAskPx'), pl.col(f'{rt}SizeAskSpd')]).fill_null(0).alias(f'{rt}AskSize'),
        pl.mean_horizontal([
            pl.col(f'{rt}BidPx').cast(pl.Float64, strict=False),
            pl.col(f'{rt}AskPx').cast(pl.Float64, strict=False)
        ]).alias(f'{rt}MidPx'),
        pl.mean_horizontal([
            pl.col(f'{rt}BidSpd').cast(pl.Float64, strict=False),
            pl.col(f'{rt}AskSpd').cast(pl.Float64, strict=False)
        ]).alias(f'{rt}MidSpd')
    ]).with_columns([
        pl.when(pl.col(f'{rt}BidSize').is_not_null() & (pl.col(f'{rt}BidSize')!=0))
        .then(pl.lit(1, pl.Int8))
        .otherwise(pl.lit(0, pl.Int8))
        .alias(f'is{rtc}BidAxe'),
        pl.when(pl.col(f'{rt}AskSize').is_not_null() & (pl.col(f'{rt}AskSize')!=0))
        .then(pl.lit(1, pl.Int8))
        .otherwise(pl.lit(0, pl.Int8))
        .alias(f'is{rtc}AskAxe'),
    ]).with_columns([
        pl.when((pl.col(f'is{rtc}BidAxe')==1) & (pl.col(f'is{rtc}AskAxe')==1))
        .then(pl.lit(1, pl.Int8))
        .otherwise(pl.lit(0, pl.Int8))
        .alias(f'is{rtc}MktAxe'),
    ]).join(refresh.select('isin',refresh_time), on='isin', how='left').select([
        'isin', f'{rt}BidPx', f'{rt}AskPx', f'{rt}MidPx', f'{rt}BidSpd', f'{rt}AskSpd', f'{rt}MidSpd',
        f'{rt}BenchmarkIsin', f'{rt}BidSize', f'{rt}AskSize',
        f'is{rtc}BidAxe', f'is{rtc}AskAxe', f'is{rtc}MktAxe', refresh_time
    ])


# dont use this
async def quote_stats_sgp_bad(my_pt, region="US", dates=None, **kwargs):
    isins = my_pt.hyper.to_list('isin')
    triplet = construct_gateway_triplet("marketdata", "SGP", "quoteevent", 1)
    cols = ['marketTimestamp', 'price', 'size', 'side', 'eventType', 'market', 'priceType', 'quoteType', 'benchmark']
    cols = kdb_col_select_helper(cols, "last")
    dates = latest_biz_date(dates, True)
    q = build_pt_query(triplet, cols, dates, filters={'market': 'CHP_STATS', 'sym': isins, 'not side': 'BIDOFFER'}, by=['sym'])
    r = await query_kdb(q, fconn(GATEWAY))
    if (r is None) or r.hyper.is_empty(): return None
    return await pivot_quoteevent_raw(r)


async def axe_coalesce(my_pt, region="US", dates=None, **kwargs):
    s = my_pt.hyper.schema()

    isBids = [
        pl.col(col).cast(pl.Float64, strict=False).fill_null(0).abs()
        for col in ['statsUsBidSize', 'statsEuBidSize', 'statsSgpBidSize'] if col in s
    ]
    isAsks = [
        pl.col(col).cast(pl.Float64, strict=False).fill_null(0).abs()
        for col in ['statsUsAskSize', 'statsEuAskSize', 'statsSgpAskSize'] if col in s
    ]
    _flat = [
        pl.lit(0, pl.Float64).alias('_flat')
    ]

    my_pt = ensure_lazy(my_pt)
    my_pt = my_pt.hyper.ensure_columns(['desigRegion'], default='US', dtypes={'desigRegion':pl.String})
    
    def _sort_by_region(template):
        return (pl.when(pl.col('desigRegion') == 'US').then(
            pl.coalesce([template % "Us", template % "Eu", template % "Sgp"])
        ).when(pl.col('desigRegion')=='EU').then(
            pl.coalesce([template % "Eu", template % "Us", template % "Sgp"])
        ).when(pl.col('desigRegion')=='SGP').then(
            pl.coalesce([template % "Sgp", template % "Us", template % "Eu"])
        ).otherwise(pl.coalesce([template % "Us", template % "Eu", template % "Sgp"]))
         .alias(template.replace("%s", "")))

    bench = my_pt.hyper.ensure_columns(['statsUsBenchmarkIsin', 'statsEuBenchmarkIsin', 'statsSgpBenchmarkIsin']).select([
        pl.col('isin'), _sort_by_region("stats%sBenchmarkIsin")
    ])

    quotes = my_pt.hyper.ensure_columns([
        'statsUsBidPx','statsUsAskPx','statsUsMidPx','statsUsBidSpd','statsUsAskSpd','statsUsMidSpd',
        'statsEuBidPx', 'statsEuAskPx', 'statsEuMidPx', 'statsEuBidSpd', 'statsEuAskSpd', 'statsEuMidSpd',
        'statsSgpBidPx', 'statsSgpAskPx', 'statsSgpMidPx', 'statsSgpBidSpd', 'statsSgpAskSpd', 'statsSgpMidSpd',
    ]).select([
        pl.col('isin'), 
        _sort_by_region("stats%sBidPx"), _sort_by_region("stats%sMidPx"), _sort_by_region("stats%sAskPx"),
        _sort_by_region("stats%sBidSpd"), _sort_by_region("stats%sMidSpd"), _sort_by_region("stats%sAskSpd"),
    ])

    axes = (await my_pt.with_columns([
        pl.max_horizontal(isBids + _flat).alias('axeFullBidSize'),
        pl.max_horizontal(isAsks + _flat).alias('axeFullAskSize')
    ]).with_columns([
        pl.when(pl.col('axeFullBidSize') > 0).then(FLAG_YES).otherwise(FLAG_NO).alias('isBidAxe'),
        pl.when(pl.col('axeFullAskSize') > 0).then(FLAG_YES).otherwise(FLAG_NO).alias('isAskAxe'),
    ]).with_columns([
        pl.when((pl.col('isBidAxe')==1) & (pl.col('isAskAxe')==1)).then(FLAG_YES).otherwise(FLAG_NO).alias('isMktAxe'),
    ]).with_columns([
        pl.when(pl.col('isMktAxe')==1).then(FLAG_NO)
        .when((pl.col('side')=='BUY') & (pl.col('isAskAxe')==1)).then(FLAG_YES)
        .when((pl.col('side')=='SELL') & (pl.col('isBidAxe')==1)).then(FLAG_YES)
        .otherwise(FLAG_NO).alias('isAntiAxe')
    ]).hyper.compress_plan_async()).select([
        pl.col('isin'),
        pl.col('isBidAxe'), pl.col('isAskAxe'), pl.col('isMktAxe'), pl.col('isAntiAxe'),
        pl.col('axeFullBidSize'), pl.col('axeFullAskSize')
    ])

    return (my_pt.select(['isin']).unique('isin')
            .join(bench, on='isin', how='left')
            .join(quotes, on='isin', how='left')
            .join(axes, on='isin', how='left'))


async def cached_quoteevent(my_pt, region=None, dates=None, **kwargs):
    seed = my_pt.select('isin').unique()
    try:
        from app.services.kdb.tickerplant import FileClient
        client = FileClient()
        isins = my_pt.hyper.ul('isin')
        date_guard = parse_date(dates)
        next_date = next_biz_date(date_guard, n=1)
        res = client.query(isins, 'quoteevent').filter([
            pl.col('marketTimestamp') >= date_guard,
            pl.col('marketTimestamp') < next_date,
        ])
        pivoted = await pivot_quoteevent_raw(res)
        if pivoted is not None and not pivoted.hyper.is_empty():
            return seed.join(pivoted, on='isin', how='left')
        return seed

    except Exception as e:
        await log.error(f"QuoteEvent Cache: {e}")
        return seed


QE_MARKET = re.compile(r'^[^_]+_([^_]+(?:_[^_]+)?)_(?:Bid|Ask)_')
async def pivot_quoteevent_raw(res, remove_outliers=True, outlier_std_threshold=3):
    res = ensure_lazy(res)
    cached_r = res.select([
        pl.col('sym').alias('isin'),
        pl.col("side").replace("OFFER", "ASK").str.to_titlecase().alias('side'),
        pl.col('region'),
        pl.col("market").replace(
            QUOTE_EVENT_MARKET_MAP
        ).str.to_lowercase().alias('market'),
        pl.col('marketTimestamp'),
        pl.col('price'),
        pl.col('priceType').alias('quoteType').replace_strict({
            'PRICE': 'Px',
            'SPREAD': 'Spd',
            'YIELD': 'Yld',
        }, default=None),
        pl.col('quoteType').eq('FIRM').cast(pl.Int8).alias('isFirm'),
        pl.col('eventType'),
        pl.col('size'),
        pl.col('benchmark'),
    ]).with_columns([

        pl.when((pl.col('market')=='tmc') & pl.col('quoteType').is_null()).then(
            pl.lit('Px', pl.String)
        ).otherwise(pl.col('quoteType')).alias('quoteType'),

        pl.when((pl.col('market')=='axi') & (pl.col('quoteType')=='Px') & pl.col('price').is_not_null()).then(
            (pl.col('price') * 100)
        ).otherwise(pl.col('price')).alias('price'),

    ])
    if remove_outliers:
        cached_r = cached_r.hyper.zscore_by_group(['price'], group_by=['isin', 'quoteType']).filter([
            pl.col('price_z').abs() < outlier_std_threshold
        ]).drop(['price_z'], strict=False)
    cached_r = await cached_r.hyper.compress_plan_async()

    markets = cached_r.hyper.ul('market')

    benchmarks = cached_r.filter([
        pl.col("benchmark").is_not_null() & pl.col('benchmark').ne("")
    ]).select([
        pl.col("isin"),
        pl.col("market"),
        pl.col('benchmark').alias("benchmarkBond")
    ]).pivot(
        index="isin",
        on="market",
        on_columns=markets,
        values="benchmarkBond",
        aggregate_function="first"
    ).rename({x: clean_camel(x, 'BenchmarkIsin') for x in markets})

    size_mkts = [x for x in markets if (x.startswith("allq")) or (x=='stats') or (x=='axi') or (x=='am')]
    size_markets = ["_".join(x) for x in product(size_mkts, ['Bid_Size', 'Ask_Size'])]
    sizes = cached_r.filter([
        pl.col('market').is_in(size_mkts)
    ]).select([
        pl.col('isin'),
        pl.col('market'),
        pl.when((pl.col('market')=='tmc') & (pl.col('size')!=0)).then(pl.col('size') * 1_000).otherwise(pl.col('size')).alias('size'),
        pl.col('side')
    ]).group_by(['isin', 'market', 'side']).agg([
        pl.col('size').last(ignore_nulls=False).fill_null(pl.lit(0, pl.Float64)).alias('size'),
        pl.concat_str([pl.col('market').last(), pl.col('side').last(), pl.lit("Size")], separator="_").alias('_t')
    ]).pivot(
        index=['isin'],
        values=['size'],
        on=['_t'],
        on_columns=size_markets
    ).fill_null(pl.lit(0, pl.Float64)).rename({k: clean_camel(k) for k in size_markets})

    base_r = cached_r.filter([
        pl.col('price').is_not_null(),
        pl.col("eventType")!="DELETE_QUOTE",
        pl.col('isFirm')==1,
        pl.col('quoteType').is_not_null()
    ]).select([
        pl.col("marketTimestamp").max().over('isin', 'market').alias("_maxTimestamp"),
        pl.col("marketTimestamp"),
        pl.col("isin"),
        pl.col("price"),
        pl.concat_str([pl.col('market'), pl.col('side'), pl.col('quoteType')], separator="_").alias("_mkt")
    ])

    mkts = base_r.hyper.ul('_mkt')
    finale = base_r.sort(['marketTimestamp'], descending=True).group_by(['isin', '_mkt']).agg([
        pl.col('_maxTimestamp').first().alias('marketTimestamp'),
        pl.col('price').first()
    ]).pivot(
        index=['isin'], values=['price', 'marketTimestamp'], on=['_mkt'], on_columns=mkts, aggregate_function='first'
    )

    s = finale.hyper.fields

    def _extract_mkt(x):
        m = QE_MARKET.search(x)
        return m.group(1) if m else None

    def _fmt(x):
        mkt = _extract_mkt(x)
        if not mkt: return x
        if x.startswith("marketTimestamp"):
            return clean_camel(mkt, "RefreshTime")
        return clean_camel(x.replace("price_", ""))

    joiners = {}
    for k in s:
        v = _fmt(k)
        if v not in joiners: joiners[v] = set()
        joiners[v].add(k)
    return await finale.select([
        pl.coalesce(list(v)).alias(k) for k, v in joiners.items()
    ]).join(benchmarks, on='isin', how='left').join(sizes.hyper.select_existing([
        'isin', 'statsBidSize', 'statsAskSize'
    ]), on='isin', how='left').hyper.compress_plan_async()



def after_quote_token(s: str):
    i_bid = s.find("Bid")
    i_mid = s.find("Mid")
    i_ask = s.find("Ask")

    # Find earliest non-negative index
    i = len(s) + 1
    if i_bid != -1 and i_bid < i: i = i_bid + 3
    if i_mid != -1 and i_mid < i: i = i_mid + 3
    if i_ask != -1 and i_ask < i: i = i_ask + 3

    return s[i:] if i <= len(s) else s

async def cached_quoteevent_cleanup(my_pt, region="US", dates=None, frames=None, **kwargs):
    return my_pt.select(['isin'] + my_pt.hyper.cols_like('^(_)|(usIs)|(euIs)|(sgpIs)'))

async def merge_cached_quoteevent(my_pt, region="US", market=None, dates=None, frames=None, **kwargs):
    if market is None: return
    all_mkts = my_pt.hyper.cols_like(f"^[_]?({market})[A-Za-z0-9_]*(Bid|Mid|Ask|Size)")
    all_static = list(set(my_pt.hyper.cols_like(f"^[_]?({market})[A-Za-z0-9_]")) - set(all_mkts))

    qts = {after_quote_token(x) for x in all_mkts}

    bid_cols = [f'{market}Bid{qt}' for qt in qts]
    mid_cols = [f'{market}Mid{qt}' for qt in qts]
    ask_cols = [f'{market}Ask{qt}' for qt in qts]
    full_cols = bid_cols + ask_cols + mid_cols

    all_input = all_static + all_mkts
    missing = [x for x in my_pt.hyper.fields if x.startswith(market) and (x not in all_input)]
    if missing:
        await log.critical(f"Detected missing: {', '.join(missing)}")

    all_output = all_static + full_cols
    return my_pt.select(['isin'] + all_input).hyper.ensure_columns(all_output).with_columns([
        pl.coalesce([
            pl.col(mid),
            (pl.col(mid.replace("Mid", "Bid")) + pl.col(mid.replace("Mid", "Ask")))/2
        ]).alias(mid) for mid in mid_cols
    ])

async def quote_runz_panoproxy(my_pt, region="US", dates=None, frames=None, swap_juniors=True, **kwargs):
    dates = parse_date(dates, biz=True)
    if region != "US": return
    my_pt = ensure_lazy(my_pt)
    isins = my_pt.hyper.ul('isin')
    cusips = my_pt.hyper.ul('cusip')
    fake_isins = set(['XS' + c for c in cusips] + ['US' + c for c in cusips]) - set(isins)
    isins += list(fake_isins)
    cols = [
        'cusip', 'date', 'time', 'benchmarkCusip:fills[benchmark]', 'QuoteConvention:quoteType',
        'BidPx:bid', 'AskPx:ask', 'BidSpd:bidSpread', 'AskSpd:askSpread', 'bidx', 'askx'
    ]
    cols = kdb_col_select_helper(cols, method='last')
    dates = latest_biz_date(dates, True)
    triplet = construct_panoproxy_triplet('us', 'bbgrunz', dates)
    filter_list = []
    if isins: filter_list.append('(isin in (%s))' % kdb_convert_series_to_sym(isins))
    if cusips: filter_list.append('(cusip in (%s))' % kdb_convert_series_to_sym(cusips))
    filters = []
    if not is_today(dates, utc=True):
        filters.append('date = %s' % dates.strftime('%Y.%m.%d'))
    filters.append(f'({" | ".join(filter_list)})')
    filters.append('priceSource in (%s)' % kdb_convert_series_to_sym(['AM', 'ALGO', 'BBG', 'VELOCITY_US_', 'MARK']))
    filters = ",".join(filters)

    q = build_pt_query(triplet, cols=cols, by=['isin', 'runzSubject:subject', 'runzSenderName:senderName', 'priceSource'], filters=filters, raw_filter=True)
    r = await query_kdb(q, config=fconn(PANOPROXY_US))
    if (r is None) or (r.hyper.is_empty()): return

    rr = r.with_columns([
        pl.when(pl.col('bidSpd').is_not_null() & pl.col('askSpd').is_not_null())
        .then((pl.col('bidSpd') + pl.col('askSpd')) / 2)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias('midSpd'),
        pl.when(pl.col('bidPx').is_not_null() & pl.col('askPx').is_not_null())
        .then((pl.col('bidPx') + pl.col('askPx')) / 2)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias('midPx'),
        pl.when(
            (pl.col('priceSource')=="ALGO") |
            (pl.col('runzSubject').cast(pl.String, strict=False).str.to_lowercase().str.contains("algo", literal=True, strict=False))
        ).then(FLAG_YES).otherwise(FLAG_NO)
        .alias('isAlgoRunz'),
        pl.when(
            (pl.col('bidx') == 1) &
            (pl.col('runzSubject').cast(pl.String, strict=False).str.to_lowercase().str.contains("(bid|buy|buys|bids)", literal=False, strict=False))
        ).then(FLAG_YES).otherwise(pl.lit(None, pl.Int8)).alias('isRunzBidAxe'),
        pl.when(
            (pl.col('askx')==1) &
            (pl.col('runzSubject').cast(pl.String, strict=False).str.to_lowercase().str.contains("(ask|sell|asks|sells|offer|offers)", literal=False, strict=False))
        ).then(FLAG_YES).otherwise(pl.lit(None, pl.Int8)).alias('isRunzAskAxe'),
        pl.col('date').dt.combine(pl.col('time')).dt.replace_time_zone('UTC').alias('refreshTime'),
        pl.col('runzSenderName').str.to_titlecase().alias('runzSenderName'),
        pl.when(pl.col('priceSource').str.starts_with('VELOCITY')).then(FLAG_YES).otherwise(FLAG_NO).alias('isVelocityAxe'),
        pl.col('priceSource').replace({'ALGO': 'algoRunz', 'VELOCITY_US_': 'runzAxe', 'BBG': 'am', 'AM': 'am', 'MARK': 'am'}).alias('_source')
    ])

    rr = await coalesce_left_join(rr, my_pt.select([pl.col('isin'),  pl.col('cusip')]), on='isin')
    rr = await coalesce_left_join(rr, my_pt.select([pl.col('cusip'), pl.col('isin')]), on='cusip')
    rr = await coalesce_left_join(rr, my_pt.select([pl.col('isin').str.slice(2, 9).alias('cusip'), pl.col('isin')]), on='cusip')

    rl = region.lower()

    def c(x):
        if len(x)==1: return x[0].upper()
        return x[0].upper() + x[1:]

    quotes = rr.pivot(on='_source', on_columns=['am', 'algoRunz', 'runzAxe'], index=['isin'],
                      values=['bidPx', 'askPx', 'bidSpd', 'askSpd', 'midSpd', 'midPx', 'refreshTime'], aggregate_function='last')
    quotes = quotes.rename({k: f'{rl}{c(y[-1])}{c(y[0])}' if len(y)==2 else y[0] for k, y in {x: x.split("_") for x in quotes.hyper.fields}.items()})

    axes = rr.group_by(['isin', '_source'], maintain_order=True).agg([
        pl.col('isVelocityAxe').last(ignore_nulls=False),
        pl.col('isRunzBidAxe').last(ignore_nulls=False),
        pl.col('isRunzAskAxe').last(ignore_nulls=False),
        pl.col('isAlgoRunz').last(ignore_nulls=False),
        pl.col('benchmarkCusip').last(ignore_nulls=True).alias('amBenchmarkCusip')
    ]).select([
        pl.col('isin'), pl.col('amBenchmarkCusip'),
        pl.col('isVelocityAxe'),
        pl.when(pl.col('isAlgoRunz')==1).then(pl.col('isRunzBidAxe')).otherwise(pl.lit(None, pl.Int8)).alias('isAlgoRunzBidAxe'),
        pl.when(pl.col('isAlgoRunz')==1).then(pl.col('isRunzAskAxe')).otherwise(pl.lit(None, pl.Int8)).alias('isAlgoRunzAskAxe'),
        pl.when(pl.col('isAlgoRunz')==0).then(pl.col('isRunzBidAxe')).otherwise(pl.lit(None, pl.Int8)).alias('isRunzBidAxe'),
        pl.when(pl.col('isAlgoRunz')==0).then(pl.col('isRunzAskAxe')).otherwise(pl.lit(None, pl.Int8)).alias('isRunzAskAxe'),
    ]).group_by(['isin'], maintain_order=True).agg([
        pl.col('amBenchmarkCusip').first(),
        pl.col('isVelocityAxe').max().fill_null(0).alias(f'{rl}IsVelocityAxe'),
        pl.col('isAlgoRunzBidAxe').last(ignore_nulls=True).fill_null(0).alias(f'{rl}IsAlgoRunzBidAxe'),
        pl.col('isAlgoRunzAskAxe').last(ignore_nulls=True).fill_null(0).alias(f'{rl}IsAlgoRunzAskAxe'),
        pl.col('isRunzBidAxe').last(ignore_nulls=True).fill_null(0).alias(f'{rl}IsRunzBidAxe'),
        pl.col('isRunzAskAxe').last(ignore_nulls=True).fill_null(0).alias(f'{rl}IsRunzAskAxe'),
    ])

    qc = rr.group_by(
        ["isin", "quoteConvention"]
    ).agg(n=pl.len()).sort(
        ["isin", "n", "quoteConvention"], descending=[False, True, False], nulls_last=True
    ).group_by(["isin"], maintain_order=True).agg([
        pl.first("quoteConvention").replace({'PRC': 'PX', 'YS': 'SPD', 'DM': 'DM', 'MMY': 'MMY', 'YLD': 'MMY', 'SPD':'SPD'}).alias(f"{rl}AmQuoteConvention")
    ])

    jmap, nmap = {}, {}
    if swap_juniors:
        if (frames is not None) and ('junior_map' in frames):
            j = frames.get('junior_map')
            if (j is None) or (j.hyper.is_empty()):
                j = await junior_traders()
        else:
            j = await junior_traders()

        if j is not None:
            jmap = j.hyper.to_map('_runzSenderLastName', '_traderLastName')
            nmap = j.hyper.to_map('_runzSenderLastName', '_traderName')

    senders = rr.with_columns([
        pl.when(pl.col('isAlgoRunz').is_not_null() & (pl.col('isAlgoRunz')==1)).then(
            pl.lit(None, pl.String)
        ).when(
            pl.col('runzSenderName').is_not_null() & pl.col('runzSenderName').str.split(" ").list.last().is_in(list(nmap.keys()))
        ).then(
            pl.col('runzSenderName').str.split(" ").list.last().replace(nmap)
        ).otherwise(
            pl.col('runzSenderName')
        ).alias('runzSenderName')
    ]).with_columns([
        pl.col('runzSenderName').str.split(" ").list.last().alias('_runzSenderLastName')
    ]).group_by('isin').agg([
        pl.col('runzSenderName').unique().alias(f'_{rl}RunzSenderName'),
        pl.col('_runzSenderLastName').unique().alias(f'_{rl}RunzSenderLastName')
    ]).with_columns([
        pl.col(f'_{rl}RunzSenderName').list.join(",").alias(f'{rl}RunzSenderName')
    ])

    r = (
        my_pt.select(['isin'])
        .join(quotes.unique(subset=['isin']), on='isin', how='left')
        .join(axes.unique(subset=['isin']), on='isin', how='left')
        .join(qc.unique(subset=['isin']), on='isin', how='left')
        .join(senders.unique(subset=['isin']), on='isin', how='left')
    )

    return await runz_cleaner(r, region=region)

async def quote_runz_creditext(my_pt, region="US", dates=None, frames=None, swap_juniors=True, **kwargs):
    dates = parse_date(dates, biz=True)
    if (region=="EU") and (not is_today(dates)): return
    my_pt = ensure_lazy(my_pt)
    isins = my_pt.hyper.ul('isin')
    cusips = my_pt.hyper.ul('cusip')
    fake_isins = ['XS' + c for c in cusips] + ['US' + c for c in cusips]
    syms = list(set(isins + cusips + fake_isins))

    if syms:
        cols = [
            '0^askSize', '0^askx', '0^bidSize', '0^bidx', 'AskDm:askDM', 'AskMmy:askMMY',
            'AskPx:ask', 'AskSpd:askSpread', 'AskYld:askYieldToWorst', 'benchmarkCusip:fills[benchmark]', 'BidDm:bidDM',
            'BidMmy:bidMMY', 'BidPx:bid', 'BidSpd:bidSpread', 'BidYld:bidYieldToWorst', 'cusip',
            'date', 'isin', 'QuoteConvention:quoteType', 'time'
        ]
        cols = kdb_col_select_helper(cols, method='last')
        dates = latest_biz_date(dates, True)
        q = build_pt_query(
            runz_triplet(region),
            cols=cols,
            by=['sym', 'runzSubject:subject', 'runzSenderName:senderName', 'priceSource'],
            dates=dates,
            filters={'sym'   : syms,
                'priceSource': {'dtype': 'string_exact', 'value': ['AM', 'ALGO', 'BBG', 'VELOCITY_US_', 'MARK']}
            }
        )
        r = await query_kdb(q, config=fconn(GATEWAY_US), name=f"creditext", timeout=5)
        if r is None: return

        rr = r.with_columns(
            [
                pl.when(pl.col('bidSpd').is_not_null() & pl.col('askSpd').is_not_null())
                .then((pl.col('bidSpd') + pl.col('askSpd')) / 2)
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias('midSpd'),
                pl.when(pl.col('bidPx').is_not_null() & pl.col('askPx').is_not_null())
                .then((pl.col('bidPx') + pl.col('askPx')) / 2)
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias('midPx'),
                pl.when(pl.col('bidYld').is_not_null() & pl.col('askYld').is_not_null())
                .then((pl.col('bidYld') + pl.col('askYld')) / 2)
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias('midYld'),
                pl.when(pl.col('bidMmy').is_not_null() & pl.col('askMmy').is_not_null())
                .then((pl.col('bidMmy') + pl.col('askMmy')) / 2)
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias('midMmy'),
                pl.when(pl.col('bidDm').is_not_null() & pl.col('askDm').is_not_null())
                .then((pl.col('bidDm') + pl.col('askDm')) / 2)
                .otherwise(pl.lit(None, dtype=pl.Float64))
                .alias('midDm'),
                pl.when(
                    (pl.col('priceSource')=="ALGO") |
                    (pl.col('runzSubject').cast(pl.String, strict=False).str.to_lowercase().str.contains(
                        "algo", literal=True, strict=False)
                    )
                ).then(FLAG_YES).otherwise(FLAG_NO)
                .alias('isAlgoRunz'),
                pl.when(
                    ((pl.col('bidSize').is_not_null()) &
                     (pl.col('bidSize') > 0) & (pl.col('bidx')==1)) |
                    (pl.col('runzSubject').cast(pl.String, strict=False).str.to_lowercase().str.contains(
                        "(bid|buy|buys|bids)", literal=False, strict=False
                        ))
                ).then(FLAG_YES).otherwise(FLAG_NO).alias('isRunzBidAxe'),
                pl.when(
                    ((pl.col('askSize').is_not_null()) &
                     (pl.col('askSize') > 0) & (pl.col('askx')==1)) |
                    (pl.col('runzSubject').cast(pl.String, strict=False).str.to_lowercase().str.contains(
                        "(ask|sell|asks|sells|offer|offers)", literal=False, strict=False
                        ))
                ).then(FLAG_YES).otherwise(FLAG_NO).alias('isRunzAskAxe'),
                pl.col('date').dt.combine(pl.col('time')).dt.replace_time_zone('UTC').alias('refreshTime'),
                pl.col('bidSize').alias('runzBidAxeSize'),
                pl.col('askSize').alias('runzAskAxeSize'),
                pl.col('runzSenderName').str.to_titlecase().alias('runzSenderName'),
                pl.when(pl.col('priceSource').str.starts_with('VELOCITY')).then(FLAG_YES).otherwise(FLAG_NO).alias(
                    'isVelocityAxe'
                    ),
                pl.col('priceSource').replace(
                    {'ALGO': 'algoRunz', 'VELOCITY_US_': 'runzAxe', 'BBG': 'am', 'AM': 'am', 'MARK': 'am'}
                    ).alias('_source')
            ]
        )

        rr = await coalesce_left_join(rr, my_pt.select([pl.col('isin').alias('sym'), pl.col('isin'), pl.col('cusip')]), on='sym')
        rr = await coalesce_left_join(rr, my_pt.select([pl.col('cusip').alias('sym'), pl.col('isin'), pl.col('cusip')]), on='sym')
        rr = await coalesce_left_join(rr, my_pt.select([pl.col('isin').str.slice(2, 9).alias('sym'), pl.col('isin'), pl.col('cusip')]), on='sym')

        rl = region.lower()

        def c(x):
            if len(x)==1: return x[0].upper()
            return x[0].upper() + x[1:]

        quotes = rr.pivot(on='_source', on_columns=['am', 'algoRunz', 'runzAxe'], index=['isin'],
                          values=['bidPx', 'askPx', 'bidSpd', 'askSpd', 'bidYld', 'askYld', 'bidMmy', 'askMmy', 'bidDm', 'askDm', 'bidSize', 'askSize', 'midSpd', 'midPx', 'midYld', 'midMmy', 'midDm',
                                  'refreshTime'], aggregate_function='last')
        quotes = quotes.rename({k: f'{rl}{c(y[-1])}{c(y[0])}' if len(y)==2 else y[0] for k, y in {x: x.split("_") for x in quotes.hyper.fields}.items()})

        axes = rr.group_by(['isin', '_source'], maintain_order=True).agg([
            pl.col('runzBidAxeSize').last(ignore_nulls=False),
            pl.col('runzAskAxeSize').last(ignore_nulls=False),
            pl.col('isVelocityAxe').last(ignore_nulls=False),
            pl.col('isRunzBidAxe').last(ignore_nulls=False),
            pl.col('isRunzAskAxe').last(ignore_nulls=False),
            pl.col('isAlgoRunz').last(ignore_nulls=False),
            pl.col('benchmarkCusip').last(ignore_nulls=True).alias('amBenchmarkCusip')
        ]).select([
            pl.col('isin'), pl.col('amBenchmarkCusip'),
            pl.when(pl.col('isAlgoRunz')==1).then(pl.col('runzBidAxeSize')).otherwise(pl.lit(None, pl.Float64)).alias('algoRunzBidAxeSize'),
            pl.when(pl.col('isAlgoRunz')==1).then(pl.col('runzAskAxeSize')).otherwise(pl.lit(None, pl.Float64)).alias('algoRunzAskAxeSize'),
            pl.when(pl.col('isAlgoRunz')==0).then(pl.col('runzBidAxeSize')).otherwise(pl.lit(None, pl.Float64)).alias('runzBidAxeSize'),
            pl.when(pl.col('isAlgoRunz')==0).then(pl.col('runzAskAxeSize')).otherwise(pl.lit(None, pl.Float64)).alias('runzAskAxeSize'),
            pl.col('isVelocityAxe'),
            pl.when(pl.col('isAlgoRunz')==1).then(pl.col('isRunzBidAxe')).otherwise(pl.lit(None, pl.Int8)).alias('isAlgoRunzBidAxe'),
            pl.when(pl.col('isAlgoRunz')==1).then(pl.col('isRunzAskAxe')).otherwise(pl.lit(None, pl.Int8)).alias('isAlgoRunzAskAxe'),
            pl.when(pl.col('isAlgoRunz')==0).then(pl.col('isRunzBidAxe')).otherwise(pl.lit(None, pl.Int8)).alias('isRunzBidAxe'),
            pl.when(pl.col('isAlgoRunz')==0).then(pl.col('isRunzAskAxe')).otherwise(pl.lit(None, pl.Int8)).alias('isRunzAskAxe'),
        ]).group_by(['isin'], maintain_order=True).agg([
            pl.col('amBenchmarkCusip').first(),
            pl.col('algoRunzBidAxeSize').last(ignore_nulls=True).fill_null(0).alias(f'{rl}AlgoRunzBidAxeSize'),
            pl.col('algoRunzAskAxeSize').last(ignore_nulls=True).fill_null(0).alias(f'{rl}AlgoRunzAskAxeSize'),
            pl.col('runzBidAxeSize').last(ignore_nulls=True).fill_null(0).alias(f'{rl}RunzBidAxeSize'),
            pl.col('runzAskAxeSize').last(ignore_nulls=True).fill_null(0).alias(f'{rl}RunzAskAxeSize'),
            pl.col('isVelocityAxe').max().fill_null(0).alias(f'{rl}IsVelocityAxe'),
            pl.col('isAlgoRunzBidAxe').last(ignore_nulls=True).fill_null(0).alias(f'{rl}IsAlgoRunzBidAxe'),
            pl.col('isAlgoRunzAskAxe').last(ignore_nulls=True).fill_null(0).alias(f'{rl}IsAlgoRunzAskAxe'),
            pl.col('isRunzBidAxe').last(ignore_nulls=True).fill_null(0).alias(f'{rl}IsRunzBidAxe'),
            pl.col('isRunzAskAxe').last(ignore_nulls=True).fill_null(0).alias(f'{rl}IsRunzAskAxe'),
        ])

        qc = rr.group_by(
            ["isin", "quoteConvention"]
        ).agg(n=pl.len()).sort(
            ["isin", "n", "quoteConvention"], descending=[False, True, False], nulls_last=True
        ).group_by(["isin"], maintain_order=True).agg([
            pl.first("quoteConvention").replace({'PRC': 'PX', 'YS': 'SPD', 'DM': 'DM', 'MMY': 'MMY', 'YLD': 'MMY'}).alias(f"{rl}AmQuoteConvention")
        ])

        jmap, nmap = {}, {}
        if swap_juniors:
            if (frames is not None) and ('junior_map' in frames):
                j = frames.get('junior_map')
                if (j is None) or (j.hyper.is_empty()):
                    j = await junior_traders()
            else:
                j = await junior_traders()

            if j is not None:
                jmap = j.hyper.to_map('_runzSenderLastName', '_traderLastName')
                nmap = j.hyper.to_map('_runzSenderLastName', '_traderName')

        senders = rr.with_columns([
            pl.when(pl.col('isAlgoRunz').is_not_null() & (pl.col('isAlgoRunz')==1)).then(
                pl.lit(None, pl.String)
            ).when(
                pl.col('runzSenderName').is_not_null() & pl.col('runzSenderName').str.split(" ").list.last().is_in(list(nmap.keys()))
            ).then(
                pl.col('runzSenderName').str.split(" ").list.last().replace(nmap)
            ).otherwise(
                pl.col('runzSenderName')
            ).alias('runzSenderName')
        ]).with_columns([
            pl.col('runzSenderName').str.split(" ").list.last().alias('_runzSenderLastName')
        ]).group_by('isin').agg([
            pl.col('runzSenderName').unique().alias(f'_{rl}RunzSenderName'),
            pl.col('_runzSenderLastName').unique().alias(f'_{rl}RunzSenderLastName')
        ]).with_columns([
            pl.col(f'_{rl}RunzSenderName').list.join(",").alias(f'{rl}RunzSenderName')
        ])

        r = (
            my_pt.select(['isin'])
            .join(quotes.unique(subset=['isin']), on='isin', how='left')
            .join(axes.unique(subset=['isin']), on='isin', how='left')
            .join(qc.unique(subset=['isin']), on='isin', how='left')
            .join(senders.unique(subset=['isin']), on='isin', how='left')
        )

        return await runz_cleaner(r, region=region)


def rem_region(x):
    if x.startswith("us"): return x[2:]
    if x.startswith("eu"): return x[2:]
    if x.startswith("sgp"): return x[3:]
    return x


def l(x):
    if len(x)==1: return x.lower()
    return x[0].lower() + x[1:]


async def runz_cleaner(my_pt, region="US", dates=None, **kwargs):
    rl = region.lower()
    mkt_cols = my_pt.hyper.cols_like(f"({rl}Am|{rl}RunzAxe|{rl}AlgoRunz)(Bid|Mid|Ask)(Px|Spd|Mmy|Yld|Dm)")
    if not mkt_cols: return
    clean_mkts = dict()
    for c in mkt_cols:
        new_col = l(rem_region(c))
        if new_col not in clean_mkts: clean_mkts[new_col] = []
        clean_mkts[new_col].append(c)

    return my_pt.with_columns([
        pl.when(pl.col(f'{rl}RunzSenderName').is_not_null()).then(FLAG_YES).otherwise(FLAG_NO).alias(f'_{rl}HasRunz'),
        # TODO
    ])


async def runz_coalesce(my_pt, region="US", dates=None, frames=None, **kwargs):
    mkt_cols = my_pt.hyper.cols_like("(usAm|euAm|sgpAm|usRunzAxe|euRunzAxe|sgpRunzAxe|usAlgoRunz|euAlgoRunz|sgpAlgoRunz)(Bid|Mid|Ask)(Px|Spd|Mmy|Yld|Dm)")
    if not mkt_cols: return

    clean_mkts = dict()
    for c in mkt_cols:
        new_col = l(rem_region(c))
        if new_col not in clean_mkts: clean_mkts[new_col] = []
        clean_mkts[new_col].append(c)

    return my_pt.with_columns([
            pl.max_horizontal([pl.col(c) for c in my_pt.hyper.cols_like(".(AmRefreshTime)$")]).alias('amRefreshTime'),
            pl.max_horizontal([pl.col(c) for c in my_pt.hyper.cols_like(".(RunzRefreshTime)$")]).alias('algoRunzRefreshTime'),
            pl.max_horizontal([pl.col(c) for c in my_pt.hyper.cols_like(".(RunzAxeRefreshTime)$")]).alias('runzAxeRefreshTime')
        ]).with_columns([
            pl.max_horizontal([pl.col('_sgpHasRunz'), pl.col('_euHasRunz'), pl.col('_usHasRunz')]).alias('_hasRunz'),
             pl.concat_list(['_usRunzSenderName', '_euRunzSenderName', '_sgpRunzSenderName']).list.unique().alias('_runzSenderName')
        ]).with_columns([
            pl.col('_runzSenderName').list.eval(pl.element().str.split(" ").list.first()).list.unique().alias('_runzSenderFirstName'),
             pl.col('_runzSenderName').list.eval(pl.element().str.split(" ").list.last()).list.unique().alias('_runzSenderLastName')
        ]).sort(
            ['amRefreshTime', 'algoRunzRefreshTime', 'runzAxeRefreshTime'],descending=[True, True, True]
        ).group_by('isin',maintain_order=True).agg([
            pl.coalesce(v).first().alias(k) for k, v in clean_mkts.items()] + [
            pl.col('_usHasRunz').first(ignore_nulls=True).fill_null(FLAG_NO).alias('_usHasRunz'),
            pl.col('_euHasRunz').first(ignore_nulls=True).fill_null(FLAG_NO).alias('_euHasRunz'),
            pl.col('_sgpHasRunz').first(ignore_nulls=True).fill_null(FLAG_NO).alias('_sgpHasRunz'),
            pl.col('_hasRunz').first(ignore_nulls=True).fill_null(FLAG_NO).alias('_hasRunz'),
            pl.col('_runzSenderName').flatten().unique(), pl.col('_runzSenderFirstName').flatten().unique(),
            pl.col('_runzSenderLastName').flatten().unique(), pl.col('amRefreshTime').first(ignore_nulls=True),
            pl.col('algoRunzRefreshTime').first(ignore_nulls=True), pl.col('runzAxeRefreshTime').first(ignore_nulls=True),
            
            pl.max_horizontal(my_pt.hyper.cols_like("(AlgoRunzBidAxeSize)")).first().fill_null(pl.lit(0, pl.Float64)).alias('algoRunzBidAxeSize'),
            pl.max_horizontal(my_pt.hyper.cols_like("(AlgoRunzBidAxe)$")).first().fill_null(pl.lit(0, pl.Int8)).alias('isAlgoRunzBidAxe'),
            pl.max_horizontal(my_pt.hyper.cols_like("(AmBidSize)")).first().fill_null(pl.lit(0, pl.Float64)).alias('amBidSize'),
            pl.max_horizontal(my_pt.hyper.cols_like("^[a-z]+(RunzBidAxeSize)")).first().fill_null(pl.lit(0, pl.Float64)).alias('runzBidAxeSize'),
            pl.max_horizontal(my_pt.hyper.cols_like("(IsRunzBidAxe)$")).first().fill_null(pl.lit(0, pl.Int8)).alias('isRunzBidAxe'),
            pl.max_horizontal(my_pt.hyper.cols_like("(AlgoRunzAskAxeSize)")).first().fill_null(pl.lit(0, pl.Float64)).alias('algoRunzAskAxeSize'),
            pl.max_horizontal(my_pt.hyper.cols_like("(AlgoRunzAskAxe)$")).first().fill_null(pl.lit(0, pl.Int8)).alias('isAlgoRunzAskAxe'),
            pl.max_horizontal(my_pt.hyper.cols_like("(AmAskSize)")).first().fill_null(pl.lit(0, pl.Float64)).alias('amAskSize'),
            pl.max_horizontal(my_pt.hyper.cols_like("^[a-z]+(RunzAskAxeSize)")).first().fill_null(pl.lit(0, pl.Float64)).alias('runzAskAxeSize'),
            pl.max_horizontal(my_pt.hyper.cols_like("(IsRunzAskAxe)$")).first().fill_null(pl.lit(0, pl.Int8)).alias('isRunzAskAxe'),

            pl.max_horizontal(my_pt.hyper.cols_like("(IsVelocityAxe)$")).first().fill_null(pl.lit(0, pl.Int8)).alias('isVelocityAxe'),
        ]).with_columns([
            pl.when((pl.col('_runzSenderName').list.len()==1) & (pl.col('_runzSenderName').list.get(0).is_null())).then(pl.lit(None, pl.List)).otherwise(pl.col('_runzSenderName')).alias('_runzSenderName'),
             pl.when((pl.col('_runzSenderFirstName').list.len()==1) & (pl.col('_runzSenderFirstName').list.get(0).is_null())).then(pl.lit(None, pl.List)).otherwise(pl.col('_runzSenderFirstName')).alias('_runzSenderFirstName'),
             pl.when((pl.col('_runzSenderLastName').list.len()==1) & (pl.col('_runzSenderLastName').list.get(0).is_null())).then(pl.lit(None, pl.List)).otherwise(pl.col('_runzSenderLastName')).alias('_runzSenderLastName')
            ])


@async_timer
async def _quotes_runz_panoproxy(my_pt, region="US", dates=None, **kwargs):
    schema = my_pt.hyper.schema()
    cusips = my_pt.hyper.to_list('cusip')
    if cusips:
        cols = [
            'benchmarkCusip:benchmark', 'runzSenderName:senderName',
            "runzSubject:`$(subject)", "runzQuoteType:quoteType",
            'amBidPx:bid', 'amAskPx:ask', 'amMidPx:(bid+ask)%2',
            'amBidSpd:bidSpread', 'amAskSpd:askSpread', 'priceSource',
            "runzRefreshTime:time", 'isin'
        ]
        cols = kdb_col_select_helper(cols, method='last')
        dates = latest_biz_date(dates, True)
        triplet = construct_panoproxy_triplet(region, 'bbgrunz', dates)
        q = build_pt_query(triplet, cols=cols, dates=dates, filters={'cusip': cusips, 'priceSource': ['AM', 'ALGO', 'BBG', 'MARK']}, by='cusip')
        r = await query_kdb(q, config=fconn(PANOPROXY_US))

        return r.with_columns([
            pl.when(pl.col('amBidSpd').is_not_null() & pl.col('amAskSpd').is_not_null())
            .then((pl.col('amBidSpd') + pl.col('amAskSpd')) / 2)
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias('amMidSpd'),
            pl.when(
                pl.col('runzSubject')
                .cast(pl.String, strict=False)
                .str.to_lowercase()
                .str.contains("algo", literal=True, strict=False) |
                (pl.col('priceSource')=='ALGO')
            ).then(FLAG_YES).otherwise(FLAG_NO)
            .alias('isAlgoRunz'),
            pl.col('runzSenderName').str.to_titlecase().alias('runzSenderName')
        ]).drop(['priceSource'], strict=False)


# -----------------------------------------------
# -- Positions
# -----------------------------------------------

async def positions(my_pt, region="US", dates=None, frames=None, lookback=90, *, incl_funges=True, as_sequence=True, **kwargs):
    table = "catsPosition" + (region_to_gateway(region).title() if region!="EU" else "")
    triplet = construct_gateway_triplet('internalfeeds', 'EU', table)
    isins = my_pt.hyper.ul('isin')
    funge_map, funge_isins = {}, []
    frames = frames or {}
    if incl_funges and (frames.get('funges') is not None):
        funge_map = frames.get('funges').filter(~pl.col('fungibleIsin').is_in(isins)).hyper.to_map('fungibleIsin', 'isin')
        funge_isins = list(funge_map.keys())
        isins += funge_isins
    rt = region.title()
    cols = [
        f'netPosition:last Position',
        'last IsDesig',
        'last TraderId',
        'last DeskType',
        f'house{rt}BenchmarkIsin:last BenchmarkIsin',
        'totalGrossTradeQuantity:sum abs[TradeQuantity]',
        'numberTrades:count TradeQuantity>0',
        'daysSinceLastTrade:last (.z.d - TradeExecutionTimeUtc.date)',
        'totalHistoricalPosition:sum[abs[Position]]',
        'maxHistoricalPosition:max[abs[Position]]',
        'historicalDesig:avg[IsDesig]'
    ]
    cols = kdb_col_select_helper(cols, None)
    lookback = abs(lookback) if lookback else 0
    if lookback > 0:
        lookback_date = safe_date_lookback(dates, lookback)
        q = build_pt_query(triplet, cols, dates=lookback_date, date_kwargs={'between': True, 'as_sequence': as_sequence}, filters={'sym': isins, 'not BookId': BAD_BOOKS}, by='isin:sym, BookId',
                           lastby=['sym'])
    else:
        dates = latest_biz_date(dates, True)
        q = build_pt_query(triplet, cols, dates=dates, date_kwargs={'return_today': False}, filters={'sym': isins, 'not BookId': BAD_BOOKS}, by='isin:sym, BookId', lastby=['sym'])

    r = await query_kdb(q, fconn(GATEWAY))
    if r is None: return
    funge_mask = pl.col('isin').is_in(funge_isins)
    return await r.select([
        pl.col('isin').replace(funge_map).alias('isin'),
        pl.col('deskType').replace_strict(L1_DESK_MAP, default="OTHER", return_dtype=pl.String).alias('deskAsset'),
        pl.col('bookId'),
        pl.lit(region, pl.String).alias('bookRegion'),
        pl.col('isDesig'),
        pl.col('traderId'),
        pl.col(f'house{rt}BenchmarkIsin'),
        pl.col('totalGrossTradeQuantity').fill_null(0).alias('_totalGrossTradeQuantity'),
        pl.col('totalHistoricalPosition').fill_null(0).alias('_totalHistoricalPosition'),
        pl.col('historicalDesig').fill_null(0).alias('_historicalDesig'),
        pl.col('maxHistoricalPosition').fill_null(0).alias('_maxHistoricalPosition'),
        pl.col('numberTrades').fill_null(0).alias('_numberTrades'),
        pl.col('daysSinceLastTrade').fill_null(0).alias('_daysSinceLastTrade'),
        pl.when(funge_mask).then(FLAG_YES).otherwise(FLAG_NO).alias('_isFungePosition'),
        pl.col('netPosition').cast(pl.Float64, strict=False).fill_null(0).alias('netPosition'),
    ]).with_columns([
        pl.when(pl.col('_isFungePosition')==1).then(FLAG_NO).otherwise(FLAG_YES).alias('_isTruePosition')
    ]).group_by([pl.col('isin', 'bookId')]).agg([
        pl.col('netPosition').sum().alias('netPosition'),
        (pl.col('_isTruePosition') * pl.col('netPosition')).sum().alias('netTruePosition'),
        (pl.col('_isFungePosition') * pl.col('netPosition')).sum().alias('netFungePosition'),
        pl.col(f'house{rt}BenchmarkIsin').first(ignore_nulls=True),
        pl.col(f'deskAsset').first(ignore_nulls=True),
        pl.col('traderId').first(ignore_nulls=True),
        pl.col('bookRegion').first(ignore_nulls=True),
        pl.col('_totalGrossTradeQuantity').sum().alias('_totalGrossTradeQuantity'),
        pl.col('_totalHistoricalPosition').sum().alias('_totalHistoricalPosition'),
        pl.col('_maxHistoricalPosition').max().alias('_maxHistoricalPosition'),
        pl.col('_numberTrades').sum().alias('_numberTrades'),
        pl.col('_daysSinceLastTrade').min().alias('_daysSinceLastTrade'),
        pl.col('isDesig').max().alias('isDesig'),
        (pl.col('_isTruePosition') * pl.col('isDesig')).max().alias('_isTrueDesig'),
        (pl.col('_isFungePosition') * pl.col('isDesig')).max().alias('_isFungeDesig'),
        (pl.col('_isTruePosition') * pl.col('_historicalDesig')).max().alias('_historicalTrueDesig'),
        (pl.col('_isFungePosition') * pl.col('_historicalDesig')).max().alias('_historicalFungeDesig'),
    ]).hyper.compress_plan_async()


@hypercache.cached(ttl=timedelta(hours=12), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'region'])
async def lcs(my_pt, region="US", dates=None, **kwargs):
    isins = my_pt.hyper.ul('isin')
    triplet = construct_panoproxy_triplet(region, 'bondpositions', dates)
    cols = kdb_col_select_helper(['lcsLiqScore:liquidityCostScore'], "last")
    filters = [{"securityAltId3": isins, 'not liquidityCostScore': 0}, {'not liquidityCostScore': None}]
    dates = latest_biz_date(dates, True)
    q = build_pt_query(triplet, cols, dates, date_kwargs={"return_today": False}, filters=filters, by=['isin:securityAltId3'])
    pano_region = "US" if region=="SGP" else region
    return await query_kdb(q, fconn(PANOPROXY, region=pano_region))


IS_FUNGE = pl.col('_isFungePosition')==1

async def pano_positions(my_pt, region="US", dates=None, frames=None, lookback=0, incl_funges=True, filter_books=False, remove_bad_books=True, **kwargs):
    isins = my_pt.hyper.ul('isin')
    funge_map, funge_isins = {}, []
    frames = frames or {}
    if incl_funges and (frames.get('funges') is not None) and (not frames.get('funges').hyper.is_empty()):
        funge_map = frames.get('funges').filter(
            ~pl.col('fungibleIsin').is_in(isins)
        ).hyper.to_map('fungibleIsin', 'isin')
        funge_isins = list(funge_map.keys())
        isins += funge_isins
    dates = latest_biz_date(dates, True)
    triplet = construct_panoproxy_triplet(region, 'bondpositions', dates)
    rt = region.title()
    cols = [
        'sym',
        'netPosition:position',
        'lcsLiqScore:liquidityCostScore',
        'traderId',
        f'house{rt}BenchmarkIsin:benchmarkISIN',
        'isDesig:desig',
        'deskType',
        'weightedPositionAge:weightedAge',
        'weightedPositionAgeBucket:weightedAgeBucket',
        'hasTradedToday:min[isSnapshot]=0b'
    ]
    cols = kdb_col_select_helper(cols, "last")
    filters = {"securityAltId3": isins}
    if filter_books and my_pt.hyper.has_columns('bookId'):
        filters['bookId'] = my_pt.hyper.ul('bookId')
    if lookback:
        q = build_pt_query(
            triplet,
            cols,
            safe_date_lookback(dates, lookback),
            date_kwargs={"return_today": False, 'between': True, "as_sequence": True},
            filters=filters,
            by=['isin:securityAltId3', 'bookId']
        )
    else:
        q = build_pt_query(
            triplet, cols, dates, date_kwargs={"return_today": False}, filters=filters,
            by=['isin:securityAltId3', 'bookId']
        )
    if remove_bad_books and ('bookId' not in filters):
        q += (',((not bookId in (%s)) | (desig=1))' % kdb_convert_series_to_sym(BAD_BOOKS))
    pano_region = "US" if region=="SGP" else region
    r = await query_kdb(q, fconn(PANOPROXY, region=pano_region))
    if r is None: return
    funge_mask = pl.col('isin').is_in(funge_isins)

    cachable = r.filter([
        pl.col('lcsLiqScore').is_not_null(),
        pl.col('lcsLiqScore') != 0
    ]).select(['isin', 'lcsLiqScore']).unique(subset=['isin'])
    await write_to_cache_async('lcs', cachable, cachable.select(['isin']), force=False)

    return await r.with_columns([
        pl.lit(region, pl.String).alias('bookRegion'),
        pl.when(funge_mask).then(FLAG_YES).otherwise(FLAG_NO).alias('_isFungePosition'),
        pl.col('isin').replace(funge_map).alias('isin'),
        pl.col('deskType').replace_strict(L1_DESK_MAP, default="OTHER", return_dtype=pl.String).alias("deskAsset"),
        pl.col('netPosition').cast(pl.Float64, strict=False).fill_null(0).alias('netPosition'),
        pl.col('lcsLiqScore').cast(pl.Float64, strict=False).fill_null(0).alias('lcsLiqScore'),
    ]).with_columns([
        pl.when(IS_FUNGE).then(FLAG_NO).otherwise(FLAG_YES).alias('_isTruePosition'),
        pl.when(IS_FUNGE).then(FLAG_YES).otherwise(pl.lit(None, pl.Int8)).alias('_isFungeDesig'),
        pl.when(~IS_FUNGE).then(pl.col('isDesig')).otherwise(pl.lit(None, pl.Int8)).alias('_isTrueDesig'),
    ]).group_by([pl.col('isin', 'bookId')]).agg([
        pl.col('netPosition').sum().alias('netPosition'),
        (pl.col('_isTruePosition') * pl.col('netPosition')).sum().alias('netTruePosition'),
        (pl.col('_isFungePosition') * pl.col('netPosition')).sum().alias('netFungePosition'),
        pl.col(f'house{rt}BenchmarkIsin').first(ignore_nulls=True),
        pl.col('weightedPositionAge').min().alias('weightedPositionAge'),
        pl.col('traderId').first(ignore_nulls=True),
        pl.col('isDesig').max().alias('isDesig'),
        pl.col('_isFungeDesig').max().alias('_isFungeDesig'),
        pl.col('_isTrueDesig').max().alias('_isTrueDesig'),
        pl.coalesce([
            (pl.col('lcsLiqScore') * pl.col('_isTruePosition')),
            (pl.col('lcsLiqScore') * pl.col('_isFungePosition'))
        ]).first(ignore_nulls=True).alias('lcsLiqScore'),
        pl.col('deskAsset').first(ignore_nulls=True),
        pl.col('bookRegion').first(ignore_nulls=True)
    ]).hyper.compress_plan_async()


async def position_aggregator(my_pt, region="US", dates=None, frames=None, **kwargs):
    # We dont know WHICH books are desig at this point
    us, eu, sgp = frames.get('realtime_positions_us', None), frames.get('realtime_positions_eu', None), frames.get('realtime_positions_sgp', None)
    regions = [r for r in (us, eu, sgp) if r is not None]
    if not regions: return
    r = await pl.concat(regions, how='diagonal_relaxed', strict=False).hyper.compress_plan_async()

    from app.helpers.common import get_algo_books, CRB_STRATEGY_BOOKS
    algo_books = list(set(await get_algo_books()) - set(CRB_STRATEGY_BOOKS))
    strategy_books = CRB_STRATEGY_BOOKS

    algo_mask = pl.col("bookId").is_in(algo_books)
    strategy_mask = pl.col("bookId").is_in(strategy_books)
    desk_mask = (~algo_mask & ~strategy_mask)
    us_mask = pl.col('bookRegion')=='US'
    eu_mask = pl.col('bookRegion')=='EU'
    sgp_mask = pl.col('bookRegion')=='SGP'

    return await r.with_columns([
        pl.when(algo_mask).then(FLAG_YES).otherwise(FLAG_NO).alias('_algoMask'),
        pl.when(strategy_mask).then(FLAG_YES).otherwise(FLAG_NO).alias('_strategyMask'),
        pl.when(desk_mask).then(FLAG_YES).otherwise(FLAG_NO).alias('_deskMask'),
        pl.when(us_mask).then(FLAG_YES).otherwise(FLAG_NO).alias('_usMask'),
        pl.when(eu_mask).then(FLAG_YES).otherwise(FLAG_NO).alias('_euMask'),
        pl.when(sgp_mask).then(FLAG_YES).otherwise(FLAG_NO).alias('_sgpMask'),
    ]).with_columns([
        pl.when((pl.col('netPosition')!=0) | (pl.col('isDesig')==1)).then(pl.col('bookId')).otherwise(pl.lit(None, pl.String)).alias('_firmBook'),
        pl.when((pl.col('_algoMask')==1) & ((pl.col('netPosition')!=0) | (pl.col('isDesig')==1))).then(pl.col('bookId')).otherwise(pl.lit(None, pl.String)).alias('_algoBook'),
        pl.when((pl.col('_strategyMask')==1) & ((pl.col('netPosition')!=0) | (pl.col('isDesig')==1))).then(pl.col('bookId')).otherwise(pl.lit(None, pl.String)).alias('_strategyBook'),
        pl.when((pl.col('_deskMask')==1) & ((pl.col('netPosition')!=0) | (pl.col('isDesig')==1))).then(pl.col('bookId')).otherwise(pl.lit(None, pl.String)).alias('_deskBook'),
    ]).group_by(['isin']).agg([
        pl.col('netPosition').sum().alias('netFirmPosition'),
        pl.col('netTruePosition').sum().alias('netTrueFirmPosition'),
        pl.col('netFungePosition').sum().alias('netFungeFirmPosition'),

        pl.struct(['netPosition', '_firmBook']).sort_by('netPosition').struct.field('_firmBook').drop_nulls().implode().list.unique(maintain_order=False).list.join(",").alias('firmBookIds'),
        pl.struct(['netPosition', '_algoBook']).sort_by('netPosition').struct.field('_algoBook').drop_nulls().implode().list.unique(maintain_order=False).list.join(",").alias('algoBookIds'),
        pl.struct(['netPosition', '_strategyBook']).sort_by('netPosition').struct.field('_strategyBook').drop_nulls().implode().list.unique(maintain_order=False).list.join(",").alias(
            'strategyBookIds'),
        pl.struct(['netPosition', '_deskBook']).sort_by('netPosition').struct.field('_deskBook').drop_nulls().implode().list.unique(maintain_order=False).list.join(",").alias('deskBookIds'),

        (pl.col('netPosition') * pl.col('_algoMask')).sum().alias('netAlgoPosition'),
        (pl.col('netPosition') * pl.col('_strategyMask')).sum().alias('netStrategyPosition'),
        (pl.col('netPosition') * pl.col('_deskMask')).sum().alias('netDeskPosition'),
        (pl.col('netPosition') * pl.col('_usMask')).sum().alias('netUsFirmPosition'),
        (pl.col('netPosition') * pl.col('_euMask')).sum().alias('netEuFirmPosition'),
        (pl.col('netPosition') * pl.col('_sgpMask')).sum().alias('netSgpFirmPosition'),

        (pl.col('netTruePosition') * pl.col('_algoMask')).sum().alias('netTrueAlgoPosition'),
        (pl.col('netFungePosition') * pl.col('_algoMask')).sum().alias('netFungeAlgoPosition'),
        (pl.col('netTruePosition') * pl.col('_strategyMask')).sum().alias('netTrueStrategyPosition'),
        (pl.col('netFungePosition') * pl.col('_strategyMask')).sum().alias('netFungeStrategyPosition'),
        (pl.col('netTruePosition') * pl.col('_deskMask')).sum().alias('netTrueDeskPosition'),
        (pl.col('netFungePosition') * pl.col('_deskMask')).sum().alias('netFungeDeskPosition'),
    ]).select([
        pl.col('isin'), pl.col('netAlgoPosition'), pl.col('netStrategyPosition'), pl.col('netDeskPosition'),
        pl.col('netUsFirmPosition'), pl.col('netEuFirmPosition'), pl.col('netSgpFirmPosition'), pl.col('netFirmPosition'),
        pl.col('netTrueFirmPosition'), pl.col('netFungeFirmPosition'),
        pl.col('netTrueAlgoPosition'), pl.col('netFungeAlgoPosition'),
        pl.col('netTrueStrategyPosition'), pl.col('netFungeStrategyPosition'),
        pl.col('netTrueDeskPosition'), pl.col('netFungeDeskPosition'),
        pl.when(pl.col('firmBookIds')=="").then(pl.lit(None, pl.String)).otherwise(pl.col('firmBookIds')).alias('firmBookIds'),
        pl.when(pl.col('algoBookIds')=="").then(pl.lit(None, pl.String)).otherwise(pl.col('algoBookIds')).alias('algoBookIds'),
        pl.when(pl.col('strategyBookIds')=="").then(pl.lit(None, pl.String)).otherwise(pl.col('strategyBookIds')).alias('strategyBookIds'),
        pl.when(pl.col('deskBookIds')=="").then(pl.lit(None, pl.String)).otherwise(pl.col('deskBookIds')).alias('deskBookIds'),
    ]).hyper.compress_plan_async()


# OH SNAP, select from .internalfeeds.ldn.catsPosition where sym = `FR001400WJR8
# dv01, cs01pct, select from .creditcalc.bk2risk where date=.z.d, sym=`1000115560, cs01


##########################################################
## SALES
##########################################################


@hypercache.cached(ttl=timedelta(days=3))
async def tw_sales(region="US"):
    gw_table = construct_gateway_triplet('enablement', region, 'twUsers')
    kdb_date = next_biz_date(None, -1).strftime('%Y.%m.%d')
    q = f"select last legalName, last businessType, last statusDate by firstName, lastName, client:sym, salesName:primarySales from {gw_table} where date>={kdb_date}"
    res = await query_kdb(q, fconn(GATEWAY))
    if res is None: return
    return res.select([
        pl.col('client').str.to_titlecase().alias("client"),
        pl.col('statusDate').cast(pl.Date, strict=False),
        pl.concat_str([pl.col('firstName'), pl.col('lastName')], separator=" ").str.to_titlecase().alias('clientTraderName'),
        pl.col('salesName').str.to_titlecase().alias('salesName'),
        pl.col('legalName').str.to_uppercase().alias('legalName'),
        pl.col('businessType').alias('businessType')
    ]).with_columns([
        pl.col('salesName').is_not_null().any().over(['client', 'clientTraderName', 'legalName', 'businessType']).alias('_has_non_null')
    ]).filter(~(pl.col('salesName').is_null() & pl.col('_has_non_null'))).drop(['_has_non_null'])


@hypercache.cached(ttl=timedelta(days=3))
async def mx_sales():
    gw_table = construct_gateway_triplet('enablement', "US", 'maDcsr')  # regions are identical here
    cols = kdb_col_select_helper(['aggressorShortName', 'statusDate:date'], "last")
    kdb_date = next_biz_date(None, -1).strftime('%Y.%m.%d')
    q = f"select {cols} by client, clientTraderName:contact, salesName:dealerContact from {gw_table} where date>={kdb_date}"
    res = await query_kdb(q, fconn(GATEWAY))
    if res is None: return
    return res.select([
        pl.col('client').str.to_titlecase().alias("client"),
        pl.col('aggressorShortName').str.to_titlecase().alias('clientShortName'),
        pl.col('statusDate').cast(pl.Date, strict=False),
        pl.col('clientTraderName').str.to_titlecase().alias('clientTraderName'),
        pl.col('salesName').replace({"-": None}).str.replace_all("*", "", literal=True).str.to_titlecase().alias('salesName'),
    ]).with_columns([
        pl.col('salesName').is_not_null().any().over(['client', 'clientTraderName']).alias('_has_non_null')
    ]).filter(~(pl.col('salesName').is_null() & pl.col('_has_non_null'))).drop(['_has_non_null'])


@hypercache.cached(ttl=timedelta(days=3))
async def mx_sales_alt():
    gw_table = construct_gateway_triplet('enablement', "US", 'mxDccp')  # regions are identical here
    cols = kdb_col_select_helper(['clientShortName', 'statusDate:date'], "last")
    kdb_date = next_biz_date(None, -1).strftime('%Y.%m.%d')
    q = f"select {cols} by client:clientCompany, clientTraderName:userFullName, salesName:salesCoverageUserLevel, salesNameAlt:salesCoverageUserLevel, clientType from {gw_table} where date>={kdb_date}"
    res = await query_kdb(q, fconn(GATEWAY))
    if res is None: return
    res = pl.concat([
        res.select([pl.col('client'), pl.col('clientTraderName'), pl.col('clientType'), pl.col('clientShortName'), pl.col('statusDate'), pl.col('salesName')]),
        res.select([pl.col('client'), pl.col('clientTraderName'), pl.col('clientType'), pl.col('clientShortName'), pl.col('statusDate'), pl.col('salesNameAlt').alias('salesName')])
    ], how="vertical")
    return res.select([
        pl.col('client').str.to_titlecase().alias("client"),
        pl.col('statusDate').cast(pl.Date, strict=False),
        pl.col('clientTraderName').str.to_titlecase().alias('clientTraderName'),
        pl.col('clientType'),
        pl.col('clientShortName'),
        pl.col('salesName').replace({"-": None}).str.replace_all("*", "", literal=True).str.to_titlecase().alias('salesName'),
    ]).with_columns([
        pl.col('salesName').is_not_null().any().over(['client', 'clientTraderName']).alias('_has_non_null')
    ]).filter(~(pl.col('salesName').is_null() & pl.col('_has_non_null'))).drop(['_has_non_null'])

@hypercache.cached(ttl=timedelta(days=3))
async def trumid_sales():
    gw_table = construct_gateway_triplet('enablement', "US", 'trumidClientList')
    kdb_date = next_biz_date(None, -1).strftime('%Y.%m.%d')
    cols = kdb_col_select_helper(['igSds', 'hySds', 'emSds', 'statusDate:lastModified'], "last")
    q = f"select {cols} by client:sym, clientTraderName:trader, igSalesName:igPrimarySales, hySalesName:hyPrimarySales, emSalesName:emPrimarySales from {gw_table} where date>={kdb_date}"
    res = await query_kdb(q, fconn(GATEWAY))
    if res is None: return
    base_cols = [pl.col('client'), pl.col('clientTraderName'), pl.col('statusDate')]

    res = pl.concat([
        res.select(base_cols + [pl.col('igSds').alias('sds'), pl.col('igSalesName').alias('salesName'), pl.lit('IG').alias('asset')]),
        res.select(base_cols + [pl.col('hySds').alias('sds'), pl.col('hySalesName').alias('salesName'), pl.lit('HY').alias('asset')]),
        res.select(base_cols + [pl.col('emSds').alias('sds'), pl.col('emSalesName').alias('salesName'), pl.lit('EM').alias('asset')]),
    ], how="vertical")

    return res.select([
        pl.col('client').str.to_titlecase().alias("client"),
        pl.col('statusDate').cast(pl.Date, strict=False),
        pl.col('clientTraderName').str.to_titlecase().alias('clientTraderName'),
        pl.col('sds').cast(pl.String, strict=False),
        pl.col('asset'),
        pl.col('salesName').replace({"-": None}).str.replace_all("*", "", literal=True).str.to_titlecase().alias('salesName'),
    ]).with_columns([
        pl.col('salesName').is_not_null().any().over(['client', 'clientTraderName']).alias('_has_non_null')
    ]).filter(~(pl.col('salesName').is_null() & pl.col('_has_non_null'))).drop(['_has_non_null'])


async def sws_sales(sdsId, source=None, product=None, raw=False):
    source = source or 'BB;BV;TW;MX;YB;CTI;BLACKBIRD;TR'
    product = product or 'ADN;APCRED;CDS-EM;CDS-EUHG;CDS-EUHY;CDS-USHG;CDS-USHY;CP;EMS;EUCR;EUCRED;EUEMS;EUHGS;EUHYS;MTG;MUNI-LONG;USCRED;USEMS;USGHS;USHGS;USHYS;USMTX;USPRD;VRDN'
    if isinstance(source, list): source = ";".join(source)
    if isinstance(product, list): product = ";".join(product)

    cookies = {
        'JSESSIONID': '7A032D7F1600AB42F19BED35B7743B3F.sysswsnyssprod_CREDITMAIN_nykpsr000009302',
        'utag_main_v_id': '01986158c23900177db3cbe109230506f009e06700c48',
        'utag_main__sn': '6',
        'utag_main_dc_visit': '6',
        'mbox': 'PC#8521c53b5b50459ebea4130e4ce04a24.34_0#1834836706|session#49fbc8a7eda94c5090b6d78b3ca74707#1771593766',
    }
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Origin': 'http://sws-nyk-all.barcapint.com',
        'Referer': 'http://sws-nyk-all.barcapint.com/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
        'Cache-Control': 'no-cache',
        'Host': 'sws-nyk-credit.barcapint.com',
        'Pragma': 'no-cache',
        'Proxy-Connection': 'keep-alive',
        'Sec-Fetch-Mode': 'cors',
        'Access-Control-Request-Headers': 'content-type',
        'Access-Control-Request-Method': 'POST'
    }

    data = {'boolean': False,
            'ServiceName': 'GetEcnMappingsService',
            'source': f'{source}',
            'product': f'{product}',
            'companyId': None,
            'salesPersonId': None,
            'ecnUserId': None,
            'isdefaultSP': True,
            'isBothSPTypeSelected': False,
            'counterpartyId': f'{sdsId}',
            'requestId': '33fcaaf9-3ae8-dfe7-1cdd-66afe7a9c998',
            'userTimeZone': 'America/New_York'}

    from httpx_negotiate_sspi import HttpSspiAuth
    auth = HttpSspiAuth(
        service='HTTP',
        host='sws-nyk-credit.barcapint.com',
        delegate=False
    )
    async with httpx.AsyncClient(follow_redirects=True, verify=False) as s:
        r = await s.options(
            'http://sws-nyk-credit.barcapint.com/SWS/RESTful/RequestReply/GetEcnCompanyNameBySDSService',
            # json={"boolean":False,"ServiceName":"GetEcnCompanyNameBySDSService","searchString":"42777913","requestId":"e8e13829-1152-2b81-1ad0-5bb43390178f","userTimeZone":"America/New_York"}
        )
        headers['www-authenticate'] = r.headers['www-authenticate']
        r = await s.post(
            'http://sws-nyk-credit.barcapint.com/SWS/RESTful/RequestReply/GetEcnCompanyNameBySDSService',
            headers=headers,
            json={"boolean": False, "ServiceName": "GetEcnCompanyNameBySDSService", "searchString": "42777913", "requestId": "e8e13829-1152-2b81-1ad0-5bb43390178f", "userTimeZone": "America/New_York"}
        )

        r = await s.post(
            'http://sws-nyk-credit.barcapint.com/SWS/RESTful/RequestReply/GetEcnMappingsService',
            headers=headers,
            data=data,
            cookies=cookies,
            timeout=5
        )

    async with httpx.AsyncClient(auth=auth, follow_redirects=True, verify=False) as client:
        # r1 = await client.get('http://sws-nyk-all.barcapint.com/', headers=headers, cookies=cookies)

        res = await client.post(
            'http://sws-nyk-credit.barcapint.com/SWS/RESTful/RequestReply/GetEcnMappingsService',
            json=data,
            cookies=cookies,
            headers=headers
        )

    if res.status_code==200:
        base = pl.DataFrame(res.json()).sort('updateDate', descending=True, nulls_last=True)
        return base if raw else base.filter([
            pl.col('salesPersonName').is_not_null(),
            pl.col('salesPersonName').str.strip_chars()!=''
        ]).select('companyName', 'ecn', 'companyId', 'updateDate', 'login', 'salesPersonName', 'productType', 'compDefSalesPersonId').group_by(['productType', 'ecn'], maintain_order=True).agg(
            pl.all().first())

async def get_mx_sales():
    try:
        tbl = await mx_sales()
    except Exception:
        tbl = await mx_sales_alt()
    return tbl

async def sales_person_lookup(my_pt, region="US", dates=None, similar_threshold=0.8, force_all=False, frames=None, **kwargs):
    rfq_sales = frames['raw'].hyper.peek('partyName')
    if (not force_all) and (rfq_sales is not None) and (str(rfq_sales).strip() != ''):
        lastName = rfq_sales.strip().split(" ")[-1]
        return my_pt.select(
            [
                META_MERGE_EXPR,
                pl.lit(lastName, pl.String).alias('salesName'),
                pl.col('venueShort'),
                pl.col('rfqClient'),
                pl.col('assetClass')
            ]
        )
    from app.server import get_pb
    venue = my_pt.hyper.peek('venueShort')
    if (venue =='MX') and not force_all:
        tbl = await get_mx_sales()
    elif (venue =="TW") and not force_all:
        tbl = await tw_sales()
    elif (venue =="TRM") and not force_all:
        tbl = await trumid_sales()
    else:
        force_all = True
        comb = await asyncio.gather(*[
            asyncio.create_task(get_mx_sales()),
            asyncio.create_task(tw_sales()),
            asyncio.create_task(trumid_sales())
        ], return_exceptions=True)
        comb = [tbl for tbl in comb if isinstance(tbl, pl.LazyFrame)]
        tbl = pl.concat(comb, how='diagonal_relaxed')

    if (tbl is None) or (tbl.hyper.is_empty()): return

    res = None

    sds = frames['raw'].hyper.peek('rfqSdsId')
    if (sds is not None) and (str(sds).strip() != ""):
        sds_match = tbl.hyper.ensure_columns(['sds'], dtypes={"sds": pl.String}).filter(pl.col('sds') == str(sds))
        if not sds_match.hyper.is_empty():
            res = sds_match

    if res is None:
        rfq_trader = str((my_pt.hyper.peek('clientTraderName') or '')).strip()
        for client_source  in ['rfqClient', 'client', 'clientBcName', 'clientAltName', 'clientUbcName', 'client']:

            rfq_client = (my_pt.hyper.peek(client_source) or '').upper()
            try:
                res = await _sales_person_lookup_inner(tbl, rfq_client, rfq_trader)
                if res.hyper.peek('_similarity_client') > similar_threshold:
                    break
            except Exception:
                pass
            res = None

    if (res is not None) and (not res.hyper.is_empty()):
        res = res.with_columns([
            pl.col('salesName').str.split(" ").list.get(-1).alias('lastName')
        ])
        pb = (await get_pb(trading_only=False, lazy=True)).select([
            pl.col('name'), pl.col('lastName'), pl.col('asset')
        ])
        res = await coalesce_left_join(res, pb.select([pl.col('name').alias('salesName'), pl.col('asset')]), on='salesName')
        res = await coalesce_left_join(res, pb.select([pl.col('lastName'), pl.col('asset')]), on='lastName')
        my_asset = my_pt.hyper.peek('assetClass') or ''
        res = res.filter(pl.col('asset').is_in(my_asset.split(",")))

    if (res is not None) and (not res.hyper.is_empty()) and (len(res.hyper.ul('salesName')) == 1):
        return my_pt.select([
            META_MERGE_EXPR,
            pl.lit(", ".join(sorted(res.hyper.ul('salesName'))), pl.String).alias('salesName'),
            pl.col('venueShort'),
            pl.col('rfqClient'),
            pl.col('assetClass')
        ])
    elif (res is not None) and (not res.hyper.is_empty()) and (len(res.hyper.ul('lastName')) <= 3):
            return my_pt.select([
                META_MERGE_EXPR,
                pl.lit(", ".join(sorted(res.hyper.ul('lastName'))), pl.String).alias('salesName'),
                pl.col('venueShort'),
                pl.col('rfqClient'),
                pl.col('assetClass')
            ])
    if not force_all:
        return await sales_person_lookup(my_pt, region=region, dates=dates, similar_threshold=similar_threshold, force_all=True)

async def _sales_person_lookup_inner(tbl, client, trader):
    from polars_strsim import jaro_winkler
    return tbl.with_columns([
        pl.col('client').str.replace_all(r'[^\w\s]', '').str.to_uppercase().alias('client'),
        pl.col('clientTraderName').str.replace_all(r'[^\w\s]', '').str.to_uppercase().alias('clientTraderName'),
        pl.lit(client, pl.String).str.replace_all(r'[^\w\s]', '').str.to_uppercase().alias('targetClient'),
        pl.lit(trader, pl.String).str.replace_all(r'[^\w\s]', '').str.to_uppercase().alias('targetTrader')
    ]).with_columns([
        jaro_winkler(pl.col('targetClient'), pl.col('client')).alias('_similarity_client'),
        pl.when(pl.col('targetTrader').is_not_null()).then(
            jaro_winkler(pl.col('targetTrader'), pl.col('clientTraderName'))
        ).otherwise(pl.lit(0.0, pl.Float64))
        .alias('_similarity_trader'),
    ]).filter([
        pl.col('_similarity_client') == pl.col('_similarity_client').max()
    ]).filter([
        pl.col('_similarity_trader')==pl.col('_similarity_trader').max()
    ])

@hypercache.cached(ttl=timedelta(days=2), deep={"my_pt": True}, primary_keys={'my_pt': ["rfqClient", "assetClass"]}, key_params=['my_pt', 'similar_threshold'])
async def client_flag(my_pt, region="US", dates=None, similar_threshold=0.8, frames=None, **kwargs):
    from app.server import get_db
    from polars_strsim import jaro_winkler
    my_asset = my_pt.hyper.peek('assetClass')
    my_asset = my_asset.split(",") if my_asset is not None else None
    if my_asset:
        res = await get_db().select('sales', filters={'asset': my_asset})
    else:
        res = await get_db().select('sales')

    m = None
    for client_source in ['rfqClient', 'client', 'clientBcName']:
        source_value = my_pt.hyper.peek(client_source)
        if (source_value is None) or (source_value == ''): continue
        m = res.with_columns([
            pl.lit(source_value, pl.String).str.strip_chars().str.to_uppercase().alias('target'),
            pl.col('account').str.strip_chars().str.to_uppercase().alias('account')
        ]).with_columns([
            pl.max_horizontal([
                jaro_winkler(pl.col('target'), pl.col('account')),
                jaro_winkler(pl.col('target'), pl.col('altNames'))
            ]).alias('_similarity'),
        ]).filter([
            pl.col('_similarity') > similar_threshold,
            (pl.col('_similarity') == pl.col('_similarity').max())
        ])
        if (m is not None) and (not m.hyper.is_empty()):
            break

    is_ttt, is_m100 = 0, 0
    sales = None
    if (m is not None) and (not m.hyper.is_empty()):
        flags = m.hyper.to_set('clientFlag')
        if 'TTT' in flags:
            is_ttt = 1
            is_m100 = 1
        elif 'M100' in flags:
            is_m100 = 1

        raw_sales = m.hyper.ul('salesName')
        if raw_sales:
            joined = ", ".join([s for s in raw_sales if s is not None])
            if joined: sales = joined

    return my_pt.select([
        META_MERGE_EXPR,
        pl.lit(is_ttt, pl.Int8).alias('isClientTtt'),
        pl.lit(is_m100, pl.Int8).alias('isClientM100'),
        pl.lit(sales, pl.String).alias('salesName'),
        pl.col("rfqClient"),
        pl.col("assetClass")
    ])


############################################
## Benchmark new path
############################################

async def benchmark_fill(my_pt=None, region="US", dates=None, **kwargs):
    isins = my_pt.hyper.to_list('isin')
    cols = ['benchmarkIsin']
    triplet = construct_gateway_triplet('enhancedfinratrace', 'US', 'BenchmarkBondMapping')
    if not is_today(dates):
        await log.warning("enhancedfinratrace only contains benchmarks as of today!")
    q = build_pt_query(triplet, cols=cols, dates=None, filters={'bondIsin': isins}, by='isin:bondIsin')
    return await query_kdb(q, fconn(GATEWAY))

TERM_BUCKETS = [
    (0.0, 1.75, "1Y"),
    (1.75, 2.5, "2Y"),
    (2.5, 4.0, "3Y"),
    (4.0, 6.0, "5Y"),
    (6.0, 8.5, "7Y"),
    (8.5, 15.0, "10Y"),
    (15.0, 25.0, "20Y"),
    (25.0, 999.0, "30Y"),
]

def assign_bench_names(df: pl.LazyFrame, region="US") -> pl.LazyFrame:
    if region == "US":
        return (
            df.with_columns([
                pl.when(pl.col('benchmarkTerm') == '1.5Y')
                .then(pl.lit('1Y', pl.String))
                .otherwise(pl.col('benchmarkTerm'))
                .alias('benchmarkTerm')
            ]).sort(
                by=["benchmarkTerm", "benchmarkIssueDate"],
                descending=[False, True],
            ).with_columns(
                (pl.cum_count("benchmarkTerm").over("benchmarkTerm") - 1).alias("_rank")
            ).with_columns(
                pl.when(pl.col('_rank') > 3).then(pl.col('benchDescription')).otherwise(
                    pl.concat_str([pl.lit("o").repeat_by("_rank").list.join(""), pl.col("benchmarkTerm")])
                ).alias("benchmarkName"),
                pl.when(pl.col('_rank')==0).then(pl.lit(1, pl.Int8)).otherwise(pl.lit(0, pl.Int8)).alias('isBenchmarkOtr')
            ).drop(["_rank"], strict=False)
        )
    td = get_today(True)
    bucket_expr = pl.lit(None).cast(pl.Utf8)
    yrs = (pl.col("benchmarkMaturityDate").cast(pl.Date) - pl.lit(td)).dt.total_days() / 365.25
    for lo, hi, label in reversed(TERM_BUCKETS):
        bucket_expr = pl.when((yrs >= lo) & (yrs < hi)).then(pl.lit(label)).otherwise(bucket_expr)

    return df.with_columns([
            bucket_expr.alias("__bucket")
        ]).sort("__bucket", "benchmarkIssueDate", descending=[False, True]).with_columns([
            pl.col("benchmarkIssueDate")
            .rank(method="ordinal", descending=True)
            .over("__bucket")
            .cast(pl.UInt32)
            .alias("__rank")
        ]).with_columns([
            pl.when(pl.col("__rank")==1)
            .then(pl.col("__bucket"))
            .when(pl.col("__rank")==2)
            .then(pl.lit("o") + pl.col("__bucket"))
            .when(pl.col("__rank")==3)
            .then(pl.lit("oo") + pl.col("__bucket"))
            .when(pl.col("__rank")==4)
            .then(pl.lit("ooo") + pl.col("__bucket"))
            .otherwise(pl.col("benchDescription"))
            .alias("benchmarkName")
        ]).with_columns([
            pl.when(pl.col('__rank')==1).then(pl.lit(1, pl.Int8)).otherwise(pl.lit(0, pl.Int8)).alias('isBenchmarkOtr')
        ]).drop(["__rank", '__bucket'], strict=False)

async def ust_ref(my_pt=None, region="US", dates=None, **kwargs):
    if region == 'EU':
        isins = my_pt.filter([
            pl.col('benchmarkIsin').is_not_null(),
            pl.col('benchmarkName').is_null(),
            ~pl.col('benchmarkIsin').str.starts_with('US')
        ])
        if isins.hyper.is_empty(): return

        isins = isins.hyper.to_kdb_sym('benchmarkIsin')
        q = 'select benchDescription:first `$TickerSymbol, benchmarkIssueDate:first IssueDate.date, benchmarkTerm:first Sector, benchmarkIsin:first `$ISIN, benchmarkCusip:first CUSIP, benchmarkMaturityDate:first Maturity.date by benchFido:`$Fido from .mt.get[`.rates.eu.fixedIncome.realtime] where date=.z.d, (`$ISIN) in (%s)' % isins
    else:
        q = 'select benchDescription:first `$TickerSymbol, benchmarkIssueDate:first IssueDate.date, benchmarkTerm:first Sector, benchmarkIsin:first `$ISIN, benchmarkCusip:first CUSIP, benchmarkMaturityDate:first Maturity.date by benchFido:`$Fido from .mt.get[`.rates.us.fixedIncome.realtime] where Ticker=`UST'

    res = await query_kdb(q, fconn(PANOPROXY_EU if region == 'EU' else PANOPROXY_US))
    return assign_bench_names(res, region=region)

async def ust_yields(my_pt=None, region="US", dates=None, **kwargs):
    triplet = '.rates.fixedIncomeUSAnalytics.realtime' if is_today(dates, True) else '.rates.fixedIncomeUSAnalytics.historical'
    fidos = my_pt.hyper.to_kdb_sym('benchFido')
    q = "select benchmarkUnitDv01:last dv01, benchmarkRefreshTime:last time, benchmarkMidPx:last twMid*100, benchmarkMidYld:last yield, benchmarkDuration:last (dv01%%twMid) by benchFido:fido from .mt.get[`%s] where fido in (%s)" % (triplet, fidos)
    res = await query_kdb(q, fconn(PANOPROXY_US))
    td = get_today(True)
    return res.with_columns(pl.lit(td,pl.Date).dt.combine(pl.col('benchmarkRefreshTime')).dt.replace_time_zone('utc').alias('benchmarkRefreshTime'))



async def ust_bval_yields(my_pt=None, region="US", dates=None, **kwargs):
    my_pt = ensure_lazy(my_pt)
    bval_isins, rfq_isins = my_pt.hyper.ul(['bvalBenchmarkIsin', 'benchmarkIsin'])
    super_isins = list(set(bval_isins + rfq_isins))
    quotes = await quote_bval_non_dm(pl.LazyFrame({'isin':super_isins}), last_snapshot_only=True, live_only=True)
    quotes = quotes.hyper.ensure_columns(['duration']).with_columns([
        (pl.col('duration').cast(pl.Float64, strict=False) * pl.col('bvalMidPx').cast(pl.Float64, strict=False) * 0.01).alias('_bvalUnitDv01')
    ])
    true_quotes = quotes.select([
        pl.col('isin').alias('bvalBenchmarkIsin'),
        pl.col('bvalMidPx').alias('bvalBenchmarkMidPx'),
        pl.col('bvalMidYld').alias('bvalBenchmarkMidYld'),
    ])
    aligned_quotes = quotes.select([
        pl.col('isin').alias('benchmarkIsin'),
        pl.col('bvalMidPx').alias('bvalAlignedBenchmarkMidPx'),
        pl.col('bvalMidYld').alias('bvalAlignedBenchmarkMidYld'),
        pl.col('_bvalUnitDv01').alias('benchmarkUnitDv01'),
    ])
    return (my_pt.select([
        'isin','bvalBenchmarkIsin', 'benchmarkIsin'
    ]).join(true_quotes, on='bvalBenchmarkIsin', how='left')
     .join(aligned_quotes, on='benchmarkIsin', how='left')
    ).unique(subset=['isin'])

async def non_ust_yields(my_pt=None, region="US", dates=None, **kwargs):
    triplet = construct_gateway_triplet('externalfeeds', "EU", 'twebBenchmark')
    non_us = my_pt.filter(~pl.col('benchmarkIsin').str.starts_with('US'))
    if non_us.hyper.is_empty(): return
    isins = non_us.hyper.ul('benchmarkIsin')
    cols = [
        'benchmarkMidPx:(AskPrice+BidPrice)%2',
        'benchmarkMidYld:(BidYield+AskYield)%2',
        'benchmarkRefreshTime:ReutersSubscriberSendingEpoch'
    ]
    cols = kdb_col_select_helper(cols, "last")
    dates = latest_biz_date(dates, True)
    q = build_pt_query(triplet, cols=cols, dates=dates, filters={'sym': isins}, by='benchmarkIsin:sym')
    return await query_kdb(q, fconn(GATEWAY))


############################################
## Desigs
############################################
# from app.services.loaders.desigs_redux import (
#     _frame_aggregator, apply_rules_to_frame, rank_scored_frame
#     FAST_REQUIRED, FAST_MODIFIERS, FAST_RULES
# )

@hypercache.cached(deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, write_mode="async")
async def test(my_pt):
    syms = my_pt.hyper.to_kdb_sym('isin')
    q = "select from .mt.get[`.credit.refData] where isin in %s" % syms
    return await query_kdb(q, fconn(PANOPROXY))


# quote_bval_full

@hypercache.cached(ttl=timedelta(hours=1), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'region'])
async def desig_fast_path(my_pt, region="US", dates=None, **kwargs):
    from app.services.loaders.desigs_redux import (
        build_fast_rule_expressions,
        desig_frame_enhancer, apply_rules_to_frame, rank_scored_frame
    )
    if (my_pt is None) or (my_pt.hyper.is_empty()): return
    rl = region.lower()
    rt = region.title()
    required_cols = {
        'currency': pl.String,
        'bvalSubAssetClass': pl.String,
        'bvalAssetClass': pl.String,
        'regionBarclaysRegion': pl.String,
        'ratingCombined': pl.String,
        f'_{rl}RunzSenderLastName': pl.List,
        f'house{rt}RefreshTime': pl.Datetime(time_zone="UTC"),
        'isMuni': pl.Int8,
    }
    req_cols = list(required_cols.keys())
    req_frame = my_pt.hyper.ensure_columns(req_cols, dtypes=required_cols)
    FAST_MODIFIERS, FAST_RULES = build_fast_rule_expressions(region=region)
    res = await desig_frame_enhancer(req_frame, region, FAST_MODIFIERS)
    if (res is None) or (res.hyper.is_empty()): return
    return await apply_rules_to_frame(res, FAST_RULES)


async def desig_fast_join(my_pt, region="US", dates=None, frames=None, **kwargs):
    from app.services.loaders.desigs_redux import rank_scored_frame

    my_frames = [
        frames.get('desig_us_scored'),
        frames.get('desig_eu_scored'),
        frames.get('desig_sgp_scored')
    ]
    r = pl.concat([x for x in my_frames if x is not None], how="diagonal_relaxed")

    res = await rank_scored_frame(r)

    _maps = await book_maps()
    return res.join(_maps.with_columns([
        pl.col('bookId').alias('desigBookId'),
    ]), on='desigBookId', how='left')


async def desig_waterfall_portfolio(my_pt, region="US", dates=None, frames=None, **kwargs):
    from app.services.loaders.desigs_redux import apply_waterfall, _WATERFALL_MATCH_COLS, _PROMOTABLE_LABELS
    _UNIVERSE_LABELS = set(['HIGH_CONFIDENCE', 'MEDIUM_CONFIDENCE'] + list(_PROMOTABLE_LABELS))
    _COMPLETE_LABELS = set(['HIGH_CONFIDENCE'] + list(_PROMOTABLE_LABELS))
    my_pt = ensure_lazy(my_pt)
    if (my_pt is None) or (my_pt.hyper.is_empty()): return

    # Read ranked desigs from intermediate frame
    joined = frames.get('desig_joined') if frames else None
    if joined is None or joined.hyper.is_empty(): return

    # Join matching columns from main frame into the desig frame
    match_cols = [c for c in _WATERFALL_MATCH_COLS if c in my_pt.columns]
    if match_cols:
        ref = my_pt.select(['isin'] + match_cols).unique(subset=['isin'])
        joined = joined.join(ref, on='isin', how='left')


    high = joined.filter(
        pl.col('desigConfidence') == 'HIGH_CONFIDENCE'
    )
    universe = joined.filter(
        pl.col('desigConfidence').is_in(_UNIVERSE_LABELS)
    )
    basket = joined.filter(
        ~pl.col('desigConfidence').is_in(['HIGH_CONFIDENCE'])
    )

    # If nothing to waterfall, return all joined results as-is
    if basket.hyper.is_empty(): return joined
    result = await apply_waterfall(basket, universe)

    # Combine: unchanged HIGH bonds + waterfall-processed bonds
    if result is not None and not result.hyper.is_empty():
        return pl.concat([high, result], how='diagonal_relaxed')

    # Waterfall failed or empty - return original joined results
    return joined


_DESIG_SPLITTER_HIGH_COLS = [
    'isin', 'desigBookId', 'desigTraderId', 'desigName', 'desigRegion',
    'desigConfidence', 'desigGapRatio', 'desigScore', 'deskAsset',
]


def _is_empty_frame(df) -> bool:
    """True when a frame is None, empty, or has no columns at all.

    The loader occasionally hands splitters a zero-column DataFrame when
    an upstream task returned None (e.g., pano_positions timed out and
    desig_waterfall_portfolio short-circuited). `.filter(pl.col(...))`
    on such a frame raises ColumnNotFoundError with 'valid columns: []'
    -- guarding up-front is the only safe thing to do.
    """
    if df is None:
        return True
    try:
        if df.hyper.is_empty():
            return True
    except Exception:
        pass
    try:
        # Zero-column frames are the specific pathology we're guarding.
        if not df.hyper.fields:
            return True
    except Exception:
        # If we can't introspect, assume non-empty and let downstream fail.
        return False
    return False


async def desig_splitter_high(my_pt, region="US", dates=None, frames=None, **kwargs):
    from app.services.loaders.desigs_redux import _PROMOTABLE_LABELS
    _COMPLETE_LABELS = set(['HIGH_CONFIDENCE'] + list(_PROMOTABLE_LABELS))
    if _is_empty_frame(my_pt):
        return None
    if 'desigConfidence' not in my_pt.hyper.fields:
        return None
    return my_pt.filter(pl.col('desigConfidence').is_in(_COMPLETE_LABELS)).select(
        [c for c in _DESIG_SPLITTER_HIGH_COLS if c in my_pt.hyper.fields]
    )


async def desig_splitter_low(my_pt, region="US", dates=None, frames=None, **kwargs):
    from app.services.loaders.desigs_redux import _PROMOTABLE_LABELS
    _COMPLETE_LABELS = set(['HIGH_CONFIDENCE'] + list(_PROMOTABLE_LABELS))
    if _is_empty_frame(my_pt):
        return None
    if 'desigConfidence' not in my_pt.hyper.fields:
        return None
    return my_pt.filter(~pl.col('desigConfidence').is_in(_COMPLETE_LABELS))

async def desig_expander(my_pt, region="US", dates=None, frames=None, **kwargs):
    """Re-export of the orchestrator from `desig_expansion.py`. The
    previous in-line stub here built a per-ticker query but never
    returned the result frame -- it was effectively dead code. The new
    implementation runs a second `apply_waterfall` round against a
    firm-wide PANOPROXY universe (cached 12h) and resolves each
    trader's main book via `book_maps`. See `desig_expansion.py` for
    the full design rationale.
    """
    from app.services.loaders.desig_expansion import desig_expander as _impl
    return await _impl(my_pt, region=region, dates=dates, frames=frames, **kwargs)


async def desig_expanded_splitter(my_pt, region="US", dates=None, frames=None, **kwargs):
    """Merge expansion-derived HIGH/P1/P2 desigs onto main, never
    overwriting an ISIN already resolved by the portfolio round.
    See `desig_expansion.py`.
    """
    from app.services.loaders.desig_expansion import (
        desig_expanded_splitter as _impl,
    )
    return await _impl(my_pt, region=region, dates=dates, frames=frames, **kwargs)




@hypercache.cached(ttl=timedelta(hours=12), deep={"my_pt": True}, primary_keys={'my_pt': ["isin"]}, key_params=['my_pt', 'bleed_liq_sources'])
async def cz_liq_score(my_pt, region="US", dates=None, bleed_liq_sources=False, **kwargs):
    from datetime import datetime
    lf = ensure_lazy(my_pt)

    TRACE_W1 = 0.55
    TRACE_W10 = 0.25
    TRACE_W30 = 0.12
    TRACE_W60 = 0.05
    TRACE_W90 = 0.03
    TRACE_VOL_BONUS_MAX = 0.8
    TRACE_VOL_SCALE = 200.0
    MOMENTUM_SCALE = 0.35
    EPISODIC_PEN_SCALE = 0.40

    BO_TIGHT_BPS = 10.0
    BO_UNLOCK_BPS = 20.0
    BO_NOISY_BPS = 30.0
    WEDGE_PEN_SCALE = 0.20
    CONSISTENCY_PEN_SCALE = 0.10

    CLASS_MULT_HIGH_LIQ = 1.15
    CLASS_MULT_IG = 1.08
    CLASS_MULT_HY = 0.97
    CLASS_MULT_MUNI = 1.01
    CLASS_MULT_DISTRESS = 0.82
    CLASS_MULT_DEFAULTED = 0.55

    EM_CLASS_SHORT = 1.02
    EM_CLASS_MID = 0.95
    EM_CLASS_LONG = 0.88
    EM_CLASS_ULTRA = 0.78

    MAT_ANCHOR_3Y = 1.08
    MAT_ANCHOR_7Y = 1.02
    MAT_ANCHOR_15Y = 0.95
    MAT_ANCHOR_30Y = 0.85
    MAT_ANCHOR_50Y = 0.72

    DUR_BO_NORM_REF = 5.0

    QUOTE_QUALITY_DISCOUNT_MAX = -2.0
    QUOTE_QUALITY_YTM_THRESH = 20.0

    EM_BENCH_BOOST_MAX = 1.5
    EM_BENCH_SIZE_THRESH = 1_000_000_000.0
    EM_BENCH_YTM_THRESH = 7.0

    ETF_AGG_ADJ = 0.15
    ETF_LQD_ADJ = 0.20
    ETF_HYG_ADJ = 0.15
    ETF_EMB_ADJ = 0.10
    ETF_OTHER_ADJ = 0.08
    ETF_MULTI_BONUS = 0.10

    NEW_ISSUE_ADJ = 0.20
    DTC_ADJ = 0.30
    CCY_MAJOR_ADJ = 0.20

    P144A_US = 0.10
    P144A_NONUS = 0.05
    REGS_US = -0.20
    REGS_NONUS = -0.10

    FLOAT_STRENGTH = 1.0

    POS_BASE_BOOST = 0.10
    POS_RATIO_SLOPE = 0.60
    POS_RATIO_CAP = 0.25

    ALGO_IG = 0.6
    ALGO_HY = 0.2
    ALGO_EM = 0.15
    ALLQ_ADJ = 0.4

    COVERAGE_WEIGHT = 0.6
    BVAL_MISSING_PEN = -0.2

    CROSS_CCY_US_PEN = -0.2

    SIZE_REF_US = 1_000_000_000.0
    SIZE_REF_EXUS = 500_000_000.0
    SIZE_REF_MUNI = 50_000_000.0

    GLOBAL_SHIFT = -0.25
    TAIL_DAMP_PIVOT = 6.0
    TAIL_DAMP = 0.75

    VENDOR_CAP = 2.0
    SMAD_DELTA_SCALE = 0.30
    SMAD_DELTA_CAP = 0.8

    STALENESS_PEN_MAX = -1.5
    STALENESS_DECAY_DAYS = 14.0

    CDS_BASIS_BONUS = 0.25
    CDS_WIDE_PEN = -0.15

    SEASONING_PEAK_YRS = 0.5
    SEASONING_FADE_YRS = 5.0
    SEASONING_PEN_MAX = -0.30

    SECTOR_LIQ_MAP = {
        "FINANCIALS": 0.15,
        "UTILITIES": 0.10,
        "COMMUNICATION SERVICES": 0.05,
        "CONSUMER STAPLES": 0.05,
        "HEALTH CARE": 0.00,
        "INDUSTRIALS": 0.00,
        "INFORMATION TECHNOLOGY": -0.05,
        "CONSUMER DISCRETIONARY": -0.05,
        "MATERIALS": -0.10,
        "ENERGY": -0.10,
        "REAL ESTATE": -0.15,
    }

    liq_sources = [
        "macpLiqScore", "blsLiqScore", "dkLiqScore", "muniLiqScore",
        "lqaLiqScore", "smadLiqScore", "mlcrLiqScore", 'idcLiqScore'
    ]

    ref_mids = [
        "bvalMidPx", "macpMidPx", "cbbtMidPx", "idcMidPx", "houseMidPx", "allqMidPx",
    ]

    cols_primary = ["isin"]

    cols_trace = [
        "traceCount1D", "traceCount10D", "traceCount30D", "traceCount60D", "traceCount90D",
        "traceVolume1D", "traceVolume10D", "traceVolume30D", "traceVolume60D", "traceVolume90D",
    ]

    cols_quote = [
        "houseBidPx", "houseAskPx", "houseBidSpd", "houseAskSpd",
        "cbbtBidPx", "cbbtAskPx", "cbbtBidSpd", "cbbtAskSpd",
    ]

    cols_etf = [
        "inEtfAgg", "inEtfLqd", "inEtfHyg", "inEtfJnk", "inEtfEmb",
        "inEtfSpsb", "inEtfSpib", "inEtfSplb", "inEtfVclt", "inEtfSpab",
        "inEtfIgib", "inEtfIglb", "inEtfIgsb", "inEtfIemb",
        "inEtfUsig", "inEtfUshy", "inEtfSjnk",
    ]

    cols_new = [
        "gicsSector", "gicsSubIndustry", "gicsIndustryGroup",
        "issuerSector", "issuerSubIndustry", "issuerIndustry",
        "yrsToMaturity", "maturityDate", "duration", "yrsSinceIssuance",
        "houseUsRefreshTime", "houseEuRefreshTime", "houseSgpRefreshTime",
        "cdsBasisToWorst", "cdsParSpdW",
    ]

    cols_extra = [
        "allqMidPx", "amountIssued", "amountOutstanding", "bvalSubAssetClass", "currency",
        "desigRegion", "deskAsset", "isCalled", "isConvertible", "isDtcEligible", "isEmAlgoEligible",
        "isFloater", "isHyAlgoEligible", "isIgAlgoEligible", "isNewIssue", "isPerpetual", "isRegS",
        "isRule144A", "netFirmPosition", "ratingCombined"
    ] + ref_mids

    all_cols = cols_primary + cols_trace + cols_quote + cols_extra + cols_etf + cols_new
    seen = set()
    all_cols_deduped = []
    for c in all_cols:
        if c not in seen:
            seen.add(c)
            all_cols_deduped.append(c)

    lf = lf.hyper.ensure_columns(all_cols_deduped)
    lf = lf.fill_nan(None)

    def _bool_col(name):
        return pl.col(name).cast(pl.Int8, strict=False).cast(pl.Boolean, strict=False).fill_null(False)

    def _f64_col(name, default=0.0):
        return pl.col(name).cast(pl.Float64, strict=False).fill_null(default)

    sub_u = pl.col("bvalSubAssetClass").cast(pl.Utf8, strict=False).str.to_uppercase()
    ast_u = pl.col("deskAsset").cast(pl.Utf8, strict=False).str.to_uppercase()
    rat_u = pl.col("ratingCombined").cast(pl.Utf8, strict=False).str.to_uppercase()
    ccy_u = pl.col("currency").cast(pl.Utf8, strict=False).str.to_uppercase()
    region_u = pl.coalesce([
        pl.col("desigRegion").cast(pl.Utf8, strict=False).str.to_uppercase(),
        pl.when(ccy_u == "USD").then(pl.lit("US"))
        .when(ccy_u.is_in(["EUR", "GBP", "CHF", "SEK", "NOK", "DKK"])).then(pl.lit("EU"))
        .otherwise(None)
    ])
    sector_u = pl.col("gicsSector").cast(pl.Utf8, strict=False).str.to_uppercase()
    is_high_liq_early = sub_u.is_in(["AGENCY", "SSA", "UST", "IRS"])

    tc1 = _f64_col("traceCount1D")
    tc10 = _f64_col("traceCount10D") / 10.0
    tc30 = _f64_col("traceCount30D") / 30.0
    tc60 = _f64_col("traceCount60D") / 60.0
    tc90 = _f64_col("traceCount90D") / 90.0

    def _piecewise_trace(tc):
        """0→1, 1-3→linear to 5, 3-15→linear to 8, 15+→log tail to 10."""
        return (
            pl.when(tc <= 0).then(1.0)
            .when(tc <= 3).then(1.0 + tc * (4.0 / 3.0))
            .when(tc <= 15).then(5.0 + (tc - 3.0) * (3.0 / 12.0))
            .otherwise(8.0 + ((tc - 15.0 + 1).log() / pl.lit(86.0).log()) * 2.0)
        ).clip(1.0, 10.0)

    trace_score = (
            _piecewise_trace(tc1) * TRACE_W1 +
            _piecewise_trace(tc10) * TRACE_W10 +
            _piecewise_trace(tc30) * TRACE_W30 +
            _piecewise_trace(tc60) * TRACE_W60 +
            _piecewise_trace(tc90) * TRACE_W90
    ).clip(1.0, 10.0)

    tc_recent = tc1 * 0.6 + tc10 * 0.4
    tc_older = tc30 * 0.4 + tc60 * 0.35 + tc90 * 0.25
    trade_momentum = (tc_recent - tc_older) / (tc_older + 0.1)
    momentum_adj = trade_momentum.clip(-1.5, 1.5) * MOMENTUM_SCALE

    size_out = _f64_col("amountOutstanding")
    v1 = _f64_col("traceVolume1D")
    v10 = _f64_col("traceVolume10D") / 10.0
    v30 = _f64_col("traceVolume30D") / 30.0
    v60 = _f64_col("traceVolume60D") / 60.0
    v90 = _f64_col("traceVolume90D") / 90.0

    vol_daily = (v1 * TRACE_W1 + v10 * TRACE_W10 + v30 * TRACE_W30 + v60 * TRACE_W60 + v90 * TRACE_W90)
    vol_ratio = vol_daily / (size_out + 1e-9)
    vol_bonus = (
            ((vol_ratio * TRACE_VOL_SCALE) + 1).log()
            / (pl.lit(TRACE_VOL_SCALE) + 1).log()
            * TRACE_VOL_BONUS_MAX
    ).fill_null(0.0)

    vol_total = vol_daily + 1e-9
    vol_shares = [v1 / vol_total, v10 / vol_total, v30 / vol_total, v60 / vol_total, v90 / vol_total]
    vol_hhi = pl.sum_horizontal([s.pow(2) for s in vol_shares])
    episodic_pen = -((vol_hhi - 0.2).clip(0.0, 1.0)) * EPISODIC_PEN_SCALE

    trace_score_plus = (trace_score + vol_bonus + momentum_adj + episodic_pen).clip(1.0, 10.0)

    has_trace = pl.any_horizontal([
        _f64_col("traceCount1D") > 0,
        _f64_col("traceCount10D") > 0,
        _f64_col("traceCount30D") > 0,
        _f64_col("traceCount60D") > 0,
        _f64_col("traceCount90D") > 0,
    ])

    dur = _f64_col("duration", default=5.0).clip(0.5, 30.0)
    dur_scale = DUR_BO_NORM_REF / (dur + 1e-9)

    house_mid = (pl.col("houseBidPx") + pl.col("houseAskPx")) / 2
    cbbt_mid = (pl.col("cbbtBidPx") + pl.col("cbbtAskPx")) / 2

    house_w_px_bps = (
            (pl.col("houseAskPx") - pl.col("houseBidPx")) / (house_mid.abs() + 1e-9) * 10000
    ).cast(pl.Float64, strict=False)
    cbbt_w_px_bps = (
            (pl.col("cbbtAskPx") - pl.col("cbbtBidPx")) / (cbbt_mid.abs() + 1e-9) * 10000
    ).cast(pl.Float64, strict=False)

    # Validate house quotes: zero spread or large deviation from consensus = suspect/stale
    independent_mids = ["bvalMidPx", "macpMidPx", "cbbtMidPx", "idcMidPx"]
    independent_consensus = pl.mean_horizontal(*[pl.col(c) for c in independent_mids])
    house_consensus_dev_pct = (
        (house_mid - independent_consensus).abs() / (independent_consensus.abs() + 1e-9)
    ).fill_null(0.0)
    house_spread_zero = (
        pl.col("houseBidPx").is_not_null()
        & pl.col("houseAskPx").is_not_null()
        & (pl.col("houseBidPx") == pl.col("houseAskPx"))
    )
    house_suspect = (
        (house_spread_zero & ~is_high_liq_early)
        | (house_consensus_dev_pct > 0.05)
    )
    house_w_px_validated = pl.when(house_suspect).then(pl.lit(None, pl.Float64)).otherwise(house_w_px_bps)

    px_w_bps_min = pl.min_horizontal(house_w_px_validated, cbbt_w_px_bps)

    house_w_spd_bps = (pl.col("houseAskSpd") - pl.col("houseBidSpd")).cast(pl.Float64, strict=False)
    cbbt_w_spd_bps = (pl.col("cbbtAskSpd") - pl.col("cbbtBidSpd")).cast(pl.Float64, strict=False)
    spd_w_bps_min = pl.min_horizontal(house_w_spd_bps, cbbt_w_spd_bps)

    eff_w_bps = pl.coalesce([
        pl.min_horizontal(px_w_bps_min, spd_w_bps_min),
        px_w_bps_min, spd_w_bps_min, pl.lit(None, pl.Float64),
    ])

    eff_w_dur_norm = eff_w_bps * dur_scale

    NO_QUOTE_DEFAULT = 3.0
    bo_score = pl.when(eff_w_bps.is_null()).then(NO_QUOTE_DEFAULT).otherwise(
        (10.0 / (1.0 + (-1 * eff_w_dur_norm / 25.0).pow(1.2))).clip(1.0, 10.0)
    ).fill_null(NO_QUOTE_DEFAULT)

    has_px = px_w_bps_min.is_not_null()
    has_spd = spd_w_bps_min.is_not_null()
    w_px = has_px.cast(pl.Float64) * 0.55
    w_spd = has_spd.cast(pl.Float64) * 0.45
    w_sum = w_px + w_spd + 1e-9

    wedge_px = (house_w_px_validated - cbbt_w_px_bps)
    wedge_spd = (house_w_spd_bps - cbbt_w_spd_bps)
    wedge_pen_px = -((wedge_px.clip(0.0, None) + 1).log() * WEDGE_PEN_SCALE).fill_null(0.0)
    wedge_pen_spd = -((wedge_spd.clip(0.0, None) + 1).log() * WEDGE_PEN_SCALE).fill_null(0.0)
    wedge_pen = ((wedge_pen_px * w_px + wedge_pen_spd * w_spd) / w_sum).fill_null(0.0)

    both_widths = has_px & has_spd
    consistency_pen = (
        pl.when(both_widths)
        .then(-((px_w_bps_min - spd_w_bps_min).abs() + 1).log() * CONSISTENCY_PEN_SCALE)
        .otherwise(0.0)
    )

    mid_cols_present = [pl.col(c) for c in ref_mids]
    mid_consensus = pl.mean_horizontal(*mid_cols_present)
    house_dev = ((house_mid - mid_consensus) / (mid_consensus.abs() + 1e-9)).abs()
    cbbt_dev = ((cbbt_mid - mid_consensus) / (mid_consensus.abs() + 1e-9)).abs()
    quote_deviation_pen = -(pl.max_horizontal(house_dev, cbbt_dev) * 10).clip(0.0, 1.5)

    now_lit = pl.lit(datetime.utcnow())
    refresh_cols = ["houseUsRefreshTime", "houseEuRefreshTime", "houseSgpRefreshTime"]
    most_recent_refresh = pl.max_horizontal([
        pl.col(c).cast(pl.Datetime, strict=False) for c in refresh_cols
    ])
    days_since_refresh = (
            (now_lit - most_recent_refresh).dt.total_seconds() / 86400.0
    ).cast(pl.Float64, strict=False).fill_null(STALENESS_DECAY_DAYS)

    refresh_staleness_pen = (
            STALENESS_PEN_MAX * (1.0 - (-days_since_refresh / (STALENESS_DECAY_DAYS / 3.0)).exp())
    ).clip(STALENESS_PEN_MAX, 0.0)

    staleness_total = (quote_deviation_pen + refresh_staleness_pen).fill_null(0)

    ytm = _f64_col("yrsToMaturity", default=5.0)
    is_dtc = _bool_col("isDtcEligible")

    has_house_quote = pl.col("houseBidPx").is_not_null() & pl.col("houseAskPx").is_not_null()
    has_cbbt_quote = pl.col("cbbtBidPx").is_not_null() & pl.col("cbbtAskPx").is_not_null()
    cbbt_only = has_cbbt_quote & ~has_house_quote

    is_em_flag_early = (sub_u=="EM") | (ast_u=="EM")

    maturity_ramp = ((ytm - 15.0) / 35.0).clip(0.0, 1.0)

    quote_quality_discount = (
        pl.when(~has_trace & is_em_flag_early & (ytm > QUOTE_QUALITY_YTM_THRESH))
        .then(
            QUOTE_QUALITY_DISCOUNT_MAX
            * maturity_ramp
            * pl.when(cbbt_only & ~is_dtc).then(1.0)
            .when(cbbt_only).then(0.7)
            .when(~is_dtc).then(0.5)
            .otherwise(0.3)
        )
        .otherwise(0.0)
    )

    allq_adj = pl.when(pl.col("allqMidPx").is_not_null()).then(ALLQ_ADJ).otherwise(0.0)

    ig_el = _bool_col("isIgAlgoEligible")
    hy_el = _bool_col("isHyAlgoEligible")
    em_el = _bool_col("isEmAlgoEligible")
    algo_adj = ig_el.cast(pl.Int8) * ALGO_IG + hy_el.cast(pl.Int8) * ALGO_HY + em_el.cast(pl.Int8) * ALGO_EM

    dtc_adj = pl.when(is_dtc).then(DTC_ADJ).otherwise(0.0)
    ccy_adj = pl.when(ccy_u.is_in(["USD", "EUR", "GBP"])).then(CCY_MAJOR_ADJ).otherwise(0.0)

    coverage_count = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int8) for c in ref_mids])
    coverage_adj = (
            ((coverage_count + 1).log() / pl.lit(len(ref_mids) + 1).log()) * COVERAGE_WEIGHT
    ).fill_null(0.0)
    bval_missing_pen = pl.when(pl.col("bvalMidPx").is_null()).then(BVAL_MISSING_PEN).otherwise(0.0)

    cross_ccy_pen = pl.when((region_u=="US") & (ccy_u!="USD")).then(CROSS_CCY_US_PEN).otherwise(0.0)

    high_liq_complex = is_high_liq_early
    is_ig_flag = (sub_u=="IG") | (ast_u=="IG")
    is_hy_flag = (sub_u=="HY") | (ast_u=="HY")
    is_muni_flag = (sub_u=="MUNI") | (ast_u=="MUNI")
    is_em_flag = (sub_u=="EM") | (ast_u=="EM")

    is_defaulted = rat_u.is_in(["D", "SD"])
    is_distressed = rat_u.is_in(["CCC", "CCC+", "CCC-", "CC", "C", "NR"]) & ~is_defaulted

    mat_date = pl.col("maturityDate").cast(pl.Date, strict=False)
    today = pl.lit(datetime.utcnow().date())
    is_past_maturity = mat_date.is_not_null() & (mat_date < today)
    is_dead_bond = is_past_maturity | is_defaulted

    em_class_mult = (
        pl.when(ytm <= 5.0).then(EM_CLASS_SHORT)
        .when(ytm <= 15.0).then(EM_CLASS_MID)
        .when(ytm <= 30.0).then(EM_CLASS_LONG)
        .otherwise(EM_CLASS_ULTRA)
    )

    class_mult = (
        pl.when(is_dead_bond).then(CLASS_MULT_DEFAULTED)
        .when(is_distressed).then(CLASS_MULT_DISTRESS)
        .when(high_liq_complex).then(CLASS_MULT_HIGH_LIQ)
        .when(is_ig_flag).then(CLASS_MULT_IG)
        .when(is_hy_flag).then(CLASS_MULT_HY)
        .when(is_em_flag).then(em_class_mult)
        .when(is_muni_flag).then(CLASS_MULT_MUNI)
        .otherwise(1.0)
    )

    mat_mult = (
        pl.when(ytm <= 3.0).then(MAT_ANCHOR_3Y)
        .when(ytm <= 7.0).then(
            MAT_ANCHOR_3Y + (ytm - 3.0) / (7.0 - 3.0) * (MAT_ANCHOR_7Y - MAT_ANCHOR_3Y)
        )
        .when(ytm <= 15.0).then(
            MAT_ANCHOR_7Y + (ytm - 7.0) / (15.0 - 7.0) * (MAT_ANCHOR_15Y - MAT_ANCHOR_7Y)
        )
        .when(ytm <= 30.0).then(
            MAT_ANCHOR_15Y + (ytm - 15.0) / (30.0 - 15.0) * (MAT_ANCHOR_30Y - MAT_ANCHOR_15Y)
        )
        .when(ytm <= 50.0).then(
            MAT_ANCHOR_30Y + (ytm - 30.0) / (50.0 - 30.0) * (MAT_ANCHOR_50Y - MAT_ANCHOR_30Y)
        )
        .otherwise(MAT_ANCHOR_50Y)
    )

    is_sovereign_like = sub_u.is_in(["EM", "SOVEREIGN", "GOVT"]) | ast_u.is_in(["EM", "SOVEREIGN", "GOVT"])
    size_val_bench = pl.coalesce([
        pl.col("amountOutstanding"), pl.col("amountIssued"),
    ]).cast(pl.Float64, strict=False).fill_null(0.0)

    bench_signals = pl.sum_horizontal([
        (is_em_flag & (ytm <= EM_BENCH_YTM_THRESH)).cast(pl.Int8),
        (size_val_bench >= EM_BENCH_SIZE_THRESH).cast(pl.Int8),
        is_sovereign_like.cast(pl.Int8),
        ccy_u.is_in(["USD", "EUR"]).cast(pl.Int8),
        (_bool_col("inEtfEmb") | _bool_col("inEtfIemb")).cast(pl.Int8),
    ])

    em_bench_boost_frac = (
        pl.when(bench_signals <= 1).then(0.0)
        .when(bench_signals==2).then(0.5)
        .when(bench_signals==3).then(0.8)
        .otherwise(1.0)
    )
    em_bench_mat_scale = ((EM_BENCH_YTM_THRESH - ytm) / EM_BENCH_YTM_THRESH).clip(0.0, 1.0)

    em_bench_boost = (EM_BENCH_BOOST_MAX * em_bench_boost_frac * em_bench_mat_scale).fill_null(0.0)

    def _etf_flag(name):
        return _bool_col(name).cast(pl.Int8)

    etf_score = (
            _etf_flag("inEtfAgg") * ETF_AGG_ADJ +
            _etf_flag("inEtfLqd") * ETF_LQD_ADJ +
            _etf_flag("inEtfHyg") * ETF_HYG_ADJ +
            _etf_flag("inEtfJnk") * ETF_HYG_ADJ +
            _etf_flag("inEtfEmb") * ETF_EMB_ADJ +
            _etf_flag("inEtfIemb") * ETF_EMB_ADJ +
            _etf_flag("inEtfSpsb") * ETF_OTHER_ADJ +
            _etf_flag("inEtfSpib") * ETF_OTHER_ADJ +
            _etf_flag("inEtfSplb") * ETF_OTHER_ADJ +
            _etf_flag("inEtfVclt") * ETF_OTHER_ADJ +
            _etf_flag("inEtfSpab") * ETF_OTHER_ADJ +
            _etf_flag("inEtfIgib") * ETF_OTHER_ADJ +
            _etf_flag("inEtfIglb") * ETF_OTHER_ADJ +
            _etf_flag("inEtfIgsb") * ETF_OTHER_ADJ +
            _etf_flag("inEtfUsig") * ETF_OTHER_ADJ +
            _etf_flag("inEtfUshy") * ETF_OTHER_ADJ +
            _etf_flag("inEtfSjnk") * ETF_OTHER_ADJ
    )
    etf_membership_count = pl.sum_horizontal([_etf_flag(c) for c in cols_etf])
    etf_multi_bonus = pl.when(etf_membership_count >= 3).then(ETF_MULTI_BONUS).otherwise(0.0)
    etf_adj_total = (etf_score + etf_multi_bonus).clip(0.0, 1.0)

    effective_sector = pl.coalesce([
        pl.col("gicsSector").cast(pl.Utf8, strict=False).str.to_uppercase(),
        pl.col("issuerSector").cast(pl.Utf8, strict=False).str.to_uppercase(),
    ])

    sector_adj = pl.lit(0.0)
    for sec_name, sec_val in SECTOR_LIQ_MAP.items():
        sector_adj = pl.when(effective_sector==sec_name).then(sec_val).otherwise(sector_adj)

    cds_basis = _f64_col("cdsBasisToWorst", default=0.0)
    cds_par = _f64_col("cdsParSpdW", default=0.0)
    has_cds = (pl.col("cdsParSpdW").is_not_null()) & (cds_par.abs() > 0)

    cds_adj = (
        pl.when(~has_cds).then(0.0)
        .when(cds_basis > -20).then(CDS_BASIS_BONUS)
        .when(cds_basis > -80).then(0.0)
        .otherwise(CDS_WIDE_PEN)
    )

    yrs_since = _f64_col("yrsSinceIssuance", default=1.0).clip(0.0, 30.0)

    seasoning_adj = (
        pl.when(yrs_since <= SEASONING_PEAK_YRS).then(NEW_ISSUE_ADJ * 0.5)
        .when(yrs_since <= SEASONING_FADE_YRS).then(
            SEASONING_PEN_MAX * ((yrs_since - SEASONING_PEAK_YRS) / (SEASONING_FADE_YRS - SEASONING_PEAK_YRS))
        )
        .otherwise(SEASONING_PEN_MAX)
    )

    size_ref = pl.when(region_u=="US").then(SIZE_REF_US).otherwise(SIZE_REF_EXUS)
    muni_ref = pl.when(is_muni_flag).then(SIZE_REF_MUNI).otherwise(size_ref)
    size_val = pl.coalesce([
        pl.col("amountOutstanding"), pl.col("amountIssued"),
    ]).cast(pl.Float64, strict=False).fill_null(0.0)
    size_adj = (((size_val + 1).log() / (muni_ref + 1).log()) * 2 - 1).clip(-1.0, 1.0)

    size_issue = _f64_col("amountIssued")
    float_ratio = (size_out / (size_issue + 1e-9)).clip(0.0, 1.0)
    float_adj = ((float_ratio - 0.5) * 2).clip(-1.0, 1.0) * FLOAT_STRENGTH

    is_perp = _bool_col("isPerpetual")
    is_float = _bool_col("isFloater")
    is_conv = _bool_col("isConvertible")
    is_called = _bool_col("isCalled")
    struct_adj = (
            pl.when(is_conv).then(-0.8).otherwise(0.0) +
            pl.when(is_perp).then(-0.4).otherwise(0.0) +
            pl.when(is_float).then(0.2).otherwise(0.0) +
            pl.when(is_called).then(-0.2).otherwise(0.0)
    )

    p144a_adj = pl.when(_bool_col("isRule144A")).then(
        pl.when(region_u=="US").then(P144A_US).otherwise(P144A_NONUS)
    ).otherwise(0.0)
    regS_adj = pl.when(_bool_col("isRegS")).then(
        pl.when(region_u=="US").then(REGS_US).otherwise(REGS_NONUS)
    ).otherwise(0.0)
    rules_adj = p144a_adj + regS_adj

    new_issue_adj = pl.when(_bool_col("isNewIssue")).then(NEW_ISSUE_ADJ).otherwise(0.0)

    pos = _f64_col("netFirmPosition")
    pos_ratio = (pos.abs() / (size_out + 1e-9)).fill_null(0.0)
    pos_adj = (
            pl.when(pos.abs() > 0).then(POS_BASE_BOOST).otherwise(0.0)
            - (pos_ratio.clip(0.0, POS_RATIO_CAP) * POS_RATIO_SLOPE)
    )

    miss_cols = ["traceCount1D", "traceCount10D", "deskAsset", "bvalSubAssetClass", "currency"]
    miss_pen = -(pl.sum_horizontal([pl.col(c).is_null().cast(pl.Int8) for c in miss_cols]) * 0.15).clip(-0.6, 0.0)

    macp = pl.col("macpLiqScore") if bleed_liq_sources else pl.lit(None, pl.Float64)
    macp_adj = (
        pl.when(macp.is_null()).then(0.0)
        .when(macp >= 9.0).then(1.00)
        .when(macp >= 8.0).then(0.70)
        .when(macp >= 7.0).then(0.35)
        .when(macp >= 4.0).then(0.00)
        .when(macp >= 2.5).then(-0.50)
        .otherwise(-1.00)
    )

    lqa = pl.col("lqaLiqScore") if bleed_liq_sources else pl.lit(None, pl.Float64)
    lqa_adj = (
        pl.when(lqa.is_null()).then(0.0)
        .when(lqa >= 95).then(0.50)
        .when(lqa >= 80).then(0.35)
        .when(lqa >= 40).then(0.00)
        .when(lqa >= 25).then(-0.35)
        .otherwise(-0.70)
    )

    hi_count = pl.sum_horizontal([(macp >= 8.0).cast(pl.Int8), (lqa >= 80).cast(pl.Int8)])
    lo_count = pl.sum_horizontal([(macp <= 3.5).cast(pl.Int8), (lqa <= 35).cast(pl.Int8)])
    vendor_consensus = pl.when(hi_count >= 2).then(0.25).when(lo_count >= 2).then(-0.25).otherwise(0.0)

    smad = pl.col("smadLiqScore") if bleed_liq_sources else pl.lit(None, pl.Float64)
    vendor_base_now = pl.coalesce([macp, lqa / 10])
    smad_delta = smad - vendor_base_now
    smad_adj = (
        pl.when(smad.is_not_null() & vendor_base_now.is_not_null())
        .then(smad_delta * SMAD_DELTA_SCALE)
        .otherwise(0.0)
    ).clip(-SMAD_DELTA_CAP, SMAD_DELTA_CAP)

    vendor_delta_raw = (macp_adj + lqa_adj + vendor_consensus + smad_adj).clip(-VENDOR_CAP, VENDOR_CAP)

    trace_recent = has_trace
    gate_unlock = 1.0 / (1.0 + ((eff_w_bps - BO_UNLOCK_BPS) / 5.0).exp())
    gate_algo = (pl.col("allqMidPx").is_not_null() & (ig_el | hy_el | em_el)).cast(pl.Float64)
    gate_trace = trace_recent.cast(pl.Float64)

    vendor_gate = (pl.max_horizontal(gate_unlock, gate_algo, gate_trace)).clip(0.2, 1.0)

    vendor_up_damper = pl.when(
        (vendor_delta_raw > 0) & (~trace_recent) & (eff_w_bps > BO_NOISY_BPS)
    ).then(0.25).otherwise(1.0)

    street_tighter = (
            cbbt_w_spd_bps.is_not_null()
            & house_w_spd_bps.is_not_null()
            & ((cbbt_w_spd_bps + 20) <= house_w_spd_bps)
    )
    street_tight_damper = pl.when(street_tighter & (vendor_delta_raw < 0)).then(0.5).otherwise(1.0)

    vendor_adj = vendor_delta_raw * vendor_gate * vendor_up_damper * street_tight_damper

    flow_proxy = (
            bo_score + wedge_pen + consistency_pen +
            algo_adj + allq_adj + dtc_adj + ccy_adj +
            coverage_adj + bval_missing_pen + cross_ccy_pen +
            quote_quality_discount
    ).clip(1.0, 10.0)

    # When house quote is suspect and no CBBT, there's no real quote data
    has_any_valid_quote = (
        (~house_suspect & has_px) | has_spd
        | pl.col("cbbtBidPx").is_not_null()
    )
    NO_DATA_CAP = 5.0
    core_liq = pl.when(has_trace).then(
        (trace_score_plus * 0.55 + bo_score * 0.45).clip(1.0, 10.0)
    ).when(has_any_valid_quote).then(
        (flow_proxy * 0.90 + size_adj * 0.10).clip(1.0, 10.0)
    ).otherwise(
        # No trade data AND no valid quotes - heavily weight size, cap low
        (flow_proxy * 0.40 + size_adj * 0.60).clip(1.0, NO_DATA_CAP)
    )

    # Small issue structural penalty - issues under 100M are structurally illiquid
    SMALL_ISSUE_THRESH = 100_000_000.0
    SMALL_ISSUE_PEN_MAX = -1.0
    small_issue_pen = pl.when(size_val < SMALL_ISSUE_THRESH).then(
        SMALL_ISSUE_PEN_MAX * (1.0 - size_val / SMALL_ISSUE_THRESH)
    ).otherwise(0.0)

    additive_adj = (
        size_adj
        + float_adj
        + etf_adj_total
        + new_issue_adj
        + rules_adj
        + struct_adj
        + pos_adj
        + miss_pen
        + vendor_adj
        + staleness_total
        + sector_adj
        + cds_adj
        + seasoning_adj
        + em_bench_boost
        + small_issue_pen
        + GLOBAL_SHIFT
    ).fill_null(0.0)

    inferred_raw = (core_liq * class_mult.fill_null(1.0) * mat_mult.fill_null(1.0)) + additive_adj

    inferred_capped = pl.when(is_dead_bond).then(
        inferred_raw.clip(1.0, 2.0)
    ).otherwise(inferred_raw)

    FALLBACK_SCORE = 3.0
    inferred = (
        pl.when(inferred_capped <= TAIL_DAMP_PIVOT)
        .then(inferred_capped)
        .otherwise(TAIL_DAMP_PIVOT + (inferred_capped - TAIL_DAMP_PIVOT) * TAIL_DAMP)
    ).fill_null(FALLBACK_SCORE).clip(1.0, 10.0).alias("czLiqScore")

    return lf.select(["isin", inferred]).filter(pl.col('czLiqScore').is_not_null() & pl.col('czLiqScore').is_not_nan())


# ----------------------------------------------------------------
# -- META
# ---------------------------------------------------------------
META_MERGE_KEY = '__meta_merge_key'
META_MERGE_EXPR = pl.lit(1, pl.UInt8).alias(META_MERGE_KEY)

async def init_meta(my_pt, region='US', dates=None, frames=None, **kwargs):
    # note my_pt here is raw
    n = now_time(utc=True)
    return my_pt.head(1).select([
        META_MERGE_EXPR,
        pl.col("rfqCreateDate").alias("date"),
        pl.when(
            pl.col('rfqCreateTime').is_null()
        ).then(
            pl.lit(n, pl.String).str.strptime(dtype=pl.Time)
        ).otherwise(pl.col('rfqCreateTime'))
        .alias('time'),
        pl.when(pl.col("rfqAon").is_null()).then(FLAG_NO).otherwise(FLAG_YES).cast(pl.Int8).alias("isAon"),
        pl.col("rfqType").alias("rfqType"),
        pl.when(pl.col("rfqType").cast(pl.String, strict=False).eq("Cross")).then(FLAG_YES).otherwise(FLAG_NO).alias("isCrossed"),
        pl.col("rfqCustFirm").alias("client"),
        pl.when(
            (pl.col('rfqCustUserInfo').is_null()) &
            (pl.col('rfqCustUserName').str.contains("@", literal=True))
        ).then(
            pl.col("rfqCustUserName").str.split("@").list.first().str.to_lowercase()
        ).otherwise(
            pl.col('rfqCustUserInfo')
        ).alias('clientTraderUsername'),
        pl.when(
            (pl.col('rfqCustUserName').str.contains("@", literal=True))
        ).then(
            pl.lit(None, pl.String)
        ).otherwise(
            pl.col('rfqCustUserName').str.to_titlecase()
        ).alias('clientTraderName'),
        pl.col("rfqEnquiryType").alias("inquiryType"),
        pl.col("rfqListId").alias("rfqListId"),
        pl.col("rfqMarketId").alias("venue"),
        pl.when(pl.col("rfqNumOfDealers").is_null()).then(pl.lit(0, pl.Int64)).otherwise(pl.col("rfqNumOfDealers")).alias("numDealers"),
        pl.when(pl.col("rfqOnTheWire").is_null()).then(pl.lit(0, pl.Float64)).otherwise(pl.col("rfqOnTheWire")).alias("wireTime"),
        pl.col("rfqSdsId").alias("clientSdsId").cast(pl.String, strict=False),
        pl.col("rfqValidityTime").alias("timeToPrice").cast(pl.Float64, strict=False),
        pl.col("refPriceSrc").str.to_uppercase().alias("_fwdStrikeMkt"),
        pl.col("refPriceTime").dt.replace_time_zone("UTC").alias("fwdStrikeTime"),
        pl.col("refPriceSide").alias("fwdStrikeSide"),
        pl.col("refPriceType").alias("fwdStrikeQuoteType"),
        pl.col("rfqConsolidatedState").alias("rfqState"),
        pl.when(pl.col("rfqValidityTime").is_null()).then(
            pl.lit("NEW", pl.String)
        ).otherwise(
            pl.lit("LIVE", pl.String)
        ).alias("state"),
        pl.col('refPriceSrc').str.to_uppercase().str.extract('([0-9 ]+(PM|AM))').str.strip_chars().alias('_fwdTimeHint')
    ]).with_columns([
        pl.col('date').dt.combine(pl.col('time')).dt.replace_time_zone("UTC").alias('datetime'),
        pl.when(
            pl.col('_fwdStrikeMkt').is_null() | (pl.col('_fwdStrikeMkt').str.strip_chars()=='')
        ).then(pl.lit(None, pl.String))
        .when(
            pl.col('_fwdStrikeMkt').str.contains('BVAL', literal=True)
        ).then(pl.lit('BVAL', pl.String))
        .when(
            pl.col('_fwdStrikeMkt').str.contains('(IDC|ICE|CEP)', literal=False)
        ).then(pl.lit('IDC', pl.String))
        .otherwise(pl.col('_fwdStrikeMkt'))
        .alias('fwdStrikeMkt'),
        pl.when(
            pl.col("fwdStrikeTime").is_null()
        ).then(pl.lit(None, pl.String)).otherwise(
            pl.col("fwdStrikeTime").dt.time().cast(pl.String, strict=False).replace(BVALSnapshotTimes().time_to_moniker)
        ).alias("fwdStrikeTimeMnemonic"),
        pl.lit(None, pl.String).alias("syncedTime"),
        pl.col("venue").str.to_uppercase().replace_strict(
            MARKET_MAPS, return_dtype=pl.String, default=None
        ).alias('venueShort'),
        pl.col('venue').str.to_uppercase().str.replace_all('[\s_-]', "", literal=False).alias('_venue')
    ]).with_columns([
        pl.when(pl.col('venueShort').is_not_null()).then(pl.col('venueShort')).otherwise(
            pl.when(
                pl.col('_venue').str.contains("TRADEWEB")  |
                pl.col('_venue').str.contains("TW")        |
                pl.col('_venue').str.contains("CORI")
            ).then(pl.lit("TW", pl.String))
            .when(
                pl.col('_venue').str.contains("BBG")       |
                pl.col('_venue').str.contains("BLOOMBERG") |
                pl.col('_venue').str.contains("BX")
            ).then(pl.lit("BBG", pl.String))
            .when(
                pl.col('_venue').str.contains("MX")        |
                pl.col('_venue').str.contains("MARKET")    |
                pl.col('_venue').str.contains("AXE")
            ).then(pl.lit("MX", pl.String))
            .when(
                pl.col('_venue').str.contains("TRU")
            ).then(pl.lit("TRM", pl.String))
            .when(
                pl.col('_venue').str.contains("MANUAL")   |
                pl.col('_venue').str.contains("EMAIL")    |
                pl.col('_venue').str.contains("UPLOAD")
            ).then(pl.lit("MANUAL", pl.String))
            .when(
                pl.col('_venue').str.contains("OTHER")
            ).then(pl.lit("OTHER", pl.String))
            .otherwise(
                pl.col('_venue')
            )
        ).alias('venueShort')
    ]).with_columns([
        (pl.col("datetime") + (pl.col("timeToPrice") * 1000).cast(pl.Duration("ms"))).alias("dueTime"),
        pl.when(pl.col('fwdStrikeMkt').is_not_null() & pl.col('fwdStrikeTimeMnemonic').is_not_null()).then(
            pl.concat_str([pl.col('fwdStrikeMkt'), pl.col('fwdStrikeTimeMnemonic')], separator=" ")
        ).otherwise(pl.lit(None, pl.String)).alias('fwdStrikeName'),
        pl.when(
            pl.col('fwdStrikeMkt').is_not_null() |
            pl.col('fwdStrikeTimeMnemonic').is_not_null()
        ).then(FLAG_YES).otherwise(FLAG_NO).alias('isFwdStrike')
    ]).drop(['_venue'], strict=False)

async def meta_aggregates(my_pt, region='US', dates=None, frames=None, **kwargs):
    return my_pt.select(
        [
            'grossSize', 'netSize', 'bidSize', 'askSize', 'grossDv01', 'netDv01', 'cs01', 'cs01Pct', 'accruedInterest',
            'axeSize', 'antiSize', 'axeBidSize', 'axeBidDv01', 'axeBidCs01', 'axeBidCs01Pct',
            'axeAskSize', 'axeAskDv01', 'axeAskCs01', 'axeAskCs01Pct', 'firmBsrSize', 'firmBsiSize', 'firmBsnSize',
            'firmBsinSize', 'firmBsrDv01', 'firmBsiDv01', 'firmBsnDv01', 'firmBsinDv01',
            'firmBsrCs01', 'firmBsiCs01', 'firmBsnCs01', 'firmBsinCs01', 'firmBsrCs01Pct', 'firmBsiCs01Pct',
            'firmBsnCs01Pct', 'firmBsinCs01Pct', 'algoBsrSize', 'algoBsiSize', 'algoBsnSize',
            'algoBsinSize', 'algoBsrDv01', 'algoBsiDv01', 'algoBsnDv01', 'algoBsinDv01', 'algoBsrCs01', 'algoBsiCs01',
            'algoBsnCs01', 'algoBsinCs01', 'algoBsrCs01Pct', 'algoBsiCs01Pct', 'algoBsnCs01Pct',
            'algoBsinCs01Pct', 'strategyBsrSize', 'strategyBsiSize', 'strategyBsnSize', 'strategyBsinSize',
            'strategyBsrDv01', 'strategyBsiDv01', 'strategyBsnDv01', 'strategyBsinDv01', 'strategyBsrCs01',
            'strategyBsiCs01', 'strategyBsnCs01', 'strategyBsinCs01', 'strategyBsrCs01Pct', 'strategyBsiCs01Pct',
            'strategyBsnCs01Pct', 'strategyBsinCs01Pct', 'deskBsrSize', 'deskBsiSize', 'deskBsnSize',
            'deskBsinSize', 'deskBsrDv01', 'deskBsiDv01', 'deskBsnDv01', 'deskBsinDv01', 'deskBsrCs01', 'deskBsiCs01',
            'deskBsnCs01', 'deskBsinCs01', 'deskBsrCs01Pct', 'deskBsiCs01Pct', 'deskBsnCs01Pct',
            'deskBsinCs01Pct', 'firmAxeBsrSize', 'firmAxeBsiSize', 'firmAxeBsnSize', 'firmAxeBsinSize',
            'firmAntiBsrSize', 'firmAntiBsiSize', 'firmAntiBsnSize', 'firmAntiBsinSize', 'firmAxeBsrDv01',
            'firmAxeBsiDv01', 'firmAxeBsnDv01', 'firmAxeBsinDv01', 'firmAntiBsrDv01', 'firmAntiBsiDv01',
            'firmAntiBsnDv01', 'firmAntiBsinDv01', 'firmAxeBsrCs01', 'firmAxeBsiCs01', 'firmAxeBsnCs01',
            'firmAxeBsinCs01', 'firmAntiBsrCs01', 'firmAntiBsiCs01', 'firmAntiBsnCs01', 'firmAntiBsinCs01',
            'firmAxeBsrCs01Pct', 'firmAxeBsiCs01Pct', 'firmAxeBsnCs01Pct', 'firmAxeBsinCs01Pct',
            'firmAntiBsrCs01Pct', 'firmAntiBsiCs01Pct', 'firmAntiBsnCs01Pct', 'firmAntiBsinCs01Pct',
            'signalAlignedAxeSize', 'signalAlignedAxeDv01', 'signalAlignedAxeCs01', 'signalAlignedAxeCs01Pct',
            'signalUnalignedAntiSize', 'signalUnalignedAntiDv01', 'signalUnalignedAntiCs01',
            'signalUnalignedAntiCs01Pct', 'signalAlignedBsrSize', 'signalAlignedBsrDv01', 'signalAlignedBsrCs01',
            'signalAlignedBsrCs01Pct', 'signalUnalignedBsinSize', 'signalUnalignedBsinDv01', 'signalUnalignedBsinCs01',
            'signalUnalignedBsinCs01Pct', 'signalAlignedAxeBsrSize', 'signalAlignedAxeBsrDv01',
            'signalAlignedAxeBsrCs01', 'signalAlignedAxeBsrCs01Pct', 'signalUnalignedAntiBsinSize',
            'signalUnalignedAntiBsinDv01', 'signalUnalignedAntiBsinCs01', 'signalUnalignedAntiBsinCs01Pct'
        ]
    ).select(pl.all().sum()).with_columns(META_MERGE_EXPR)


@hypercache.cached(ttl=timedelta(days=3), key_params=['sdsId'])
async def client_meta_data(sdsId, lookback=3):
    from app.services.kdb.hosts.connections import P1, fconn
    dates = safe_date_lookback(None, lookback * -1)
    min_dt = dates[0].strftime("%Y.%m.%d")
    q = f'select first client, first ultimateBuyingCentreId, 4h$string (first ultimateBuyingCentreName), first buyingCentreId, first buyingCentreId, 4h$string (first buyingCentreName), first businessType, first businessClassification, first industry, first country, first region, first dpgs, first tier by sdsId from ratesClientsMetadata where date>=.z.d-{min_dt}, sdsId={sdsId}, not ultimateBuyingCentreName = `, not buyingCentreName = `'
    r = await query_kdb(q, config=fconn(P1, timeline='historical'))
    if r is not None:
        return r.with_columns([
            pl.col('ultimateBuyingCentreName').map_elements(lambda s: ''.join(chr(int(b)) for b in s), return_dtype=pl.String),
            pl.col('buyingCentreName').map_elements(lambda s: ''.join(chr(int(b)) for b in s), return_dtype=pl.String)
        ])


@hypercache.cached(ttl=timedelta(days=3))
async def all_client_meta_data(lookback=3):
    from app.services.kdb.hosts.connections import P1, fconn
    dates = safe_date_lookback(None, lookback * -1)
    min_dt = dates[0].strftime("%Y.%m.%d")
    q = f'select first client, first ultimateBuyingCentreId, 4h$string (first ultimateBuyingCentreName), first buyingCentreId, first buyingCentreId, 4h$string (first buyingCentreName), first businessType, first businessClassification, first industry, first country, first region, first dpgs, first tier by sdsId from ratesClientsMetadata where date>=.z.d-{min_dt}, sdsId>0, not ultimateBuyingCentreName = `, not buyingCentreName = `'
    r = await query_kdb(q, config=fconn(P1, timeline='historical'))
    if r is not None:
        return r.with_columns([
            pl.col('ultimateBuyingCentreName').map_elements(lambda s: ''.join(chr(int(b)) for b in s), return_dtype=pl.String),
            pl.col('buyingCentreName').map_elements(lambda s: ''.join(chr(int(b)) for b in s), return_dtype=pl.String)
        ])


@hypercache.cached(ttl=timedelta(days=1))
async def _fuzz_client_meta_data(key: str, value: Any, fuzzy=True, fuzzy_threshold=10):
    from rapidfuzz import fuzz
    try:
        cached_data = await all_client_meta_data(__verbose=False)
        if cached_data is None: return
        s = cached_data.hyper.schema()
        cast_dtype = s.get(key)
        if cast_dtype==pl.String:
            if fuzzy:
                fuzzy_data = cached_data.with_columns(
                    pl.col(key).map_elements(lambda x: fuzz.ratio(x, str(value)), return_dtype=pl.Float64).alias("similarity")
                )
                return fuzzy_data.filter(pl.col("similarity") > fuzzy_threshold).sort("similarity", descending=True)
            return cached_data.filter(pl.col(key).str.to_uppercase()==pl.lit(value).cast(cast_dtype).str.to_uppercase())
        return cached_data.filter(pl.col(key)==pl.lit(value).cast(cast_dtype))
    except Exception as e:
        await log.error(e)
        return None


non_alnum = re.compile(r'[^A-Za-z0-9 ]+')

async def client_enhance(my_pt, region='US', dates=None, frames=None, **kwargs):
    from app.helpers.string_helpers import extract_most_similar, similarity, Compare, Dist
    from app.helpers.common import CLASSIFICATION_MAP

    if 'raw' not in frames: return
    sds = frames['raw'].hyper.peek('rfqSdsId')
    cust = frames['raw'].hyper.peek('rfqCustFirm')
    username = frames['raw'].hyper.peek('rfqCustUserInfo')
    cname = frames['raw'].hyper.peek('rfqCustUserName')

    data = {
        'clientSdsId': str(sds),
        'rfqClient': cust,
        'clientUsername': username,
        'clientTrader': cname
    }

    cust_simple = non_alnum.sub('', cust)
    cust_first = cust_simple.split(" ")[0]

    from app.server import get_db
    _group_map = await get_db().select('clients')
    _rates = await all_client_meta_data(__verbose=False)

    # 1) CLASSIFICATION
    my_client = None
    if (sds is not None) and (not _group_map.filter(pl.col('clientSds').cast(pl.String, strict=False)==str(sds)).hyper.is_empty()):
        my_client = _group_map.filter(pl.col('clientSds').cast(pl.String, strict=False)==str(sds)).head(1)
    else:
        for c in (cust, cust_simple, cust_first):
            t = similarity(c, _group_map.hyper.ul('clientName'), return_scores=True, top=1, workers=3, dist=Dist.WINKLER)
            if t:
                my_client = _group_map.filter(pl.col('clientName')==t[0][0])
                break

    if my_client is not None:
        my_client = _group_map.filter(pl.col('clientId')==my_client.hyper.peek('clientId')).sort(pl.col('index')).head(1)

        new_data = (await my_client.select([
            pl.col('clientName').alias('client'),
            pl.col('clientGroup'),
            pl.col('clientSubGroup')
        ]).hyper.collect_async()).to_dicts()[0]
        data = {**new_data, **data}

    my_rates = _rates.filter(pl.col('sdsId').cast(pl.String, strict=False)==str(sds))
    if (my_rates is None) or (my_rates.hyper.is_empty()):
        rates = []
        for c in (cust, cust_simple, cust_first):
            c_sim = similarity(c, _rates.hyper.ul('client'), top=5, dist=Dist.WINKLER) or []
            rates_client = [x for x in c_sim if x[1] > 0.8]
            rates.extend(rates_client)

            b_sim = similarity(c, _rates.hyper.ul('buyingCentreName'), top=5, dist=Dist.WINKLER) or []
            rates_bc = [x for x in b_sim if x[1] > 0.8]
            rates.extend(rates_bc)

            u_sim = similarity(c, _rates.hyper.ul('ultimateBuyingCentreName'), top=5, dist=Dist.WINKLER) or []
            rates_ubc = [x for x in u_sim if x[1] > 0.8]
            rates.extend(rates_ubc)

        rate_guess = sorted(rates, key=lambda x: x[1])[-1] if rates else None

        if rate_guess is not None:
            my_rates = _rates.filter([
                (pl.col('client') == rate_guess[0]) |
                (pl.col('buyingCentreName') == rate_guess[0]) |
                (pl.col('ultimateBuyingCentreName') == rate_guess[0])
            ])

    if (my_rates is not None) and (not my_rates.hyper.is_empty()):
        new_data = (await my_rates.head(1).select([
            pl.col('sdsId').cast(pl.String, strict=False).alias('clientSdsId'),
            pl.col('client').alias('clientAltName'),
            pl.col('ultimateBuyingCentreName').alias('clientUbcName'),
            pl.col('buyingCentreName').str.to_titlecase().alias('clientBcName'),
            pl.col('businessType').str.to_titlecase().alias('clientType'),
            pl.col('businessClassification').alias('clientClassification'),
            pl.col('industry').alias('clientIndustry'),
            pl.col('dpgs').alias('clientGrade'),
            pl.col('country').alias('clientCountryOfOrigin'),
            pl.col('tier').alias('clientRatesTier')
        ]).hyper.collect_async()).to_dicts()[0]
        data = {**new_data, **data}

    v1 = CLASSIFICATION_MAP.get(data.get('clientGroup', None), None)
    v2 = CLASSIFICATION_MAP.get(data.get('clientClassification', None), None)
    v3 = CLASSIFICATION_MAP.get(data.get('clientSubGroup', None), None)

    client_bucket = v1 or v2 or v3
    data['clientBucket'] = client_bucket

    data_empty = {
        'client'               : pl.String,
        'clientUbcName'        : pl.String,
        'clientBcName'         : pl.String,
        'clientType'           : pl.String,
        'clientClassification' : pl.String,
        'clientIndustry'       : pl.String,
        'clientRatesGrant'     : pl.String,
        'clientCountryOfOrigin': pl.String,
        'clientRatesTier'      : pl.String,
        'clientGroup'          : pl.String,
        'clientSubGroup'       : pl.String,
        'clientSdsId'          : pl.String,
        'rfqClient'            : pl.String,
        'clientAltName'        : pl.String,
        'clientUsername'       : pl.String,
        'clientTrader'         : pl.String,
        'clientBucket'         : pl.String,
    }
    return pl.DataFrame(data).hyper.ensure_columns(list(data_empty.keys()), dtypes=data_empty).select(
        list(data_empty.keys())
    ).with_columns([
        pl.coalesce([
            pl.col('client'),
            pl.col('rfqClient'),
            pl.col('clientBcName'),
            pl.col('clientAltName'),
            pl.col('clientUbcName')
        ]).alias('client')
    ]).with_columns(META_MERGE_EXPR).lazy()


async def custom_meta_fields(my_pt, region='US', dates=None, frames=None, **kwargs):
    raw = frames.get('raw', pl.DataFrame())
    list_id = raw.hyper.peek('rfqListId')

    if list_id is None:
        venue = my_pt.hyper.peek('venueShort') or ''
        meta_region = my_pt.hyper.peek('region') or ''
        list_id = generate_fake_rfq_list_id(venue, date=dates, region=meta_region)

    iso = isonow(utc=True)
    portfolio_key = generate_portfolio_key(list_id)

    return pl.LazyFrame().with_columns([
        META_MERGE_EXPR,
        pl.lit(portfolio_key, pl.String).alias("portfolioKey"),
        pl.lit(list_id, pl.String).alias('rfqListId'),
        pl.lit(1, pl.Int8).alias('isReal'),
        pl.lit(0, pl.Float64).alias('pctPriced'),
        pl.lit("", pl.String).alias("barcRank"),
        pl.lit(0, pl.Float64).alias("barcPreference"),
        pl.lit(0, pl.Float64).alias('bidCoverProceeds'),
        pl.lit(0, pl.Float64).alias('bidCoverSkew'),
        pl.lit(0, pl.Float64).alias('askCoverProceeds'),
        pl.lit(0, pl.Float64).alias('askCoverSkew'),
        pl.lit(0, pl.Float64).alias('coverProceeds'),  # net
        pl.lit(0, pl.Float64).alias('coverSkew'),  # net
        pl.lit("", pl.String).alias("comment"),
        pl.lit(iso, pl.String).alias("updatedAt"),
        pl.lit(region, pl.String).alias("kdbRegion"),
        pl.lit(iso, pl.String).alias("serverTimeUtc"),
        pl.lit(None, pl.Float64).alias('serverTimeToLoad'),
        pl.lit(None, pl.String).alias("lastEditUser"),
        pl.lit(None, pl.String).alias("lastEditTime"),
    ])

SIG_THRESHOLD = 0.4
async def meta_classify(my_pt, region='US', dates=None, frames=None, **kwargs):
    from datetime import time as _time
    raw = frames['raw']
    my_pt = ensure_lazy(my_pt)

    pt_region = my_pt.group_by('desigRegion').len().with_columns([
        (pl.col('len') / pl.col('len').sum()).alias('_pct')
    ]).filter(pl.col('_pct') > SIG_THRESHOLD).hyper.ul('desigRegion')

    SGP_CUTOFF = _time(hour=10, minute=0, second=0)  # UTC
    if ('US' in pt_region) and (raw.hyper.peek('rfqCreateTime') < SGP_CUTOFF):
        pt_region.remove('US')
        if 'SGP' not in pt_region:
            pt_region.append('SGP')

    s = my_pt.hyper.schema()
    if 'deskAsset' not in s:
        _maps = await book_maps()
        my_pt = my_pt.select('desigBookId').join(_maps.select([
            pl.col('bookId').alias('desigBookId'),
            pl.col('deskAsset')
        ]), on='desigBookId', how='left')

    pt_asset = my_pt.group_by('deskAsset').len().with_columns([
        (pl.col('len') / pl.col('len').sum()).alias('_pct')
    ]).filter(pl.col('_pct') > SIG_THRESHOLD).hyper.ul('deskAsset')

    pt_cur = my_pt.group_by('currency').len().with_columns([
        (pl.col('len') / pl.col('len').sum()).alias('_pct')
    ]).filter(pl.col('_pct') > SIG_THRESHOLD).hyper.ul('currency')

    all_region = ",".join(sorted(my_pt.hyper.ul('desigRegion') or ["NA"]))
    pt_region = ",".join(sorted(pt_region or ['OTHER']))

    all_asset = ",".join(sorted(my_pt.hyper.ul('deskAsset') or ["NA"]))
    pt_asset = ",".join(sorted(pt_asset or ['OTHER']))

    all_currency = ",".join(sorted(my_pt.hyper.ul('currency') or []))
    dom_currency = ",".join(sorted(pt_cur or []))

    return pl.LazyFrame().with_columns([
        META_MERGE_EXPR,
        pl.lit(pt_region, pl.String).alias('region'),
        pl.lit(all_currency, pl.String).alias('currency'),
        pl.lit(pt_asset, pl.String).alias('assetClass'),
        pl.lit(dom_currency, pl.String).alias('dominateCurrency'),
        pl.lit(all_region, pl.String).alias('allRegions'),
        pl.lit(all_asset, pl.String).alias('allAssets')
    ])

async def agg_liquidity_score(my_pt, region='US', dates=None, frames=None, **kwargs):
    return my_pt.with_columns([
            (pl.col('lqaLiqScore') / 100).alias('lqaLiqScore')
         ]).with_columns([
            pl.coalesce([
                pl.col('macpLiqScore').hyper.fill_zero(None),
                pl.col('dkLiqScore').hyper.fill_zero(None),
                pl.col('smadLiqScore').hyper.fill_zero(None),
                pl.col('mlcrLiqScore').hyper.fill_zero(None),
                pl.col('blsLiqScore').hyper.fill_zero(None),
                pl.col('idcLiqScore').hyper.fill_zero(None),
                pl.col('lqaLiqScore').hyper.fill_zero(None),
                pl.col('czLiqScore').hyper.fill_zero(None)
            ]).alias('liqScore')
        ]).hyper.horizontal_winsor_mean([
            'blsLiqScore', 'czLiqScore', 'dkLiqScore', 'lqaLiqScore', 'macpLiqScore', 'mlcrLiqScore',
            'muniLiqScore', 'smadLiqScore', "idcLiqScore"
    ], result_alias="avgLiqScore").select([
        pl.col('isin'),
        pl.col('liqScore'),
        pl.col('avgLiqScore').hyper.fill_null(
            pl.lit(1.0, pl.Float64), include_zero_as_null=True
        ).alias('avgLiqScore')
    ])

async def meta_wavgs(my_pt, region='US', dates=None, frames=None, **kwargs):
    from app.helpers.generic_helpers import convert_credit_rating_to_numeric, convert_numeric_to_credit_rating
    weight = kwargs.get('weight', 'grossDv01')
    s = my_pt.hyper.schema()
    if weight not in s: weight = 'grossSize'

    ratings = my_pt.hyper.ul('ratingCombined')
    _map = {r:convert_credit_rating_to_numeric(r) for r in ratings}

    my_pt = my_pt.with_columns([
        pl.when(pl.col('quoteType') == 'SPD').then(FLAG_YES).otherwise(FLAG_NO).alias('isSpd'),
        pl.when(pl.col('quoteType') == 'PX').then(FLAG_YES).otherwise(FLAG_NO).alias('isPx'),
        pl.when(pl.col('quoteType') == 'MMY').then(FLAG_YES).otherwise(FLAG_NO).alias('isMmy'),
        pl.when(pl.col('quoteType') == 'DM').then(FLAG_YES).otherwise(FLAG_NO).alias('isDm'),
    ])

    cols = [
        'coupon', 'convexity', 'duration', 'yrsToMaturity', 'yrsSinceIssuance',
        'avgLife', 'cdsBasisToWorst',
        'bvalMidPx', 'bvalMidSpd', 'macpMidPx', 'macpMidSpd', 'cbbtMidPx', 'cbbtMidSpd',
        "inEtfAgg", "inEtfLqd", "inEtfHyg", "inEtfJnk", "inEtfEmb", "inEtfSpsb", "inEtfSpib", "inEtfSplb", "inEtfVcst",
        "inEtfVcit", "inEtfVclt", "inEtfSpab", "inEtfIgib", "inEtfIglb", "inEtfIgsb", "inEtfIemb", "inEtfUsig",
        "inEtfUshy", "inEtfSjnk", 'isIgAlgoEligible', 'isHybridAlgoEligible', 'isHyAlgoEligible','isEmAlgoEligible',
        'isInAlgoUniverse', "isBidAxe", "isAskAxe", "isMktAxe", "isAntiAxe",
        "isNewIssue", "isSpd", "isPx", "isMmy", "isDm"
    ]

    return my_pt.select(
        [
            META_MERGE_EXPR,
            pl.col('ratingCombined').replace_strict(
                _map, default=None, return_dtype=pl.String
            ).hyper.wavg(weight).map_elements(convert_numeric_to_credit_rating).alias('creditRating')
        ] + [
            pl.col(col).hyper.wavg(weight).alias(col) for col in cols
        ]
    )


async def meta_liquidity(my_pt, region='US', dates=None, frames=None, **kwargs):
    weight = kwargs.get('weight', 'grossDv01')
    s = my_pt.hyper.schema()
    if weight not in s: weight = 'grossSize'
    return my_pt.select([
        META_MERGE_EXPR,
        pl.col('avgLiqScore').hyper.wavg(weight).alias('avgLiqScore'),
        pl.col('blsLiqScore').hyper.wavg(weight).alias('blsLiqScore'),
        pl.col('czLiqScore').hyper.wavg(weight).alias('czLiqScore'),
        pl.col('dkLiqScore').hyper.wavg(weight).alias('dkLiqScore'),
        pl.col('lcsLiqScore').hyper.wavg(weight).alias('lcsLiqScore'),
        pl.col('liqScore').hyper.wavg(weight).alias('liqScore'),
        pl.col('lqaLiqScore').hyper.wavg(weight).alias('lqaLiqScore'),
        pl.col('macpLiqScore').hyper.wavg(weight).alias('macpLiqScore'),
        pl.col('mlcrLiqScore').hyper.wavg(weight).alias('mlcrLiqScore'),
        pl.col('muniLiqScore').hyper.wavg(weight).alias('muniLiqScore'),
        pl.col('smadLiqScore').hyper.wavg(weight).alias('smadLiqScore'),
        pl.col('idcLiqScore').hyper.wavg(weight).alias('idcLiqScore'),
    ])

async def meta_counts(my_pt, region='US', dates=None, **kwargs):
    return my_pt.select([
        META_MERGE_EXPR,
        pl.col('isin').count().alias('count'),
        pl.col('isin').n_unique().alias('uniqueCount'),
        pl.when(pl.col('side')=='BUY').then(FLAG_YES).otherwise(FLAG_NO).alias('bwicCount'),
        pl.when(pl.col('side')=='SELL').then(FLAG_YES).otherwise(FLAG_NO).alias('owicCount'),
        pl.col('isReal').sum().alias('_real')
    ]).with_columns([
        pl.when((pl.col('bwicCount') > 0) & (pl.col('owicCount') > 0)).then(pl.lit('BOWIC', pl.String))
        .when(pl.col('bwicCount') > 0).then(pl.lit('BWIC', pl.String))
        .otherwise(pl.lit('OWIC', pl.String)).alias('rfqSide'),
        (pl.col('count') - pl.col('_real')).alias('removedCount')
    ]).drop(['_real'], strict=False)

async def meta_signals(my_pt, region='US', dates=None, **kwargs):
    px_cols = [
        'signalPxAggSignal', 'signalPxCreditMom', 'signalPxCreditReversal', 'signalPxEqMom', 'signalPxHf',
        'signalPxIndexRebalance', 'signalPxLiveStatsRaw', 'signalPxRelVal', 'signalPxRfqImbalance', 'signalPxStats',
        'signalPxTraceImbalance', 'signalPxLiveFrontEnd', 'signalPxLiveStats', 'signalPxEodStats', 'signalPxTotal'
    ]
    spd_cols = [
        'signalBpsAggSignal', 'signalBpsCreditMom', 'signalBpsCreditReversal', 'signalBpsEqMom', 'signalBpsHf',
        'signalBpsIndexRebalance', 'signalBpsLiveStatsRaw', 'signalBpsRelVal', 'signalBpsRfqImbalance',
        'signalBpsStats', 'signalBpsTraceImbalance', 'signalBpsLiveFrontEnd', 'signalBpsLiveStats', 'signalBpsEodStats',
        'signalBpsTotal'
    ]
    risk_cols = [
        'signalAlignedAxeSize', 'signalAlignedAxeDv01', 'signalAlignedAxeCs01', 'signalAlignedAxeCs01Pct',
        'signalUnalignedAntiSize', 'signalUnalignedAntiDv01', 'signalUnalignedAntiCs01', 'signalUnalignedAntiCs01Pct',
        'signalAlignedBsrSize', 'signalAlignedBsrDv01', 'signalAlignedBsrCs01', 'signalAlignedBsrCs01Pct',
        'signalUnalignedBsinSize', 'signalUnalignedBsinDv01', 'signalUnalignedBsinCs01', 'signalUnalignedBsinCs01Pct',
        'signalAlignedAxeBsrSize', 'signalAlignedAxeBsrDv01', 'signalAlignedAxeBsrCs01', 'signalAlignedAxeBsrCs01Pct',
        'signalUnalignedAntiBsinSize', 'signalUnalignedAntiBsinDv01', 'signalUnalignedAntiBsinCs01',
        'signalUnalignedAntiBsinCs01Pct'
    ]

    my_pt = my_pt.hyper.ensure_columns(px_cols + spd_cols + risk_cols)
    return my_pt.select([META_MERGE_EXPR] + [
        pl.col(c).hyper.wavg('grossSize').alias(c) for c in px_cols
    ] + [
        pl.col(c).hyper.wavg('grossDv01').alias(c)  for c in spd_cols
    ] + [
        pl.col(c).sum().alias(c) for c in risk_cols
    ])

async def meta_risk(my_pt, region='US', dates=None, **kwargs):
    risk_cols = []
    return my_pt.select([META_MERGE_EXPR] + [
        pl.col(c).sum().alias(c) for c in risk_cols
    ])


@hypercache.cached(ttl=timedelta(days=7), deep={"my_pt": True}, primary_keys={'my_pt': ["benchmarkIsin"]}, key_params=[
    'my_pt'])
async def add_esm_to_benchmarks(my_pt, region='US', dates=None, source="pano", **kwargs):
    isins = my_pt.select('benchmarkIsin').hyper.to_kdb_sym('benchmarkIsin')
    if source == 'pano':
        q = 'select benchmarkEsm:sym, benchmarkIsin:isin from .mt.get[`.credit.refData] where isin in (%s), i=(last;i) fby isin' % isins
        return await query_kdb(q, PANOPROXY)
    q = ('select benchmarkEsm:sym, benchmarkIsin:isin from .esm.fixedinc where isin in (%s), i=(last;i) fby '
         'isin' % isins)
    return await query_kdb(q, GATEWAY)


async def maturity_benchmark(my_pt, region='US', dates=None, frames=None, otr_only=True, **kwargs):
    bench = frames['benchmarks']
    if otr_only:
        bench = bench.filter(pl.col('isBenchmarkOtr') == 1)

    bench = bench.select(
        pl.col("benchmarkMaturityDate"),
        pl.col("benchmarkIsin"),
    )

    bonds_with_id = my_pt.select([
        pl.col('isin'),
        pl.when([
            pl.col('pseudoWorkoutDate').is_not_null(),
            pl.col('isCallable')==1,
            pl.col('isFloater')==1,
            pl.col('isPerpetual')==1
        ]).then(pl.col('pseudoWorkoutDate'))
        .otherwise(pl.col('maturityDate')).alias('maturityDate')
    ]).with_row_index("__bond_idx")

    crossed = bonds_with_id.join(bench, how="cross")

    closest = (
        crossed.with_columns(
            (pl.col("maturityDate") - pl.col("benchmarkMaturityDate")).dt.total_days().abs().alias("__day_diff"),
        )
        .sort("__day_diff", "benchmarkMaturityDate")
        .group_by("__bond_idx")
        .agg(pl.col("benchmarkIsin").first())
        .rename({"benchmarkIsin": "maturityBenchmarkIsin"})
    )
    return bonds_with_id.join(closest, on="__bond_idx", how="left").drop(["__bond_idx", 'maturityDate'], strict=False)

BENCHMARK_PRIORITY = [
    'rfq', 'bval', 'house', 'smad', 'houseUs', 'am', 'mlcr', 'macp', 'allq', 'houseEu', 'markit', 'maturity'
]
BENCHMARK_IDX = {v:i for i,v in enumerate(BENCHMARK_PRIORITY)}

async def coalesce_benchmarks(my_pt, region='US', dates=None, frames=None, **kwargs):
    my_pt = ensure_lazy(my_pt)
    benches = ensure_lazy(frames['benchmarks'])
    candidates = my_pt.hyper.cols_like("[a-zA-Z0-9]*Benchmark(Cusip|Isin|Esm)$")
    markets = {x.split("Benchmark")[0] for x in candidates}
    market_cols = {m: (
        f"{m}BenchmarkIsin", f"{m}BenchmarkCusip", f"{m}BenchmarkEsm"
    ) for m in markets}

    all_cols = [y for x in market_cols.values() for y in x]
    my_pt = my_pt.hyper.ensure_columns(all_cols, dtypes={m:pl.String for m in all_cols})
    cusip_to_isin = benches.hyper.to_map('benchmarkCusip', 'benchmarkIsin')
    esm_to_isin = benches.hyper.to_map('benchmarkEsm', 'benchmarkIsin')

    source_exprs = []
    for m in markets:
        isin, cusip, esm = market_cols[m]
        expr = (
            pl.when(pl.col(isin).is_not_null()).then(pl.col(isin))
            .when(pl.col(cusip).is_not_null()).then(pl.col(cusip).replace_strict(cusip_to_isin, default=None, return_dtype=pl.String))
            .when(pl.col(esm).is_not_null()).then(pl.col(esm).replace_strict(esm_to_isin, default=None, return_dtype=pl.String))
            .otherwise(pl.lit(None, pl.String))
            .alias(f"{m}BenchmarkIsin")
        )
        source_exprs.append(expr)

    benchmark_sources = [
        f'{x}BenchmarkIsin' for x in sorted(markets, key=lambda x: (x not in BENCHMARK_IDX, BENCHMARK_IDX.get(x, 0)))
    ]

    isin_to_term = benches.hyper.to_map('benchmarkIsin', 'benchmarkTerm')
    isin_to_name = benches.hyper.to_map('benchmarkIsin', 'benchmarkName')

    market_term_name_exprs = []
    market_term_name_cols = []
    for m in markets:
        isin_col = f"{m}BenchmarkIsin"
        term_col = f"{m}BenchmarkTerm"
        name_col = f"{m}BenchmarkName"
        market_term_name_exprs.append(
            pl.col(isin_col).replace_strict(isin_to_term, default=None, return_dtype=pl.String).alias(term_col)
        )
        market_term_name_exprs.append(
            pl.col(isin_col).replace_strict(isin_to_name, default=None, return_dtype=pl.String).alias(name_col)
        )
        market_term_name_cols.extend([term_col, name_col])

    return my_pt.with_columns(source_exprs).hyper.ensure_columns(benchmark_sources).with_columns([
        pl.coalesce(benchmark_sources).alias('benchmarkIsin')
    ]).select(['isin', 'benchmarkIsin'] + benchmark_sources).join(
        benches.select([
            'benchFido', 'benchDescription', 'benchmarkTerm', 'benchmarkName', 'isBenchmarkOtr', 'benchmarkIsin'
        ]), on='benchmarkIsin', how='left'
    ).with_columns(market_term_name_exprs)

async def bval_yest(my_pt, region='US', dates=None, **kwargs):
    quotes = await quote_bval_non_dm(my_pt, dates=next_biz_date(dates, -1), last_snapshot_only=True, live_only=True)
    if (quotes is None) or (quotes.hyper.is_empty()): return
    return quotes.select([
        pl.col('isin'),
        pl.col('bvalBidSpd').alias('bvalYestBidSpd'),
        pl.col('bvalMidSpd').alias('bvalYestMidSpd'),
        pl.col('bvalAskSpd').alias('bvalYestAskSpd'),
        pl.col('bvalBidPx').alias('bvalYestBidPx'),
        pl.col('bvalMidPx').alias('bvalYestMidPx'),
        pl.col('bvalAskPx').alias('bvalYestAskPx'),
        pl.col('bvalBidYld').alias('bvalYestBidYld'),
        pl.col('bvalMidYld').alias('bvalYestMidYld'),
        pl.col('bvalAskYld').alias('bvalYestAskYld'),
        pl.col('bvalSnapshot').alias('bvalYestSnapshot')
    ])

async def macp_yest(my_pt, region='US', dates=None, **kwargs):
    quotes = await quote_macp_pano(my_pt, dates=next_biz_date(dates, -1))
    if (quotes is None) or (quotes.hyper.is_empty()): return
    return quotes.select([
        pl.col('isin'),
        pl.col('macpBidSpd').alias('macpYestBidSpd'),
        pl.col('macpMidSpd').alias('macpYestMidSpd'),
        pl.col('macpAskSpd').alias('macpYestAskSpd'),
        pl.col('macpBidPx').alias('macpYestBidPx'),
        pl.col('macpMidPx').alias('macpYestMidPx'),
        pl.col('macpAskPx').alias('macpYestAskPx')
    ])

async def bval_cod(my_pt, region='US', dates=None, **kwargs):
    return my_pt.select([
        pl.col('isin'),
        # labeled like this to avoid getting picked up as markets
        (pl.col('bvalBidSpd') - pl.col('bvalYestBidSpd')).alias('bvalBidSpdCod'),
        (pl.col('bvalBidPx') - pl.col('bvalYestBidPx')).alias('bvalBidPxCod'),
        (pl.col('bvalBidYld') - pl.col('bvalYestBidYld')).alias('bvalBidYldCod'),
        
        (pl.col('bvalMidSpd') - pl.col('bvalYestMidSpd')).alias('bvalMidSpdCod'),
        (pl.col('bvalMidPx') - pl.col('bvalYestMidPx')).alias('bvalMidPxCod'),
        (pl.col('bvalMidYld') - pl.col('bvalYestMidYld')).alias('bvalMidYldCod'),

        (pl.col('bvalAskSpd') - pl.col('bvalYestAskSpd')).alias('bvalAskSpdCod'),
        (pl.col('bvalAskPx') - pl.col('bvalYestAskPx')).alias('bvalAskPxCod'),
        (pl.col('bvalAskYld') - pl.col('bvalYestAskYld')).alias('bvalAskYldCod'),
    ])

async def macp_cod(my_pt, region='US', dates=None, **kwargs):
    return my_pt.select([
        pl.col('isin'),
        # labeled like this to avoid getting picked up as markets
        (pl.col('macpBidSpd') - pl.col('macpYestBidSpd')).alias('macpBidSpdCod'),
        (pl.col('macpBidPx') - pl.col('macpYestBidPx')).alias('macpBidPxCod'),
        (pl.col('macpMidSpd') - pl.col('macpYestMidSpd')).alias('macpMidSpdCod'),
        (pl.col('macpMidPx') - pl.col('macpYestMidPx')).alias('macpMidPxCod'),
        (pl.col('macpAskSpd') - pl.col('macpYestAskSpd')).alias('macpAskSpdCod'),
        (pl.col('macpAskPx') - pl.col('macpYestAskPx')).alias('macpAskPxCod'),
    ])

NYK = pl.lit('Nyk', pl.String)
EU = pl.lit('Eu', pl.String)
SGP = pl.lit('Sgp', pl.String)
async def coalesce_house(my_pt, region='US', dates=None, frames=None, **kwargs):
    house_str_cols = {
        'houseQuoteConvention', 'houseRefreshTime','houseBenchmarkIsin','houseBenchmarkCusip', 'houseBenchmarkEsm'
    }
    house_flt_cols = {
        'houseBidPx', 'houseMidPx', 'houseAskPx',
        'houseBidSpd','houseMidSpd', 'houseAskSpd',
        'houseBidYld', 'houseMidYld', 'houseAskYld'
    }
    house_cols = list(house_str_cols | house_flt_cols)
    all_cols = [x.replace('house', f'house{region}') for x in house_cols for region in ('Us', 'Eu', 'Sgp')]
    dtypes = {k:pl.Float64 if k in house_flt_cols else pl.String for k in all_cols}

    return my_pt.hyper.ensure_columns(all_cols, dtypes=dtypes).with_columns([
        pl.when(pl.col('desigRegion') == 'US').then(NYK)
        .when(pl.col('desigRegion') == 'EU').then(EU)
        .when(pl.col('desigRegion')=='SGP').then(SGP)
        .otherwise(NYK).alias('_houseRegion')
    ]).select([pl.col('isin')] + [
        pl.col('_houseRegion').hyper.case([
            ('Nyk', pl.col(col.replace('house','houseUs'))),
            ('Eu', pl.col(col.replace('house','houseEu'))),
            ('Sgp', pl.col(col.replace('house','houseSgp')))
        ]).cast(pl.Float64 if col in house_flt_cols else pl.String).alias(col) for col in house_cols
    ])

async def coalesce_stats(my_pt, region='US', dates=None, frames=None, **kwargs):
    str_cols = {'statsBenchmarkIsin'}
    flt_cols = {'statsBidPx','statsAskPx','statsMidPx','statsBidSpd','statsAskSpd','statsMidSpd'}
    full_cols = list(str_cols | flt_cols)
    all_cols = [x.replace('stats', f'stats{region}').replace('Stats',f'Stats{region}') for x in full_cols for region in ('Us', 'Eu', 'Sgp')]
    return my_pt.hyper.ensure_columns(all_cols).with_columns([
        pl.when(pl.col('desigRegion') == 'US').then(NYK)
        .when(pl.col('desigRegion') == 'EU').then(EU)
        .when(pl.col('desigRegion')=='SGP').then(SGP)
        .otherwise(NYK).alias('_region')
    ]).with_columns([pl.col('isin')] + [
        pl.col('_region').hyper.case([
            ('Nyk', pl.col(col.replace('stats','statsUs').replace('Stats','StatsUs'))),
            ('Eu', pl.col(col.replace('stats','statsEu').replace('Stats','StatsEu'))),
            ('Sgp', pl.col(col.replace('stats','statsSgp').replace('Stats','StatsSgp')))
        ]).cast(pl.Float64 if col in flt_cols else pl.String).alias(col)
        for col in full_cols
    ]).select(['isin'] + [
        pl.coalesce([
            pl.col(col.replace('stats', f'stats{region}').replace('Stats',f'Stats{region}')).alias(col+region) for region in ('Us', 'Eu', 'Sgp')
        ])
        for col in full_cols
    ])

async def quote_adj_trace(my_pt, region="US", dates=None, source='credit', **kwargs):
    cusips = my_pt.hyper.ul('cusip')
    if source == 'credit':
        triplet = construct_gateway_triplet("credit", "US", "misreport")
        server = GATEWAY_US
        cols = [
            'bidPx:col11', 'askPx:col13', 'midPx:(col11+col13)%2',
            'date','adjTraceRefreshTime:time'
        ]
    else:
        triplet = construct_panoproxy_triplet(region, 'misReport', None)
        server = PANOPROXY_US
        cols = [
            'adjTraceRefreshTime:time', 'bidPx:bidPrice', 'askPx:offerPrice', 'midPx:(bidPrice+offerPrice)%2',
            'date:.z.d'
        ]
        if not is_today(dates): return

    q = build_pt_query(
            table=triplet,
            cols=kdb_col_select_helper(cols, method='last'),
            by='cusip:sym, market:security',
            dates=dates,
            filters={'sym': cusips, "security": ['TRACE_HY', 'TRACE']},
            date_kwargs={"return_today": False},
            lastby=['sym', 'security']
        )
    r = await query_kdb(q, config=fconn(server))
    if (r is None) or (r.hyper.is_empty()): return
    r = r.hyper.utc_datetime(time_col="adjTraceRefreshTime", date_col='date', output_name="adjTraceRefreshTime")
    r = r.pivot(index='cusip', on='market', on_columns=['TRACE_HY', 'TRACE'])

    r_schema = r.hyper.schema()
    priority = ['TRACE_HY', 'TRACE']
    time_cols = [col for col in [f'adjTraceRefreshTime_{mkt}' for mkt in priority] if col in r_schema]
    bid_cols = [col for col in [f'bidPx_{mkt}' for mkt in priority] if col in r_schema]
    ask_cols = [col for col in [f'askPx_{mkt}' for mkt in priority] if col in r_schema]
    mid_cols = [col for col in [f'midPx_{mkt}' for mkt in priority] if col in r_schema]

    trace_data = r.select(
        [col for col in [
            pl.col('cusip'),
            pl.coalesce(bid_cols).alias('adjTraceBidPx') if bid_cols else None,
            pl.coalesce(mid_cols).alias('adjTraceMidPx') if mid_cols else None,
            pl.coalesce(ask_cols).alias('adjTraceAskPx') if ask_cols else None,
            pl.coalesce(time_cols).alias('adjTraceRefreshTime') if time_cols else None
        ] if not col is None])

    return trace_data.join(my_pt.select(['isin', 'cusip']), on='cusip', how='inner').drop(['cusip'], strict=False)

async def quote_trace(my_pt, region="US", dates=None, **kwargs):
    cusips = my_pt.hyper.ul('cusip')
    cols = [
        'lastTraceTime:trade_time', 'dailyTraceSize:sum[trade_quantity]', 'dailyTraceCount:count[trade_time]',
        'traceTradeType:trade_type', 'traceMidPx:price', 'traceMidYld:yield', 'traceMidYtw:yield_to_worst',
        'traceBenchmarkIsin:benchmark_isin',
        'traceMidSpd:spread2benchmark'
    ]
    q = build_pt_query(
        table=construct_gateway_triplet("enhancedfinratrace", "US", "EnhancedTracecalcTbl"),
        cols=kdb_col_select_helper(cols, method='last'),
        by='cusip',
        dates=dates,
        filters={'cusip': cusips}
    )
    r = await query_kdb(q, config=fconn(GATEWAY), name="enhancedfinratrace", timeout=5)
    if (r is None) or (r.hyper.is_empty()): return
    return r.join(my_pt.select(['isin', 'cusip']), on='cusip', how='inner').drop(['cusip'], strict=False)

async def quote_ibval(my_pt, region="EU", dates=None, **kwargs):
    isins = my_pt.hyper.ul('isin')
    triplet = construct_gateway_triplet("seriesdata", "EU", "ibval")
    cols = ['date', 'time', 'ibvalBidPx:bid', 'ibvalAskPx:offer', 'ibvalMidPx:(bid+offer)%2']
    q = build_pt_query(
            table=triplet,
            cols=kdb_col_select_helper(cols, method='last'),
            by='isin:sym',
            dates=dates,
            filters={'sym': isins},
            lastby='sym'
        )
    r = await query_kdb(q, config=fconn(GATEWAY_EU))
    if (r is None) or (r.hyper.is_empty()): return
    r = (r.hyper.utc_datetime(time_col="time", date_col='date', output_name="ibvalRefreshTime")).drop(
        ['date', 'time'], strict=False
    )
    return r

async def coop_bond(my_pt, dates=None, region="US", **kwargs):
    my_pt = ensure_lazy(my_pt)
    isins = my_pt.hyper.to_kdb_sym('isin')
    q = 'select isin:sym from .seriesconfig.latestCoopBonds where date=.z.d, sym in (%s), active=1b' % isins
    r = await query_kdb(q, fconn(GATEWAY))
    if r is None:
        r = pl.LazyFrame({'isin':[]}, schema_overrides={'isin': pl.String})
    r = r.with_columns(FLAG_YES.alias('isCoop'))
    return my_pt.select('isin').join(r, on='isin', how='left').with_columns(pl.col('isCoop').fill_null(FLAG_NO).alias('isCoop'))

async def corp_action(my_pt, dates=None, region="US", **kwargs):
    my_pt = ensure_lazy(my_pt)
    isins = my_pt.hyper.to_kdb_sym('isin')
    q = 'select corpActionType:last actionType by isin:sym from .seriesconfig.latestCorporateAction where date=.z.d, sym in (%s), active=1b' % isins
    r = await query_kdb(q, fconn(GATEWAY))
    if r is None:
        r = pl.LazyFrame({'isin':[]}, schema_overrides={'isin': pl.String})
    r = r.with_columns(FLAG_YES.alias('isCorpAction'))
    return my_pt.select('isin').join(r, on='isin', how='left').with_columns(pl.col('isCorpAction').fill_null(FLAG_NO).alias('isCorpAction'))

async def bond_targets(my_pt, dates=None, region="US", **kwargs):
    q = 'select sym, book, isin:targetBonds, targetSizes*100000j from .seriesconfig.latestBondTargets where date=.z.d, sym in `extraWideningBuySkew`extraWideningSellSkew, active=1b'
    r = await query_kdb(q, fconn(GATEWAY_US))
    if (r is None) or (r.hyper.is_empty()): return
    r = r.explode(['isin', 'targetSizes'])
    isins = my_pt.hyper.ul('isin')
    r = (
        r.filter(
            pl.col('isin').is_in(isins),
            pl.col('targetSizes').abs() > 0,
        )
        .group_by(['isin', 'sym'])
        .agg(pl.col('targetSizes').sum())
        .pivot(index='isin', on='sym', values='targetSizes', on_columns=['extraWideningBuySkew', 'extraWideningSellSkew'])
    )
    r = r.hyper.ensure_columns(
        ['extraWideningBuySkew', 'extraWideningSellSkew'],
        dtypes={'extraWideningBuySkew': pl.Int64, 'extraWideningSellSkew': pl.Int64},
    ).join(my_pt.select([
        'isin', 'side',
        'firmBsinSize', 'algoBsinSize',
        'unitDv01', 'unitCs01', 'unitCs01Pct'
    ]), on='isin', how='left')

    return r.filter(
        ((pl.col('side')=='BUY') & (pl.col('extraWideningBuySkew') > 0)) |
        ((pl.col('side')=='SELL') & (pl.col('extraWideningSellSkew') < 0))
    ).with_columns([
        pl.when(pl.col('side') == 'BUY')
        .then(pl.col('extraWideningBuySkew'))
        .otherwise(pl.col('extraWideningSellSkew'))
        .abs().alias('bsifr')
    ]).with_columns([
        pl.col('isin'),
        pl.min_horizontal([pl.col('bsifr'), pl.col('firmBsinSize')]).alias('firmBsifrSize'),
        pl.min_horizontal([pl.col('bsifr'), pl.col('algoBsinSize')]).alias('algoBsifrSize'),
    ]).select([
        pl.col('isin'),
        pl.col('firmBsifrSize'),
        pl.col('algoBsifrSize'),
        (pl.col("unitDv01") / 10_000 * pl.col("firmBsifrSize")).alias("firmBsifrDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("firmBsifrSize")).alias("firmBsifrCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("firmBsifrSize")).alias("firmBsifrCs01Pct"),
        (pl.col("unitDv01") / 10_000 * pl.col("algoBsifrSize")).alias("algoBsifrDv01"),
        (pl.col("unitCs01") / 10_000 * pl.col("algoBsifrSize")).alias("algoBsifrCs01"),
        (pl.col("unitCs01Pct") / 10_000 * pl.col("algoBsifrSize")).alias("algoBsifrCs01Pct"),
    ])


async def quote_illiquids(my_pt, region="US", dates=None, **kwargs):
    isins = my_pt.hyper.ul('isin')
    triplet = construct_gateway_triplet("smad", "US", "bondMarking")
    cols = ['date', 'time', 'illiquid:val', 'illiquidSnapshot:snapshot']
    q = build_pt_query(
            table=triplet,
            cols=kdb_col_select_helper(cols, method='last'),
            by='isin:sym, param',
            dates=dates,
            filters={'sym': isins},
        )
    r = await query_kdb(q, config=fconn(GATEWAY))
    if (r is None) or (r.hyper.is_empty()): return

    r = (r.hyper.utc_datetime(time_col="time", date_col='date', output_name="illiquidRefreshTime")).drop(['date', 'time'], strict=False)
    r = r.pivot(index=['isin','illiquidRefreshTime', 'illiquidSnapshot'], on='param', on_columns=[
        'BidPrice', 'MidPrice', 'AskPrice',
        'BidSpread', 'MidSpread', 'AskSpread',
        'SpreadBasis', 'LeaderSpread'
    ], separator="", values=['illiquid'], aggregate_function='first')
    r = r.rename({
        'BidPrice':'illiquidBidPx',
        'MidPrice': 'illiquidMidPx',
        'AskPrice': 'illiquidAskPx',
        'BidSpread': 'illiquidBidSpd',
        'MidSpread': 'illiquidMidSpd',
        'AskSpread': 'illiquidAskSpd',
        'SpreadBasis': 'illiquidSpdBasis',
        'LeaderSpread': 'illiquidLeaderSpd',
    })
    return r

async def landmines_eu(my_pt, dates=None, region="EU",  **kwargs):
    my_pt = ensure_lazy(my_pt)
    triplet = construct_gateway_triplet('seriesmodels', 'EU', 'landmines')
    query_date = next_biz_date(dates, -1) if is_today(dates) else latest_biz_date(dates, True)
    isins = my_pt.hyper.to_kdb_sym('isin')
    filters = 'sym in (%s), expiryTimestamp>=.z.p' % isins
    cols = [
        'date', 'time', 'landmineSeverity:severity', 'landmineComment:comment',
        'expiry:'
    ]
    q = build_pt_query(
        triplet,
        cols=kdb_col_select_helper(cols, method='last'),
        by=['isin:sym', 'landmineCode:warningCode', 'side'],
        dates=query_date,
        filters=filters,
        raw_filter=True,
        lastby='sym'
        )
    r = await query_kdb(q, fconn(GATEWAY_EU))
    if (r is None) or (r.hyper.is_empty()): return
    generic = r.filter(pl.col('side').is_null())
    warnings = pl.concat([
        r.filter(pl.col('side').is_not_null()).with_columns([
        pl.when(pl.col('side') == 'OFFER').then(pl.lit('SELL', pl.String))
        .otherwise(pl.lit('BUY', pl.String)).alias('side')
    ]).join(my_pt.select(['isin','side']), on=['isin', 'side'], how='inner'),
        generic
    ], how='vertical_relaxed')

    return warnings.pivot(
        index='isin',
        on='landmineSeverity',
        on_columns=['LOW', 'MEDIUM', 'HIGH'],
        values=['landmineComment']
    ).group_by(['isin']).agg([
        pl.col('LOW').alias('lowLandmineWarning'),
        pl.col('MEDIUM').alias('mediumLandmineWarning'),
        pl.col('HIGH').alias('highLandmineWarning'),
    ]).with_columns([
        pl.col('lowLandmineWarning').list.join(", ", ignore_nulls=True).alias('lowLandmineWarning'),
        pl.col('mediumLandmineWarning').list.join(", ", ignore_nulls=True).alias('mediumLandmineWarning'),
        pl.col('highLandmineWarning').list.join(", ", ignore_nulls=True).alias('highLandmineWarning')
    ]).with_columns([
        pl.col('lowLandmineWarning').hyper.fill_when("", None).alias('lowLandmineWarning'),
        pl.col('mediumLandmineWarning').hyper.fill_when("", None).alias('mediumLandmineWarning'),
        pl.col('highLandmineWarning').hyper.fill_when("", None).alias('highLandmineWarning')
    ])

# NOT DONE
async def holding_time_us(my_pt, dates=None, region="US", **kwargs):
    isins = my_pt.hyper.to_kdb_sym('isin')
    q = 'select from .mt.get[`.credit.us.ehtStaticData.realtime] where isin in (%s), i=(last;i) fby isin' % isins
    base = await query_kdb(q, fconn(PANOPROXY_US))
    base = base.join(my_pt.select([
        'isin', 'netSize', 'grossSize', 'side', 'quoteType'
    ]), on='isin', how='left')

    isSell = pl.col('side') == 'SELL'
    isBuy = pl.col('side') == 'BUY'

    d = parse_date(ensure_list(dates)[0])
    t = date_to_datetime(d).time()
    isEvening = t > datetime.strptime("15:00:00", "%H:%M:%S").time()

    dbase = base.with_columns([
        pl.lit(d, pl.Date).alias('tradeDate')
    ]).with_columns([
        pl.when(isSell).then(pl.col('liqBktBuy')).otherwise(pl.col('liqBktSell')).alias('lqbkt_factor'),
        pl.when(isSell).then(pl.col('liqBkt10ScaleSell')).otherwise(pl.col('liqBkt10ScaleBuy')).alias('lqBkt10_factor'),
        (pl.col('grossSize').sqrt() / 1e6).alias('absBSIInitToUseSqrt_factor'),
        pl.when(isSell & pl.col('principalAmountOutstanding').is_not_null()).then(pl.col('grossSize') / pl.col('principalAmountOutstanding'))
        .otherwise(pl.lit(0, pl.Float64)).alias('pctAmtOutSell_factor'),
        pl.when(pl.col('principalAmountOutstanding').is_not_null() & pl.col('principalAmountOutstanding')<=350e6).then(pl.lit(1, pl.Float64)).otherwise(pl.lit(0, pl.Float64)).alias('subIndex_factor'),
        pl.when((pl.col('newlyIssued') == 1) & (pl.col('issueDate') > pl.col('tradeDate'))).then(pl.lit(0, pl.Float64)).otherwise(
            (pl.col('tradeDate') - pl.col('issueDate')).dt.total_days().sqrt()
        ).alias('daysSinceIssuanceSqrt_factor'),
        pl.when(pl.col('quoteType') == 'PX').then(pl.lit(1, pl.Float64)).otherwise(pl.lit(0, pl.Float64)).alias('pricingMethodPRICE_factor'),
        pl.when(pl.col('grossSize') >= 5e6).then(pl.lit(1, pl.Float64)).otherwise(pl.lit(0, pl.Float64)).alias('blockLot_factor'),
        pl.when(isSell).then(pl.lit(1, pl.Float64)).otherwise(pl.lit(0, pl.Float64)).alias('bsiSideShort_factor'),
        pl.when(isSell & pl.col('principalAmountOutstanding').is_not_null() & pl.col('principalAmountOutstanding') > 0).then(pl.col('insHoldingHealth') / pl.col('principalAmountOutstanding')).otherwise(pl.lit(0, pl.Float64)).alias('healthPctAOSell_factor'),
        pl.lit(1, pl.Float64).alias('bsiActivityTagPTRFQ_factor'),
        pl.lit(0, pl.Float64).alias('bsiActivityTagAlgoRFQ_factor'),
        pl.when(isEvening).then(pl.lit(1, pl.Float64)).otherwise(pl.lit(0, pl.Float64)).alias('evening_factor'),
        pl.when(
            ((pl.col('maturityDate') - pl.col('tradeDate')) < 366) &
            isSell
        ).then(pl.lit(1, pl.Float64)).otherwise(pl.lit(0, pl.Float64))
        .alias('sub1YrMtySell_factor'),
        pl.col('insHoldingLife').sqrt().alias('insHoldingLifeSqrt_factor'),
        pl.when(
            (pl.col('etfMembership').is_null()) |
            (pl.col('etfMembership') == "") |
            (pl.col('etfMembership')=="NA_ETF")
        ).then(pl.lit(0, pl.Float64)).otherwise(pl.lit(1, pl.Float64)).alias('etfEligible_factor'),
        pl.when(
            (pl.col('grossSize') < 2e6) & (pl.col('grossSize') > 50e3)
        ).then(pl.lit(1, pl.Float64)).otherwise(pl.lit(0, pl.Float64)).alias('oddlot_factor'),
        pl.when(isSell).then(pl.col('franchiseScoreBuy')).otherwise(pl.col('franchiseScoreSell')).alias('franchiseScore_factor'),
        pl.when(pl.col('grossSize') <= 50e3).then(pl.lit(1, pl.Float64)).otherwise(pl.lit(0, pl.Float64)).alias('microlot_factor'),
        (pl.col('principalAmountOutstanding') / 1e6).sqrt().alias('bsiPrincipalAmtOutSqrt_factor')
    ]).with_columns([
        (((pl.col('lqBkt10_factor')) / (
        (pl.col('liqBkt10ScaleBuy') + pl.col('liqBkt10ScaleSell') + 1e-5))) - 0.5).alias('blsScoreImbalance_factor')
    ])

async def holding_time_eu(my_pt, dates=None, region="US", large_threshold = 7_000_000, **kwargs):
    my_pt = ensure_lazy(my_pt)
    triplet = construct_gateway_triplet('seriesmodels', 'EU', 'holdingTime')
    query_date = next_biz_date(dates, -1) if is_today(dates) else latest_biz_date(dates, True)
    isins = my_pt.hyper.ul('isin')
    filters = {'sym':isins, 'modelName': 'HOLDING_TIME'}
    cols=['date', 'time', 'x:xValues', 'y:yValues']
    q = build_pt_query(triplet, cols=kdb_col_select_helper(cols, method='last'), by=['isin:sym'], dates=query_date, filters=filters, lastby='sym')
    r = await query_kdb(q, fconn(GATEWAY_EU))
    if (r is None) or (r.hyper.is_empty()): return
    curves_ht = (
        r.with_columns(
            pl.col("x").alias("x_expl"),
            pl.col("y").alias("y_expl"),
        ).explode(["x_expl", "y_expl"]).with_columns([
            ((pl.col("y_expl") * 2.0).fill_null(0.0) / pl.col("x_expl").abs()).alias("ht_day_expl")
        ])
        .group_by("isin", maintain_order=True)
        .agg(
            pl.col("x_expl").implode().alias("x"),
            pl.col("ht_day_expl").implode().alias("ht_days"),
        )
    ).join(my_pt.select('isin', 'netSize').group_by(['isin']).agg(
        ((pl.col('netSize').sum())/1_000_000).alias('notionals_m')
    ), on='isin', how='left').with_row_count(name="pos_id", offset=0).with_columns(
            pl.col("x").list.len().cast(pl.Int64).alias("len")
        )

    idx_tbl = (
        curves_ht.select(["pos_id", "notionals_m", "x"])
        .explode("x")
        .with_columns((pl.col("x") < pl.col("notionals_m")).cast(pl.Int64).alias("lt"))
        .group_by("pos_id", maintain_order=True)
        .agg(pl.col("lt").sum().cast(pl.Int64).alias("idx"))
    )
    df = curves_ht.join(idx_tbl, on="pos_id", how="left")

    len_expr = pl.col("x").list.len().cast(pl.Int64, strict=False).alias("len")
    df = df.with_columns(
        pl.when(pl.col("len") < 2)
        .then(pl.lit(None, dtype=pl.Int64))
        .otherwise(
            pl.when(pl.col("idx") < 1)
            .then(pl.lit(1, pl.Int64))
            .when(pl.col("idx") > (pl.col("len") - 1))
            .then(pl.col("len") - 1)
            .otherwise(pl.col("idx")).cast(pl.Int64, strict=False)
        ).alias("idxc")
    )
    eht_small = (
        pl.when(pl.col("idxc").is_null() | pl.col("x").is_null() | pl.col("ht_days").is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(
            pl.col("ht_days").list.get(pl.col("idxc") - 1)
            + (
                    (pl.col("ht_days").list.get(pl.col("idxc")) - pl.col("ht_days").list.get(pl.col("idxc") - 1))
                    * (pl.col("notionals_m") - pl.col("x").list.get(pl.col("idxc") - 1))
                    / (pl.col("x").list.get(pl.col("idxc")) - pl.col("x").list.get(pl.col("idxc") - 1))
            )
        )
        .alias("EHT_days_small")
    )

    extrap_pos = (
            pl.col("ht_days").list.get(pl.col("len") - 2)
            + (pl.col("ht_days").list.get(pl.col("len") - 1) - pl.col("ht_days").list.get(pl.col("len") - 2))
            * (pl.col("notionals_m") - pl.col("x").list.get(pl.col("len") - 2))
            / (pl.col("x").list.get(pl.col("len") - 1) - pl.col("x").list.get(pl.col("len") - 2))
    )

    extrap_neg = (
            pl.col("ht_days").list.get(0)
            + (pl.col("ht_days").list.get(1) - pl.col("ht_days").list.get(0))
            * (pl.col("notionals_m") - pl.col("x").list.get(0))
            / (pl.col("x").list.get(1) - pl.col("x").list.get(0))
    )

    eht_large = (
        pl.when((pl.col("len") < 2) | pl.col("x").is_null() | pl.col("ht_days").is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(pl.col("notionals_m") > 0).then(extrap_pos)
        .otherwise(extrap_neg)
        .alias("EHT_days_large")
    )

    small_mask = (pl.col("notionals_m") > -1*(large_threshold/1e6)) & (pl.col("notionals_m") < (large_threshold/1e6))
    return (
        df.with_columns([eht_small, eht_large])
        .with_columns(
            pl.when(small_mask)
            .then(pl.col("EHT_days_small"))
            .otherwise(pl.col("EHT_days_large"))
            .alias("EHT_days")
        )
        .with_columns(
            pl.when(pl.col("EHT_days").is_finite()).then(pl.col("EHT_days")).otherwise(
                pl.lit(None, dtype=pl.Float64)
            ).alias("EHT_days")
        )
        .drop(["EHT_days_small", "EHT_days_large"], strict=False)
    ).with_columns([
        pl.when(pl.col('EHT_days').is_null()).then(
            pl.col('EHT_days').hyper.wavg('notionals_m')
        ).otherwise(
            pl.col('EHT_days')
        ).alias('eht')
    ]).select('isin', 'eht')


# -----------------------------------------------------------------------------
# ETF
# ----------------------------------------------------------------------------

async def etf_snapshot(my_pt, dates=None, region="US", tickers=None, depth=0, **kwargs):
    from app.helpers.common import KEY_ETFS
    tickers = tickers or KEY_ETFS
    triplet = construct_gateway_triplet('creditext', "US", "etfNAV")
    cols = kdb_col_select_helper([
        'nav',
        'cash:(estimatedCash+accrued+coupons)%creationUnits',
        'premiumDiscount',
        'bvalBidNav:navBvalBid', 'bvalAskNav:navBvalAsk',
        'macpBidNav:navCpplusBid', 'macpAskNav:navCpplusAsk',
        'creationUnits'
    ], method="last")
    q = build_pt_query(triplet, cols, dates=dates, by="ticker:sym", filters={'sym':tickers})
    etf = await query_kdb(q, fconn(GATEWAY, region="US"), lazy=False)
    if (etf.hyper.is_empty()) and (depth == 0):
        return await etf_snapshot(my_pt, dates=next_biz_date(dates, -1), region=region, tickers=tickers, depth=1, **kwargs)
    return etf


'''
etf.with_columns([
        pl.lit(1).alias('index')
    ]).pivot(index='index', on='ticker', separator="", maintain_order=False).with_columns([
        META_MERGE_EXPR
    ]).drop(['index'], strict=False)

'''

async def etf_constituents(my_pt, dates=None, region="US", frames=None, tickers=None, **kwargs):
    from app.helpers.date_helpers import next_biz_date
    from app.helpers.common import KEY_ETFS
    tickers = tickers or KEY_ETFS
    prev_biz_date = next_biz_date(dates, -1)
    triplet = construct_gateway_triplet('creditext', "US", "etfConstituents")
    q = build_pt_query(
        table=triplet,
        cols = kdb_col_select_helper(['quantity', 'houseMidPx:(catsBid+catsAsk)%2', 'houseMidSpd:spreadVsBenchmark'], "last"),
        dates=prev_biz_date,
        filters={'sym': tickers},
        by=['ticker:sym', 'isin']
    )
    return await query_kdb(q, GATEWAY)

async def etf_mlcr_levels(my_pt, dates=None, region="US", frames=None, tickers=None, **kwargs):
    levels = await quote_mlcr(my_pt, region=region, dates=None, **kwargs)
    if (levels is None) or (levels.hyper.is_empty()): return
    return levels.select([
        pl.col('isin'),
        pl.col('mlcrBidPx'), pl.col('mlcrMidPx'), pl.col('mlcrAskPx'),
        pl.col('mlcrBidSpd'), pl.col('mlcrMidSpd'), pl.col('mlcrAskSpd'),
    ])

async def hedge_risk(my_pt, **kwargs):
    """
    Compute hedge ratio and hedge-scaled sizes/DV01 against the benchmark.

    hedgeRatio        = unitDv01 / benchmarkUnitDv01
    grossHedgeSize    = grossSize * hedgeRatio
    netHedgeSize      = netSize   * hedgeRatio
    gross/netHedgeDv01= benchmarkUnitDv01 / 10_000 * hedgeSize
    hedgeDirection    = opposite of side (BUY -> SELL, SELL -> BUY)

    Rows without a benchmarkIsin get 0.  Rows with a benchmarkIsin but no
    resolvable benchmarkUnitDv01 also get 0, with a warning logged.
    """
    my_pt = ensure_lazy(my_pt)

    # Ensure fallback columns exist (may be null if benchmark pipeline didn't provide them)
    my_pt = my_pt.hyper.ensure_columns(
        ['benchmarkUnitDv01', 'benchmarkDuration', 'benchmarkMidPx', 'benchmarkIsin'],
        dtypes={
            'benchmarkUnitDv01': pl.Float64,
            'benchmarkDuration': pl.Float64,
            'benchmarkMidPx':    pl.Float64,
            'benchmarkIsin':     pl.String,
        },
    )

    # Estimate benchmarkUnitDv01 from duration * price when the direct value is missing
    bench_dv01 = pl.coalesce([
        pl.col('benchmarkUnitDv01').cast(pl.Float64, strict=False),
        (pl.col('benchmarkDuration').cast(pl.Float64, strict=False)
         * pl.col('benchmarkMidPx').cast(pl.Float64, strict=False)
         * 0.01),
    ]).alias('_benchDv01')

    has_benchmark = pl.col('benchmarkIsin').is_not_null() & (pl.col('benchmarkIsin') != '')
    has_dv01      = pl.col('_benchDv01').is_not_null() & (pl.col('_benchDv01').abs() > 1e-9)
    can_hedge     = has_benchmark & has_dv01

    result = (
        my_pt
        .with_columns(bench_dv01)
        .with_columns([
            pl.when(can_hedge)
              .then(pl.col('unitDv01').cast(pl.Float64, strict=False) / pl.col('_benchDv01'))
              .otherwise(0.0)
              .alias('hedgeRatio'),

            pl.when(pl.col('side') == 'BUY').then(pl.lit('SELL'))
              .when(pl.col('side') == 'SELL').then(pl.lit('BUY'))
              .otherwise(pl.lit(None, pl.String))
              .alias('hedgeDirection'),
        ])
        .with_columns([
            (pl.col('hedgeRatio') * pl.col('grossSize')).alias('grossHedgeSize'),
            (pl.col('hedgeRatio') * pl.col('netSize')).alias('netHedgeSize'),
        ])
        .with_columns([
            (pl.col('_benchDv01') / 10_000 * pl.col('grossHedgeSize')).alias('grossHedgeDv01'),
            (pl.col('_benchDv01') / 10_000 * pl.col('netHedgeSize')).alias('netHedgeDv01'),
        ])
        .select([
            'tnum', 'hedgeRatio', 'grossHedgeSize', 'netHedgeSize',
            'grossHedgeDv01', 'netHedgeDv01', 'hedgeDirection',
        ])
        .collect()
    )

    # Warn when rows have a benchmark but no usable DV01
    warn_count = my_pt.with_columns(bench_dv01).filter(
        has_benchmark & ~has_dv01
    ).select(pl.len()).collect().item()
    if warn_count:
        log.warning(f"hedge_risk: {warn_count} row(s) have benchmarkIsin but no resolvable benchmarkUnitDv01 — hedge set to 0")

    return result


async def etf_bval_levels(my_pt, dates=None, region="US", frames=None, tickers=None, **kwargs):
    my_pt = ensure_lazy(my_pt)
    levels = await quote_bval_non_dm(my_pt, live_only=True)
    if (levels is None) or (levels.hyper.is_empty()): return
    return levels.select(
        [
            pl.col('isin'),
            pl.col('bvalBidPx'), pl.col('bvalMidPx'), pl.col('bvalAskPx'),
            pl.col('bvalBidSpd'), pl.col('bvalMidSpd'), pl.col('bvalAskSpd'),
        ]
    )

async def etf_macp_levels(my_pt, dates=None, region="US", frames=None, tickers=None, **kwargs):
    my_pt = ensure_lazy(my_pt)
    levels = await quote_macp(my_pt)
    if (levels is None) or (levels.hyper.is_empty()): return
    return levels.select(
        [
            pl.col('isin'),
            pl.col('macpBidPx'), pl.col('macpMidPx'), pl.col('macpAskPx'),
            pl.col('macpBidSpd'), pl.col('macpMidSpd'), pl.col('macpAskSpd'),
        ]
    )

    # select sym, nav, cash: (estimatedCash+accrued+coupons)%creationUnits, premiumDiscount, navBvalBid, navBvalAsk, navCpplusBid, navCpplusAsk, creationUnits from .creditext.nyk.etfNAV where sym in `LQD, i=(last;i) fby sym


# ====================================================================
## ICE Data Services R+
# ====================================================================

ICE_RPLUS_DEFAULT_FIELDS = {
    # -- ICE Evaluated Pricing --------------------------------------
    "IEBID"            : ["idcEval3pmBidPx"],
    "IEMID"            : ["idcEval3pmMidPx"],
    "IEASK"            : ["idcEval3pmAskPx"],
    "IEBYLD"           : ["idcEval3pmBidYld"],
    "IEMYLD"           : ["idcEval3pmMidYld"],
    "IEAYLD"           : ["idcEval3pmAskYld"],
    "IEBID4"           : ["idcEval4pmBidPx"],
    "IEMID4"           : ["idcEval4pmMidPx"],
    "IEBYLD4"          : ["idcEval4pmBidYld"],
    "IEMYLD4"          : ["idcEval4pmMidYld"],
    "IEAYLD4"          : ["idcEval4pmAskYld"],
    'SYSP'             : ['idcEval4pmMidSpd'],

    # -- CEP (Continuous Evaluated Pricing) -------------------------
    "CEP:BID"          : ["idcBidPx"],
    "CEP:BSPRD"        : ["idcBidSpd"],
    "CEP:BYLD"         : ["idcBidYld"],
    "CEP:ASK"          : ["idcAskPx"],
    "CEP:ASPRD"        : ["idcAskSpd"],
    "CEP:AYLD"         : ["idcAskYld"],
    "CEP:MID"          : ["idcMidPx"],
    "CEP:MSPRD"        : ["idcMidSpd"],
    "CEP:MYLD"         : ["idcMidYld"],
    "CEP:TIME"         : ["idcRefreshTime"],

    # -- Trax Pricing -----------------------------------------------
    "ISBID"            : ["traxBidPx"],
    "ISMID"            : ["traxMidPx"],
    "ISASK"            : ["traxAskPx"],

    "LIQUIDITYPOINTREC": [
        "idcLiqScore", "idcLiqScoreAsset", "idcLiqScoreSector",
        "idcLiqScoreIssuer", "idcLiqScoreDuration", "idcLiqScoreYield",
        "idcLiqScoreAtmo", "_iceVolPtProjection", "_iceTradeVolPtCapacity",
        "_iceLiqPtRatio", "_iceTimeWeightedPtVol",
    ],

    # -- Reference / Static -----------------------------------------
    "DES1"             : ["issuerName"],
    "IAMT"             : ["amountIssued"],
    "AMTO"             : ["amountOutstanding"],
    "ID9"              : ["cusip"],
    "LEI"              : ["lei"],
    "PERPETUAL"        : ["isPerpetual"],
    "TTIC"             : ["ticker"],

    "BBID"             : [
        '_na',
        '_exch',
        '_ticker2',
        '_bbid1',
        '_bbid2',
        '_bbid3',
        'description'
    ]
}


async def quote_ice_rplus(my_pt, field_map=None, batch_size=200, dates=None, **kwargs):
    if field_map is None:
        field_map = ICE_RPLUS_DEFAULT_FIELDS
    field_map = {k: ensure_list(v) for k, v in field_map.items()}

    isins = my_pt.hyper.ul('isin')
    if not isins:
        return None

    from base64 import b64encode
    from app.helpers.fire_and_forget import PostClientConfig, AsyncPostClient, ParseMode
    from app.config.config import from_env

    # Build ordered field list and flatten output column names
    ice_fields = list(field_map.keys())
    output_cols = ['isin']
    for fc in ice_fields:
        output_cols.extend(field_map[fc])
    expected_width = len(output_cols)

    # Deduplicate column names (dict schema would silently collapse dupes)
    seen = {}
    unique_cols = []
    for c in output_cols:
        if c in seen:
            seen[c] += 1
            unique_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            unique_cols.append(c)

    auth = b64encode(from_env("ICE_LICENSE").encode()).decode('ascii')
    url = from_env("ICE_RPLUS_URL")
    headers = {'Authorization': f'Basic {auth}'}
    proxy = from_env('PROXY_URL', None)
    cfg = PostClientConfig(proxy=proxy, parse_mode=ParseMode.TEXT, max_retries=1)
    field_str = ','.join(['ISIN'] + ice_fields)

    all_rows = []
    date = parse_date(dates).strftime("%Y%m%d")
    isin_set = set(isins)
    async with AsyncPostClient(cfg) as client:
        for i in range(0, len(isin_set), batch_size):
            batch = isins[i:i + batch_size]
            selector = ','.join(f'IS:{isin}@US' for isin in batch)
            params = {'Request': f'GET,({selector}),({field_str}),{date}', 'Done': 'flag'}

            try:
                r = await client.post(
                    url=url, payload=params,
                    headers=headers, as_form=True,
                )
                body = r.body if isinstance(r.body, str) else r.body.decode('utf-8')
            except Exception as e:
                log.warning(f"quote_ice_rplus: batch {i // batch_size} failed: {e}")
                continue

            current = []
            for line in body.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('\\CRC') or line.startswith('\\ERR'):
                    continue
                values = [v.strip().strip('"') for v in line.split(',')]
                if values[0] in isin_set:
                    if current:
                        if len(current) == expected_width:
                            all_rows.append(current)
                        else:
                            log.trace(f"quote_ice_rplus: skipping row ({len(current)} vals, expected {expected_width}): {current[0]}")
                    current = values
                else:
                    current.extend(values)
            if current:
                if len(current) == expected_width:
                    all_rows.append(current)
                else:
                    log.trace(f"quote_ice_rplus: skipping row ({len(current)} vals, expected {expected_width}): {current[0]}")

    if not all_rows:
        return None

    df = pl.DataFrame(all_rows, schema=[(c, pl.String) for c in unique_cols], orient='row')

    # Null out sentinel values
    _sentinels = {'', 'N/A', 'n/a', '!NA', 'null', '-'}
    df = df.with_columns([
        pl.when(pl.col(c).is_in(_sentinels)).then(None).otherwise(pl.col(c)).alias(c)
        for c in unique_cols if c != 'isin'
    ])

    for c in unique_cols:
        if c == 'isin': continue
        attempted = df[c].cast(pl.Float64, strict=False)
        has_original = df[c].null_count() < len(df)
        has_numeric = attempted.null_count() < len(df)
        if has_numeric or not has_original:
            df = df.with_columns(attempted.alias(c))

    df = df.hyper.ensure_columns(['ticker', '_ticker2', 'idcRefreshTime', 'isPerpetual']).with_columns([
        pl.coalesce(['ticker', '_ticker2']).alias('ticker'),
        pl.col('idcRefreshTime')
            .str.strptime(pl.Datetime, format="%Y%m%d %H:%M:%S")
            .dt.replace_time_zone("America/New_York")
            .dt.convert_time_zone("UTC").alias('idcRefreshTime'),
        pl.when(pl.col('isPerpetual') == "Y").then(FLAG_YES).otherwise(FLAG_NO).alias('isPerpetual')
    ])

    # -- 1) Dedup 3pm vs 4pm: null out 4pm when identical to 3pm -------
    _3pm_4pm_pairs = [
        ('idcEval3pmBidPx',  'idcEval4pmBidPx'),
        ('idcEval3pmMidPx',  'idcEval4pmMidPx'),
        ('idcEval3pmAskPx', 'idcEval4pmAskPx'),
        ('idcEval3pmBidYld', 'idcEval4pmBidYld'),
        ('idcEval3pmMidYld', 'idcEval4pmMidYld'),
        ('idcEval3pmAskYld', 'idcEval4pmAskYld'),
    ]
    dedup_exprs = []
    s = set(df.hyper.fields)
    for c3, c4 in _3pm_4pm_pairs:
        if c3 in s and c4 in s:
            dedup_exprs.append(
                pl.when(pl.col(c3) == pl.col(c4))
                .then(None)
                .otherwise(pl.col(c4))
                .alias(c4)
            )
    if dedup_exprs:
        df = df.with_columns(dedup_exprs)

    # -- 2) Infer missing sides: bid/mid/ask from the other two --------
    _triplets = [
        ('idcEval3pmBidPx',  'idcEval3pmMidPx',  'idcEval3pmAskPx'),
        ('idcEval3pmBidYld', 'idcEval3pmMidYld', 'idcEval3pmAskYld'),
        ('idcEval4pmBidPx', 'idcEval4pmMidPx', 'idcEval4pmAskPx'),
        ('idcEval4pmBidYld','idcEval4pmMidYld', 'idcEval4pmAskYld'),
        ('idcBidPx',      'idcMidPx',       'idcAskPx'),
        ('idcBidSpd',     'idcMidSpd',      'idcAskSpd'),
        ('idcBidYld',     'idcMidYld',      'idcAskYld'),
        ('idcTraxBidPx',  'idcTraxMidPx',   'idcTraxAskPx'),
    ]
    infer_exprs = []
    for bid, mid, ask in _triplets:
        if not ({bid, mid, ask} <= s):
            continue
        b, m, a = pl.col(bid), pl.col(mid), pl.col(ask)
        # infer mid from bid+ask
        infer_exprs.append(
            pl.when(m.is_null() & b.is_not_null() & a.is_not_null())
            .then((b + a) / 2)
            .otherwise(m)
            .alias(mid)
        )
        # infer bid from mid+ask
        infer_exprs.append(
            pl.when(b.is_null() & m.is_not_null() & a.is_not_null())
            .then(2 * m - a)
            .otherwise(b)
            .alias(bid)
        )
        # infer ask from mid+bid
        infer_exprs.append(
            pl.when(a.is_null() & m.is_not_null() & b.is_not_null())
            .then(2 * m - b)
            .otherwise(a)
            .alias(ask)
        )

    if infer_exprs:
        df = df.with_columns(infer_exprs)

    # -- 3) Compute 4pm spreads via implied treasury yield -------------
    df = df.hyper.ensure_columns([
        'idcEval4pmMidYld', 'idcEval4pmMidSpd',
        'idcEval4pmBidYld', 'idcEval4pmAskYld',
    ])
    tsy = pl.col('idcEval4pmMidYld') - (pl.col('idcEval4pmMidSpd')/100)
    tsy3 = pl.col('idcEval3pmMidYld') - (pl.col('idcEval4pmMidSpd') / 100)
    df = df.with_columns([
        ((pl.col('idcEval4pmBidYld') - tsy)*100).alias('idcEval4pmBidSpd'),
        ((pl.col('idcEval4pmAskYld') - tsy)*100).alias('idcEval4pmAskSpd'),
    ]).with_columns([
        pl.when(
            pl.col('idcEval4pmBidYld').is_null() &
            pl.col('idcEval4pmBidPx').is_null()
        ).then(pl.col('idcEval4pmMidSpd'))
        .otherwise(pl.lit(None, pl.Float64)).alias('idcEval3pmMidSpd')
    ]).with_columns([
        pl.when(pl.col('idcEval3pmMidSpd').is_not_null()).then(
            (pl.col('idcEval3pmBidYld') - tsy3) * 100
        ).otherwise(pl.lit(None, pl.Float64)).alias('idcEval3pmBidSpd'),
        pl.when(pl.col('idcEval3pmMidSpd').is_not_null()).then(
            (pl.col('idcEval3pmAskYld') - tsy3) * 100
        ).otherwise(pl.lit(None, pl.Float64)).alias('idcEval3pmAskSpd'),
    ])

    uc = df.hyper.fields
    return df.drop([x for x in uc if x.startswith("_")], strict=False)

