

from app.helpers.pandas_helpers import pd
import polars as pl
import numpy as np
import hashlib
import pytz, time
from datetime import datetime
import copy
from app.helpers.common import MARKET_SNAPSHOT_TICKERS

def wavg(data, v, w):
    t = data[[v, w]].astype(float).dropna()
    t1 = t[v] * t[w]
    t2 = t[w].sum()
    if t2 == 0:
        return np.nan
    return sum(list(t1)) / t2


# Herfindahl–Hirschman index
def hhi(data, v, w):
    t = data[[v, w]].dropna()
    t[w] = t[w].astype(float)
    t = t.dropna()  # non floats
    t2 = t[w].sum()
    if t2 == 0:
        return np.nan
    weights = t.groupby(v, group_keys=False).sum() / t2
    return sum([x ** 2 for x in list(weights[w])])


def generate_md5_hash(msg):
    return hashlib.md5(msg.encode()).hexdigest()

L0_DESK_MAP = {
    "C": "CDO",
    "D": "Distressed",
    "E": "Emerging Market",
    "G": "High Grade",
    "H": "Hybrid Capital",
    "J": "High Yield",
    "L": "Loans",
    "M": "Municipals",
    "P": "PROP",
    "R": "Structured IR",
    "S": "Syndicates",
    "Y": "Yankees",
}

def desk_type_mapping_l0(x):
    return L0_DESK_MAP.get(x.upper(), "")


L1_DESK_MAP = {
    "C": "OTHER",
    "D": "HY",
    "E": "EM",
    "G": "IG",
    "H": "IG",
    "J": "HY",
    "L": "LOAN",
    "M": "MUNI",
    "P": "OTHER",
    "R": "OTHER",
    "S": "OTHER",
    "Y": "IG",
}

def desk_type_mapping_l1(x):
    return L1_DESK_MAP.get(x.upper(), "")

MOODY_TO_SP = {
    'AAA': 'AAA', 'AA1': 'AA+', 'AA2': 'AA', 'AA3': 'AA-',
    'A1': 'A+', 'A2': 'A', 'A3': 'A-',
    'BAA1': 'BBB+', 'BAA2': 'BBB', 'BAA3': 'BBB-',
    'BA1': 'BB+', 'BA2': 'BB', 'BA3': 'BB-',
    'B1': 'B+', 'B2': 'B', 'B3': 'B-',
    'CAA1': 'CCC+', 'CAA2': 'CCC', 'CAA3': 'CCC-',
    'CA': 'CC', 'C': 'D'
}

SP_TO_MNEMONIC = {
    'AAA': 'PRIME',
    'AA+': 'IG_HIGH_GRADE', 'AA': 'IG_HIGH_GRADE', 'AA-': 'IG_HIGH_GRADE',
    'A+': 'IG_MEDIUM_GRADE', 'A': 'IG_MEDIUM_GRADE', 'A-': 'IG_MEDIUM_GRADE',
    'BBB+': 'IG_LOW_GRADE', 'BBB': 'IG_LOW_GRADE', 'BBB-': 'IG_LOW_GRADE',
    'BB+': 'HY_UPPER_GRADE', 'BB': 'HY_UPPER_GRADE', 'BB-': 'HY_UPPER_GRADE',
    'B+': 'HY_LOWER_GRADE', 'B': 'HY_LOWER_GRADE', 'B-': 'HY_LOWER_GRADE',
    'CCC+': 'JUNK_UPPER_GRADE', 'CCC': 'JUNK_MEDIUM_GRADE', 'CCC-': 'JUNK_LOW_GRADE',
    'CC': 'JUNK_LOW_GRADE', 'C': 'JUNK_LOW_GRADE',
    'D': 'IN_DEFAULT'
}

# Rating mappings
SP_RATINGS = {
    'AAA': 'IG', 'AA+': 'IG', 'AA': 'IG', 'AA-': 'IG',
    'A+': 'IG', 'A': 'IG', 'A-': 'IG',
    'BBB+': 'IG', 'BBB': 'IG', 'BBB-': 'IG',
    'BB+': 'HY', 'BB': 'HY', 'BB-': 'HY',
    'B+': 'HY', 'B': 'HY', 'B-': 'HY',
    'CCC+': 'Distressed', 'CCC': 'Distressed', 'CCC-': 'Distressed',
    'CC': 'Distressed', 'C': 'Distressed',
    'D': 'Distressed'
}

MOODYS_RATINGS = {
    'Aaa': 'IG', 'Aa1': 'IG', 'Aa2': 'IG', 'Aa3': 'IG',
    'A1': 'IG', 'A2': 'IG', 'A3': 'IG',
    'Baa1': 'IG', 'Baa2': 'IG', 'Baa3': 'IG',
    'Ba1': 'HY', 'Ba2': 'HY', 'Ba3': 'HY',
    'B1': 'HY', 'B2': 'HY', 'B3': 'HY',
    'Caa1': 'Distressed', 'Caa2': 'Distressed', 'Caa3': 'Distressed',
    'Ca': 'Distressed', 'C': 'Distressed'
}

FITCH_RATINGS = {
    'AAA': 'IG', 'AA+': 'IG', 'AA': 'IG', 'AA-': 'IG',
    'A+': 'IG', 'A': 'IG', 'A-': 'IG',
    'BBB+': 'IG', 'BBB': 'IG', 'BBB-': 'IG',
    'BB+': 'HY', 'BB': 'HY', 'BB-': 'HY',
    'B+': 'HY', 'B': 'HY', 'B-': 'HY',
    'CCC+': 'Distressed', 'CCC': 'Distressed', 'CCC-': 'Distressed',
    'CC': 'Distressed', 'C': 'Distressed',
    'RD': 'Distressed', 'D': 'Distressed'
}

def get_rating_mnemonic(rating):
    if rating in MOODYS_RATINGS.keys():
        rating = MOODY_TO_SP[rating.upper()]
    return SP_TO_MNEMONIC.get(rating, None)

def get_sp_asset_class(rating, include_distressed= True):
    rating = rating.upper().strip().replace("U","").replace("*","")
    asset_class = SP_RATINGS.get(rating)
    if asset_class is None:
        raise ValueError(f"Invalid S&P rating: {rating}")
    return asset_class if include_distressed or asset_class != "Distressed" else "HY"

def get_moodys_asset_class(rating, include_distressed=True):
    rating = rating.upper().strip().replace("U","").replace("*","")
    asset_class = MOODYS_RATINGS.get(rating)
    if asset_class is None:
        raise ValueError(f"Invalid Moody's rating: {rating}")
    return asset_class if include_distressed or asset_class != "Distressed" else "HY"

def get_fitch_asset_class(rating, include_distressed = True):
    rating = rating.upper().strip().replace("U","").replace("*","")
    asset_class = FITCH_RATINGS.get(rating)
    if asset_class is None:
        raise ValueError(f"Invalid Fitch rating: {rating}")
    return asset_class if include_distressed or asset_class != "Distressed" else "HY"

def get_asset_class_from_rating_agency(rating, include_distressed=True):
    rating = rating.upper().strip().replace("U","").replace("*","")
    if rating.upper() in SP_RATINGS: # Same as fitch
        return get_sp_asset_class(rating, include_distressed)
    if rating in MOODYS_RATINGS:
        return get_moodys_asset_class(rating, include_distressed)
    return None

def convert_rating_to_sp(rating, return_none=True):
    if rating is None: return
    rating = rating.upper()
    if rating in SP_RATINGS: return rating
    return MOODY_TO_SP.get(rating, None if return_none else rating)


def convert_credit_rating_to_numeric(x):
    if x is None:
        return None
    if x == "AAA":
        return 0.26
    if x == "AA+":
        return 0.36
    if x == "AA":
        return 0.48
    if x == "AA-":
        return 0.61
    if x == "A+":
        return 0.72
    if x == "A":
        return 0.89
    if x == "A-":
        return 1.04
    if x == "BBB+":
        return 1.20
    if x == "BBB":
        return 1.42
    if x == "BBB-":
        return 1.69
    if x == "BB+":
        return 3.16
    if x == "BB":
        return 4.01
    if x == "BB-":
        return 5.19
    if x == "B+":
        return 5.35
    if x == "B":
        return 7.19
    if x == "B-":
        return 9.96
    if x == "CCC+":
        return 14.58
    if x == "CCC":
        return 14.58
    if x == "CCC-":
        return 14.58
    if x == "CC+":
        return 19.56
    if x == "CC":
        return 19.56
    if x == "CC-":
        return 19.56
    if x == "C+":
        return 25.28
    if x == "C":
        return 25.28
    if x == "C-":
        return 25.28
    if x == "D":
        return 29.82
    return 10


def convert_numeric_to_credit_rating(x):
    if x is None:
        return None
    if x == 0:
        return None
    if x <= 0.31:
        return "AAA"
    if x <= 0.42:
        return "AA+"
    if x <= 0.545:
        return "AA"
    if x <= 0.665:
        return "AA-"
    if x <= 0.805:
        return "A+"
    if x <= 0.965:
        return "A"
    if x <= 1.12:
        return "A-"
    if x <= 1.31:
        return "BBB+"
    if x <= 1.555:
        return "BBB"
    if x <= 2.425:
        return "BBB-"
    if x <= 3.585:
        return "BB+"
    if x <= 4.6:
        return "BB"
    if x <= 5.27:
        return "BB-"
    if x <= 6.27:
        return "B+"
    if x <= 8.575:
        return "B"
    if x <= 12.27:
        return "B-"
    if x <= 17.07:
        return "CCC"
    if x <= 22.42:
        return "CC"
    if x <= 27.55:
        return "C"
    return "D"

# TODO: this ALLQ map isnt correct i dont think.

ALLQ_CODE_MAP = {
    "BXAL": ('ALGO', ['IG', 'HY']),
    "BXCR": ('US', ['IG', 'HY']),
    'BXEM': ('EU', ['EM']),
    'BXCA': ('SGP', ['IG', 'HY', 'EM']),
}

QUOTE_EVENT_MARKET_MAP = {
    'AMSTEL_CASH': 'AMSTEL',
    'AMSTEL_CASH_AP': 'AMSTEL',
    'AMSTEL_CASH_LN': 'AMSTEL',
    'BATS_BBG_AXE_T0': 'BBG_AXE_T0',
    'BATS_BBG_AXE_T1': 'BBG_AXE_T1',
    'BATS_BBG_AXE_T2': 'BBG_AXE_T2',
    'BATS_BBG_AXE_T3': 'BBG_AXE_T3',
    'BATS_BBG_BXAL_T0': 'BXAL_T0',
    'BATS_BBG_BXAL_T1': 'BXAL_T1',
    'BATS_BBG_BXAL_T2': 'BXAL_T2',
    'BATS_BBG_BXAL_T3': 'BXAL_T3',
    'BATS_BBG_BXOL_T0': 'BXOL_T0',
    'BATS_BBG_BXOL_T1': 'BXOL_T1',
    'BATS_BBG_BXOL_T2': 'BXOL_T2',
    'BATS_BBG_BXOL_T3': 'BXOL_T3',
    'BATS_BXEC_T0': 'BXEC_T0',
    'BATS_BXEC_T1': 'BXEC_T1',
    'BATS_BXEC_T2': 'BXEC_T2',
    'BATS_BXEC_T3': 'BXEC_T3',
    'BATS_F_TW_BXOL_T0': 'FTW_BXOL_T0',
    'BATS_F_TW_BXOL_T1': 'FTW_BXOL_T1',
    'BATS_F_TW_BXOL_T2': 'FTW_BXOL_T2',
    'BATS_F_TW_BXOL_T3': 'FTW_BXOL_T3',
    'BATS_MX_EUCR_T0': 'MX_EUCR_T0',
    'BATS_MX_EUCR_T1': 'MX_EUCR_T1',
    'BATS_MX_EUCR_T2': 'MX_EUCR_T2',
    'BATS_MX_EUCR_T3': 'MX_EUCR_T3',
    'BATS_NEPTUNE_AXE_T0': 'NEPTUNE_AXE_T0',
    'BATS_NEPTUNE_AXE_T1': 'NEPTUNE_AXE_T1',
    'BATS_NEPTUNE_AXE_T2': 'NEPTUNE_AXE_T2',
    'BATS_NEPTUNE_AXE_T3': 'NEPTUNE_AXE_T3',
    'BATS_TW_AXE_T0': 'TW_AXE_T0',
    'BATS_TW_AXE_T1': 'TW_AXE_T1',
    'BATS_TW_AXE_T2': 'TW_AXE_T2',
    'BATS_TW_AXE_T3': 'TW_AXE_T3',
    'BATS_TW_BXOL_T0': 'TW_BXOL_T0',
    'BATS_TW_BXOL_T1': 'TW_BXOL_T1',
    'BATS_TW_BXOL_T2': 'TW_BXOL_T2',
    'BATS_TW_BXOL_T3': 'TW_BXOL_T3',
    'BATS_TW_EUCR_T0': 'TW_EUCR_T0',
    'BATS_TW_EUCR_T1': 'TW_EUCR_T1',
    'BATS_TW_EUCR_T2': 'TW_EUCR_T2',
    'BATS_TW_EUCR_T3': 'TW_EUCR_T3',
    'BBG_BPIPE': 'CBBT_CDS',
    'BBG_BPIPE_CASH': 'CBBT',
    'BBG_BPIPE_CASH_LN': 'CBBT',
    'BBG_BPIPE_CASH_AP': 'CBBT',
    'BGC_CASH': 'BGC',
    'BGC_CASH_AP': 'BGC',
    'BGC_CASH_LN': 'BGC',
    'BONDPOINT': 'BONDPOINT',
    'BONDSPRO': 'BONDSPRO',
    'CHP_ALLMARKETS': 'AM',
    'CHP_AXI': 'AXI',
    'CHP_BEMUBBG_NY': 'ALLQ_BEMU',
    'CHP_BLOOMBERG': 'ALLQ_BXCR',
    'CHP_BXCABBG_AP': 'ALLQ_BXCA',
    'CHP_BXEMBBG_LN': 'ALLQ_BXEM',
    'CHP_STATS': 'STATS',
    'CREDIT_IBVAL': 'IBVAL',
    'CREDIT_IEVAL': 'IEVAL',
    'CREDIT_MLCR': 'MLCR',
    'MLCR':'MLCR',
    'CREDIT_SMM': 'SMM',
    'CREDIT_SMM_DEALER': 'SMM_DEALER',
    'ETF_MD': 'ETF',
    'GFI_CASH': 'GFI',
    'GFI_CASH_AP': 'GFI',
    'GFI_CASH_LN': 'GFI',
    'HDAT_CASH': 'HDAT',
    'HDAT_CASH_AP': 'HDAT',
    'HDAT_CASH_LN': 'HDAT',
    'ICAP_CASH': 'ICAP',
    'ICAP_CASH_AP': 'ICAP',
    'ICAP_CASH_LN': 'ICAP',
    'MA_LIVE_MD': 'MA_LIVE',
    'MACP': 'MACP',
    'MACP_LN': 'MACP',
    'MACP_AP': 'MACP',
    'MARKIT': 'MARKIT',
    'MARKIT_LN': 'MARKIT',
    'MARKIT_AP': 'MARKIT',
    'REUTERS': 'REUTERS',
    'TMC': 'TMC',
    'TRAD_CASH': 'TRAD',
    'TRAD_CASH_AP': 'TRAD',
    'TRAD_CASH_LN': 'TRAD',
    'TRADEWEB_MD': 'TW',
    'TULLET_CASH': 'TULLET',
    'TULLET_CASH_AP': 'TULLET',
    'TULLETT_CASH_LN': 'TULLET',
    'TW_AI_MD': 'TW_AI'
}

def quoteevent_market_maps(x):
    return QUOTE_EVENT_MARKET_MAP.get(x.upper(), x.upper())

MARKET_MAPS = {
    "TRADEWEBEUPT"     : "TW",
    "TRADEWEB"         : "TW",
    "TRADEWEBCORI"     : "TW",
    "TRADEWEBCORIPT"   : "TW",
    "TRADEWEBCORIEUPT" : "TW",
    "BXPT"             : "BBG",
    "BXEUPT"           : "BBG",
    "BBG"              : "BBG",
    "BBGEUPT"          : "BBG",
    "MXUSCRPTTRADING"  : "MX",
    "MXEUCRPTTRADING"  : "MX",
    "MARKETAXESS"      : "MX",
    "MARKETAXESSEUPT"  : "MX",
    "TRUMID"           : "TRM",
    "TRUMIDEUPT"       : "TRM",
    "TRUMIDPT"         : "TRM",
}

def market_id_maps(x):
    # note: not all of these exist
    r = MARKET_MAPS.get(x.upper().strip(), None)
    return r if not r is None else guess_market(x)


def guess_market(mkt):
    x = mkt.upper()
    if "TRADEWEB" in x: return "TW"
    if "TW" in x: return "TW"
    if "CORI" in x: return "TW"
    if "BBG" in x: return "BBG"
    if "BLOOMBERG" in x: return "BBG"
    if "BX" in x: return "BBG"
    if "MX" in x: return "MX"
    if "MARKET" in x: return "MX"
    if "AXE" in x: return "MX"
    if "TRU" in x: return "TRUMID"
    if "MANUAL" in x: return "MANUAL"
    if "EMAIL" in x: return "MANUAL"
    if "UPLOAD" in x: return "MANUAL"
    if "OTHER" in x: return "OTHER"
    return mkt


def vals_over_threshold(vals, threshold=0.25):
    if len(vals) == 0:
        return "NA"
    groups = vals.value_counts()
    div = groups.sum()
    if div == 0:
        return "NA"
    groups = groups / div
    return ",".join(sorted(groups[groups >= threshold].index))

BVAL_ASSET_MAP = {
    1 : "Sovereign",
    2 : "Emerging Market Sovereign",
    3 : "Investment Grade Corporate",
    4 : "High Yield Corporate",
    5 : "Emerging Market Corporate",
    6 : "Interest Rate Swap",
    7 : "UST",
    8 : "Agency",
    9 : "Emerging Market Agency",
    11: "Convertible",
    12: "CDS Rate",
    13: "CDS Deal",
    15: "IRS Deal",
    34: "Private Placement Bonds",
    35: "Syndicated Loan",
    39: "Structured Note",
    40: "Emerging Market Debt - IG",
    41: "Emerging Market Debt - HY",
    42: "G20 Sovereign",
    43: "Non-G20 Sovereign",
    64: "Defaulted",
    65: "CLO"
}

BVAL_SUB_ASSET_MAP = {
    1 : "SSA",
    2 : "EM",
    3 : "IG",
    4 : "HY",
    5 : "EM",
    6 : "IRS",
    7 : "UST",
    8 : "Agency",
    9 : "EM",
    11: "Convertible",
    12: "CDS",
    13: "CDS",
    15: "IRS",
    34: "CORP",
    35: "LOAN",
    39: "STRUCTURED NOTE",
    40: "EM",
    41: "EM",
    42: "IG",
    43: "EM",
    64: "HY",
    65: "CLO"
}

def bval_asset_class_map(x):
    x = int(x) if str(x).isnumeric() else None
    return BVAL_ASSET_MAP.get(x, None)

def bval_sub_asset_class_map(x):
    x = int(x) if str(x).isnumeric() else None
    return BVAL_SUB_ASSET_MAP.get(x, None)

def first_intersection(source, priority_list):
    for x in priority_list:
        if x in source:
            return x
    return None

def get_emini_future_ticker(reference_date=None):
    if reference_date is None:
        reference_date = datetime.now()
    elif isinstance(reference_date, str):
        reference_date = datetime.strptime(reference_date, "%Y-%m-%d")

    # E-mini S&P 500 futures contract months
    contract_months = ['H', 'M', 'U', 'Z']
    month_to_letter = {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}

    # Find the most recent contract month
    current_month = reference_date.month
    current_year = reference_date.year

    # Adjust for the fact that contracts expire on the third Friday of the contract month
    if reference_date.day >= 15:
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    # Find the next valid contract month
    while current_month not in [3, 6, 9, 12]:
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    contract_letter = month_to_letter[current_month]
    year_code = str(current_year)[-2:]  # Last two digits of the year

    return f"ES{contract_letter}{year_code}"

def clean_market_snapshot_tickers(ms):
    ms = ms.copy()
    ms.loc[ms.sym.str.startswith('ES'), "sym"] = "ESA"
    ms.sym = ms.sym.str.replace(".", "", regex=False)
    ms.sym = ms.sym.str.replace("=TE", "", regex=False)

    ms.loc[ms.sym == "CDXHY5Y", 'sym'] = "CDX-HY"
    ms.loc[ms.sym == "CDXIG5Y", 'sym'] = "CDX-IG"

    return ms

def get_market_snapshot_tickers():
    my_tickers = copy.copy(MARKET_SNAPSHOT_TICKERS)
    emini_contract = get_emini_future_ticker()
    if 'ESA' in MARKET_SNAPSHOT_TICKERS:
        my_tickers[my_tickers.index("ESA")] = emini_contract
    return my_tickers

async def unique_left_merge(df1, df2, id_col, id_col2=None, fill_nulls=True, safety_check=False, sort_first=False):
    '''Performs an optimized left merge between two dataframes with null coalescing for common columns'''

    # Optimize input handling
    id_col2 = id_col2 or id_col

    # Fast conversion to LazyFrame without cloning
    df1 = (pl.LazyFrame(df1, nan_to_null=True) if isinstance(df1, (pd.DataFrame, pl.DataFrame))
           else df1 if isinstance(df1, pl.LazyFrame) else pl.LazyFrame(df1, nan_to_null=True))
    df2 = (pl.LazyFrame(df2, nan_to_null=True) if isinstance(df2, (pd.DataFrame, pl.DataFrame))
           else df2 if isinstance(df2, pl.LazyFrame) else pl.LazyFrame(df2, nan_to_null=True))

    if sort_first:
        df1 = df1.sort(id_col)
        df1 = df1.set_sorted(id_col)

    schema1 = df1.collect_schema()
    schema2 = df2.collect_schema()
    if id_col not in schema1:
        raise ValueError(f"Merge key '{id_col}' not found in first dataframe")
    if id_col2 not in schema2:
        raise ValueError(f"Merge key '{id_col2}' not found in second dataframe")

    # Type alignment for merge keys
    id_col_type = schema1[id_col]
    if schema2[id_col2] != id_col_type:
        df2 = df2.with_columns(pl.col(id_col2).cast(id_col_type, strict=False))

    # Identify common columns once
    common_cols = set(schema1.keys()) & set(schema2.keys()) - {id_col, id_col2}

    if fill_nulls and common_cols:
        # Prepare type casting expressions in bulk
        cast_exprs = []
        successful_common_cols = []

        for col in common_cols:
            target_type = schema1[col]
            from_type = schema2[col]
            try:
                if from_type != target_type:
                    cast_exprs.append(pl.col(col).cast(target_type, strict=False))
                else:
                    cast_exprs.append(pl.col(col))
                successful_common_cols.append(col)
            except Exception as e:
                print(f"Warning: Skipping column '{col}': {str(e)}")
                continue

        if successful_common_cols:
            # Bulk type casting
            df2_common = df2.select(cast_exprs + [pl.col(id_col2)])

            # Optimized merge for common columns
            temp_merged = df1.join(
                df2_common,
                left_on=id_col,
                right_on=id_col2,
                coalesce=False,
                how='left'
            )

            # Bulk coalesce operation
            coalesce_exprs = [
                pl.coalesce([pl.col(col), pl.col(f"{col}_right")]).alias(col)
                for col in successful_common_cols
            ]

            # Select all columns in one operation
            other_cols = [col for col in schema1.keys() if col not in successful_common_cols]
            df1 = temp_merged.select(other_cols + coalesce_exprs).unique()

    # Optimize final merge
    df2_unique = df2.drop(list(common_cols)).unique(subset=[id_col2])

    # Final merge with optimized column selection
    merged = df1.join(
        df2_unique,
        left_on=id_col,
        right_on=id_col2,
        how='left'
    ).unique()

    if safety_check:
        counts = await pl.concat([
            merged.select(pl.len().alias('count')),
            df1.select(pl.len().alias('count'))
        ]).collect_async()

        if counts[0, 0] > counts[1, 0]:
            merged = merged.unique(subset=[id_col], keep="first")

    return merged

def sort_by_other_list(list_to_sort, reference_list):
    """Sorts a list based on the position of elements in another list."""
    index_dict = {element: index for index, element in enumerate(reference_list)}
    return sorted(list_to_sort, key=lambda x: index_dict.get(x, float('inf')))

def is_number_repl_isdigit(s):
    """ Returns True if string is a number. """
    return s.replace('.','',1).isdigit()

def coerce_numeric(x, onNaN=None, emptyStringIsZero=False):
    if x is None: return onNaN
    if isinstance(x, (int, float)): return x if not np.isnan(x) else onNaN
    if isinstance(x, str) and is_number_repl_isdigit(x.replace(",","")): return float(x.replace(",",""))
    if isinstance(x, str) and (x.lower() in {'true', 'false', 'y', 'n'}): return int(x.lower() in {'true', 'y'})
    if isinstance(x, str) and (x == ''): return 0 if emptyStringIsZero else onNaN
    return onNaN

def classify_desk(desig_desk: str):
    if desig_desk is None:
        return None

    s = desig_desk.lower()

    if (
            "ig" in s
            or "investment grade" in s
            or "hg" in s
            or "high grade" in s
    ):
        return "IG"

    # HY group
    if (
            "hy" in s
            or "high yield" in s
            or "distressed" in s
            or "illiquid" in s
    ):
        return "HY"

    # EM group
    if (
            "em" in s
            or "japan" in s
            or "singapore" in s
            or "asia" in s
            or "emerging" in s
            or "nja" in s
            or "ja" in s
    ):
        return "EM"

    # Loan
    if "loan" in s:
        return "LOAN"

    return "OTHER"

def flatten_list_single_level(nested_list):
    flattened = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            flattened.extend(sublist)
        else:
            flattened.append(sublist)
    return flattened


class DummyAsyncContextManager:
    async def __aenter__(self):
        return self  # Return the resource to be managed
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False  # Re-raise exceptions


# market
# HOUSE
# REUTERS
# CREDIT_MLCR
# TWUSCDS_BLK
# TWUSCDS_MINI
# BXCZBBG
# BXCYBBG
# TWUSCDS_T2
# TWUSCDS_T1
# BXCXBBG_T2
# BXCXBBG_T1
# BXCXBBG_T0
# ETF_MD
# BGC
# CREDIT_SMM
# GFI
# TULLETT
# ICAP
# BATS_BBG_BXAL_T1
# BATS_BBG_BXAL_T0
# BATS_BBG_BXAL_T2
# BATS_NEPTUNE_AXE_T1
# BATS_BBG_AXE_T1
# BATS_BBG_AXE_T0
# BATS_NEPTUNE_AXE_T0
# CREDIT_SMM_DEALER
# TW_AI_MD
# BBG_BPIPE_CASH
# CHP_ALLMARKETS
# CHP_AXI
# CHP_BLOOMBERG
# CHP_STATS
# TMC
# BONDPOINT
# MACP
# MARKIT
# BBG_BPIPE
# BONDSPRO
# MA_LIVE_MD
# CHP_BEMUBBG_NY
# market
# TRADEWEB_MD
# MACP
# BATS_TW_BXOL_T0
# BATS_TW_BXOL_T1
# BATS_TW_BXOL_T2
# BATS_TW_BXOL_T3
# BATS_F_TW_BXOL_T0
# BATS_F_TW_BXOL_T1
# BATS_F_TW_BXOL_T2
# BATS_F_TW_BXOL_T3
# BATS_TradingScreen_T0
# BATS_TradingScreen_T1
# BATS_TradingScreen_T2
# BATS_TradingScreen_T3
# BATS_BBG_BXOL_T0
# BATS_BBG_BXOL_T1
# BATS_BBG_BXOL_T2
# BATS_BBG_BXOL_T3
# ETF_MD
# BATS_BBG_AXE_T0
# BATS_BBG_AXE_T1
# BATS_BBG_AXE_T2
# BATS_BBG_AXE_T3
# BATS_TW_AXE_T0
# BATS_TW_AXE_T1
# BATS_TW_AXE_T2
# BATS_TW_AXE_T3
# BATS_BondVision_AXE_T0
# BATS_BondVision_AXE_T1
# BATS_BondVision_AXE_T2
# BATS_BondVision_AXE_T3
# BATS_NEPTUNE_AXE_T0
# BATS_NEPTUNE_AXE_T1
# BATS_NEPTUNE_AXE_T2
# BATS_NEPTUNE_AXE_T3
# BBG_BPIPE_CASH_LN
# GFI_CASH_LN
# TRAD_CASH_LN
# TULLETT_CASH_LN
# BGC_CASH_LN
# AMSTEL_CASH_LN
# ICAP_CASH_LN
# MARKIT_LN
# CHP_ALLMARKETS
# CHP_STATS
# CHP_AXI
# CHP_BLOOMBERG
# HDAT_CASH_LN
# CHP_BXEMBBG_LN
#
# market
# CHP_BXCABBG_AP
# CHP_STATS
