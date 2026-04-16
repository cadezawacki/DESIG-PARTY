
import polars as pl
from app.services.loaders.kdb_queries_dev_v3 import *
from app.services.portfolio.optimization_v3 import DataTask, DataLoader, FakeTask, EchoTask, QuickCacheTask

try:
    import re2 as re
except ImportError:
    import re


async def create_task_schema(my_pt, func):
    r = await func(my_pt)
    schema = r.hyper.schema()
    sorted_by_key = dict(sorted(schema.items(), key=lambda item: item))
    return {'"' + k + '"': ('pl.' + str(v)) for k, v in sorted_by_key.items()}


def check_query_log(name):
    with open('app/logs/kdb/query.log', 'r') as f:
        x = f.readlines()
    z = [y for y in x if name in y]
    p = re.compile(r"duration=([0-9]*\.?[0-9]+)")
    data = [y for y in [float(p.search(zz).group(1)) for zz in z if p.search(zz)] if y!=0]
    print(f"min:{min(data)}; mean:{np.mean(data)}; median:{np.median(data)}; max:{max(data)}")


async def compare_task_schema(t1, t2=None):
    from app.services.portfolio.optimization_v3 import _provider_cache_path, _read_provider_cache
    t1_path = _provider_cache_path(t1)
    t2_path = _provider_cache_path(t2) if t2 else None
    t1_cache = await _read_provider_cache(t1_path)
    t2_cache = await _read_provider_cache(t2_path) if t2 else None
    t1_keys = set(t1_cache.get('columns_last', {}).keys())
    t2_keys = set(t2_cache.get('columns_last', {}).keys()) if t2 else None
    if t2 is None:
        print(f"""{"'" + "','".join(sorted(list(t1_keys))) + "'"}""")
    else:
        print(
            f"""
        ## 1) IN {t1} BUT NOT {t2} ##
        {"'" + "','".join(sorted(list(t1_keys.difference(t2_keys)))) + "'"}
        
        ## 2) IN {t2} BUT NOT {t1} ##
        {"'" + ",".join(sorted(list(t2_keys.difference(t1_keys)))) + "'"}
        
        ## 3) Overlap count: {len(t1_keys.intersection(t2_keys))}
        
        """
            )


"""
@dataclass
class DataTask:
    task_name: str
    func: callable  # async function returning pl.DataFrame

    merge_key: Union[str, List[str], Set[str]]
    fromFrame: str = "main"
    toFrame: str = "main"
    
    mergePolicy: str = MERGE_POLICY_DEFAULT  # coalesce_left | coalesce_right | overwrite | overwrite_non_null
    dedupe_right: bool = True # when merging, dedupe the right side? (safer)

    # strict requirements - ALL must be met
    strict_col_requirements: Union[list, set] = field(default_factory=list)
    strict_task_requirements: Union[list, set] = field(default_factory=list)

    # Failed requirements
    failed_task_requirements: Union[list, set] = field(default_factory=list)

    # Outputs from these tasks are ignored
    ignored_tasks: Union[list, set] = field(default_factory=list)
    global_ignore: bool = False

    # expected_col_provides:
    #   - dict: {col: dtype} where dtype can be pl.DataType or "Float64"/"Utf8"/etc
    #   - list/set/tuple: ["colA","colB"] treated as dtype unknown (pl.Null)
    expected_col_provides: dict = field(default_factory=dict)
    
    use_cached_providers: bool = True
    cache_providers: bool = True

    # Data source
    host: Optional[str] = None
    port: Optional[int] = None
    tbl_name: Optional[str] = None
    region: Optional[str] = None

    isTemp: bool = False
    isOptional: bool = False
    backup_tasks: Union[list, set] = field(default_factory=list)
    
    max_retries: int = 1
    retry_delay: float = 0
    broadcast_name: Optional[str] = None

    frameContext: Union[list, set] = field(default_factory=list)
    results: Optional[Any] = None
    kwargs: dict = field(default_factory=dict)

"""

setup_tasks = [
    DataTask(
        task_name="init_values",
        func=init_values,
        broadcast_name="Initializing",
        merge_key='tnum',
        mergePolicy="coalesce_left",
        strict_col_requirements=['id', 'size', 'side', 'quoteType'],
        expected_col_provides={
            'askSize'             : pl.Float64,
            'bidSize'             : pl.Float64,
            'comment'             : pl.String,
            'grossSize'           : pl.Float64,
            'isLocked'            : pl.Int8,
            'isMarked'            : pl.Int8,
            'isReal'              : pl.Int8,
            'manualBidPx'         : pl.Float64,
            'manualMidPx'         : pl.Float64,
            'manualAskPx'         : pl.Float64,
            'manualBidSpd'        : pl.Float64,
            'manualMidSpd'        : pl.Float64,
            'manualAskSpd'        : pl.Float64,
            'manualBidMmy'        : pl.Float64,
            'manualMidMmy'        : pl.Float64,
            'manualAskMmy'        : pl.Float64,
            'manualBidDm'         : pl.Float64,
            'manualMidDm'         : pl.Float64,
            'manualAskDm'         : pl.Float64,
            'manualRefMktOverride': pl.Int8,
            'manualRefMktTime'    : pl.Datetime,
            'manualRefMktUser'    : pl.String,
            'netSize'             : pl.Float64,
            'originalId'          : pl.String,
            'originalQuoteType'   : pl.String,
            'originalSide'        : pl.String,
            'originalSize'        : pl.Float64,
            'portfolioKey'        : pl.String,
            'rfqId'               : pl.String,
            'tnum'                : pl.String,
        },
        use_cached_providers=False
    ),
    DataTask(
        task_name="init_pricing",
        func=init_pricing,
        broadcast_name="Intializing pricing",
        merge_key='tnum',
        mergePolicy="coalesce_left",
        strict_col_requirements=[],
        expected_col_provides={
            'lastAdminEditNewLevel'      : pl.Float64,
            'lastAdminEditTimestamp'     : pl.String,
            'lastAdminEditUser'          : pl.String,
            'lastComputeTimstamp'        : pl.String,
            'lastEditQuoteType'          : pl.String,
            'lastEditSource'             : pl.String,
            'lastEditTime'               : pl.String,
            'lastEditTraceId'            : pl.String,
            'lastEditUser'               : pl.String,
            'lastTraderEditNewLevel'     : pl.Float64,
            'lastTraderEditTimestamp'    : pl.String,
            'lastTraderEditUser'         : pl.String,
            'newLevel'                   : pl.Float64,
            'newLevelDm'                 : pl.Float64,
            'newLevelMmy'                : pl.Float64,
            'newLevelPx'                 : pl.Float64,
            'newLevelSpd'                : pl.Float64,
            'newLevelYld'                : pl.Float64,
            'relativeSkewTargetMkt'      : pl.String,
            'relativeSkewTargetQuoteType': pl.String,
            'relativeSkewTargetSide'     : pl.String,
            'relativeSkewValue'          : pl.Float64,
            'skewType'                   : pl.Float64,
            'tnum'                       : pl.String
        },
        use_cached_providers=False
    )
]

ref_data_tasks = [
    DataTask(
        task_name='kdb_bond_pano_static',
        broadcast_name="Static data from panoproxy",
        func=kdb_bond_pano_static,
        merge_key='id',
        isOptional=True,
        critical_columns=[
            'accrualMethod', 'amountIssued', 'amountOutstanding', 'country', 'coupon', 'couponType', 'currency',
            'cusip', 'description', 'esmi', 'esmp', 'isCallable', 'isConvertible', 'isin', 'isRegS', 'isRule144A',
            'issueDate', 'issuerCountry', 'issuerIndustryGroup', 'issuerIndustrySector', 'issuerIndustrySubGroup',
            'issuerName', 'isWhenIssued', 'maturityDate', 'minDenomination', 'nextCouponDate', 'sym', 'ticker'
        ],
        use_cached_providers=True,
    ),
    DataTask(
        task_name='kdb_bond_static_data',
        broadcast_name="Bond Static",
        func=kdb_bond_static_data,
        merge_key='id',
        isOptional=True,
        critical_columns=[
            'accrualMethod', 'amountIssued', 'amountOutstanding', 'country', 'coupon', 'currency', 'cusip',
            'description', 'esmi', 'esmp', 'isCallable', 'isCalled', 'isConvertible', 'isin', 'isRegS', 'isRule144A',
            'issueDate', 'issuerCountry', 'issuerIndustryGroup', 'issuerIndustrySector', 'issuerIndustrySubGroup',
            'issuerName', 'isWhenIssued', 'maturityDate', 'minDenomination', 'sym', 'ticker'
        ],
        use_cached_providers=True,
    ),
    DataTask(
        task_name='kdb_series_static',
        broadcast_name="Bond Static",
        func=kdb_series_static,
        merge_key='isin',
        isOptional=True,
        max_retries=0,
        critical_columns=[
            'esmp', 'cusip', 'description', 'ticker', 'issueDate', 'maturityDate',
            'currency', 'coupon', 'isRule144A', 'isRegS', 'nextCallDate'
        ],
        use_cached_providers=True
    ),
    DataTask(
        task_name='kdb_bond_static_data_eu',
        broadcast_name="Bond Static",
        func=kdb_bond_static_data,
        merge_key='id',
        kwargs={"region": "EU"},
        isOptional=True,
        critical_columns=[
            'accrualMethod', 'amountIssued', 'amountOutstanding', 'country', 'coupon', 'currency', 'cusip',
            'description', 'esmi', 'esmp', 'isCallable', 'isCalled', 'isConvertible', 'isin', 'isRegS', 'isRule144A',
            'issueDate', 'issuerCountry', 'issuerIndustryGroup', 'issuerIndustrySector', 'issuerIndustrySubGroup',
            'issuerName', 'isWhenIssued', 'maturityDate', 'minDenomination', 'sym', 'ticker'
        ],
        use_cached_providers=True,
    ),
    DataTask(
        task_name='house_eu_benchmark',
        func=house_eu_benchmark,
        merge_key='isin',
        isOptional=True,
        max_retries=0,
        critical_columns=[
            'houseEuBenchmarkIsin'
        ],
        use_cached_providers=False,
        expected_col_provides={
            'houseEuBenchmarkIsin': pl.String,
        }
    ),
    DataTask(
        task_name='instrument_static',
        broadcast_name="Bond Static",
        func=instrument_static,
        merge_key='id',
        isOptional=True,
        critical_columns=[
            'accrualMethod', 'amountIssued', 'amountOutstanding', 'country', 'countryOfRisk', 'coupon', 'currency',
            'cusip', 'daysToSettle', 'description', 'esmp', 'industryGroup', 'industrySector', 'industrySubGroup',
            'isCallable', 'isCalled', 'isConvertible', 'isCovered', 'isFloater', 'isin', 'isPerpetual',
            'isRegS', 'isRule144A', 'issuerIndustry', 'issuerIndustryGroup', 'issuerIndustrySector',
            'issuerIndustrySubGroup', 'issuerName', 'maturityDate', 'minDenomination', 'productType',
            'pseudoWorkoutDate', 'ratingCombined', 'sym', 'ticker', 'ultimateParentCountryOfRisk'
        ],
        use_cached_providers=True,
    ),
    DataTask(
        task_name='kdb_smad_bondStaticData',
        func=kdb_smad_bondStaticData,
        kwargs={"region": "EU"},
        merge_key='id',
        isOptional=True,
        critical_columns=[
            'accrualMethod', 'amountIssued', 'amountOutstanding', 'country', 'countryOfRisk', 'coupon',
            'couponFrequency', 'couponType', 'currency', 'cusip', 'daysToSettle', 'description', 'esmp',
            'industryGroup', 'industrySector', 'industrySubGroup', 'isCallable', 'isCalled', 'isConvertible',
            'isCovered', 'isFloater', 'isInDefault', 'isLitigationDefault', 'isMakeWholeCall', 'isPerpetual',
            'isPrivatePlacement', 'isPutable', 'isRegS', 'isRule144A', 'isSinkable', 'isStructuredNote',
            'isSubordinated', 'issueDate', 'issuerIndustry', 'issuerIndustryGroup', 'issuerIndustrySector',
            'issuerIndustrySubGroup', 'issuerName', 'isWhenIssued', 'maturityDate', 'maturityType', 'minDenomination',
            'minIncrement', 'pseudoWorkoutDate', 'ratingCombined', 'sym', 'ticker', 'ultimateParentCountryOfRisk'
        ],
        use_cached_providers=True,
    ),
    DataTask(
        task_name='kdb_smad_bondStaticData_pano_eu',
        func=kdb_smad_bondStaticData_pano,
        kwargs={"region": "EU"},
        merge_key='id',
        isOptional=True,
        critical_columns=[
            'accrualMethod', 'amountIssued', 'amountOutstanding', 'country', 'countryOfRisk', 'coupon',
            'couponFrequency', 'couponType', 'currency', 'cusip', 'daysToSettle', 'description', 'esmp',
            'industryGroup', 'industrySector', 'industrySubGroup', 'isCallable', 'isCalled', 'isConvertible',
            'isCovered', 'isFloater', 'isInDefault', 'isLitigationDefault', 'isMakeWholeCall', 'isPerpetual',
            'isPrivatePlacement', 'isPutable', 'isRegS', 'isRule144A', 'isSinkable', 'isStructuredNote',
            'isSubordinated', 'issueDate', 'issuerIndustry', 'issuerIndustryGroup', 'issuerIndustrySector',
            'issuerIndustrySubGroup', 'issuerName', 'isWhenIssued', 'maturityDate', 'maturityType',
            'minDenomination', 'minIncrement', 'pseudoWorkoutDate', 'ratingCombined', 'sym', 'ticker',
            'ultimateParentCountryOfRisk'
        ],
        use_cached_providers=True
    ),
    DataTask(
        task_name='kdb_smad_bondStaticData_pano_us',
        func=kdb_smad_bondStaticData_pano,
        kwargs={"region": "US"},
        merge_key='id',
        isOptional=True,
        critical_columns=[
            'accrualMethod', 'amountIssued', 'amountOutstanding', 'country', 'countryOfRisk', 'coupon',
            'couponFrequency', 'couponType', 'currency', 'cusip', 'daysToSettle', 'description', 'esmp',
            'industryGroup', 'industrySector', 'industrySubGroup', 'isCallable', 'isCalled', 'isConvertible',
            'isCovered', 'isFloater', 'isInDefault', 'isLitigationDefault', 'isMakeWholeCall', 'isPerpetual',
            'isPrivatePlacement', 'isPutable', 'isRegS', 'isRule144A', 'isSinkable', 'isStructuredNote',
            'isSubordinated', 'issueDate', 'issuerIndustry', 'issuerIndustryGroup', 'issuerIndustrySector',
            'issuerIndustrySubGroup', 'issuerName', 'isWhenIssued', 'maturityDate', 'maturityType', 'minDenomination',
            'minIncrement', 'pseudoWorkoutDate', 'ratingCombined', 'sym', 'ticker', 'ultimateParentCountryOfRisk'
        ],
        use_cached_providers=True
    ),
    DataTask(
        task_name='settle_dater',
        func=settle_dater,
        merge_key='isin',
        strict_col_requirements=['isRegS', 'isRule144A', 'daysToSettle'],
        expected_col_provides={
            'settleDate'        : pl.Date,
            'standardSettleDate': pl.Date
        }
    ),
    DataTask(
        task_name='kdb_bbg_internalDomBond',
        broadcast_name="Bond Static BBG",
        func=kdb_bbg_domBond,
        strict_col_requirements=[],
        merge_key='id',
        isOptional=True,
        critical_columns=[
            'convexity', 'country', 'coupon', 'currency', 'cusip', 'duration', 'isin', 'issrClass', 'issuerIndustry',
            'issuerIndustryGroup', 'issuerName', 'issuerSector', 'issuerSubIndustry', 'maturityDate', 'ticker'
        ]
    ),
    DataTask(
        task_name='kdb_eag_bond',
        broadcast_name="Bond Static BBG",
        func=kdb_eag_bond,
        strict_col_requirements=[],
        merge_key='id',
        isOptional=True,
        critical_columns=[
            'convexity', 'country', 'coupon', 'currency', 'cusip', 'duration', 'isin', 'issrClass', 'issuerIndustry',
            'issuerIndustryGroup', 'issuerName', 'issuerSector', 'issuerSubIndustry', 'maturityDate', 'ticker'
        ]
    ),
    DataTask(
        task_name='kdb_bbg_internalEmgBond',
        broadcast_name="Bond Static BBG Emg",
        func=kdb_bbg_emgBond,
        merge_key='id',
        isOptional=True,
        critical_columns=[
            'convexity', 'country', 'coupon', 'currency', 'cusip', 'duration', 'isin', 'issrClass', 'issuerIndustry',
            'issuerIndustryGroup', 'issuerName', 'issuerSector', 'issuerSubIndustry', 'maturityDate', 'ticker'
        ]
    ),
    DataTask(
        task_name='kdb_smad_bondLevelData',
        broadcast_name="SMAD Static Data",
        func=kdb_smad_bondLevelData,
        merge_key='id',
    ),
    DataTask(
        task_name='trade_to_flag',
        broadcast_name="Trade To Flag",
        func=trade_to_flag,
        strict_col_requirements=["isCallable", "isCalled", "isMakeWholeCall", "isHybrid", "isPerpetual"],
        merge_key='isin',
    ),
    DataTask(
        task_name='map_gic_sector',
        broadcast_name="GICS Sector",
        func=map_gic_sector,
        strict_col_requirements=[
            'issuerGicsIndustry', 'issuerGicsIndustryGroup', 'issuerGicsSector', 'issuerGicsSubIndustry'
        ],
        merge_key='isin',
        expected_col_provides={
            "gicsSector"       : pl.String,
            "gicsIndustryGroup": pl.String,
            "gicsIndustry"     : pl.String,
            "gicsSubIndustry"  : pl.String,
        },
        use_cached_providers=False,
    ),
    DataTask(
        task_name='ref_data_transforms',
        broadcast_name="Cleaning Ref Data",
        func=ref_data_transforms,
        strict_col_requirements=[
            'bondType', 'description', 'industrySector', 'isin', 'isPerpetual', 'issueDate', 'issuerName',
            'maturityDate', 'pseudoWorkoutDate', 'ratingCombined', 'ratingMoody', 'ratingSandP', 'shortDescription',
            'ticker'
        ],
        merge_key='tnum',
        mergePolicy="coalesce_right",
        isFinalizer=True
    ),

    EchoTask(
        task_name='ref_data_desig_us',
        broadcast_name="Desigs",
        strict_col_requirements=[
            'isin', 'currency', 'bvalSubAssetClass', 'bvalAssetClass', 'regionBarclaysRegion', 'ratingCombined'
        ],
        columns=[
            'isin', 'currency', 'bvalSubAssetClass', 'bvalAssetClass', 'regionBarclaysRegion', 'ratingCombined'
        ],
        strict_task_requirements=['_pano_positions_us_desig'],
        merge_key='isin',
        toFrame='desig_us'
    ),
    DataTask(
        task_name='desig_fast_path_us',
        broadcast_name="Desigs",
        func=desig_fast_path,
        merge_key='',
        strict_col_requirements=[
            '_isFungeDesig', '_isTrueDesig', '_usRunzSenderLastName', 'bookId', 'bookRegion', 'bvalAssetClass',
            'bvalSubAssetClass', 'currency', 'deskAsset', 'houseUsRefreshTime', 'isDesig', 'isin', 'netPosition',
            'ratingCombined', 'regionBarclaysRegion', 'traderId', 'usIsRunzAskAxe', 'usIsRunzBidAxe', 'isMuni'
        ],
        fromFrame='desig_us',
        toFrame='desig_us_scored',
        kwargs={"region": "US"}
    ),
    EchoTask(
        task_name='ref_data_desig_eu',
        broadcast_name="Desigs",
        strict_col_requirements=[
            'isin', 'currency', 'bvalSubAssetClass', 'bvalAssetClass', 'regionBarclaysRegion', 'ratingCombined'
        ],
        columns=[
            'isin', 'currency', 'bvalSubAssetClass', 'bvalAssetClass', 'regionBarclaysRegion', 'ratingCombined'
        ],
        strict_task_requirements=['ref_data_transforms', '_pano_positions_eu_desig'],
        merge_key='isin',
        toFrame='desig_eu'
    ),
    DataTask(
        task_name='desig_fast_path_eu',
        broadcast_name="Desigs",
        func=desig_fast_path,
        merge_key='',
        strict_col_requirements=[
            '_euRunzSenderLastName', '_isFungeDesig', '_isTrueDesig', 'bookId', 'bookRegion', 'bvalAssetClass',
            'bvalSubAssetClass', 'currency', 'deskAsset', 'euIsRunzAskAxe', 'euIsRunzBidAxe', 'houseEuRefreshTime',
            'isDesig', 'isin', 'netPosition', 'ratingCombined', 'regionBarclaysRegion', 'traderId', 'isMuni'
        ],
        fromFrame='desig_eu',
        toFrame='desig_eu_scored',
        kwargs={"region": "EU"}
    ),
    EchoTask(
        task_name='ref_data_desig_sgp',
        broadcast_name="Desigs",
        strict_col_requirements=[
            'isin', 'currency', 'bvalSubAssetClass', 'bvalAssetClass', 'regionBarclaysRegion', 'ratingCombined'
        ],
        columns=['isin', 'currency', 'bvalSubAssetClass', 'bvalAssetClass', 'regionBarclaysRegion', 'ratingCombined'],
        strict_task_requirements=['ref_data_transforms', '_pano_positions_sgp_desig'],
        merge_key='isin',
        toFrame='desig_sgp'
    ),
    DataTask(
        task_name='desig_fast_path_sgp',
        broadcast_name="Desigs",
        func=desig_fast_path,
        merge_key='',
        strict_col_requirements=[
            '_isFungeDesig', '_isTrueDesig', '_sgpRunzSenderLastName', 'bookId', 'bookRegion', 'bvalAssetClass',
            'bvalSubAssetClass', 'currency', 'deskAsset', 'houseSgpRefreshTime', 'isDesig', 'isin', 'netPosition',
            'ratingCombined', 'regionBarclaysRegion', 'sgpIsRunzAskAxe', 'sgpIsRunzBidAxe', 'traderId', 'isMuni'
        ],
        fromFrame='desig_sgp',
        toFrame='desig_sgp_scored',
        kwargs={"region": "SGP"}
    ),
    DataTask(
        task_name='desig_fast_join',
        broadcast_name="Desigs",
        func=desig_fast_join,
        merge_key='isin',
        toFrame='',
        strict_task_requirements=[
            'desig_fast_path_us', 'desig_fast_path_eu', 'desig_fast_path_sgp'
        ],
        frameContext=['desig_us_scored', 'desig_eu_scored', 'desig_sgp_scored'],
    ),
    DataTask(
        task_name='desig_waterfall_portfolio',
        broadcast_name="Desigs - Waterfall",
        func=desig_waterfall_portfolio,
        merge_key='isin',
        strict_task_requirements=['desig_fast_join', 'ref_data_transforms'],
        frameContext=['desig_joined'],
        strict_col_requirements=[
            'ticker', 'ratingAssetClass', 'issuerCountry', 'yieldCurvePosition',
            'industryGroup', 'currency',
        ],
        use_cached_providers=False,
        expected_col_provides={
            'desigBookId'    : pl.String,
            'desigTraderId'  : pl.String,
            'desigName'      : pl.String,
            'desigRegion'    : pl.String,
            'desigConfidence': pl.String,
            'desigGapRatio'  : pl.Float64,
            'desigScore'     : pl.Float64,
            'deskAsset'      : pl.String,
            'topTradersIds'  : pl.List(pl.String),
            'topScores'      : pl.List(pl.Float64),
            'topBooks'       : pl.List(pl.String),
            'topNames'       : pl.List(pl.String),
            'topRegions'     : pl.List(pl.String),
        },
    ),
    DataTask(
        task_name='desig_splitter_high',
        broadcast_name="Desigs",
        func=desig_splitter_high,
        merge_key='isin',
        strict_task_requirements=['desig_waterfall_portfolio'],
        fromFrame='desig_joined',
        use_cached_providers=False,
        expected_col_provides={
            'desigBookId'    : pl.String,
            'desigTraderId'  : pl.String,
            'desigName'      : pl.String,
            'desigRegion'    : pl.String,
            'desigConfidence': pl.String,
            'desigGapRatio'  : pl.Float64,
            'desigScore'     : pl.Float64,
            'deskAsset'      : pl.String,
        },
    ),
    DataTask(
        task_name='desig_splitter_low',
        broadcast_name="Desigs",
        func=desig_splitter_low,
        merge_key='isin',
        strict_task_requirements=['desig_waterfall_portfolio'],
        fromFrame='desig_joined',
        toFrame='',
        use_cached_providers=False,
        expected_col_provides={
            'desigBookId'    : pl.String,
            'desigTraderId'  : pl.String,
            'desigName'      : pl.String,
            'desigRegion'    : pl.String,
            'desigConfidence': pl.String,
            'desigGapRatio'  : pl.Float64,
            'desigScore'     : pl.Float64,
            'deskAsset'      : pl.String,
            'topTradersIds': pl.List(pl.String),
            'topScores'    : pl.List(pl.Float64),
            'topBooks'     : pl.List(pl.String),
            'topNames'     : pl.List(pl.String),
            'topRegions'   : pl.List(pl.String),
        },
    ),
    # todo: expand universe using LOW, for not just join low
    # DataTask(
    #     task_name='desig_expander',
    #     broadcast_name="Desigs",
    #     func=desig_expander,
    #     merge_key='isin',
    #     strict_task_requirements=['desig_splitter_low'],
    #     fromFrame='main',
    #     frameContext=['desig_low'],
    #     toFrame='desig_low',
    #     isOptional=True,
    #     critical_columns=['desigName']
    # ),
]

funge_tasks = [
    DataTask(
        task_name='fungible_series',
        broadcast_name="Fungible Bonds",
        func=fungible_series,
        strict_col_requirements=['isin'],
        expected_col_provides={
            "fungibleSeries": pl.String,
            "fungibleIsin"  : pl.String
        },
        use_cached_providers=False,
        merge_key='',
        toFrame="funges",
    ),
    DataTask(
        task_name='fungible_enhance',
        broadcast_name="Fungible Bonds Enhance",
        func=fungible_enhance,
        strict_col_requirements=['fungibleIsin'],
        expected_col_provides={
            "fungibleSym"        : pl.String,
            "fungibleDescription": pl.String
        },
        strict_task_requirements=['fungible_series'],
        use_cached_providers=False,
        merge_key='fungibleIsin',
        fromFrame="funges",
        toFrame="funges",
    ),
    DataTask(
        task_name='fungible_join',
        broadcast_name="Fungible Bonds Join",
        func=fungible_join,
        strict_col_requirements=['fungibleSym'],
        strict_task_requirements=['fungible_series', 'fungible_enhance'],
        use_cached_providers=False,
        expected_col_provides={
            "fungibleIsin"       : pl.String,
            "fungibleSeries"     : pl.String,
            "fungibleSym"        : pl.String,
            "fungibleDescription": pl.String
        },
        merge_key='isin',
        fromFrame="funges",
        toFrame="main",
    ),
]

flag_tasks = [
    DataTask(
        task_name='pano_restricted_list',
        func=pano_restricted_list,
        broadcast_name="Restricted List",
        merge_key='esmi',
        strict_col_requirements=['esmi'],
        use_cached_providers=False,
        expected_col_provides={
            "restrictedCode" : pl.String,
            "restrictionTier": pl.String,
        },
    ),
    DataTask(
        task_name='coop_bond',
        func=coop_bond,
        broadcast_name="Checking COOP Bonds",
        merge_key='isin',
        use_cached_providers=False,
        expected_col_provides={
            "isCoop": pl.Int8,
        },
    ),
    DataTask(
        task_name='corp_action',
        func=corp_action,
        broadcast_name="Checking Corp Actions",
        merge_key='isin',
        use_cached_providers=False,
        expected_col_provides={
            "isCorpAction"  : pl.Int8,
            'corpActionType': pl.String,
        },
    ),
    DataTask(
        task_name='bond_targets',
        func=bond_targets,
        merge_key='',
        toFrame='bond_targets'
    ),
    DataTask(
        task_name='pano_dnt_warnings_by_ticker',
        func=pano_dnt_warnings_by_ticker,
        broadcast_name="DNT",
        strict_col_requirements=['ticker', 'isin'],
        use_cached_providers=False,
        expected_col_provides={
            "dntComment"   : pl.String,
            "dntUpdateTime": pl.String,
        },
        merge_key='ticker',
    ),
]

risk_metric_tasks = [
    DataTask(
        task_name='pano_dv01',
        broadcast_name="DV01",
        func=pano_dv01,
        isOptional=True,
        critical_columns=['unitDv01', 'unitCs01Pct'],
        strict_col_requirements=[],
        use_cached_providers=False,
        expected_col_provides={
            "unitCs01Pct": pl.Float64,
            "unitDv01"   : pl.Float64,
        },
        merge_key='isin',
    ),
    DataTask(
        task_name='approx_dv01',
        broadcast_name="Approx DV01",
        func=approx_dv01,
        isOptional=True,
        strict_col_requirements=['bvalMidPx', 'macpMidPx', 'duration', 'unitAccrued'],
        strict_task_requirements=['pano_dv01'],
        use_cached_providers=False,
        critical_columns=['unitDv01'],
        ignored_tasks=['dv01_to_duration'],
        expected_col_provides={
            "unitDv01": pl.Float64,
        },
        merge_key='isin',
    ),
    DataTask(
        task_name='interp_dv01',
        broadcast_name="Approx DV01",
        func=interpolated_dv01,
        isOptional=True,
        strict_col_requirements=['duration'],
        strict_task_requirements=['pano_dv01', 'approx_dv01', 'benchmark_join', 'bk2_risk'],
        use_cached_providers=False,
        frameContext=["benchmarks"],
        critical_columns=['unitDv01'],
        ignored_tasks=['dv01_to_duration'],
        expected_col_provides={
            "unitDv01": pl.Float64,
        },
        merge_key='isin',
    ),
    DataTask(
        task_name='bk2_risk',
        broadcast_name="BK2 Risk Measures",
        func=bk2_risk,
        strict_col_requirements=['isin'],
        strict_task_requirements=['pano_dv01', 'approx_dv01'],
        use_cached_providers=False,
        expected_col_provides={
            "unitCs01Pct": pl.Float64,
            "unitCs01"   : pl.Float64,
            "unitDv01"   : pl.Float64,  # A bad estimate of dv01, use as a last resort
            "bk2BidPx"   : pl.Float64,
            "bk2MidPx"   : pl.Float64,
            "bk2AskPx"   : pl.Float64,
        },
        merge_key='isin',
    ),
    DataTask(
        task_name='risk_transforms',
        broadcast_name="Risk Transforms",
        func=risk_transforms,
        strict_col_requirements=[
            'axeFullAskSize', 'axeFullBidSize', 'grossSize', 'netAlgoPosition', 'netDeskPosition', 'netFirmPosition',
            'netStrategyPosition', 'side', 'signalFlag', 'unitAccrued', 'unitCs01', 'unitCs01Pct', 'unitDv01'
        ],
        merge_key='tnum',
    ),
    DataTask(
        task_name='cs01_estimate',
        broadcast_name="CS01 Estimate",
        func=cs01_estimate,
        strict_col_requirements=['yrsToMaturity', 'unitDv01'],
        expected_col_provides={
            "unitCs01": pl.Float64
        },
        merge_key='isin',
    ),
    DataTask(
        task_name='holding_time_eu',
        broadcast_name="Holding Time",
        func=holding_time_eu,
        strict_col_requirements=['netSize'],
        use_cached_providers=False,
        expected_col_provides={
            "eht": pl.Float64
        },
        merge_key='isin',
    ),
    DataTask(
        task_name='landmines_eu',
        broadcast_name="Landmines",
        func=landmines_eu,
        use_cached_providers=False,
        expected_col_provides={
            "lowLandmineWarning"   : pl.String,
            "mediumLandmineWarning": pl.String,
            "highLandmineWarning"  : pl.String,
        },
        merge_key='isin',
    ),
    DataTask(
        task_name="etf_overlap",
        broadcast_name="ETF Overlap",
        func=etf_overlap,
        merge_key='isin',
    ),
    DataTask(
        task_name="funding_rate",
        broadcast_name="Funding Rate",
        func=funding_rate,
        merge_key='isin',
    ),
    DataTask(
        task_name="dv01_to_duration",
        broadcast_name="Duration",
        func=estimate_duration_from_dv01,
        merge_key='isin',
        isOptional=True,
        critical_columns=['duration'],
        strict_col_requirements=['duration', 'unitDv01', 'bvalMidPx'],
        ignored_tasks=['approx_dv01', 'bk2_risk', 'interp_dv01'],
        use_cached_providers=False,
        expected_col_provides={
            "duration": pl.Float64,
        },
        isFinalizer=True
    ),
    DataTask(
        task_name="rough_maturity_to_duration_polars",
        broadcast_name="Duration",
        func=rough_maturity_to_duration_polars,
        merge_key='isin',
        isOptional=True,
        critical_columns=['duration'],
        strict_col_requirements=[
            'accrualMethod', 'calledDate', 'coupon', 'couponFrequency', 'duration', 'isFloater', 'isHybrid',
            'isPerpetual', 'isVariable', 'maturityDate', 'nextCallDate', 'nextCouponDate', 'pseudoWorkoutDate'
        ],
        strict_task_requirements=['dv01_to_duration'],
        use_cached_providers=False,
        expected_col_provides={
            "duration": pl.Float64,
        },
        isFinalizer=True
    ),
    DataTask(
        task_name="cds_basis",
        broadcast_name="CDS Basis",
        func=cds_basis,
        merge_key='sym',
        strict_col_requirements=['ticker', 'sym'],
        strict_task_requirements=['quote_bval_non_dm'],
        use_cached_providers=False,
        global_ignore=True,  # fallback ticker
        max_retries=0,
        expected_col_provides={
            "cdsBasisToWorst": pl.Float64,
            "cdsCurve"       : pl.String,
            "cdsParSpdW"     : pl.Float64,
            "cdsTicker"      : pl.String,
            "ticker"         : pl.String
        },
    ),
    DataTask(
        task_name="cds_percentiles",
        broadcast_name="Percentiles",
        func=cds_percentiles,
        merge_key='sym',
        max_retries=0,
        use_cached_providers=False,
        expected_col_provides={
            "maxRangeCdsBasis"                    : pl.Float64,
            "minRangeCdsBasis"                    : pl.Float64,
            "rangeCdsBasisPercentile"             : pl.Float64,
            "rangeCdsBasisPercentileShiftOver1D"  : pl.Float64,
            "rangeCdsBasisPercentileShiftOver7D"  : pl.Float64,
            "rangeCdsBasisPercentileShiftOver30D" : pl.Float64,
            "rangeCdsBasisPercentileShiftOver90D" : pl.Float64,
            "rangeCdsBasisPercentileShiftOver365D": pl.Float64
        },
    ),
]

liq_score_tasks = [
    DataTask(
        task_name='internal_liquidity_score',
        broadcast_name="Internal Liq Scores",
        func=internal_liquidity_score,
        use_cached_providers=False,
        strict_col_requirements=['netSize'],
        expected_col_provides={
            "blsLiqScore": pl.Float64,
            "dkLiqScore" : pl.Float64
        },
        merge_key='isin',
    ),
    QuickCacheTask(
        task_name='macp_liq_score',
        broadcast_name="MACP LiqScore",
        func=macp_liqScore,
        merge_key='isin',
    ),
    DataTask(
        task_name='macp_liq_score_eu',
        broadcast_name="MACP LiqScore",
        func=macp_liq_score_eu,
        merge_key='isin',
        isOptional=True,
        critical_columns=['macpLiqScore'],
        use_cached_providers=False,
        strict_task_requirements=['quote_macp'],
        expected_col_provides={
            "macpLiqScore": pl.Float64,
        },
    ),
    QuickCacheTask(
        task_name='lcs',
        broadcast_name="LCS LiqScore",
        strict_task_requirements=['pano_positions_us', 'pano_positions_eu', 'pano_positions_sgp'],
        func=lcs,
        merge_key='isin',
    ),
    DataTask(
        task_name='cz_liq_score',
        broadcast_name="CZ Liq Scores",
        func=cz_liq_score,
        use_cached_providers=False,
        strict_col_requirements=[
            'allqMidPx', 'amountIssued', 'amountOutstanding', 'bvalMidPx', 'bvalSubAssetClass', 'cbbtAskPx',
            'cbbtAskSpd', 'cbbtBidPx', 'cbbtBidSpd', 'cbbtMidPx', 'cdsBasisToWorst', 'cdsParSpdW', 'currency',
            'desigRegion', 'deskAsset', 'duration', 'gicsIndustryGroup', 'gicsSector', 'gicsSubIndustry',
            'houseAskPx', 'houseAskSpd', 'houseBidPx', 'houseBidSpd', 'houseEuRefreshTime', 'houseMidPx',
            'houseSgpRefreshTime', 'houseUsRefreshTime', 'idcMidPx', 'inEtfAgg', 'inEtfEmb', 'inEtfHyg',
            'inEtfIemb', 'inEtfIgib', 'inEtfIglb', 'inEtfIgsb', 'inEtfJnk', 'inEtfLqd',
            'inEtfSjnk', 'inEtfSpab', 'inEtfSpib', 'inEtfSplb', 'inEtfSpsb', 'inEtfUshy', 'inEtfUsig', 'inEtfVclt',
            'isCalled', 'isConvertible', 'isDtcEligible', 'isEmAlgoEligible', 'isFloater', 'isHyAlgoEligible',
            'isIgAlgoEligible', 'isin', 'isNewIssue', 'isPerpetual', 'isRegS', 'isRule144A', 'issuerIndustry',
            'issuerSector', 'issuerSubIndustry', 'macpMidPx', 'maturityDate', 'netFirmPosition', 'ratingCombined',
            'traceCount1D', 'traceCount10D', 'traceCount30D', 'traceCount60D', 'traceCount90D', 'traceVolume1D',
            'traceVolume10D', 'traceVolume30D', 'traceVolume60D', 'traceVolume90D', 'yrsSinceIssuance', 'yrsToMaturity'
        ],
        expected_col_provides={
            "czLiqScore": pl.Float64,
        },
        merge_key='isin',
    ),
    DataTask(
        task_name='liq_aggregate',
        broadcast_name="Liq Scores",
        func=agg_liquidity_score,
        use_cached_providers=False,
        strict_task_requirements=['cz_liq_score'],
        strict_col_requirements=[
            'blsLiqScore', 'czLiqScore', 'macpLiqScore', 'dkLiqScore', 'lqaLiqScore', 'mlcrLiqScore', 'muniLiqScore',
            'smadLiqScore', 'idcLiqScore'
        ],
        expected_col_provides={
            "liqScore"   : pl.Float64,
            'avgLiqScore': pl.Float64
        },
        merge_key='isin',
    ),
]

region_tasks = [
    DataTask(
        task_name='isin_countries',
        broadcast_name="ISIN Countries",
        func=isin_countries,
        isTemp=True,  # remove this column before persisting
        use_cached_providers=False,
        expected_col_provides={
            "_isinCountry": pl.String,
        },
        merge_key='isin',
    ),
    DataTask(
        task_name='map_regions_quick',
        broadcast_name="Mapping regions",
        func=map_regions,
        strict_col_requirements=["countryOfRisk", "country"],
        kwargs={'country_columns': ["countryOfRisk", "country"]},
        merge_key='isin',
    ),
    DataTask(
        task_name='map_regions_deep',
        broadcast_name="Mapping regions",
        func=map_regions,
        isOptional=True,
        strict_col_requirements=[
            '_isinCountry', 'issuerCountry', 'ultimateParentCountryOfRisk', "country", "countryOfRisk"
        ],
        kwargs={
            'country_columns': [
                '_isinCountry', 'countryOfRisk', 'issuerCountry', 'ultimateParentCountryOfRisk', "country"
            ]
        },
        critical_columns=['regionBarclaysDesk'],
        merge_key='isin',
    )
]

muni_tasks = [
    DataTask(
        task_name='muni_min_increment',
        broadcast_name="Muni Min/Incr",
        func=muni_min_increment,
        isOptional=True,
        critical_columns=['minDenomination', 'minIncrement', 'country'],
        use_cached_providers=False,
        expected_col_provides={
            'country'        : pl.String,
            'isMuni'         : pl.Int8,
            'minDenomination': pl.Float64,
            'minIncrement'   : pl.Float64,
            'productType'    : pl.String,
        },
        merge_key='cusip'
    ),
    DataTask(
        task_name='kdb_muni_data',
        broadcast_name="Muni Static",
        func=kdb_muni_data,
        isOptional=True,
        critical_columns=[
            'amountIssued', 'amountOutstanding', 'country', 'currency', 'cusip', 'description', 'esmi',
            'esmp', 'isin', 'maturityDate', 'sym'
        ],
        merge_key='id'
    ),
    DataTask(
        task_name='muni_direct',
        broadcast_name="Muni Risk",
        func=muni_direct,
        strict_col_requirements=['isin'],
        isOptional=True,
        critical_columns=[
            'convexity', 'coupon', 'duration', 'issuerName', 'maturityDate', 'unitAccrued'
        ],
        merge_key='isin'
    ),
    DataTask(
        task_name='muni_filler',
        broadcast_name="Muni Filler",
        func=muni_filler,
        strict_col_requirements=[],
        strict_task_requirements=['kdb_muni_data', 'muni_direct'],
        use_cached_providers=False,
        expected_col_provides={
            'tnum'  : pl.String,
            'isMuni': pl.Int8
        },
        merge_key='tnum'
    ),
    DataTask(
        task_name='muni_ticker',
        broadcast_name="Muni Ticker",
        func=muni_ticker,
        global_ignore=True,
        strict_col_requirements=[
            'coupon', 'maturityDate', 'isMuni', 'description', 'ticker', 'issuerName'
        ],
        strict_task_requirements=['kdb_muni_data', 'muni_direct', 'muni_filler'],
        merge_key='isin',
    ),
]

_REFRESH_RE = re.compile(r'^(.+)RefreshTime$')
refresh_time_based_on_col = _REFRESH_RE
_SIDES = ("Bid", "Mid", "Ask")
_METRICS = ("Px", "Spd", "Yld", "Mmy", "Dm")


def based_on_refresh_time_levels(bo_col: str) -> list[str]:
    market = _REFRESH_RE.fullmatch(bo_col).group(1)
    return [f"{market}{side}{metric}"
        for side in _SIDES
        for metric in _METRICS]


quote_tasks = [
    DataTask(
        task_name='quote_bval_non_dm',
        broadcast_name="BVAL Quotes",
        func=quote_bval_non_dm,
        merge_key='isin',
        based_on_col='bvalRefreshTime',
    ),
    DataTask(
        task_name='quote_bval_full',
        broadcast_name="BVAL Quotes",
        func=quote_bval_full,
        merge_key='isin',
        based_on_col='bvalRefreshTime',
    ),
    DataTask(
        task_name='bval_yest',
        func=bval_yest,
        merge_key='isin',
    ),
    DataTask(
        task_name='macp_yest',
        func=macp_yest,
        merge_key='isin',
    ),
    DataTask(
        task_name='bval_cod',
        func=bval_cod,
        merge_key='isin',
        strict_task_requirements=['bval_yest'],
        strict_col_requirements=[
            'bvalYestMidSpd', 'bvalYestMidPx', 'bvalYestMidYld',
            'bvalMidSpd', 'bvalMidPx', 'bvalMidYld',
        ]
    ),
    DataTask(
        task_name='macp_cod',
        func=macp_cod,
        merge_key='isin',
        strict_task_requirements=['macp_yest'],
        strict_col_requirements=[
            'macpYestMidSpd', 'macpYestMidPx',
            'macpMidSpd', 'macpMidPx',
        ]
    ),
    DataTask(
        task_name='macp_ingress',
        func=macp_ingress,
        merge_key='isin',
        strict_col_requirements=[
            'macpMidSpd', 'macpMidPx',
        ],
        use_cached_providers=False,
        expected_col_provides={
            'macpMidSpdIngress': pl.Float64,
            'macpMidPxIngress' : pl.Float64
        },
    ),
    DataTask(
        task_name='quote_markit',
        broadcast_name="EVB Quotes",
        func=quote_markit,
        merge_key='isin',
        based_on_col='markitRefreshTime',
    ),
    DataTask(
        task_name='quote_idc',
        broadcast_name="IDC Quotes",
        func=quote_idc,
        merge_key='isin',
        based_on_col='idcRefreshTime',
    ),
    DataTask(
        task_name='quote_ice_rplus',
        broadcast_name="ICE Quotes",
        func=quote_ice_rplus,
        merge_key='isin',
        based_on_col='idcRefreshTime',
        isOptional=True,
        critical_columns={
            'idcEvalBidPx', 'idcEvalMidPx', 'idcEvalAskPx',
            'idcEval4BidPx', 'idcEval4MidPx',
            'idcBidPx', 'idcMidPx', 'idcAskPx',
            'idcBidSpd', 'idcMidSpd', 'idcAskSpd',
            'idcLiqScore',
        },
        max_retries=0,
    ),
    DataTask(
        task_name='adj_trace_credit',
        broadcast_name="Trace Quotes",
        func=quote_adj_trace,
        merge_key='isin',
        kwargs={'source': 'credit'},
        strict_task_requirements=["cached_quoteevent"],
        based_on_col='adjTraceRefreshTime',
        use_cached_providers=False,
        toFrame="quoteevent",
        isOptional=True,
        critical_columns=['adjTraceRefreshTime'],
        expected_col_provides={
            'adjTraceBidPx'      : pl.Float64,
            'adjTraceMidPx'      : pl.Float64,
            'adjTraceAskPx'      : pl.Float64,
            'adjTraceRefreshTime': pl.Datetime
        },
    ),
    DataTask(
        task_name='adj_trace_pano',
        broadcast_name="Trace Quotes",
        func=quote_adj_trace,
        merge_key='isin',
        kwargs={'source': 'pano'},
        strict_task_requirements=["cached_quoteevent"],
        based_on_col='adjTraceRefreshTime',
        use_cached_providers=False,
        isOptional=True,
        toFrame="quoteevent",
        critical_columns=['adjTraceRefreshTime'],
        expected_col_provides={
            'adjTraceBidPx'      : pl.Float64,
            'adjTraceMidPx'      : pl.Float64,
            'adjTraceAskPx'      : pl.Float64,
            'adjTraceRefreshTime': pl.Datetime
        },
    ),
    DataTask(
        task_name='quote_trace',
        broadcast_name="Trace Prices",
        func=quote_trace,
        max_retries=0,
        strict_col_requirements=["cusip"],
        merge_key='isin',
    ),
    DataTask(
        task_name='quote_illiquids',
        broadcast_name="Illiquid Quotes",
        func=quote_illiquids,
        merge_key='isin',
    ),
    DataTask(
        task_name='quote_ibval',
        broadcast_name="IBVAL Quotes",
        func=quote_ibval,
        merge_key='isin',
    ),
    DataTask(
        task_name='quote_cbbt',
        broadcast_name="CBBT Quotes",
        strict_task_requirements=["cached_quoteevent"],
        func=quote_cbbt,
        merge_key='isin',
        toFrame="quoteevent",
        based_on_col='cbbtRefreshTime',
    ),
    DataTask(
        task_name='quote_cbbt_basket',
        broadcast_name="CBBT Quotes",
        strict_task_requirements=["cached_quoteevent"],
        func=quote_cbbt_basket,
        merge_key='isin',
        toFrame="quoteevent",
        based_on_col='cbbtRefreshTime',
    ),
    DataTask(
        task_name='quote_cbbt_pano_us',
        broadcast_name="CBBT Quotes",
        isOptional=True,
        critical_columns=['cbbtRefreshTime'],
        strict_task_requirements=["cached_quoteevent", 'quote_cbbt'],
        func=quote_cbbt_pano,
        merge_key='isin',
        toFrame="quoteevent",
        kwargs={'region': 'US'},
        based_on_col='cbbtRefreshTime',
    ),
    DataTask(
        task_name='quote_cbbt_pano_eu',
        broadcast_name="CBBT Quotes",
        isOptional=True,
        critical_columns=['cbbtRefreshTime'],
        strict_task_requirements=["cached_quoteevent", 'quote_cbbt'],
        func=quote_cbbt_pano,
        merge_key='isin',
        toFrame="quoteevent",
        kwargs={'region': 'EU'},
        based_on_col='cbbtRefreshTime',
    ),
    DataTask(
        task_name='quote_cbbt_pano_sgp',
        broadcast_name="CBBT Quotes",
        isOptional=True,
        critical_columns=['cbbtRefreshTime'],
        strict_task_requirements=["cached_quoteevent", 'quote_cbbt'],
        func=quote_cbbt_pano,
        merge_key='isin',
        toFrame="quoteevent",
        kwargs={'region': 'SGP'},
        based_on_col='cbbtRefreshTime',
    ),
    DataTask(
        task_name='quote_cbbt_ext',
        broadcast_name="CBBT Quotes",
        strict_task_requirements=["cached_quoteevent"],
        strict_col_requirements=['bbpk'],
        func=quote_cbbt_ext,
        merge_key='isin',
        toFrame="quoteevent",
        max_retries=1,
        retry_delay=2,
        kwargs={"timeout": 5},
        based_on_col='cbbtRefreshTime',
    ),
    DataTask(
        task_name='quote_macp',
        broadcast_name="MACP Quotes",
        strict_task_requirements=["cached_quoteevent"],
        func=quote_macp,
        merge_key='isin',
        toFrame="quoteevent",
        based_on_col='macpRefreshTime',
    ),
    DataTask(
        task_name='quote_macp_pano',
        broadcast_name="MACP Quotes",
        isOptional=True,
        critical_columns=['macpRefreshTime'],
        strict_task_requirements=["cached_quoteevent"],
        func=quote_macp_pano,
        merge_key='isin',
        toFrame="quoteevent",
        based_on_col='macpRefreshTime',
    ),
    DataTask(
        task_name='quote_markitRt',
        broadcast_name="MARKIT RT Quotes",
        strict_task_requirements=["cached_quoteevent"],
        func=quote_markitRt,
        merge_key='isin',
        toFrame="quoteevent",
        based_on_col='markitRtRefreshTime',
    ),
    DataTask(
        task_name='quote_mlcr',
        broadcast_name="MLCR Quotes",
        func=quote_mlcr,
        strict_task_requirements=["cached_quoteevent"],
        merge_key='isin',
        based_on_col='mlcrRefreshTime',
        toFrame="quoteevent",
    ),
    DataTask(
        task_name='quote_house_tm_us',
        broadcast_name="House Quotes",
        func=quote_house_tm,
        merge_key='isin',
        isOptional=True,
        kwargs={"region": "US"},
        toFrame='quoteevent',
        critical_columns=[
            'houseUsMidPx', 'houseUsMidSpd', 'houseUsMidYld', 'houseUsQuoteConvention'
        ],
        based_on_col='houseUsRefreshTime',
        strict_task_requirements=['cached_quoteevent']
    ),
    DataTask(
        task_name='quote_house_bq_us',
        broadcast_name="House Quotes",
        func=quote_house_bq,
        merge_key='isin',
        isOptional=True,
        kwargs={"region": "US"},
        toFrame='quoteevent',
        critical_columns=[
            'houseUsMidPx', 'houseUsMidSpd', 'houseUsMidYld', 'houseUsQuoteConvention', 'unitDv01'
        ],
        based_on_col='houseUsRefreshTime',
        strict_task_requirements=['cached_quoteevent']
    ),
    EchoTask(
        task_name='_quote_house_bq_us_desig',
        broadcast_name="Desigs",
        columns=['isin', 'houseUsRefreshTime'],
        strict_col_requirements=['houseUsRefreshTime'],
        strict_task_requirements=['_pano_positions_us_desig', 'cached_quoteevent'],
        fromFrame='quoteevent',
        merge_key='isin',
        toFrame='desig_us',
    ),
    DataTask(
        task_name='quote_house_bq_eu',
        broadcast_name="House Quotes",
        func=quote_house_bq,
        merge_key='isin',
        isOptional=True,
        kwargs={"region": "EU"},
        critical_columns=[
            'houseEuMidPx', 'houseEuMidSpd', 'houseEuMidYld', 'houseEuQuoteConvention', 'unitDv01'
        ],
        based_on_col='houseEuRefreshTime',
        strict_task_requirements=['cached_quoteevent'],
        toFrame='quoteevent'
    ),
    EchoTask(
        task_name='_quote_house_bq_eu_desig',
        broadcast_name="Desigs",
        columns=['isin', 'houseEuRefreshTime'],
        strict_col_requirements=['houseEuRefreshTime'],
        strict_task_requirements=['_pano_positions_eu_desig', 'cached_quoteevent'],
        fromFrame='quoteevent',
        merge_key='isin',
        toFrame='desig_eu'
    ),
    DataTask(
        task_name='quote_house_bq_sgp',
        broadcast_name="House Quotes",
        func=quote_house_bq,
        merge_key='isin',
        isOptional=True,
        kwargs={"region": "SGP"},
        toFrame='quoteevent',
        strict_task_requirements=['cached_quoteevent'],
        critical_columns=[
            'houseSgpMidPx', 'houseSgpMidSpd', 'houseSgpMidYld', 'houseSgpQuoteConvention', 'unitDv01'
        ],
        based_on_col="houseSgpRefreshTime"
    ),
    EchoTask(
        task_name='_quote_house_bq_sgp_desig',
        broadcast_name="Desigs",
        columns=['isin', 'houseSgpRefreshTime'],
        strict_col_requirements=['houseSgpRefreshTime'],
        strict_task_requirements=['_pano_positions_sgp_desig', 'cached_quoteevent'],
        fromFrame='quoteevent',
        merge_key='isin',
        toFrame='desig_sgp'
    ),
    DataTask(
        task_name='coalesce_house',
        broadcast_name="House Quotes",
        func=coalesce_house,
        merge_key='isin',
        fromFrame='main',
        toFrame='main',
        strict_col_requirements=[
            'houseUsMidPx', 'houseUsMidSpd', 'houseUsMidYld', 'houseUsQuoteConvention',
            'houseEuMidPx', 'houseEuMidSpd', 'houseEuMidYld', 'houseEuQuoteConvention',
            'houseSgpMidPx', 'houseSgpMidSpd', 'houseSgpMidYld', 'houseSgpQuoteConvention',
            'desigRegion'
        ],
        based_on_col='houseRefreshTime',
        strict_task_requirements=['cached_quoteevent']
    ),

    DataTask(
        task_name='cached_quoteevent_stats',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=[
            'cached_quoteevent', 'quote_stats_eu', 'quote_stats_us', 'coalesce_stats'
        ],
        strict_col_requirements=['isin', 'statsRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "stats"},
        global_ignore=True,
        based_on_col="statsRefreshTime"
    ),
    DataTask(
        task_name='axe_coalesce',
        broadcast_name="Axe Quotes",
        func=axe_coalesce,
        merge_key='isin',
        strict_col_requirements=[
            'isStatsUsBidAxe', 'isStatsUsAskAxe', 'isStatsUsMktAxe',
            'isStatsEuBidAxe', 'isStatsEuAskAxe', 'isStatsEuMktAxe',
            'isStatsSgpBidAxe', 'isStatsSgpAskAxe', 'isStatsSgpMktAxe',
        ],
        use_cached_providers=False,
        expected_col_provides={
            'axeFullAskSize': pl.Float64,
            'axeFullBidSize': pl.Float64,
            'isAntiAxe'     : pl.Int8,
            'isAskAxe'      : pl.Int8,
            'isBidAxe'      : pl.Int8,
            'isMktAxe'      : pl.Int8
        },
    ),
    DataTask(
        task_name='coalesce_stats',
        broadcast_name="Stats Quotes",
        func=coalesce_stats,
        merge_key='isin',
        fromFrame='main',
        toFrame='main',
        strict_col_requirements=[
            'desigRegion', 'statsUsBidPx', 'statsEuBidPx', 'statsSgpBidPx',
        ],
        based_on_col='statsRefreshTime',
        strict_task_requirements=[
            'cached_quoteevent', 'quote_stats_us', 'quote_stats_eu'
        ]
    ),
    DataTask(
        task_name='quote_stats_us',
        broadcast_name="Axe Quotes",
        func=quote_stats,
        merge_key='isin',
        isOptional=True,
        strict_task_requirements=['cached_quoteevent'],
        toFrame='quoteevent',
        critical_columns=['isStatsUsBidAxe', 'isStatsUsAskAxe', 'isStatsUsMktAxe'],
        kwargs={"region": "US"},
    ),
    EchoTask(
        task_name='quote_stats_us_desig',
        broadcast_name="Desigs",
        merge_key='isin',
        fromFrame='quoteevent',
        toFrame='desig_us',
        strict_task_requirements=['_pano_positions_us_desig', 'cached_quoteevent'],
        columns=['isStatsUsBidAxe', 'isStatsUsAskAxe', 'isin'],
        strict_col_requirements=['isStatsUsBidAxe', 'isStatsUsAskAxe', 'isStatsUsMktAxe'],

    ),
    DataTask(
        task_name='quote_stats_eu',
        broadcast_name="Axe Quotes",
        func=quote_stats,
        merge_key='isin',
        kwargs={"region": "EU"},
        isOptional=True,
        strict_task_requirements=['cached_quoteevent'],
        toFrame='quoteevent',
        critical_columns=['isStatsEuBidAxe', 'isStatsEuAskAxe', 'isStatsEuMktAxe']
    ),
    EchoTask(
        task_name='quote_stats_eu_desig',
        broadcast_name="Desigs",
        merge_key='isin',
        toFrame='desig_eu',
        fromFrame='quoteevent',
        columns=['isStatsEuBidAxe', 'isStatsEuAskAxe', 'isin'],
        strict_task_requirements=['_pano_positions_eu_desig'],
        strict_col_requirements=['isStatsEuBidAxe', 'isStatsEuAskAxe', 'isStatsEuMktAxe'],
    ),
    DataTask(
        task_name='quote_runz_creditext_us',
        broadcast_name="Runz Quotes",
        func=quote_runz_creditext,
        strict_task_requirements=['junior_map', 'cached_quoteevent'],
        frameContext=['junior_map'],
        merge_key='isin',
        toFrame='quoteevent',
        kwargs={"region": "US", "swap_juniors": True},
    ),
    DataTask(
        task_name='quote_runz_pano_us',
        broadcast_name="Runz Quotes",
        func=quote_runz_panoproxy,
        isOptional=True,
        critical_columns=['_usHasRunz'],
        strict_task_requirements=['junior_map', 'cached_quoteevent'],
        frameContext=['junior_map'],
        merge_key='isin',
        toFrame='quoteevent',
        kwargs={"region": "US", "swap_juniors": True},
    ),
    EchoTask(
        task_name='_quote_runz_creditext_us_desigs',
        broadcast_name="Desigs",
        columns=[
            '_usRunzSenderLastName', '_usRunzSenderName', 'isin', 'usIsRunzAskAxe', 'usIsRunzBidAxe', 'usRunzSenderName'
        ],
        strict_task_requirements=['cached_quoteevent'],
        merge_key='isin',
        fromFrame='quoteevent',
        toFrame='desig_us',
        isOptional=True,
        critical_columns=['_usHasRunz'],
        use_cached_providers=False,
        expected_col_provides={
            '_usRunzSenderLastName': pl.List(pl.String),
            '_usRunzSenderName'    : pl.List(pl.String),
            'usIsRunzAskAxe'       : pl.Int8,
            'usIsRunzBidAxe'       : pl.Int8,
            'usRunzSenderName'     : pl.String
        },
    ),
    DataTask(
        task_name='quote_runz_creditext_eu',
        broadcast_name="Runz Quotes",
        func=quote_runz_creditext,
        strict_task_requirements=['junior_map', 'cached_quoteevent'],
        frameContext=['junior_map'],
        merge_key='isin',
        kwargs={"region": "EU", "swap_juniors": True},
        expected_col_provides={
            '_euRunzSenderLastName': pl.List(pl.String),
            '_euRunzSenderName'    : pl.List(pl.String),
            'euIsRunzAskAxe'       : pl.Int8,
            'euIsRunzBidAxe'       : pl.Int8,
            'euRunzSenderName'     : pl.String
        },
        toFrame='quoteevent'
    ),
    DataTask(
        task_name='junior_map',
        broadcast_name="",
        func=junior_traders,
        merge_key='',
        toFrame='junior_map'
    ),
    EchoTask(
        task_name='_quote_runz_creditext_eu_desigs',
        broadcast_name="Desigs",
        columns=[
            '_euRunzSenderLastName',
            '_euRunzSenderName',
            'euIsRunzAskAxe',
            'euIsRunzBidAxe',
            'euRunzSenderName',
            'isin'
        ],
        strict_task_requirements=[
            'quote_runz_creditext_eu', '_pano_positions_eu_desig', 'cached_quoteevent'
        ],
        merge_key='isin',
        fromFrame='quoteevent',
        toFrame='desig_eu',
        use_cached_providers=False,
        expected_col_provides={
            '_euRunzSenderLastName': pl.List(pl.String),
            '_euRunzSenderName'    : pl.List(pl.String),
            'euIsRunzAskAxe'       : pl.Int8,
            'euIsRunzBidAxe'       : pl.Int8,
            'euRunzSenderName'     : pl.String
        },
    ),
    DataTask(
        task_name='quote_runz_creditext_sgp',
        broadcast_name="Runz Quotes",
        func=quote_runz_creditext,
        strict_task_requirements=['junior_map', 'cached_quoteevent'],
        frameContext=['junior_map'],
        merge_key='isin',
        kwargs={"region": "SGP", "swap_juniors": True},
        toFrame="quoteevent",
    ),
    EchoTask(
        task_name='_quote_runz_creditext_sgp_desigs',
        broadcast_name="Desigs",
        columns=[
            '_sgpRunzSenderLastName', '_sgpRunzSenderName', 'isin', 'sgpIsRunzAskAxe', 'sgpIsRunzBidAxe',
            'sgpRunzSenderName'
        ],
        strict_task_requirements=[
            '_pano_positions_sgp_desig', 'cached_quoteevent', 'quote_runz_creditext_sgp'
        ],
        fromFrame='quoteevent',
        merge_key='isin',
        toFrame='desig_sgp',
        use_cached_providers=False,
        expected_col_provides={
            '_sgpRunzSenderLastName': pl.List(pl.String),
            '_sgpRunzSenderName'    : pl.List(pl.String),
            'sgpIsRunzAskAxe'       : pl.Int8,
            'sgpIsRunzBidAxe'       : pl.Int8,
            'sgpRunzSenderName'     : pl.String
        },
    ),
    DataTask(
        task_name='cached_quoteevent',
        broadcast_name="Quotes",
        func=cached_quoteevent,
        strict_col_requirements=['isin'],
        fromFrame='main',
        toFrame='quoteevent',
        merge_key='',
    ),
    DataTask(
        task_name='cached_quoteevent_allqbemu',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'allqBemuRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "allqBemu"},
        based_on_col="allqBemuRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_allqbxcr',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'allqBxcrRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "allqBxcr"},
        based_on_col="allqBxcrRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_am',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=[
            'cached_quoteevent',
            'quote_runz_creditext_eu', 'quote_runz_creditext_sgp', 'quote_runz_creditext_us',
            'quote_runz_pano_eu', 'quote_runz_pano_sgp', 'quote_runz_pano_us'
        ],
        strict_col_requirements=['isin', 'amRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "am"},
        based_on_col="amRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_axi',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'axiRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "axi"},
        based_on_col="axiRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_bondpoint',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'bondpointRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "bondpoint"},
        based_on_col="bondpointRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_bondspro',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'bondsproRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "bondspro"},
        based_on_col="bondsproRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_cbbt',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent', 'quote_cbbt'],
        strict_col_requirements=['isin', 'cbbtRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "cbbt"},
        based_on_col="cbbtRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_macp',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent', 'quote_macp'],
        strict_col_requirements=['isin', 'macpRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "macp"},
        based_on_col="macpRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_markit',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'markitRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "markit"},
        based_on_col="markitRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_mlcr',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'mlcrRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "mlcr"},
        based_on_col="mlcrRefreshTime"
    ),
    DataTask(
        task_name='cached_adj_trace',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'adjTraceRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "adjTrace"},
        based_on_col="adjTraceRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_tw',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'twRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "tw"},
        based_on_col="twRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_malive',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'maLiveRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "maLive"},
        based_on_col="maLiveRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_markitrt',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'markitRtRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "markitRt"},
        based_on_col="markitRtRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_allq',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'allqRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "allq"},
        based_on_col="allqRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_houseus',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'houseUsRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "houseUs"},
        based_on_col="houseUsRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_houseeu',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'houseEuRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "houseEu"},
        based_on_col="houseEuRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_housesgp',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'houseSgpRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "houseSgp"},
        based_on_col="houseSgpRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_usam',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'usAmRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "usAm"},
        based_on_col="usAmRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_usam_runz',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'usRunzAxeRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "usRunz"},
        based_on_col="usRunzAxeRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_usam_algo',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'usAlgoRunzRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "usAlgo"},
        based_on_col="usAlgoRunzRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_euam',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'euAmRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "euAm"},
        based_on_col="euAmRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_euam_runz',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'euRunzAxeRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "euRunz"},
        based_on_col="euRunzAxeRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_euam_algo',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'euAlgoRunzRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "euAlgo"},
        based_on_col="euAlgoRunzRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_sgpam',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'sgpAmRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "sgpAm"},
        based_on_col="sgpAmRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_sgpam_runz',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'sgpRunzAxeRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "sgpRunz"},
        based_on_col="sgpRunzAxeRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_sgpam_algo',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'sgpAlgoRunzRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "sgpAlgo"},
        based_on_col="sgpAlgoRunzRefreshTime"
    ),
    DataTask(
        task_name='cached_quoteevent_tmc',
        broadcast_name="Quotes",
        func=merge_cached_quoteevent,
        strict_task_requirements=['cached_quoteevent'],
        strict_col_requirements=['isin', 'tmcRefreshTime'],
        fromFrame='quoteevent',
        merge_key='isin',
        kwargs={"market": "tmc"},
        based_on_col="tmcRefreshTime"
    ),
    DataTask(
        task_name="cached_quoteevent_cleanup",
        strict_task_requirements=['(|)?((cached_quoteevent_)[a-z]+)'],
        merge_key='isin',
        fromFrame='quoteevent',
        func=cached_quoteevent_cleanup
    )
]

algo_tasks = [
    DataTask(
        task_name='stats_signals',
        broadcast_name="Algo Signals",
        func=stats_signals,
        strict_col_requirements=['unitCs01'],
        strict_task_requirements=['cs01_estimate'],
        merge_key='isin',
    ),
    DataTask(
        task_name='realtime_signals_eu',
        broadcast_name="Algo Signals",
        func=realtime_signals,
        kwargs={"region": "EU"},
        strict_col_requirements=['unitCs01'],
        strict_task_requirements=['cs01_estimate'],
        merge_key='isin'
    ),
    DataTask(
        task_name='signal_filler',
        broadcast_name="Algo Signals",
        strict_task_requirements=['stats_signals', 'realtime_signals_eu'],
        func=signal_filler,
        merge_key='tnum'
    ),
    DataTask(
        task_name='us_algo_eligibility',
        broadcast_name="Algo Eligibility",
        func=us_algo_eligibility,
        merge_key='isin'
    ),
    # DataTask(task_name='assign_algo',
    #          broadcast_name="Algo Eligibility",
    #          func=us_algo_elibility,
    #          merge_key='isin'
    #         provides=['whichAlgo'],
    #          ),
]

benchmark_tasks = [
    DataTask(
        task_name='maturity_benchmark',
        func=maturity_benchmark,
        merge_key='isin',
        frameContext=["benchmarks"],
        strict_task_requirements=['ust_ref'],
        strict_col_requirements=[
            'isCallable', 'isFloater', 'isPerpetual', 'maturityDate', 'pseudoWorkoutDate'
        ],
        use_cached_providers=False,
        expected_col_provides={
            'maturityBenchmarkIsin': pl.String,
        },
    ),
    DataTask(
        task_name='ust_ref',
        broadcast_name="UST",
        func=ust_ref,
        merge_key='',
        toFrame="benchmarks",
        kwargs={'market': "US"}
    ),
    DataTask(
        task_name='ust_bval_yields',
        broadcast_name="UST Yields",
        func=ust_bval_yields,
        strict_col_requirements=['bvalBenchmarkIsin', 'benchmarkIsin'],
        strict_task_requirements=['coalesce_benchmarks', 'ust_yields'],
        merge_key='isin',
        expected_col_provides={
            'bvalBenchmarkMidPx'        : pl.Float64,
            'bvalBenchmarkMidYld'       : pl.Float64,
            'bvalAlignedBenchmarkMidPx' : pl.Float64,
            'bvalAlignedBenchmarkMidYld': pl.Float64,
            'benchmarkUnitDv01'         : pl.Float64,
        },
    ),
    DataTask(
        task_name='ust_yields',
        broadcast_name="UST Yields",
        func=ust_yields,
        strict_col_requirements=['benchFido'],
        strict_task_requirements=['coalesce_benchmarks'],
        merge_key='benchFido',
    ),
    DataTask(
        task_name='add_esm_to_benchmarks_quick',
        broadcast_name="UST ESM",
        func=add_esm_to_benchmarks,
        merge_key='benchmarkIsin',
        fromFrame='benchmarks',
        toFrame="benchmarks",
        kwargs={'source': 'pano'},
        strict_task_requirements=['ust_ref'],
        isOptional=True,
        critical_columns=['benchmarkEsm'],
        use_cached_providers=False,
        expected_col_provides={
            'benchmarkEsm': pl.String,
        },
    ),
    DataTask(
        task_name='add_esm_to_benchmarks_full',
        broadcast_name="UST ESM",
        func=add_esm_to_benchmarks,
        merge_key='benchmarkIsin',
        fromFrame='benchmarks',
        toFrame="benchmarks",
        kwargs={'source': 'esm'},
        strict_task_requirements=['ust_ref'],
        isOptional=True,
        critical_columns=['benchmarkEsm'],
        use_cached_providers=False,
        expected_col_provides={
            'benchmarkEsm': pl.String,
        },
    ),
    EchoTask(
        task_name='benchmarks_loaded_benchmarks',
        fromFrame='benchmarks',
        toFrame="benchmarks",
        columns=None,
        merge_key='benchmarkIsin',
        strict_task_requirements=['add_esm_to_benchmarks_quick', 'add_esm_to_benchmarks_full', 'ust_ref'],
        strict_col_requirements=['benchmarkEsm']
    ),
    EchoTask(
        task_name='benchmarks_loaded_main',
        columns=None,
        merge_key='isin',
        strict_task_requirements=[
            "cached_quoteevent_am", "cached_quoteevent_cbbt", "cached_quoteevent_houseeu", "cached_quoteevent_housesgp",
            "cached_quoteevent_houseus", "cached_quoteevent_macp", "cached_quoteevent_markitrt",
            "cached_quoteevent_mlcr", "kdb_bond_pano_static", "quote_bval_full", "quote_bval_non_dm", "quote_stats",
            "quote_stats_eu", "quote_stats_us", 'maturity_benchmark', 'coalesce_house',
        ],
    ),
    DataTask(
        task_name='coalesce_benchmarks',
        func=coalesce_benchmarks,
        merge_key='isin',
        frameContext=["benchmarks"],
        strict_task_requirements=['benchmarks_loaded_benchmarks', 'benchmarks_loaded_main'],
        strict_col_requirements=[
            'isCallable', 'isFloater', 'isPerpetual', 'maturityDate', 'pseudoWorkoutDate'
        ],
    ),
    DataTask(
        task_name='non_us_ref',
        func=ust_ref,
        kwargs={'region': 'EU'},
        isOptional=True,
        critical_columns=['benchName'],
        strict_task_requirements=['coalesce_benchmarks'],
        merge_key='benchmarkIsin',
    ),
    DataTask(
        task_name='non_ust_yields',
        broadcast_name="UST Yields",
        func=non_ust_yields,
        isOptional=True,
        critical_columns=['benchmarkMidPx'],
        strict_task_requirements=['coalesce_benchmarks', 'non_us_ref'],
        merge_key='benchmarkIsin',
    )
]

# bgc benchmarks = select from .externalfeeds.ldn.bgcAuction where date=.z.d

POSITION_LOOKBACK = 90
positions_tasks = [
    # DataTask(task_name='positions_us',
    #          broadcast_name="Positions",
    #          func=positions,
    #          merge_key='',
    #          strict_col_requirements=['isin'],
    #          strict_task_requirements=['fungible_series'],
    #          frameContext=['funges'],
    #          kwargs={"region":"US", "lookback":POSITION_LOOKBACK},
    #          toFrame="longterm_positions_us"
    #          ),
    # DataTask(task_name='positions_eu',
    #          broadcast_name="Positions",
    #          func=positions,
    #          merge_key='',
    #          strict_col_requirements=['isin'],
    #          strict_task_requirements=['fungible_series'],
    #          frameContext=['funges'],
    #          kwargs={"region": "EU", "lookback": POSITION_LOOKBACK},
    #          toFrame="longterm_positions_eu"
    #          ),
    # DataTask(task_name='positions_sgp',
    #          broadcast_name="Positions",
    #          func=positions,
    #          merge_key='',
    #          strict_col_requirements=['isin'],
    #          strict_task_requirements=['fungible_series'],
    #          frameContext=['funges'],
    #          kwargs={"region": "SGP", "lookback": POSITION_LOOKBACK},
    #          toFrame="longterm_positions_sgp"
    #          ),
    DataTask(
        task_name='pano_positions_us',
        broadcast_name="Positions",
        func=pano_positions,
        merge_key='',
        strict_col_requirements=['isin'],
        strict_task_requirements=['fungible_series'],
        frameContext=['funges'],
        kwargs={"region": "US", "lookback": 0},
        toFrame="realtime_positions_us"
    ),
    EchoTask(
        task_name='_pano_positions_us_desig',
        broadcast_name="Desigs",
        merge_key='',
        strict_task_requirements=['pano_positions_us'],
        strict_col_requirements=[
            '_isFungeDesig', '_isTrueDesig', 'bookId', 'bookRegion', 'deskAsset', 'isDesig', 'isin', 'netPosition',
            'traderId'
        ],
        columns=[
            '_isFungeDesig', '_isTrueDesig', 'bookId', 'bookRegion', 'deskAsset', 'isDesig', 'isin', 'netPosition',
            'traderId'
        ],
        fromFrame='realtime_positions_us',
        toFrame="desig_us"
    ),
    DataTask(
        task_name='pano_positions_eu',
        broadcast_name="Positions",
        func=pano_positions,
        merge_key='',
        strict_col_requirements=['isin'],
        strict_task_requirements=['fungible_series'],
        frameContext=['funges'],
        kwargs={"region": "EU", "lookback": 0},
        toFrame="realtime_positions_eu"
    ),
    EchoTask(
        task_name='_pano_positions_eu_desig',
        broadcast_name="Desigs",
        merge_key='',
        strict_col_requirements=[
            '_isFungeDesig', '_isTrueDesig', 'bookId', 'bookRegion', 'deskAsset', 'isDesig', 'isin', 'netPosition',
            'traderId'
        ],
        columns=[
            '_isFungeDesig', '_isTrueDesig', 'bookId', 'bookRegion', 'deskAsset', 'isDesig', 'isin', 'netPosition',
            'traderId'
        ],
        strict_task_requirements=['pano_positions_eu'],
        fromFrame='realtime_positions_eu',
        toFrame="desig_eu"
    ),
    DataTask(
        task_name='pano_positions_sgp',
        broadcast_name="Positions",
        func=pano_positions,
        merge_key='',
        strict_col_requirements=['isin'],
        strict_task_requirements=['fungible_series'],
        frameContext=['funges'],
        kwargs={"region": "SGP", "lookback": 0},
        toFrame="realtime_positions_sgp"
    ),
    EchoTask(
        task_name='_pano_positions_sgp_desig',
        broadcast_name="Desigs",
        merge_key='',
        strict_col_requirements=[
            '_isFungeDesig', '_isTrueDesig', 'bookId', 'bookRegion', 'deskAsset', 'isDesig', 'isin', 'netPosition',
            'traderId'
        ],
        columns=[
            '_isFungeDesig', '_isTrueDesig', 'bookId', 'bookRegion', 'deskAsset', 'isDesig', 'isin', 'netPosition',
            'traderId'
        ],
        strict_task_requirements=['pano_positions_sgp'],
        fromFrame='realtime_positions_sgp',
        toFrame="desig_sgp"
    ),
    DataTask(
        task_name='position_aggregator',
        broadcast_name="Positions",
        func=position_aggregator,
        merge_key='isin',
        strict_task_requirements=[
            'pano_positions_us', 'pano_positions_eu', 'pano_positions_sgp'
        ],
        frameContext=[
            "realtime_positions_us", "realtime_positions_eu", "realtime_positions_sgp"
        ],
        toFrame="main",
        use_cached_providers=False,
        expected_col_provides={
            "algoBookIds"             : pl.String,
            "deskBookIds"             : pl.String,
            "firmBookIds"             : pl.String,
            "netAlgoPosition"         : pl.Float64,
            "netDeskPosition"         : pl.Float64,
            "netEuFirmPosition"       : pl.Float64,
            "netFirmPosition"         : pl.Float64,
            "netFungeAlgoPosition"    : pl.Float64,
            "netFungeDeskPosition"    : pl.Float64,
            "netFungeFirmPosition"    : pl.Float64,
            "netFungeStrategyPosition": pl.Float64,
            "netSgpFirmPosition"      : pl.Float64,
            "netStrategyPosition"     : pl.Float64,
            "netTrueAlgoPosition"     : pl.Float64,
            "netTrueDeskPosition"     : pl.Float64,
            "netTrueFirmPosition"     : pl.Float64,
            "netTrueStrategyPosition" : pl.Float64,
            "netUsFirmPosition"       : pl.Float64,
            "strategyBookIds"         : pl.String
        }
    ),
]

meta_tasks = [
    DataTask(
        task_name='init_meta',
        broadcast_name="",
        func=init_meta,
        merge_key='',
        fromFrame='raw',
        toFrame="meta"
    ),
    DataTask(
        task_name='meta_aggregates',
        broadcast_name="",
        func=meta_aggregates,
        merge_key='__meta_merge_key',
        strict_task_requirements=['risk_transforms', 'init_meta'],
        fromFrame='main',
        toFrame="meta"
    ),
    DataTask(
        task_name='client_enhance',
        broadcast_name="",
        func=client_enhance,
        merge_key='__meta_merge_key',
        strict_task_requirements=['init_meta'],
        fromFrame='meta',
        frameContext=['raw'],
        toFrame="meta",
        mergePolicy='overwrite',
    ),
    DataTask(
        task_name='custom_fields',
        broadcast_name="",
        func=custom_meta_fields,
        merge_key='__meta_merge_key',
        strict_task_requirements=['init_meta'],
        fromFrame='meta',
        frameContext=['raw'],
        toFrame="meta",
        mergePolicy='overwrite',
    ),
    DataTask(
        task_name='meta_classify',
        func=meta_classify,
        merge_key='__meta_merge_key',
        strict_task_requirements=['init_meta'],
        strict_col_requirements=["desigRegion", "desigBookId", "deskAsset", "currency"],
        frameContext=['raw', 'meta'],
        toFrame="meta"
    ),
    DataTask(
        task_name='meta_liquidity',
        broadcast_name="",
        func=meta_liquidity,
        merge_key='__meta_merge_key',
        strict_task_requirements=['init_meta'],
        strict_col_requirements=[
            'avgLiqScore', 'blsLiqScore', 'czLiqScore', 'czLiqScore', 'dkLiqScore', 'grossDv01', 'grossSize',
            'lcsLiqScore', 'liqScore', 'lqaLiqScore', 'macpLiqScore', 'mlcrLiqScore', 'muniLiqScore', 'smadLiqScore',
            'idcLiqScore'
        ],
        kwargs={"weight": "grossDv01"},
        fromFrame='main',
        toFrame="meta"
    ),
    DataTask(
        task_name='meta_sales',
        broadcast_name="Identifying Sales",
        func=sales_person_lookup,
        merge_key='__meta_merge_key',
        frameContext=['raw'],
        strict_task_requirements=['init_meta', 'meta_classify', 'client_enhance'],
        strict_col_requirements=[
            'rfqClient', 'client', 'clientBcName', 'clientAltName', 'clientUbcName', 'client', 'assetClass'
        ],
        fromFrame='meta',
        toFrame="meta",
        use_cached_providers=False,
        expected_col_provides={
            "salesName" : pl.String,
            "venueShort": pl.String,
            "rfqClient" : pl.String,
            "assetClass": pl.String,
        }
    ),
    DataTask(
        task_name='client_flag',
        func=client_flag,
        merge_key='__meta_merge_key',
        strict_task_requirements=['init_meta', 'meta_sales'],
        strict_col_requirements=['salesName', 'assetClass', 'rfqClient'],
        fromFrame='meta',
        toFrame="meta",
        global_ignore=True,
        use_cached_providers=False,
        expected_col_provides={
            "isClientTtt" : pl.Int8,
            "isClientM100": pl.Int8,
            "salesName"   : pl.String,
            "rfqClient"   : pl.String,
            "assetClass"  : pl.String,
        }
    ),
    DataTask(
        task_name='meta_counts',
        func=meta_counts,
        merge_key='__meta_merge_key',
        strict_task_requirements=['init_meta', 'init_values'],
        strict_col_requirements=['isin', 'isReal'],
        fromFrame='main',
        toFrame="meta"
    ),
    DataTask(
        task_name='meta_wavgs',
        func=meta_wavgs,
        merge_key='__meta_merge_key',
        strict_task_requirements=['init_meta', 'ref_data_transforms'],
        strict_col_requirements=[
            'coupon', 'convexity', 'duration', 'yrsToMaturity', 'yrsSinceIssuance',
            'avgLife', 'cdsBasisToWorst', "isNewIssue"
                                          'bvalMidPx', 'bvalMidSpd', 'macpMidPx', 'macpMidSpd', 'cbbtMidPx',
            'cbbtMidSpd',
            "inEtfAgg", "inEtfLqd", "inEtfHyg", "inEtfJnk",
            "inEtfEmb", "inEtfSpsb", "inEtfSpib", "inEtfSplb", "inEtfVcst",
            "inEtfVcit", "inEtfVclt", "inEtfSpab", "inEtfIgib", "inEtfIglb",
            "inEtfIgsb", "inEtfIemb", "inEtfUsig", "inEtfUshy", "inEtfSjnk",
            'isIgAlgoEligible', 'isHybridAlgoEligible', 'isHyAlgoEligible', 'isEmAlgoEligible',
            'isInAlgoUniverse',
            "isBidAxe", "isAskAxe", "isMktAxe", "isAntiAxe",
        ],
        fromFrame='main',
        toFrame="meta"
    ),
    DataTask(
        task_name='meta_signals',
        func=meta_signals,
        merge_key='__meta_merge_key',
        strict_task_requirements=['init_meta', 'risk_transforms'],
        strict_col_requirements=[],
        fromFrame='main',
        toFrame="meta"
    ),
]

etf_tasks = [
    DataTask(
        task_name='etf_snapshot',
        func=etf_snapshot,
        merge_key='',
        fromFrame='main',
        toFrame="etf",
        max_retries=0
    ),
    DataTask(
        task_name='etf_constituents',
        func=etf_constituents,
        merge_key='',
        strict_task_requirements=[],
        fromFrame='main',
        toFrame="etf_bonds"
    ),
    DataTask(
        task_name='etf_mlcr_levels',
        func=etf_mlcr_levels,
        merge_key='isin',
        strict_task_requirements=['etf_constituents'],
        fromFrame='etf_bonds',
        toFrame="etf_bonds"
    ),
    DataTask(
        task_name='etf_bval_levels',
        func=etf_bval_levels,
        merge_key='isin',
        strict_task_requirements=['etf_constituents'],
        fromFrame='etf_bonds',
        toFrame="etf_bonds"
    ),
    DataTask(
        task_name='etf_macp_levels',
        func=etf_macp_levels,
        merge_key='isin',
        strict_task_requirements=['etf_constituents'],
        fromFrame='etf_bonds',
        toFrame="etf_bonds"
    ),
]

hedge_tasks = [
    DataTask(
        task_name='hedge_risk',
        broadcast_name="Hedge Risk",
        func=hedge_risk,
        merge_key='tnum',
        strict_col_requirements=[
            'benchmarkIsin', 'grossSize', 'netSize', 'side', 'unitDv01',
        ],
        strict_task_requirements=['risk_transforms', 'ust_bval_yields'],
        expected_col_provides={
            'hedgeRatio'    : pl.Float64,
            'grossHedgeSize': pl.Float64,
            'netHedgeSize'  : pl.Float64,
            'grossHedgeDv01': pl.Float64,
            'netHedgeDv01'  : pl.Float64,
            'hedgeDirection': pl.String,
        },
    ),
]


async def load_portfolio(base_pt, dates, portfolio_key, return_loader=False, broadcaster=None):
    schema = base_pt.collect_schema() if isinstance(base_pt, pl.LazyFrame) else base_pt.schema
    is_raw = 'rfqL0InstrumentId' in schema
    constituents = (await init_rfq_leg(base_pt)) if is_raw else base_pt
    all_tasks = (
            algo_tasks +
            benchmark_tasks +
            etf_tasks +
            funge_tasks +
            hedge_tasks +
            liq_score_tasks +
            meta_tasks +
            muni_tasks +
            positions_tasks +
            quote_tasks +
            ref_data_tasks +
            region_tasks +
            risk_metric_tasks +
            setup_tasks
    )
    loader = DataLoader(
        constituents,
        tasks=all_tasks,
        max_concurrency=20,
        dates=dates,
        frames={'raw': base_pt if is_raw else constituents},
        broadcaster=broadcaster,
        portfolio_key=portfolio_key,
        debug=True,
        optional_policy="background",
    )
    my_pt, timers, frames = await loader.run()
    await loader.log_duration("error")
    meta = frames.get('meta')
    if return_loader:
        return my_pt, meta, loader
    return my_pt, meta


if __name__=="__main__":
    async def main():

        from app.services.storage.portfolioManager import PortfolioManager
        DB = PortfolioManager(os.getenv("DB_BACKUP"))
        rfqListId, dates = await DB.select_random_rfq()
        await DB.disconnect()
        dates = parse_date(dates)

        rfqListId = 'LSTAON_20260416_BARX_CORI_NY9773758.1'
        # dates = "T-1"

        # rfqListId = '193796126'
        # dates = "2026.03.06"
        from app.server import get_db
        rfqListId, dates = await get_db().select_random_rfq()

        # rfqListId = '195852175'
        # dates = "T"

        base_pt = await query_pt_constituents_from_kdb(rfqListId, dates=dates)
        if (base_pt is None) or (base_pt.hyper.is_empty()):
            base_pt = await query_pt_constituents_from_kdb(rfqListId, dates=dates, region="EU")
        constituents = await init_rfq_leg(base_pt)
        print(constituents.hyper.shape, constituents.hyper.peek('isin'))

        dates = "T"
        all_tasks = (
                algo_tasks +
                benchmark_tasks +
                etf_tasks +
                funge_tasks +
                hedge_tasks +
                liq_score_tasks +
                meta_tasks +
                muni_tasks +
                positions_tasks +
                quote_tasks +
                ref_data_tasks +
                region_tasks +
                risk_metric_tasks +
                setup_tasks
        )

        # TODO:
        # MUNI BOOST FOR DESIGS
        # ASSET WEIGHTS? Or additional EM asset boosts
        # consolidate marks

        # does bondpositions have a ticker?
        loader = DataLoader(
            constituents,
            tasks=all_tasks,
            max_concurrency=20,
            dates=dates,
            frames={'raw': base_pt},
            broadcaster=None,
            debug=True,
            optional_policy="background",
        )
        my_pt, timers, frames = await loader.run(); await loader.log_duration(level="error")

        def export_frames(base="temp"):
            if not os.path.exists(f'app/data/{base}'):
                os.mkdir(f'app/data/{base}/')
            for name, df in frames.items():
                path = f'app/data/{base}/{name}.parquet'
                df.hyper.collect().write_parquet(path)

        def import_frames(base="temp"):
            d = {}
            import os
            base_dir = f'app/data/{base}/'
            for file in os.listdir(base_dir):
                name = file.split(".")[0]
                if name=='timers': continue
                d[name] = pl.read_parquet(os.path.join(base_dir, file)).lazy()
            return d

        frames = import_frames('help')
        my_pt = frames['main'].collect()
