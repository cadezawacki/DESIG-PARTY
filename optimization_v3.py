
from __future__ import annotations

import asyncio
import atexit
import datetime
import inspect
import json
import msgspec.json
import os
from functools import partial
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from typing import Union, List, Callable, Optional, Set
from functools import lru_cache
from app.helpers.fancy_text_helpers import UnicodeTextStyler
from app.services.cache.hyperCache._decorator import CachedFunction
from app.helpers.asyncThreadExecutor import AsyncThreadExecutor

import rapfiles
try:
    import re2 as _re
except ImportError:
    import re as _re

from app.helpers.polars_hyper_plugin import *
import polars as pl

from datetime import timedelta
from collections.abc import Sequence
import functools
from app.config.config import from_env
from app.helpers.asyncThreadExecutor import AsyncThreadExecutor
from app.helpers.string_helpers import clean_camel
from app.helpers.taskContext import TaskContext
from app.logs.logging import log
from app.helpers.type_helpers import ensure_list, ensure_lazy
from app.services.portfolio.s3 import S3Service
from app.helpers.hash import code_fingerprint
from app.services.loaders.kdb_queries_dev_v3 import hypercache
from app.helpers.lruCache import CacheDict

# Register new color
log.register('loader', color="violet")

# Tuneables ----------------------------------

# Providers
_PROVIDER_CACHE_MEM = CacheDict(cache_len=4096)
PROVIDER_CACHE_VERSION = 2

# Shared executor for synchronous provider cache reads at import time.
# Lazily created on first use, avoids creating N throwaway executors in DataTask.__post_init__.
_INIT_EXECUTOR = None
_INIT_EXECUTOR_REFCOUNT = 0

def _get_init_executor():
    global _INIT_EXECUTOR, _INIT_EXECUTOR_REFCOUNT
    if _INIT_EXECUTOR is None:
        _INIT_EXECUTOR = AsyncThreadExecutor(name="provider-cache-init")
        _INIT_EXECUTOR.start()
    _INIT_EXECUTOR_REFCOUNT += 1
    return _INIT_EXECUTOR

def _release_init_executor():
    global _INIT_EXECUTOR, _INIT_EXECUTOR_REFCOUNT
    if _INIT_EXECUTOR is None:
        return
    _INIT_EXECUTOR_REFCOUNT -= 1
    if _INIT_EXECUTOR_REFCOUNT <= 0:
        try:
            _INIT_EXECUTOR.shutdown()
        except Exception:
            pass
        _INIT_EXECUTOR = None
        _INIT_EXECUTOR_REFCOUNT = 0

# Defaults
OPTIONAL_POLICY_DEFAULT = "background" # "run" | "skip" | "background"
COLUMN_SCAN_MODE_DEFAULT = "adaptive"   # "off" | "adaptive" | "eager"
MAX_QUEUED_MERGES_DEFAULT = 64
MATERIALIZE_EVERY_N_MERGES_DEFAULT = 1  # 0 disables; >0 collects frame every N merges into that frame
MERGE_POLICY_DEFAULT = "coalesce_left"
MERGE_FAST_PATH_LEVEL = 2 # 0 (none), 1 (conservative speed ups), 2 (aggressive shortcuts)

_BASED_ON_REGEX_META = _re.compile(r'[\\^$*+?{}()\[\]|]')
_KNOWN_MERGE_POLICIES = frozenset({
    "default", "overwrite", "coalesce_left", "coalesce_right",
    "overwrite_non_null", "based_on",
})

# --------------------------------------------
def clean_column(x):
    if not isinstance(x, str): return x
    if x.startswith("_"): return x
    return clean_camel(x)

def collapse_list(
    expr: pl.Expr | str,
    dtype: pl.DataType,
    *,
    separators: str | Sequence[str] = ",",
    ignore_nulls: bool = True,
    parallel: bool = False,
) -> pl.Expr:
    value = pl.col(expr) if isinstance(expr, str) else expr
    depth = _list_depth(dtype)
    normalized = _normalize_separators(separators, depth)
    return _collapse(value, dtype, normalized, ignore_nulls, parallel)


def collapse_flat_list(
    expr: pl.Expr | str,
    *,
    separator: str = ",",
    ignore_nulls: bool = True,
) -> pl.Expr:
    value = pl.col(expr) if isinstance(expr, str) else expr
    return value.list.eval(pl.element().cast(pl.String)).list.join(
        separator,
        ignore_nulls=ignore_nulls,
    )


def _collapse(
    expr: pl.Expr,
    dtype: pl.DataType,
    separators: tuple[str, ...],
    ignore_nulls: bool,
    parallel: bool,
) -> pl.Expr:
    if dtype.base_type() is not pl.List:
        return expr.cast(pl.String)

    inner_expr = _collapse(
        pl.element(),
        dtype.inner,
        separators[1:],
        ignore_nulls,
        parallel,
    )

    return expr.list.eval(inner_expr, parallel=parallel).list.join(
        separators[0],
        ignore_nulls=ignore_nulls,
    )


def _list_depth(dtype: pl.DataType) -> int:
    depth = 0
    current = dtype
    while current.base_type() is pl.List:
        depth += 1
        current = current.inner
    if depth == 0:
        raise TypeError(f"expected a Polars List dtype, got {dtype!r}")
    return depth


def _normalize_separators(
    separators: str | Sequence[str],
    depth: int,
) -> tuple[str, ...]:
    if isinstance(separators, str):
        return (separators,) * depth

    values = tuple(separators)
    if not values:
        raise ValueError("separators must not be empty")
    if len(values) == 1:
        return values * depth
    if len(values) != depth:
        raise ValueError(
            f"expected 1 or {depth} separators for a {depth}-level list, got {len(values)}"
        )
    return values

def polars_dtype_to_string(dtype):
    if dtype is None: return "Null"
    if isinstance(dtype, str):
        s = dtype.strip()
        return s or "Null"
    try:
        return str(dtype)
    except Exception:
        return "Null"

def _provider_cols_to_dtype_str_map(raw_cols):
    # Normalizes cache payload column representations into:
    #   { "col": "DTypeString", ... }
    # Backwards compatible with old formats that were:
    #   - list[str]
    #   - dict[str -> str]
    #   - a single string (one col OR "colA,colB" OR "colA\ncolB")
    # Unknown dtype => "Null"
    if raw_cols is None:
        return {}

    if isinstance(raw_cols, dict):
        out = {}
        for k, v in raw_cols.items():
            if not isinstance(k, str):
                continue
            kk = clean_column(k)
            if not kk:
                continue
            if v is None:
                out[kk] = "Null"
            elif isinstance(v, str):
                out[kk] = v.strip() or "Null"
            else:
                out[kk] = polars_dtype_to_string(v)
        return out

    if isinstance(raw_cols, (bytes, bytearray)):
        try:
            raw_cols = raw_cols.decode("utf-8", errors="ignore")
        except Exception:
            return {}

    if isinstance(raw_cols, str):
        cols = _split_legacy_columns_text(raw_cols)
        if cols:
            return {c: "Null" for c in cols}
        # If it's a single col name that happened to start with "["/"{" we still guard:
        cc = clean_column(raw_cols)
        return {cc: "Null"} if cc else {}

    # list/tuple/set/etc => treat as list of names
    out = {}
    for c in _clean_list(raw_cols):
        out[c] = "Null"
    return out

def _provider_cache_state():
    st = getattr(_provider_cache_state, "_st", None)
    if st is None:
        st = {
            "read_inflight": {},   # path -> asyncio.Task
            "write_inflight": {},  # path -> asyncio.Task
            "write_token": {},     # path -> int (monotonic)
        }
        setattr(_provider_cache_state, "_st", st)
    return st

def _spawn_provider_cache_read(path: str):
    if not path: return None
    if path in _PROVIDER_CACHE_MEM: return None

    st = _provider_cache_state()
    inflight = st["read_inflight"]

    existing = inflight.get(path)
    if existing is not None and (not existing.done()):
        return existing

    try:
        from app.server import get_ctx
        at = get_ctx().spawn(_read_provider_cache(path))
    except Exception as e:
        log.error(f"Error spawning cache reader: {e}")
        try:
            loop = asyncio.get_running_loop()
            at = loop.create_task(_read_provider_cache(path))
        except Exception:
            return None

    inflight[path] = at
    try:
        at.add_done_callback(lambda _t, p=path: inflight.pop(p, None))
    except Exception:
        pass
    return at

async def _spawn_provider_cache_write(path: str):
    if not path: return None

    st = _provider_cache_state()
    inflight = st["write_inflight"]

    existing = inflight.get(path)
    if existing is not None and (not existing.done()):
        return existing

    try:
        from app.server import get_ctx
        at = get_ctx().spawn(_write_provider_cache_file(path))
    except Exception:
        # Fallback when server ctx isn't available; best-effort only.
        try:
            loop = asyncio.get_running_loop()
            at = loop.create_task(_write_provider_cache_file(path))
        except Exception:
            return None

    inflight[path] = at
    try:
        at.add_done_callback(lambda _t, p=path: inflight.pop(p, None))
    except Exception:
        pass
    return at

async def _write_provider_cache_file(path: str):
    st = _provider_cache_state()
    tokens = st["write_token"]

    while True:
        token_before = tokens.get(path, 0)
        record = _PROVIDER_CACHE_MEM.get(path)
        if not isinstance(record, dict):
            return

        try:
            payload = msgspec.json.encode(record)
        except Exception:
            try:
                payload = json.dumps(record, cls=SetEncoder).encode("utf-8")
            except Exception:
                return

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass

        tmp_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"
        try:
            await rapfiles.atomic_write_file_bytes(path, payload)
        except Exception as e:
            try:
                log.error(f"Failed to write provider cache {path}: {e}")
            except Exception:
                pass
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return

        if tokens.get(path, 0) == token_before:
            return
        # Else: loop and write the newest record.

def _split_legacy_columns_text(s: str):
    # Accepts legacy cache formats like:
    #  - "colA,colB"
    #  - "colA\ncolB"
    #  - "colA"
    # Returns cleaned column names.
    if not isinstance(s, str):
        return []
    t = s.strip()
    if not t:
        return []

    # If it looks like JSON-ish, let the caller parse it instead.
    if t[:1] in "[{":
        return []

    import re
    if ("\n" in t) or ("," in t):
        parts = [p.strip() for p in _re.split(r"[\n,]+", t) if p and p.strip()]
        return _clean_list(parts)

    return [clean_column(t)]

# --------------------------------------------

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def _safe_schema(obj):
    if obj is None:
        return {}

    try:
        return obj.hyper.schema()
    except Exception:
        pass

    try:
        if isinstance(obj, pl.LazyFrame):
            return obj.collect_schema()
        elif isinstance(obj, pl.DataFrame):
            return obj.schema
        else:
            return {}
    except Exception:
        pass

    return {}

def _clean_list(cols):
    if not cols: return []
    seen = set()
    for c in cols:
        if c is None: continue
        cc = clean_column(c) if isinstance(c, str) else c
        if not cc or not isinstance(cc, str): continue
        seen.add(cc)
    return list(seen)

def _as_lazy(df) -> pl.LazyFrame:
    if df is None: return pl.LazyFrame()
    return df.lazy() if isinstance(df, pl.DataFrame) else df

def _normalize_merge_key(merge_key):
    if merge_key is None:
        return ""

    if isinstance(merge_key, (set, tuple)):
        merge_key = list(merge_key)

    if isinstance(merge_key, list):
        out = [clean_column(x) for x in merge_key if isinstance(x, str) and x.strip()]
        return out if out else ""

    if isinstance(merge_key, str):
        mk = merge_key.strip()
        return clean_column(mk) if mk else ""

    return ""

def _merge_key_list(merge_key):
    mk = _normalize_merge_key(merge_key)
    if isinstance(mk, list): return mk
    if isinstance(mk, str) and mk != "": return [mk]
    return []

def _task_signature(func):

    if isinstance(func, CachedFunction):
        func = func.func

    sig = {
        "qualname": getattr(func, "__qualname__", None) or getattr(func, "__name__", None) or str(func),
        "module": getattr(func, "__module__", None),
        "file": None,
        "mtime": None,
        "hash": code_fingerprint(func)
    }
    try:
        f = inspect.getsourcefile(func) or inspect.getfile(func)
        sig["file"] = f
        if f and os.path.exists(f):
            sig["mtime"] = os.path.getmtime(f)
    except Exception:
        pass
    return sig

def _provider_cache_path(task_name: str):
    base = from_env('PROVIDER_BASE_PATH', default='./app/data/providers')
    return os.path.join(base, f"{task_name}.json")

def _normalize_provider_record(raw):
    # Backwards compatible inputs:
    #   - ["colA", "colB", ...]                               (legacy v0)
    #   - {"columns": [...]} / {"columns_union":[...], ...}   (legacy v1)
    #   - {"columns_union": {"col":"DTypeStr", ...}, ...}     (v2+)
    #   - "colA,colB" or "colA\ncolB"                         (very legacy / manual)
    if raw is None:
        return None

    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8", errors="ignore")
        except Exception:
            return None

    if isinstance(raw, str):
        cols = _split_legacy_columns_text(raw)
        cols_map = _provider_cols_to_dtype_str_map(cols)
        return {
            "version": 0,
            "columns_union": cols_map,
            "columns_last": dict(cols_map),
            "merge_key": [],
            "signature": None,
            "updated_at": None,
        }

    if isinstance(raw, (list, tuple, set)):
        cols_map = _provider_cols_to_dtype_str_map(list(raw))
        return {
            "version": 0,
            "columns_union": cols_map,
            "columns_last": dict(cols_map),
            "merge_key": [],
            "signature": None,
            "updated_at": None,
        }

    if isinstance(raw, dict):
        cols_union_raw = raw.get("columns_union", raw.get("columns", None))
        cols_last_raw = raw.get("columns_last", raw.get("columns", cols_union_raw))

        cols_union = _provider_cols_to_dtype_str_map(cols_union_raw)
        cols_last = _provider_cols_to_dtype_str_map(cols_last_raw)

        mk = raw.get("merge_key") or []
        if isinstance(mk, str):
            mk = [mk] if mk else []
        mk = _clean_list(mk)

        return {
            "version": int(raw.get("version", 1)),
            "columns_union": cols_union,
            "columns_last": cols_last,
            "merge_key": mk,
            "signature": raw.get("signature"),
            "updated_at": raw.get("updated_at"),
        }

    return None

async def _read_provider_cache(path: str):
    cached = _PROVIDER_CACHE_MEM.get(path, "___MISS___")
    if cached != "___MISS___":
        return cached

    if not path or (not os.path.exists(path)):
        _PROVIDER_CACHE_MEM[path] = None
        return None

    try:
        async with rapfiles.open(path, "rb") as f:
            data = await f.read()

        if not data:
            _PROVIDER_CACHE_MEM[path] = None
            return None

        raw = None

        try:
            raw = msgspec.json.decode(data, strict=False)
        except Exception:
            text = None
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                text = None

            if text:
                t = text.strip()
                if t:
                    try:
                        raw = json.loads(t)
                    except Exception:
                        try:
                            import ast
                            raw = ast.literal_eval(t)
                        except Exception:
                            raw = t  # treat as legacy "colA,colB" or "colA\ncolB"

        rec = _normalize_provider_record(raw)
        _PROVIDER_CACHE_MEM[path] = rec
        return rec

    except Exception as e:
        try:
            r = log.error(f"Provider cache read failed for {path}: {e}")
            if asyncio.iscoroutine(r):
                await r
        except Exception:
            pass
        _PROVIDER_CACHE_MEM[path] = None
        return None

def _signature_matches(a, b):
    if not a or not b: return False
    return (
            a.get("qualname") == b.get("qualname")
            and a.get("module") == b.get("module")
            and a.get("file") == b.get("file")
            and a.get("mtime") == b.get("mtime")
    )


def string_to_polars_dtype(dtype_str: str):
    if not isinstance(dtype_str, str):
        return dtype_str

    s = dtype_str.strip()
    if not s or s == "None":
        return pl.Null

    if s.startswith("pl."):
        s = s[3:].strip()

    # Fast path: no params
    if "(" not in s:
        if s == "String" and hasattr(pl, "Utf8"):
            s = "Utf8"
        dt = getattr(pl, s, None)
        return dt if dt is not None else pl.Null

    # List(inner)
    if s.startswith("List(") and s.endswith(")"):
        inner_s = s[5:-1].strip()
        inner_dt = string_to_polars_dtype(inner_s)
        try:
            return pl.List(inner_dt)
        except Exception:
            return pl.Null

    # Array(inner, width)
    if s.startswith("Array(") and s.endswith(")"):
        inner = s[6:-1].strip()
        parts = [p.strip() for p in inner.rsplit(",", 1)]
        if len(parts) == 2:
            inner_dt = string_to_polars_dtype(parts[0])
            try:
                width = int(parts[1])
            except Exception:
                width = None
            if width is not None and hasattr(pl, "Array"):
                try:
                    return pl.Array(inner_dt, width)
                except Exception:
                    try:
                        return pl.Array(inner_dt, width=width)
                    except Exception:
                        return pl.Null
        return pl.Null

    # Datetime(time_unit='us', time_zone=None|'UTC')
    if s.startswith("Datetime"):
        m = _re.search(r"time_unit\s*=\s*['\"]?([a-zA-Z]+)['\"]?", s)
        unit = m.group(1) if m else None

        m = _re.search(r"time_zone\s*=\s*(None|['\"]([^'\"]*)['\"])", s)
        tz = None
        if m:
            if m.group(1) != "None":
                tz = m.group(2)

        try:
            if unit is None and tz is None:
                return pl.Datetime
            return pl.Datetime(time_unit=unit or "us", time_zone=tz)
        except Exception:
            try:
                return pl.Datetime(unit or "us", tz)
            except Exception:
                return pl.Null

    # Duration(time_unit='us')
    if s.startswith("Duration"):
        m = _re.search(r"time_unit\s*=\s*['\"]?([a-zA-Z]+)['\"]?", s)
        unit = m.group(1) if m else None
        try:
            if unit is None:
                return pl.Duration
            return pl.Duration(time_unit=unit)
        except Exception:
            try:
                return pl.Duration(unit)
            except Exception:
                return pl.Null

    # Decimal(precision=38, scale=9)
    if s.startswith("Decimal"):
        m1 = _re.search(r"precision\s*=\s*(\d+)", s)
        m2 = _re.search(r"scale\s*=\s*(\d+)", s)
        if m1:
            precision = int(m1.group(1))
            scale = int(m2.group(1)) if m2 else 0
            if hasattr(pl, "Decimal"):
                try:
                    return pl.Decimal(precision=precision, scale=scale)
                except Exception:
                    try:
                        return pl.Decimal(precision, scale)
                    except Exception:
                        return pl.Null
        return pl.Null

    # Fallback: try base name
    base = s.split("(", 1)[0].strip()
    if base == "String" and hasattr(pl, "Utf8"):
        base = "Utf8"
    dt = getattr(pl, base, None)
    return dt if dt is not None else pl.Null

@dataclass
class DataTask:
    task_name: str
    func: callable  # async function returning pl.DataFrame

    merge_key: Union[str, List[str], Set[str]]
    fromFrame: str = "main"
    toFrame: str = "main"
    mergePolicy: Optional[str] = None # coalesce_left | coalesce_right | overwrite | overwrite_non_null | based_on
    dedupe_right: bool = True # when merging, dedupe the right side? (safer)
    based_on_col: Optional[str] = None
    based_on_comparator: Optional[Any] = "greater" # greater | lesser | callable(left, right) -> True means right wins
    cache: bool = False
    deep_cache: bool = False
    cache_ttl: Optional[timedelta] = None
    cache_kwargs: Dict = field(default_factory=dict)

    # strict requirements - ALL must be met
    strict_col_requirements: Union[list, set] = field(default_factory=list)
    strict_task_requirements: Union[list, set] = field(default_factory=list)

    # Failed requirements
    failed_task_requirements: Union[list, set] = field(default_factory=list)

    # Outputs from these tasks are ignored
    ignored_tasks: Union[list, set] = field(default_factory=list)
    global_ignore: bool = False
    isFinalizer: bool = False
    closeFrame: bool = False

    # expected_col_provides:
    #   - dict: {col: dtype} where dtype can be pl.DataType or "Float64"/"Utf8"/etc
    #   - list/set/tuple: ["colA","colB"] treated as dtype unknown (pl.Null)
    expected_col_provides: dict = field(default_factory=dict)
    actual_col_provides: dict = field(default_factory=dict)
    use_cached_providers: bool = True
    cache_providers: bool = True
    empty_on_fail: bool = True

    # Prevents auto-cancel on Optional tasks
    isOptional: bool = False
    critical_columns: Union[list, set] = field(default_factory=list)
    requestFullFrame: bool = True

    # Data source
    host: Optional[str] = None
    port: Optional[int] = None
    tbl_name: Optional[str] = None
    region: Optional[str] = None

    isTemp: bool = False
    backup_tasks: Union[list, set] = field(default_factory=list)

    startTime: float = None
    stopTime: float = None
    numExecutions: int = 0
    max_retries: int = 1
    retry_delay: Optional[float] = None # if not explicitly specified, DataLoader handles scaled per attempt
    timeout: Optional[float] = 30.0  # per-attempt timeout in seconds; None disables
    broadcast_name: Optional[str] = None

    frameContext: Union[list, set, str] = field(default_factory=list)
    results: Optional[Any] = None
    kwargs: dict = field(default_factory=dict)
    debug_input: pl.LazyFrame = field(default=None, init=False)

    # Internal
    _provider_path: str = field(default=None, init=False, repr=False)
    _provider_record: dict = field(default=None, init=False, repr=False)
    _provider_exists: bool = True
    _provider_sig: dict = field(default=None, init=False, repr=False)
    _last_outcome: str = field(default="pending", init=False, repr=False)  # success|failed|cancelled
    _last_error: str = field(default=None, init=False, repr=False)

    _future: Any = field(default=None, init=False, repr=False)  # concurrent.futures.Future from executor.submit()
    c_func: Optional[Callable] = field(default=None, init=False, repr=False) # cached version of the func

    def __post_init__(self):
        self.task_name = str(self.task_name).lower().strip()

        if self.fromFrame:
            self.fromFrame = str(self.fromFrame).lower().strip()

        if self.toFrame:
            self.toFrame = str(self.toFrame).lower().strip()

        # Merge key normalization
        self.merge_key = _normalize_merge_key(self.merge_key)

        # Column normalization
        self.strict_col_requirements = {clean_column(col) for col in ensure_list(self.strict_col_requirements)}

        self.critical_columns = {
            clean_column(col)
            for col in ensure_list(getattr(self, "critical_columns", None))
            if isinstance(col, str) and col.strip()
        }
        if self.requestFullFrame and (not self.critical_columns):
            self.requestFullFrame = False

        tokens = set()
        for x in ensure_list(self.strict_task_requirements):
            if x is None: continue
            s = str(x).lower().strip()
            if s: tokens.add(s)
        self.strict_task_requirements = tokens

        # Dynamic triggers are TASK names: normalize to lowercase
        self.failed_task_requirements = {
            str(x).lower().strip() for x in ensure_list(self.failed_task_requirements) if x is not None
        }

        if self.mergePolicy is None:
            self.mergePolicy = "based_on" if (self.based_on_col is not None) else MERGE_POLICY_DEFAULT

        # Task-name sets
        self.ignored_tasks = {str(x).lower().strip() for x in ensure_list(self.ignored_tasks) if x is not None}
        self.backup_tasks = {str(x).lower().strip() for x in ensure_list(self.backup_tasks) if x is not None}
        self.frameContext = {str(x).lower().strip() for x in ensure_list(self.frameContext) if x is not None}
        self.broadcast_name = None if (self.broadcast_name == '') else self.broadcast_name

        # expected_col_provides normalization
        declared = {}

        raw = self.expected_col_provides
        if raw is None:
            declared = {}
        elif isinstance(raw, dict):
            declared = dict(raw)
        else:
            for item in ensure_list(raw):
                if item is None:
                    continue
                if isinstance(item, str):
                    s = item.strip()
                    if s:
                        declared[s] = pl.Null
                elif isinstance(item, (tuple, list)) and len(item)==2 and isinstance(item[0], str):
                    k = item[0].strip()
                    if k:
                        declared[k] = item[1]

        declared = {
            clean_column(col): (string_to_polars_dtype(dtype) if dtype is not None else pl.Null)
            for col, dtype in declared.items()
            if isinstance(col, str) and col.strip()
        }

        self._provider_sig = _task_signature(self.func)
        self._provider_path = _provider_cache_path(self.task_name)

        if self.cache:
            primary_keys = self.cache_kwargs.pop('primary_keys', {'my_pt': [self.merge_key]})

            if self.deep_cache:
                deep = self.cache_kwargs.pop('deep', {"my_pt": True})
            else:
                deep = None
            ttl = self.cache_kwargs.pop('ttl', self.cache_ttl)
            lazy_hash_mode = self.cache_kwargs.pop('lazy_hash_mode', 'collect_keys')
            lazy_hash_keys = self.cache_kwargs.pop('lazy_hash_keys', ensure_set(self.strict_col_requirements) | set([self.merge_key]))
            # exclude_key_params = self.cache_kwargs.pop('exclude_key_params', ['kwargs'])
            key_params = self.cache_kwargs.pop('key_params', ['my_pt', 'region', 'dates'])

            cache_kwargs = {
                'key_params': key_params,
                'primary_keys':primary_keys,
                'deep':deep,
                'ttl':ttl,
                'lazy_hash_mode':lazy_hash_mode,
                'lazy_hash_keys':lazy_hash_keys,

                **self.cache_kwargs
            }

            self.c_func = hypercache.wrap(self.func, **cache_kwargs)

        if self.use_cached_providers:
            a = _get_init_executor()
            try:
                cached_rec = a.run(_read_provider_cache(self._provider_path)) or {}
            finally:
                _release_init_executor()
            cached_cols = cached_rec.get("columns_union") or cached_rec.get("columns_last") or {}
            if isinstance(cached_cols, dict):
                cached_map = {
                    clean_column(k): (string_to_polars_dtype(v) if v is not None else pl.Null)
                    for k, v in cached_cols.items()
                    if isinstance(k, str) and k.strip()
                }
            else:
                cached_map = {clean_column(c): pl.Null for c in _clean_list(cached_cols)}

            merged = dict(cached_map)
            merged.update(declared)  # declared wins
            declared = merged

        self.expected_col_provides = declared

    @property
    def duration(self, sigs=3, strict=False):
        stop = self.stopTime or (time.monotonic() if not strict else None)
        if self.startTime is None or stop is None:
            return None
        return round(stop - self.startTime, sigs)

    @property
    def run(self, suppress_cache=False):
        return self.c_func if self.cache and self.c_func and (not suppress_cache) else self.func

@dataclass
class FakeTask(DataTask):
    pass

@dataclass
class QuickCacheTask(DataTask):
    def __init__(self, **kwargs):
        inner = dict(kwargs.get('kwargs', {}))
        inner['__cache_only'] = True
        super().__init__(**{**kwargs, "kwargs": inner})


async def echo(columns=None, my_pt=None, *args, **kwargs):
    if my_pt is None: return
    if columns == "*": return my_pt
    columns = ensure_list(columns)
    if columns:
        return my_pt.hyper.ensure_columns(columns).select(columns)
    return None

@dataclass
class EchoTask(DataTask):
    def __init__(self, columns, **kwargs):
        kwargs['func'] = partial(echo, columns)
        kwargs['use_cached_providers'] = False
        kwargs['cache_providers'] = False
        super().__init__(**kwargs)
        self.strict_col_requirements |= ensure_set(columns)


class DependencyCycleError(Exception): pass

class DependencyGraph:
    def __init__(self, tasks: Union[list, set, dict]):

        if isinstance(tasks, dict):
            self.task_dict = tasks
            self.task_names = set(self.task_dict.keys())
            self.data_tasks = list(self.task_dict.values())
        else:
            self.data_tasks = [t for t in ensure_list(tasks) if isinstance(t, DataTask)]
            self.task_names = {x.task_name for x in self.data_tasks if hasattr(x, "task_name")}
            self.task_dict = {t.task_name: t for t in self.data_tasks}

        self.global_ignores = {t.task_name for t in self.data_tasks if getattr(t, "global_ignore", False)}
        self.finalizers = {t.task_name for t in self.data_tasks if getattr(t, "isFinalizer", False)}

        # Providers/consumers
        #   - column_providers_all: includes finalizers
        #   - column_providers_non_finalizer: excludes finalizers
        #   - column_finalizers: finalizers only
        self.column_providers_all = defaultdict(lambda: defaultdict(set))          # frame -> col -> set(task)
        self.column_providers_non_finalizer = defaultdict(lambda: defaultdict(set))  # frame -> col -> set(task)
        self.column_finalizers = defaultdict(lambda: defaultdict(set))            # frame -> col -> set(task)
        self.column_consumers = defaultdict(lambda: defaultdict(set))             # frame -> col -> set(task)
        self._compute_column_maps()

        # Backwards-compat alias: existing code expects dep_graph.column_providers
        # to mean "providers for a column". We keep it as "ALL providers",
        # and task-aware filtering is done via _providers_for_task_column().
        self.column_providers = self.column_providers_all

        self.strict_requirements = {}  # task -> set(tasks that must complete before it, excluding dynamic triggers)
        self._compute_static_requirements()

        self.upstream_deps = {}
        self.downstream_deps = defaultdict(set)
        self._compute_transitive_deps()

    def _compute_column_maps(self):
        for t in self.data_tasks:
            provides = t.expected_col_provides or {}
            if not isinstance(provides, dict):
                provides = {c: pl.Null for c in _clean_list(provides)}

            for col in provides.keys():
                self.column_providers_all[t.toFrame][col].add(t.task_name)
                if getattr(t, "isFinalizer", False):
                    self.column_finalizers[t.toFrame][col].add(t.task_name)
                else:
                    self.column_providers_non_finalizer[t.toFrame][col].add(t.task_name)

            for col in (t.strict_col_requirements or ()):
                self.column_consumers[t.fromFrame][col].add(t.task_name)

    def _ignored_for_task(self, task: DataTask):
        return ensure_set(getattr(task, "ignored_tasks", None)) | ensure_set(self.global_ignores)

    @classmethod
    def _task_provides_col(cls, task: DataTask, col: str) -> bool:
        provides = getattr(task, "expected_col_provides", None) or {}
        if isinstance(provides, dict):
            return col in provides
        return col in set(_clean_list(provides))

    def _providers_for_task_column(self, task: DataTask, frame: str, col: str):
        """
        Provider selection as seen by `task` for column `col` in `frame`.

        Rules:
          - Start with ALL providers (including finalizers)
          - Remove global ignores, task ignores, and self
          - If `task` provides `col`, remove finalizers that provide `col`
            (prevents provider<->finalizer dependency loops)
        """
        base = self.column_providers_all.get(frame, {}).get(col, set())
        if not base:
            return set()

        ignored = self._ignored_for_task(task)
        provs = base - ignored - {task.task_name}

        # Provider masking rule: providers must not depend on finalizers for overlapping cols.
        if provs and self._task_provides_col(task, col):
            finals = self.column_finalizers.get(frame, {}).get(col, set())
            if finals:
                provs -= finals

        return provs

    def _finalizer_gate_met(self, task: DataTask, completed_tasks: set):
        """
        Finalizer run condition:
          A finalizer is runnable only when *no non-finalizer provider* remains
          for any column it provides (in its toFrame), excluding ignored/global ignores.

        Finalizers do not wait on other finalizers (otherwise F<->F loops are easy).
        """
        if not getattr(task, "isFinalizer", False):
            return True

        provides = task.expected_col_provides or {}
        if not provides:
            return True

        if not isinstance(provides, dict):
            provides = {c: pl.Null for c in _clean_list(provides)}

        ignored = self._ignored_for_task(task)
        frame = task.toFrame

        for col in provides.keys():
            non_final = self.column_providers_non_finalizer.get(frame, {}).get(col, set())
            if not non_final:
                continue

            remaining = (non_final - ignored - {task.task_name})
            if remaining and (not remaining.issubset(completed_tasks)):
                return False

        return True

    def _compute_static_requirements(self):
        task_names = set(self.task_dict.keys())

        for t in self.data_tasks:
            deps = set()

            # strict_task_requirements supports BOTH:
            #   - task names (case-insensitive)
            #   - column names (expanded to provider tasks on t.fromFrame)
            for token in (t.strict_task_requirements or set()):
                if token is None:
                    continue
                raw = str(token).strip()
                if not raw:
                    continue

                as_task = raw.lower()
                if as_task in task_names:
                    if as_task != t.task_name:
                        # explicit task edges always respected (user intent),
                        # except global/task ignores handled in scheduler logic.
                        if as_task not in self._ignored_for_task(t):
                            deps.add(as_task)
                    continue

                col = clean_column(raw)
                if not col:
                    continue
                provs = self._providers_for_task_column(t, t.fromFrame, col)
                if provs:
                    deps |= provs

            self.strict_requirements[t.task_name] = deps

    def _compute_transitive_deps(self):
        graph = {name: set(deps) for name, deps in self.strict_requirements.items()}

        def dfs(n, seen):
            if n in seen:
                return set()
            seen.add(n)
            out = set(graph.get(n, set()))
            for d in list(out):
                out |= dfs(d, seen)
            return out

        for name in self.task_names:
            self.upstream_deps[name] = dfs(name, set())

        for name, deps in self.upstream_deps.items():
            for d in deps:
                self.downstream_deps[d].add(name)

    def get_upstream(self, task_name):
        return self.upstream_deps.get(task_name, set())

    def get_downstream(self, task_name):
        return self.downstream_deps.get(task_name, set())

    def detect_dependency_loops(self, verbose=False):
        graph = defaultdict(set)
        task_names = set(self.task_dict.keys())

        for t in self.data_tasks:
            name = t.task_name

            # strict_task_requirements tokens: task or column
            for token in (t.strict_task_requirements or set()):
                if token is None:
                    continue
                raw = str(token).strip()
                if not raw:
                    continue

                as_task = raw.lower()
                if as_task in task_names:
                    if as_task != name:
                        graph[name].add(as_task)
                else:
                    col = clean_column(raw)
                    if not col:
                        continue
                    provs = self._providers_for_task_column(t, t.fromFrame, col)
                    if provs:
                        graph[name].update(provs)

            # strict_col_requirements (implicit edges to providers)
            for col in (t.strict_col_requirements or ()):
                provs = self._providers_for_task_column(t, t.fromFrame, col)
                if provs:
                    graph[name].update(provs)

        visited = set()
        stack = []
        stack_set = set()
        cycles = []

        def edge_info(a, b):
            ta = self.task_dict.get(a)
            if not ta:
                return "unknown"

            # explicit task edge?
            for tok in (ta.strict_task_requirements or set()):
                if tok is None:
                    continue
                raw = str(tok).strip()
                if raw and raw.lower() == b:
                    return "explicit"

            # column token edge?
            col_hits = []
            for tok in (ta.strict_task_requirements or set()):
                if tok is None:
                    continue
                raw = str(tok).strip()
                if not raw:
                    continue
                if raw.lower() in task_names:
                    continue
                col = clean_column(raw)
                if not col:
                    continue
                provs = self._providers_for_task_column(ta, ta.fromFrame, col)
                if b in provs:
                    col_hits.append(col)

            # strict_col edge?
            for c in (ta.strict_col_requirements or ()):
                provs = self._providers_for_task_column(ta, ta.fromFrame, c)
                if b in provs:
                    col_hits.append(c)

            if not col_hits:
                return "unknown"
            if len(col_hits) == 1:
                return f"col:{col_hits[0]}"
            return f"cols:{col_hits[0]}+"

        def dfs(n):
            visited.add(n)
            stack.append(n)
            stack_set.add(n)

            for nb in graph.get(n, set()):
                if nb not in visited:
                    dfs(nb)
                elif nb in stack_set:
                    i = stack.index(nb)
                    cyc = stack[i:] + [nb]
                    cycles.append([(cyc[j], edge_info(cyc[j], cyc[j + 1])) for j in range(len(cyc) - 1)])

            stack.pop()
            stack_set.remove(n)

        for n in list(graph.keys()):
            if n not in visited:
                dfs(n)

        formatted = []
        seen = set()
        for cyc in cycles:
            nodes = tuple(sorted(x for x, _ in cyc))
            if nodes in seen:
                continue
            seen.add(nodes)
            if not verbose:
                chain = [cyc[0][0]]
                for (a, info), (b, _) in zip(cyc, cyc[1:] + [cyc[0]]):
                    if info.startswith("col:"):
                        chain.append(f" --[{info[4:]}]--> {b}")
                    elif info.startswith("cols:"):
                        chain.append(f" --[{info[5:]}]--> {b}")
                    else:
                        chain.append(f" --[explicit]--> {b}")
                formatted.append("Loop: " + "".join(chain))
            else:
                lines = ["Dependency loop detected:"]
                for (a, info), (b, _) in zip(cyc, cyc[1:] + [cyc[0]]):
                    ta = self.task_dict.get(a)
                    lines.append(f"  {a} -> {b} ({info})")
                    if ta:
                        provs = ta.expected_col_provides or {}
                        prov_list = list(provs.keys()) if isinstance(provs, dict) else _clean_list(provs)
                        lines.append(f"    isFinalizer: {bool(getattr(ta, 'isFinalizer', False))}")
                        lines.append(f"    provides: {', '.join(prov_list) if prov_list else '∅'}")
                        lines.append(f"    requires: {', '.join(ta.strict_col_requirements) if ta.strict_col_requirements else '∅'}")
                formatted.append("\n".join(lines))

        return formatted

    def column_requirements_met(self, task, completed_tasks, column_status):
        for col in (task.strict_col_requirements or ()):
            # 1) Column is DONE
            if column_status.get(task.fromFrame, {}).get(col, False):
                continue

            # 2) No potential providers left (as seen by this task)
            provs = self._providers_for_task_column(task, task.fromFrame, col)
            if not provs:
                continue

            # 3) Still some providers left to run
            if not provs.issubset(completed_tasks):
                return False

        return True

    def get_ready_tasks(self, completed_tasks, running_task_names, column_status):
        ready = set()
        all_started = set(completed_tasks) | set(running_task_names)

        for name, t in self.task_dict.items():
            # 1) Already Done / Running
            if name in all_started:
                continue

            # 2) Static (task-based) deps must be done
            if not self.strict_requirements.get(name, set()).issubset(completed_tasks):
                continue

            # 3) Columns requirements must be satisfiable / settled
            if not self.column_requirements_met(t, completed_tasks, column_status):
                continue

            # 4) Finalizer gate (no remaining non-finalizer providers for its output cols)
            if not self._finalizer_gate_met(t, completed_tasks):
                continue

            ready.add(name)

        return ready

    def blockers(self, task, completed_tasks, column_status):
        b = set()
        b |= (self.strict_requirements.get(task.task_name, set()) - set(completed_tasks))

        for col in (task.strict_col_requirements or ()):
            if column_status.get(task.fromFrame, {}).get(col, False):
                continue

            provs = self._providers_for_task_column(task, task.fromFrame, col)
            if provs and (not provs.issubset(completed_tasks)):
                b |= (provs - set(completed_tasks))

        # Include finalizer gate blockers for diagnostics (not for scheduling)
        if getattr(task, "isFinalizer", False):
            provides = task.expected_col_provides or {}
            if not isinstance(provides, dict):
                provides = {c: pl.Null for c in _clean_list(provides)}
            ignored = self._ignored_for_task(task)
            frame = task.toFrame

            for col in provides.keys():
                non_final = self.column_providers_non_finalizer.get(frame, {}).get(col, set())
                if not non_final:
                    continue
                remaining = (non_final - ignored - {task.task_name}) - set(completed_tasks)
                if remaining:
                    b |= remaining

        return b

def _is_regex_pattern(val):
    if isinstance(val, _re.Pattern): return True
    return isinstance(val, str) and _BASED_ON_REGEX_META.search(val) is not None

LOG_LEVELS = {
    "debug": 0,
    "warning": 1,
    "error": 2
}

class DataLoader:
    def __init__(
            self,
            main_df: Union[pl.DataFrame, pl.LazyFrame],
            tasks: list,
            dates: Optional[str] = None,

            max_concurrency: Optional[int] = 16,
            use_providers_cache: Optional[bool] = True,
            cleanup_func: Optional[Callable] = None,

            loop: Optional[Any] = None,
            portfolio_key: Optional[str] = None,

            toast_hint="META",
            broadcaster: Optional[Any] = None,
            debug: bool = False,

            optional_policy: Optional[str] = OPTIONAL_POLICY_DEFAULT,
            column_scan_mode: Optional[str] = COLUMN_SCAN_MODE_DEFAULT,
            max_queued_merges: Optional[int] = MAX_QUEUED_MERGES_DEFAULT,
            materialize_every_n_merges: Optional[int] = MATERIALIZE_EVERY_N_MERGES_DEFAULT,

            suppress_cache: bool = False,
            force_cache: bool = False,

            frames: Optional[Dict] = None
    ):
        self.dates = dates
        self.max_concurrency = max_concurrency
        self.use_providers_cache = use_providers_cache
        self.broadcaster = broadcaster
        self.portfolio_key = portfolio_key
        self.toast_hint = toast_hint
        self.cleanup_func = cleanup_func
        self.debug = debug

        self.loop = loop

        self.s3 = None
        self.executor = None
        self._force_quit = asyncio.Event()
        self._scheduler_wake = asyncio.Event()
        self._broadcasts = []

        self.optional_policy = optional_policy
        self.column_scan_mode = column_scan_mode
        self.max_queued_merges = int(max_queued_merges) if max_queued_merges else 0
        self.materialize_every_n_merges = int(materialize_every_n_merges) if materialize_every_n_merges else 0

        self.start_time = None
        self.stop_time = None

        self.frames = frames or {}
        self.locks = {}
        self.schema_cache = {}
        self._merge_counts = defaultdict(int)

        # Main frame normalization
        s = main_df.hyper.schema()
        main_df = main_df.collect() if isinstance(main_df, pl.LazyFrame) else main_df
        main_df = main_df.rename({c: clean_column(c) for c in s}, strict=False)

        self.frames["main"] = main_df.lazy()
        self.main_df = main_df

        self.initial_row_count = int(self.main_df.hyper.height())

        # Task state
        self.tasks = {t.task_name: t for t in tasks if hasattr(t, 'task_name') and (not isinstance(t, FakeTask))}
        self.task_to_name = {}  # asyncio.Task -> name
        self.name_to_task = {} # name -> asyncio.Task
        self.task_status = {name: "pending" for name in self.tasks.keys()}
        self.task_outcome = {name: "pending" for name in self.tasks.keys()}  # pending|running|success|failed|cancelled

        self._cancel_requested = set() # names requested for cancel
        self._cancel_events = {} # task_name -> threading.Event
        self._detached = set() # asyncio.Task objects detached from scheduler tracking

        self.suppress_cache = suppress_cache
        self.force_cache = force_cache

        self.frame_writers = {}
        self.frame_creators = {}

        # Ensure all referenced frames exist in dict
        for task in tasks:
            if task.fromFrame not in self.frames:
                self.frames[task.fromFrame] = None
            if task.toFrame not in self.frames:
                self.frames[task.toFrame] = None

            if task.toFrame not in self.frame_writers:
                self.frame_writers[task.toFrame] = set()
            self.frame_writers[task.toFrame].add(task.task_name)

            if (task.merge_key==''):
                if (task.toFrame not in self.frame_creators):
                    self.frame_creators[task.toFrame] = task.task_name
                else:
                    raise DependencyCycleError('Multiple writers for the same frame')

        self.frames_open = {frame: True for frame in self.frames.keys()}

        joined_task_names = "|".join(self.tasks.keys())
        for task in tasks:

            tasks_to_check = task.strict_task_requirements.copy()
            for req in tasks_to_check:
                if _is_regex_pattern(req):
                    task.strict_task_requirements.discard(req)
                    matches = re.findall(req, joined_task_names)
                    if matches:
                        valid_matches = [m[1] for m in matches if (m is not None) and (len(m)>1)]
                        for v in valid_matches:
                            if v != task.task_name:
                                task.strict_task_requirements.add(v)

            creator = self.frame_creators.get(task.toFrame)
            if (task.task_name != creator) and (creator not in task.strict_task_requirements):
                task.strict_task_requirements.add(creator)
            source = self.frame_creators.get(task.fromFrame)
            if (source is not None) and (source != 'main') and (source not in task.strict_task_requirements):
                task.strict_task_requirements.add(source)

        for frame_name in self.frames.keys():
            self.locks[frame_name] = asyncio.Lock()

        # Creators

        # Providers map (static, based on provides lists)
        self.providers = defaultdict(lambda: defaultdict(set))
        for t in tasks:
            for col in t.expected_col_provides:
                self.providers[t.toFrame][col].add(t.task_name)

        self.temp_columns = {col for t in tasks if t.isTemp for col in t.expected_col_provides}
        self.dep_graph = DependencyGraph(self.tasks)
        dependency_loops = self.dep_graph.detect_dependency_loops(verbose=False)
        if dependency_loops:
            msg = "\n".join(dependency_loops)
            log.critical(msg)
            raise DependencyCycleError(msg)

        self.global_ignores = self.dep_graph.global_ignores

        # Scheduler sets
        self.completed_tasks = set()
        self.pending_tasks = set(self.tasks.keys())
        self.running_tasks = set()  # asyncio.Task objects
        self.ctx = TaskContext()
        self._completed_order = []

        # Column completeness cache: frame -> col -> bool
        self.column_complete = defaultdict(dict)

        # Queues
        self.queued_dfs = []  # (task_name, df)
        self._provider_write_queue = {}  # path -> record dict
        self.retry_delay = 0.1

        # Semaphores
        self.sem = asyncio.Semaphore(self.max_concurrency)

        # Optional task policy: pre-skip
        if self.optional_policy == "skip":
            for name, t in self.tasks.items():
                if t.isOptional:
                    self._skip_task(name, "optional_policy=skip")

    def _get_or_create_cancel_event(self, task_name: str):
        import threading
        ev = self._cancel_events.get(task_name)
        if ev is None:
            ev = threading.Event()
            self._cancel_events[task_name] = ev
        return ev

    def __getitem__(self, key):
        return self.tasks.get(key, None)

    @property
    def duration(self, sigs=3):
        if self.start_time is None or self.stop_time is None:
            return "Loader has not ran yet."
        return round(self.stop_time - self.start_time, sigs)

    @classmethod
    def _consume_task_terminal_state(cls, at: asyncio.Task):
        try:
            if at.cancelled():
                return
            _ = at.exception()
        except asyncio.CancelledError:
            return
        except Exception:
            return

    def _detach_running_task(self, at: asyncio.Task, name: str, reason: str):
        self.running_tasks.discard(at)
        self.task_to_name.pop(at, None)
        self.name_to_task.pop(name, None)
        self.ctx.discard(at)

        # Ensure no "Task exception was never retrieved" warnings if it errors.
        at.add_done_callback(self._consume_task_terminal_state)
        self._detached.add(at)

        self._cancel_requested.add(name)
        self.pending_tasks.discard(name)
        self.completed_tasks.add(name)
        self.task_outcome[name] = "cancelled"
        self.task_status[name] = f"cancelled: {reason}"

        # Wake the scheduler so it doesn't sit blocked on asyncio.wait.
        self._scheduler_wake.set()

    @classmethod
    @lru_cache(maxsize=128)
    def _func_accepts_kw(cls, func, kw: str) -> bool:
        try:
            sig = inspect.signature(func)
        except Exception:
            return False

        for p in sig.parameters.values():
            if p.kind==inspect.Parameter.VAR_KEYWORD:
                return True
            if p.name==kw:
                return True
        return False

    async def broadcast_log(self, msg):
        try:
            if self.broadcaster and self.portfolio_key:
                self.ctx.spawn(self.broadcaster._send_loading_toast(
                    self.portfolio_key, msg, toast_hint=self.toast_hint
                ), name='loading-toast')
        except Exception as e:
            log.error(f"Error while broadcasting.. {e}")

    def invalidate_schema(self, frame_name):
        self.schema_cache.pop(frame_name, None)

    def get_schema(self, frame_name):
        if frame_name not in self.schema_cache:
            lf = self.frames.get(frame_name)
            s = _safe_schema(lf)
            if s is not None:
                self.schema_cache[frame_name] = s
        return self.schema_cache[frame_name]

    def _task_critical_columns(self, task: DataTask):
        raw = getattr(task, "critical_columns", None)
        if not raw: return set()
        return set(raw)

    def _get_critical_cols_by_frame(self):
        cached = getattr(self, "_critical_cols_by_frame_cache", None)
        if cached: return cached

        out = defaultdict(set)
        for t in self.tasks.values():
            if not getattr(t, "isOptional", False):
                continue
            cols = self._task_critical_columns(t)
            if cols:
                out[t.toFrame].update(cols)

        self._critical_cols_by_frame_cache = out
        return out

    def get_complete_columns(self, frame_name):
        st = self.column_complete.get(frame_name, {})
        return dict(st) if st else {}

    async def _task_has_unfilled_critical_columns(self, task: DataTask):
        if not getattr(task, "isOptional", False):
            return False

        cols = self._task_critical_columns(task)
        if not cols: return False

        st = self.get_complete_columns(task.toFrame)
        for c in cols:
            if not st.get(c, False):
                await log.loader(f'Cannot cancel {task.task_name} because missing in {c}', color="#d6a83c")
                return True

        await log.loader(f'Cancelling {task.task_name} because critical columns have been filled.', color="green")
        return False

    def _compute_optional_keep_set_for_missing_critical_columns(self, remaining_names: set):
        crit_by_frame = self._get_critical_cols_by_frame()
        if not crit_by_frame:
            return set(), []

        missing = []
        for frame, cols in crit_by_frame.items():
            if not cols:
                continue
            st = self.get_complete_columns(frame)
            for c in cols:
                if not st.get(c, False):
                    missing.append((frame, c))

        if not missing:
            return set(), []

        # Seed: remaining tasks that are known providers for missing critical columns
        keep = set()
        col_providers = self.dep_graph.column_providers
        for frame, col in missing:
            provs = col_providers.get(frame, {}).get(col, set())
            if provs:
                keep |= (provs & remaining_names)

        if not keep:
            # Missing critical cols exist, but no remaining task is a known provider.
            return set(), missing

        # Closure over blockers so we don't cancel tasks needed for the keep-set to run.
        stack = list(keep)
        while stack:
            name = stack.pop()
            t = self.tasks.get(name)
            if t is None:
                continue
            try:
                blockers = self.dep_graph.blockers(t, self.completed_tasks, self.column_complete)
            except Exception:
                blockers = set()

            for b in blockers:
                if b in remaining_names and b not in keep:
                    keep.add(b)
                    stack.append(b)

        return keep, missing

    def _running_task_names_by_task(self):
        out = set()
        for t in self.running_tasks:
            n = self.task_to_name.get(t)
            if n:
                out.add(n)
        return out

    def _running_task_names(self):
        return set(self.name_to_task.keys())

    def _eager_cancel_optional_tasks(self):

        crit_by_frame = self._get_critical_cols_by_frame()
        if not crit_by_frame:
            return

        running_names = self._running_task_names()
        if not running_names:
            return

        for name in list(running_names):
            if name in self._cancel_requested:
                continue
            t = self.tasks.get(name)
            if t is None or not t.isOptional:
                continue

            crit_cols = t.critical_columns
            if not crit_cols:
                continue

            st = self.column_complete.get(t.toFrame, {})
            all_filled = all(st.get(c, False) for c in crit_cols)
            if not all_filled:
                continue

            # All critical columns for this optional task are filled.
            # Signal cancellation eagerly.
            log.loader(f"Eager cancel: {name} — critical columns filled", color="green")
            self._cancel_requested.add(name)

            ev = self._cancel_events.get(name)
            if ev is not None:
                ev.set()

            # Cancel the inner executor work.
            if t._future is not None:
                ex = self.executor
                if ex is not None:
                    try:
                        ex.cancel_inner(t._future)
                    except Exception:
                        pass
                try:
                    t._future.cancel()
                except Exception:
                    pass

            # Cancel the outer asyncio.Task so _run_task_with_retries
            # gets CancelledError at its next await point.
            at = self.name_to_task.get(name)
            if at is not None and not at.done():
                at.cancel()

    async def _batch_initialize_column_status(self):
        required_cols = defaultdict(set)

        for t in self.tasks.values():
            for col in t.strict_col_requirements:
                required_cols[t.fromFrame].add(col)

        crit_by_frame = self._get_critical_cols_by_frame()
        for frame, cols in crit_by_frame.items():
            if cols:
                required_cols[frame].update(cols)

        main_required = required_cols.get("main", set())
        available = set(_safe_schema(self.main_df).keys())
        cols_to_check = list(main_required & available)

        if cols_to_check:
            exprs = [pl.col(c).is_null().sum().alias(c) for c in cols_to_check]
            try:
                res_df = await self.main_df.lazy().select(exprs).collect_async()
                res = res_df.to_dict(as_series=False)
                for c in cols_to_check:
                    self.column_complete["main"][c] = (res.get(c, [1])[0]==0)
            except Exception:
                for c in cols_to_check:
                    self.column_complete["main"][c] = False

        for frame, cols in required_cols.items():
            for c in cols:
                if c not in self.column_complete[frame]:
                    self.column_complete[frame][c] = False

    async def _warm_provider_cache(self):
        if not self.use_providers_cache: return

        from app.services.loaders.kdb_queries_dev_v3 import hypercache

        # Ensure all reads are spawned (some may already be in-flight from DataTask.__post_init__)
        unique_paths = set()
        read_tasks = []
        for t in self.tasks.values():
            if not t.use_cached_providers:
                continue
            p = t._provider_path or _provider_cache_path(t.task_name)
            if not p or p in unique_paths:
                continue
            unique_paths.add(p)
            rt = _spawn_provider_cache_read(p)
            if rt is not None:
                read_tasks.append(rt)

        if read_tasks:
            await asyncio.gather(*read_tasks, return_exceptions=True)

        # Apply loaded caches into expected_col_provides (declared dtypes win)
        changed_any = False
        for t in self.tasks.values():
            if not t.use_cached_providers:
                continue
            rec = await _read_provider_cache(t._provider_path) or {}
            my_hash = code_fingerprint(t.func)
            cached_hash = rec.get('signature', {}).get('hash', None)
            if (cached_hash is not None) and (cached_hash != my_hash):
                if os.path.exists(t._provider_path):
                    os.remove(t._provider_path)
                    try:
                        await hypercache.clear(t.func.__name__)
                    except Exception:
                        pass
                    log.notify(f'Removing stale cache: {t._provider_path}: {my_hash} vs {cached_hash}')
                continue
            cols = rec.get("columns_union") or rec.get("columns_last") or {}
            if not cols:
                continue

            if isinstance(cols, dict):
                cached_map = {
                    clean_column(k): (string_to_polars_dtype(v) if v is not None else pl.Null)
                    for k, v in cols.items()
                    if isinstance(k, str) and k.strip()
                }
            else:
                cached_map = {clean_column(c): pl.Null for c in _clean_list(cols)}

            if not cached_map:
                continue

            merged = dict(cached_map)
            merged.update(t.expected_col_provides or {})  # declared wins
            if merged!=(t.expected_col_provides or {}):
                t.expected_col_provides = merged
                changed_any = True

        if not changed_any:
            return

        # Rebuild provider maps + dependency graph with the augmented provides
        self.providers = defaultdict(lambda: defaultdict(set))
        for t in self.tasks.values():
            for col in (t.expected_col_provides or {}):
                self.providers[t.toFrame][col].add(t.task_name)

        self.temp_columns = {
            col for t in self.tasks.values() if t.isTemp for col in (t.expected_col_provides or {})
        }

        self.dep_graph = DependencyGraph(self.tasks)
        dependency_loops = self.dep_graph.detect_dependency_loops(verbose=False)
        if dependency_loops:
            msg = "\n".join(dependency_loops)
            log.critical(msg)
            raise DependencyCycleError(msg)

    async def log_duration(self, level="debug"):
        key = self.main_df.hyper.ensure_columns('portfolioKey').hyper.peek('portfolioKey')
        str_key = f'-------------------------------------- {key}'
        if self.duration <= 10:
            color = "emerald"
        elif self.duration <= 30:
            color = "#a67400"
        else:
            color = "#a60000"

        log.blank("", hide_title=True, show_time=False)
        await log.loader(str_key, color=color, show_time=False)
        for tn in self._completed_order:
            task = self.tasks.get(tn)
            dur = task.duration
            status = self.task_status[tn]
            lvl = LOG_LEVELS.get(level, 0)

            if status == 'cancelled':
                if lvl <= 0:
                    await log.loader(f"[{tn}] {UnicodeTextStyler().format(text=status,strike=True)}", color="#33373d", show_time=False)
            elif status.startswith('failed'):
                await log.loader(f"[{tn}] {status}", color="#990f00", show_time=False)
            elif dur < 1:
                if lvl <= 0:
                    await log.loader(f"[{tn}] {status}", color="#335266", show_time=False)
            elif dur < 5:
                if lvl <= 1:
                    await log.loader(f"[{tn}] {status}", color="#b3a700", show_time=False)
            else:
                await log.loader(f"[{tn}] {status}", color="#cc7f02", show_time=False)
        comp_str = f"Completed: {self.duration}s"
        fin_str = "-"*max(0,(len(str_key)-(1+len(comp_str)))) + " " + comp_str
        await log.loader(fin_str, color=color, show_time=False)
        log.blank("", hide_title=True, show_time=False)

    async def run(self, debug=False):
        self.start_time = time.monotonic()
        await self._warm_provider_cache()
        await self._batch_initialize_column_status()
        res = await self._run_full()
        self.stop_time = time.monotonic()
        return res

    def _requirements_met(self, task):
        required_tasks = self.dep_graph.strict_requirements.get(task.task_name, set())  # FIX
        if not required_tasks.issubset(self.completed_tasks):
            return False
        return self.dep_graph.column_requirements_met(task, self.completed_tasks, self.column_complete)

    async def stop(self):
        cancel_exc = None
        try:
            self._force_quit.set()
            self._scheduler_wake.set()

            # Cancel running scheduler tasks
            running = list(self.running_tasks) if self.running_tasks else []
            for t in running:
                try:
                    t.cancel()
                except (Exception, asyncio.CancelledError):
                    pass
            if running:
                try:
                    await asyncio.gather(*running, return_exceptions=True)
                except asyncio.CancelledError as e:
                    cancel_exc = cancel_exc or e
                except Exception:
                    pass

            # Cancel broadcasts
            try:
                await self.ctx.close()
            except Exception:
                pass

            # Shutdown executor
            ex: AsyncThreadExecutor = self.executor
            self.executor = None
            if ex is not None:
                try:
                    ex.shutdown(cancel_pending=True)
                except Exception:
                    pass

        finally:
            # Release references
            self.pending_tasks.clear()
            self.running_tasks.clear()
            self.completed_tasks.clear()
            self.task_to_name.clear()
            self.name_to_task.clear()
            self.queued_dfs.clear()
            self._provider_write_queue.clear()
            self.frames.clear()
            self.schema_cache.clear()

            if cancel_exc is not None:
                raise cancel_exc

    def _skip_task(self, task_name: str, reason: str):
        log.loader(f"SKIPPING TASK {task_name}", color='#d6a83c')
        if task_name not in self.tasks: return
        if task_name in self.completed_tasks: return
        self.pending_tasks.discard(task_name)
        self.completed_tasks.add(task_name)
        self._cancel_requested.add(task_name)
        self.task_outcome[task_name] = "cancelled"
        self.task_status[task_name] = f"cancelled: {reason}"
        self._completed_order.append(task_name)

    def _cancel_running_task_by_name(self, task_name: str, reason: str = "cancel_requested", detach: bool = True):
        if not task_name:
            return

        # Sticky: once cancelled, this name must never merge during this run.
        self._cancel_requested.add(task_name)

        # Set the cooperative cancel event so the task function can check it.
        ev = self._cancel_events.get(task_name)
        if ev is not None:
            ev.set()

        at = self.name_to_task.get(task_name)
        if at is None:
            # Not currently tracked as running; mark as cancelled if still pending.
            self.pending_tasks.discard(task_name)
            self.completed_tasks.add(task_name)
            self.task_outcome[task_name] = "cancelled"
            self.task_status[task_name] = f"cancelled: {reason}"
            return

        log.loader(f"Cancel request {task_name}: at={id(at) if at else None} done={at.done() if at else None} cancelled={at.cancelled() if at else None}", color='#d6a83c')

        # Cancel the inner executor task (the real work on the executor thread).
        t = self.tasks.get(task_name)
        if t is not None and t._future is not None:
            ex = self.executor
            if ex is not None:
                try:
                    ex.cancel_inner(t._future)
                except Exception:
                    pass
            try:
                t._future.cancel()
            except Exception:
                pass

        # Request cancellation of the outer asyncio.Task.
        try:
            at.cancel()
        except Exception:
            pass

        if not detach:
            self.task_outcome[task_name] = "cancelling"
            self.task_status[task_name] = f"cancelling: {reason}"
            return

        # Detach immediately so scheduler stops waiting.
        self._detach_running_task(at, task_name, reason)


    def _auto_skip_untriggered_failed_tasks(self):
        # If a task has requiresFailedTask and all triggers completed with no failures => it will never fi_re.
        for name in list(self.pending_tasks):
            t = self.tasks[name]
            if not t.failed_task_requirements: continue
            triggers = [x for x in t.failed_task_requirements if x in self.tasks]
            if not triggers: continue
            if not set(triggers).issubset(self.completed_tasks): continue
            any_failed = any(self.task_outcome.get(x) == "failed" for x in triggers)
            if not any_failed:
                self._skip_task(name, "requiresFailedTask not satisfied")

    async def _maybe_skip_optional_blockers(self):
        if self.optional_policy!="background":
            return False

        running_names = self._running_task_names()
        ready = self.dep_graph.get_ready_tasks(self.completed_tasks, running_names, self.column_complete)

        progressed = False

        for name in list(self.pending_tasks):
            t = self.tasks[name]
            if t.isOptional:
                continue

            if name in ready:
                continue
            if name in running_names:
                continue

            blockers = self.dep_graph.blockers(t, self.completed_tasks, self.column_complete)
            if not blockers:
                continue

            optional_blockers = {b for b in blockers if (b in self.tasks and self.tasks[b].isOptional)}
            if optional_blockers and (optional_blockers==blockers):
                skippable_pending = []
                cancellable_running = []

                for ob in optional_blockers:
                    tob = self.tasks.get(ob)
                    # await log.loader(f"checking optinoal_blocker: {tob.task_name}")
                    if tob is None:
                        continue

                    if await self._task_has_unfilled_critical_columns(tob):
                        continue

                    if ob in self.pending_tasks:
                        # await log.loader(f"Adding {tob.task_name} to skippable")
                        skippable_pending.append(ob)

                    elif ob in running_names:
                        # await log.loader(f"Adding {tob.task_name} to cancellable")
                        cancellable_running.append(ob)

                if not skippable_pending and not cancellable_running:
                    continue

                for ob in skippable_pending:
                    self._skip_task(ob, f"optional blocker for {name}")
                    progressed = True

                for ob in cancellable_running:
                    await log.loader(f"Cancelling {ob}", color='#d6a83c')
                    self._cancel_running_task_by_name(ob, reason=f"optional blocker for {name}", detach=True)
                    progressed = True

        return progressed

    def _prune_optional_keep_set_for_readiness(self, keep: set, remaining_names: set, running_names: set):

        if not keep:
            return set(), set()

        tasks = self.tasks
        dep_graph = self.dep_graph
        task_outcome = self.task_outcome
        column_complete = self.column_complete
        col_providers = dep_graph.column_providers
        strict_reqs = dep_graph.strict_requirements

        keep_work = set(keep)
        removed = set()
        running_names = set(running_names)

        NON_SUCCESS = frozenset({"failed", "cancelled"})

        changed = True
        while changed:
            changed = False

            # Iterate over a snapshot since we may remove during loop.
            for name in list(keep_work):
                t = tasks.get(name)
                if t is None:
                    keep_work.remove(name)
                    removed.add(name)
                    changed = True
                    continue

                bad = False

                # ---- (1) Hard task deps already finalized non-success ----
                deps = strict_reqs.get(name, set())
                if deps:
                    for dep in deps:
                        if dep==name:
                            continue
                        dep_state = task_outcome.get(dep, None)
                        if dep_state in NON_SUCCESS:
                            bad = True
                            break
                        # dep not successful and also not something we will run (or currently running)
                        if dep_state!="success" and (dep not in keep_work) and (dep not in running_names):
                            bad = True
                            break

                if bad:
                    keep_work.remove(name)
                    removed.add(name)
                    changed = True
                    continue

                # ---- (2) Strict column requirements cannot be improved (and were never successfully provided) ----
                req_cols = t.strict_col_requirements or ()
                if req_cols:
                    frame = t.fromFrame
                    st = column_complete.get(frame, {})
                    providers_map = col_providers.get(frame, {})
                    ignored = ensure_set(t.ignored_tasks) | ensure_set(self.global_ignores)

                    for col in req_cols:
                        if st.get(col, False):
                            continue

                        provs = providers_map.get(col)
                        if not provs:
                            # No known providers => do not treat as unreachable.
                            continue

                        # Any remaining provider in keep_work means the column might still improve.
                        active_provider_found = False
                        any_provider_succeeded = False

                        for p in provs:
                            if p==name or p in ignored:
                                continue
                            if p in keep_work:
                                active_provider_found = True
                                break
                            if task_outcome.get(p)=="success":
                                any_provider_succeeded = True

                        if active_provider_found:
                            continue

                        # No active providers left.
                        # If at least one provider succeeded historically, allow (partial data may exist).
                        if any_provider_succeeded:
                            continue

                        # Providers exist but none active and none ever succeeded => requirement is dead.
                        bad = True
                        break

                if bad:
                    keep_work.remove(name)
                    removed.add(name)
                    changed = True
                    continue

        return keep_work, removed

    async def _should_early_finish_due_to_optional_only(self):
        if self.optional_policy!="background":
            return False

        # Any non-optional still pending => not in optional-only mode.
        if any((not self.tasks[n].isOptional) for n in self.pending_tasks):
            return False

        running_names = self._running_task_names()
        if any((not self.tasks[n].isOptional) for n in running_names):
            return False

        if not (self.pending_tasks or self.running_tasks):
            return False

        crit_by_frame = self._get_critical_cols_by_frame()

        # No critical columns configured => preserve original behavior
        if not crit_by_frame:
            return True

        # If any critical column is still incomplete, do NOT early-finish.
        missing = []
        for frame, cols in crit_by_frame.items():
            st = self.get_complete_columns(frame)
            for c in cols:
                if not st.get(c, False):
                    missing.append((frame, c))

        if not missing:
            # All critical columns complete => safe to cancel remaining optionals.
            return True

        # Critical columns missing:
        remaining = set(self.pending_tasks) | set(running_names)

        protected = set()
        for n in remaining:
            t = self.tasks.get(n)
            if t is None or (not t.isOptional):
                continue

            if await self._task_has_unfilled_critical_columns(t):
                protected.add(n)

        keep = set(protected)

        # Known providers for missing critical columns (best-effort, based on expected_col_provides graph)
        for frame, col in missing:
            provs = self.dep_graph.column_providers.get(frame, {}).get(col, set())
            if provs:
                keep |= (provs & remaining)

        # Closure over blockers for keep-set tasks
        stack = list(keep)
        while stack:
            name = stack.pop()
            t = self.tasks.get(name)
            if t is None:
                continue
            blockers = self.dep_graph.blockers(t, self.completed_tasks, self.column_complete)
            for b in blockers:
                if b in remaining and b not in keep:
                    keep.add(b)
                    stack.append(b)

        # If we can't determine anything useful to keep, do no pruning.
        if not keep:
            return False

        # Skip only pending, non-protected, non-keep optionals.
        for n in list(self.pending_tasks):
            if n in keep:
                continue
            t = self.tasks.get(n)
            if t is None or (not t.isOptional):
                continue
            if await self._task_has_unfilled_critical_columns(t):
                continue
            self._skip_task(n, "optional pruned (waiting for critical columns)")

        return False

    async def _ensure_frame_has_columns(self, frame_name: str, cols: list):
        cols = _clean_list(cols)
        if not cols:
            return
        async with self.locks[frame_name]:
            lf = self.frames.get(frame_name)
            if lf is None:
                schema_pairs = [(c, pl.Null) for c in cols]
                self.frames[frame_name] = pl.DataFrame(schema=schema_pairs).lazy()  # FIX
                self.invalidate_schema(frame_name)
                return

            schema = self.get_schema(frame_name)
            missing = [c for c in cols if c not in schema]
            if not missing:
                return

            self.frames[frame_name] = lf.with_columns([pl.lit(None).alias(c) for c in missing])
            self.invalidate_schema(frame_name)

    async def _prepare_task_inputs(self, task: DataTask):
        await self._ensure_frame_has_columns(task.fromFrame, task.strict_col_requirements)
        mk = _merge_key_list(task.merge_key)
        if mk: await self._ensure_frame_has_columns(task.toFrame, mk)

    def _empty_result_frame(self, task: DataTask):
        mk = _merge_key_list(task.merge_key)

        provides = task.expected_col_provides or {}
        if not isinstance(provides, dict):
            provides = {c: pl.Null for c in _clean_list(provides)}

        cols = _clean_list(mk + list(provides.keys()))

        left_schema = self.get_schema(task.toFrame)
        schema_pairs = []

        for c in cols:
            if c in provides:
                dt = provides.get(c, pl.Null)
                dt = string_to_polars_dtype(dt) if isinstance(dt, str) else (dt if dt is not None else pl.Null)
            else:
                # Join keys should match left frame when possible
                dt = left_schema.get(c, pl.Null)

            schema_pairs.append((c, dt))

        return pl.DataFrame(schema=schema_pairs)

    def _normalize_task_output(self, task: DataTask, df):
        if df is None:
            if task.expected_col_provides:
                return self._empty_result_frame(task)
            return None

        # Normalize column names to clean_column for consistent merges/requirements.
        sch = _safe_schema(df)
        if sch:
            rename_map = {c: clean_column(c) for c in sch.keys()}
            try:
                df = df.rename(rename_map) if rename_map else df
            except Exception:
                pass

        # Ensure merge keys exist on right.
        mk = _merge_key_list(task.merge_key)
        if mk:
            right_schema = _safe_schema(df)
            missing_mk = [c for c in mk if c not in right_schema]
            if missing_mk:
                log.error(f"Missing merge key for: {task.task_name}")
                log.error(f"Missing: {missing_mk}")
                log.error(f"Columns: {df.hyper.fields}")
                return self._empty_result_frame(task)

        # If task claims to provide columns, but they are missing, add them as *typed* nulls.
        if task.expected_col_provides:
            right_schema = _safe_schema(df)
            missing = [c for c in task.expected_col_provides if c not in right_schema]
            if missing:
                try:
                    exprs = []
                    for c in missing:
                        dt = task.expected_col_provides.get(c, pl.Null)
                        dt = string_to_polars_dtype(dt) if isinstance(dt, str) else (dt if dt is not None else pl.Null)

                        e = pl.lit(None)
                        if polars_dtype_to_string(dt)!="Null":
                            try:
                                e = e.cast(dt, strict=False)
                            except TypeError:
                                # older Polars: no strict kwarg
                                try:
                                    e = e.cast(dt)
                                except Exception:
                                    pass
                            except Exception:
                                pass

                        exprs.append(e.alias(c))

                    df = df.with_columns(exprs)
                except Exception:
                    # Fall back: ignore; merge will still proceed.
                    pass

        return df

    async def _queue_provider_cache_update(self, task: DataTask, df):

        if not self.use_providers_cache: return
        if df is None: return

        try:
            mk = _merge_key_list(task.merge_key)
            sch = _safe_schema(df)
            if not sch: return

            mk_set = set(mk) if mk else set()

            # Actual output schema (excluding merge keys), stored as dtype strings
            actual_map = {}
            for col, dt in sch.items():
                cc = clean_column(col) if isinstance(col, str) else col
                if not isinstance(cc, str) or (not cc) or (cc in mk_set):
                    continue
                actual_map[cc] = polars_dtype_to_string(dt)

            # Declared expected schema (dtype strings)
            expected = task.expected_col_provides or {}
            if not isinstance(expected, dict):
                expected = {c: pl.Null for c in _clean_list(expected)}

            expected_map = {
                clean_column(c): polars_dtype_to_string(dt)
                for c, dt in expected.items()
                if isinstance(c, str) and c.strip()
            }

            path = task._provider_path or _provider_cache_path(task.task_name)

            existing = await _read_provider_cache(path) or {}
            existing_sig = existing.get("signature") if isinstance(existing, dict) else None
            current_sig = task._provider_sig or _task_signature(task.func)

            sig_ok = isinstance(existing, dict) and _signature_matches(existing_sig, current_sig)

            existing_union_raw = existing.get("columns_union") if sig_ok and isinstance(existing, dict) else None
            union_map = _provider_cols_to_dtype_str_map(existing_union_raw) if existing_union_raw else {}

            # Ensure expected columns exist in union (don't clobber a non-null dtype with Null)
            for c, dts in expected_map.items():
                prev = union_map.get(c, None)
                if prev is None or prev=="Null":
                    union_map[c] = dts

            # Actual output should override Null/missing; if it differs from old, take latest non-null
            for c, dts in actual_map.items():
                if dts!="Null":
                    union_map[c] = dts
                else:
                    union_map.setdefault(c, "Null")

            record = {
                "version": PROVIDER_CACHE_VERSION,
                "columns_union": union_map,
                "columns_last": actual_map,
                "merge_key": mk,
                "signature": current_sig,
                "updated_at": time.time(),
            }

            # Update in-memory view immediately
            self._provider_write_queue[path] = record
            _PROVIDER_CACHE_MEM[path] = record

            # Fire-and-forget background write, coalesced by token.
            st = _provider_cache_state()
            tokens = st["write_token"]
            tokens[path] = tokens.get(path, 0) + 1
            await _spawn_provider_cache_write(path)

        except Exception as e:
            # Best-effort cache update; never fail the task on cache issues.
            try:
                log.error(f"Provider cache update failed for {task.task_name}: {e}")
            except Exception:
                pass

    async def _run_task_with_retries(self, task: DataTask):
        last_error = None

        cancel_event = self._get_or_create_cancel_event(task.task_name)

        accepts_cancel_event = getattr(task, "_accepts_cancel_event", None)
        accepts_is_cancelled = getattr(task, "_accepts_is_cancelled", None)

        pass_full_frame = getattr(task, "requestFullFrame", False)
        is_optional = getattr(task, "isOptional", False)
        trim_frame_check = (not pass_full_frame) and is_optional

        input_cols = set()
        if trim_frame_check:
            input_cols = ensure_set(task.merge_key) | ensure_set(task.critical_columns)

        if accepts_cancel_event is None:
            accepts_cancel_event = self._func_accepts_kw(task.func, "cancel_event")
            setattr(task, "_accepts_cancel_event", accepts_cancel_event)

        if accepts_is_cancelled is None:
            accepts_is_cancelled = self._func_accepts_kw(task.func, "is_cancelled")
            setattr(task, "_accepts_is_cancelled", accepts_is_cancelled)

        def is_cancelled():
            return cancel_event.is_set() or self._force_quit.is_set() or (task.task_name in self._cancel_requested)

        task.startTime = time.monotonic()
        for attempt in range(task.max_retries + 1):
            if self._force_quit.is_set() or (task.task_name in self._cancel_requested):
                cancel_event.set()
                task._last_outcome = "cancelled"
                task._last_error = "force_quit" if self._force_quit.is_set() else "cancel_requested"
                raise asyncio.CancelledError()

            fut = None
            try:
                task.numExecutions += 1
                await self._prepare_task_inputs(task)

                my_pt_lf = _as_lazy(self.frames.get(task.fromFrame))
                if trim_frame_check and input_cols:
                    my_pt_lf = my_pt_lf.select(input_cols)

                ctx_frames = {}
                if "*" in task.frameContext:
                    ctx_frames = {c:_as_lazy(d) for c,d in self.frames.items()}
                else:
                    for cotx in (task.frameContext or []):
                        ctx_frames[cotx] = _as_lazy(self.frames.get(cotx))

                    if (task.fromFrame != task.toFrame) and (task.toFrame not in ctx_frames):
                        ctx_frames[task.toFrame] = _as_lazy(self.frames.get(task.toFrame))

                kwargs = dict(task.kwargs or {})
                kwargs.update(
                    {
                        "my_pt": my_pt_lf,
                        "dates": self.dates,
                        "frames": ctx_frames,
                        "s3": self.s3,
                    }
                )

                if accepts_cancel_event:
                    kwargs["cancel_event"] = cancel_event
                if accepts_is_cancelled:
                    kwargs["is_cancelled"] = is_cancelled

                def _stop_timer(t, future):
                    t.stopTime = time.monotonic()
                stop_timer = functools.partial(_stop_timer, task)

                async def _run():
                    task.debug_input = my_pt_lf.clone()
                    kwargs.setdefault('__loop', asyncio.get_running_loop())
                    fut_local = self.executor.submit(task.run, **kwargs)
                    fut_local.add_done_callback(stop_timer)
                    task._future = fut_local
                    return fut_local, await asyncio.wrap_future(fut_local)

                async with self.sem:
                    log.loader(f"Running {task.task_name}")
                    try:
                        if task.timeout is not None and task.timeout > 0:
                            fut, result = await asyncio.wait_for(_run(), timeout=task.timeout)
                        else:
                            fut, result = await _run()
                    except asyncio.TimeoutError:
                        # Cancel the inner executor task so it doesn't linger.
                        cancel_event.set()
                        ex = self.executor
                        if ex is not None and task._future is not None:
                            try:
                                ex.cancel_inner(task._future)
                            except Exception:
                                pass
                            try:
                                task._future.cancel()
                            except Exception:
                                pass
                        raise

                if is_cancelled():
                    cancel_event.set()
                    task._last_outcome = "cancelled"
                    task._last_error = "cancel_requested (late)"
                    self._completed_order.append(task.task_name)
                    raise asyncio.CancelledError()

                self._completed_order.append(task.task_name)
                # log.loader(f"Task completed: {task.task_name} ({task.duration:.2f}s)")
                if task.broadcast_name is not None:
                    await self.broadcast_log(f"Complete: {task.broadcast_name} ({task.duration:.2f}s)")

                result = self._normalize_task_output(task, result)
                task._last_outcome = "success"
                task._last_error = None

                if (result is not None) and (task.cache_providers):
                    try:
                        await self._queue_provider_cache_update(task, result)
                    except Exception:
                        pass

                return result

            except asyncio.CancelledError:
                cancel_event.set()
                # Cancel the inner executor task (the real work).
                ex = self.executor
                if ex is not None and fut is not None:
                    try:
                        ex.cancel_inner(fut)
                    except Exception:
                        pass
                try:
                    if fut is not None:
                        fut.cancel()
                except Exception:
                    pass
                task._last_outcome = "cancelled"
                task._last_error = last_error
                self._completed_order.append(task.task_name)
                raise

            except asyncio.TimeoutError:
                last_error = f"Task {task.task_name} timed out after {task.duration}s"
                log.error(last_error)
                await self.broadcast_log(last_error)

            except (ConnectionRefusedError, ConnectionError) as e:
                last_error = f"Task {task.task_name} connection error after {task.duration}s: {e}"
                log.error(last_error)
                await self.broadcast_log(last_error)

            except RuntimeError as e:
                last_error = f"Task {task.task_name} encountered runtime error {task.duration}s: {e}"
                log.error(last_error)
                await self.broadcast_log(last_error)
                task._last_outcome = "failed"
                task._last_error = last_error
                self._completed_order.append(task.task_name)
                raise

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                log.error(f"Task {task.task_name} failed after {task.duration}s: {e}")
                if task.broadcast_name is not None:
                    await self.broadcast_log(f"Task {task.broadcast_name} failed: {e}")

            if attempt < task.max_retries:
                delay = task.retry_delay or (self.retry_delay*attempt)
                log.warning(f"Retrying task {task.task_name} in {delay}s...")
                if (task.broadcast_name is not None):
                    await self.broadcast_log(f"Retrying task {task.broadcast_name} in {delay}s...")
                await asyncio.sleep(delay)

        task._last_outcome = "failed"
        task._last_error = last_error
        self._completed_order.append(task.task_name)

        if task.expected_col_provides and task.empty_on_fail:
            log.error(f"Task {task.task_name} failed; degrading to empty typed frame. Last error: {last_error}")
            empty = self._empty_result_frame(task)
            if (empty is not None) and (task.cache_providers):
                try:
                    await self._queue_provider_cache_update(task, empty)
                except Exception:
                    pass
            return empty

        log.error(f"Task {task.task_name} failed; no provides => treated as no-op. Last error: {last_error}")
        return None

    async def _merge_frames(
            self,
            to_frame: str,
            right_df,
            *,
            on=None,
            left_on=None,
            right_on=None,
            merge_policy: str = "coalesce_left",
            dedupe_right: bool = True,
            based_on_col=None,
            based_on_comparator=None,
            based_on_col_to_merge_cols=None,
            based_on_tie_break: str = "left",
            suppress_new_columns: bool = False,
    ):
        """Merge *right_df* into the frame identified by *to_frame*.

        Parameters
        ----------
        based_on_col : str or _re.Pattern, optional
            When *merge_policy* is ``"based_on"``, identifies the comparator
            column(s).

            * **plain str** – a single literal column name (original
              single-column behaviour).
            * **str containing regex meta-characters / compiled _re.Pattern** –
              matched via ``_re.fullmatch`` against every shared column name to
              discover *all* based-on columns (multi-group mode).

            A plain string is promoted to a regex pattern when it contains any
            of ``^ $ * + ? { } [ ] | ( ) \\``.  Note that ``.`` is **not**
            treated as a meta-character to avoid misinterpreting dotted column
            names; use ``_re.compile()`` for patterns requiring ``.``.

        based_on_comparator : ``"greater"`` | ``"less"`` | callable, optional
            Comparison applied to every based-on column.  A callable receives
            ``(left_value, right_value)`` and returns ``True`` when the
            *right* side wins.

        based_on_col_to_merge_cols : callable, optional
            ``(based_on_col_name: str) -> list[str]``

            Given the name of a based-on column, return the value columns
            governed by that comparison.  Called once per discovered based-on
            column at plan-build time (not per row).

            If multiple based-on groups claim the same value column, the first
            group (in schema iteration order) wins and subsequent claims are
            skipped with a debug log.

            When ``None`` and the pattern matched exactly **one** column, all
            remaining shared columns are governed by that column (backwards-
            compatible).  When ``None`` and multiple columns matched, the
            merge is skipped with an error.

        based_on_tie_break : ``"left"`` | ``"right"``, default ``"left"``
            Which side wins when the based-on column values are equal.
            ``"left"`` favours the existing *to_frame* values (default);
            ``"right"`` favours the incoming *right_df* values.  Only applies
            to the ``"greater"`` and ``"less"`` comparators — for a callable
            comparator, tie-breaking is the caller's responsibility.

        suppress_new_columns : bool, default False
            When ``True``, columns present in *right_df* but absent from the
            existing *to_frame* are silently dropped before the join.  The
            resulting frame's schema will never widen — only existing columns
            can be updated.  Composes with any merge policy.

        """
        if right_df is None: return

        frames = self.frames
        locks = self.locks
        invalidate_schema = self.invalidate_schema
        fast_level = int(MERGE_FAST_PATH_LEVEL)

        pl_col = pl.col
        pl_coalesce = pl.coalesce

        if merge_policy not in _KNOWN_MERGE_POLICIES:
            log.error(f"Merge skipped for frame={to_frame} (unknown merge_policy={merge_policy!r})")
            return

        # Dtype constants (handle older/newer Polars naming)
        NULL_DT = getattr(pl, "Null", None)
        STR_DT = getattr(pl, "Utf8", None) or getattr(pl, "String", None)
        FLOAT64_DT = getattr(pl, "Float64", None)
        INT64_DT = getattr(pl, "Int64", None)

        pl_datatypes = getattr(pl, "hyper", None)
        get_supertype = getattr(pl_datatypes, "get_supertype", None) if pl_datatypes is not None else None

        def _is_null(dt):
            if dt is None:
                return True
            if NULL_DT is not None and dt==NULL_DT:
                return True
            try:
                return str(dt)=="Null"
            except Exception:
                return False

        def _dtype_str(dt):
            try:
                return str(dt)
            except Exception:
                return ""

        def _is_float(dt):
            s = _dtype_str(dt)
            return s.startswith("Float")

        def _is_stringlike(dt):
            s = _dtype_str(dt)
            return s in ("Utf8", "String", "Categorical", "Enum")

        def _numeric_info(dt):
            s = _dtype_str(dt)
            if s.startswith("Float"):
                try:
                    return "float", True, int(s[5:])
                except Exception:
                    return "float", True, 64
            if s.startswith("UInt"):
                try:
                    return "int", False, int(s[4:])
                except Exception:
                    return "int", False, 64
            if s.startswith("Int"):
                try:
                    return "int", True, int(s[3:])
                except Exception:
                    return "int", True, 64
            return None

        def _numeric_supertype(ldt, rdt):
            if get_supertype is not None:
                try:
                    st = get_supertype(ldt, rdt)
                    if st is not None:
                        return st
                except Exception:
                    pass

            li = _numeric_info(ldt)
            ri = _numeric_info(rdt)
            if li is None or ri is None:
                return None

            if li[0]=="float" or ri[0]=="float":
                return FLOAT64_DT or ldt

            _, lsigned, lbits = li
            _, rsigned, rbits = ri

            if lsigned and rsigned:
                return INT64_DT or ldt
            if (not lsigned) and (not rsigned):
                u64 = getattr(pl, "UInt64", None)
                return u64 or INT64_DT or ldt

            return FLOAT64_DT or INT64_DT or ldt

        target_cache = {}

        def _choose_merge_dtype(ldt, rdt, prefer_right: bool):
            if ldt==rdt:
                return ldt
            if _is_null(ldt):
                return rdt
            if _is_null(rdt):
                return ldt

            ck = (_dtype_str(ldt), _dtype_str(rdt), 1 if prefer_right else 0)
            got = target_cache.get(ck, None)
            if got is not None:
                return got

            st_num = _numeric_supertype(ldt, rdt)
            if st_num is not None:
                target_cache[ck] = st_num
                return st_num

            if _is_stringlike(ldt) or _is_stringlike(rdt):
                out = STR_DT or (rdt if prefer_right else ldt)
                target_cache[ck] = out
                return out

            if get_supertype is not None:
                try:
                    st = get_supertype(ldt, rdt)
                    if st is not None:
                        target_cache[ck] = st
                        return st
                except Exception:
                    pass

            out = rdt if prefer_right else ldt
            target_cache[ck] = out
            return out

        def _cast_relaxed(expr, dtype):
            if dtype is None:
                return expr
            try:
                return expr.cast(dtype, strict=False)
            except TypeError:
                try:
                    return expr.cast(dtype)
                except Exception:
                    return expr
            except Exception:
                return expr

        def _maybe_fill_nan(expr, dtype):
            if dtype is None:
                return expr
            if _is_float(dtype):
                try:
                    return expr.fill_nan(None)
                except Exception:
                    return expr
            return expr

        def _build_right_wins(bo_col_name, s_left, s_right, comparator, tie_break):
            left_bo = pl_col(bo_col_name)
            right_bo = pl_col(f"{bo_col_name}_right")

            ldt_bo = s_left.get(bo_col_name)
            rdt_bo = s_right.get(bo_col_name)
            if _is_float(ldt_bo):
                left_bo = left_bo.fill_nan(None)
            if _is_float(rdt_bo):
                right_bo = right_bo.fill_nan(None)

            bo_target = _choose_merge_dtype(ldt_bo, rdt_bo, prefer_right=False)
            if bo_target is not None and ldt_bo!=bo_target:
                left_bo = _cast_relaxed(left_bo, bo_target)
            if bo_target is not None and rdt_bo!=bo_target:
                right_bo = _cast_relaxed(right_bo, bo_target)

            # Null on right => right loses.  Null on left => right wins.
            # Tie behaviour controlled by tie_break:
            #   "left"  => strict comparison (right must be strictly better)
            #   "right" => non-strict comparison (equal values favour right)
            right_wins_tie = (tie_break=="right")

            if comparator=="greater":
                if right_wins_tie:
                    return right_bo.is_not_null() & (left_bo.is_null() | (right_bo >= left_bo))
                return right_bo.is_not_null() & (left_bo.is_null() | (right_bo > left_bo))
            if comparator=="less":
                if right_wins_tie:
                    return right_bo.is_not_null() & (left_bo.is_null() | (right_bo <= left_bo))
                return right_bo.is_not_null() & (left_bo.is_null() | (right_bo < left_bo))
            if callable(comparator):
                _l_alias = f"_bo_l_{bo_col_name}"
                _r_alias = f"_bo_r_{bo_col_name}"

                def _make_udf_mask(_left_bo, _right_bo, _cmp=comparator,
                                   _la=_l_alias, _ra=_r_alias):
                    return (
                            _right_bo.is_not_null()
                            & (
                                    _left_bo.is_null()
                                    | pl.struct([_left_bo.alias(_la), _right_bo.alias(_ra)])
                                    .map_elements(
                                lambda s, __cmp=_cmp, __la=_la, __ra=_ra:
                                bool(__cmp(s[__la], s[__ra])),
                                return_dtype=pl.Boolean,
                            )
                            )
                    )

                return _make_udf_mask(left_bo, right_bo)
            return None  # invalid comparator

        def _build_value_expr(col_name, right_wins_expr, s_left, s_right):
            """Build the when/then/otherwise expression for a single value
            column governed by *right_wins_expr*.  Returns ``None`` when the
            column can be skipped (both Null dtype, or right all-Null)."""
            rc = f"{col_name}_right"
            ldt = s_left.get(col_name)
            rdt = s_right.get(col_name)

            if _is_null(ldt) and _is_null(rdt):
                return None
            if _is_null(ldt):
                # Left is all-null -> right value is the only real data.
                return pl_col(rc).alias(col_name)
            if _is_null(rdt):
                return None

            target = _choose_merge_dtype(ldt, rdt, prefer_right=False)

            lc = _maybe_fill_nan(pl_col(col_name), ldt)
            rcx = _maybe_fill_nan(pl_col(rc), rdt)

            if target is not None:
                if ldt!=target:
                    lc = _cast_relaxed(lc, target)
                if rdt!=target:
                    rcx = _cast_relaxed(rcx, target)

            # Winner is greedy side of coalesce; loser fills nulls.
            return (
                pl.when(right_wins_expr)
                .then(pl_coalesce([rcx, lc]))
                .otherwise(pl_coalesce([lc, rcx]))
                .alias(col_name)
            )

        # ---- Validate based_on parameters (before lock) ----
        if merge_policy=="based_on":
            if based_on_col is None:
                log.error(f"Merge skipped for frame={to_frame} (based_on policy requires based_on_col)")
                return
            if based_on_comparator is None:
                log.error(f"Merge skipped for frame={to_frame} (based_on policy requires based_on_comparator)")
                return
            if based_on_comparator not in ("greater", "less") and not callable(based_on_comparator):
                log.error(
                    f"Merge skipped for frame={to_frame} "
                    f"(based_on_comparator must be 'greater', 'less', or callable, "
                    f"got {based_on_comparator!r})"
                )
                return
            if based_on_tie_break not in ("left", "right"):
                log.error(
                    f"Merge skipped for frame={to_frame} "
                    f"(based_on_tie_break must be 'left' or 'right', "
                    f"got {based_on_tie_break!r})"
                )
                return

        async with locks[to_frame]:
            left_lf = frames.get(to_frame)
            if left_lf is None:
                right_lf_init = ensure_lazy(right_df)
                try:
                    result = await right_lf_init.hyper.compress_plan_async()
                except Exception:
                    log.exception(f"Merge failed for frame={to_frame} during initial compress")
                    raise
                invalidate_schema(to_frame)
                frames[to_frame] = result
                return result
            left_lf = ensure_lazy(left_lf)

            # Normalize key params
            if on is not None:
                left_on = right_on = on

            left_keys = ensure_list(left_on, allow_none=False)
            right_keys = ensure_list(right_on, allow_none=False)

            left_keys = [clean_column(k) for k in left_keys if isinstance(k, str) and k]
            right_keys = [clean_column(k) for k in right_keys if isinstance(k, str) and k]

            if (not left_keys) or (not right_keys):
                log.error(f"Merge skipped for frame={to_frame} (missing join keys)")
                return
            if len(left_keys)!=len(right_keys):
                log.error(f"Merge skipped for frame={to_frame} (join key length mismatch)")
                return

            right_lf = ensure_lazy(right_df)
            schema_left = left_lf.hyper.schema()
            schema_right = right_lf.hyper.schema()

            missing_left = [k for k in left_keys if k not in schema_left]
            missing_right = [k for k in right_keys if k not in schema_right]

            if missing_left or missing_right:
                log.error(
                    f"Merge skipped for frame={to_frame} (missing join keys). "
                    f"left_missing={missing_left[:6]} right_missing={missing_right[:6]}"
                )
                return

            bo_groups = None  # dict | None

            if merge_policy=="based_on":
                is_pattern = _is_regex_pattern(based_on_col)

                if is_pattern:

                    pat = based_on_col if isinstance(based_on_col, _re.Pattern) else _re.compile(based_on_col)
                    l_match = [c for c in schema_left if pat.fullmatch(c)]
                    r_match = [c for c in schema_right if pat.fullmatch(c)]
                    if (r_match and not l_match):
                        if not suppress_new_columns:
                            return -1
                        log.error(
                            f"Merge skipped for frame={to_frame} "
                            f"(based_on_col pattern {based_on_col!r} matched no left columns and suppress_new is ON)"
                        )
                        return

                    bo_col_names = [c for c in schema_left if c in schema_right and pat.fullmatch(c)]
                    if not bo_col_names:
                        log.error(
                            f"Merge skipped for frame={to_frame} "
                            f"(based_on_col pattern {based_on_col!r} matched no shared columns)"
                        )
                        return

                    if based_on_col_to_merge_cols is None:
                        if len(bo_col_names)==1:
                            bo_groups = {bo_col_names[0]: None}  # deferred
                        else:
                            log.error(
                                f"Merge skipped for frame={to_frame} "
                                f"(based_on_col pattern matched {len(bo_col_names)} columns "
                                f"but based_on_col_to_merge_cols was not provided)"
                            )
                            return
                    else:
                        bo_groups = {}
                        for bc in bo_col_names:
                            try:
                                vcs = based_on_col_to_merge_cols(bc)
                            except Exception:
                                log.exception(
                                    f"Merge skipped for frame={to_frame} "
                                    f"(based_on_col_to_merge_cols raised for {bc!r})"
                                )
                                return
                            if vcs is None:
                                vcs = []
                            elif isinstance(vcs, str):
                                vcs = [vcs]
                            else:
                                vcs = list(vcs)
                            bo_groups[bc] = vcs
                else:
                    if based_on_col not in schema_left:
                        if suppress_new_columns:
                            log.error(
                                f"Merge skipped for frame={to_frame} "
                                f"(based_on_col={based_on_col!r} not found in left frame)"
                            )
                            return
                        else:
                            return -1

                    if based_on_col not in schema_right:
                        log.error(
                            f"Merge skipped for frame={to_frame} "
                            f"(based_on_col={based_on_col!r} not found in right frame)"
                        )
                        return

                    if based_on_col_to_merge_cols is not None:
                        try:
                            vcs = based_on_col_to_merge_cols(based_on_col)
                        except Exception:
                            log.exception(
                                f"Merge skipped for frame={to_frame} "
                                f"(based_on_col_to_merge_cols raised for {based_on_col!r})"
                            )
                            return
                        if vcs is None:
                            vcs = []
                        elif isinstance(vcs, str):
                            vcs = [vcs]
                        else:
                            vcs = list(vcs)
                        bo_groups = {based_on_col: vcs}
                    else:
                        bo_groups = {based_on_col: None}  # deferred

            # ---- Determine shared columns ----
            key_excl = set(left_keys)
            key_excl.update(right_keys)

            if suppress_new_columns:
                left_col_set = set(schema_left)
                keep_right = [c for c in schema_right
                              if c in left_col_set or c in key_excl]
                if len(keep_right) < len(schema_right):
                    right_lf = right_lf.select(keep_right)
                    schema_right = right_lf.hyper.schema()

            if len(schema_left) <= len(schema_right):
                shared = [c for c in schema_left.keys() if (c in schema_right) and (c not in key_excl)]
            else:
                shared = [c for c in schema_right.keys() if (c in schema_left) and (c not in key_excl)]

            if bo_groups is not None:
                deferred_keys = [bc for bc, vcs in bo_groups.items() if vcs is None]
                if deferred_keys:
                    for bc in deferred_keys:
                        bo_groups[bc] = [c for c in shared if c!=bc]

            need_key_cast = False
            if fast_level >= 1:
                for lk, rk in zip(left_keys, right_keys):
                    if schema_left.get(lk)!=schema_right.get(rk):
                        need_key_cast = True
                        break
            else:
                need_key_cast = True

            if need_key_cast:
                left_exprs = []
                right_exprs = []
                for lk, rk in zip(left_keys, right_keys):
                    ldt = schema_left.get(lk)
                    rdt = schema_right.get(rk)
                    if ldt==rdt:
                        continue
                    target = _choose_merge_dtype(ldt, rdt, prefer_right=False)
                    if target is None:
                        continue
                    if ldt!=target:
                        left_exprs.append(_cast_relaxed(pl_col(lk), target).alias(lk))
                    if rdt!=target:
                        right_exprs.append(_cast_relaxed(pl_col(rk), target).alias(rk))

                if left_exprs:
                    left_lf = left_lf.with_columns(left_exprs)
                if right_exprs:
                    right_lf = right_lf.with_columns(right_exprs)

            # ---- Dedupe right ----
            if dedupe_right:
                if (merge_policy=="based_on" and bo_groups is not None
                        and len(bo_groups)==1
                        and based_on_comparator in ("greater", "less")):
                    sort_desc = (based_on_comparator=="greater")
                    bo_sort_col = next(iter(bo_groups))
                    if bo_sort_col in schema_right:
                        right_lf = (
                            right_lf
                            .sort(bo_sort_col, descending=sort_desc)
                            .unique(subset=right_keys, keep="first")
                        )
                    else:
                        right_lf = right_lf.unique(subset=right_keys)
                else:
                    right_lf = right_lf.unique(subset=right_keys)

            # ---- Fast path: no shared columns ----
            if merge_policy!="based_on" and fast_level >= 1 and not shared:
                joined = left_lf.join(
                    right_lf, left_on=left_keys, right_on=right_keys,
                    how="left", suffix="_right",
                )
                try:
                    result = await joined.hyper.compress_plan_async()
                except Exception:
                    log.exception(f"Merge failed for frame={to_frame} during plan compression")
                    raise
                invalidate_schema(to_frame)
                frames[to_frame] = result
                return result

            # ---- Join ----
            joined = left_lf.join(
                right_lf, left_on=left_keys, right_on=right_keys,
                how="left", suffix="_right",
            )

            if not shared and merge_policy!="based_on":
                try:
                    result = await joined.hyper.compress_plan_async()
                except Exception:
                    log.exception(f"Merge failed for frame={to_frame} during plan compression")
                    raise
                invalidate_schema(to_frame)
                frames[to_frame] = result
                return result

            drop_cols = [f"{c}_right" for c in shared]

            if merge_policy=="based_on":
                exprs = []

                mask_cache = {}
                for bo_col_name in bo_groups:
                    rw = _build_right_wins(bo_col_name, schema_left, schema_right, based_on_comparator, based_on_tie_break)
                    if rw is None:
                        log.error(
                            f"Merge skipped for frame={to_frame} "
                            f"(failed to build comparator for {bo_col_name!r})"
                        )
                        return
                    mask_cache[bo_col_name] = rw

                governed = {}

                for bo_col_name, value_cols in bo_groups.items():
                    right_wins = mask_cache[bo_col_name]

                    value_set = set(value_cols)
                    if bo_col_name in value_set:
                        cols_for_group = value_cols
                    else:
                        cols_for_group = [*value_cols, bo_col_name]

                    for c in cols_for_group:
                        if c in key_excl: continue
                        if c not in schema_left or c not in schema_right:
                            log.debug(
                                f"based_on: column {c!r} from transformer for "
                                f"{bo_col_name!r} not found in both frames, skipping"
                            )
                            continue
                        if c in governed:
                            log.debug(
                                f"based_on: column {c!r} already governed by "
                                f"{governed[c]!r}, skipping claim from {bo_col_name!r}"
                            )
                            continue
                        governed[c] = bo_col_name

                        expr = _build_value_expr(c, right_wins, schema_left, schema_right)
                        if expr is not None:
                            exprs.append(expr)

                for c in shared:
                    if c in governed:
                        continue

                    rc = f"{c}_right"
                    ldt = schema_left.get(c)
                    rdt = schema_right.get(c)

                    if _is_null(ldt) and _is_null(rdt):
                        continue
                    if _is_null(ldt):
                        exprs.append(pl_col(rc).alias(c))
                        continue
                    if _is_null(rdt):
                        continue

                    target = _choose_merge_dtype(ldt, rdt, prefer_right=False)
                    lc = _maybe_fill_nan(pl_col(c), ldt)
                    rcx = _maybe_fill_nan(pl_col(rc), rdt)
                    if target is not None:
                        if ldt!=target:
                            lc = _cast_relaxed(lc, target)
                        if rdt!=target:
                            rcx = _cast_relaxed(rcx, target)
                    exprs.append(pl_coalesce([lc, rcx]).alias(c))

                if exprs:
                    joined = joined.with_columns(exprs)
                joined = joined.drop(drop_cols, strict=False)

                try:
                    result = await joined.hyper.compress_plan_async()
                except Exception:
                    log.exception(f"Merge failed for frame={to_frame} during plan compression")
                    raise
                invalidate_schema(to_frame)
                frames[to_frame] = result
                return result

            if merge_policy=="overwrite":
                exprs = [pl_col(f"{c}_right").alias(c) for c in shared]
                joined = joined.with_columns(exprs).drop(drop_cols)
                try:
                    result = await joined.hyper.compress_plan_async()
                except Exception:
                    log.exception(f"Merge failed for frame={to_frame} during plan compression")
                    raise
                invalidate_schema(to_frame)
                frames[to_frame] = result
                return result

            prefer_right = (merge_policy=="coalesce_right") or (merge_policy=="overwrite_non_null")

            exprs = []

            if fast_level >= 2:
                for c in shared:
                    rc = f"{c}_right"
                    ldt = schema_left.get(c)
                    rdt = schema_right.get(c)

                    if _is_null(ldt):
                        exprs.append(pl_col(rc).alias(c))
                        continue
                    if _is_null(rdt):
                        continue

                    if ldt==rdt:
                        lc = _maybe_fill_nan(pl_col(c), ldt)
                        rcx = _maybe_fill_nan(pl_col(rc), rdt)
                        if prefer_right:
                            exprs.append(pl_coalesce([rcx, lc]).alias(c))
                        else:
                            exprs.append(pl_coalesce([lc, rcx]).alias(c))
                        continue

                    target = _choose_merge_dtype(ldt, rdt, prefer_right=prefer_right)
                    lc = _maybe_fill_nan(pl_col(c), ldt)
                    rcx = _maybe_fill_nan(pl_col(rc), rdt)

                    if target is not None:
                        if ldt!=target:
                            lc = _cast_relaxed(lc, target)
                        if rdt!=target:
                            rcx = _cast_relaxed(rcx, target)

                    if prefer_right:
                        exprs.append(pl_coalesce([rcx, lc]).alias(c))
                    else:
                        exprs.append(pl_coalesce([lc, rcx]).alias(c))

            else:
                for c in shared:
                    rc = f"{c}_right"
                    ldt = schema_left.get(c)
                    rdt = schema_right.get(c)

                    if _is_null(ldt):
                        exprs.append(pl_col(rc).alias(c))
                        continue
                    if _is_null(rdt):
                        continue

                    target = _choose_merge_dtype(ldt, rdt, prefer_right=prefer_right)

                    lc = _maybe_fill_nan(pl_col(c), ldt)
                    rcx = _maybe_fill_nan(pl_col(rc), rdt)

                    if target is not None:
                        lc = _cast_relaxed(lc, target) if ldt!=target else lc
                        rcx = _cast_relaxed(rcx, target) if rdt!=target else rcx

                    if prefer_right:
                        exprs.append(pl_coalesce([rcx, lc]).alias(c))
                    else:
                        exprs.append(pl_coalesce([lc, rcx]).alias(c))

            if exprs:
                joined = joined.with_columns(exprs)
            joined = joined.drop(drop_cols, strict=False)

            try:
                result = await joined.hyper.compress_plan_async()
            except Exception:
                log.exception(f"Merge failed for frame={to_frame} during plan compression")
                raise
            invalidate_schema(to_frame)
            frames[to_frame] = result
            return result

    async def _merge_task_result(self, task: DataTask):
        if task.results is None: return
        mk = _normalize_merge_key(task.merge_key)
        if (mk == "") and (self.frames.get(task.toFrame) is None):
            collected = ensure_lazy(await task.results.hyper.compress_plan_async())
            async with self.locks[task.toFrame]:
                if self.frames.get(task.toFrame) is None:
                    self.frames[task.toFrame] = collected
                    self.frames_open[task.toFrame] = True
                    self.invalidate_schema(task.toFrame)
                    return

        async with self.locks[task.toFrame]:
            if (task.toFrame in self.frames_open) and (not self.frames_open.get(task.toFrame)):
                await log.error(f"Task {task.task_name} is writing to a supposedly closed frame! Inefficient.")
            if task.closeFrame and (task.toFrame != 'main'):
                self.frames_open[task.toFrame] = False
                await log.loader(f"Frame {task.toFrame} closed by {task.task_name}")

        if self.debug:
            # {task.results.hyper.height()}x{len(task.results.hyper.fields)} from
            await log.loader(f'Merging {task.task_name} -> {task.toFrame}')

        res = await self._merge_frames(
            to_frame=task.toFrame,
            right_df=task.results,
            left_on=mk,
            right_on=mk,
            merge_policy=task.mergePolicy,
            based_on_col=task.based_on_col,
            based_on_comparator=task.based_on_comparator
        )
        if (res is not None) and isinstance(res, int):
            await self._merge_frames(
                to_frame=task.toFrame,
                right_df=task.results,
                left_on=mk,
                right_on=mk,
                merge_policy="coalesce_left",
            )

    async def _update_column_status(self, task: DataTask):
        provides = task.expected_col_provides or {}
        if not provides:
            return

        lf = self.frames.get(task.toFrame)
        if lf is None:
            return

        if self.column_scan_mode=="off":
            return

        running_names = self._running_task_names()
        pending_or_running = set(self.pending_tasks) | set(running_names)

        cols = set()

        # Existing consumer-driven scanning
        for c in provides:
            consumers = self.dep_graph.column_consumers[task.toFrame].get(c, set())
            if not consumers:
                continue
            if self.column_scan_mode=="adaptive":
                if consumers & pending_or_running:
                    cols.add(c)
            else:
                cols.add(c)

        if self.optional_policy=="background":
            crit_cols = self._get_critical_cols_by_frame().get(task.toFrame, set())
            if crit_cols:

                st = self.get_complete_columns(task.toFrame)
                actual_provides = getattr(task, "actual_col_provides", None) or {}
                for c in crit_cols:
                    # Only scan if this task plausibly touched it (declared or actually present)
                    if (c in provides) or (c in actual_provides):
                        if task.mergePolicy=="overwrite" or (not st.get(c, False)):
                            cols.add(c)

        if not cols:
            return

        available = set(_safe_schema(lf).keys())
        exprs = [pl.col(c).is_null().sum().alias(c) for c in cols if c in available]
        if not exprs:
            async with self.locks[task.toFrame]:
                for c in cols:
                    self.column_complete[task.toFrame][c] = False
            return

        try:
            res = (await lf.select(exprs).collect_async()).to_dict(as_series=False)
            async with self.locks[task.toFrame]:
                for c in cols:
                    if c in available:
                        self.column_complete[task.toFrame][c] = (res.get(c, [1])[0]==0)
                    else:
                        self.column_complete[task.toFrame][c] = False
        except Exception:
            async with self.locks[task.toFrame]:
                for c in cols:
                    self.column_complete[task.toFrame][c] = False

    async def _handle_task_completion(self, task: DataTask, df):

        if task.task_name in self._cancel_requested: return
        task.results = ensure_lazy(df)

        declared = task.expected_col_provides or {}
        if not isinstance(declared, dict):
            declared = {c: pl.Null for c in _clean_list(declared)}

        actual_schema = _safe_schema(task.results) if task.results is not None else {}
        provided = dict(declared)
        provided.update(actual_schema)
        task.actual_col_provides = provided

        downstream = self.dep_graph.get_downstream(task.task_name)
        pending_downstream = downstream & set(self.pending_tasks)
        running_names = self._running_task_names()

        needs_immediate_merge = False
        if task.actual_col_provides:
            consumers = set()
            for c in task.actual_col_provides:
                consumers |= self.dep_graph.column_consumers[task.toFrame].get(c, set())
            if consumers & (set(self.pending_tasks) | set(running_names)):
                needs_immediate_merge = True

        if (not needs_immediate_merge) and (self.optional_policy=="background"):
            crit_cols = self._get_critical_cols_by_frame().get(task.toFrame, set())
            if crit_cols and task.actual_col_provides:
                st = self.get_complete_columns(task.toFrame)
                for c in crit_cols:
                    if c in task.actual_col_provides:
                        # If overwrite is possible, the column can become "worse" (introduce nulls) => force merge.
                        if task.mergePolicy=="overwrite" or (not st.get(c, False)):
                            needs_immediate_merge = True
                            break

        if pending_downstream or needs_immediate_merge:
            if self.debug:
                # {task.results.hyper.height()}x{len(task.results.hyper.fields)} from
                log.loader(f"Merging {task.task_name} immediately.")
            if self.frames.get(task.toFrame) is None:
                collected = ensure_lazy(await df.hyper.compress_plan_async())
                _check = False
                async with self.locks[task.toFrame]:
                    if self.frames.get(task.toFrame) is None:
                        self.frames[task.toFrame] = collected.lazy()
                        self.invalidate_schema(task.toFrame)
                        _check = True

                if not _check:
                    await self._merge_task_result(task)
            else:
                await self._merge_task_result(task)

            if not self.debug:
                task.results = None

            if self.materialize_every_n_merges > 0:
                self._merge_counts[task.toFrame] += 1
                if self._merge_counts[task.toFrame] >= self.materialize_every_n_merges:
                    self._merge_counts[task.toFrame] = 0
                    async with self.locks[task.toFrame]:
                        try:
                            collected = await self.frames[task.toFrame].collect_async()
                            self.frames[task.toFrame] = collected.lazy()
                            self.invalidate_schema(task.toFrame)
                            if task.toFrame=="main":
                                self.main_df = collected
                        except Exception as e:
                            log.critical(f"Error materializing {task.toFrame}: {e}")

        else:
            if self.debug:
                await log.loader(f'Queuing {task.task_name} to merge later')
            self.queued_dfs.append((task.task_name, df))
            if not self.debug:
                task.results = None
            if self.max_queued_merges and (len(self.queued_dfs) >= self.max_queued_merges):
                await self._process_queued_dataframes()

    async def _process_queued_dataframes(self):
        if not self.queued_dfs:
            await log.warning('nothing queued')
            return

        items = self.queued_dfs
        self.queued_dfs = []

        batch_groups = defaultdict(list)
        sequential = []

        for name, df in items:
            t = self.tasks[name]
            mk = _normalize_merge_key(t.merge_key)
            if t.mergePolicy == "coalesce_left" and mk != "":
                if self.debug:
                    await log.loader(t.task_name)
                    # await log.loader(f'Batched Merging {df.hyper.height()}x{len(df.hyper.fields)} from {t.task_name} -> {t.toFrame}')
                mk_key = ",".join(_merge_key_list(mk))
                batch_groups[(t.toFrame, mk_key)].append((t, df))
            else:
                if self.debug:
                    await log.loader(f'Seq. Merging {t.task_name} -> {t.toFrame}')
                sequential.append((t, df))

        for (to_frame, mk_key), grp in batch_groups.items():
            if not grp: continue
            mk = [x for x in (mk_key.split(",") if mk_key else []) if x]

            combined = None
            for t, df in grp:
                if df is None: continue
                df = self._normalize_task_output(t, df)
                if df is None: continue
                rf = ensure_lazy(df)

                if combined is None:
                    combined = rf
                else:
                    joined = combined.join(rf, on=mk, how="left", suffix="_right")
                    sch_l = _safe_schema(combined)
                    sch_r = _safe_schema(rf)
                    shared = (set(sch_l.keys()) & set(sch_r.keys())) - set(mk)

                    joined = joined.fill_nan(pl.lit(None))
                    if shared:
                        joined = joined.with_columns(
                            [pl.coalesce([pl.col(c), pl.col(f"{c}_right")]).alias(c) for c in shared]
                        ).drop([f"{c}_right" for c in shared])
                    combined = joined

            if combined is not None:
                if self.debug:
                    # {combined.hyper.height()}x{len(combined.hyper.fields)}
                    await log.loader(f'Merging from BATCHED -> {to_frame}')
                try:
                    await self._merge_frames(
                        to_frame=to_frame,
                        right_df=combined,
                        left_on=mk,
                        right_on=mk,
                        merge_policy="coalesce_left",
                    )
                except Exception as e:
                    log.error(f"Batched merge to '{to_frame}' failed: {e}")

        for t, df in sequential:
            try:
                df = self._normalize_task_output(t, df)
                if df is None and not (t.expected_col_provides or {}):  # FIX: no t.provides
                    continue

                t.results = df
                if self.frames.get(t.toFrame) is None:
                    collected = ensure_lazy(await df.hyper.compress_plan_async())
                    _check = False
                    async with self.locks[t.toFrame]:
                        if self.frames.get(t.toFrame) is None:
                            self.frames[t.toFrame] = collected
                            self.invalidate_schema(t.toFrame)
                            _check = True
                    if not _check:
                        await self._merge_task_result(t)
                else:
                    await self._merge_task_result(t)
            except Exception as e:
                log.error(f"Sequential merge for '{t.task_name}' failed: {e}")

            if not self.debug:
                t.results = None

    async def _finalize_main_frame(self, key="main"):
        if key.startswith("_"): return

        await log.loader(f"Finalizing {key} frame...")
        await self.broadcast_log(f"Finalizing {key} frame...")

        frame = self.frames.get(key)
        if (frame is None) or (frame.hyper.is_empty()): return frame

        frame = await frame.hyper.compress_plan_async()
        s = frame.hyper.schema()

        # 1) Remove temp_columns
        declared_temp_cols = {c for c in self.temp_columns if c in s} if self.temp_columns else set()
        temp_cols = {col for col in s.keys() if col.startswith("_")} #if not self.debug else set()
        cols_to_remove = list(declared_temp_cols | temp_cols)

        if cols_to_remove:
            await log.debug(f'Removing from {key}...', cols_to_remove=cols_to_remove)
            frame = frame.drop(cols_to_remove, strict=False)
            self.invalidate_schema(key)
            s = frame.hyper.schema()

        # 2) Standardize nulls
        frame = frame.fill_nan(None)

        # 3) Clean boolean casts
        b = [c for c, d in s.items() if d==pl.Int8]
        frame = frame.with_columns([pl.col(c).fill_null(0).alias(c) for c in b])

        # 4) Collapse Lists
        l = {c:d for c, d in s.items() if d==pl.List}
        frame = frame.with_columns([
            collapse_list(c, d).alias(c) for c,d in l.items()
        ])

        # last) Pass back
        self.frames[key] = await frame.hyper.compress_plan_async()

        if key =="main":
            self.main_df = await self.frames["main"].hyper.collect_async()

        # Rowcount invariant check (only here; not per-join).
        if (key == "main") and (self.initial_row_count is not None):
            try:
                final_height = int(self.main_df.hyper.height())
                initial_height = int(self.initial_row_count)
                if final_height != initial_height:
                    await log.critical(f"Rowcount invariant violated:",
                        initial = initial_height,
                        final = final_height
                    )
            except Exception:
                pass

    async def flush_provider_cache(self):
        if not self._provider_write_queue:
            return

        items = list(self._provider_write_queue.items())
        self._provider_write_queue.clear()

        st = _provider_cache_state()
        tokens = st["write_token"]

        # Schedule writes in background; do not await file IO.
        for path, record in items:
            try:
                _PROVIDER_CACHE_MEM[path] = record
                tokens[path] = tokens.get(path, 0) + 1
                await _spawn_provider_cache_write(path)
            except Exception:
                pass

    async def _run_full(self):
        from app.server import get_s3
        self.s3 = get_s3()
        self.executor = AsyncThreadExecutor(name='data-loader-executor')
        self.executor.start()

        # Reset state for a full run
        self._force_quit.clear()
        self._scheduler_wake.clear()
        self.pending_tasks = set(self.tasks.keys()) - set(self.completed_tasks)
        self.running_tasks = set()
        self.completed_tasks = {n for n, o in self.task_outcome.items() if o == "cancelled"}  # preserve pre-cancelled

        cancel_exc = None
        try:
            while (self.pending_tasks or self.running_tasks) and (not self._force_quit.is_set()):
                self._auto_skip_untriggered_failed_tasks()

                # Early finish if only optional remain
                if await self._should_early_finish_due_to_optional_only():
                    # Cancel/skip all remaining optionals
                    for n in list(self.pending_tasks):
                        if self.tasks[n].isOptional:
                            self._skip_task(n, "optional-only remainder")

                    for t in list(self.running_tasks):
                        n = self.task_to_name.get(t)
                        if n and self.tasks[n].isOptional:
                            self._cancel_running_task_by_name(n, reason="optional-only remainder", detach=True)
                    break

                running_names = self._running_task_names()
                ready = self.dep_graph.get_ready_tasks(self.completed_tasks, running_names, self.column_complete)

                # Apply dynamic triggers (ANY, FAILED)
                filtered_ready = set()
                for name in ready:
                    t = self.tasks[name]

                    # requiresFailedTask: wait for triggers to complete; if none failed => skip.
                    if t.failed_task_requirements:
                        trig = [x for x in t.failed_task_requirements if x in self.tasks]
                        if trig and not set(trig).issubset(self.completed_tasks):
                            continue
                        if trig and not any(self.task_outcome.get(x)=="failed" for x in trig):
                            self._skip_task(name, "requiresFailedTask not satisfied")
                            continue

                    filtered_ready.add(name)

                # If nothing ready, attempt to unblock by skipping optional blockers.
                if not filtered_ready and not self.running_tasks:
                    if await self._maybe_skip_optional_blockers():
                        continue

                    # Deadlock: mark remaining as cancelled with diagnostic.
                    pending = list(self.pending_tasks)
                    diag = []
                    for n in pending[:25]:
                        t = self.tasks[n]
                        blockers = self.dep_graph.blockers(t, self.completed_tasks, self.column_complete)
                        diag.append(f"{n} blocked by: {sorted(blockers)[:8]}")
                    await log.error("Deadlock detected. Skipping remaining tasks.\n" + "\n".join(diag))
                    for n in list(self.pending_tasks):
                        self._skip_task(n, "deadlock")
                    break

                # Schedule ready tasks
                for name in sorted(filtered_ready):
                    if name not in self.pending_tasks:
                        continue
                    t: DataTask = self.tasks[name]
                    self.pending_tasks.remove(name)
                    self.task_outcome[name] = "running"
                    self.task_status[name] = "running"

                    at = self.ctx.spawn(self._run_task_with_retries(t), name=name)
                    self.task_to_name[at] = name
                    self.name_to_task[name] = at
                    self.running_tasks.add(at)

                if self.running_tasks:
                    # Use a wake event as an extra "task" so detach/cancel
                    # can break us out of the wait immediately instead of
                    # blocking up to the full timeout.
                    self._scheduler_wake.clear()
                    wake_task = asyncio.ensure_future(self._scheduler_wake.wait())
                    wait_set = self.running_tasks | {wake_task}
                    done, wait_set = await asyncio.wait(
                        wait_set,
                        timeout=0.5,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # Clean up: remove the wake sentinel from running_tasks tracking.
                    wait_set.discard(wake_task)
                    self.running_tasks = wait_set
                    if not wake_task.done():
                        wake_task.cancel()
                        try:
                            await wake_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    done.discard(wake_task)

                    for dt in done:
                        name = self.task_to_name.pop(dt, None)
                        if not name: continue
                        self.name_to_task.pop(name, None)
                        t = self.tasks[name]

                        try:
                            # If cancellation was requested, ignore any late outputs even if the task "succeeded".
                            if name in self._cancel_requested:
                                try:
                                    _ = dt.result()
                                except asyncio.CancelledError:
                                    pass
                                except Exception:
                                    pass

                                self.task_outcome[name] = "cancelled"
                                self.task_status[name] = "cancelled"
                                t._last_outcome = "cancelled"
                                t._last_error = "cancel_requested"
                                self.completed_tasks.add(name)
                                continue

                            if dt.cancelled():
                                self.task_outcome[name] = "cancelled"
                                self.task_status[name] = "cancelled"
                                t._last_outcome = "cancelled"
                                t._last_error = "cancelled"
                                self.completed_tasks.add(name)
                                continue

                            df = dt.result()
                            df = self._normalize_task_output(t, df)

                            if (df is not None) or t.expected_col_provides:
                                try:
                                    await self._handle_task_completion(t, df)
                                except Exception as merge_err:
                                    log.error(f"Task {name} merge failed: {merge_err}")
                                    t._last_outcome = "failed"
                                    t._last_error = str(merge_err)
                                finally:
                                    try:
                                        await self._update_column_status(t)
                                    except Exception as col_err:
                                        log.error(f"Task {name} column status update failed: {col_err}")

                            # Eagerly cancel optional tasks that are no longer
                            # needed now that column status may have changed.
                            # This fires the cancel_event/cancel_inner immediately
                            # so running tasks see it ASAP — even if they finish
                            # before the next scheduler loop iteration.
                            if self.optional_policy == "background":
                                self._eager_cancel_optional_tasks()

                            outcome = t._last_outcome or "success"
                            self.task_outcome[name] = outcome

                            if outcome=="success":
                                self.task_status[name] = f"completed - {t.duration}s"
                            elif outcome=="failed":
                                self.task_status[name] = f"failed: {t._last_error}"
                            elif outcome=="cancelled":
                                self.task_status[name] = "cancelled"
                            else:
                                self.task_status[name] = outcome

                        except Exception as e:
                            self.task_outcome[name] = "failed"
                            self.task_status[name] = f"failed: {e}"
                            log.error(f"Task {name} failed in scheduler: {e}")
                        finally:
                            self.completed_tasks.add(name)
                            # Cleanup cancellation bookkeeping so a retry/run doesn't inherit stale state.
                            try:
                                ev = self._cancel_events.get(name)
                                if ev is not None and ev.is_set():
                                    # keep object for reuse if you want; otherwise delete:
                                    pass
                            except Exception:
                                pass

            if self._force_quit.is_set():
                return

            provider_flush = self.ctx.spawn(self.flush_provider_cache(), name='flushing provider cache')
            try:
                await self._process_queued_dataframes()
            except Exception as e:
                log.error(f"Post-loop queued merge failed: {e}")
            for frame in ["main", "meta", "benchmarks"]: #list(self.frames.keys()):
                try:
                    await self._finalize_main_frame(key=frame)
                except Exception as e:
                    log.error(f"Finalize frame '{frame}' failed: {e}")
            try:
                await provider_flush
            except Exception as e:
                log.error(f"Provider cache write failed: {e}")

            if self.cleanup_func is not None and isinstance(self.cleanup_func, Callable):
                if asyncio.iscoroutinefunction(self.cleanup_func):
                    self.main_df = await self.cleanup_func(self.main_df)
                else:
                    self.main_df = self.cleanup_func(self.main_df)

            return self.main_df, self.task_status, self.frames

        except asyncio.CancelledError as e:
            cancel_exc = e
            raise

        finally:
            ex = self.executor
            self.executor = None
            if ex is not None:
                try:
                    ex.shutdown()
                except Exception:
                    pass

            # Cancel any still-running tasks
            running = list(self.running_tasks) if self.running_tasks else []
            for t in running:
                try:
                    t.cancel()
                except (Exception, asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            if running:
                try:
                    await asyncio.gather(*running, return_exceptions=True)
                except (asyncio.CancelledError, Exception, asyncio.TimeoutError):
                    pass
            self.running_tasks.clear()
            self.task_to_name.clear()
            self.name_to_task.clear()

            # Cancel broadcasts
            await self.ctx.close()
            if cancel_exc is not None:
                raise cancel_exc

