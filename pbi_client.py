# pbi_client.py

import os
import sys
import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd

# ==== CONFIG ====
WORKSPACE = "UGC UK Trading Reporting"
DATASET = "Strategy document"
TENANT_ID = "db8e2f82-8a37-4c09-b7de-ed06547b5a20"

# If ADOMD.NET installed here:
ADOMD_DIR = r"C:\Program Files\Microsoft.NET\ADOMD.NET\160"

if os.path.isdir(ADOMD_DIR):
    if ADOMD_DIR not in sys.path:
        sys.path.append(ADOMD_DIR)
    os.environ["PATH"] = ADOMD_DIR + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(ADOMD_DIR)
    except Exception:
        pass

try:
    from pyadomd import Pyadomd
    _PYADOMD_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    Pyadomd = None  # type: ignore[assignment]
    _PYADOMD_IMPORT_ERROR = exc

_TOKEN_CACHE: dict[str, object] = {"token": None, "expires_at": 0.0}
_RUNTIME_WARNING_EMITTED = False
_POWERBI_RESOURCE_CACHE: dict[str, str | None] = {"workspace_id": None, "dataset_id": None}


def _pyadomd_unavailable_message() -> str:
    detail = ""
    if _PYADOMD_IMPORT_ERROR is not None:
        detail = f" Import error: {_PYADOMD_IMPORT_ERROR}"
    return (
        "Power BI query runtime is unavailable in this environment. "
        "pyadomd/pythonnet requires a local .NET runtime (Mono/.NET), which is not present here."
        + detail
    )


def _require_pyadomd(raise_on_error: bool = True) -> bool:
    if Pyadomd is not None:
        return True
    if raise_on_error:
        raise RuntimeError(_pyadomd_unavailable_message())
    return False


def _build_base_conn_str() -> str:
    return (
        "Provider=MSOLAP;"
        f"Data Source=powerbi://api.powerbi.com/v1.0/myorg/{WORKSPACE};"
        f"Initial Catalog={DATASET};"
    )


def _get_secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return default


def _build_sp_conn_str() -> str | None:
    client_id = _get_secret("PBI_CLIENT_ID")
    client_secret = _get_secret("PBI_CLIENT_SECRET")
    tenant_id = _get_secret("PBI_TENANT_ID", TENANT_ID)
    if not (client_id and client_secret and tenant_id):
        return None
    return (
        _build_base_conn_str()
        + f"User ID=app:{client_id}@{tenant_id};"
        + f"Password={client_secret};"
    )


def _request_access_token_from_aad() -> str | None:
    client_id = _get_secret("PBI_CLIENT_ID")
    client_secret = _get_secret("PBI_CLIENT_SECRET")
    tenant_id = _get_secret("PBI_TENANT_ID", TENANT_ID)
    if not (client_id and client_secret and tenant_id):
        return None

    now = time.time()
    cached = _TOKEN_CACHE.get("token")
    expires_at = float(_TOKEN_CACHE.get("expires_at", 0.0) or 0.0)
    if cached and now < (expires_at - 60):
        return str(cached)

    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    body = urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://analysis.windows.net/powerbi/api/.default",
        }
    ).encode("utf-8")

    req = Request(token_url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    with urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    token = str(payload.get("access_token", "")).strip()
    if not token:
        return None
    expires_in = int(payload.get("expires_in", 3600))
    _TOKEN_CACHE["token"] = token
    _TOKEN_CACHE["expires_at"] = now + max(expires_in, 60)
    return token


def _get_access_token() -> str | None:
    env_token = _get_secret("PBI_ACCESS_TOKEN")
    if env_token:
        return env_token
    try:
        return _request_access_token_from_aad()
    except Exception:
        return None


def _pbi_api_request_json(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=45) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                return {}
            return json.loads(raw)
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(f"Power BI API HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Power BI API network error: {exc}") from exc


def _resolve_workspace_dataset_ids(token: str) -> tuple[str, str]:
    workspace_id = _POWERBI_RESOURCE_CACHE.get("workspace_id")
    dataset_id = _POWERBI_RESOURCE_CACHE.get("dataset_id")
    if workspace_id and dataset_id:
        return workspace_id, dataset_id

    groups = _pbi_api_request_json("GET", "https://api.powerbi.com/v1.0/myorg/groups?$top=5000", token)
    group_values = groups.get("value", []) if isinstance(groups, dict) else []
    group_match = next(
        (g for g in group_values if str(g.get("name", "")).strip().lower() == WORKSPACE.strip().lower()),
        None,
    )
    if not group_match:
        raise RuntimeError(f"Workspace not found or inaccessible: {WORKSPACE}")
    workspace_id = str(group_match.get("id", "")).strip()
    if not workspace_id:
        raise RuntimeError(f"Workspace id missing for: {WORKSPACE}")

    ds_url = f"https://api.powerbi.com/v1.0/myorg/groups/{workspace_id}/datasets?$top=5000"
    datasets = _pbi_api_request_json("GET", ds_url, token)
    dataset_values = datasets.get("value", []) if isinstance(datasets, dict) else []
    dataset_match = next(
        (d for d in dataset_values if str(d.get("name", "")).strip().lower() == DATASET.strip().lower()),
        None,
    )
    if not dataset_match:
        raise RuntimeError(f"Dataset not found or inaccessible in workspace '{WORKSPACE}': {DATASET}")
    dataset_id = str(dataset_match.get("id", "")).strip()
    if not dataset_id:
        raise RuntimeError(f"Dataset id missing for: {DATASET}")

    _POWERBI_RESOURCE_CACHE["workspace_id"] = workspace_id
    _POWERBI_RESOURCE_CACHE["dataset_id"] = dataset_id
    return workspace_id, dataset_id


def _run_dax_via_rest(dax_query: str, token: str) -> pd.DataFrame:
    workspace_id, dataset_id = _resolve_workspace_dataset_ids(token)
    url = f"https://api.powerbi.com/v1.0/myorg/groups/{workspace_id}/datasets/{dataset_id}/executeQueries"
    payload = {
        "queries": [{"query": dax_query}],
        "serializerSettings": {"includeNulls": True},
    }
    resp = _pbi_api_request_json("POST", url, token, payload=payload)
    if not isinstance(resp, dict):
        return pd.DataFrame()
    if "error" in resp:
        raise RuntimeError(f"Power BI executeQueries error: {resp['error']}")
    results = resp.get("results", [])
    if not results:
        return pd.DataFrame()
    first = results[0] if isinstance(results[0], dict) else {}
    tables = first.get("tables", [])
    if not tables:
        return pd.DataFrame()
    rows = tables[0].get("rows", []) if isinstance(tables[0], dict) else []
    if not rows:
        return pd.DataFrame()
    if isinstance(rows, list) and isinstance(rows[0], dict):
        return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def run_dax(dax_query: str) -> pd.DataFrame:
    strict_runtime = os.getenv("PBI_STRICT_RUNTIME", "").strip().lower() in {"1", "true", "yes", "on"}
    token = _get_access_token()
    if not _require_pyadomd(raise_on_error=strict_runtime):
        global _RUNTIME_WARNING_EMITTED
        if not _RUNTIME_WARNING_EMITTED:
            print(_pyadomd_unavailable_message(), file=sys.stderr)
            _RUNTIME_WARNING_EMITTED = True
        if token:
            try:
                return _run_dax_via_rest(dax_query, token)
            except Exception:
                if strict_runtime:
                    raise
        return pd.DataFrame()
    last_exc: Exception | None = None
    candidates = []
    sp_conn = _build_sp_conn_str()
    if sp_conn:
        candidates.append(("service_principal_conn_str", sp_conn, None))
    candidates.append(("access_token", _build_base_conn_str(), token))
    candidates.append(("base", _build_base_conn_str(), None))

    for mode, conn_str, access_token in candidates:
        conn = Pyadomd(conn_str)
        try:
            if access_token:
                try:
                    conn.conn.AccessToken = access_token
                except Exception:
                    # Some Adomd builds may not expose AccessToken; fallback to next mode.
                    pass
            conn.open()
            with conn.cursor().execute(dax_query) as cur:
                rows = cur.fetchall()
                cols = [c.name for c in cur.description]
                return pd.DataFrame(rows, columns=cols)
        except Exception as exc:
            last_exc = exc
        finally:
            try:
                conn.close()
            except Exception:
                pass

    if token:
        try:
            return _run_dax_via_rest(dax_query, token)
        except Exception as exc:
            last_exc = exc

    hint = (
        "Authentication failed. Set one of: "
        "PBI_ACCESS_TOKEN, or PBI_CLIENT_ID + PBI_CLIENT_SECRET (+ optional PBI_TENANT_ID)."
    )
    if last_exc is None:
        raise RuntimeError(hint)
    raise RuntimeError(f"{hint} Last error: {last_exc}") from last_exc


def _strip_table_prefix(col_name: str) -> str:
    s = str(col_name).strip()
    if "[" in s and s.endswith("]"):
        return s[s.rfind("[") + 1 : -1]
    return s


def _canon(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _pick_col(cols: list[str], candidates: list[str]) -> str | None:
    by_canon = {_canon(c): c for c in cols}
    for c in candidates:
        exact = by_canon.get(_canon(c))
        if exact:
            return exact
    for c in candidates:
        key = _canon(c)
        for col in cols:
            if key in _canon(col):
                return col
    return None


def _to_series_frame(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Date", value_name])

    out = df.copy()
    out.columns = [_strip_table_prefix(c) for c in out.columns]
    cols = list(out.columns)

    date_col = _pick_col(cols, ["FORECASTDATE", "DELIVERYDATE", "SETTLEMENT DAY", "DATE"])
    time_col = _pick_col(cols, ["TIME"])
    value_col = _pick_col(cols, [value_name, f"{value_name}_value", "VALUE", "WIND_MED", "WIND", "SOLAR"])

    if date_col is None:
        date_col = cols[0]
    if value_col is None:
        value_col = cols[-1]

    dt = pd.to_datetime(out[date_col], errors="coerce")
    date_has_intraday_time = False
    if dt.notna().any():
        valid_dt = dt.dropna()
        date_has_intraday_time = bool((valid_dt.dt.normalize() != valid_dt).any())
    if time_col is not None and not date_has_intraday_time:
        t = pd.to_datetime(out[time_col], errors="coerce")
        t_delta = (t - t.dt.normalize()).fillna(pd.Timedelta(0))
        dt = dt.dt.normalize() + t_delta

    series = pd.DataFrame(
        {
            "Date": dt,
            value_name: pd.to_numeric(out[value_col], errors="coerce"),
        }
    ).dropna(subset=["Date", value_name])
    # Only aggregate if duplicate timestamps remain after Date + Time construction.
    if series["Date"].duplicated().any():
        series = series.groupby("Date", as_index=False)[value_name].max()
    return series.sort_values("Date")


def _extract_series_from_raw_table(
    table_name: str,
    date_candidates: list[str],
    value_candidates: list[str],
    out_name: str,
    agg: str = "mean",
) -> pd.DataFrame:
    try:
        raw = run_dax(f"EVALUATE TOPN(200000, '{table_name}')")
    except Exception:
        return pd.DataFrame(columns=["Settlement Day", out_name])
    if raw.empty:
        return pd.DataFrame(columns=["Settlement Day", out_name])

    df = raw.copy()
    df.columns = [_strip_table_prefix(c) for c in df.columns]
    cols = list(df.columns)

    date_col = _pick_col(cols, date_candidates)
    value_col = _pick_col(cols, value_candidates)
    if date_col is None:
        return pd.DataFrame(columns=["Settlement Day", out_name])
    if value_col is None:
        numeric_cols = [c for c in cols if c != date_col and pd.to_numeric(df[c], errors="coerce").notna().any()]
        if not numeric_cols:
            return pd.DataFrame(columns=["Settlement Day", out_name])
        value_col = numeric_cols[0]

    out = pd.DataFrame()
    out["Settlement Day"] = pd.to_datetime(df[date_col], errors="coerce")
    out[out_name] = pd.to_numeric(df[value_col], errors="coerce")
    out = out.dropna(subset=["Settlement Day", out_name])
    if out.empty:
        return pd.DataFrame(columns=["Settlement Day", out_name])

    if agg == "sum":
        out = out.groupby("Settlement Day", as_index=False)[out_name].sum()
    else:
        out = out.groupby("Settlement Day", as_index=False)[out_name].mean()
    return out.sort_values("Settlement Day")


def get_wind_forecast() -> pd.DataFrame:
    dax_candidates = [
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Wattsight Wind Forecast'[FORECASTDATE],
    "Wind", SUM('Wattsight Wind Forecast'[WIND_AVG])
)
ORDER BY 'Wattsight Wind Forecast'[FORECASTDATE]
""",
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Wattsight Wind Forecast'[FORECASTDATE],
    "Wind", MAX('Wattsight Wind Forecast'[WIND_AVG])
)
ORDER BY 'Wattsight Wind Forecast'[FORECASTDATE]
""",
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Wattsight Wind Forecast'[FORECASTDATE],
    "Wind", SUM('Wattsight Wind Forecast'[WIND_MED])
)
ORDER BY 'Wattsight Wind Forecast'[FORECASTDATE]
""",
    ]
    for dax in dax_candidates:
        try:
            return _to_series_frame(run_dax(dax), "Wind")
        except Exception:
            continue
    return pd.DataFrame(columns=["Date", "Wind"])


def get_solar_forecast() -> pd.DataFrame:
    dax_candidates = [
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Solar'[DELIVERYDATE],
    'Solar'[Time],
    "Solar", [Forecast]
)
ORDER BY 'Solar'[DELIVERYDATE], 'Solar'[Time]
""",
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Solar'[DELIVERYDATE],
    'Solar'[Time],
    "Solar", CALCULATE([Forecast])
)
ORDER BY 'Solar'[DELIVERYDATE], 'Solar'[Time]
""",
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Solar'[DELIVERYDATE],
    'Solar'[Time],
    "Solar", SUM('Solar'[VALUE])
)
ORDER BY 'Solar'[DELIVERYDATE], 'Solar'[Time]
""",
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Solar'[FORECASTDATE],
    "Solar", [Forecast]
)
ORDER BY 'Solar'[FORECASTDATE]
""",
    ]
    for dax in dax_candidates:
        try:
            return _to_series_frame(run_dax(dax), "Solar")
        except Exception:
            continue
    return pd.DataFrame(columns=["Date", "Solar"])


def get_niv_forecast() -> pd.DataFrame:
    dax = r"""
EVALUATE
ADDCOLUMNS(
    VALUES('NIV Forecast Display Query'[sp]),
    "Uniper_NIV_Forecast", CALCULATE(SUM('NIV Forecast Display Query'[Uniper NIV Forecast])),
    "Outturn_NIV",         CALCULATE(SUM('NIV Forecast Display Query'[Outturn NIV])),
    "NGC_NIV_Forecast",    CALCULATE([!NGC NIV Forecast]),
    "NGT_Forecast_Error",  CALCULATE([!NGT forecast error])
)
ORDER BY 'NIV Forecast Display Query'[sp]
"""
    return run_dax(dax)


def get_ic_flows() -> pd.DataFrame:
    dax = """
EVALUATE
SUMMARIZECOLUMNS(
    'IC Flows 2'[Settlement Day],
    "IFA", SUM('IC Flows 2'[IFA]),
    "ElecLink", SUM('IC Flows 2'[ElecLink]),
    "East West", SUM('IC Flows 2'[East West]),
    "Britned", SUM('IC Flows 2'[Britned]),
    "IFA 2", SUM('IC Flows 2'[IFA 2]),
    "Moyles", SUM('IC Flows 2'[Moyles]),
    "NEMO", SUM('IC Flows 2'[NEMO]),
    "NSL", SUM('IC Flows 2'[NSL])
)
ORDER BY 'IC Flows 2'[Settlement Day]
"""
    return run_dax(dax)


def get_temperature_forecast() -> pd.DataFrame:
    dax = r"""
EVALUATE
TOPN(
  400,
  SELECTCOLUMNS(
    'climate_uk',
    "Date", 'climate_uk'[Date],
    "Temperature", 'climate_uk'[Normal],
    "Topo", 'climate_uk'[topo],
    "Valid", 'climate_uk'[valid]
  ),
  [Date], DESC
)
"""
    return run_dax(dax)


def get_availability_matrix() -> pd.DataFrame:
    dax = r"""
EVALUATE
SUMMARIZECOLUMNS(
    'REMIT Availability'[bmu],
    'REMIT Availability'[SD],
    "AvailableCapacity", SUM('REMIT Availability'[Available Capacity])
)
ORDER BY 'REMIT Availability'[SD], 'REMIT Availability'[bmu]
"""
    return run_dax(dax)


def get_price_curve() -> pd.DataFrame:
    # Raw-table extraction first (most resilient to schema/measure alias drift).
    mip = _extract_series_from_raw_table(
        table_name="Within Day Prices",
        date_candidates=["Settlement Day", "SETTLEMENT_DAY"],
        value_candidates=["PXP", "MIP"],
        out_name="MIP",
        agg="mean",
    )
    sys = _extract_series_from_raw_table(
        table_name="System Price",
        date_candidates=["Settlement Day", "SD"],
        value_candidates=["SYSTEM_BUY_PRICE", "System Price"],
        out_name="System Price",
        agg="mean",
    )
    da = pd.DataFrame(columns=["Settlement Day", "DA Price"])
    for da_table in ["DA price", "DA Price"]:
        da = _extract_series_from_raw_table(
            table_name=da_table,
            date_candidates=["Settlement Day", "SETTLEMENT_DAY", "Date"],
            value_candidates=["DA Price", "Price", "VALUE"],
            out_name="DA Price",
            agg="mean",
        )
        if not da.empty:
            break

    if not mip.empty or not sys.empty or not da.empty:
        merged = None
        for part in [mip, da, sys]:
            if part.empty:
                continue
            merged = part if merged is None else pd.merge(merged, part, on="Settlement Day", how="outer")
        if merged is not None and not merged.empty:
            return merged.sort_values("Settlement Day")

    mip_candidates = [
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Within Day Prices'[Settlement Day],
    "MIP", AVERAGE('Within Day Prices'[PXP])
)
ORDER BY 'Within Day Prices'[Settlement Day]
""",
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Within Day Prices'[Settlement Day],
    "MIP", SUM('Within Day Prices'[PXP])
)
ORDER BY 'Within Day Prices'[Settlement Day]
""",
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Within Day Prices'[SETTLEMENT_DAY],
    "MIP", AVERAGE('Within Day Prices'[PXP])
)
ORDER BY 'Within Day Prices'[SETTLEMENT_DAY]
""",
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Within Day Prices'[SETTLEMENT_DAY],
    "MIP", SUM('Within Day Prices'[PXP])
)
ORDER BY 'Within Day Prices'[SETTLEMENT_DAY]
""",
    ]
    da_candidates = [
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'DA price'[Settlement Day],
    "DA Price", [DA Price]
)
ORDER BY 'DA price'[Settlement Day]
""",
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'DA price'[Settlement Day],
    "DA Price", CALCULATE([DA Price])
)
ORDER BY 'DA price'[Settlement Day]
""",
    ]
    sys_candidates = [
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'System Price'[Settlement Day],
    "System Price", AVERAGE('System Price'[SYSTEM_BUY_PRICE])
)
ORDER BY 'System Price'[Settlement Day]
""",
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'System Price'[SD],
    "System Price", AVERAGE('System Price'[SYSTEM_BUY_PRICE])
)
ORDER BY 'System Price'[SD]
""",
    ]

    mip = pd.DataFrame()
    for dax in mip_candidates:
        try:
            mip = run_dax(dax)
        except Exception:
            continue
        if not mip.empty:
            break

    da = pd.DataFrame()
    for dax in da_candidates:
        try:
            da = run_dax(dax)
        except Exception:
            continue
        if not da.empty:
            break

    sys = pd.DataFrame()
    for dax in sys_candidates:
        try:
            sys = run_dax(dax)
        except Exception:
            continue
        if not sys.empty:
            break

    if mip.empty and sys.empty and da.empty:
        return pd.DataFrame()

    def _prep(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        if df.empty:
            return df
        date_col = df.columns[0]
        out = df.rename(columns={date_col: "Settlement Day"}).copy()
        # Some engines return aliased column names with table/prefix wrappers.
        if value_col not in out.columns:
            target = "".join(ch for ch in value_col.lower() if ch.isalnum())
            match = None
            for c in out.columns:
                if c == "Settlement Day":
                    continue
                norm = "".join(ch for ch in str(c).lower() if ch.isalnum())
                if target in norm:
                    match = c
                    break
            if match is None and len(out.columns) > 1:
                match = out.columns[1]
            if match is not None and match != value_col:
                out = out.rename(columns={match: value_col})
        out["Settlement Day"] = pd.to_datetime(out["Settlement Day"], errors="coerce")
        out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
        return out.dropna(subset=["Settlement Day"])

    mip = _prep(mip, "MIP") if not mip.empty else pd.DataFrame(columns=["Settlement Day", "MIP"])
    da = _prep(da, "DA Price") if not da.empty else pd.DataFrame(columns=["Settlement Day", "DA Price"])
    sys = _prep(sys, "System Price") if not sys.empty else pd.DataFrame(columns=["Settlement Day", "System Price"])

    merged = mip.copy()
    if merged.empty:
        merged = da.copy() if not da.empty else sys.copy()
    else:
        if not da.empty:
            merged = pd.merge(merged, da, on="Settlement Day", how="outer")
        if not sys.empty:
            merged = pd.merge(merged, sys, on="Settlement Day", how="outer")

    if merged.empty:
        return pd.DataFrame()
    return merged.sort_values("Settlement Day")
