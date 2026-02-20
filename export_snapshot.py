from pathlib import Path

import pandas as pd

from pbi_client import (
    get_availability_matrix,
    get_ic_flows,
    get_niv_forecast,
    get_price_curve,
    get_solar_forecast,
    get_temperature_forecast,
    get_wind_forecast,
    run_dax,
)


OUT_DIR = Path("data_snapshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _write(name: str, df: pd.DataFrame) -> None:
    path = OUT_DIR / f"{name}.csv"
    if df is None:
        df = pd.DataFrame()
    df.to_csv(path, index=False)
    print(f"{name}: {len(df)} rows")


def main() -> None:
    _write("overview_query1", run_dax("EVALUATE TOPN(100, 'Query1')"))
    _write("wind_forecast", get_wind_forecast())
    _write("solar_forecast", get_solar_forecast())

    _write(
        "margins",
        run_dax(
            """
EVALUATE
SUMMARIZECOLUMNS(
    'Margins'[Date],
    "Demand", SUM('Margins'[Demand]),
    "MarginForecast", SUM('Margins'[Margin Forecast])
)
ORDER BY 'Margins'[Date]
"""
        ),
    )
    _write("niv_forecast", get_niv_forecast())
    _write("ic_flows", get_ic_flows())
    _write("temperature_forecast", get_temperature_forecast())
    _write("availability_matrix", get_availability_matrix())
    _write("price_curve", get_price_curve())

    _write(
        "prices_main",
        run_dax(
            r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Prices'[Type],
    "Order", MAX('Prices'[Order]),
    "Sum of WD", SUM('Prices'[WD]),
    "Sum of DA", SUM('Prices'[DA]),
    "Sum of Weekend", SUM('Prices'[Weekend]),
    "Week Ahead", SUM('Prices'[Week Ahead])
)
ORDER BY [Order]
"""
        ),
    )
    _write(
        "prices_sparkdark",
        run_dax(
            r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Prices'[Type],
    "Order", MAX('Prices'[Order]),
    "DA Spark", SUM('Prices'[DA Spark]),
    "DA Dark", SUM('Prices'[DA Dark]),
    "WKND Spark", SUM('Prices'[WKND Spark]),
    "WKND Dark", SUM('Prices'[WKND Dark]),
    "Week Ahead Spark", SUM('Prices'[Week Ahead Spark]),
    "Week Ahead Dark", SUM('Prices'[Week Ahead Dark])
)
ORDER BY [Order]
"""
        ),
    )

    cdc = run_dax("EVALUATE TOPN(200000, 'Query2')")
    if cdc.empty:
        cdc = run_dax("EVALUATE TOPN(200000, 'Query 2')")
    _write("cdc_killpg", cdc)
    _write("unit_status", run_dax("EVALUATE TOPN(5000, 'Query1')"))


if __name__ == "__main__":
    main()
