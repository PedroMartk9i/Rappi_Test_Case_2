"""
Carga y preparación de datos operacionales de Rappi.
"""
import pandas as pd
from pathlib import Path


DATA_PATH = Path(__file__).parent.parent / "data" / "rappi_data.xlsx"

WEEK_COLS_METRICS = [
    "L8W_ROLL", "L7W_ROLL", "L6W_ROLL", "L5W_ROLL",
    "L4W_ROLL", "L3W_ROLL", "L2W_ROLL", "L1W_ROLL", "L0W_ROLL",
]

WEEK_COLS_ORDERS = [
    "L8W", "L7W", "L6W", "L5W", "L4W", "L3W", "L2W", "L1W", "L0W",
]

METRIC_DICTIONARY = {
    "% PRO Users Who Breakeven": "Usuarios Pro cuyo valor generado cubre el costo de su membresía / Total Pro",
    "% Restaurants Sessions With Optimal Assortment": "Sesiones con mínimo 40 restaurantes / Total sesiones",
    "Gross Profit UE": "Margen bruto de ganancia / Total de órdenes",
    "Lead Penetration": "Tiendas habilitadas / (Leads + Habilitadas + Salieron)",
    "MLTV Top Verticals Adoption": "Usuarios con órdenes en múltiples verticales / Total usuarios",
    "Non-Pro PTC > OP": "Conversión No-Pro de 'Proceed to Checkout' a 'Order Placed'",
    "Perfect Orders": "Órdenes sin cancelaciones, defectos ni demora / Total órdenes",
    "Pro Adoption (Last Week Status)": "Usuarios Pro / Total usuarios Rappi",
    "Restaurants Markdowns / GMV": "Descuentos restaurantes / Gross Merchandise Value restaurantes",
    "Restaurants SS > ATC CVR": "Conversión restaurantes 'Select Store' a 'Add to Cart'",
    "Restaurants SST > SS CVR": "Conversión 'Store Type' a 'Select Store' en restaurantes",
    "Retail SST > SS CVR": "Conversión 'Store Type' a 'Select Store' en retail/supermercados",
    "Turbo Adoption": "Usuarios Turbo / Total usuarios con Turbo disponible",
}

COUNTRY_NAMES = {
    "AR": "Argentina", "BR": "Brasil", "CL": "Chile", "CO": "Colombia",
    "CR": "Costa Rica", "EC": "Ecuador", "MX": "México", "PE": "Perú", "UY": "Uruguay",
}


def load_data(path: str | Path = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los datasets de métricas y órdenes desde el Excel."""
    df_metrics = pd.read_excel(path, sheet_name="RAW_INPUT_METRICS")
    df_orders = pd.read_excel(path, sheet_name="RAW_ORDERS")
    return df_metrics, df_orders


def get_schema_description() -> str:
    """Genera una descripción textual del schema para inyectar en el prompt del LLM."""
    schema = """
## DataFrames disponibles

### df_metrics (métricas operacionales por zona)
Columnas:
- COUNTRY (str): Código de país — {countries}
- CITY (str): Ciudad
- ZONE (str): Zona operacional / barrio
- ZONE_TYPE (str): "Wealthy" o "Non Wealthy"
- ZONE_PRIORITIZATION (str): "High Priority", "Prioritized", "Not Prioritized"
- METRIC (str): Nombre de la métrica medida
- L8W_ROLL a L0W_ROLL (float): Valor de la métrica en las últimas 9 semanas
  (L8W = hace 8 semanas, L0W = semana actual)

Métricas disponibles y sus significados:
{metrics}

### df_orders (órdenes por zona)
Columnas:
- COUNTRY (str): Código de país
- CITY (str): Ciudad
- ZONE (str): Zona operacional
- METRIC (str): Siempre "Orders"
- L8W a L0W (float): Número de órdenes en cada semana

### Mapeo de países
{country_map}

### Notas para generar código:
- Las columnas de semanas en df_metrics son: {week_cols_m}
- Las columnas de semanas en df_orders son: {week_cols_o}
- L0W/L0W_ROLL = semana más reciente, L8W/L8W_ROLL = hace 8 semanas
- Para calcular cambio WoW (Week over Week) usa L0W vs L1W
- Para tendencias temporales, usa las 9 columnas de semana como serie
- "Zonas problemáticas" = métricas con deterioro consistente o valores bajos
""".format(
        countries=", ".join(f"{k}={v}" for k, v in COUNTRY_NAMES.items()),
        metrics="\n".join(f'- "{k}": {v}' for k, v in METRIC_DICTIONARY.items()),
        country_map=", ".join(f"{k} → {v}" for k, v in COUNTRY_NAMES.items()),
        week_cols_m=", ".join(WEEK_COLS_METRICS),
        week_cols_o=", ".join(WEEK_COLS_ORDERS),
    )
    return schema


if __name__ == "__main__":
    df_m, df_o = load_data()
    print(f"Métricas: {df_m.shape}")
    print(f"Órdenes: {df_o.shape}")
    print(get_schema_description())
