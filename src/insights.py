"""
Sistema de Insights Automáticos.
Detecta anomalías, tendencias, correlaciones y oportunidades
en los datos operacionales de Rappi.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.data_loader import (
    WEEK_COLS_METRICS, WEEK_COLS_ORDERS,
    METRIC_DICTIONARY, COUNTRY_NAMES,
)


@dataclass
class Insight:
    category: str          # anomaly | trend | benchmark | correlation | opportunity
    severity: str          # critical | warning | info
    title: str
    finding: str
    impact: str
    recommendation: str
    data: dict | None = None


def detect_anomalies(df_metrics: pd.DataFrame, threshold: float = 0.10) -> list[Insight]:
    """Detecta zonas con cambios drásticos WoW (>threshold)."""
    insights = []

    df = df_metrics.copy()
    df["wow_change"] = (df["L0W_ROLL"] - df["L1W_ROLL"]) / df["L1W_ROLL"].replace(0, np.nan)

    # Filtrar cambios absurdos (baseline near zero) y duplicados
    df = df[df["wow_change"].abs() < 10]  # máximo 1000% cambio
    df = df.drop_duplicates(subset=["ZONE", "METRIC"])

    # Deterioro fuerte
    deteriorated = df[df["wow_change"] < -threshold].sort_values("wow_change")

    for _, row in deteriorated.head(10).iterrows():
        change_pct = row["wow_change"] * 100
        severity = "critical" if change_pct < -20 else "warning"
        country = COUNTRY_NAMES.get(row["COUNTRY"], row["COUNTRY"])

        insights.append(Insight(
            category="anomaly",
            severity=severity,
            title=f"Caída de {row['METRIC']} en {row['ZONE']}",
            finding=f"{row['METRIC']} cayó {abs(change_pct):.1f}% WoW en {row['ZONE']} ({country}, {row['CITY']}). "
                    f"Pasó de {row['L1W_ROLL']:.4f} a {row['L0W_ROLL']:.4f}.",
            impact=f"Zona {row['ZONE_TYPE']}, prioridad: {row['ZONE_PRIORITIZATION']}. "
                   f"{'Alto impacto por ser zona prioritaria.' if row['ZONE_PRIORITIZATION'] == 'High Priority' else 'Monitorear evolución.'}",
            recommendation=f"Investigar causa raíz de la caída en {row['ZONE']}. "
                          f"Revisar si hay cambios operacionales o factores externos.",
            data={"zone": row["ZONE"], "metric": row["METRIC"],
                  "change_pct": change_pct, "country": row["COUNTRY"]},
        ))

    # Mejora fuerte
    improved = df[df["wow_change"] > threshold].sort_values("wow_change", ascending=False)

    for _, row in improved.head(5).iterrows():
        change_pct = row["wow_change"] * 100
        country = COUNTRY_NAMES.get(row["COUNTRY"], row["COUNTRY"])

        insights.append(Insight(
            category="anomaly",
            severity="info",
            title=f"Mejora de {row['METRIC']} en {row['ZONE']}",
            finding=f"{row['METRIC']} mejoró {change_pct:.1f}% WoW en {row['ZONE']} ({country}).",
            impact="Identificar qué está funcionando para replicar en otras zonas.",
            recommendation=f"Analizar acciones recientes en {row['ZONE']} que expliquen la mejora "
                          f"y evaluar replicabilidad.",
            data={"zone": row["ZONE"], "metric": row["METRIC"],
                  "change_pct": change_pct, "country": row["COUNTRY"]},
        ))

    return insights


def detect_trends(df_metrics: pd.DataFrame, consecutive_weeks: int = 3) -> list[Insight]:
    """Detecta métricas en deterioro o mejora consistente por N semanas consecutivas."""
    insights = []
    df = df_metrics.copy()
    weeks = WEEK_COLS_METRICS

    # Calcular cambios semana a semana
    changes = pd.DataFrame()
    for i in range(1, len(weeks)):
        col = f"change_{i}"
        changes[col] = df[weeks[i]] - df[weeks[i - 1]]

    # Deterioro: N semanas consecutivas cayendo (últimas N)
    recent_changes = changes.iloc[:, -(consecutive_weeks):]
    declining = (recent_changes < 0).all(axis=1)

    declining_rows = df[declining].copy()
    if not declining_rows.empty:
        # Calcular magnitud total del deterioro
        declining_rows["total_decline"] = (
            (declining_rows["L0W_ROLL"] - declining_rows[weeks[-(consecutive_weeks + 1)]]) /
            declining_rows[weeks[-(consecutive_weeks + 1)]].replace(0, np.nan)
        )
        declining_rows = declining_rows.sort_values("total_decline")

        for _, row in declining_rows.head(8).iterrows():
            decline_pct = row["total_decline"] * 100 if pd.notna(row["total_decline"]) else 0
            country = COUNTRY_NAMES.get(row["COUNTRY"], row["COUNTRY"])

            if abs(decline_pct) < 1:
                continue

            insights.append(Insight(
                category="trend",
                severity="critical" if abs(decline_pct) > 15 else "warning",
                title=f"Tendencia negativa: {row['METRIC']} en {row['ZONE']}",
                finding=f"{row['METRIC']} lleva {consecutive_weeks}+ semanas en deterioro en "
                        f"{row['ZONE']} ({country}). Caída acumulada: {abs(decline_pct):.1f}%.",
                impact=f"Tendencia sostenida sugiere problema estructural, no puntual.",
                recommendation=f"Escalar a equipo de operaciones de {row['CITY']} para diagnóstico.",
                data={"zone": row["ZONE"], "metric": row["METRIC"],
                      "total_decline_pct": decline_pct, "weeks": consecutive_weeks},
            ))

    return insights


def detect_benchmarking(df_metrics: pd.DataFrame) -> list[Insight]:
    """Compara zonas similares (mismo país/tipo) con performance divergente."""
    insights = []

    for metric in df_metrics["METRIC"].unique():
        df_m = df_metrics[df_metrics["METRIC"] == metric].copy()

        for (country, zone_type), group in df_m.groupby(["COUNTRY", "ZONE_TYPE"]):
            if len(group) < 3:
                continue

            values = group["L0W_ROLL"].dropna()
            if values.empty:
                continue

            mean_val = values.mean()
            std_val = values.std()

            if std_val == 0 or pd.isna(std_val):
                continue

            # Zonas que están >1.5 std por debajo del promedio de su grupo
            underperformers = group[group["L0W_ROLL"] < (mean_val - 1.5 * std_val)]

            for _, row in underperformers.iterrows():
                gap_pct = ((row["L0W_ROLL"] - mean_val) / mean_val) * 100 if mean_val != 0 else 0
                country_name = COUNTRY_NAMES.get(country, country)

                insights.append(Insight(
                    category="benchmark",
                    severity="warning",
                    title=f"{row['ZONE']} bajo en {metric}",
                    finding=f"{row['ZONE']} ({country_name}) tiene {metric} = {row['L0W_ROLL']:.4f}, "
                            f"un {abs(gap_pct):.1f}% por debajo del promedio de zonas {zone_type} "
                            f"en {country_name} ({mean_val:.4f}).",
                    impact=f"Oportunidad de mejora si se equipara al promedio de su grupo.",
                    recommendation=f"Comparar operaciones de {row['ZONE']} con las zonas top del mismo grupo.",
                    data={"zone": row["ZONE"], "metric": metric, "value": row["L0W_ROLL"],
                          "group_mean": mean_val, "gap_pct": gap_pct},
                ))

    # Limitar a los más relevantes
    insights.sort(key=lambda x: abs(x.data.get("gap_pct", 0)) if x.data else 0, reverse=True)
    return insights[:10]


def detect_correlations(df_metrics: pd.DataFrame) -> list[Insight]:
    """Encuentra relaciones entre métricas a nivel zona."""
    insights = []

    # Pivotar para tener métricas como columnas
    pivot = df_metrics.pivot_table(
        index=["COUNTRY", "CITY", "ZONE"],
        columns="METRIC",
        values="L0W_ROLL",
        aggfunc="first",
    )

    if pivot.shape[1] < 2:
        return insights

    # Calcular correlación entre métricas
    corr = pivot.corr()

    # Encontrar pares con alta correlación (positiva o negativa)
    pairs_seen = set()
    for i, m1 in enumerate(corr.columns):
        for j, m2 in enumerate(corr.columns):
            if i >= j:
                continue
            pair = tuple(sorted([m1, m2]))
            if pair in pairs_seen:
                continue

            r = corr.loc[m1, m2]
            if pd.isna(r) or abs(r) < 0.5:
                continue

            pairs_seen.add(pair)
            direction = "positiva" if r > 0 else "negativa"

            insights.append(Insight(
                category="correlation",
                severity="info",
                title=f"Correlación {direction} entre {m1[:25]} y {m2[:25]}",
                finding=f"Se detectó correlación {direction} (r={r:.2f}) entre "
                        f"'{m1}' y '{m2}'. "
                        f"{'Cuando una sube, la otra también.' if r > 0 else 'Cuando una sube, la otra baja.'}",
                impact=f"Puede indicar causalidad o un factor común subyacente.",
                recommendation=f"Investigar si hay una relación causal entre estas métricas "
                              f"para optimizar ambas simultáneamente.",
                data={"metric_1": m1, "metric_2": m2, "correlation": r},
            ))

    insights.sort(key=lambda x: abs(x.data["correlation"]) if x.data else 0, reverse=True)
    return insights[:8]


def detect_order_growth(df_orders: pd.DataFrame) -> list[Insight]:
    """Detecta zonas con crecimiento o caída en órdenes."""
    insights = []
    weeks = WEEK_COLS_ORDERS

    df = df_orders.copy()
    df["growth_5w"] = (df["L0W"] - df["L4W"]) / df["L4W"].replace(0, np.nan)
    df["growth_wow"] = (df["L0W"] - df["L1W"]) / df["L1W"].replace(0, np.nan)

    # Top crecimiento
    top_growth = df.nlargest(5, "growth_5w")
    for _, row in top_growth.iterrows():
        if pd.isna(row["growth_5w"]):
            continue
        growth_pct = row["growth_5w"] * 100
        country = COUNTRY_NAMES.get(row["COUNTRY"], row["COUNTRY"])

        insights.append(Insight(
            category="opportunity",
            severity="info",
            title=f"Crecimiento de órdenes en {row['ZONE']}",
            finding=f"Las órdenes en {row['ZONE']} ({country}) crecieron {growth_pct:.1f}% "
                    f"en las últimas 5 semanas ({row['L4W']:.0f} → {row['L0W']:.0f}).",
            impact="Zona con momentum positivo, potencial para inversión incremental.",
            recommendation="Evaluar si la zona puede absorber más oferta de restaurantes/tiendas.",
            data={"zone": row["ZONE"], "growth_pct": growth_pct, "country": row["COUNTRY"]},
        ))

    # Mayor caída
    bottom = df.nsmallest(3, "growth_5w")
    for _, row in bottom.iterrows():
        if pd.isna(row["growth_5w"]):
            continue
        decline_pct = row["growth_5w"] * 100
        country = COUNTRY_NAMES.get(row["COUNTRY"], row["COUNTRY"])

        insights.append(Insight(
            category="opportunity",
            severity="warning",
            title=f"Caída de órdenes en {row['ZONE']}",
            finding=f"Las órdenes en {row['ZONE']} ({country}) cayeron {abs(decline_pct):.1f}% "
                    f"en 5 semanas ({row['L4W']:.0f} → {row['L0W']:.0f}).",
            impact="Requiere atención para evitar pérdida de cuota de mercado.",
            recommendation=f"Diagnóstico urgente en {row['ZONE']}: revisar competencia, "
                          f"promociones y satisfacción.",
            data={"zone": row["ZONE"], "decline_pct": decline_pct, "country": row["COUNTRY"]},
        ))

    return insights


def generate_all_insights(
    df_metrics: pd.DataFrame,
    df_orders: pd.DataFrame,
) -> list[Insight]:
    """Ejecuta todos los detectores y retorna insights priorizados."""
    all_insights = []

    all_insights.extend(detect_anomalies(df_metrics))
    all_insights.extend(detect_trends(df_metrics))
    all_insights.extend(detect_benchmarking(df_metrics))
    all_insights.extend(detect_correlations(df_metrics))
    all_insights.extend(detect_order_growth(df_orders))

    # Priorizar: critical > warning > info
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    all_insights.sort(key=lambda x: severity_order.get(x.severity, 3))

    return all_insights


def format_executive_report(insights: list[Insight]) -> str:
    """Genera reporte ejecutivo en Markdown."""
    critical = [i for i in insights if i.severity == "critical"]
    warnings = [i for i in insights if i.severity == "warning"]
    info = [i for i in insights if i.severity == "info"]

    report = "# 📊 Reporte Ejecutivo de Insights — Rappi Operations\n\n"

    # Resumen ejecutivo
    report += "## Resumen Ejecutivo\n\n"
    report += f"Se identificaron **{len(insights)}** insights en total: "
    report += f"**{len(critical)}** críticos, **{len(warnings)}** advertencias, "
    report += f"**{len(info)}** informativos.\n\n"

    # Top 5 hallazgos
    report += "### Top 5 Hallazgos Críticos\n\n"
    for i, insight in enumerate(insights[:5], 1):
        emoji = "🔴" if insight.severity == "critical" else "🟡" if insight.severity == "warning" else "🟢"
        report += f"**{i}. {emoji} {insight.title}**\n"
        report += f"- **Hallazgo:** {insight.finding}\n"
        report += f"- **Impacto:** {insight.impact}\n"
        report += f"- **Recomendación:** {insight.recommendation}\n\n"

    # Detalle por categoría
    categories = {
        "anomaly": ("🚨 Anomalías", "Cambios drásticos semana a semana"),
        "trend": ("📉 Tendencias Preocupantes", "Deterioro consistente por 3+ semanas"),
        "benchmark": ("📊 Benchmarking", "Zonas por debajo de su grupo comparable"),
        "correlation": ("🔗 Correlaciones", "Relaciones entre métricas"),
        "opportunity": ("💡 Oportunidades", "Crecimiento y áreas de atención"),
    }

    for cat_key, (cat_title, cat_desc) in categories.items():
        cat_insights = [i for i in insights if i.category == cat_key]
        if not cat_insights:
            continue

        report += f"## {cat_title}\n"
        report += f"*{cat_desc}*\n\n"

        for insight in cat_insights[:5]:
            emoji = "🔴" if insight.severity == "critical" else "🟡" if insight.severity == "warning" else "🟢"
            report += f"### {emoji} {insight.title}\n"
            report += f"**Hallazgo:** {insight.finding}\n\n"
            report += f"**Impacto:** {insight.impact}\n\n"
            report += f"**Recomendación:** {insight.recommendation}\n\n"
            report += "---\n\n"

    return report
