"""Boxplot de dos cajas: todas las bandejas con iman (pisos 1-8) vs control
(piso 9), juntando todos los experimentos de iman a 5 mL.

A 5 mL solo hay pesos, asi que el endpoint es la imbibicion a 72 h. El control
negativo queda fuera: tiene 5 mL pero no tiene iman. El p-valor sale del mismo
modelo de la tesis (pct_change ~ con_iman + C(bloque)), con TOST a +-10%.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats

from analisis_confirmatorio import (
    ANALYSIS_DIR, CONTROL_FLOOR, DOSIS_CSV_MAP, EXTRA_DIR, MARGEN_REL, METADATA_CSV,
    cargar_datos_peso, guardar_figura,
)

OUT_DIR = ANALYSIS_DIR / "confirmatorio_iman_pooled_5ml"


def cargar() -> pd.DataFrame:
    meta = pd.read_csv(METADATA_CSV, dtype={"exp_id": str})
    meta = meta[meta["experiment_type"].str.startswith("IMAN_") & (meta["water_ml"] == 5)]
    dfs = []
    for (tipo, polo), g in meta.groupby(["experiment_type", "magnet_pole"]):
        df = cargar_datos_peso(sorted(g["exp_id"]), EXTRA_DIR / DOSIS_CSV_MAP[(tipo, polo)])
        df["iman"] = tipo
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    df = cargar()
    df["grupo"] = np.where(df["con_iman"] == 1, "Con imán\n(pisos 1–8)", "Control\n(piso 9, sin campo)")
    orden = ["Control\n(piso 9, sin campo)", "Con imán\n(pisos 1–8)"]
    df["pct"] = 100 * df["pct_change"]  # solo para el eje: el modelo sigue en fraccion

    media_ctrl = df.loc[df["con_iman"] == 0, "pct_change"].mean()
    m = smf.ols("pct_change ~ con_iman + C(bloque)", data=df).fit()
    est, se, dfres = m.params["con_iman"], m.bse["con_iman"], m.df_resid
    lo, hi = m.conf_int().loc["con_iman"]
    p = m.pvalues["con_iman"]
    delta = MARGEN_REL * media_ctrl
    p_tost = max(stats.t.sf((est + delta) / se, dfres), stats.t.cdf((est - delta) / se, dfres))

    n_ctrl, n_iman = (df["con_iman"] == 0).sum(), (df["con_iman"] == 1).sum()
    fechas = sorted(df["fecha"].unique())
    print(f"Fechas: {', '.join(fechas)}")
    print(f"n control = {n_ctrl} | n con iman = {n_iman}")
    print(f"Efecto: {100*est/media_ctrl:+.2f}% IC95 [{100*lo/media_ctrl:+.2f}%, {100*hi/media_ctrl:+.2f}%] "
          f"p = {p:.4f} | p_TOST(+-{MARGEN_REL:.0%}) = {p_tost:.2e}")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.boxplot(data=df, x="grupo", y="pct", order=orden, palette=["tab:red", "tab:blue"],
                width=0.55, showfliers=False, ax=ax,
                medianprops={"color": "black", "linewidth": 2},
                boxprops={"alpha": 0.45, "edgecolor": "k"},
                whiskerprops={"color": "k", "linewidth": 1.3},
                capprops={"color": "k", "linewidth": 1.3})
    sns.stripplot(data=df, x="grupo", y="pct", order=orden, hue="fecha", palette="dark",
                  size=4.5, jitter=0.2, alpha=0.75, ax=ax, zorder=5)
    medias = df.groupby("grupo")["pct"].mean().reindex(orden)
    ax.scatter(range(len(orden)), medias.values, marker="D", s=90, facecolors="white",
               edgecolors="black", linewidths=1.6, zorder=10, label="media")
    ax.plot([], [], color="black", linewidth=2, label="mediana")

    signif = "Diferencia significativa" if p < 0.05 else "Sin diferencia significativa"
    equiv = "equivalentes" if p_tost < 0.05 else "equivalencia no concluyente"
    texto = (f"{signif} (p = {p:.2f})\n"
             f"Δ = {100*est/media_ctrl:+.1f}%  IC95 [{100*lo/media_ctrl:+.1f}%, {100*hi/media_ctrl:+.1f}%]\n"
             f"TOST ±{MARGEN_REL:.0%}: p = {p_tost:.1e} → {equiv}")
    ax.text(0.03, 0.97, texto, transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "gray", "alpha": 0.95},
            zorder=20)

    ymin, ymax = df["pct"].min(), df["pct"].max()
    ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.35 * (ymax - ymin))
    ax.set_xticklabels([f"{o}\nn = {n}" for o, n in zip(orden, [n_ctrl, n_iman])])
    ax.set_xlabel("")
    ax.set_ylabel("Cambio de peso a 72 h (%)\n(post − pre) / pre × 100")
    ax.set_title("Imbibición: imán vs control | todos los experimentos de imán a 5 mL\n"
                 "Cerámica N + Neodimio N, 6 fechas | p ajustado por bloque (fecha × piso físico)",
                 fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, title="fecha", title_fontsize=8)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    guardar_figura(fig, OUT_DIR / "fig_boxplot_iman_vs_control")
    print(f"-> {OUT_DIR / 'fig_boxplot_iman_vs_control'}.pdf")


def contraste(df: pd.DataFrame) -> dict:
    """Con iman vs control con el modelo de la tesis, sobre un subconjunto."""
    media_ctrl = df.loc[df["con_iman"] == 0, "pct_change"].mean()
    m = smf.ols("pct_change ~ con_iman + C(bloque)", data=df).fit()
    est, se, dfres = m.params["con_iman"], m.bse["con_iman"], m.df_resid
    lo, hi = m.conf_int().loc["con_iman"]
    delta = MARGEN_REL * media_ctrl
    p_tost = max(stats.t.sf((est + delta) / se, dfres), stats.t.cdf((est - delta) / se, dfres))
    return {"dif": 100 * est / media_ctrl, "lo": 100 * lo / media_ctrl, "hi": 100 * hi / media_ctrl,
            "p": m.pvalues["con_iman"], "p_tost": p_tost}


def fig_por_iman() -> None:
    """Tres cajas: control (piso 9 de las 6 fechas), Ceramica N y Neodimio N.
    Cada iman se testea contra los controles de sus propias fechas: los bloques
    de un iman no contienen bandejas del otro."""
    df = cargar()
    df["pct"] = 100 * df["pct_change"]
    nombres = {"IMAN_CERAMICA": "Cerámica N\n(pisos 1–8)", "IMAN_NEODIMIO": "Neodimio N\n(pisos 1–8)"}
    ctrl = "Control\n(piso 9, sin campo)"
    df["grupo"] = np.where(df["con_iman"] == 0, ctrl, df["iman"].map(nombres))
    orden = [ctrl, nombres["IMAN_CERAMICA"], nombres["IMAN_NEODIMIO"]]

    lineas = []
    for iman, nombre in nombres.items():
        r = contraste(df[df["iman"] == iman])
        print(f"{nombre.split(chr(10))[0]}: {r['dif']:+.2f}% IC95 [{r['lo']:+.2f}%, {r['hi']:+.2f}%] "
              f"p = {r['p']:.4f} | p_TOST = {r['p_tost']:.2e}")
        signif = "significativo" if r["p"] < 0.05 else "no significativo"
        equiv = "equivalente" if r["p_tost"] < 0.05 else "equivalencia no concluyente"
        lineas.append(f"{nombre.split(chr(10))[0]} vs control: Δ = {r['dif']:+.1f}% "
                      f"[{r['lo']:+.1f}, {r['hi']:+.1f}]\n"
                      f"   p = {r['p']:.2f} ({signif}) · TOST ±{MARGEN_REL:.0%}: p = {r['p_tost']:.1e} ({equiv})")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.boxplot(data=df, x="grupo", y="pct", order=orden, hue="grupo", legend=False,
                palette=dict(zip(orden, ["tab:red", "tab:orange", "tab:blue"])), width=0.55, showfliers=False, ax=ax,
                medianprops={"color": "black", "linewidth": 2},
                boxprops={"alpha": 0.45, "edgecolor": "k"},
                whiskerprops={"color": "k", "linewidth": 1.3},
                capprops={"color": "k", "linewidth": 1.3})
    sns.stripplot(data=df, x="grupo", y="pct", order=orden, hue="fecha", palette="dark",
                  size=4.5, jitter=0.2, alpha=0.75, ax=ax, zorder=5)
    medias = df.groupby("grupo")["pct"].mean().reindex(orden)
    ax.scatter(range(len(orden)), medias.values, marker="D", s=90, facecolors="white",
               edgecolors="black", linewidths=1.6, zorder=10, label="media")
    ax.plot([], [], color="black", linewidth=2, label="mediana")

    ax.text(0.02, 0.97, "\n".join(lineas) + "\n(Δ relativo a la media del control; p ajustado por bloque)",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "gray", "alpha": 0.95},
            zorder=20)

    ymin, ymax = df["pct"].min(), df["pct"].max()
    ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.45 * (ymax - ymin))
    ns = df["grupo"].value_counts().reindex(orden)
    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels([f"{o}\nn = {n}" for o, n in zip(orden, ns)])
    ax.set_xlabel("")
    ax.set_ylabel("Cambio de peso a 72 h (%)\n(post − pre) / pre × 100")
    ax.set_title("Imbibición por tipo de imán vs control | experimentos a 5 mL\n"
                 "Cerámica N: 20260331, 0608, 0622 | Neodimio N: 20260626, 0721, 0727", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, title="fecha", title_fontsize=8)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    guardar_figura(fig, OUT_DIR / "fig_boxplot_por_iman_vs_control")
    print(f"-> {OUT_DIR / 'fig_boxplot_por_iman_vs_control'}.pdf")


if __name__ == "__main__":
    main()
    fig_por_iman()
