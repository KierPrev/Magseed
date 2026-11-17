"""Analisis confirmatorio (peso, germinacion, largo de radicula) para los
experimentos de iman, con seleccion interactiva de fechas por TUI.

Uso:
    python src/analisis_confirmatorio.py

Elegis las fechas con espacio, confirmas con enter. El script las agrupa
por (experiment_type, magnet_pole), corre el analisis por grupo y guarda
figuras/tablas/reporte en data/analysis/confirmatorio_<grupo>/.

Metodologia (igual para todos los grupos):
  - Unidad experimental = LA BANDEJA (nunca la semilla).
  - Diseno por bloques: bloque = fecha-floor. Los grupos de neodimio son
    completos y balanceados; ceramica tiene un desbalance declarado en 20260331.
  - pct_change = imbibicion a 72h, no biomasa.
  - Germinacion (si el experimento la registro) se analiza como proporcion
    por bandeja, nunca a nivel semilla (seria pseudo-replicacion).
  - Largo de radicula: endpoint principal = mediana por bandeja.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.figure import Figure
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
EXTRA_DIR = PROJECT_ROOT / "data" / "extra"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
METADATA_CSV = PROJECT_ROOT / "data" / "experiments_metadata.csv"

CONTROL_FLOOR = 9
N_SEMILLAS = 20
MARGEN_REL = 0.10
MIN_RAICES = 3
FACTOR_SOSPECHA = 3

# Tabla de dosis (mT por piso) segun tipo de iman y polo. Si una combinacion
# no esta aca, el grupo se excluye del analisis dosis-respuesta (y se avisa).
DOSIS_CSV_MAP: dict[tuple[str, str], str] = {
    ("IMAN_CERAMICA", "N"): "dosis_ceramica_compart.csv",
    ("IMAN_NEODIMIO", "N"): "dosis_neodimio_compart.csv",
}


# =============================================================================
# Inventario de experimentos
# =============================================================================


@dataclass
class EstadoExperimento:
    exp_id: str
    experiment_type: str
    magnet_pole: str
    water_ml: str
    pesos_ok: bool
    fotos_raw: bool
    radicula_procesada: bool
    germinacion_registrada: bool
    dosis_disponible: bool


def _tiene_radicula(path: Path) -> bool:
    """root_lengths.csv puede existir con solo el header: el pipeline de fotos
    corrio pero no midio ninguna raiz. Para el analisis eso es 'no hay datos'."""
    if not path.exists():
        return False
    return not pd.read_csv(path).empty


def construir_inventario() -> list[EstadoExperimento]:
    meta = pd.read_csv(METADATA_CSV)
    inventario = []
    for _, row in meta.iterrows():
        exp_id = str(row["exp_id"])
        exp_dir = EXPERIMENTS_DIR / exp_id
        trays_path = exp_dir / "trays.csv"

        pesos_ok = False
        germinacion_registrada = False
        if trays_path.exists():
            trays = pd.read_csv(trays_path)
            trays.columns = trays.columns.str.strip()
            if {"pre_weight", "post_weight"}.issubset(trays.columns):
                pesos_ok = trays[["pre_weight", "post_weight"]].notna().all(axis=None)
            germinacion_registrada = "no_germinadas" in trays.columns and trays["no_germinadas"].notna().any()

        fotos_raw = (exp_dir / "raw" / "photos").exists() and any((exp_dir / "raw" / "photos").iterdir())
        radicula_procesada = _tiene_radicula(exp_dir / "processed" / "root_lengths.csv")
        dosis_disponible = (row["experiment_type"], row["magnet_pole"]) in DOSIS_CSV_MAP

        # pandas lee la columna como float, asi que "5" vuelve como "5.0" y
        # ensuciaria el nombre del grupo. Se normaliza a entero cuando lo es.
        agua = row.get("water_ml", "")
        if pd.isna(agua) or str(agua).strip() == "":
            agua = ""
        else:
            agua = f"{float(agua):g}"

        inventario.append(EstadoExperimento(
            exp_id=exp_id,
            experiment_type=row["experiment_type"],
            magnet_pole=row["magnet_pole"],
            water_ml=agua,
            pesos_ok=pesos_ok,
            fotos_raw=fotos_raw,
            radicula_procesada=radicula_procesada,
            germinacion_registrada=germinacion_registrada,
            dosis_disponible=dosis_disponible,
        ))
    return inventario


def _marca(ok: bool) -> str:
    return "OK " if ok else "-- "


def etiqueta(e: EstadoExperimento) -> str:
    agua = f"{e.water_ml}mL" if e.water_ml else "?mL"
    tipo = f"{e.experiment_type}/{e.magnet_pole} {agua}"
    extra = "" if e.dosis_disponible else "  [sin tabla de dosis, no se puede graficar]"
    return (
        f"{e.exp_id}  {tipo:22s}  "
        f"pesos:{_marca(e.pesos_ok)} fotos_raw:{_marca(e.fotos_raw)} "
        f"radicula:{_marca(e.radicula_procesada)} germinacion:{_marca(e.germinacion_registrada)}"
        f"{extra}"
    )


def seleccionar_fechas(inventario: list[EstadoExperimento]) -> list[str]:
    from prompt_toolkit.shortcuts import checkboxlist_dialog

    values = [(e.exp_id, etiqueta(e)) for e in inventario]
    seleccion = checkboxlist_dialog(
        title="Analisis confirmatorio",
        text="Elegi las fechas a incluir (espacio para marcar, enter para confirmar).\n"
             "Se agrupan automaticamente por tipo de iman + polo.",
        values=values,
    ).run()
    return seleccion or []


# =============================================================================
# Carga y analisis estadistico (peso / germinacion / radicula)
# =============================================================================


@dataclass
class ResultadoPeso:
    conf_df: pd.DataFrame
    vs_control: pd.DataFrame
    media_control: float
    modelo_iman: Any
    modelo_dosis: Any
    p_tost: float
    mde_rel: float
    margen_rel: float
    diagnostico: str


@dataclass
class ResultadoGerminacion:
    disponible: bool
    gm_df: pd.DataFrame | None = None
    vs_control: pd.DataFrame | None = None
    modelo_iman: Any = None
    modelo_dosis: Any = None
    modelo_niveles: Any = None
    p_perm: float | None = None
    diagnostico: str = ""


@dataclass
class ResultadoRadicula:
    disponible: bool
    rl_df: pd.DataFrame | None = None
    vs_control: pd.DataFrame | None = None
    media_control: float | None = None
    modelo_iman: Any = None
    modelo_dosis: Any = None
    mde_mm: float | None = None
    diagnostico: str = ""


@dataclass
class ResultadoGrupo:
    nombre: str
    titulo: str
    fechas: list[str]
    peso: ResultadoPeso
    germinacion: ResultadoGerminacion
    radicula: ResultadoRadicula
    figuras: dict[str, Figure] = field(default_factory=dict)


def cargar_datos_peso(fechas: list[str], dosis_csv: Path, control_floor: int = CONTROL_FLOOR,
                       n_semillas: int = N_SEMILLAS) -> pd.DataFrame:
    dfs = []
    for fecha in fechas:
        path = EXPERIMENTS_DIR / fecha / "trays.csv"
        tmp = pd.read_csv(path)
        tmp.columns = tmp.columns.str.strip()
        tmp["fecha"] = fecha
        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)
    df["treatment_floor"] = pd.to_numeric(df["treatment_floor"], errors="coerce")
    df = df.dropna(subset=["treatment_floor", "pre_weight", "post_weight"]).copy()
    df["treatment_floor"] = df["treatment_floor"].astype(int)
    df["position"] = df["position"].astype(int)
    df["pct_change"] = (df["post_weight"] - df["pre_weight"]) / df["pre_weight"]
    df["bloque"] = df["fecha"] + "-" + df["floor"].astype(str)
    df["con_iman"] = (df["treatment_floor"] != control_floor).astype(int)

    if "no_germinadas" in df.columns:
        df["no_germinadas"] = pd.to_numeric(df["no_germinadas"], errors="coerce")
        df["germinadas"] = n_semillas - df["no_germinadas"]
        df["prop_germ"] = df["germinadas"] / n_semillas

    dosis = pd.read_csv(dosis_csv).rename(columns={"nivel": "treatment_floor"})
    df = df.merge(dosis, on="treatment_floor", how="left")
    df["log10_B"] = np.log10(df["B_compart_mT"])
    return df


def analizar_peso(conf_df: pd.DataFrame, control_floor: int = CONTROL_FLOOR,
                   margen_rel: float = MARGEN_REL) -> ResultadoPeso:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dosis = conf_df.drop_duplicates("treatment_floor")[["treatment_floor", "B_compart_mT"]]
        media_control = conf_df.loc[conf_df["treatment_floor"] == control_floor, "pct_change"].mean()
        formula_bloque = " + C(bloque)"

        print("=" * 72)
        print("VERIFICACION DEL DISENO")
        print("=" * 72)
        tab = pd.crosstab(conf_df["treatment_floor"], conf_df["bloque"])
        print(tab)
        print(f"\nBloques completos y balanceados: {(tab.values == 1).all()}")
        print(f"n bandejas = {len(conf_df)} | control = {(conf_df.con_iman == 0).sum()} | tratadas = {(conf_df.con_iman == 1).sum()}")

        print("\n" + "=" * 72)
        print(f"ENDPOINT 1: IMBIBICION A 72H  (media control = {media_control:.4f})")
        print("=" * 72)
        f_full = f"pct_change ~ C(treatment_floor, Treatment(reference={control_floor})){formula_bloque}"
        m_full = smf.ols(f_full, data=conf_df).fit()
        print("Test global (omnibus) del tratamiento:")
        print(anova_lm(m_full, typ=2).round(5))

        pre = f"C(treatment_floor, Treatment(reference={control_floor}))[T."
        ci = m_full.conf_int()
        rows = []
        for term in m_full.params.index:
            if term.startswith(pre):
                fl = int(term.split("T.")[1].rstrip("]"))
                rows.append({
                    "floor": fl,
                    "B_mT": float(dosis.loc[dosis.treatment_floor == fl, "B_compart_mT"].iloc[0]),
                    "diff": m_full.params[term],
                    "ci_low": ci.loc[term, 0],
                    "ci_high": ci.loc[term, 1],
                    "p": m_full.pvalues[term],
                })
        vs_control = pd.DataFrame(rows).sort_values("floor").reset_index(drop=True)
        vs_control["p_holm"] = multipletests(vs_control["p"], method="holm")[1]
        for col, src in [("dif_%", "diff"), ("ic_low_%", "ci_low"), ("ic_high_%", "ci_high")]:
            vs_control[col] = 100 * vs_control[src] / media_control

        print("\nCada piso vs control, en % relativo a la media del control:")
        print(vs_control[["floor", "B_mT", "dif_%", "ic_low_%", "ic_high_%", "p", "p_holm"]]
              .round(3).to_string(index=False))

        print("\n--- Contrastes de 1 grado de libertad ---")
        m_iman = smf.ols(f"pct_change ~ con_iman{formula_bloque}", data=conf_df).fit()
        est = m_iman.params["con_iman"]
        se = m_iman.bse["con_iman"]
        dfres = m_iman.df_resid
        lo, hi = m_iman.conf_int().loc["con_iman"]
        print(f"A) Cualquier iman vs control: {100*est/media_control:+.2f}% "
              f"IC95 [{100*lo/media_control:+.2f}%, {100*hi/media_control:+.2f}%] "
              f"p = {m_iman.pvalues['con_iman']:.4f}")
        tratados = conf_df[conf_df["con_iman"] == 1]
        m_dosis = smf.ols(f"pct_change ~ log10_B{formula_bloque}", data=tratados).fit()
        print(f"B) Tendencia dosis-respuesta: {m_dosis.params['log10_B']:+.5f} por decada de mT "
              f"p = {m_dosis.pvalues['log10_B']:.4f}")

        print("\n--- EQUIVALENCIA (TOST) ---")
        delta = margen_rel * media_control
        p_tost = max(stats.t.sf((est + delta) / se, dfres), stats.t.cdf((est - delta) / se, dfres))
        ic90 = m_iman.conf_int(alpha=0.10).loc["con_iman"]
        mde = (stats.t.ppf(0.975, dfres) + stats.t.ppf(0.80, dfres)) * se
        mde_rel = mde / media_control
        print(f"Margen de relevancia: +-{margen_rel:.0%} = +-{delta:.5f}")
        print(f"IC90 del efecto: [{ic90[0]:+.5f}, {ic90[1]:+.5f}]  p_TOST = {p_tost:.5f}")
        print(f"  -> {'EQUIVALENTE' if p_tost < 0.05 else 'NO concluyente'}")
        print(f"Efecto minimo detectable (80% potencia): {100*mde_rel:.2f}% relativo")

        print("\n--- Supuestos y robustez ---")
        print(f"Shapiro-Wilk sobre residuos: p = {stats.shapiro(m_full.resid)[1]:.4f}")
        print(f"Levene entre grupos: p = {stats.levene(*[g['pct_change'].values for _, g in conf_df.groupby('treatment_floor')])[1]:.4f}")
        cook = m_full.get_influence().cooks_distance[0]
        print(f"Cook D max = {cook.max():.3f} (umbral 4/n = {4/len(conf_df):.3f}) -> "
              f"{(cook > 4/len(conf_df)).sum()} bandejas influyentes")
        print("\nSensibilidad:")
        for nombre, extra in [("solo tratamiento", ""), ("+ bloque", " + C(bloque)"),
                               ("+ position", " + C(position)"), ("+ bloque + position", " + C(bloque) + C(position)")]:
            ms = smf.ols(f"pct_change ~ con_iman{extra}", data=conf_df).fit()
            print(f"  {nombre:22s} efecto {100*ms.params['con_iman']/media_control:+.2f}% "
                  f"SE {ms.bse['con_iman']:.5f} p {ms.pvalues['con_iman']:.4f}")

    return ResultadoPeso(
        conf_df=conf_df, vs_control=vs_control, media_control=media_control,
        modelo_iman=m_iman, modelo_dosis=m_dosis, p_tost=p_tost, mde_rel=mde_rel,
        margen_rel=margen_rel, diagnostico=buf.getvalue(),
    )


def analizar_germinacion(conf_df: pd.DataFrame, n_perm: int = 5000, seed: int = 0) -> ResultadoGerminacion:
    if "germinadas" not in conf_df.columns or conf_df["germinadas"].isna().all():
        return ResultadoGerminacion(
            disponible=False,
            diagnostico="Este grupo no registro germinacion (sin columna no_germinadas en trays.csv).",
        )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gm_df = conf_df.dropna(subset=["germinadas"]).copy()
        gm_tratados = gm_df[gm_df["con_iman"] == 1]

        print("=" * 72)
        print(f"ENDPOINT 2: GERMINACION (unidad = bandeja, n = {len(gm_df)})")
        print("=" * 72)
        print(gm_df.groupby("con_iman")["prop_germ"]
              .agg(["count", "mean", "std"]).rename(index={0: "control", 1: "con iman"}).round(3))

        m_iman = smf.ols("prop_germ ~ con_iman + C(bloque)", data=gm_df).fit()
        lo, hi = m_iman.conf_int().loc["con_iman"]
        print(f"\nA) Cualquier iman vs control: {100*m_iman.params['con_iman']:+.1f} pp "
              f"IC95 [{100*lo:+.1f}, {100*hi:+.1f}] p = {m_iman.pvalues['con_iman']:.4f}")

        m_dosis = smf.ols("prop_germ ~ log10_B + C(bloque)", data=gm_tratados).fit()
        obs = m_dosis.params["log10_B"]
        print(f"B) Tendencia dosis: {100*obs:+.1f} pp por decada de mT p = {m_dosis.pvalues['log10_B']:.4f}")

        # Contrastes secundarios nivel contra control con el mismo ajuste por
        # bloque declarado para el resto del analisis. La figura anterior usaba
        # t de Welch sin bloque y, por lo tanto, no coincidia con el modelo del
        # texto. Treatment(reference=9) hace que cada coeficiente sea la
        # diferencia del nivel correspondiente respecto del control interno.
        m_niveles = smf.ols(
            f"prop_germ ~ C(treatment_floor, Treatment(reference={CONTROL_FLOOR})) + C(bloque)",
            data=gm_df,
        ).fit()
        filas = []
        for nivel in sorted(gm_tratados["treatment_floor"].unique()):
            termino = (
                f"C(treatment_floor, Treatment(reference={CONTROL_FLOOR}))"
                f"[T.{int(nivel)}]"
            )
            lo_nivel, hi_nivel = m_niveles.conf_int().loc[termino]
            filas.append({
                "nivel": int(nivel),
                "B_mT": gm_df.loc[
                    gm_df["treatment_floor"] == nivel, "B_compart_mT"
                ].iloc[0],
                "dif_semillas": N_SEMILLAS * m_niveles.params[termino],
                "dif_pp": 100 * m_niveles.params[termino],
                "ic95_low_pp": 100 * lo_nivel,
                "ic95_high_pp": 100 * hi_nivel,
                "p": m_niveles.pvalues[termino],
            })
        vs_control = pd.DataFrame(filas)
        vs_control["p_holm"] = multipletests(vs_control["p"], method="holm")[1]
        print("\nC) Cada nivel vs control, ajustado por bloque (Holm):")
        print(vs_control.round({
            "B_mT": 1, "dif_semillas": 2, "dif_pp": 1,
            "ic95_low_pp": 1, "ic95_high_pp": 1, "p": 4, "p_holm": 4,
        }).to_string(index=False))

        rng = np.random.default_rng(seed)
        perm = gm_tratados.copy()
        extremos = 0
        for _ in range(n_perm):
            perm["log10_B"] = perm.groupby("bloque")["log10_B"].transform(lambda s: rng.permutation(s.values))
            coef = smf.ols("prop_germ ~ log10_B + C(bloque)", data=perm).fit().params["log10_B"]
            extremos += abs(coef) >= abs(obs)
        p_perm = (extremos + 1) / (n_perm + 1)
        print(f"   p por permutacion dentro de bloque ({n_perm} remuestreos): {p_perm:.4f}")

        print("\n--- Sensibilidad (peso ajustado por germinacion; sesgo de mediador) ---")
        sens_m = smf.ols("pct_change ~ con_iman + no_germinadas + C(bloque)", data=gm_df).fit()
        sens_base = smf.ols("pct_change ~ con_iman + C(bloque)", data=gm_df).fit()
        media_control_peso = conf_df.loc[conf_df["con_iman"] == 0, "pct_change"].mean()
        print(f"  sin ajustar : {100*sens_base.params['con_iman']/media_control_peso:+.2f}% p = {sens_base.pvalues['con_iman']:.4f}")
        print(f"  ajustado    : {100*sens_m.params['con_iman']/media_control_peso:+.2f}% p = {sens_m.pvalues['con_iman']:.4f}")

    return ResultadoGerminacion(
        disponible=True, gm_df=gm_df, vs_control=vs_control,
        modelo_iman=m_iman, modelo_dosis=m_dosis, modelo_niveles=m_niveles,
        p_perm=p_perm, diagnostico=buf.getvalue(),
    )


def cargar_datos_radicula(fechas: list[str], dosis_csv: Path, control_floor: int = CONTROL_FLOOR,
                           min_raices: int = MIN_RAICES, factor_sospecha: int = FACTOR_SOSPECHA) -> pd.DataFrame | None:
    # Todo o nada: la radicula solo se analiza si las 3 fechas del grupo tienen
    # fotos medidas. Con una sola fecha el "bloque" seria la fecha misma y la
    # comparacion no es la que se aleatorizo.
    faltantes = [f for f in fechas if not _tiene_radicula(EXPERIMENTS_DIR / f / "processed" / "root_lengths.csv")]
    if faltantes:
        print(f"Advertencia: sin raices medidas en {faltantes}; se omite el endpoint de radicula "
              f"para este grupo (se necesitan las {len(fechas)} fechas).")
        return None

    raices = []
    for fecha in fechas:
        tmp = pd.read_csv(EXPERIMENTS_DIR / fecha / "processed" / "root_lengths.csv")
        tmp["fecha"] = fecha
        raices.append(tmp)
    raices = pd.concat(raices, ignore_index=True)

    trays = []
    for fecha in fechas:
        tmp = pd.read_csv(EXPERIMENTS_DIR / fecha / "trays.csv")
        tmp.columns = tmp.columns.str.strip()
        tmp["fecha"] = fecha
        trays.append(tmp[["tray_id", "fecha", "treatment_floor", "floor", "position"]])
    trays = pd.concat(trays, ignore_index=True)
    trays["treatment_floor"] = pd.to_numeric(trays["treatment_floor"], errors="coerce")
    trays = trays.dropna(subset=["treatment_floor"])
    trays["treatment_floor"] = trays["treatment_floor"].astype(int)
    trays["bloque"] = trays["fecha"] + "-" + trays["floor"].astype(str)
    trays["con_iman"] = (trays["treatment_floor"] != control_floor).astype(int)
    dosis = pd.read_csv(dosis_csv).rename(columns={"nivel": "treatment_floor"})
    trays = trays.merge(dosis, on="treatment_floor", how="left")
    trays["log10_B"] = np.log10(trays["B_compart_mT"])

    por_bandeja = (raices.groupby("tray_id")["length_mm"]
                   .agg(mediana_mm="median", media_mm="mean", n_raices="count").reset_index())
    rl_df = trays.merge(por_bandeja, on="tray_id", how="left")
    rl_df["n_raices"] = rl_df["n_raices"].fillna(0).astype(int)
    rl_df["sospechosa"] = rl_df["n_raices"] > factor_sospecha * rl_df.groupby("fecha")["n_raices"].transform("median")
    rl_df = rl_df[rl_df["n_raices"] >= min_raices].copy()
    return rl_df


def analizar_radicula(rl_df: pd.DataFrame | None, control_floor: int = CONTROL_FLOOR) -> ResultadoRadicula:
    if rl_df is None or rl_df.empty:
        return ResultadoRadicula(disponible=False, diagnostico="Sin datos de radicula procesados para este grupo.")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dosis = rl_df.drop_duplicates("treatment_floor")[["treatment_floor", "B_compart_mT"]]

        print("=" * 72)
        print("ENDPOINT 3: LARGO DE RADICULA (unidad = bandeja)")
        print("=" * 72)
        print(rl_df.groupby("fecha")["n_raices"].describe()[["count", "min", "50%", "max"]].to_string())
        if rl_df["sospechosa"].any():
            print("\nMarcadas como probable sobre-segmentacion:")
            print(rl_df.loc[rl_df["sospechosa"], ["tray_id", "treatment_floor", "n_raices", "mediana_mm"]]
                  .round(2).to_string(index=False))

        tab = pd.crosstab(rl_df["treatment_floor"], rl_df["bloque"])
        print(f"\nn bandejas = {len(rl_df)} | bandejas por piso: {tab.sum(axis=1).to_dict()}")
        print(f"Bloques completos: {(tab.values == 1).all()}")

        media_control = rl_df.loc[rl_df["treatment_floor"] == control_floor, "mediana_mm"].mean()
        print(f"\nMediana por bandeja, media del control = {media_control:.2f} mm")

        f_full = f"mediana_mm ~ C(treatment_floor, Treatment(reference={control_floor})) + C(bloque)"
        m_full = smf.ols(f_full, data=rl_df).fit()
        print("\nTest global (omnibus) del tratamiento:")
        print(anova_lm(m_full, typ=2).round(5))

        pre = f"C(treatment_floor, Treatment(reference={control_floor}))[T."
        ci = m_full.conf_int()
        rows = []
        for term in m_full.params.index:
            if term.startswith(pre):
                fl = int(term.split("T.")[1].rstrip("]"))
                rows.append({
                    "floor": fl,
                    "B_mT": float(dosis.loc[dosis.treatment_floor == fl, "B_compart_mT"].iloc[0]),
                    "diff_mm": m_full.params[term],
                    "ci_low": ci.loc[term, 0],
                    "ci_high": ci.loc[term, 1],
                    "p": m_full.pvalues[term],
                })
        vs_control = pd.DataFrame(rows).sort_values("floor").reset_index(drop=True)
        vs_control["p_holm"] = multipletests(vs_control["p"], method="holm")[1]
        for col, src in [("dif_%", "diff_mm"), ("ic_low_%", "ci_low"), ("ic_high_%", "ci_high")]:
            vs_control[col] = 100 * vs_control[src] / media_control
        print("\nCada piso vs control, ajustado por bloque, en % relativo al control:")
        print(vs_control[["floor", "B_mT", "diff_mm", "dif_%", "ic_low_%", "ic_high_%", "p", "p_holm"]]
              .round(3).to_string(index=False))

        print("\n--- Contrastes de 1 grado de libertad ---")
        m_iman = smf.ols("mediana_mm ~ con_iman + C(bloque)", data=rl_df).fit()
        est = m_iman.params["con_iman"]
        lo, hi = m_iman.conf_int().loc["con_iman"]
        print(f"A) Cualquier iman vs control: {est:+.2f} mm ({100*est/media_control:+.1f}%) "
              f"IC95 [{100*lo/media_control:+.1f}%, {100*hi/media_control:+.1f}%] "
              f"p = {m_iman.pvalues['con_iman']:.4f}")
        m_dosis = smf.ols("mediana_mm ~ log10_B + C(bloque)", data=rl_df[rl_df["con_iman"] == 1]).fit()
        print(f"B) Tendencia dosis-respuesta: {m_dosis.params['log10_B']:+.2f} mm por decada de mT "
              f"p = {m_dosis.pvalues['log10_B']:.4f}")
        mde = (stats.t.ppf(0.975, m_iman.df_resid) + stats.t.ppf(0.80, m_iman.df_resid)) * m_iman.bse["con_iman"]
        print(f"Efecto minimo detectable (80% potencia): {mde:.2f} mm = {100*mde/media_control:.1f}% relativo")

        print("\n--- Supuestos ---")
        print(f"Shapiro-Wilk sobre residuos: p = {stats.shapiro(m_full.resid)[1]:.4f}")
        print(f"Levene entre grupos: p = {stats.levene(*[g['mediana_mm'].values for _, g in rl_df.groupby('treatment_floor')])[1]:.4f}")

        print("\n--- Sensibilidad ---")
        for nombre, data, y in [
            ("mediana por bandeja (principal)", rl_df, "mediana_mm"),
            ("media por bandeja", rl_df, "media_mm"),
            ("sin bandejas sospechosas", rl_df[~rl_df["sospechosa"]], "mediana_mm"),
        ]:
            ms = smf.ols(f"{y} ~ con_iman + C(bloque)", data=data).fit()
            print(f"  {nombre:32s} {ms.params['con_iman']:+.2f} mm  p = {ms.pvalues['con_iman']:.4f}  (n={len(data)})")

    return ResultadoRadicula(
        disponible=True, rl_df=rl_df, vs_control=vs_control, media_control=media_control,
        modelo_iman=m_iman, modelo_dosis=m_dosis, mde_mm=mde, diagnostico=buf.getvalue(),
    )


# =============================================================================
# Figuras
# =============================================================================


def fig_forest_peso_germinacion(resultado_peso: ResultadoPeso, resultado_germ: ResultadoGerminacion,
                                 titulo: str, control_floor: int = CONTROL_FLOOR) -> Figure:
    import matplotlib.pyplot as plt

    vs_control = resultado_peso.vs_control
    margen_rel = resultado_peso.margen_rel

    n_panels = 2 if resultado_germ.disponible else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(13 if n_panels == 2 else 7, 6), sharey=n_panels == 2)
    ax_peso = axes[0] if n_panels == 2 else axes

    ax_peso.axvspan(-100 * margen_rel, 100 * margen_rel, color="tab:green", alpha=0.10,
                     label=f"zona de equivalencia (+-{margen_rel:.0%})")
    ax_peso.axvline(0, color="black", linestyle="-", linewidth=0.8)
    for signo in (-1, 1):
        ax_peso.axvline(signo * 100 * margen_rel, color="tab:green", linestyle=":", linewidth=1.2)
    ax_peso.errorbar(vs_control["dif_%"], vs_control["floor"],
                      xerr=[vs_control["dif_%"] - vs_control["ic_low_%"],
                            vs_control["ic_high_%"] - vs_control["dif_%"]],
                      fmt="o", color="black", ecolor="gray", elinewidth=1.6, capsize=4, markersize=6)
    ax_peso.set_yticks(vs_control["floor"])
    ax_peso.set_yticklabels([f"nivel {int(f)}\n{b:.0f} mT" for f, b in zip(vs_control["floor"], vs_control["B_mT"])])
    ax_peso.invert_yaxis()
    # Aire arriba del primer piso para que la leyenda no tape el marcador.
    pisos = vs_control["floor"]
    ax_peso.set_ylim(pisos.max() + 0.6, pisos.min() - 1.1)
    ax_peso.set_xlabel("Diferencia de imbibicion a 72 h vs control (%)")
    ax_peso.set_title("Imbibición: efecto acotado dentro del margen")
    ax_peso.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax_peso.legend(loc="upper left", fontsize=9)

    if resultado_germ.disponible:
        ax_gm = axes[1]
        gm_df = resultado_germ.gm_df
        gm_plot = (gm_df[gm_df["con_iman"] == 1]
                   .groupby("treatment_floor")
                   .agg(media=("germinadas", "mean"), sd=("germinadas", "std"), n=("germinadas", "count"))
                   .reset_index())
        gm_plot["ee"] = gm_plot["sd"] / gm_plot["n"] ** 0.5
        gm_ctrl = gm_df[gm_df["con_iman"] == 0]["germinadas"]
        gm_ctrl_ee = gm_ctrl.std() / len(gm_ctrl) ** 0.5
        ax_gm.axvspan(gm_ctrl.mean() - 1.96 * gm_ctrl_ee, gm_ctrl.mean() + 1.96 * gm_ctrl_ee,
                      color="tab:red", alpha=0.12, label="control IC95")
        ax_gm.axvline(gm_ctrl.mean(), color="tab:red", linestyle="--", linewidth=1.2,
                      label=f"control ({gm_ctrl.mean():.1f}/20)")
        ax_gm.errorbar(gm_plot["media"], gm_plot["treatment_floor"],
                       xerr=1.96 * gm_plot["ee"], fmt="s", color="tab:blue", ecolor="steelblue",
                       elinewidth=1.6, capsize=4, markersize=6)
        ax_gm.set_xlabel("Semillas germinadas de 20 (media +- IC95)")
        ax_gm.set_title("Germinacion: senal sugestiva, sin potencia")
        ax_gm.grid(True, axis="x", linestyle="--", alpha=0.4)
        ax_gm.legend(loc="lower right", fontsize=9)

    n_total = len(resultado_peso.conf_df)
    n_control = (resultado_peso.conf_df["con_iman"] == 0).sum()
    n_tratadas = (resultado_peso.conf_df["con_iman"] == 1).sum()
    fig.suptitle(f"{titulo} | n = {n_total} cajas ({n_control} control / {n_tratadas} tratadas) "
                 f"| modelo ajustado por bloque", fontsize=10)
    fig.tight_layout()
    return fig


def fig_boxplot_germinacion(resultado_germ: ResultadoGerminacion, titulo: str,
                             control_floor: int = CONTROL_FLOOR) -> Figure | None:
    if not resultado_germ.disponible:
        return None
    import matplotlib.pyplot as plt

    box_df = resultado_germ.gm_df.copy()
    box_orden = [f for f in range(1, control_floor + 1) if f in set(box_df["treatment_floor"])]
    box_ctrl = box_df.loc[box_df["treatment_floor"] == control_floor, "germinadas"]
    box_ctrl_ee = box_ctrl.std() / np.sqrt(len(box_ctrl))
    box_colores = ["tab:red" if f == control_floor else "tab:blue" for f in box_orden]

    fig_box, ax_box = plt.subplots(figsize=(12, 6.5))

    ax_box.axhspan(box_ctrl.mean() - 1.96 * box_ctrl_ee, box_ctrl.mean() + 1.96 * box_ctrl_ee,
                   color="tab:red", alpha=0.10, zorder=0,
                   label=f"control IC95 [{box_ctrl.mean() - 1.96 * box_ctrl_ee:.1f}, {box_ctrl.mean() + 1.96 * box_ctrl_ee:.1f}]")
    ax_box.axhline(box_ctrl.mean(), color="tab:red", linestyle="--", linewidth=1.4, zorder=1,
                   label=f"media control = {box_ctrl.mean():.1f}/20")

    sns.boxplot(data=box_df, x="treatment_floor", y="germinadas", order=box_orden,
                palette=box_colores, width=0.62, showfliers=False, ax=ax_box,
                medianprops={"color": "black", "linewidth": 2},
                boxprops={"alpha": 0.45, "edgecolor": "k"},
                whiskerprops={"color": "k", "linewidth": 1.3},
                capprops={"color": "k", "linewidth": 1.3})
    sns.stripplot(data=box_df, x="treatment_floor", y="germinadas", order=box_orden,
                  color="black", size=5, jitter=0.18, alpha=0.75, ax=ax_box, zorder=5)

    box_medias = box_df.groupby("treatment_floor")["germinadas"].mean().reindex(box_orden)
    ax_box.scatter(range(len(box_orden)), box_medias.values, marker="D", s=85,
                   facecolors="white", edgecolors="black", linewidths=1.6, zorder=10,
                   label="media del nivel")

    contrastes = resultado_germ.vs_control.set_index("nivel")

    box_top = box_df["germinadas"].max()
    for i, f in enumerate(box_orden):
        if f == control_floor:
            ax_box.text(i, box_top + 1.4, "CONTROL", ha="center", fontsize=9, color="tab:red", fontweight="bold")
            continue
        p = contrastes.loc[f, "p"]
        p_holm = contrastes.loc[f, "p_holm"]
        ax_box.text(i, box_top + 1.1, f"p={p:.2f}\nHolm {p_holm:.2f}",
                    ha="center", fontsize=8, color="black" if p < 0.05 else "gray")

    ax_box.set_xticklabels([f"{f}\n{'sin campo' if f == control_floor else format(box_df.loc[box_df.treatment_floor == f, 'B_compart_mT'].iloc[0], '.0f') + ' mT'}"
                            for f in box_orden])
    ax_box.set_xlabel("Nivel de la torre (campo en el centro del compartimento)")
    ax_box.set_ylabel("Semillas germinadas de 20")
    ax_box.set_ylim(top=box_top + 3.2)
    ax_box.set_title(f"Germinación por nivel vs control | {titulo} | n=9 cajas por nivel")
    ax_box.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_box.legend(loc="lower right", fontsize=9, ncol=2)
    fig_box.tight_layout()
    return fig_box


def fig_boxplot_radicula(resultado_rad: ResultadoRadicula, titulo: str,
                          control_floor: int = CONTROL_FLOOR) -> Figure | None:
    if not resultado_rad.disponible:
        return None
    import matplotlib.pyplot as plt

    # Se grafica el desvio respecto del bloque, no el mm absoluto: cada fecha
    # midio en su propia escala (el pipeline de fotos no es comparable entre
    # fechas), asi que juntar los mm crudos haria que el alto de cada caja sea
    # la distancia entre fechas y no el efecto del tratamiento. Centrar por
    # bloque es exactamente lo que hace el termino C(bloque) del modelo, asi
    # que la figura muestra lo mismo que testean los p-valores anotados.
    rl_df = resultado_rad.rl_df.copy()
    rl_df["centrado"] = rl_df["mediana_mm"] - rl_df.groupby("bloque")["mediana_mm"].transform("mean")
    vs_control = resultado_rad.vs_control

    rl_orden = [f for f in range(1, control_floor + 1) if f in set(rl_df["treatment_floor"])]
    rl_ctrl = rl_df.loc[rl_df["treatment_floor"] == control_floor, "centrado"]
    rl_ctrl_ee = rl_ctrl.std() / np.sqrt(len(rl_ctrl))
    rl_colores = ["tab:red" if f == control_floor else "tab:green" for f in rl_orden]

    fig_rl, ax_rl = plt.subplots(figsize=(13, 6.5))
    ax_rl.axhline(0, color="gray", linestyle="-", linewidth=0.9, zorder=0)
    ax_rl.axhspan(rl_ctrl.mean() - 1.96 * rl_ctrl_ee, rl_ctrl.mean() + 1.96 * rl_ctrl_ee,
                  color="tab:red", alpha=0.10, zorder=0,
                  label=f"control IC95 [{rl_ctrl.mean() - 1.96 * rl_ctrl_ee:+.1f}, {rl_ctrl.mean() + 1.96 * rl_ctrl_ee:+.1f}] mm")
    ax_rl.axhline(rl_ctrl.mean(), color="tab:red", linestyle="--", linewidth=1.4, zorder=1,
                  label=f"media control = {rl_ctrl.mean():+.1f} mm vs su bloque")
    sns.boxplot(data=rl_df, x="treatment_floor", y="centrado", order=rl_orden,
                palette=rl_colores, width=0.62, showfliers=False, ax=ax_rl,
                medianprops={"color": "black", "linewidth": 2},
                boxprops={"alpha": 0.45, "edgecolor": "k"},
                whiskerprops={"color": "k", "linewidth": 1.3},
                capprops={"color": "k", "linewidth": 1.3})
    sns.stripplot(data=rl_df, x="treatment_floor", y="centrado", order=rl_orden, hue="fecha",
                  palette="dark", size=5, jitter=0.18, alpha=0.8, ax=ax_rl, zorder=5)
    rl_medias = rl_df.groupby("treatment_floor")["centrado"].mean().reindex(rl_orden)
    ax_rl.scatter(range(len(rl_orden)), rl_medias.values, marker="D", s=85,
                  facecolors="white", edgecolors="black", linewidths=1.6, zorder=10,
                  label="media del nivel")

    p_map = vs_control.set_index("floor")[["p", "p_holm"]]
    rl_top = rl_df["centrado"].max()
    rl_rango = rl_top - rl_df["centrado"].min()
    for i, f in enumerate(rl_orden):
        if f == control_floor:
            ax_rl.text(i, rl_top + 0.10 * rl_rango, "CONTROL", ha="center", fontsize=9, color="tab:red", fontweight="bold")
            continue
        p, ph = p_map.loc[f]
        ax_rl.text(i, rl_top + 0.05 * rl_rango, f"p={p:.2f}\nHolm {ph:.2f}",
                   ha="center", fontsize=8, color="black" if p < 0.05 else "gray")

    ax_rl.set_xticklabels([f"{f}\n{'sin campo' if f == control_floor else format(rl_df.loc[rl_df.treatment_floor == f, 'B_compart_mT'].iloc[0], '.0f') + ' mT'}"
                           for f in rl_orden])
    ax_rl.set_xlabel("Nivel de la torre (campo en el centro del compartimento)")
    ax_rl.set_ylabel("Largo de radicula: desvio respecto de su bloque (mm)")
    ax_rl.set_ylim(top=rl_top + 0.22 * rl_rango)
    fechas_usadas = ", ".join(sorted(rl_df["fecha"].unique()))
    ax_rl.set_title(
        f"Largo de radícula por nivel vs control | {titulo} ({fechas_usadas})\n"
        f"Centrado por bloque (fecha x piso fisico); en escala absoluta la media del control "
        f"es {resultado_rad.media_control:.1f} mm | p ajustado por bloque"
    )
    ax_rl.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_rl.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig_rl.tight_layout()
    return fig_rl


# =============================================================================
# Orquestacion por grupo
# =============================================================================


def guardar_figura(fig: Figure, destino: Path) -> None:
    """PNG para mirar y PDF para LaTeX. El PDF es vectorial y, de paso, es el
    unico de los dos que queda versionado: .gitignore bloquea *.png."""
    fig.savefig(destino.with_suffix(".png"), dpi=200)
    fig.savefig(destino.with_suffix(".pdf"))


def run_grupo(nombre: str, fechas: list[str], titulo: str, dosis_csv: Path, out_dir: Path) -> ResultadoGrupo:
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_df = cargar_datos_peso(fechas, dosis_csv)
    resultado_peso = analizar_peso(conf_df)
    resultado_germ = analizar_germinacion(conf_df)

    rl_df = cargar_datos_radicula(fechas, dosis_csv)
    resultado_rad = analizar_radicula(rl_df)

    fig_forest = fig_forest_peso_germinacion(resultado_peso, resultado_germ, titulo)
    fig_germ_box = fig_boxplot_germinacion(resultado_germ, titulo)
    fig_rad_box = fig_boxplot_radicula(resultado_rad, titulo)

    resultado_peso.vs_control.to_csv(out_dir / "peso_vs_control.csv", index=False)
    guardar_figura(fig_forest, out_dir / "fig_forest_peso_germinacion")

    if resultado_germ.disponible:
        resultado_germ.gm_df.groupby("treatment_floor").agg(
            B_mT=("B_compart_mT", "first"), media=("germinadas", "mean"),
            sd=("germinadas", "std"), n=("germinadas", "count"),
        ).round(3).to_csv(out_dir / "germinacion_por_piso.csv")
        resultado_germ.vs_control.to_csv(out_dir / "germinacion_vs_control.csv", index=False)
        guardar_figura(fig_germ_box, out_dir / "fig_germinacion_boxplot")

    if resultado_rad.disponible:
        resultado_rad.vs_control.to_csv(out_dir / "radicula_vs_control.csv", index=False)
        guardar_figura(fig_rad_box, out_dir / "fig_radicula_boxplot")

    reporte = "\n\n".join(filter(None, [
        f"GRUPO: {titulo} ({nombre})\nFechas: {', '.join(fechas)}\n",
        resultado_peso.diagnostico,
        resultado_germ.diagnostico,
        resultado_rad.diagnostico,
    ]))
    (out_dir / "reporte.txt").write_text(reporte, encoding="utf-8")

    figuras = {"forest_peso_germinacion": fig_forest}
    if fig_germ_box is not None:
        figuras["germinacion_boxplot"] = fig_germ_box
    if fig_rad_box is not None:
        figuras["radicula_boxplot"] = fig_rad_box

    return ResultadoGrupo(nombre=nombre, titulo=titulo, fechas=fechas, peso=resultado_peso,
                           germinacion=resultado_germ, radicula=resultado_rad, figuras=figuras)


def comparativa_dosis_respuesta(resultados: list[ResultadoGrupo], out_dir: Path) -> Figure | None:
    """Imbibicion vs intensidad de campo, juntando los grupos corridos.
    Reemplaza a las celdas 7-8 del notebook, que apuntaban a fechas descartadas."""
    if len(resultados) < 2:
        return None
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat(
        [r.peso.conf_df.assign(iman=r.titulo) for r in resultados],
        ignore_index=True,
    )
    # El control (piso 9) no tiene fila en la tabla de dosis porque no recibe
    # campo aplicado: se le asigna B = 0 ANTES de descartar nulos, si no el
    # dropna se lleva puesto justamente al grupo de referencia.
    df.loc[df["treatment_floor"] == CONTROL_FLOOR, "B_compart_mT"] = 0.0
    df = df.dropna(subset=["B_compart_mT"])
    b_order = sorted(df["B_compart_mT"].unique())

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(data=df, x="B_compart_mT", y="pct_change", hue="iman", order=b_order,
                width=0.6, ax=ax, medianprops={"color": "red", "linewidth": 2},
                whiskerprops={"color": "k", "linewidth": 1.5},
                capprops={"color": "k", "linewidth": 1.5},
                boxprops={"alpha": 0.8, "edgecolor": "k"})
    sns.stripplot(data=df, x="B_compart_mT", y="pct_change", hue="iman", order=b_order,
                  dodge=True, alpha=0.35, jitter=True, legend=False, ax=ax)
    # Una linea por iman: los controles de cada grupo estan en niveles muy
    # distintos (distinta tanda de semillas), promediarlos no significa nada.
    for color, (iman, g) in zip(["tab:blue", "tab:orange", "tab:green"], df.groupby("iman")):
        media = g.loc[g["treatment_floor"] == CONTROL_FLOOR, "pct_change"].mean()
        ax.axhline(media, color=color, linestyle="--", linewidth=1.5,
                   label=f"Control {iman} = {media:.3f}")
    ax.set_xlabel("|B| en el centro del compartimento (mT)")
    ax.set_ylabel("Imbibicion a 72 h ((post - pre) / pre)")
    ax.set_title("Imbibicion vs intensidad de campo aplicada")
    ax.set_xticks(range(len(b_order)))
    ax.set_xticklabels([f"{b:g}" for b in b_order])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Iman")
    fig.tight_layout()
    guardar_figura(fig, out_dir / "fig_dosis_respuesta_combinada")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("Grupos incluidos:")
        for r in resultados:
            print(f"  {r.titulo}: {', '.join(r.fechas)}")
        print("\nModelo: pct_change ~ B_compart_mT + C(iman)")
        print(smf.ols("pct_change ~ B_compart_mT + C(iman)", data=df).fit().summary())
        print("\nModelo con interaccion: pct_change ~ B_compart_mT * C(iman)")
        print(smf.ols("pct_change ~ B_compart_mT * C(iman)", data=df).fit().summary())
    (out_dir / "reporte_dosis_respuesta.txt").write_text(buf.getvalue(), encoding="utf-8")
    return fig


def run_control_positivo(fechas: list[str], out_dir: Path) -> Figure:
    """Control positivo: las 3 fechas usan la misma dosis de agua (2/5/6 mL).
    Sirve para mostrar que el metodo SI detecta un efecto real cuando existe,
    que es lo que le da sentido al resultado nulo de los imanes."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for fecha in fechas:
        trays = pd.read_csv(EXPERIMENTS_DIR / fecha / "trays.csv")
        trays.columns = trays.columns.str.strip()
        # El mL de cada nivel vive en treatment.csv (603 escribe "ml" y los
        # otros "mL", de ahi el lower()).
        trt = pd.read_csv(EXPERIMENTS_DIR / fecha / "treatment.csv")
        trt["treatment_type"] = trt["treatment_type"].str.lower()
        assert (trt["treatment_type"] == "ml").all(), f"{fecha}: treatment.csv no esta en mL"
        mapa = dict(zip(trt["treatment_id"], trt["treatment_value"]))
        trays["water_ml"] = trays["treatment_floor"].map(mapa)
        trays["fecha"] = fecha
        dfs.append(trays)

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["pre_weight", "post_weight", "water_ml"]).copy()
    df["pct_change"] = (df["post_weight"] - df["pre_weight"]) / df["pre_weight"]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("=" * 72)
        print(f"CONTROL POSITIVO: imbibicion vs agua aplicada (n = {len(df)} bandejas)")
        print("=" * 72)
        print(f"Fechas: {', '.join(fechas)}")
        print("\nBandejas por fecha y nivel de agua:")
        print(pd.crosstab(df["fecha"], df["water_ml"]))
        print("\nResumen por nivel de agua:")
        print(df.groupby("water_ml")["pct_change"]
              .agg(["count", "mean", "std", "min", "max"]).round(4).to_string())

        # La fecha entra como bloque: distinta tanda de semillas por experimento.
        modelo = smf.ols("pct_change ~ water_ml + C(fecha)", data=df).fit()
        lo, hi = modelo.conf_int().loc["water_ml"]
        print(f"\nModelo: pct_change ~ water_ml + C(fecha)   (fecha = bloque)")
        print(f"Pendiente: {modelo.params['water_ml']:+.5f} por mL "
              f"IC95 [{lo:+.5f}, {hi:+.5f}] p = {modelo.pvalues['water_ml']:.2e}")
        print(f"R2 ajustado = {modelo.rsquared_adj:.3f}")

        print("\nTest global del nivel de agua (como factor):")
        m_factor = smf.ols("pct_change ~ C(water_ml) + C(fecha)", data=df).fit()
        print(anova_lm(m_factor, typ=2).round(5))

        niveles = sorted(df["water_ml"].unique())
        print(f"\nContrastes entre niveles (Welch, Holm):")
        pares, ps = [], []
        for i, a in enumerate(niveles):
            for b in niveles[i + 1:]:
                va = df.loc[df.water_ml == a, "pct_change"]
                vb = df.loc[df.water_ml == b, "pct_change"]
                pares.append((a, b, vb.mean() - va.mean()))
                ps.append(stats.ttest_ind(va, vb, equal_var=False).pvalue)
        p_holm = multipletests(ps, method="holm")[1]
        for (a, b, dif), p, ph in zip(pares, ps, p_holm):
            print(f"  {a} mL vs {b} mL: {dif:+.4f}  p = {p:.4g}  Holm = {ph:.4g}")

    (out_dir / "reporte_control_positivo.txt").write_text(buf.getvalue(), encoding="utf-8")

    resumen = df.groupby("water_ml")["pct_change"].agg(["count", "mean", "std"]).round(5)
    resumen.to_csv(out_dir / "control_positivo_por_nivel.csv")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=df, x="water_ml", y="pct_change", order=niveles, width=0.55, ax=ax,
                showfliers=False, medianprops={"color": "black", "linewidth": 2},
                boxprops={"alpha": 0.5, "edgecolor": "k"},
                whiskerprops={"color": "k", "linewidth": 1.3},
                capprops={"color": "k", "linewidth": 1.3})
    sns.stripplot(data=df, x="water_ml", y="pct_change", order=niveles, hue="fecha",
                  dodge=False, size=6, jitter=0.18, alpha=0.85, ax=ax, zorder=5)
    medias = df.groupby("water_ml")["pct_change"].mean().reindex(niveles)
    ax.scatter(range(len(niveles)), medias.values, marker="D", s=85, facecolors="white",
               edgecolors="black", linewidths=1.6, zorder=10, label="media del nivel")
    ax.set_xlabel("Agua aplicada (mL)")
    ax.set_ylabel("Imbibicion a 72 h ((post - pre) / pre)")
    ax.set_title(f"Control positivo: imbibición vs agua | n = {len(df)} cajas\n"
                 f"pendiente {modelo.params['water_ml']:+.4f}/mL, p = {modelo.pvalues['water_ml']:.1e}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Fecha", fontsize=9)
    fig.tight_layout()
    guardar_figura(fig, out_dir / "fig_control_positivo")
    return fig


def agrupar_por_tipo(
    fechas: list[str], inventario: list[EstadoExperimento]
) -> dict[tuple[str, str, str], list[str]]:
    """El agua entra en la clave de grupo: las dos campanas de neodimio N se
    corrieron con volumenes distintos (5 mL saturacion, 3 mL estres hidrico), y
    agruparlas juntas mezclaria dos condiciones experimentales en un solo
    analisis. Son grupos separados, con directorios de salida separados."""
    por_exp_id = {e.exp_id: e for e in inventario}
    grupos: dict[tuple[str, str, str], list[str]] = {}
    for fecha in fechas:
        e = por_exp_id[fecha]
        key = (e.experiment_type, e.magnet_pole, e.water_ml)
        grupos.setdefault(key, []).append(fecha)
    return grupos


def main() -> None:
    import argparse
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Analisis confirmatorio por grupo (tipo de iman, polo, agua)."
    )
    parser.add_argument(
        "--fechas",
        nargs="+",
        metavar="EXP_ID",
        help="Fechas a analizar (ej. --fechas 20260626 20260721 20260727). "
             "Sin este flag se abre la seleccion interactiva.",
    )
    args = parser.parse_args()

    inventario = construir_inventario()

    if args.fechas:
        conocidas = {e.exp_id for e in inventario}
        desconocidas = [f for f in args.fechas if f not in conocidas]
        if desconocidas:
            parser.error(
                "No estan en experiments_metadata.csv: " + ", ".join(desconocidas)
            )
        fechas_elegidas = list(args.fechas)
    else:
        fechas_elegidas = seleccionar_fechas(inventario)

    if not fechas_elegidas:
        print("No se eligio ninguna fecha. Nada que hacer.")
        return

    grupos = agrupar_por_tipo(fechas_elegidas, inventario)
    resultados = []
    for (tipo, polo, agua), fechas in grupos.items():
        if tipo == "CTRL_POSITIVO":
            out_dir = ANALYSIS_DIR / "control_positivo_combinado"
            print(f"\n>>> Control positivo ({', '.join(sorted(fechas))}) -> {out_dir}")
            run_control_positivo(sorted(fechas), out_dir)
            continue

        clave_dosis = DOSIS_CSV_MAP.get((tipo, polo))
        if clave_dosis is None:
            print(f"[OMITIDO] {tipo}/{polo} ({', '.join(fechas)}): no hay tabla de dosis mapeada en DOSIS_CSV_MAP.")
            continue

        if not agua:
            print(f"[OMITIDO] {tipo}/{polo} ({', '.join(fechas)}): sin water_ml en "
                  f"experiments_metadata.csv. El agua define el grupo, asi que "
                  f"hay que completarla antes de analizar.")
            continue

        sufijo_agua = f"_{agua}ml"
        nombre = f"{tipo.replace('IMAN_', '').lower()}_{polo.lower()}{sufijo_agua}"
        titulo = f"{tipo.replace('IMAN_', '').title()} {polo} ({agua} mL)"
        out_dir = ANALYSIS_DIR / f"confirmatorio_{nombre}"
        print(f"\n>>> Corriendo grupo {titulo} ({', '.join(sorted(fechas))}) -> {out_dir}")
        resultado = run_grupo(nombre, sorted(fechas), titulo, EXTRA_DIR / clave_dosis, out_dir)
        resultados.append(resultado)
        print(f"    guardado: {list(resultado.figuras.keys())} + CSVs + reporte.txt en {out_dir}")

    if comparativa_dosis_respuesta(resultados, ANALYSIS_DIR / "comparativa_imanes") is not None:
        print(f"\n>>> Comparativa dosis-respuesta -> {ANALYSIS_DIR / 'comparativa_imanes'}")


if __name__ == "__main__":
    main()
