"""Linea de base por fecha y comparacion entre regimenes de agua.

Produce los dos numeros que el documento cita y que no salian de ningun
reporte del analisis confirmatorio:

  1. La linea de base de cada experimento (imbibicion de las muestras sin
     campo), que muestra que a igual volumen de agua no hay deriva temporal
     sino variabilidad entre fechas.

  2. El test de interaccion entre la exposicion al campo y el regimen
     hidrico: responde si el efecto del iman difiere entre 5 y 3 mL, que es
     la hipotesis que motivo ampliar el protocolo.

Uso:  python src/analisis_linea_base.py
Salida: data/analysis/linea_base_y_regimenes/reporte.txt
"""

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
METADATA_CSV = PROJECT_ROOT / "data" / "experiments_metadata.csv"
OUT_DIR = PROJECT_ROOT / "data" / "analysis" / "linea_base_y_regimenes"

CONTROL_FLOOR = 9


def cargar(exp_ids: list[str]) -> pd.DataFrame:
    dfs = []
    for exp_id in exp_ids:
        path = EXPERIMENTS_DIR / exp_id / "trays.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        d.columns = d.columns.str.strip()
        d = d.dropna(subset=["pre_weight", "post_weight"]).copy()
        d["fecha"] = exp_id
        d["pct"] = (d["post_weight"] - d["pre_weight"]) / d["pre_weight"]
        d["bloque"] = d["fecha"] + "-" + d["floor"].astype(str)
        d["con_iman"] = (d["treatment_floor"] != CONTROL_FLOOR).astype(int)
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def tabla_linea_base(meta: pd.DataFrame, lineas: list[str]) -> None:
    lineas.append("=" * 72)
    lineas.append("LINEA DE BASE POR EXPERIMENTO")
    lineas.append("=" * 72)
    lineas.append("Imbibicion de las muestras SIN campo (nivel 9) y de todas las muestras.")
    lineas.append("")
    lineas.append(f"{'fecha':<10} {'tipo':<16} {'agua':>5} {'control':>8} {'todas':>8} "
                  f"{'pre (g)':>8} {'post (g)':>9} {'CV pre':>7}")

    df = cargar(meta["exp_id"].tolist())
    for _, row in meta.iterrows():
        sub = df[df.fecha == row.exp_id]
        if sub.empty:
            continue
        ctrl = sub.loc[sub.treatment_floor == CONTROL_FLOOR, "pct"]
        # en los controles negativos no hay tratamiento: todas las muestras son control
        base = ctrl.mean() if len(ctrl) else sub["pct"].mean()
        agua = "" if pd.isna(row.water_ml) else f"{row.water_ml:g}"
        cv = sub["pre_weight"].std() / sub["pre_weight"].mean() * 100
        lineas.append(f"{row.exp_id:<10} {row.experiment_type.replace('IMAN_','').replace('CTRL_','ctrl '):<16} "
                      f"{agua:>5} {base:8.3f} {sub['pct'].mean():8.3f} "
                      f"{sub['pre_weight'].mean():8.3f} {sub['post_weight'].mean():9.3f} {cv:6.1f}%")

    lineas.append("")
    lineas.append("Lectura: a volumen de agua constante la linea de base varia entre fechas")
    lineas.append("sin direccion definida. No es una deriva del sistema. El CV del peso seco")
    lineas.append("delata el dosificador: cae al incorporarse el medidor de celdas.")
    lineas.append("")


def interaccion_regimenes(meta: pd.DataFrame, lineas: list[str]) -> None:
    lineas.append("=" * 72)
    lineas.append("INTERACCION: EL EFECTO DEL IMAN DEPENDE DEL REGIMEN DE AGUA?")
    lineas.append("=" * 72)

    iman = meta[(meta.experiment_type == "IMAN_NEODIMIO") & (meta.magnet_pole == "N")]
    iman = iman.dropna(subset=["water_ml"])
    if iman.water_ml.nunique() < 2:
        lineas.append("Se necesitan dos regimenes de agua del mismo iman y polo. No disponible.")
        return

    df = cargar(iman.exp_id.tolist())
    agua_por_fecha = dict(zip(iman.exp_id, iman.water_ml))
    df["agua"] = df.fecha.map(agua_por_fecha).map(lambda x: f"{x:g}")

    lineas.append(f"Mismo iman (neodimio N) en dos regimenes. n = {len(df)} muestras.")
    lineas.append("")
    for agua, sub in sorted(df.groupby("agua"), key=lambda kv: -float(kv[0])):
        m = smf.ols("pct ~ C(bloque) + con_iman", data=sub).fit()
        ctrl = sub.loc[sub.con_iman == 0, "pct"].mean()
        e, se = m.params["con_iman"], m.bse["con_iman"]
        fechas = ", ".join(sorted(sub.fecha.unique()))
        lineas.append(f"  {agua} mL ({fechas}):")
        lineas.append(f"     control = {ctrl:.4f} | efecto {e / ctrl * 100:+.2f}% "
                      f"(SE {se / ctrl * 100:.2f}%)  p = {m.pvalues['con_iman']:.4f}")

    modelo = smf.ols("pct ~ C(bloque) + con_iman*C(agua)", data=df).fit()
    termino = next(t for t in modelo.params.index if t.startswith("con_iman:"))
    ic = modelo.conf_int().loc[termino]
    lineas.append("")
    lineas.append(f"  Modelo: pct ~ C(bloque) + con_iman*C(agua)")
    lineas.append(f"  Termino de interaccion ({termino}):")
    lineas.append(f"     coef = {modelo.params[termino]:+.5f}  SE = {modelo.bse[termino]:.5f}  "
                  f"t = {modelo.tvalues[termino]:+.3f}  p = {modelo.pvalues[termino]:.4f}")
    lineas.append(f"     IC95 absoluto = [{ic[0]:+.5f}, {ic[1]:+.5f}]")
    lineas.append("")
    veredicto = ("NO hay evidencia de que el regimen de agua modifique el efecto del campo"
                 if modelo.pvalues[termino] >= 0.05 else
                 "El efecto del campo DIFIERE entre regimenes")
    lineas.append(f"  -> {veredicto}")
    lineas.append("")
    lineas.append("  Limite: los dos regimenes difieren tambien en el dosificador de semillas")
    lineas.append("  (ver seccion de regimen hidrico en el documento), asi que el resultado no")
    lineas.append("  se atribuye al agua de manera aislada.")
    lineas.append("")


def main() -> None:
    meta = pd.read_csv(METADATA_CSV, dtype={"exp_id": str})
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lineas: list[str] = []
    tabla_linea_base(meta, lineas)
    interaccion_regimenes(meta, lineas)

    reporte = "\n".join(lineas)
    (OUT_DIR / "reporte.txt").write_text(reporte, encoding="utf-8")
    print(reporte)
    print(f"-> {OUT_DIR / 'reporte.txt'}")


if __name__ == "__main__":
    main()
