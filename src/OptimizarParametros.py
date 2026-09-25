"""
Busca los parámetros de Analizar.py (parametros_calibracion.json) que mejor
reproducen las máscaras pintadas a mano con EtiquetarMascara.py.

Uso:
    python src/EtiquetarMascara.py data/experiments/20260911/raw/photos/IMG_3887.jpg
    python src/OptimizarParametros.py 20260911

Qué hace:
  1. Busca <experimento>/raw/labels/*_mask.png y las empareja con su foto en
     raw/photos/ por nombre de archivo.
  2. Para cada par (foto, máscara pintada) precalcula UNA sola vez la
     referencia de escala (regla blanca / QR / etiqueta / regla amarilla):
     no depende de los parámetros que se están buscando, así que recalcularla
     en cada evaluación sería tiempo tirado.
  3. Corre scipy.optimize.differential_evolution en paralelo (todos los
     núcleos por defecto) minimizando una pérdida = Dice de raíz + Dice de
     semilla + penalización por diferencia en la cantidad de raíces, contra
     lo pintado.
  4. Guarda el resultado en <experimento>/processed/parametros_calibracion.json
     (el mismo archivo que ya usa el pipeline normal para ese experimento) y
     deja un testigo visual optim_<foto>.png para revisar a ojo.

No hace falta re-pintar nada para volver a correrlo: podés agregar más fotos
etiquetadas después y volver a ejecutar este script, usa todas las que
encuentre.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import differential_evolution

import Analizar
from EtiquetarMascara import COLOR_RAIZ, COLOR_SEMILLA, cargar_imagen_base

# (nombre_parametro, minimo, maximo, tipo)
DIMENSIONES = [
    ("blur_k", 1, 9, "int"),
    ("clean_k", 0, 5, "int"),
    ("seed_glue", 1, 21, "int"),
    ("seed_h_max", 15, 45, "int"),
    ("seed_s_min", 30, 150, "int"),
    ("seed_v_min", 10, 150, "int"),
    ("root_s_max", 30, 120, "int"),
    ("root_v_min", 100, 220, "int"),
    ("min_area_seed", 15, 200, "int"),
    ("root_bridge", 1, 15, "int"),
    ("min_area_root", 3, 80, "int"),
    ("root_solidity_max", 0.3, 0.9, "float"),
    ("root_aspect_min", 0.03, 0.4, "float"),
    ("root_aspect_max", 3.0, 15.0, "float"),
]

PESO_DICE_RAIZ = 0.65
PESO_DICE_SEMILLA = 0.15
PESO_CONTEO = 0.20


def dice(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    inter = np.logical_and(pred_bool, gt_bool).sum()
    total = int(pred_bool.sum()) + int(gt_bool.sum())
    if total == 0:
        return 1.0
    return 2.0 * float(inter) / float(total)


def construir_params(x, base: dict) -> dict:
    params = dict(base)
    for valor, (nombre, lo, hi, tipo) in zip(x, DIMENSIONES):
        v = min(max(valor, lo), hi)
        params[nombre] = int(round(v)) if tipo == "int" else float(v)
    return params


def encontrar_pares_etiquetados(exp_dir: Path, filtro_fotos):
    labels_dir = exp_dir / "raw" / "labels"
    fotos_dir = exp_dir / "raw" / "photos"
    if not labels_dir.exists():
        return []

    extensiones = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG", ".heic", ".HEIC"]
    pares = []
    for archivo_mask in sorted(labels_dir.glob("*_mask.png")):
        stem = archivo_mask.stem[: -len("_mask")]
        if filtro_fotos and stem not in filtro_fotos:
            continue
        foto = None
        for ext in extensiones:
            candidata = fotos_dir / f"{stem}{ext}"
            if candidata.exists():
                foto = candidata
                break
        if foto is None:
            print(f"[ALERTA] {archivo_mask.name}: no encontré la foto '{stem}.*' en {fotos_dir}")
            continue
        pares.append((foto, archivo_mask))
    return pares


def preparar_contexto(pares):
    contexto = []
    for foto, archivo_mask in pares:
        img = cargar_imagen_base(foto)
        mask = cv2.imread(str(archivo_mask), cv2.IMREAD_COLOR)
        if mask is None or mask.shape != img.shape:
            print(
                f"[ALERTA] {archivo_mask.name} no coincide en tamaño con {foto.name}, "
                "se ignora este par (¿la foto cambió después de pintar?)."
            )
            continue

        gt_root = np.all(mask == np.array(COLOR_RAIZ), axis=2)
        gt_seed = np.all(mask == np.array(COLOR_SEMILLA), axis=2)
        n_gt_root = max(cv2.connectedComponents(gt_root.astype(np.uint8))[0] - 1, 0)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        px_por_mm, ref_ok, ref_origen, rect_regla = Analizar.detectar_referencia(
            hsv, None, Analizar.PARAMS, img_original=img
        )
        if not ref_ok:
            print(
                f"[ALERTA] No se encontró referencia de escala en {foto.name}: "
                "los mm de esta foto no van a ser confiables en el puntaje."
            )

        contexto.append(
            {
                "nombre": foto.name,
                "img": img,
                "gt_root": gt_root,
                "gt_seed": gt_seed,
                "n_gt_root": n_gt_root,
                "referencia_fija": (px_por_mm, ref_ok, ref_origen, rect_regla),
            }
        )
        print(
            f"  {foto.name}: raiz={int(gt_root.sum())}px ({n_gt_root} trazos) "
            f"semilla={int(gt_seed.sum())}px  ref={ref_origen}"
        )
    return contexto


def costo_individual(params, item) -> float:
    res = Analizar.procesar_imagen(
        item["img"],
        params,
        generar_debug=False,
        recortar=False,
        devolver_mascaras=True,
        referencia_fija=item["referencia_fija"],
    )
    pred_root = res["mask_radiculas_pred"] > 0
    pred_seed = res["mask_semillas_pred"] > 0

    dice_root = dice(pred_root, item["gt_root"])
    dice_seed = dice(pred_seed, item["gt_seed"])

    n_gt = max(item["n_gt_root"], 1)
    n_pred = len(res["lista_radiculas"])
    penal_conteo = abs(n_pred - item["n_gt_root"]) / n_gt

    perdida = (
        PESO_DICE_RAIZ * (1.0 - dice_root)
        + PESO_DICE_SEMILLA * (1.0 - dice_seed)
        + PESO_CONTEO * penal_conteo
    )
    return perdida, dice_root, dice_seed, n_pred


def _objetivo(x, contexto, base):
    params = construir_params(x, base)
    total = 0.0
    for item in contexto:
        perdida, _, _, _ = costo_individual(params, item)
        total += perdida
    return total / len(contexto)


def reportar(titulo, params, contexto):
    print(f"\n{titulo}")
    total = 0.0
    for item in contexto:
        perdida, dice_root, dice_seed, n_pred = costo_individual(params, item)
        total += perdida
        print(
            f"  {item['nombre']:<24} dice_raiz={dice_root:.3f}  "
            f"dice_semilla={dice_seed:.3f}  raices={n_pred} (pintadas: {item['n_gt_root']})"
            f"  perdida={perdida:.3f}"
        )
    print(f"  -> perdida promedio: {total / len(contexto):.4f}")
    return total / len(contexto)


def guardar_testigos(exp_dir: Path, params, contexto):
    carpeta = exp_dir / "processed" / "procesadas_debug"
    carpeta.mkdir(parents=True, exist_ok=True)
    for item in contexto:
        res = Analizar.procesar_imagen(
            item["img"],
            params,
            generar_debug=True,
            recortar=False,
            referencia_fija=item["referencia_fija"],
        )
        stem = Path(item["nombre"]).stem
        ruta = carpeta / f"optim_{stem}.png"
        cv2.imwrite(str(ruta), res["img_debug"])
        print(f"  Testigo guardado: {ruta}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exp_id", help="Carpeta del experimento, ej: 20260911")
    parser.add_argument(
        "--foto",
        action="append",
        default=None,
        help="Limitar a esta foto (nombre sin extensión). Puede repetirse. Por defecto usa todas las etiquetadas.",
    )
    parser.add_argument("--popsize", type=int, default=10, help="Tamaño de población por dimensión (default 10)")
    parser.add_argument("--maxiter", type=int, default=20, help="Generaciones máximas (default 20)")
    parser.add_argument("--workers", type=int, default=-1, help="Procesos en paralelo, -1 = todos los núcleos (default -1)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad (default 42)")
    args = parser.parse_args()

    exp_dir = Path(Analizar.DIRECTORIO_RAIZ) / args.exp_id
    if not exp_dir.exists():
        raise SystemExit(f"[ERROR] No existe el experimento: {exp_dir}")

    filtro_fotos = set(args.foto) if args.foto else None
    pares = encontrar_pares_etiquetados(exp_dir, filtro_fotos)
    if not pares:
        raise SystemExit(
            f"[ERROR] No hay máscaras pintadas en {exp_dir / 'raw' / 'labels'}.\n"
            f"Corré primero: python src/EtiquetarMascara.py <ruta a una foto de {args.exp_id}>"
        )

    print(f"Encontré {len(pares)} foto(s) etiquetada(s) en {args.exp_id}:")
    contexto = preparar_contexto(pares)
    if not contexto:
        raise SystemExit("[ERROR] Ninguna máscara pudo usarse (ver alertas arriba).")

    # Config base: la que ya tenga el experimento (si corrió antes) o la global.
    archivo_config_exp = exp_dir / "processed" / "parametros_calibracion.json"
    Analizar.cargar_configuracion(
        str(archivo_config_exp) if archivo_config_exp.exists() else Analizar.ARCHIVO_CONFIG
    )
    base = dict(Analizar.PARAMS)

    perdida_antes = reportar("=== ANTES (config actual) ===", base, contexto)

    bounds = [(lo, hi) for _, lo, hi, _ in DIMENSIONES]
    x0 = [min(max(base[nombre], lo), hi) for nombre, lo, hi, _ in DIMENSIONES]

    def callback(xk, convergence):
        p = construir_params(xk, base)
        perdida = _objetivo(xk, contexto, base)
        print(f"    ... mejor pérdida parcial: {perdida:.4f}")

    print(
        f"\nBuscando parámetros (popsize={args.popsize}, maxiter={args.maxiter}, "
        f"workers={args.workers})..."
    )
    t0 = time.time()
    resultado = differential_evolution(
        _objetivo,
        bounds,
        args=(contexto, base),
        popsize=args.popsize,
        maxiter=args.maxiter,
        seed=args.seed,
        workers=args.workers,
        updating="deferred" if args.workers != 1 else "immediate",
        # polish=False: la mayoría de las dimensiones son enteras (se
        # redondean recién adentro de construir_params), así que la función
        # que ve differential_evolution es escalonada. El pulido final de
        # scipy asume un objetivo suave y usa gradiente — sobre una función
        # escalonada el gradiente es ~0 en casi todos lados, así que en el
        # mejor de los casos no hace nada y en el peor puede terminar
        # devolviendo un punto ligeramente distinto sin garantía de que sea
        # mejor. Con el chequeo de abajo (guardar solo si mejora de verdad)
        # esto ya no podría arruinar nada, pero mejor evitarlo directamente.
        polish=False,
        init="sobol",
        x0=x0,
        tol=1e-4,
        callback=callback,
    )
    elapsed = time.time() - t0
    print(f"\nListo en {elapsed:.1f}s ({resultado.nfev} evaluaciones).")

    mejores_params = construir_params(resultado.x, base)
    perdida_despues = reportar("=== DESPUÉS (optimizado) ===", mejores_params, contexto)

    # La búsqueda es estocástica (más todavía en paralelo: con workers != 1
    # scipy no garantiza el mismo camino corrida a corrida aunque la semilla
    # sea la misma). differential_evolution en teoría nunca debería devolver
    # algo peor que el punto de partida, pero NO lo demos por sentado y
    # sobreescribamos el archivo ciegamente: si por lo que sea el resultado
    # es peor que la config que ya había, la conservamos y avisamos, en vez
    # de pisar silenciosamente una config que ya andaba bien con una peor.
    TOLERANCIA = 1e-9
    if perdida_despues > perdida_antes + TOLERANCIA:
        print(
            f"\n[ALERTA] La búsqueda no mejoró la config actual "
            f"(antes={perdida_antes:.4f}, encontrado={perdida_despues:.4f}). "
            "Se mantiene la config existente sin cambios."
        )
        mejores_params = base
    else:
        archivo_config_exp.parent.mkdir(parents=True, exist_ok=True)
        with open(archivo_config_exp, "w") as f:
            json.dump(mejores_params, f, indent=4)
        print(f"\n[OK] Parámetros guardados en {archivo_config_exp}")

    print("\nGenerando testigos visuales...")
    guardar_testigos(exp_dir, mejores_params, contexto)

    print(
        f"\nListo. Corré Analizar.py sobre '{args.exp_id}' de nuevo (con --reprocesar "
        "si ya habías medido estas fotos) para remedir todo con estos parámetros."
    )


if __name__ == "__main__":
    main()
