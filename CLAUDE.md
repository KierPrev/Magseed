# Tesina de magnetobiología — mapa del proyecto

Tesina de Licenciatura en Física (UNSL, MBLab): ¿afecta un campo magnético
estático la germinación de semillas de maíz? Autora: Kiernan Preve.

El proyecto vive en `/Users/kier/MEGA/tesina/`, que **no es un repositorio
git**. Contiene dos repos independientes:

| Carpeta | Qué es |
|---|---|
| `tesis-magnetobiologia-software/` | Datos crudos, pipeline de análisis, generación de figuras |
| `tesis-magnetobiologia-latex/` | El documento de la tesis en LaTeX |

> Este archivo vive en `tesis-magnetobiologia-software/CLAUDE.md` y está
> enlazado desde el directorio padre, así que se carga trabajes desde donde
> trabajes. **Las rutas de abajo son relativas al padre** (`/Users/kier/MEGA/tesina/`).
> Lo recomendado es trabajar desde ahí, para tener los dos repos a mano.

## Antes de tocar nada: leé el inventario

**`tesis-magnetobiologia-software/INVENTARIO_RESULTADOS.md`** dice qué datos
existen, qué endpoints tiene cada grupo de experimentos y cuáles no (y por qué),
dónde está cada figura, cuáles están obsoletas y los números principales de cada
resultado.

No asumas que un endpoint existe para un grupo porque existe para otro: los
experimentos difieren en qué se midió. El inventario lo detalla.

## Los dos repos son independientes

Cada uno tiene su propio historial. **Hay que commitear en cada uno por
separado**, con `git -C` o entrando a la carpeta:

```bash
git -C tesis-magnetobiologia-latex    status
git -C tesis-magnetobiologia-software status
```

Un error fácil es commitear cambios del documento dentro del repo de software o
al revés. Verificá siempre en cuál estás parado antes de commitear.

## Las figuras: symlink, y siempre `.pdf`

El repo de LaTeX tiene un symlink al `data/` del repo de software:

```
tesis-magnetobiologia-latex/datos/software_data
    -> /Users/kier/MEGA/tesina/tesis-magnetobiologia-software/data
```

Por eso **no hay que copiar figuras**. Una figura generada en
`tesis-magnetobiologia-software/data/analysis/...` se referencia desde el `.tex`
como `datos/software_data/analysis/...` y el documento la toma en la siguiente
compilación.

**Usá siempre la versión `.pdf`, nunca la `.png`.** Cada figura se exporta en
ambos formatos. El PDF es vectorial y, sobre todo, es el único de los dos que
queda versionado: el `.gitignore` del repo de software bloquea `*.png`.

Ojo: el symlink es de **ruta absoluta**. Si se mueven las carpetas, se rompe.

## Regenerar una figura

```bash
cd tesis-magnetobiologia-software
.venv/bin/python src/analisis_confirmatorio.py
```

Abre una TUI que lista los experimentos con su tipo, polo y disponibilidad de
pesos / fotos / radícula / germinación. Se eligen fechas con espacio y se
confirma con enter. Escribe en `data/analysis/confirmatorio_<grupo>/`.

Usar el Python del venv (`.venv/bin/python`), no el `python3` del sistema: las
dependencias están solo en el venv.

El control negativo usa un pipeline anterior distinto y no se regenera con este
script.

## Compilar la tesis

```bash
cd tesis-magnetobiologia-latex
latexmk -xelatex tesis.tex     # -> tesis.pdf
latexmk -c                     # limpia auxiliares, conserva el PDF
```

**Pausá la sincronización de MEGA mientras compilás.** Los auxiliares se
escriben junto a las fuentes y MEGA sincronizando en el medio rompe la
compilación. Está explicado en `.latexmkrc`.

## Contexto experimental mínimo

Lo necesario para no escribir cualquier cosa; el detalle está en el inventario.

- **El piso 9 es SIEMPRE el control** (sin campo). Los pisos 1 a 8 reciben campo
  decreciente: el piso 1 es el más intenso.
- **La unidad experimental es la BANDEJA, nunca la semilla.** Analizar semillas
  sueltas sería pseudo-replicación (las 20 semillas de una bandeja comparten
  tratamiento).
- El diseño es de bloques completos aleatorizados, bloque = `fecha × floor`.
  Todos los modelos llevan `C(bloque)`.
- `pct_change` es **imbibición** (absorción de agua a 72 h), **no biomasa**.
  Nombrarlo así en el texto.
- Solo entran en la tesis **3 experimentos de Cerámica N** (20260331, 20260608,
  20260622) y **3 de Neodimio N** (20260804, 20260907, 20260911). Todo el resto
  de los experimentos de imán está descartado — el inventario lista cuáles y por
  qué.

## Cómo se presenta el resultado

El resultado de los imanes es **nulo**, y la forma de decirlo importa:

- No es "no encontramos nada". Los tests de equivalencia (TOST) **acotan** el
  efecto por debajo del margen de relevancia biológica (±10%).
- El **control positivo** (agua: 2/5/6 mL, p=2.9e-07) demuestra que el método sí
  detecta un efecto real cuando existe. Es lo que convierte "ausencia de
  evidencia" en "evidencia de ausencia".
- El piso 3 (77 mT) aparece alto en germinación y radícula, pero **no sobrevive
  la corrección por multiplicidad** (Holm). Es una hipótesis a testear, nunca un
  hallazgo. No lo presentes como resultado.

## Advertencia sobre la figura de radícula

`fig_radicula_boxplot.pdf` **no está en milímetros absolutos**: el eje Y es el
desvío respecto del bloque. Las tres fechas de Neodimio N midieron en escalas
distintas porque el pipeline de fotos no es consistente entre fechas, así que en
mm crudos el alto de cada caja reflejaba la distancia entre fechas y no el
efecto del tratamiento.

Si citás largos en mm absolutos en el texto, tomalos del `reporte.txt`, no de la
figura.
