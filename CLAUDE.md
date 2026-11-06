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

- **El nivel 9 es SIEMPRE el control** (sin campo). Los niveles 1 a 8 de la torre
  reciben campo decreciente: el nivel 1 es el más intenso. En el código la
  columna se llama `treatment_floor`, pero **en el texto de la tesis los de la
  torre son "niveles"**; "pisos" son solo A, B y C del sistema de germinación.
- **La unidad experimental es la muestra (una caja), nunca la semilla.** En el
  código se llama "bandeja"; en el texto de la tesis, "caja". Analizar semillas
  sueltas sería pseudo-replicación (las semillas de una caja comparten
  tratamiento).
- El diseño es de bloques completos aleatorizados, bloque = `fecha × floor`.
  Todos los modelos llevan `C(bloque)`.
- `pct_change` es **imbibición** (absorción de agua a 72 h), **no biomasa**.
  Nombrarlo así en el texto.
- Entran en la tesis **tres grupos de 3 experimentos**: Cerámica N 5 mL
  (20260331, 20260608, 20260622), Neodimio N 5 mL (20260626, 20260721,
  20260727) y Neodimio N 3 mL (20260804, 20260907, 20260911). Germinación y
  radícula existen **solo** en Neodimio N 3 mL. El resto de los experimentos de
  imán está descartado — el inventario lista cuáles y por qué.
- El plan de análisis estadístico se definió **después** de tener los datos;
  antes solo se fijó qué medir y cómo, y el umbral del 10%. No escribas que el
  análisis fue pre-especificado.

## Cómo se presenta el resultado

El resultado de los imanes es **nulo**, y la forma de decirlo importa:

- No es "no encontramos nada". Los tests de equivalencia (TOST) **acotan** el
  efecto por debajo del margen de relevancia biológica (±10%).
- El **control positivo** (agua: 2/5/6 mL, p=2.9e-07) demuestra que el método sí
  detecta un efecto real cuando existe. Es lo que convierte "ausencia de
  evidencia" en "evidencia de ausencia".
- El nivel 3 (77 mT) aparece alto en radícula (p=0.012, Holm 0.093) y el nivel 6
  (22 mT) en germinación (p=0.01, Holm 0.06), pero **ninguno sobrevive la
  corrección por multiplicidad** (Holm). Es una hipótesis a testear, nunca un
  hallazgo. No lo presentes como resultado.

## Advertencia sobre la figura de radícula

`fig_radicula_boxplot.pdf` **no está en milímetros absolutos**: el eje Y es el
desvío respecto del bloque. La escala px→mm NO es el problema: cada foto se
calibra por separado con la regla de referencia (la cámara estaba a distancia
variable). Lo que no es consistente entre fechas es la *segmentación*: por la
resolución distinta de las fotos, el algoritmo parte o pierde radículas en
proporciones distintas según la fecha. No escribas "escalas distintas". En
mm crudos el alto de cada caja reflejaba la distancia entre fechas y no el
efecto del tratamiento.

Si citás largos en mm absolutos en el texto, tomalos del `reporte.txt`, no de la
figura.
