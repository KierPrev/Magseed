# Inventario de datos y figuras — tesina magnetobiología

Documento de referencia para ubicar cada figura y tabla en el documento final de
tesis. Describe **qué existe realmente en el repositorio**, qué no existe y por
qué. Estado verificado al 2026-10-16.

> Regla general: no asumas que un endpoint existe para un grupo porque existe
> para otro. Las tablas de abajo dicen exactamente qué hay. Los experimentos
> difieren en qué se midió.

---

## 1. Qué es el experimento

Se evalúa si un campo magnético estático afecta la germinación de semillas.

- Las semillas se colocan en **bandejas** (20 semillas por bandeja).
- Las bandejas se ubican en una torre de **9 pisos** (`treatment_floor`). El
  campo decae con la altura, así que cada piso es una **dosis distinta de campo**.
- **El piso 9 es SIEMPRE el control: no recibe campo.** Los pisos 1 a 8 reciben
  campo decreciente (piso 1 = más intenso).
- Cada piso tiene 3 posiciones físicas (`floor` = A, B, C), o sea 27 bandejas
  por experimento.
- Se midió con dos imanes distintos: **cerámica** y **neodimio**. Cada uno tiene
  su propia tabla de campo por piso (en mT), en `data/extra/dosis_*_compart.csv`.

### Unidad experimental y diseño

**La unidad experimental es LA BANDEJA, nunca la semilla.** Analizar semillas
individuales sería pseudo-replicación (las 20 semillas de una bandeja comparten
el mismo tratamiento). Todos los análisis respetan esto.

El diseño es **por bloques**: el bloque es `fecha × floor` (ej.
`20260804-A`). En los dos grupos de neodimio cada bloque contiene los 9
tratamientos exactamente una vez. Cerámica presenta un desbalance parcial en
20260331, documentado en Métodos. Todos los modelos incluyen el término
`C(bloque)` y permiten ese desbalance.

### Los tres endpoints

1. **Imbibición a 72 h** (`pct_change` = `(post_weight - pre_weight) / pre_weight`).
   Es **absorción de agua, NO biomasa** — importante nombrarlo así en el texto.
2. **Germinación**: semillas germinadas sobre 20, por bandeja.
3. **Largo de radícula**: mediana del largo por bandeja, medido por análisis de
   fotos.

---

## 2. Qué datos existen por experimento

| exp_id | Tipo / polo | Bandejas | Pesos | Germinación | Raíces medidas | Fotos |
|---|---|---|---|---|---|---|
| 20260331 | Cerámica N | 27 | sí | **no** | 542 | 27 (HEIC) |
| 20260608 | Cerámica N | 27 | sí | **no** | **0 (vacío)** | no hay |
| 20260622 | Cerámica N | 27 | sí | **no** | **sin archivo** | no hay |
| 20260804 | Neodimio N | 27 | sí | **sí** | 1670 | 26 @ 4618×3464 |
| 20260907 | Neodimio N | 27 | sí | **sí** | 280 | 27 @ **960×1280** |
| 20260911 | Neodimio N | 27 | sí | **sí** | 648 | 27 @ 4618×3464 |
| 20260404 | Control negativo | 27 | sí | no | — | no hay |
| 20260428 | Control negativo | 27 | sí | no | — | no hay |
| 20260507 | Control negativo | 27 | sí | no | — | no hay |
| 20260518 | Control positivo | 9 | sí | no | — | no hay |
| 20260522 | Control positivo | 15 | 1 faltante | no | — | no hay |
| 20260603 | Control positivo | 15 | sí | no | — | no hay |
| 20260626 | Neodimio N | 27 | sí | no | — | no hay |
| 20260727 | Neodimio N | 27 | sí | no | — | no hay |
| 20260721 | Neodimio (sin polo) | 27 | sí | no | — | no hay |
| 20260814 | Neodimio **S** | 27 | sí | no | **0 (vacío)** | no hay |

### Los dos regímenes de agua

**Dato crítico, incorporado al metadata el 2026-10-15.** Las campañas se corrieron
con volúmenes de agua distintos, y el agua es la variable más influyente del
ensayo (+0,064 de imbibición por mL según el control positivo):

| Grupo | Fechas | Agua | Condición |
|---|---|---|---|
| Cerámica N | 20260331, 20260608, 20260622 | **5 mL** | saturación |
| Neodimio N | 20260626, 20260721, 20260727 | **5 mL** | saturación |
| Neodimio N | 20260804, 20260907, 20260911 | **3 mL** | **estrés hídrico** |
| Control negativo | 20260404, 20260428, 20260507 | 5 mL | saturación |
| Control positivo | 20260518, 20260522, 20260603 | 2 / 5 / 6 mL | factor deliberado |

El régimen de 3 mL se decidió **a posteriori** del resultado nulo de cerámica. El
agua está ahora en la columna `water_ml` de `experiments_metadata.csv`, y
`analisis_confirmatorio.py` **agrupa por ella**: dos campañas del mismo imán con
agua distinta son grupos separados.

El 20260804 es además una **discontinuidad de protocolo**: cambiaron a la vez el
agua, el dosificador (de cuchara volumétrica a medidor de celdas exacto de 20), el
conteo de germinación y el registro fotográfico.

### Los tres grupos de imán de la tesis

| Grupo | Fechas | Directorio de salida |
|---|---|---|
| Cerámica N, 5 mL | 20260331, 20260608, 20260622 | `confirmatorio_ceramica_n_5ml/` |
| Neodimio N, 5 mL | 20260626, 20260721, 20260727 | `confirmatorio_neodimio_n_5ml/` |
| Neodimio N, 3 mL | 20260804, 20260907, 20260911 | `confirmatorio_neodimio_n_3ml/` |

> **20260626, 20260721 y 20260727 estaban marcados como descartados en versiones
> anteriores de este inventario. La exclusión se revirtió el 2026-10-15**: se
> basaba en que no mostraron diferencias significativas, es decir en el resultado,
> lo que no es un criterio admisible. Por los datos disponibles son el mejor grupo
> del proyecto: bloques completos y balanceados (cerámica no los tiene) y
> consistencia entre fechas tres veces mejor.

### Experimentos que sí quedan fuera

- **20260814 (Neodimio S)**: no existe tabla de dosis para el polo sur ni
  caracterización de campo del neodimio S, así que los niveles no se pueden
  asociar a una intensidad. Queda fuera por eso, no por su resultado.
- **20260701**: ya eliminado del repositorio y de `experiments_metadata.csv`.

### Limitaciones de datos que hay que declarar en el texto

- **Cerámica N no tiene germinación.** La columna `no_germinadas` no existe en
  los `trays.csv` de esas 3 fechas. No es que falte procesarla: nunca se contó.
- **Cerámica N no tiene figura de radícula.** Solo 20260331 tiene raíces
  medidas; 20260608 quedó vacío y 20260622 no tiene fotos (la carpeta
  `raw/photos` está vacía, las fotos no existen). El análisis exige las 3 fechas
  del grupo, porque con una sola el "bloque" sería la fecha misma y la
  comparación dejaría de ser la que se aleatorizó.
- **Las fotos de 20260907 son exports reducidos** (960×1280 = 1.2 MP) contra
  4618×3464 (16 MP) de las otras dos. Se reprocesaron con parámetros de
  calibración tuneados (150 → 280 raíces), pero **la segmentación de las tres
  fechas de Neodimio N sigue sin ser comparable** (la escala px→mm sí: cada foto
  se calibra con la regla): el efecto de bloque da F≈37, p<0.0001,
  mucho más fuerte que cualquier efecto de tratamiento. Por eso la figura de
  radícula se grafica **centrada por bloque** (ver sección 4).

---

## 3. Dónde está cada figura

Todas las rutas son relativas a la raíz de **este** repositorio
(`tesis-magnetobiologia-software`).

### Cómo las ve el documento LaTeX

El repositorio de la tesis (`../tesis-magnetobiologia-latex`) tiene un symlink:

```
datos/software_data -> /Users/kier/MEGA/tesina/tesis-magnetobiologia-software/data
```

Por eso, desde el `.tex`, una figura de acá se referencia así:

| Ruta en este repo | Ruta en el `.tex` |
|---|---|
| `data/analysis/confirmatorio_neodimio_n_3ml/fig_radicula_boxplot.pdf` | `datos/software_data/analysis/confirmatorio_neodimio_n_3ml/fig_radicula_boxplot.pdf` |

**No hay que copiar nada**: al regenerar una figura, el documento la toma en la
siguiente compilación.

**Usar siempre la versión `.pdf`, no la `.png`.** Cada figura se exporta en los
dos formatos. El PDF es vectorial (mejor calidad en LaTeX) y es el único de los
dos que queda versionado en git, porque `.gitignore` bloquea `*.png`. El
documento ya usa `.pdf` para las figuras de caracterización de campo.

### 3.1 Resultados principales — experimentos de imán

| Figura | Ruta | Contenido |
|---|---|---|
| Cerámica N 5 mL, peso | `data/analysis/confirmatorio_ceramica_n_5ml/fig_forest_peso_germinacion.pdf` | Forest plot de imbibición por piso vs control, con banda de equivalencia ±10%. **Un solo panel** (no hay germinación). |
| Neodimio N 5 mL, peso | `data/analysis/confirmatorio_neodimio_n_5ml/fig_forest_peso_germinacion.pdf` | Forest plot de imbibición. Un panel: este grupo no tiene germinación. |
| Neodimio N 3 mL, peso + germinación | `data/analysis/confirmatorio_neodimio_n_3ml/fig_forest_peso_germinacion.pdf` | **Dos paneles**: izquierda forest plot de imbibición con banda de equivalencia, derecha germinación por piso vs control. |
| Neodimio N, germinación | `data/analysis/confirmatorio_neodimio_n_3ml/fig_germinacion_boxplot.pdf` | Boxplot de semillas germinadas de 20 por piso, con las bandejas individuales superpuestas y p-valores (crudo y Holm). |
| Neodimio N, radícula | `data/analysis/confirmatorio_neodimio_n_3ml/fig_radicula_boxplot.pdf` | Boxplot del largo de radícula **centrado por bloque**. Ver sección 4 antes de usar. |
| Línea de base y regímenes | `data/analysis/linea_base_y_regimenes/reporte.txt` | Línea de base por fecha y test de interacción entre campo y régimen de agua. Sin figura. |

> **`comparativa_imanes/` quedó fuera del documento.** Juntaba en un mismo modelo
> dos regímenes de agua, así que su coeficiente de imán (−0,112, p<0,001) no era
> un efecto del imán sino la diferencia de agua entre campañas. Su lugar lo ocupa
> el test de interacción de `linea_base_y_regimenes/`.

### 3.2 Controles

| Figura | Ruta | Contenido |
|---|---|---|
| Control positivo | `data/analysis/control_positivo_combinado/fig_control_positivo.png` | Imbibición vs agua aplicada (2/5/6 mL), las 3 fechas combinadas, n=38. |
| Control negativo | `data/analysis/negative_control_combined/*.png` | 13 figuras: variación por piso, posición, celda de bandeja, etc. Generadas por un pipeline anterior. |

### 3.3 Caracterización del campo magnético (sección de métodos)

| Figura | Ruta |
|---|---|
| Torre de exposición con campos por piso | `data/extra/plots/torre_exposicion_campos.png` (+ `.pdf`) |
| Mapa de dosis vs literatura | `data/extra/plots/mapa_dosis_literatura.png` (+ `.pdf`) |
| Ajuste de Bz, cerámica N | `data/extra/plots/ceramica_n_bz_fit.png` y `ceramica_n_bz_fit_field_map.png` |
| Ajuste de Bz, cerámica S | `data/extra/plots/ceramica_s_bz_fit.png` y `ceramica_s_bz_fit_field_map.png` |
| Ajuste de Bz, neodimio N | `data/extra/plots/neodimio_n_bz_fit.png` y `neodimio_n_bz_fit_field_map.png` |
| Heatmaps de campo por nivel | `data/extra/plots/heatmaps/heatmap_{ceramica_n,neodimio_n}_nivel{1..8}.png` (+ `.pdf`) |
| Grillas de los 8 niveles | `data/extra/plots/heatmaps/grilla_heatmaps_{ceramica_n,neodimio_n}.png` (+ `.pdf`) |

**No existe caracterización del neodimio S** y no se va a generar (experimento
descartado).

### 3.4 Figuras obsoletas — NO usar

- `data/analysis/selected_combined_plots/analisis_combinado_20260331_20260608.png`
  — metodología vieja, solo 2 de las 3 fechas de cerámica.
- `data/experiments/*/plots/*.png` — figuras por experimento individual de un
  pipeline anterior (`trays_*`, `root_lengths_boxplot`, etc.). Superadas por los
  análisis combinados de `data/analysis/`. Las de 20260331 y 20260608 incluyen
  `temps_over_time.png`, única fuente de temperatura (solo esas 2 fechas la
  registraron).

---

## 4. Advertencia sobre la figura de radícula

`fig_radicula_boxplot.png` **no está en milímetros absolutos**: el eje Y es el
desvío respecto del bloque (`mediana_mm` menos el promedio de su bloque).

Motivo: la escala px→mm se calibra foto por foto con la regla, pero la
segmentación no es consistente entre fechas (20260804 sobre-segmenta: 69 objetos por bandeja para
20 semillas, mediana 2.5 mm; 20260907 sub-detecta: 11 objetos, mediana 9.6 mm;
20260911 es el más plausible: 24 objetos, mediana 5.6 mm). Si se grafican los mm
crudos juntos, el alto de cada caja refleja la distancia entre fechas y no el
efecto del tratamiento.

Centrar por bloque es exactamente lo que hace el término `C(bloque)` del modelo,
así que **los p-valores anotados en la figura corresponden a lo que se ve**. La
media absoluta del control (6.4 mm) está escrita en el subtítulo para conservar
la referencia de escala.

Si en el texto se cita el largo en mm absolutos, tomarlo de `reporte.txt`, no de
la figura.

---

## 5. Tablas y reportes numéricos

Cada carpeta de `data/analysis/` tiene un `reporte.txt` con **todo** lo que
imprime el análisis: tabla de verificación del diseño, ANOVA, contrastes por
piso con IC y p-valores (crudo y Holm), TOST de equivalencia, efecto mínimo
detectable, tests de supuestos (Shapiro, Levene, Cook) y análisis de
sensibilidad. Es la fuente para cualquier número citado en el texto.

| Archivo | Contenido |
|---|---|
| `confirmatorio_{ceramica_n_5ml,neodimio_n_5ml,neodimio_n_3ml}/reporte.txt` | Reporte completo del grupo |
| `confirmatorio_*/peso_vs_control.csv` | Cada piso vs control: diferencia, IC95, p, p_holm |
| `confirmatorio_neodimio_n_3ml/germinacion_por_piso.csv` | Media, sd y n de germinadas por piso |
| `confirmatorio_neodimio_n_3ml/germinacion_vs_control.csv` | Cada nivel de germinación vs control, ajustado por bloque: diferencia, IC95, p y p_holm |
| `confirmatorio_neodimio_n_3ml/radicula_vs_control.csv` | Cada piso vs control en mm y % |
| `linea_base_y_regimenes/reporte.txt` | Línea de base por fecha y test de interacción agua × campo |
| `control_positivo_combinado/reporte_control_positivo.txt` | Modelo y contrastes del control positivo |
| `control_positivo_combinado/control_positivo_por_nivel.csv` | Resumen por nivel de agua |

---

## 6. Resultados principales en una línea

Para orientar dónde va cada figura en el argumento del documento:

- **Imbibición, Cerámica N 5 mL** (n=81): **+1.43%**, IC95 [−4.32%, +7.18%],
  p=0.62. TOST p=0.002 → **equivalente** dentro del margen de ±10%. Efecto
  mínimo detectable: 8.19%. Supuestos débiles: Shapiro p=0.0003 y bloques no
  balanceados en 20260331.
- **Imbibición, Neodimio N 5 mL** (n=81): **−1.46%**, IC95 [−6.28%, +3.36%],
  p=0.55. TOST p=0.00037 → **equivalente**. Efecto mínimo detectable: 6.87%.
  Bloques completos y balanceados; `C(bloque)` F=1.08 p=0.39 (campaña estable).
  Nótese que el signo es opuesto al de cerámica.
- **Imbibición, Neodimio N 3 mL** (n=81): **+1.59%**, IC95 [−0.80%, +3.98%],
  p=0.19. TOST p<0.00001 → **equivalente**. Efecto mínimo detectable: 3.40%,
  el mejor de los tres (dosificador exacto).
- **Germinación, Neodimio N 3 mL** (n=81): **+6.2 puntos porcentuales**,
  IC95 [−2.0, +14.5], p=0.14. Sin tendencia dosis-respuesta (p=0.23). Ningún
  nivel sobrevive Holm; el más alto es el nivel 6 (22 mT, crudo 0.022, Holm
  0.178) mientras el nivel 1 (263 mT) queda **por debajo** del control. Estos
  contrastes, como el análisis principal, están ajustados por bloque.
- **Radícula, Neodimio N 3 mL** (n=77): **+0.60 mm (+9.3%)**, IC95 [−9.2%,
  +27.8%], p=0.32. Efecto mínimo detectable 26.4% (endpoint de baja potencia).
- **Interacción agua × campo** (n=162): el efecto del imán **no difiere** entre
  5 y 3 mL (coef −0.0131, SE 0.0126, p=0.30). Responde negativamente la
  hipótesis que motivó el estrés hídrico. Límite: los dos regímenes difieren
  también en el dosificador.
- **Línea de base**: a 5 mL constantes la imbibición de los controles va de
  0.377 a 0.538 entre fechas **sin tendencia temporal**. Los dos grupos a 5 mL
  dan medias de control casi idénticas (0.4878 y 0.4872) pese a estar separados
  por meses. No hay deriva del sistema.
- **Control positivo** (n=38): imbibición sube **+0.064 por mL de agua**,
  p=2.9e-07, R² ajustado 0.59. Contrastes: 2 vs 5 mL p=0.004 (Holm), 2 vs 6 mL
  p=8.4e-05 (Holm).

**El argumento central**: el resultado de los imanes es nulo, pero no es "no
encontramos nada". El control positivo demuestra que el método **sí** detecta un
efecto real cuando existe (el agua produce un efecto enorme y limpio), y los
tests de equivalencia (TOST) acotan el efecto de los imanes por debajo del
margen de relevancia biológica. Esa es la diferencia entre "ausencia de
evidencia" y "evidencia de ausencia".

Observación menor, **no es un hallazgo**: el nivel 3 (77 mT) aparece como el más
alto en radícula (p=0.012 crudo, pero **Holm=0.09**, no sobrevive la corrección
por multiplicidad). En germinación el más alto es el nivel 6 (p=0.022 crudo,
**Holm=0.178**). Mencionable como
hipótesis a testear, nunca como resultado.

---

## 7. Cómo regenerar todo

```bash
# Los tres grupos de imán, de una sola vez
python src/analisis_confirmatorio.py --fechas \
    20260331 20260608 20260622 \
    20260626 20260721 20260727 \
    20260804 20260907 20260911

# El control positivo
python src/analisis_confirmatorio.py --fechas 20260518 20260522 20260603

# Línea de base por fecha y test de interacción entre regímenes
python src/analisis_linea_base.py
```

Sin `--fechas` se abre una TUI que lista los experimentos con su tipo, polo,
agua y disponibilidad de pesos / fotos / radícula / germinación.

El script agrupa por **`(experiment_type, magnet_pole, water_ml)`** y escribe en
`data/analysis/confirmatorio_<grupo>_<agua>ml/`. **El agua forma parte de la
clave de grupo**: sin eso, las dos campañas de neodimio N se analizarían juntas
promediando dos condiciones distintas. Un experimento sin `water_ml` declarado
se omite con aviso en lugar de analizarse mal.

El control negativo tiene un pipeline anterior distinto y no se regenera con
este script.

### Nota sobre git

`*.png` está en `.gitignore`, así que **los PNG no se versionan**. Los `.pdf`,
`.csv` y `.txt` sí. Por eso el documento LaTeX debe referenciar siempre el
`.pdf` de cada figura: es el que sobrevive a un clon limpio del repositorio.

Atención si se toca la caracterización de campo: las figuras
`extra/plots/*_bz_fit*.png` y las del control negativo
(`analysis/negative_control_combined/*.png`) **solo existen como PNG y no están
versionadas**, pero el documento ya las usa. Si alguna se pierde hay que
regenerarla con los scripts correspondientes (`plot_field_heatmaps.py`,
`simulate_magnet_field.py` y el pipeline del control negativo), no con
`analisis_confirmatorio.py`.
