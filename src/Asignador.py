import csv
import re
import sys
from typing import Optional, Union
from datetime import date, datetime
from pathlib import Path
import random
import shutil
import unicodedata


def obtener_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent.parent


BASE_DIR = obtener_base_dir()
EXPERIMENTS_DIR = BASE_DIR / "data" / "experiments"
TRAYS_FILENAME = "trays.csv"
TREATMENT_FILENAME = "treatment.csv"
METADATA_FILENAME = "experiments_metadata.csv"
DEFAULT_TREATMENT_FLOORS = 9
DEFAULT_REPLICAS = 3
MAX_TREATMENT_FLOORS = 9
MAX_PHYSICAL_FLOORS = 3
POSITIONS_PER_FLOOR = 9
MAX_TRAYS = MAX_PHYSICAL_FLOORS * POSITIONS_PER_FLOOR
MAX_REPLICAS = MAX_TRAYS
FIELDNAMES = [
    "tray_id",
    "exp_id",
    "treatment_floor",
    "floor",
    "position",
    "pre_weight",
    "post_weight",
    "no_germinadas",
]
TREATMENT_FIELDNAMES = ["treatment_id", "treatment_value", "treatment_type"]
METADATA_FIELDNAMES = ["exp_id", "experiment_type", "magnet_pole", "water_ml"]
LEGACY_METADATA_FIELDNAMES = ["exp_id", "magnet", "magnet_pole"]
DEFAULT_EXPERIMENT_TYPES = [
    "CTRL_NEGATIVO",
    "CTRL_POSITIVO",
    "IMAN_CERAMICA",
    "IMAN_NEODIMIO",
    "BOBINA",
]


def validar_exp_id(exp_id: str) -> str:
    exp_id_normalizado = exp_id.strip()
    if exp_id_normalizado == "":
        raise ValueError("La fecha del experimento no puede estar vacía.")

    try:
        datetime.strptime(exp_id_normalizado, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(
            "Fecha inválida. Debe ser una fecha real en formato YYYYMMDD."
        ) from exc

    return exp_id_normalizado


def pedir_fecha() -> str:
    hoy = date.today().strftime("%Y%m%d")

    while True:
        entrada = input(f"Fecha del experimento [YYYYMMDD] (Enter = {hoy}): ").strip()

        if entrada == "":
            return hoy

        try:
            return validar_exp_id(entrada)
        except ValueError as exc:
            print(f"{exc}\n")


def ruta_experimento(exp_id: str) -> Path:
    return EXPERIMENTS_DIR / exp_id


def ruta_trays(exp_id: str) -> Path:
    return ruta_experimento(exp_id) / TRAYS_FILENAME


def ruta_treatment(exp_id: str) -> Path:
    return ruta_experimento(exp_id) / TREATMENT_FILENAME


def ruta_raw(exp_id: str) -> Path:
    return ruta_experimento(exp_id) / "raw"


def ruta_raw_photos(exp_id: str) -> Path:
    return ruta_raw(exp_id) / "photos"


def ruta_metadata() -> Path:
    return BASE_DIR / "data" / METADATA_FILENAME


def validar_cantidad(nombre: str, valor: Union[str, int]) -> int:
    try:
        cantidad = int(valor)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"La cantidad de {nombre} debe ser un número entero.") from exc

    if cantidad <= 0:
        raise ValueError(f"La cantidad de {nombre} debe ser mayor que cero.")

    return cantidad


def validar_cantidad_maxima(nombre: str, valor: Union[str, int], maximo: int) -> int:
    cantidad = validar_cantidad(nombre, valor)

    if cantidad > maximo:
        raise ValueError(f"La cantidad de {nombre} no puede ser mayor que {maximo}.")

    return cantidad


def validar_configuracion_trays(
    cantidad_treatment_floors: Union[str, int],
    cantidad_replicas: Union[str, int],
) -> tuple[int, int]:
    treatment_floors = validar_cantidad_maxima(
        "treatment floors", cantidad_treatment_floors, MAX_TREATMENT_FLOORS
    )
    replicas = validar_cantidad("réplicas", cantidad_replicas)
    total_trays = treatment_floors * replicas

    if total_trays > MAX_TRAYS:
        raise ValueError(
            f"La configuración genera {total_trays} bandejas, pero solo hay "
            f"{MAX_TRAYS} posiciones disponibles (A1 a C9)."
        )

    return treatment_floors, replicas


def floor_desde_indice(indice: int) -> str:
    if indice < 0:
        raise ValueError("El índice del floor no puede ser negativo.")

    etiqueta = ""
    indice_actual = indice

    while True:
        indice_actual, resto = divmod(indice_actual, 26)
        etiqueta = chr(ord("A") + resto) + etiqueta
        if indice_actual == 0:
            return etiqueta
        indice_actual -= 1


def indice_floor(floor: str) -> int:
    etiqueta = floor.strip().upper()
    if not etiqueta.isalpha():
        raise ValueError(f"Floor inválido: '{floor}'.")

    indice = 0
    for caracter in etiqueta:
        indice = indice * 26 + (ord(caracter) - ord("A") + 1)
    return indice - 1


def generar_floors(cantidad_replicas: int = DEFAULT_REPLICAS) -> list[str]:
    cantidad_floors = validar_cantidad_maxima(
        "floors físicos", cantidad_replicas, MAX_PHYSICAL_FLOORS
    )
    return [floor_desde_indice(indice) for indice in range(cantidad_floors)]


def generar_posiciones_fisicas() -> list[tuple[str, int]]:
    posiciones = []
    for floor in generar_floors(MAX_PHYSICAL_FLOORS):
        for position in range(1, POSITIONS_PER_FLOOR + 1):
            posiciones.append((floor, position))
    return posiciones


def generar_posiciones(
    cantidad_treatment_floors: int = DEFAULT_TREATMENT_FLOORS,
    cantidad_replicas: int = DEFAULT_REPLICAS,
) -> list[tuple[str, int]]:
    treatment_floors, replicas = validar_configuracion_trays(
        cantidad_treatment_floors, cantidad_replicas
    )
    total_trays = treatment_floors * replicas
    return generar_posiciones_fisicas()[:total_trays]


def generar_treatment_floors(
    cantidad_treatment_floors: int = DEFAULT_TREATMENT_FLOORS,
    cantidad_replicas: int = DEFAULT_REPLICAS,
) -> list[int]:
    treatment_floors_count, replicas = validar_configuracion_trays(
        cantidad_treatment_floors, cantidad_replicas
    )
    treatment_floors = []
    for treatment_floor in range(1, treatment_floors_count + 1):
        treatment_floors.extend([treatment_floor] * replicas)
    return treatment_floors


def generar_treatment_floors_balanceados(
    posiciones: list[tuple[str, int]],
    cantidad_treatment_floors: int = DEFAULT_TREATMENT_FLOORS,
    cantidad_replicas: int = DEFAULT_REPLICAS,
) -> list[int]:
    treatment_floors_count, replicas = validar_configuracion_trays(
        cantidad_treatment_floors, cantidad_replicas
    )
    pendientes = {
        treatment_floor: replicas
        for treatment_floor in range(1, treatment_floors_count + 1)
    }
    usados_por_floor = {
        floor: {treatment_floor: 0 for treatment_floor in pendientes}
        for floor, _ in posiciones
    }
    asignados = []

    for floor, _ in posiciones:
        candidatos = [
            treatment_floor
            for treatment_floor, cantidad_pendiente in pendientes.items()
            if cantidad_pendiente > 0
        ]
        menor_uso_en_floor = min(
            usados_por_floor[floor][treatment_floor]
            for treatment_floor in candidatos
        )
        candidatos_balanceados = [
            treatment_floor
            for treatment_floor in candidatos
            if usados_por_floor[floor][treatment_floor] == menor_uso_en_floor
        ]
        elegido = random.choice(candidatos_balanceados)

        asignados.append(elegido)
        pendientes[elegido] -= 1
        usados_por_floor[floor][elegido] += 1

    return asignados


def normalizar_experiment_type(experiment_type: str) -> str:
    valor = unicodedata.normalize("NFKD", experiment_type.strip().upper())
    valor = valor.encode("ascii", "ignore").decode("ascii")
    valor = re.sub(r"\s+", "_", valor)
    valor = re.sub(r"_+", "_", valor)

    if valor == "":
        raise ValueError("El tipo de experimento no puede estar vacío.")

    return valor


def normalizar_magnet_pole(magnet_pole: str) -> str:
    magnet_pole_normalizado = magnet_pole.strip().upper()

    if magnet_pole_normalizado == "":
        raise ValueError("El magnet_pole no puede estar vacío.")

    return magnet_pole_normalizado


def normalizar_water_ml(water_ml: str) -> str:
    """Agua aplicada por bandeja, en mL. Vacio es valido: los controles
    positivos la registran por bandeja en su propio trays.csv, y hay
    experimentos viejos donde el volumen no quedo anotado."""
    valor = str(water_ml).strip()
    if valor == "":
        return ""

    try:
        numero = float(valor)
    except ValueError:
        raise ValueError(f"water_ml debe ser un numero en mL o vacio, no '{water_ml}'.")

    if numero <= 0:
        raise ValueError(f"water_ml debe ser positivo, no '{water_ml}'.")

    return f"{numero:g}"


def pedir_metadata_experimento() -> dict[str, str]:
    while True:
        experiment_type = input(
            "Tipo de experimento (ej. CTRL_NEGATIVO, IMAN_CERAMICA, BOBINA): "
        ).strip()
        try:
            experiment_type = normalizar_experiment_type(experiment_type)
        except ValueError as exc:
            print(f"{exc}\n")
            continue
        break

    while True:
        magnet_pole = input("Magnet pole del experimento (ej. N, S, NONE): ").strip()
        try:
            magnet_pole = normalizar_magnet_pole(magnet_pole)
        except ValueError as exc:
            print(f"{exc}\n")
            continue
        break

    while True:
        water_ml = input(
            "Agua aplicada por bandeja en mL (enter si no aplica o no se sabe): "
        ).strip()
        try:
            water_ml = normalizar_water_ml(water_ml)
        except ValueError as exc:
            print(f"{exc}\n")
            continue
        break

    return {
        "experiment_type": experiment_type,
        "magnet_pole": magnet_pole,
        "water_ml": water_ml,
    }


def normalizar_metadata(
    experiment_type: str, magnet_pole: str, water_ml: str = ""
) -> dict[str, str]:
    return {
        "experiment_type": normalizar_experiment_type(experiment_type),
        "magnet_pole": normalizar_magnet_pole(magnet_pole),
        "water_ml": normalizar_water_ml(water_ml),
    }


def normalizar_fila_metadata(fila: dict[str, str]) -> dict[str, str]:
    experiment_type = fila.get("experiment_type") or fila.get("magnet") or ""
    return {
        "exp_id": fila.get("exp_id", "").strip(),
        "experiment_type": normalizar_experiment_type(experiment_type),
        "magnet_pole": normalizar_magnet_pole(fila.get("magnet_pole", "")),
        "water_ml": normalizar_water_ml(fila.get("water_ml", "")),
    }


def leer_metadata_experimentos(normalizar=True) -> list[dict[str, str]]:
    archivo = ruta_metadata()
    if not archivo.exists():
        return []

    with open(archivo, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        filas = list(reader)

    if not normalizar:
        return filas

    metadata = []
    for fila in filas:
        try:
            metadata.append(normalizar_fila_metadata(fila))
        except ValueError:
            metadata.append(
                {
                    "exp_id": fila.get("exp_id", "").strip(),
                    "experiment_type": (
                        fila.get("experiment_type") or fila.get("magnet") or ""
                    ).strip().upper(),
                    "magnet_pole": fila.get("magnet_pole", "").strip().upper(),
                }
            )
    return metadata


def obtener_experiment_types_disponibles() -> list[str]:
    tipos = set(DEFAULT_EXPERIMENT_TYPES)
    for fila in leer_metadata_experimentos():
        experiment_type = fila.get("experiment_type", "").strip()
        if experiment_type:
            tipos.add(experiment_type)
    return sorted(tipos)


def migrar_metadata_csv_si_hace_falta() -> None:
    archivo = ruta_metadata()
    if not archivo.exists() or archivo.stat().st_size == 0:
        return

    with open(archivo, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        filas = list(reader)

    if fieldnames == METADATA_FIELDNAMES:
        return

    if "experiment_type" not in fieldnames and "magnet" not in fieldnames:
        return

    metadata = [normalizar_fila_metadata(fila) for fila in filas]
    with open(archivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metadata)


def asegurar_salto_final_csv(archivo: Path) -> None:
    if not archivo.exists() or archivo.stat().st_size == 0:
        return

    with open(archivo, "rb+") as f:
        f.seek(-1, 2)
        ultimo_byte = f.read(1)
        if ultimo_byte not in {b"\n", b"\r"}:
            f.write(b"\n")


def crear_metadata_csv_con_valores(
    exp_id: str, experiment_type: str, magnet_pole: str, water_ml: str = ""
) -> Path:
    exp_id = validar_exp_id(exp_id)
    archivo = ruta_metadata()

    metadata = normalizar_metadata(experiment_type, magnet_pole, water_ml)
    crear_estructura_experimento(exp_id)
    migrar_metadata_csv_si_hace_falta()

    metadata_existente = leer_metadata_experimentos()
    if any(fila["exp_id"] == exp_id for fila in metadata_existente):
        raise FileExistsError(
            f"Ya existe metadata para el experimento '{exp_id}' en '{archivo}'."
        )

    archivo.parent.mkdir(parents=True, exist_ok=True)
    asegurar_salto_final_csv(archivo)

    with open(archivo, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=METADATA_FIELDNAMES,
            lineterminator="\n",
        )
        if archivo.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(
            {
                "exp_id": exp_id,
                "experiment_type": metadata["experiment_type"],
                "magnet_pole": metadata["magnet_pole"],
                "water_ml": metadata["water_ml"],
            }
        )

    return archivo


def crear_metadata_csv(
    exp_id: str,
    experiment_type: Optional[str] = None,
    magnet_pole: Optional[str] = None,
    water_ml: Optional[str] = None,
):
    if experiment_type is None or magnet_pole is None:
        metadata = pedir_metadata_experimento()
        experiment_type = metadata["experiment_type"]
        magnet_pole = metadata["magnet_pole"]
        if water_ml is None:
            water_ml = metadata["water_ml"]

    try:
        archivo = crear_metadata_csv_con_valores(
            exp_id, experiment_type, magnet_pole, water_ml or ""
        )
    except (ValueError, FileExistsError) as exc:
        print(f"{exc}\n")
        return None

    print(f"Metadata del experimento agregada correctamente en '{archivo}'.\n")
    return archivo


def crear_estructura_experimento(exp_id: str) -> Path:
    exp_id = validar_exp_id(exp_id)
    carpeta = ruta_experimento(exp_id)
    ruta_raw_photos(exp_id).mkdir(parents=True, exist_ok=True)
    (carpeta / "processed").mkdir(parents=True, exist_ok=True)
    (carpeta / "plots").mkdir(parents=True, exist_ok=True)
    return carpeta


def nombre_disponible(destino: Path) -> Path:
    if not destino.exists():
        return destino

    stem = destino.stem
    suffix = destino.suffix
    contador = 2

    while True:
        candidato = destino.with_name(f"{stem}_{contador}{suffix}")
        if not candidato.exists():
            return candidato
        contador += 1


def copiar_fotos_a_experimento(exp_id: str, fotos: list[Union[str, Path]]) -> list[Path]:
    exp_id = validar_exp_id(exp_id)

    if not fotos:
        raise ValueError("Tenés que seleccionar al menos una foto.")

    destino_dir = ruta_raw_photos(exp_id)
    destino_dir.mkdir(parents=True, exist_ok=True)

    copiados = []
    for foto in fotos:
        origen = Path(foto).expanduser()
        if not origen.exists() or not origen.is_file():
            raise FileNotFoundError(f"No existe el archivo '{origen}'.")

        destino = nombre_disponible(destino_dir / origen.name)
        shutil.copy2(origen, destino)
        copiados.append(destino)

    return copiados


def crear_treatment_csv_para_experimento(
    exp_id: str,
    cantidad_treatment_floors: int = DEFAULT_TREATMENT_FLOORS,
    sobrescribir: bool = False,
) -> Path:
    exp_id = validar_exp_id(exp_id)
    treatment_floors_count = validar_cantidad_maxima(
        "treatment floors", cantidad_treatment_floors, MAX_TREATMENT_FLOORS
    )
    crear_estructura_experimento(exp_id)
    archivo = ruta_treatment(exp_id)

    if archivo.exists() and not sobrescribir:
        return archivo

    with open(archivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TREATMENT_FIELDNAMES)
        writer.writeheader()
        for treatment_id in range(1, treatment_floors_count + 1):
            writer.writerow(
                {
                    "treatment_id": treatment_id,
                    "treatment_value": "",
                    "treatment_type": "",
                }
            )

    return archivo


def leer_treatments(exp_id: str) -> list[dict[str, str]]:
    exp_id = validar_exp_id(exp_id)
    archivo = ruta_treatment(exp_id)
    with open(archivo, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sorted(
            list(reader),
            key=lambda fila: int(fila["treatment_id"]),
        )


def guardar_treatments(exp_id: str, datos) -> None:
    exp_id = validar_exp_id(exp_id)
    archivo = ruta_treatment(exp_id)
    with open(archivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TREATMENT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(datos)


def actualizar_treatment_en_id(
    exp_id: str,
    treatment_id: Union[str, int],
    treatment_value: str,
    treatment_type: str,
) -> dict[str, object]:
    treatment_id_num = validar_cantidad("treatment_id", treatment_id)
    datos = leer_treatments(exp_id)
    fila = next(
        (
            item
            for item in datos
            if int(item["treatment_id"]) == treatment_id_num
        ),
        None,
    )

    if fila is None:
        raise LookupError(f"No existe treatment_id {treatment_id_num}.")

    fila["treatment_value"] = treatment_value.strip()
    fila["treatment_type"] = treatment_type.strip()
    guardar_treatments(exp_id, datos)

    return {
        "fila": fila,
        "datos": datos,
    }


def crear_trays_csv_para_experimento(
    exp_id: str,
    cantidad_treatment_floors: int = DEFAULT_TREATMENT_FLOORS,
    cantidad_replicas: int = DEFAULT_REPLICAS,
) -> Path:
    exp_id = validar_exp_id(exp_id)
    treatment_floors_count, replicas = validar_configuracion_trays(
        cantidad_treatment_floors, cantidad_replicas
    )
    crear_estructura_experimento(exp_id)
    archivo = ruta_trays(exp_id)

    if archivo.exists():
        raise FileExistsError(f"El archivo '{archivo}' ya existe.")

    posiciones = generar_posiciones(treatment_floors_count, replicas)

    treatment_floors = generar_treatment_floors_balanceados(
        posiciones,
        treatment_floors_count,
        replicas,
    )

    with open(archivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for (floor, position), treatment_floor in zip(posiciones, treatment_floors):
            tray_id = f"{exp_id}_{floor}{position}"
            writer.writerow(
                {
                    "tray_id": tray_id,
                    "exp_id": exp_id,
                    "treatment_floor": treatment_floor,
                    "floor": floor,
                    "position": position,
                    "pre_weight": "",
                    "post_weight": "",
                    "no_germinadas": "",
                }
            )

    crear_treatment_csv_para_experimento(exp_id, treatment_floors_count)
    return archivo


def crear_trays_csv(
    exp_id: str,
    cantidad_treatment_floors: int = DEFAULT_TREATMENT_FLOORS,
    cantidad_replicas: int = DEFAULT_REPLICAS,
):
    try:
        archivo = crear_trays_csv_para_experimento(
            exp_id,
            cantidad_treatment_floors=cantidad_treatment_floors,
            cantidad_replicas=cantidad_replicas,
        )
    except (ValueError, FileExistsError) as exc:
        print(f"{exc}\n")
        return None

    print(f"Archivo '{archivo}' creado correctamente.\n")
    return archivo


def leer_datos(exp_id: str):
    exp_id = validar_exp_id(exp_id)
    archivo = ruta_trays(exp_id)
    with open(archivo, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        datos = list(reader)

    for fila in datos:
        fila.setdefault("no_germinadas", "")

    return datos


def guardar_datos(exp_id: str, datos):
    exp_id = validar_exp_id(exp_id)
    archivo = ruta_trays(exp_id)
    with open(archivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(datos)


def ordenar_datos(datos):
    return sorted(
        datos,
        key=lambda fila: (indice_floor(fila["floor"]), int(fila["position"])),
    )


def mostrar_bandejas(datos):
    print("\nEstado actual:")
    datos_ordenados = ordenar_datos(datos)

    for fila in datos_ordenados:
        pos = f"{fila['floor']}{fila['position']}"
        print(
            f"{fila['tray_id']} | "
            f"Treatment floor: {fila['treatment_floor']} | "
            f"Pos: {pos:<2} | "
            f"Antes: {fila['pre_weight'] or '-':>6} | "
            f"Después: {fila['post_weight'] or '-':>6} | "
            f"No germinadas: {fila.get('no_germinadas') or '-':>6}"
        )
    print()


def posiciones_en_orden(
    cantidad_treatment_floors: int = DEFAULT_TREATMENT_FLOORS,
    cantidad_replicas: int = DEFAULT_REPLICAS,
) -> list[tuple[str, int]]:
    return generar_posiciones(cantidad_treatment_floors, cantidad_replicas)


def posiciones_en_orden_desde_datos(datos) -> list[tuple[str, int]]:
    return [
        (fila["floor"], int(fila["position"]))
        for fila in ordenar_datos(datos)
    ]


def obtener_posiciones_validas(datos) -> set[tuple[str, int]]:
    return set(posiciones_en_orden_desde_datos(datos))


def obtener_configuracion_trays(datos) -> dict[str, object]:
    posiciones = posiciones_en_orden_desde_datos(datos)
    floors = sorted({floor for floor, _ in posiciones}, key=indice_floor)
    treatment_floors = sorted({int(fila["treatment_floor"]) for fila in datos})
    cantidad_treatment_floors = len(treatment_floors)
    cantidad_replicas = (
        len(datos) // cantidad_treatment_floors
        if cantidad_treatment_floors > 0
        else 0
    )

    return {
        "cantidad_replicas": cantidad_replicas,
        "cantidad_treatment_floors": cantidad_treatment_floors,
        "floors": floors,
        "posiciones": posiciones,
        "primer_posicion": f"{posiciones[0][0]}{posiciones[0][1]}" if posiciones else "A1",
        "ultima_posicion": f"{posiciones[-1][0]}{posiciones[-1][1]}" if posiciones else "A1",
    }


def validar_posicion(
    posicion: str,
    posiciones_validas: Optional[set[tuple[str, int]]] = None,
) -> tuple[str, int]:
    entrada = posicion.strip().upper()
    match = re.fullmatch(r"([A-Z]+)(\d+)", entrada)

    if match is None:
        raise ValueError("Ingresá una posición válida, por ejemplo A1 o B3.")

    floor_entrada, position_entrada = match.groups()
    position_num = int(position_entrada)
    if position_num <= 0:
        raise ValueError("La posición debe ser mayor que cero.")

    posiciones_fisicas = set(generar_posiciones_fisicas())
    if (floor_entrada, position_num) not in posiciones_fisicas:
        raise ValueError("La posición debe estar dentro del rango físico A1 a C9.")

    if posiciones_validas is not None:
        if not posiciones_validas:
            raise ValueError("Primero cargá un experimento con bandejas.")

        if (floor_entrada, position_num) in posiciones_validas:
            return floor_entrada, position_num

        posiciones_ordenadas = sorted(
            posiciones_validas,
            key=lambda item: (indice_floor(item[0]), item[1]),
        )
        primera = f"{posiciones_ordenadas[0][0]}{posiciones_ordenadas[0][1]}"
        ultima = f"{posiciones_ordenadas[-1][0]}{posiciones_ordenadas[-1][1]}"
        raise ValueError(
            f"La posición debe estar dentro de las bandejas cargadas ({primera} a {ultima})."
        )

    return floor_entrada, position_num


def buscar_fila(datos, floor: str, position: int):
    return next(
        (
            x
            for x in datos
            if x["floor"] == floor and int(x["position"]) == position
        ),
        None,
    )



def sugerir_siguiente_posicion(datos, ultima_posicion: Optional[tuple[str, int]] = None) -> str:
    orden = posiciones_en_orden_desde_datos(datos)
    if not orden:
        return "A1"

    if ultima_posicion is not None and ultima_posicion in orden:
        indice_actual = orden.index(ultima_posicion)
        if indice_actual < len(orden) - 1:
            siguiente_floor, siguiente_position = orden[indice_actual + 1]
            return f"{siguiente_floor}{siguiente_position}"

    for floor, position in orden:
        fila = next(
            (
                x
                for x in datos
                if x["floor"] == floor and int(x["position"]) == position
            ),
            None,
        )
        if fila is not None and (fila["pre_weight"] == "" or fila["post_weight"] == ""):
            return f"{floor}{position}"

    return f"{orden[0][0]}{orden[0][1]}"


def obtener_estado_experimento(
    exp_id: str, ultima_posicion: Optional[tuple[str, int]] = None
) -> dict[str, object]:
    archivo = ruta_trays(validar_exp_id(exp_id))

    if not archivo.exists():
        raise FileNotFoundError(
            f"No existe '{archivo}'. Primero tenés que crear el experimento."
        )

    datos = leer_datos(exp_id)
    configuracion = obtener_configuracion_trays(datos)
    crear_treatment_csv_para_experimento(
        exp_id,
        configuracion["cantidad_treatment_floors"],
    )
    return {
        "datos": ordenar_datos(datos),
        "siguiente_posicion": sugerir_siguiente_posicion(datos, ultima_posicion),
        "configuracion": configuracion,
        "treatments": leer_treatments(exp_id),
    }


def actualizar_pesos_en_posicion(
    exp_id: str,
    posicion: str,
    nuevo_antes: Optional[str] = None,
    nuevo_despues: Optional[str] = None,
    nuevo_no_germinadas: Optional[str] = None,
) -> dict[str, object]:
    archivo = ruta_trays(validar_exp_id(exp_id))

    if not archivo.exists():
        raise FileNotFoundError(
            f"No existe '{archivo}'. Primero tenés que crear el experimento."
        )

    datos = leer_datos(exp_id)
    posiciones_validas = obtener_posiciones_validas(datos)
    floor_entrada, position_num = validar_posicion(
        posicion,
        posiciones_validas=posiciones_validas,
    )
    fila = buscar_fila(datos, floor_entrada, position_num)

    if fila is None:
        raise LookupError("No se encontró esa bandeja.")

    if nuevo_antes is not None and nuevo_antes.strip() != "":
        fila["pre_weight"] = nuevo_antes.strip()

    if nuevo_despues is not None and nuevo_despues.strip() != "":
        fila["post_weight"] = nuevo_despues.strip()

    if nuevo_no_germinadas is not None and nuevo_no_germinadas.strip() != "":
        fila["no_germinadas"] = nuevo_no_germinadas.strip()

    guardar_datos(exp_id, datos)

    ultima_posicion_editada = (fila["floor"], int(fila["position"]))
    return {
        "fila": fila,
        "datos": ordenar_datos(datos),
        "siguiente_posicion": sugerir_siguiente_posicion(datos, ultima_posicion_editada),
    }


def actualizar_pesos():
    exp_id = pedir_fecha()
    try:
        estado = obtener_estado_experimento(exp_id)
    except FileNotFoundError as exc:
        print(f"{exc}\n")
        return

    datos = estado["datos"]
    ultima_posicion_editada = None

    while True:
        mostrar_bandejas(datos)

        sugerida = sugerir_siguiente_posicion(datos, ultima_posicion_editada)
        entrada = input(
            f"Posición a editar [sugerida: {sugerida}] (Enter = usar sugerida, 'salir' = terminar): "
        ).strip().upper()
        if entrada == "SALIR":
            break
        if entrada == "":
            entrada = sugerida

        try:
            floor_entrada, position_num = validar_posicion(
                entrada,
                posiciones_validas=obtener_posiciones_validas(datos),
            )
        except ValueError as exc:
            print(f"{exc}\n")
            continue

        fila = buscar_fila(datos, floor_entrada, position_num)
        if fila is None:
            print("No se encontró esa bandeja.\n")
            continue

        print(
            f"\nEditando posición {fila['floor']}{fila['position']} | "
            f"Tray ID {fila['tray_id']} | "
            f"Treatment floor {fila['treatment_floor']}"
        )

        print(f"Peso antes actual: {fila['pre_weight'] or '-'}")
        print(f"Peso después actual: {fila['post_weight'] or '-'}")
        print(f"No germinadas actual: {fila.get('no_germinadas') or '-'}")

        nuevo_antes = input("Nuevo peso antes (Enter para dejar igual): ").strip()
        nuevo_despues = input("Nuevo peso después (Enter para dejar igual): ").strip()
        nuevo_no_germinadas = input(
            "Nueva cantidad de no germinadas (Enter para dejar igual): "
        ).strip()

        resultado = actualizar_pesos_en_posicion(
            exp_id, entrada, nuevo_antes, nuevo_despues, nuevo_no_germinadas
        )
        datos = resultado["datos"]
        fila = resultado["fila"]
        ultima_posicion_editada = (fila["floor"], int(fila["position"]))
        siguiente_sugerida = resultado["siguiente_posicion"]

        print(f"Siguiente posición sugerida: {siguiente_sugerida}")
        print("Datos guardados.\n")


def menu():
    while True:
        print("=== MENÚ ===")
        print("1. Crear trays.csv para un experimento")
        print("2. Agregar experimento a experiments_metadata.csv")
        print("3. Ver / actualizar pesos")
        print("4. Salir")

        opcion = input("Elegí una opción: ").strip()

        if opcion == "1":
            exp_id = pedir_fecha()
            cantidad_treatment_floors = input(
                f"Cantidad de treatment floors (Enter = {DEFAULT_TREATMENT_FLOORS}): "
            ).strip()
            cantidad_replicas = input(
                f"Cantidad de réplicas (Enter = {DEFAULT_REPLICAS}): "
            ).strip()

            crear_trays_csv(
                exp_id,
                cantidad_treatment_floors=(
                    cantidad_treatment_floors
                    if cantidad_treatment_floors
                    else DEFAULT_TREATMENT_FLOORS
                ),
                cantidad_replicas=(
                    cantidad_replicas
                    if cantidad_replicas
                    else DEFAULT_REPLICAS
                ),
            )
        elif opcion == "2":
            exp_id = pedir_fecha()
            crear_estructura_experimento(exp_id)
            crear_metadata_csv(exp_id)
        elif opcion == "3":
            actualizar_pesos()
        elif opcion == "4":
            print("Fin.")
            break
        else:
            print("Opción inválida.\n")


if __name__ == "__main__":
    menu()
