import csv
import re
from typing import Optional, Union
from datetime import date, datetime
from pathlib import Path
import random
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = BASE_DIR / "data" / "experiments"
TRAYS_FILENAME = "trays.csv"
METADATA_FILENAME = "experiments_metadata.csv"
DEFAULT_TREATMENT_FLOORS = 9
DEFAULT_REPLICAS = 3
FIELDNAMES = [
    "tray_id",
    "exp_id",
    "treatment_floor",
    "floor",
    "position",
    "pre_weight",
    "post_weight",
]
METADATA_FIELDNAMES = ["exp_id", "magnet", "magnet_pole"]


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
    replicas = validar_cantidad("réplicas", cantidad_replicas)
    return [floor_desde_indice(indice) for indice in range(replicas)]


def generar_posiciones(
    cantidad_treatment_floors: int = DEFAULT_TREATMENT_FLOORS,
    cantidad_replicas: int = DEFAULT_REPLICAS,
) -> list[tuple[str, int]]:
    treatment_floors = validar_cantidad(
        "treatment floors", cantidad_treatment_floors
    )
    replicas = validar_cantidad("réplicas", cantidad_replicas)
    posiciones = []
    for floor in generar_floors(replicas):
        for position in range(1, treatment_floors + 1):
            posiciones.append((floor, position))
    return posiciones


def generar_treatment_floors(
    cantidad_treatment_floors: int = DEFAULT_TREATMENT_FLOORS,
    cantidad_replicas: int = DEFAULT_REPLICAS,
) -> list[int]:
    treatment_floors_count = validar_cantidad(
        "treatment floors", cantidad_treatment_floors
    )
    replicas = validar_cantidad("réplicas", cantidad_replicas)
    treatment_floors = []
    for treatment_floor in range(1, treatment_floors_count + 1):
        treatment_floors.extend([treatment_floor] * replicas)
    return treatment_floors


def pedir_metadata_experimento() -> dict[str, str]:
    while True:
        magnet = input("Magnet del experimento (ej. M1, M2, CONTROL): ").strip().upper()
        if magnet == "":
            print("El magnet no puede estar vacío.\n")
            continue
        break

    while True:
        magnet_pole = input("Magnet pole del experimento (ej. N, S, NONE): ").strip().upper()
        if magnet_pole == "":
            print("El magnet_pole no puede estar vacío.\n")
            continue
        break

    return {
        "magnet": magnet,
        "magnet_pole": magnet_pole,
    }


def normalizar_metadata(magnet: str, magnet_pole: str) -> dict[str, str]:
    magnet_normalizado = magnet.strip().upper()
    magnet_pole_normalizado = magnet_pole.strip().upper()

    if magnet_normalizado == "":
        raise ValueError("El magnet no puede estar vacío.")

    if magnet_pole_normalizado == "":
        raise ValueError("El magnet_pole no puede estar vacío.")

    return {
        "magnet": magnet_normalizado,
        "magnet_pole": magnet_pole_normalizado,
    }


def leer_metadata_experimentos() -> list[dict[str, str]]:
    archivo = ruta_metadata()
    if not archivo.exists():
        return []

    with open(archivo, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def asegurar_salto_final_csv(archivo: Path) -> None:
    if not archivo.exists() or archivo.stat().st_size == 0:
        return

    with open(archivo, "rb+") as f:
        f.seek(-1, 2)
        ultimo_byte = f.read(1)
        if ultimo_byte not in {b"\n", b"\r"}:
            f.write(b"\n")


def crear_metadata_csv_con_valores(exp_id: str, magnet: str, magnet_pole: str) -> Path:
    exp_id = validar_exp_id(exp_id)
    archivo = ruta_metadata()

    metadata = normalizar_metadata(magnet, magnet_pole)
    crear_estructura_experimento(exp_id)

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
                "magnet": metadata["magnet"],
                "magnet_pole": metadata["magnet_pole"],
            }
        )

    return archivo


def crear_metadata_csv(
    exp_id: str,
    magnet: Optional[str] = None,
    magnet_pole: Optional[str] = None,
):
    if magnet is None or magnet_pole is None:
        metadata = pedir_metadata_experimento()
        magnet = metadata["magnet"]
        magnet_pole = metadata["magnet_pole"]

    try:
        archivo = crear_metadata_csv_con_valores(exp_id, magnet, magnet_pole)
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


def crear_trays_csv_para_experimento(
    exp_id: str,
    cantidad_treatment_floors: int = DEFAULT_TREATMENT_FLOORS,
    cantidad_replicas: int = DEFAULT_REPLICAS,
) -> Path:
    exp_id = validar_exp_id(exp_id)
    treatment_floors_count = validar_cantidad(
        "treatment floors", cantidad_treatment_floors
    )
    replicas = validar_cantidad("réplicas", cantidad_replicas)
    crear_estructura_experimento(exp_id)
    archivo = ruta_trays(exp_id)

    if archivo.exists():
        raise FileExistsError(f"El archivo '{archivo}' ya existe.")

    posiciones = generar_posiciones(treatment_floors_count, replicas)

    treatment_floors = generar_treatment_floors(treatment_floors_count, replicas)
    random.shuffle(treatment_floors)

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
                }
            )

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
        return list(reader)


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
            f"Después: {fila['post_weight'] or '-':>6}"
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
    treatment_floors = sorted({position for _, position in posiciones})

    return {
        "cantidad_replicas": len(floors),
        "cantidad_treatment_floors": len(treatment_floors),
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
    return {
        "datos": ordenar_datos(datos),
        "siguiente_posicion": sugerir_siguiente_posicion(datos, ultima_posicion),
        "configuracion": obtener_configuracion_trays(datos),
    }


def actualizar_pesos_en_posicion(
    exp_id: str,
    posicion: str,
    nuevo_antes: Optional[str] = None,
    nuevo_despues: Optional[str] = None,
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

        nuevo_antes = input("Nuevo peso antes (Enter para dejar igual): ").strip()
        nuevo_despues = input("Nuevo peso después (Enter para dejar igual): ").strip()

        resultado = actualizar_pesos_en_posicion(exp_id, entrada, nuevo_antes, nuevo_despues)
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
