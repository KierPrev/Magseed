import importlib
import io
import re
import unicodedata
from pathlib import Path
import traceback
import tkinter as tk
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from tkinter import filedialog, messagebox, scrolledtext, ttk

BG_COLOR = "#f3eee6"
CARD_COLOR = "#fffaf3"
ACCENT_COLOR = "#2f6b5b"
ACCENT_DARK = "#234d42"
TEXT_COLOR = "#1f2a25"
MUTED_COLOR = "#63706b"
BORDER_COLOR = "#d7d0c6"
DEFAULT_IQR_FACTOR = 1.5
DEFAULT_OBJETO = "Radicula"
DEFAULT_MEDIDA = "Largo_cm"
DEFAULT_EXPERIMENTS_DIR = "./data/experiments"
NON_MAGNETIC_EXPERIMENT_TYPES = {"CTRL_NEGATIVO", "CTRL_POSITIVO", "NONE"}


def normalizar_experiment_type(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value).strip().upper())
    value = value.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"_+", "_", re.sub(r"\s+", "_", value))


def metadata_experiment_type(row) -> str:
    return normalizar_experiment_type(row.get("experiment_type") or row.get("magnet") or "")


class TextWidgetWriter(io.StringIO):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def write(self, text):
        if not text:
            return 0
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.update_idletasks()
        return len(text)

    def flush(self):
        self.widget.update_idletasks()


class PipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Pipeline de Analisis")
        self.geometry("1100x760")
        self.minsize(980, 700)
        self.configure(bg=BG_COLOR)

        self._module_cache = {}

        self.iqr_var = tk.StringVar(value=str(DEFAULT_IQR_FACTOR))
        self.objeto_var = tk.StringVar(value=DEFAULT_OBJETO)
        self.medida_var = tk.StringVar(value=DEFAULT_MEDIDA)
        self.calibracion_var = tk.BooleanVar(value=False)
        self.reprocesar_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Abriendo interfaz...")
        self.base_dir_var = tk.StringVar(value=f"Origen actual: {DEFAULT_EXPERIMENTS_DIR}")
        self.action_buttons = []

        self._configure_styles()
        self._build_layout()
        self.after(50, lambda: self.refresh_folders(select_all=True))

    def get_module(self, module_name):
        module = self._module_cache.get(module_name)
        if module is None:
            module = importlib.import_module(module_name)
            self._module_cache[module_name] = module
        return module

    def _configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("App.TFrame", background=BG_COLOR)
        style.configure(
            "Card.TFrame",
            background=CARD_COLOR,
            relief="solid",
            borderwidth=1,
            bordercolor=BORDER_COLOR,
        )
        style.configure(
            "Title.TLabel",
            background=BG_COLOR,
            foreground=TEXT_COLOR,
            font=("Helvetica", 22, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=BG_COLOR,
            foreground=MUTED_COLOR,
            font=("Helvetica", 11),
        )
        style.configure(
            "Section.TLabel",
            background=CARD_COLOR,
            foreground=TEXT_COLOR,
            font=("Helvetica", 14, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background=CARD_COLOR,
            foreground=TEXT_COLOR,
            font=("Helvetica", 10),
        )
        style.configure(
            "Muted.TLabel",
            background=CARD_COLOR,
            foreground=MUTED_COLOR,
            font=("Helvetica", 10),
        )
        style.configure(
            "TButton",
            padding=(14, 10),
            font=("Helvetica", 10, "bold"),
            borderwidth=0,
        )
        style.map(
            "TButton",
            background=[("active", ACCENT_DARK), ("!disabled", ACCENT_COLOR)],
            foreground=[("!disabled", "white")],
        )
        style.configure(
            "Secondary.TButton",
            background="#e9eee9",
            foreground=TEXT_COLOR,
            bordercolor=BORDER_COLOR,
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#dde5df"), ("!disabled", "#e9eee9")],
            foreground=[("!disabled", TEXT_COLOR)],
        )
        style.configure(
            "Card.TCheckbutton",
            background=CARD_COLOR,
            foreground=TEXT_COLOR,
            font=("Helvetica", 10),
        )

    def _build_layout(self):
        container = ttk.Frame(self, padding=24, style="App.TFrame")
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=1)

        ttk.Label(container, text="Pipeline de Analisis", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            container,
            text="Interfaz simple para reutilizar Analizar, Limpiar, Graficas y Numeros sin duplicar lógica.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 18))

        top_row = ttk.Frame(container, style="App.TFrame")
        top_row.grid(row=2, column=0, sticky="nsew")
        top_row.columnconfigure(0, weight=3)
        top_row.columnconfigure(1, weight=2)

        self._build_folders_card(top_row).grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self._build_options_card(top_row).grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self._build_log_card(container).grid(row=3, column=0, sticky="nsew", pady=(18, 0))

        status_label = tk.Label(
            container,
            textvariable=self.status_var,
            bg=BG_COLOR,
            fg=MUTED_COLOR,
            anchor="w",
            font=("Helvetica", 10),
            pady=10,
        )
        status_label.grid(row=4, column=0, sticky="ew")

    def _build_folders_card(self, parent):
        frame = ttk.Frame(parent, padding=20, style="Card.TFrame")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)

        ttk.Label(frame, text="Carpetas a Analizar", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            frame,
            textvariable=self.base_dir_var,
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.grid(row=2, column=0, sticky="ew", pady=(0, 12))

        choose_base_button = ttk.Button(
            actions,
            text="Elegir carpeta base",
            command=self.choose_base_folder,
        )
        choose_base_button.grid(row=0, column=0, padx=(0, 10))
        self.action_buttons.append(choose_base_button)

        refresh_button = ttk.Button(actions, text="Actualizar", command=self.refresh_folders)
        refresh_button.grid(row=0, column=1, padx=(0, 10))
        self.action_buttons.append(refresh_button)

        select_all_button = ttk.Button(actions, text="Seleccionar todas", command=self.select_all_folders)
        select_all_button.grid(row=0, column=2, padx=(0, 10))
        self.action_buttons.append(select_all_button)

        clear_button = ttk.Button(
            actions,
            text="Limpiar seleccion",
            command=self.clear_folder_selection,
            style="Secondary.TButton",
        )
        clear_button.grid(row=0, column=3)
        self.action_buttons.append(clear_button)

        list_frame = ttk.Frame(frame, style="Card.TFrame")
        list_frame.grid(row=3, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.folder_list = tk.Listbox(
            list_frame,
            selectmode=tk.EXTENDED,
            bg="white",
            fg=TEXT_COLOR,
            selectbackground="#58708a",
            selectforeground="white",
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            font=("Helvetica", 11),
        )
        self.folder_list.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.folder_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.folder_list.configure(yscrollcommand=scrollbar.set)

        return frame

    def _build_options_card(self, parent):
        frame = ttk.Frame(parent, padding=20, style="Card.TFrame")
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Opciones", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            frame,
            text="Analizar mide solo las fotos nuevas y guarda root_lengths.csv en processed/ para cada experimento.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 16))

        ttk.Checkbutton(
            frame,
            text="Abrir calibracion de Analizar",
            variable=self.calibracion_var,
            style="Card.TCheckbutton",
        ).grid(row=2, column=0, sticky="w", pady=(0, 4))

        ttk.Checkbutton(
            frame,
            text="Reprocesar fotos ya medidas",
            variable=self.reprocesar_var,
            style="Card.TCheckbutton",
        ).grid(row=3, column=0, sticky="w", pady=(0, 14))

        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.grid(row=4, column=0, sticky="ew")

        analizar_button = ttk.Button(actions, text="Analizar", command=self.run_analizar)
        analizar_button.grid(row=0, column=0, padx=(0, 10), pady=(0, 10), sticky="ew")
        self.action_buttons.append(analizar_button)

        graficas_button = ttk.Button(actions, text="Graficas", command=self.run_graficas)
        graficas_button.grid(row=0, column=1, padx=(0, 10), pady=(0, 10), sticky="ew")
        self.action_buttons.append(graficas_button)

        numeros_button = ttk.Button(actions, text="Estadisticas", command=self.run_numeros)
        numeros_button.grid(row=0, column=2, pady=(0, 10), sticky="ew")
        self.action_buttons.append(numeros_button)

        return frame

    def _build_log_card(self, parent):
        frame = ttk.Frame(parent, padding=20, style="Card.TFrame")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        top = ttk.Frame(frame, style="Card.TFrame")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        top.columnconfigure(0, weight=1)

        ttk.Label(top, text="Log", style="Section.TLabel").grid(row=0, column=0, sticky="w")

        clear_log_button = ttk.Button(
            top,
            text="Limpiar log",
            command=self.clear_log,
            style="Secondary.TButton",
        )
        clear_log_button.grid(row=0, column=1, sticky="e")
        self.action_buttons.append(clear_log_button)

        self.log_widget = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            bg="white",
            fg=TEXT_COLOR,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            font=("Menlo", 11),
        )
        self.log_widget.grid(row=1, column=0, sticky="nsew")

        return frame

    def log(self, text):
        self.log_widget.insert(tk.END, text)
        self.log_widget.see(tk.END)
        self.log_widget.update_idletasks()

    def clear_log(self):
        self.log_widget.delete("1.0", tk.END)

    def get_base_directory(self):
        base_dir = self.base_dir_var.get().replace("Origen actual: ", "", 1).strip()
        return base_dir or DEFAULT_EXPERIMENTS_DIR

    def is_valid_experiment_folder(self, folder_name):
        try:
            datetime.strptime(folder_name, "%Y%m%d")
        except ValueError:
            return False
        return len(folder_name) == 8

    def get_experiment_directory(self, folder_name):
        return Path(self.get_base_directory()).expanduser().resolve() / folder_name

    def get_output_paths(self, folder_name):
        experiment_dir = self.get_experiment_directory(folder_name)
        output_dir = experiment_dir / "processed"
        plots_dir = experiment_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        return {
            "experiment_dir": experiment_dir,
            "output_dir": output_dir,
            "archivo_detallado": output_dir / "root_lengths.csv",
            "archivo_limpio": output_dir / "resultados_limpios.csv",
            "carpeta_testigos": output_dir / "procesadas_debug",
            "archivo_config": output_dir / "parametros_calibracion.json",
            "plots_dir": plots_dir,
            "trays_csv": experiment_dir / "trays.csv",
            "metadata_csv": Path(self.get_base_directory()).expanduser().resolve().parent
            / "experiments_metadata.csv",
        }

    def needs_legacy_results_refresh(self, rutas):
        archivo_limpio = rutas["archivo_limpio"]
        if not archivo_limpio.exists():
            return True

        fuentes = [rutas["archivo_detallado"], rutas["trays_csv"], rutas["metadata_csv"]]
        salida_mtime = archivo_limpio.stat().st_mtime
        return any(fuente.exists() and fuente.stat().st_mtime > salida_mtime for fuente in fuentes)

    def build_legacy_results_file(self, rutas):
        import pandas as pd

        archivo_detallado = rutas["archivo_detallado"]
        if not archivo_detallado.exists():
            raise FileNotFoundError(
                f"No existe {archivo_detallado}. Ejecutá Analizar primero para generar root_lengths.csv."
            )

        df = pd.read_csv(archivo_detallado)
        columnas_requeridas = {"tray_id", "seed_id", "length_mm", "area_cm2"}
        faltantes = columnas_requeridas - set(df.columns)
        if faltantes:
            raise ValueError(
                f"{archivo_detallado} no tiene las columnas esperadas: "
                + ", ".join(sorted(faltantes))
            )

        df = df.copy()
        df["tray_id"] = df["tray_id"].astype(str).str.strip()
        df["seed_id"] = df["seed_id"].astype(str).str.strip()
        df["length_mm"] = pd.to_numeric(df["length_mm"], errors="coerce")
        df["area_cm2"] = pd.to_numeric(df["area_cm2"], errors="coerce")
        df["Replica"] = df["tray_id"].str.extract(r"^[^_]+_([^_]+)", expand=False)
        df["Replica"] = df["Replica"].fillna(df["tray_id"])

        df["Grupo"] = "Control"
        if rutas["trays_csv"].exists():
            trays = pd.read_csv(rutas["trays_csv"])
            if {"tray_id", "treatment_floor"}.issubset(trays.columns):
                trays = trays[["tray_id", "treatment_floor"]].copy()
                trays["tray_id"] = trays["tray_id"].astype(str).str.strip()
                trays["treatment_floor"] = pd.to_numeric(
                    trays["treatment_floor"].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce",
                )
                df = df.merge(trays, on="tray_id", how="left")

                active_magnet = True
                if rutas["metadata_csv"].exists():
                    metadata = pd.read_csv(rutas["metadata_csv"], dtype=str)
                    exp_id = rutas["experiment_dir"].name
                    metadata_exp = metadata[
                        metadata.get("exp_id", "").astype(str).str.strip() == exp_id
                    ]
                    if not metadata_exp.empty:
                        experiment_type = metadata_experiment_type(metadata_exp.iloc[0])
                        active_magnet = bool(
                            experiment_type
                            and experiment_type not in NON_MAGNETIC_EXPERIMENT_TYPES
                        )

                if active_magnet:
                    df["Grupo"] = df["treatment_floor"].apply(
                        lambda value: "Control" if value == 9 else "Magnetico"
                    )

        largo = df[["Grupo", "Replica", "seed_id", "length_mm"]].copy()
        largo["Tipo_Objeto"] = DEFAULT_OBJETO
        largo["Medida_Tipo"] = "Largo_cm"
        largo["Medida_Valor"] = largo["length_mm"] / 10.0

        area = df[["Grupo", "Replica", "seed_id", "area_cm2"]].copy()
        area["Tipo_Objeto"] = DEFAULT_OBJETO
        area["Medida_Tipo"] = "Area_cm2"
        area["Medida_Valor"] = area["area_cm2"]

        legacy = pd.concat(
            [
                largo[["Grupo", "Tipo_Objeto", "Medida_Tipo", "Medida_Valor", "Replica", "seed_id"]],
                area[["Grupo", "Tipo_Objeto", "Medida_Tipo", "Medida_Valor", "Replica", "seed_id"]],
            ],
            ignore_index=True,
        )
        legacy = legacy.dropna(subset=["Medida_Valor"])
        legacy.to_csv(rutas["archivo_limpio"], index=False)
        print(
            f"[OK] Archivo compatible generado: {rutas['archivo_limpio']} "
            f"({len(legacy)} mediciones)."
        )
        return rutas["archivo_limpio"]

    def ensure_legacy_results_file(self, rutas):
        if self.needs_legacy_results_refresh(rutas):
            return self.build_legacy_results_file(rutas)
        return rutas["archivo_limpio"]

    def choose_base_folder(self):
        selected_dir = filedialog.askdirectory(
            title="Seleccionar carpeta base de experimentos",
            initialdir=str(Path(self.get_base_directory()).expanduser().resolve()),
        )
        if not selected_dir:
            return

        self.base_dir_var.set(f"Origen actual: {selected_dir}")
        self.refresh_folders(select_all=True)

    def refresh_folders(self, select_all=False):
        directorio_raiz = self.get_base_directory()

        try:
            analizar = self.get_module("Analizar")
            carpetas = [
                carpeta
                for carpeta in analizar.obtener_carpetas_disponibles(directorio_raiz)
                if self.is_valid_experiment_folder(carpeta)
            ]
        except Exception as exc:
            self.folder_list.delete(0, tk.END)
            self.status_var.set(f"Error leyendo carpetas en {directorio_raiz}.")
            messagebox.showerror("Error", f"No se pudieron cargar las carpetas:\n{exc}")
            return

        self.folder_list.delete(0, tk.END)
        for carpeta in carpetas:
            self.folder_list.insert(tk.END, carpeta)

        if select_all and carpetas:
            self.folder_list.selection_set(0, tk.END)

        if carpetas:
            self.status_var.set(f"{len(carpetas)} carpetas disponibles en {directorio_raiz}.")
        else:
            self.status_var.set(f"No se encontraron carpetas en {directorio_raiz}.")

    def select_all_folders(self):
        if self.folder_list.size() > 0:
            self.folder_list.selection_set(0, tk.END)

    def clear_folder_selection(self):
        self.folder_list.selection_clear(0, tk.END)

    def selected_folders(self):
        return [self.folder_list.get(i) for i in self.folder_list.curselection()]

    def require_single_selected_folder(self, action_name):
        carpetas = self.selected_folders()
        if not carpetas:
            messagebox.showwarning("Sin carpeta", f"Seleccioná una carpeta YYYYMMDD para {action_name}.")
            return None
        if len(carpetas) > 1:
            messagebox.showwarning(
                "Selección múltiple",
                f"Para {action_name} tenés que seleccionar una sola carpeta YYYYMMDD.",
            )
            return None
        return carpetas[0]

    def parse_iqr(self):
        try:
            return float(self.iqr_var.get().strip())
        except ValueError as exc:
            raise ValueError("El factor IQR configurado no es válido.") from exc

    def set_busy(self, busy):
        state = tk.DISABLED if busy else tk.NORMAL
        for button in self.action_buttons:
            button.configure(state=state)
        self.configure(cursor="watch" if busy else "")
        self.update_idletasks()

    def execute_with_log(self, title, callback):
        self.log(f"\n=== {title} ===\n")
        self.status_var.set(f"Ejecutando: {title}...")
        self.set_busy(True)
        writer = TextWidgetWriter(self.log_widget)

        try:
            with redirect_stdout(writer), redirect_stderr(writer):
                callback()
        except Exception as exc:
            traceback.print_exc(file=writer)
            self.status_var.set(f"Falló: {title}")
            messagebox.showerror("Error", str(exc))
        else:
            self.status_var.set(f"Terminado: {title}")
        finally:
            self.set_busy(False)

    def run_analizar(self):
        carpetas = self.selected_folders()
        if not carpetas:
            messagebox.showwarning("Sin carpetas", "Seleccioná al menos una carpeta para analizar.")
            return

        analizar = self.get_module("Analizar")

        def procesar_seleccion():
            for carpeta in carpetas:
                rutas = self.get_output_paths(carpeta)
                print(f"\n--- Experimento {carpeta} ---")
                print(f"Guardando resultados en: {rutas['output_dir']}")

                resultado = analizar.procesar_carpetas(
                    [carpeta],
                    directorio_raiz=self.get_base_directory(),
                    archivo_salida=str(rutas["archivo_detallado"]),
                    carpeta_testigos=str(rutas["carpeta_testigos"]),
                    abrir_calibracion=self.calibracion_var.get(),
                    archivo_config=str(rutas["archivo_config"]),
                    reprocesar=self.reprocesar_var.get(),
                )

                if resultado is None:
                    print(f"[ALERTA] Se omitió {carpeta} porque no generó resultados analizables.")
                    continue

        self.execute_with_log(
            "Analizar fotos",
            procesar_seleccion,
        )

    def run_graficas(self):
        carpetas = self.selected_folders()
        if not carpetas:
            messagebox.showwarning("Sin carpetas", "Seleccioná al menos una carpeta para graficar.")
            return

        graficas = self.get_module("Graficas")
        archivos_multiples = []
        rutas_por_carpeta = []
        
        for carpeta in carpetas:
            rutas = self.get_output_paths(carpeta)
            if not rutas["archivo_limpio"].exists() and not rutas["archivo_detallado"].exists():
                messagebox.showwarning(
                    "Falta análisis",
                    f"No existe root_lengths.csv para {carpeta}.\nPrimero ejecutá Analizar para esta fecha.",
                )
                return
            rutas_por_carpeta.append(rutas)
            archivos_multiples.append((carpeta, str(rutas["archivo_limpio"])))

        def _ejecutar():
            for rutas in rutas_por_carpeta:
                self.ensure_legacy_results_file(rutas)
            graficas.graficar_datos(
                objeto_a_graficar=self.objeto_var.get(),
                tipo_medida=self.medida_var.get(),
                archivos_multiples=archivos_multiples,
            )

        self.execute_with_log(
            "Graficas", _ejecutar
        )

    def run_numeros(self):
        carpeta = self.require_single_selected_folder("estadísticas")
        if carpeta is None:
            return

        numeros = self.get_module("Numeros")
        rutas = self.get_output_paths(carpeta)
        if not rutas["archivo_limpio"].exists() and not rutas["archivo_detallado"].exists():
            messagebox.showwarning(
                "Falta análisis",
                f"No existe root_lengths.csv para {carpeta}.\nPrimero ejecutá Analizar para esta fecha.",
            )
            return

        self.execute_with_log(
            "Estadisticas",
            lambda: (
                self.ensure_legacy_results_file(rutas),
                numeros.analizar_estadisticas(
                    archivo=str(rutas["archivo_limpio"]),
                    tipo_objeto=self.objeto_var.get(),
                    tipo_medida=self.medida_var.get(),
                ),
            ),
        )

def main():
    app = PipelineGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
