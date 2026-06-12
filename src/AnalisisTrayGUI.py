### TODO: Esto es solo provisorio, despues hay que analizar bien que poner

import io
import os
import subprocess
import sys
import tkinter as tk
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Optional

import AnalisisTrays
import pandas as pd

BG_COLOR = "#f3eee6"
CARD_COLOR = "#fffaf3"
ACCENT_COLOR = "#2f6b5b"
ACCENT_DARK = "#234d42"
TEXT_COLOR = "#1f2a25"
MUTED_COLOR = "#63706b"
BORDER_COLOR = "#d7d0c6"


class TextWidgetWriter(io.StringIO):
    def __init__(self, widget: scrolledtext.ScrolledText):
        super().__init__()
        self.widget = widget

    def write(self, text: str) -> int:
        if not text:
            return 0
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.update_idletasks()
        return len(text)

    def flush(self) -> None:
        self.widget.update_idletasks()


class AnalisisTrayGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Analisis de Trays")
        self.geometry("1000x700")
        self.minsize(900, 600)
        self.configure(bg=BG_COLOR)

        self.status_var = tk.StringVar(value="Listo para analizar trays.csv.")
        self.base_dir_var = tk.StringVar(
            value=f"Experimentos: {AnalisisTrays.EXPERIMENTS_DIR}"
        )
        self.selected_exp_var = tk.StringVar(value="Experimentos seleccionados: -")
        self.global_dir_var = tk.StringVar(
            value=f"Consolidado global: {AnalisisTrays.GLOBAL_ANALYSIS_DIR}"
        )
        self.action_buttons: list[ttk.Button] = []
        self.last_result = None

        self._configure_styles()
        self._build_layout()
        self.after(50, self.refresh_experiments)

    def _configure_styles(self) -> None:
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

    def _build_layout(self) -> None:
        container = ttk.Frame(self, padding=24, style="App.TFrame")
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)  # top_row
        container.rowconfigure(3, weight=1)  # bottom_row for summary and log

        ttk.Label(
            container,
            text="Analisis Estadistico de Trays",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            container,
            text=(
                "Interfaz para correr AnalisisTrays.py, revisar el resumen del modelo "
                "y abrir rapidamente processed, plots y el consolidado global."
            ),
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 18))

        top_row = ttk.Frame(container, style="App.TFrame")
        top_row.grid(row=2, column=0, sticky="nsew")
        top_row.columnconfigure(0, weight=3)
        top_row.columnconfigure(1, weight=2)
        top_row.rowconfigure(0, weight=1)

        self._build_selection_card(top_row).grid(
            row=0, column=0, sticky="nsew", padx=(0, 10)
        )
        self._build_actions_card(top_row).grid(
            row=0, column=1, sticky="nsew", padx=(10, 0)
        )

        # Nuevo frame para los paneles de resumen y log, uno al lado del otro
        bottom_row = ttk.Frame(container, style="App.TFrame")
        bottom_row.grid(row=3, column=0, sticky="nsew", pady=(18, 0))
        bottom_row.columnconfigure(0, weight=1)  # Columna para el resumen
        bottom_row.columnconfigure(1, weight=1)  # Columna para el log
        bottom_row.rowconfigure(0, weight=1)  # Ambos paneles se expanden verticalmente

        self._build_summary_card(bottom_row).grid(
            row=0, column=0, sticky="nsew", padx=(0, 10)
        )
        self._build_log_card(bottom_row).grid(
            row=0, column=1, sticky="nsew", padx=(10, 0)
        )

        # El status_label se mueve una fila hacia arriba
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

    def _build_selection_card(self, parent):
        frame = ttk.Frame(parent, padding=20, style="Card.TFrame")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)

        ttk.Label(frame, text="Experimentos", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(frame, textvariable=self.base_dir_var, style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(4, 12)
        )

        actions = ttk.Frame(frame, style="Card.TFrame")
        actions.grid(row=2, column=0, sticky="ew", pady=(0, 12))

        refresh_button = ttk.Button(
            actions, text="Actualizar lista", command=self.refresh_experiments
        )
        refresh_button.grid(row=0, column=0, padx=(0, 10))
        self.action_buttons.append(refresh_button)

        open_base_button = ttk.Button(
            actions,
            text="Abrir carpeta base",
            command=lambda: self.open_path(AnalisisTrays.EXPERIMENTS_DIR),
            style="Secondary.TButton",
        )
        open_base_button.grid(row=0, column=1)
        self.action_buttons.append(open_base_button)

        list_frame = ttk.Frame(frame, style="Card.TFrame")
        list_frame.grid(row=3, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.experiment_list = tk.Listbox(
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
        self.experiment_list.grid(row=0, column=0, sticky="nsew")
        self.experiment_list.bind("<<ListboxSelect>>", self.on_selection_changed)

        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.experiment_list.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.experiment_list.configure(yscrollcommand=scrollbar.set)

        return frame

    def _build_actions_card(self, parent):
        outer_frame = ttk.Frame(parent, padding=20, style="Card.TFrame")
        outer_frame.columnconfigure(0, weight=1)
        outer_frame.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer_frame, bg=CARD_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(10, 0))

        frame = ttk.Frame(canvas, style="Card.TFrame")
        canvas_window = canvas.create_window((0, 0), window=frame, anchor="nw")

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)

        frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        def _on_mousewheel(event):
            num = getattr(event, "num", 0)
            delta = getattr(event, "delta", 0)

            if sys.platform == "darwin":
                scroll_amount = -1 * delta
            else:
                scroll_amount = int(-1 * (delta / 120)) if delta else 0

            if num == 4:
                canvas.yview_scroll(-1, "units")
            elif num == 5:
                canvas.yview_scroll(1, "units")
            elif scroll_amount:
                canvas.yview_scroll(scroll_amount, "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        outer_frame.bind("<Enter>", _bind_mousewheel)
        outer_frame.bind("<Leave>", _unbind_mousewheel)

        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Acciones", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            frame,
            text=(
                "Podés seleccionar una o varias fechas con Shift/Cmd. "
                "El backend guarda CSVs en processed/ y figuras en plots/."
            ),
            style="Muted.TLabel",
            wraplength=250,
        ).grid(row=1, column=0, sticky="w", pady=(4, 16))

        analyze_selected_button = ttk.Button(
            frame,
            text="Analizar seleccionados",
            command=self.run_selected_analysis,
        )
        analyze_selected_button.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.action_buttons.append(analyze_selected_button)

        analyze_all_button = ttk.Button(
            frame,
            text="Analizar todos",
            command=self.run_all_analysis,
        )
        analyze_all_button.grid(row=3, column=0, sticky="ew", pady=(0, 18))
        self.action_buttons.append(analyze_all_button)

        negative_control_button = ttk.Button(
            frame,
            text="Control negativo combinado",
            command=self.run_negative_control_analysis,
        )
        negative_control_button.grid(row=4, column=0, sticky="ew", pady=(0, 18))
        self.action_buttons.append(negative_control_button)

        combine_trays_button = ttk.Button(
            frame,
            text="Crear tray_combined.csv",
            command=self.create_selected_tray_combined,
        )
        combine_trays_button.grid(row=5, column=0, sticky="ew", pady=(0, 18))
        self.action_buttons.append(combine_trays_button)

        ttk.Label(frame, textvariable=self.selected_exp_var, style="Body.TLabel").grid(
            row=6, column=0, sticky="w"
        )
        ttk.Label(
            frame,
            textvariable=self.global_dir_var,
            style="Muted.TLabel",
            wraplength=250,
        ).grid(row=7, column=0, sticky="w", pady=(4, 14))

        open_processed_button = ttk.Button(
            frame,
            text="Abrir processed",
            command=self.open_selected_processed,
            style="Secondary.TButton",
        )
        open_processed_button.grid(row=8, column=0, sticky="ew", pady=(0, 10))
        self.action_buttons.append(open_processed_button)

        open_plots_button = ttk.Button(
            frame,
            text="Abrir plots",
            command=self.open_selected_plots,
            style="Secondary.TButton",
        )
        open_plots_button.grid(row=9, column=0, sticky="ew", pady=(0, 10))
        self.action_buttons.append(open_plots_button)

        open_global_button = ttk.Button(
            frame,
            text="Abrir analisis global",
            command=lambda: self.open_path(AnalisisTrays.GLOBAL_ANALYSIS_DIR),
            style="Secondary.TButton",
        )
        open_global_button.grid(row=10, column=0, sticky="ew")
        self.action_buttons.append(open_global_button)

        return outer_frame

    def _build_summary_card(self, parent):
        frame = ttk.Frame(parent, padding=20, style="Card.TFrame")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Resumen", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        self.summary_widget = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            bg="white",
            fg=TEXT_COLOR,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            font=("Menlo", 11),
        )
        self.summary_widget.grid(row=1, column=0, sticky="nsew")
        self.summary_widget.insert(
            "1.0",
            "Todavia no hay resultados cargados.\n"
            "Elegí una fecha y ejecutá el análisis para ver el resumen acá.",
        )
        self.summary_widget.configure(state=tk.DISABLED)
        return frame

    def _build_log_card(self, parent):
        frame = ttk.Frame(parent, padding=20, style="Card.TFrame")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        top = ttk.Frame(frame, style="Card.TFrame")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        top.columnconfigure(0, weight=1)

        ttk.Label(top, text="Log", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )

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

    def clear_log(self) -> None:
        self.log_widget.delete("1.0", tk.END)

    def log(self, text: str) -> None:
        self.log_widget.insert(tk.END, text)
        self.log_widget.see(tk.END)
        self.log_widget.update_idletasks()

    def _format_p_value(self, valor) -> str:
        if self._es_nan(valor):
            return "n/d"
        if valor < 0.001:
            return "<0.001"
        return f"{valor:.3f}"

    def _format_value(self, valor, decimales: int = 3) -> str:
        if self._es_nan(valor):
            return "n/d"
        return f"{valor:.{decimales}f}"

    def _format_coef_and_p(self, coef, p_value) -> str:
        if self._es_nan(coef) or self._es_nan(p_value):
            return "no disponible"
        return f"coef {coef:.3f} | p {self._format_p_value(p_value)}"

    def _describe_spatial_effect(self, resumen: dict) -> str:
        disponibles = [
            valor
            for valor in (
                resumen["floor_delta_p_value"],
                resumen["position_delta_p_value"],
            )
            if not self._es_nan(valor)
        ]
        if not disponibles:
            return "sin test disponible"

        minimo = min(disponibles)
        if minimo < 0.05:
            return "hay señal espacial"
        if minimo < 0.10:
            return "hay una señal espacial débil"
        return "sin señal espacial clara"

    def set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        for button in self.action_buttons:
            button.configure(state=state)
        self.configure(cursor="watch" if busy else "")
        self.update_idletasks()

    def execute_with_log(self, title: str, callback):
        separador = "=" * 72
        self.log(f"\n{separador}\n{title}\n{separador}\n")
        self.status_var.set(f"Ejecutando: {title}...")
        self.set_busy(True)
        writer = TextWidgetWriter(self.log_widget)

        try:
            with redirect_stdout(writer), redirect_stderr(writer):
                result = callback()
        except Exception as exc:
            traceback.print_exc(file=writer)
            self.log(f"\nError durante: {title}\n")
            self.status_var.set(f"Falló: {title}")
            messagebox.showerror("Error", str(exc))
            return None
        else:
            self.log(f"\nListo: {title}\n")
            self.status_var.set(f"Terminado: {title}")
            return result
        finally:
            self.set_busy(False)

    def get_available_experiments(self) -> list[str]:
        try:
            return [ruta.parent.name for ruta in AnalisisTrays.descubrir_trays()]
        except FileNotFoundError:
            return []

    def refresh_experiments(self) -> None:
        previos = set(self.get_selected_experiments())
        experimentos = self.get_available_experiments()

        self.experiment_list.delete(0, tk.END)
        for exp_id in experimentos:
            self.experiment_list.insert(tk.END, exp_id)

        if experimentos:
            objetivos = [exp_id for exp_id in experimentos if exp_id in previos]
            if not objetivos:
                objetivos = [experimentos[0]]
            for objetivo in objetivos:
                indice = experimentos.index(objetivo)
                self.experiment_list.selection_set(indice)
            indice_activo = experimentos.index(objetivos[0])
            self.experiment_list.activate(indice_activo)
            self.experiment_list.see(indice_activo)
            self.update_selected_label()
            self.status_var.set(
                f"{len(experimentos)} experimentos listos para analizar."
            )
        else:
            self.selected_exp_var.set("Experimentos seleccionados: -")
            self.status_var.set("No se encontraron trays.csv en data/experiments.")

    def on_selection_changed(self, _event=None) -> None:
        self.update_selected_label()

    def update_selected_label(self) -> None:
        exp_ids = self.get_selected_experiments()
        if not exp_ids:
            self.selected_exp_var.set("Experimentos seleccionados: -")
        elif len(exp_ids) == 1:
            self.selected_exp_var.set(f"Experimento seleccionado: {exp_ids[0]}")
        else:
            self.selected_exp_var.set(
                f"Experimentos seleccionados ({len(exp_ids)}): {', '.join(exp_ids)}"
            )

    def get_selected_experiments(self) -> list[str]:
        return [
            self.experiment_list.get(i) for i in self.experiment_list.curselection()
        ]

    def get_selected_experiment(self) -> Optional[str]:
        seleccion = self.get_selected_experiments()
        if not seleccion:
            return None
        return seleccion[0]

    def require_selected_experiment(self, action_name: str) -> Optional[str]:
        exp_ids = self.get_selected_experiments()
        if not exp_ids:
            messagebox.showwarning(
                "Sin experimento",
                f"Seleccioná un experimento YYYYMMDD para {action_name}.",
            )
            return None
        if len(exp_ids) > 1:
            messagebox.showwarning(
                "Demasiados experimentos",
                f"Para {action_name} tenés que seleccionar una sola fecha.",
            )
            return None
        return exp_ids[0]

    def require_selected_experiments(
        self,
        action_name: str,
        *,
        minimum: int = 1,
    ) -> list[str] | None:
        exp_ids = self.get_selected_experiments()
        if len(exp_ids) < minimum:
            messagebox.showwarning(
                "Selección insuficiente",
                f"Seleccioná al menos {minimum} experimentos para {action_name}.",
            )
            return None
        return exp_ids

    def get_experiment_paths(self, exp_id: str) -> dict[str, Path]:
        exp_dir = AnalisisTrays.EXPERIMENTS_DIR / exp_id
        return {
            "exp_dir": exp_dir,
            "processed_dir": exp_dir / "processed",
            "plots_dir": exp_dir / "plots",
            "global_dir": AnalisisTrays.GLOBAL_ANALYSIS_DIR,
        }

    def open_path(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            messagebox.showwarning(
                "No existe",
                f"La ruta todavía no existe:\n{path}",
            )
            return

        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", str(path)])
            elif os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("No se pudo abrir", f"{path}\n\n{exc}")

    def open_selected_processed(self) -> None:
        exp_id = self.require_selected_experiment("abrir processed")
        if exp_id is None:
            return
        self.open_path(self.get_experiment_paths(exp_id)["processed_dir"])

    def open_selected_plots(self) -> None:
        exp_id = self.require_selected_experiment("abrir plots")
        if exp_id is None:
            return
        self.open_path(self.get_experiment_paths(exp_id)["plots_dir"])

    def render_summary(self, result: dict) -> None:
        lineas = [
            f"Experimentos analizados: {len(result['experiment_summaries'])}",
            f"Consolidado global: {result['global_analysis_dir']}",
            "",
        ]

        for resumen in result["experiment_summaries"]:
            comparacion_disponible = any(
                not self._es_nan(valor)
                for valor in (
                    resumen["magnetic_vs_control_delta_p_value"],
                    resumen["magnetic_vs_control_pct_p_value"],
                    resumen["magnetic_vs_control_log_p_value"],
                )
            )

            lineas.extend(
                [
                    f"{resumen['exp_id']} | {resumen['experiment_kind_label']}",
                    (
                        "Datos: "
                        f"{resumen['total_rows']} trays | "
                        f"delta {resumen['valid_delta_rows']}/{resumen['total_rows']} | "
                        f"pct {resumen['valid_pct_rows']}/{resumen['total_rows']} | "
                        f"log {resumen['valid_log_rows']}/{resumen['total_rows']}"
                    ),
                    (
                        "Grupos: "
                        f"magnetic={resumen['magnetic_rows']} | "
                        f"control={resumen['control_rows']} | "
                        f"metadata={'completa' if resumen['has_complete_metadata'] else 'incompleta'}"
                    ),
                    (
                        "Referencia: "
                        f"delta_control={self._format_value(resumen['control_delta_mean'])} "
                        f"(n={resumen['control_delta_count']}) | "
                        f"log_control={self._format_value(resumen['control_log_ratio_mean'])} "
                        f"(n={resumen['control_log_ratio_count']})"
                    ),
                    (
                        "Intensidad magnética: "
                        f"delta {self._format_coef_and_p(resumen['magnetic_intensity_delta_coef'], resumen['magnetic_intensity_delta_p_value'])} | "
                        f"pct {self._format_coef_and_p(resumen['magnetic_intensity_pct_coef'], resumen['magnetic_intensity_pct_p_value'])} | "
                        f"log {self._format_coef_and_p(resumen['magnetic_intensity_log_coef'], resumen['magnetic_intensity_log_p_value'])}"
                    ),
                    (
                        "Magnetic vs control: "
                        f"delta p={self._format_p_value(resumen['magnetic_vs_control_delta_p_value'])} | "
                        f"pct p={self._format_p_value(resumen['magnetic_vs_control_pct_p_value'])} | "
                        f"log p={self._format_p_value(resumen['magnetic_vs_control_log_p_value'])}"
                        if comparacion_disponible
                        else "Magnetic vs control: no disponible"
                    ),
                    (
                        "Efecto espacial (delta): "
                        f"floor p={self._format_p_value(resumen['floor_delta_p_value'])} | "
                        f"position p={self._format_p_value(resumen['position_delta_p_value'])} | "
                        f"{self._describe_spatial_effect(resumen)}"
                    ),
                    f"Archivos: processed={resumen['processed_dir']}",
                    f"          plots={resumen['plots_dir']}",
                ]
            )

        combined_plot_path = result.get("combined_plot_path")
        if combined_plot_path:
            lineas.extend([
                "",
                f"Plot combinado: {combined_plot_path}",
            ])

        lineas.append("")

        if result["warnings"]:
            lineas.append("Advertencias:")
            for advertencia in result["warnings"]:
                lineas.append(f"- {advertencia}")
            lineas.append("")

        self.summary_widget.configure(state=tk.NORMAL)
        self.summary_widget.delete("1.0", tk.END)
        self.summary_widget.insert("1.0", "\n".join(lineas))
        self.summary_widget.configure(state=tk.DISABLED)

    def _describe_negative_control_signal(self, p_values: list[float]) -> str:
        disponibles = [valor for valor in p_values if not self._es_nan(valor)]
        if not disponibles:
            return "sin test disponible"
        minimo = min(disponibles)
        if minimo < 0.05:
            return "hay diferencias detectables"
        if minimo < 0.10:
            return "hay una señal débil"
        return "sin diferencias claras"

    def render_negative_control_summary(self, result: dict) -> None:
        resumen = result["combined_summary"]
        plot_paths = result.get("plot_paths") or {}
        assumption_tests = result.get("combined_assumption_tests")
        floor_tests = result.get("combined_floor_tests")
        position_group_tests = result.get("combined_position_group_tests")
        tray_cell_tests = result.get("combined_tray_cell_tests")
        day_p_values = [
            resumen["day_raw_delta_p_value"],
            resumen["day_pct_p_value"],
            resumen["day_log_p_value"],
        ]
        spatial_p_values = [
            resumen["floor_raw_delta_p_value"],
            resumen["position_raw_delta_p_value"],
            resumen["floor_pct_p_value"],
            resumen["position_pct_p_value"],
            resumen["floor_log_p_value"],
            resumen["position_log_p_value"],
        ]
        mean_p_values = [
            resumen["raw_delta_mean_p_value"],
            resumen["pct_change_mean_p_value"],
            resumen["raw_log_ratio_mean_p_value"],
        ]

        lineas = [
            "Control negativo combinado",
            f"Experimentos: {', '.join(resumen['exp_ids'])}",
            f"Salida: {result['output_dir']}",
            f"Supuestos: {result.get('assumption_tests_path') or 'no generado'}",
            f"Comparación A/B/C: {result.get('floor_tests_path') or 'no generado'}",
            (
                "Comparación posiciones agrupadas: "
                f"{result.get('position_group_tests_path') or 'no generado'}"
            ),
            f"Comparación A1-C9: {result.get('tray_cell_tests_path') or 'no generado'}",
            f"Tests por panel: {result.get('facet_tests_path') or 'no generado'}",
            "Plots:",
            (
                "- piso/dia: "
                f"{plot_paths.get('floor_day') or result['plot_path'] or 'no generado'}"
            ),
            (
                "- barras (media ± SD): "
                f"{plot_paths.get('barplot_exp') or 'no generado'}"
            ),
            (
                "- A/B/C raw_delta media ± SD: "
                f"{plot_paths.get('floor_barplot_raw_delta') or 'no generado'}"
            ),
            (
                "- A/B/C pct media ± SD: "
                f"{plot_paths.get('floor_barplot_pct') or 'no generado'}"
            ),
            (
                "- A/B/C log media ± SD: "
                f"{plot_paths.get('floor_barplot_log') or 'no generado'}"
            ),
            (
                "- izquierda/centro/derecha raw_delta: "
                f"{plot_paths.get('left_center_right_barplot_raw_delta') or 'no generado'}"
            ),
            (
                "- izquierda/centro/derecha pct: "
                f"{plot_paths.get('left_center_right_barplot_pct') or 'no generado'}"
            ),
            (
                "- izquierda/centro/derecha log: "
                f"{plot_paths.get('left_center_right_barplot_log') or 'no generado'}"
            ),
            (
                "- frente/medio/atras raw_delta: "
                f"{plot_paths.get('front_middle_back_barplot_raw_delta') or 'no generado'}"
            ),
            (
                "- frente/medio/atras pct: "
                f"{plot_paths.get('front_middle_back_barplot_pct') or 'no generado'}"
            ),
            (
                "- frente/medio/atras log: "
                f"{plot_paths.get('front_middle_back_barplot_log') or 'no generado'}"
            ),
            (
                "- A1-C9 raw_delta: "
                f"{plot_paths.get('tray_cell_barplot_raw_delta') or 'no generado'}"
            ),
            (
                "- A1-C9 pct: "
                f"{plot_paths.get('tray_cell_barplot_pct') or 'no generado'}"
            ),
            (
                "- A1-C9 log: "
                f"{plot_paths.get('tray_cell_barplot_log') or 'no generado'}"
            ),
            f"- A/B/C por fecha: {plot_paths.get('floor_facets') or 'no generado'}",
            (
                "- posiciones 1-9 por fecha: "
                f"{plot_paths.get('position_facets') or 'no generado'}"
            ),
            "",
            (
                "Datos: "
                f"{resumen['total_rows']} trays | "
                f"delta {resumen['valid_delta_rows']}/{resumen['total_rows']} | "
                f"pct {resumen['valid_pct_rows']}/{resumen['total_rows']} | "
                f"log {resumen['valid_log_rows']}/{resumen['total_rows']}"
            ),
            (
                "Promedios vs 0: "
                f"delta_raw={self._format_value(resumen['raw_delta_mean'])} "
                f"(p={self._format_p_value(resumen['raw_delta_mean_p_value'])}) | "
                f"pct={self._format_value(resumen['pct_change_mean'])} "
                f"(p={self._format_p_value(resumen['pct_change_mean_p_value'])}) | "
                f"log_raw={self._format_value(resumen['raw_log_ratio_mean'])} "
                f"(p={self._format_p_value(resumen['raw_log_ratio_mean_p_value'])})"
            ),
        ]

        if assumption_tests is not None and not assumption_tests.empty:
            lineas.append("Supuestos (Shapiro residuos | Brown-Forsythe días):")
            for _, fila in assumption_tests.iterrows():
                lineas.append(
                    "- "
                    f"{fila['response_metric']}: "
                    f"W={self._format_value(fila['shapiro_w'], 4)}, "
                    f"p={self._format_p_value(fila['shapiro_p_value'])} | "
                    f"F={self._format_value(fila['brown_forsythe_statistic'], 4)}, "
                    f"p={self._format_p_value(fila['brown_forsythe_p_value'])}"
                )

        if floor_tests is not None and not floor_tests.empty:
            lineas.append("Comparación pisos A/B/C ajustada por día (HC3):")
            for _, fila in floor_tests.iterrows():
                lineas.append(
                    "- "
                    f"{fila['response_metric']}: "
                    f"F={self._format_value(fila['statistic'], 4)}, "
                    f"p={self._format_p_value(fila['p_value'])}"
                )

        if position_group_tests is not None and not position_group_tests.empty:
            lineas.append("Comparación posiciones agrupadas ajustada por día (HC3):")
            for _, fila in position_group_tests.iterrows():
                lineas.append(
                    "- "
                    f"{fila['comparison']} / {fila['response_metric']}: "
                    f"F={self._format_value(fila['statistic'], 4)}, "
                    f"p={self._format_p_value(fila['p_value'])}"
                )

        if tray_cell_tests is not None and not tray_cell_tests.empty:
            lineas.append("Comparación A1-C9 ajustada por día (HC3):")
            for _, fila in tray_cell_tests.iterrows():
                lineas.append(
                    "- "
                    f"{fila['response_metric']}: "
                    f"F={self._format_value(fila['statistic'], 4)}, "
                    f"p={self._format_p_value(fila['p_value'])}"
                )

        lineas.extend(
            [
            (
                "Efecto día: "
                f"delta p={self._format_p_value(resumen['day_raw_delta_p_value'])} | "
                f"pct p={self._format_p_value(resumen['day_pct_p_value'])} | "
                f"log p={self._format_p_value(resumen['day_log_p_value'])} | "
                f"{self._describe_negative_control_signal(day_p_values)}"
            ),
            (
                "Efecto espacial delta_raw: "
                f"floor p={self._format_p_value(resumen['floor_raw_delta_p_value'])} | "
                f"position p={self._format_p_value(resumen['position_raw_delta_p_value'])}"
            ),
            (
                "Efecto espacial pct: "
                f"floor p={self._format_p_value(resumen['floor_pct_p_value'])} | "
                f"position p={self._format_p_value(resumen['position_pct_p_value'])}"
            ),
            (
                "Efecto espacial log_raw: "
                f"floor p={self._format_p_value(resumen['floor_log_p_value'])} | "
                f"position p={self._format_p_value(resumen['position_log_p_value'])} | "
                f"{self._describe_negative_control_signal(spatial_p_values)}"
            ),
            (
                "Lectura rápida: "
                f"promedios {self._describe_negative_control_signal(mean_p_values)}, "
                f"días {self._describe_negative_control_signal(day_p_values)}, "
                f"espacio {self._describe_negative_control_signal(spatial_p_values)}."
            ),
            "",
            ]
        )

        if result["warnings"]:
            lineas.append("Advertencias:")
            for advertencia in result["warnings"]:
                lineas.append(f"- {advertencia}")
            lineas.append("")

        self.summary_widget.configure(state=tk.NORMAL)
        self.summary_widget.delete("1.0", tk.END)
        self.summary_widget.insert("1.0", "\n".join(lineas))
        self.summary_widget.configure(state=tk.DISABLED)

    def render_tray_combined_summary(self, result: dict) -> None:
        lineas = [
            "Tray combinado",
            f"Experimentos: {', '.join(result['exp_ids'])}",
            f"Salida: {result['output_path']}",
            f"Filas totales: {result['total_rows']}",
            "",
            "Filas por experimento:",
        ]
        for exp_id, rows in result["rows_by_exp_id"].items():
            lineas.append(f"- {exp_id}: {rows}")

        self.summary_widget.configure(state=tk.NORMAL)
        self.summary_widget.delete("1.0", tk.END)
        self.summary_widget.insert("1.0", "\n".join(lineas))
        self.summary_widget.configure(state=tk.DISABLED)

    def _es_nan(self, valor) -> bool:
        try:
            return valor != valor
        except Exception:
            return False

    def build_selected_tray_combined(self, exp_ids: list[str]) -> dict:
        frames = []
        rows_by_exp_id = {}

        for trays_path in AnalisisTrays.descubrir_trays(exp_ids=exp_ids):
            exp_id = trays_path.parent.name
            datos = pd.read_csv(trays_path, dtype=str)
            faltantes = AnalisisTrays.REQUIRED_TRAY_COLUMNS - set(datos.columns)
            if faltantes:
                raise ValueError(
                    f"El archivo '{trays_path}' no tiene las columnas requeridas: "
                    + ", ".join(sorted(faltantes))
                )

            datos = datos.loc[:, AnalisisTrays.TRAY_COLUMNS].copy()
            frames.append(datos)
            rows_by_exp_id[exp_id] = len(datos)

        combinado = pd.concat(frames, ignore_index=True)
        output_path = AnalisisTrays.EXPERIMENTS_DIR / "tray_combined.csv"
        combinado.to_csv(output_path, index=False)

        print(f"Archivo combinado generado: {output_path}")
        print(f"Experimentos: {', '.join(exp_ids)}")
        print(f"Filas totales: {len(combinado)}")

        return {
            "exp_ids": exp_ids,
            "output_path": output_path,
            "total_rows": len(combinado),
            "rows_by_exp_id": rows_by_exp_id,
        }

    def build_selected_combined_plot(
        self, exp_ids: list[str], data: pd.DataFrame
    ) -> Path:
        if not AnalisisTrays.cargar_dependencias_plotting():
            raise RuntimeError(
                "No se pudieron cargar matplotlib/seaborn; no se puede generar el plot combinado."
            )

        plot_data = data.copy()
        plot_data["exp_id"] = plot_data["exp_id"].astype(str)
        plot_data = plot_data[plot_data["exp_id"].isin(exp_ids)]
        plot_data = plot_data.dropna(subset=["exp_id", "raw_delta_weight"])
        if plot_data.empty:
            raise ValueError("No hay datos disponibles para generar el plot combinado.")

        palette = AnalisisTrays.sns.color_palette(
            "tab20", n_colors=max(len(exp_ids), 2)
        )[: len(exp_ids)]
        plot_dir = AnalisisTrays.GLOBAL_ANALYSIS_DIR / "selected_combined_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = plot_dir / f"analisis_combinado_{'_'.join(exp_ids)}.png"

        AnalisisTrays.sns.set_theme(style="whitegrid", context="notebook")
        fig, ax = AnalisisTrays.plt.subplots(figsize=(12, 7))
        AnalisisTrays.sns.boxplot(
            data=plot_data,
            x="exp_id",
            y="raw_delta_weight",
            order=exp_ids,
            palette=palette,
            showfliers=False,
            ax=ax,
        )
        AnalisisTrays.sns.stripplot(
            data=plot_data,
            x="exp_id",
            y="raw_delta_weight",
            order=exp_ids,
            palette=palette,
            size=5,
            alpha=0.75,
            jitter=0.15,
            ax=ax,
        )
        ax.set_title("Datos combinados por experimento")
        ax.set_xlabel("Fecha del experimento")
        ax.set_ylabel("raw_delta_weight")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        AnalisisTrays.plt.close(fig)

        return output_path

    def create_selected_tray_combined(self) -> None:
        exp_ids = self.require_selected_experiments(
            "crear tray_combined.csv",
            minimum=2,
        )
        if exp_ids is None:
            return

        result = self.execute_with_log(
            f"Crear tray_combined.csv ({len(exp_ids)} experimentos)",
            lambda: self.build_selected_tray_combined(exp_ids),
        )
        if result is not None:
            self.last_result = result
            self.render_tray_combined_summary(result)

    def run_selected_analysis(self) -> None:
        exp_ids = self.require_selected_experiments("analizar", minimum=1)
        if exp_ids is None:
            return

        if len(exp_ids) == 1:
            title = f"Analisis de trays {exp_ids[0]}"
            callback = lambda: AnalisisTrays.run_analysis(
                exp_id=exp_ids[0], verbose=True
            )
        else:
            title = f"Analisis de trays seleccionados ({len(exp_ids)})"
            callback = lambda: AnalisisTrays.run_analysis(exp_ids=exp_ids, verbose=True)

        result = self.execute_with_log(
            title,
            callback,
        )
        if result is not None:
            if len(exp_ids) > 1:
                plot_path = self.execute_with_log(
                    "Generar plot combinado para seleccionados",
                    lambda: self.build_selected_combined_plot(
                        exp_ids, result["global_enriched"]
                    ),
                )
                if plot_path is not None:
                    result["combined_plot_path"] = str(plot_path)

            self.last_result = result
            self.render_summary(result)

    def run_negative_control_analysis(self) -> None:
        exp_ids = self.require_selected_experiments(
            "combinar controles negativos",
            minimum=2,
        )
        if exp_ids is None:
            return

        result = self.execute_with_log(
            f"Control negativo combinado ({len(exp_ids)} experimentos)",
            lambda: AnalisisTrays.run_negative_control_analysis(exp_ids, verbose=True),
        )
        if result is not None:
            self.last_result = result
            self.render_negative_control_summary(result)

    def run_all_analysis(self) -> None:
        if not self.get_available_experiments():
            messagebox.showwarning(
                "Sin experimentos",
                "No hay trays.csv disponibles para analizar.",
            )
            return

        result = self.execute_with_log(
            "Analisis de todos los trays",
            lambda: AnalisisTrays.run_analysis(exp_id=None, verbose=True),
        )
        if result is not None:
            self.last_result = result
            self.render_summary(result)


def main() -> None:
    app = AnalisisTrayGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
