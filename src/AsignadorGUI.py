from datetime import date
import tkinter as tk
from tkinter import messagebox, ttk

import Asignador


BG_COLOR = "#eef1f3"
CARD_COLOR = "#ffffff"
PANEL_COLOR = "#f3f5f7"
ACCENT_COLOR = "#1f6f78"
ACCENT_DARK = "#154f56"
TEXT_COLOR = "#1a2226"
MUTED_COLOR = "#647079"
BORDER_COLOR = "#dde2e6"
TABLE_ALT_COLOR = "#f7f9fa"
POSITION_HIGHLIGHT = "#d9622b"
SORT_BY_POSITION = "position"
SORT_BY_TREATMENT = "treatment_floor"
SORT_LABELS = {
    "Posición (A1, A2, A3...)": SORT_BY_POSITION,
    "Piso de tratamiento (1, 1, 1, 2...)": SORT_BY_TREATMENT,
}
DEFAULT_SORT_LABEL = "Posición (A1, A2, A3...)"
MAX_TREATMENT_FLOORS = Asignador.MAX_TREATMENT_FLOORS
MAX_REPLICAS = Asignador.MAX_REPLICAS


class AsignadorGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Asignador de Bandejas")
        self.geometry("1180x760")
        self.minsize(1024, 700)
        self.configure(bg=BG_COLOR)

        self.current_data = []
        self.current_treatments = []
        self.selected_position = ""
        self.selected_treatment_id = ""
        self.tray_cells = {}
        self.treatment_floor_cell = None
        self.location_panel_host = None
        self.loaded_config = None
        self.current_location_layout = None

        hoy = date.today().strftime("%Y%m%d")
        self.exp_id_var = tk.StringVar(value=hoy)
        self.experiment_type_var = tk.StringVar(value="CTRL_NEGATIVO")
        self.magnet_pole_var = tk.StringVar(value="NONE")
        self.treatment_floors_var = tk.StringVar(
            value=str(Asignador.DEFAULT_TREATMENT_FLOORS)
        )
        self.replicas_var = tk.StringVar(value=str(Asignador.DEFAULT_REPLICAS))
        self.total_trays_var = tk.StringVar(value="-")
        self.range_var = tk.StringVar(value="-")
        self.position_var = tk.StringVar()
        self.suggestion_var = tk.StringVar(value="-")
        self.current_tray_var = tk.StringVar(value="-")
        self.current_treatment_var = tk.StringVar(value="-")
        self.current_pre_var = tk.StringVar(value="-")
        self.current_post_var = tk.StringVar(value="-")
        self.current_no_germinadas_var = tk.StringVar(value="-")
        self.new_pre_var = tk.StringVar()
        self.new_post_var = tk.StringVar()
        self.new_no_germinadas_var = tk.StringVar()
        self.treatment_id_var = tk.StringVar()
        self.treatment_value_var = tk.StringVar()
        self.treatment_type_var = tk.StringVar()
        self.sort_var = tk.StringVar(value=DEFAULT_SORT_LABEL)
        self.status_var = tk.StringVar(
            value="Listo para trabajar. Elegí un experimento o cargá uno existente."
        )

        self._configure_styles()
        self._build_layout()
        self.treatment_floors_var.trace_add("write", self._on_tray_config_changed)
        self.replicas_var.trace_add("write", self._on_tray_config_changed)
        self._on_tray_config_changed()

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
            font=("Helvetica Neue", 24, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=BG_COLOR,
            foreground=MUTED_COLOR,
            font=("Helvetica Neue", 11),
        )
        style.configure(
            "Section.TLabel",
            background=CARD_COLOR,
            foreground=TEXT_COLOR,
            font=("Helvetica Neue", 13, "bold"),
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
            "Value.TLabel",
            background=CARD_COLOR,
            foreground=TEXT_COLOR,
            font=("Helvetica", 11, "bold"),
        )
        style.configure(
            "EditHint.TLabel",
            background=CARD_COLOR,
            foreground=ACCENT_DARK,
            font=("Helvetica", 10, "bold"),
        )
        style.configure(
            "TButton",
            padding=(16, 12),
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
            "Treeview",
            background="white",
            fieldbackground="white",
            foreground=TEXT_COLOR,
            rowheight=36,
            bordercolor=BORDER_COLOR,
            font=("Helvetica", 10),
        )
        style.map(
            "Treeview",
            background=[("selected", ACCENT_DARK)],
            foreground=[("selected", "white")],
        )
        style.configure(
            "Treeview.Heading",
            background=ACCENT_COLOR,
            foreground="white",
            font=("Helvetica", 10, "bold"),
            relief="flat",
        )
        style.map("Treeview.Heading", background=[("active", ACCENT_DARK)])

    def _build_layout(self):
        container = ttk.Frame(self, padding=24, style="App.TFrame")
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        header = ttk.Frame(container, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="Asignador de Bandejas",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Esta interfaz usa el backend de Asignador.py para crear archivos y editar pesos sin duplicar lógica.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        top_row = ttk.Frame(container, style="App.TFrame")
        top_row.grid(row=1, column=0, sticky="ew", pady=(20, 18))
        top_row.columnconfigure(0, weight=3)
        top_row.columnconfigure(1, weight=2)
        top_row.columnconfigure(2, weight=2)

        self._build_experiment_card(top_row).grid(
            row=0, column=0, sticky="nsew", padx=(0, 10)
        )
        self._build_metadata_card(top_row).grid(
            row=0, column=1, sticky="nsew", padx=(10, 10)
        )
        self._build_treatment_card(top_row).grid(
            row=0, column=2, sticky="nsew", padx=(10, 0)
        )

        main_row = ttk.Frame(container, style="App.TFrame")
        main_row.grid(row=2, column=0, sticky="nsew")
        main_row.columnconfigure(0, weight=1)
        main_row.rowconfigure(0, weight=1)

        self._build_table_card(main_row).grid(row=0, column=0, sticky="nsew")

        self.status_label = tk.Label(
            container,
            textvariable=self.status_var,
            bg=BG_COLOR,
            fg=MUTED_COLOR,
            anchor="w",
            font=("Helvetica", 10),
            pady=10,
        )
        self.status_label.grid(row=3, column=0, sticky="ew")

    def _build_experiment_card(self, parent):
        frame = ttk.Frame(parent, padding=24, style="Card.TFrame")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Experimento", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(
            frame,
            text="Trabajá siempre con el ID en formato YYYYMMDD.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 16))

        ttk.Label(frame, text="Fecha / ID", style="Body.TLabel").grid(
            row=2, column=0, sticky="w", pady=(0, 8)
        )
        self.exp_entry = ttk.Entry(frame, textvariable=self.exp_id_var, font=("Helvetica", 12))
        self.exp_entry.grid(row=2, column=1, sticky="ew", pady=(0, 8))

        ttk.Button(
            frame,
            text="Hoy",
            style="Secondary.TButton",
            command=self._use_today,
        ).grid(row=2, column=2, sticky="ew", padx=(8, 0), pady=(0, 8))

        ttk.Label(frame, text="Treatment floors", style="Body.TLabel").grid(
            row=3, column=0, sticky="w", pady=(4, 8)
        )
        ttk.Spinbox(
            frame,
            from_=1,
            to=MAX_TREATMENT_FLOORS,
            textvariable=self.treatment_floors_var,
            font=("Helvetica", 11),
            width=10,
        ).grid(row=3, column=1, sticky="w", pady=(4, 8))

        ttk.Label(frame, text="Réplicas", style="Body.TLabel").grid(
            row=4, column=0, sticky="w", pady=(0, 8)
        )
        ttk.Spinbox(
            frame,
            from_=1,
            to=MAX_REPLICAS,
            textvariable=self.replicas_var,
            font=("Helvetica", 11),
            width=10,
        ).grid(row=4, column=1, sticky="w", pady=(0, 8))

        summary = ttk.Frame(frame, style="Card.TFrame")
        summary.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        summary.columnconfigure(0, weight=1)
        summary.columnconfigure(1, weight=1)
        self._build_info_field(summary, 0, 0, "Total de bandejas", self.total_trays_var)
        self._build_info_field(summary, 0, 1, "Rango esperado", self.range_var)

        ttk.Button(
            frame,
            text="Crear trays.csv",
            command=self._create_trays,
        ).grid(row=6, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(
            frame,
            text="Cargar bandejas",
            style="Secondary.TButton",
            command=self._load_experiment,
        ).grid(row=6, column=1, sticky="ew", padx=(8, 8), pady=(12, 0))
        ttk.Button(
            frame,
            text="Refrescar",
            style="Secondary.TButton",
            command=self._refresh_current_experiment,
        ).grid(row=6, column=2, sticky="ew", pady=(12, 0))

        return frame

    def _build_metadata_card(self, parent):
        frame = ttk.Frame(parent, padding=24, style="Card.TFrame")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Metadata", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            frame,
            text="Se agrega una fila al archivo global experiments_metadata.csv usando el mismo backend del script original.",
            style="Muted.TLabel",
            wraplength=300,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 16))

        ttk.Label(frame, text="Tipo", style="Body.TLabel").grid(
            row=2, column=0, sticky="w", pady=(0, 8)
        )
        self.experiment_type_combo = ttk.Combobox(
            frame,
            textvariable=self.experiment_type_var,
            values=tuple(Asignador.obtener_experiment_types_disponibles()),
            state="normal",
            font=("Helvetica", 11),
        )
        self.experiment_type_combo.grid(
            row=2, column=1, sticky="ew", pady=(0, 8)
        )
        self.experiment_type_combo.bind("<FocusIn>", self._refresh_experiment_type_options)

        ttk.Label(frame, text="Magnet pole", style="Body.TLabel").grid(
            row=3, column=0, sticky="w", pady=(0, 8)
        )
        ttk.Combobox(
            frame,
            textvariable=self.magnet_pole_var,
            values=("N", "S", "NONE"),
            state="readonly",
            font=("Helvetica", 11),
        ).grid(row=3, column=1, sticky="ew", pady=(0, 8))

        ttk.Button(
            frame,
            text="Agregar al metadata global",
            command=self._create_metadata,
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        return frame

    def _build_treatment_card(self, parent):
        frame = ttk.Frame(parent, padding=24, style="Card.TFrame")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        ttk.Label(frame, text="Treatments", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            frame,
            text="Se guarda en treatment.csv dentro del experimento cargado.",
            style="Muted.TLabel",
            wraplength=300,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 12))

        columns = ("treatment_id", "treatment_value", "treatment_type")
        self.treatment_tree = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            selectmode="browse",
            height=4,
        )
        self.treatment_tree.heading("treatment_id", text="ID")
        self.treatment_tree.heading("treatment_value", text="Valor")
        self.treatment_tree.heading("treatment_type", text="Tipo")
        self.treatment_tree.column("treatment_id", width=52, anchor="center")
        self.treatment_tree.column("treatment_value", width=90, anchor="center")
        self.treatment_tree.column("treatment_type", width=120, anchor="center")
        self.treatment_tree.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.treatment_tree.bind("<<TreeviewSelect>>", self._on_treatment_select)

        ttk.Label(frame, text="Valor", style="Body.TLabel").grid(
            row=3, column=0, sticky="w", pady=(12, 4)
        )
        ttk.Label(frame, text="Tipo", style="Body.TLabel").grid(
            row=3, column=1, sticky="w", pady=(12, 4), padx=(8, 0)
        )
        ttk.Entry(
            frame,
            textvariable=self.treatment_value_var,
            font=("Helvetica", 11),
        ).grid(row=4, column=0, sticky="ew")
        ttk.Entry(
            frame,
            textvariable=self.treatment_type_var,
            font=("Helvetica", 11),
        ).grid(row=4, column=1, sticky="ew", padx=(8, 0))

        ttk.Button(
            frame,
            text="Guardar treatment",
            command=self._save_treatment,
        ).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 0))

        return frame

    def _build_editor_card(self, parent):
        frame = ttk.Frame(parent, padding=24, style="Card.TFrame")
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Editar pesos", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            frame,
            text="Podés escribir una posición o seleccionar una fila de la tabla.",
            style="Muted.TLabel",
            wraplength=320,
        ).grid(row=1, column=0, sticky="w", pady=(4, 16))

        suggestion_frame = ttk.Frame(frame, style="Card.TFrame")
        suggestion_frame.grid(row=2, column=0, sticky="ew")
        suggestion_frame.columnconfigure(0, weight=1)
        ttk.Label(suggestion_frame, text="Siguiente sugerida", style="Body.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            suggestion_frame,
            textvariable=self.suggestion_var,
            style="Value.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Button(
            suggestion_frame,
            text="Usar sugerida",
            style="Secondary.TButton",
            command=self._apply_suggestion,
        ).grid(row=2, column=0, sticky="ew", pady=(10, 0))

        form = ttk.Frame(frame, style="Card.TFrame")
        form.grid(row=3, column=0, sticky="ew", pady=(18, 0))
        form.columnconfigure(0, weight=1)

        ttk.Label(form, text="Posición", style="Body.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        position_entry = ttk.Entry(
            form,
            textvariable=self.position_var,
            font=("Helvetica", 12),
        )
        position_entry.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        position_entry.bind("<KeyRelease>", self._on_position_typed)

        info_grid = ttk.Frame(form, style="Card.TFrame")
        info_grid.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        info_grid.columnconfigure(0, weight=1)
        info_grid.columnconfigure(1, weight=1)

        self._build_info_field(info_grid, 0, 0, "Tray ID", self.current_tray_var)
        self._build_info_field(info_grid, 0, 1, "Treatment floor", self.current_treatment_var)
        self._build_info_field(info_grid, 1, 0, "Peso antes actual", self.current_pre_var)
        self._build_info_field(info_grid, 1, 1, "Peso después actual", self.current_post_var)
        self._build_info_field(
            info_grid, 2, 0, "No germinadas actual", self.current_no_germinadas_var
        )

        ttk.Label(form, text="Campos editables", style="EditHint.TLabel").grid(
            row=3, column=0, sticky="w"
        )
        ttk.Label(
            form,
            text="Escribí los pesos en estas cajas blancas y después apretá Guardar cambios.",
            style="Muted.TLabel",
            wraplength=320,
        ).grid(row=4, column=0, sticky="w", pady=(2, 10))

        ttk.Label(form, text="Peso antes", style="Body.TLabel").grid(
            row=5, column=0, sticky="w"
        )
        tk.Entry(
            form,
            textvariable=self.new_pre_var,
            font=("Helvetica", 13),
            bg="white",
            fg=TEXT_COLOR,
            relief="solid",
            bd=1,
            highlightthickness=2,
            highlightbackground="#cfd8d3",
            highlightcolor=ACCENT_COLOR,
            insertbackground=TEXT_COLOR,
        ).grid(row=6, column=0, sticky="ew", pady=(4, 10), ipady=7)

        ttk.Label(form, text="Peso después", style="Body.TLabel").grid(
            row=7, column=0, sticky="w"
        )
        tk.Entry(
            form,
            textvariable=self.new_post_var,
            font=("Helvetica", 13),
            bg="white",
            fg=TEXT_COLOR,
            relief="solid",
            bd=1,
            highlightthickness=2,
            highlightbackground="#cfd8d3",
            highlightcolor=ACCENT_COLOR,
            insertbackground=TEXT_COLOR,
        ).grid(row=8, column=0, sticky="ew", pady=(4, 16), ipady=7)

        ttk.Label(form, text="No germinadas", style="Body.TLabel").grid(
            row=9, column=0, sticky="w"
        )
        tk.Entry(
            form,
            textvariable=self.new_no_germinadas_var,
            font=("Helvetica", 13),
            bg="white",
            fg=TEXT_COLOR,
            relief="solid",
            bd=1,
            highlightthickness=2,
            highlightbackground="#cfd8d3",
            highlightcolor=ACCENT_COLOR,
            insertbackground=TEXT_COLOR,
        ).grid(row=10, column=0, sticky="ew", pady=(4, 16), ipady=7)

        buttons = ttk.Frame(form, style="Card.TFrame")
        buttons.grid(row=11, column=0, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        buttons.columnconfigure(2, weight=1)

        ttk.Button(
            buttons,
            text="Guardar cambios",
            command=self._save_weights,
        ).grid(row=0, column=0, sticky="ew")
        ttk.Button(
            buttons,
            text="Limpiar campos",
            style="Secondary.TButton",
            command=self._clear_weight_entries,
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(
            buttons,
            text="Abrir editor",
            style="Secondary.TButton",
            command=self._open_edit_dialog,
        ).grid(row=0, column=2, sticky="ew", padx=(8, 0))

        return frame

    def _build_info_field(self, parent, row, column, label, variable):
        block = ttk.Frame(parent, padding=(0, 0, 12, 12), style="Card.TFrame")
        block.grid(row=row, column=column, sticky="ew")
        ttk.Label(block, text=label, style="Muted.TLabel").pack(anchor="w")
        ttk.Label(block, textvariable=variable, style="Value.TLabel").pack(
            anchor="w", pady=(2, 0)
        )

    def _build_table_card(self, parent):
        frame = ttk.Frame(parent, padding=24, style="Card.TFrame")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        ttk.Label(frame, text="Estado actual", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            frame,
            text="Seleccioná una bandeja y abrí el editor con doble click o con el botón.",
            style="Muted.TLabel",
        ).grid(row=0, column=1, sticky="e")

        controls = ttk.Frame(frame, style="Card.TFrame")
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Ordenar por", style="Body.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        sort_combo = ttk.Combobox(
            controls,
            textvariable=self.sort_var,
            values=tuple(SORT_LABELS.keys()),
            state="readonly",
            font=("Helvetica", 10),
            width=34,
        )
        sort_combo.grid(row=0, column=1, sticky="w", padx=(10, 0))
        sort_combo.bind("<<ComboboxSelected>>", self._on_sort_changed)

        content = ttk.Frame(frame, style="Card.TFrame")
        content.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(16, 0))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        table_container = ttk.Frame(content, style="Card.TFrame")
        table_container.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        table_container.columnconfigure(0, weight=1)
        table_container.rowconfigure(0, weight=1)

        columns = ("position", "treatment_floor", "pre_weight", "post_weight", "no_germinadas")
        self.tree = ttk.Treeview(
            table_container,
            columns=columns,
            show="headings",
            selectmode="browse",
        )
        self.tree.heading(
            "position",
            text="Posición",
            command=lambda: self._set_sort_mode(DEFAULT_SORT_LABEL),
        )
        self.tree.heading(
            "treatment_floor",
            text="Piso",
            command=lambda: self._set_sort_mode("Piso de tratamiento (1, 1, 1, 2...)"),
        )
        self.tree.heading("pre_weight", text="Peso antes")
        self.tree.heading("post_weight", text="Peso después")
        self.tree.heading("no_germinadas", text="No germinadas")

        self.tree.column("position", width=90, anchor="center")
        self.tree.column("treatment_floor", width=120, anchor="center")
        self.tree.column("pre_weight", width=120, anchor="center")
        self.tree.column("post_weight", width=120, anchor="center")
        self.tree.column("no_germinadas", width=120, anchor="center")

        scrollbar = ttk.Scrollbar(table_container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.tree.tag_configure("odd", background=TABLE_ALT_COLOR)
        self.tree.tag_configure("even", background="white")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-1>", self._open_edit_dialog)

        self.location_panel_host = ttk.Frame(content, style="Card.TFrame")
        self.location_panel_host.grid(row=0, column=1, sticky="nsew")
        self._refresh_location_panel(force=True)

        return frame

    def _build_location_panel(self, parent, cantidad_treatment_floors, cantidad_replicas):
        frame = ttk.Frame(parent, padding=18, style="Card.TFrame")
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Ubicación", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            frame,
            text="Se marca la bandeja seleccionada.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 14))

        visual = ttk.Frame(frame, style="Card.TFrame")
        visual.grid(row=2, column=0, sticky="n")
        visual.columnconfigure(1, weight=0)

        grid_frame = ttk.Frame(visual, style="Card.TFrame")
        grid_frame.grid(row=0, column=0, sticky="n")

        self.tray_cells = {}
        floors = Asignador.generar_floors(Asignador.MAX_PHYSICAL_FLOORS)
        font_size = 13
        cell_width = 5
        positions_per_row = 3

        header_font = ("Helvetica", max(font_size - 1, 9), "bold")
        cell_font = ("Helvetica", font_size, "bold")

        for floor_index, floor in enumerate(floors):
            floor_frame = ttk.Frame(grid_frame, style="Card.TFrame")
            floor_frame.grid(
                row=0,
                column=floor_index,
                padx=(0 if floor_index == 0 else 20, 0),
                sticky="n",
            )

            tk.Label(
                floor_frame,
                text=floor,
                bg=CARD_COLOR,
                fg=ACCENT_DARK,
                font=header_font,
            ).grid(row=0, column=0, columnspan=positions_per_row, padx=3, pady=(0, 6))

            for position in range(1, Asignador.POSITIONS_PER_FLOOR + 1):
                row = (position - 1) // positions_per_row + 1
                column = (position - 1) % positions_per_row
                pos = f"{floor}{position}"
                cell = tk.Label(
                    floor_frame,
                    text=pos,
                    width=cell_width,
                    height=2,
                    bg=PANEL_COLOR,
                    fg=TEXT_COLOR,
                    font=cell_font,
                    relief="flat",
                    bd=0,
                    highlightthickness=1,
                    highlightbackground=BORDER_COLOR,
                )
                cell.grid(row=row, column=column, padx=3, pady=3, sticky="nsew")
                self.tray_cells[pos] = cell

        treatment_frame = ttk.Frame(visual, style="Card.TFrame")
        treatment_frame.grid(row=0, column=1, sticky="n", padx=(14, 0))

        ttk.Label(treatment_frame, text="Piso", style="Body.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        self.treatment_floor_cell = tk.Label(
            treatment_frame,
            text="-",
            width=6,
            height=2,
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=("Helvetica", 18, "bold"),
            relief="solid",
            bd=1,
        )
        self.treatment_floor_cell.grid(row=1, column=0, sticky="w")

        ttk.Button(
            frame,
            text="Abrir editor",
            style="Secondary.TButton",
            command=self._open_edit_dialog,
        ).grid(row=3, column=0, sticky="ew", pady=(16, 0))

        self._update_visual_position(None, None)
        return frame

    def _set_status(self, text, is_error=False):
        self.status_var.set(text)
        self.status_label.configure(fg="#9c3d35" if is_error else MUTED_COLOR)

    def _use_today(self):
        self.exp_id_var.set(date.today().strftime("%Y%m%d"))

    def _safe_get_tray_config(self):
        try:
            treatment_floors, replicas = Asignador.validar_configuracion_trays(
                self.treatment_floors_var.get(),
                self.replicas_var.get(),
            )
        except ValueError:
            return None

        return treatment_floors, replicas

    def _get_tray_config(self) -> tuple[int, int]:
        treatment_floors, replicas = Asignador.validar_configuracion_trays(
            self.treatment_floors_var.get(),
            self.replicas_var.get(),
        )
        self.treatment_floors_var.set(str(treatment_floors))
        self.replicas_var.set(str(replicas))
        return treatment_floors, replicas

    def _set_tray_config(self, treatment_floors: int, replicas: int):
        if self.treatment_floors_var.get() != str(treatment_floors):
            self.treatment_floors_var.set(str(treatment_floors))
        if self.replicas_var.get() != str(replicas):
            self.replicas_var.set(str(replicas))

    def _refresh_tray_config_preview(self):
        config = self._safe_get_tray_config()
        if config is None:
            self.total_trays_var.set("-")
            self.range_var.set("-")
            return

        treatment_floors, replicas = config
        posiciones = Asignador.generar_posiciones(treatment_floors, replicas)
        self.total_trays_var.set(str(treatment_floors * replicas))
        self.range_var.set(
            f"{posiciones[0][0]}{posiciones[0][1]} a {posiciones[-1][0]}{posiciones[-1][1]}"
        )

    def _active_location_config(self):
        if self.loaded_config is not None:
            return (
                self.loaded_config["cantidad_treatment_floors"],
                self.loaded_config["cantidad_replicas"],
            )

        return self._safe_get_tray_config()

    def _refresh_location_panel(self, force=False):
        if self.location_panel_host is None:
            return

        config = self._active_location_config()
        if config is None:
            return

        if not force and self.current_location_layout == config:
            return

        for child in self.location_panel_host.winfo_children():
            child.destroy()

        self.current_location_layout = config
        panel = self._build_location_panel(
            self.location_panel_host,
            cantidad_treatment_floors=config[0],
            cantidad_replicas=config[1],
        )
        panel.grid(row=0, column=0, sticky="nsew")
        self._update_visual_position(self.selected_position or None, self.current_treatment_var.get())

    def _on_tray_config_changed(self, *_args):
        self._refresh_tray_config_preview()
        if not self.current_data:
            self.loaded_config = None
            self._refresh_location_panel(force=True)

    def _get_exp_id(self) -> str:
        exp_id = self.exp_id_var.get().strip()
        if exp_id == "":
            exp_id = date.today().strftime("%Y%m%d")

        exp_id = Asignador.validar_exp_id(exp_id)
        self.exp_id_var.set(exp_id)
        return exp_id

    def _create_trays(self):
        try:
            exp_id = self._get_exp_id()
            treatment_floors, replicas = self._get_tray_config()
            archivo = Asignador.crear_trays_csv_para_experimento(
                exp_id,
                cantidad_treatment_floors=treatment_floors,
                cantidad_replicas=replicas,
            )
            self._set_status(f"Se creó {archivo}.")
            self._load_experiment()
        except Exception as exc:
            self._show_error(str(exc))

    def _create_metadata(self):
        try:
            exp_id = self._get_exp_id()
            archivo = Asignador.crear_metadata_csv_con_valores(
                exp_id,
                self.experiment_type_var.get(),
                self.magnet_pole_var.get(),
            )
            self.experiment_type_var.set(
                Asignador.normalizar_experiment_type(self.experiment_type_var.get())
            )
            self._refresh_experiment_type_options()
            self._set_status(f"Se agregó metadata del experimento en {archivo}.")
        except Exception as exc:
            self._show_error(str(exc))

    def _refresh_experiment_type_options(self, _event=None):
        if hasattr(self, "experiment_type_combo"):
            self.experiment_type_combo.configure(
                values=tuple(Asignador.obtener_experiment_types_disponibles())
            )

    def _populate_treatment_tree(self):
        for item in self.treatment_tree.get_children():
            self.treatment_tree.delete(item)

        for fila in self.current_treatments:
            treatment_id = str(fila["treatment_id"])
            self.treatment_tree.insert(
                "",
                "end",
                iid=treatment_id,
                values=(
                    treatment_id,
                    fila["treatment_value"] or "-",
                    fila["treatment_type"] or "-",
                ),
            )

        if self.current_treatments:
            selected = self.selected_treatment_id or str(
                self.current_treatments[0]["treatment_id"]
            )
            if self.treatment_tree.exists(selected):
                self.treatment_tree.selection_set(selected)
                self.treatment_tree.focus(selected)
                self._sync_treatment_details(selected)

    def _on_treatment_select(self, _event=None):
        selected = self.treatment_tree.selection()
        if selected:
            self._sync_treatment_details(selected[0])

    def _sync_treatment_details(self, treatment_id: str):
        fila = next(
            (
                item
                for item in self.current_treatments
                if str(item["treatment_id"]) == str(treatment_id)
            ),
            None,
        )
        if fila is None:
            self.selected_treatment_id = ""
            self.treatment_id_var.set("")
            self.treatment_value_var.set("")
            self.treatment_type_var.set("")
            return

        self.selected_treatment_id = str(fila["treatment_id"])
        self.treatment_id_var.set(self.selected_treatment_id)
        self.treatment_value_var.set(fila["treatment_value"])
        self.treatment_type_var.set(fila["treatment_type"])

    def _save_treatment(self):
        if not self.current_data:
            self._show_error("Primero cargá un experimento con bandejas.")
            return

        treatment_id = self.selected_treatment_id
        if not treatment_id:
            self._show_error("Primero seleccioná un treatment.")
            return

        try:
            exp_id = self._get_exp_id()
            resultado = Asignador.actualizar_treatment_en_id(
                exp_id,
                treatment_id,
                self.treatment_value_var.get(),
                self.treatment_type_var.get(),
            )
            self.current_treatments = resultado["datos"]
            self._populate_treatment_tree()
            self._set_status(f"Se guardó treatment_id {treatment_id}.")
        except Exception as exc:
            self._show_error(str(exc))

    def _load_experiment(self):
        try:
            exp_id = self._get_exp_id()
            estado = Asignador.obtener_estado_experimento(exp_id)
            self.current_data = estado["datos"]
            self.current_treatments = estado["treatments"]
            self.loaded_config = estado["configuracion"]
            self._set_tray_config(
                self.loaded_config["cantidad_treatment_floors"],
                self.loaded_config["cantidad_replicas"],
            )
            self._refresh_location_panel(force=True)
            self._populate_tree()
            self._populate_treatment_tree()

            sugerida = estado["siguiente_posicion"]
            self.suggestion_var.set(sugerida)
            posicion_actual = self.position_var.get().strip()
            posiciones_validas = Asignador.obtener_posiciones_validas(self.current_data)

            try:
                if posicion_actual:
                    Asignador.validar_posicion(
                        posicion_actual,
                        posiciones_validas=posiciones_validas,
                    )
                else:
                    raise ValueError
            except ValueError:
                posicion_actual = sugerida
                self.position_var.set(sugerida)

            self._sync_position_details(posicion_actual)
            self._set_status(f"Experimento {exp_id} cargado correctamente.")
        except Exception as exc:
            self._show_error(str(exc))

    def _refresh_current_experiment(self):
        self._load_experiment()

    def _floor_sort_key(self, floor: str) -> int:
        return Asignador.indice_floor(floor)

    def _sorted_current_data(self):
        sort_mode = SORT_LABELS.get(self.sort_var.get(), SORT_BY_POSITION)

        if sort_mode == SORT_BY_TREATMENT:
            return sorted(
                self.current_data,
                key=lambda fila: (
                    int(fila["treatment_floor"]),
                    self._floor_sort_key(fila["floor"]),
                    int(fila["position"]),
                ),
            )

        return sorted(
            self.current_data,
            key=lambda fila: (
                self._floor_sort_key(fila["floor"]),
                int(fila["position"]),
            ),
        )

    def _populate_tree(self):
        selected = self.tree.selection()
        selected_position = selected[0] if selected else self.selected_position

        for item in self.tree.get_children():
            self.tree.delete(item)

        for index, fila in enumerate(self._sorted_current_data()):
            pos = f"{fila['floor']}{fila['position']}"
            self.tree.insert(
                "",
                "end",
                iid=pos,
                values=(
                    pos,
                    fila["treatment_floor"],
                    fila["pre_weight"] or "-",
                    fila["post_weight"] or "-",
                    fila.get("no_germinadas") or "-",
                ),
                tags=("even" if index % 2 == 0 else "odd",),
            )

        if selected_position and self.tree.exists(selected_position):
            self.tree.selection_set(selected_position)
            self.tree.focus(selected_position)
            self.tree.see(selected_position)

    def _on_sort_changed(self, _event=None):
        self._populate_tree()

    def _set_sort_mode(self, label: str):
        self.sort_var.set(label)
        self._populate_tree()

    def _apply_suggestion(self):
        sugerida = self.suggestion_var.get().strip()
        if sugerida and sugerida != "-":
            self.position_var.set(sugerida)
            self._sync_position_details(sugerida)

    def _clear_weight_entries(self):
        self.new_pre_var.set("")
        self.new_post_var.set("")
        self.new_no_germinadas_var.set("")

    def _show_error(self, message):
        self._set_status(message, is_error=True)
        messagebox.showerror("Asignador", message)

    def _on_tree_select(self, _event=None):
        selected = self.tree.selection()
        if not selected:
            return

        posicion = selected[0]
        self.position_var.set(posicion)
        self._sync_position_details(posicion, update_tree_selection=False)

    def _on_position_typed(self, _event=None):
        self._sync_position_details(self.position_var.get().strip(), silent=True)

    def _sync_position_details(
        self, posicion: str, silent=False, update_tree_selection=True
    ):
        if not posicion:
            self._clear_current_details()
            return

        try:
            posiciones_validas = (
                Asignador.obtener_posiciones_validas(self.current_data)
                if self.current_data
                else None
            )
            floor, number = Asignador.validar_posicion(
                posicion,
                posiciones_validas=posiciones_validas,
            )
        except ValueError:
            self._clear_current_details()
            if not silent:
                self._set_status("La posición todavía no es válida.", is_error=True)
            return

        fila = Asignador.buscar_fila(self.current_data, floor, number)
        if fila is None:
            self._clear_current_details()
            if not silent:
                self._set_status("Esa posición no está cargada en la tabla.", is_error=True)
            return

        pos = f"{fila['floor']}{fila['position']}"
        previous_position = self.selected_position
        self.selected_position = pos
        self._update_visual_position(pos, fila["treatment_floor"])
        self.current_tray_var.set(fila["tray_id"])
        self.current_treatment_var.set(str(fila["treatment_floor"]))
        self.current_pre_var.set(fila["pre_weight"] or "-")
        self.current_post_var.set(fila["post_weight"] or "-")
        self.current_no_germinadas_var.set(fila.get("no_germinadas") or "-")

        if previous_position != pos:
            self.new_pre_var.set(fila["pre_weight"])
            self.new_post_var.set(fila["post_weight"])
            self.new_no_germinadas_var.set(fila.get("no_germinadas", ""))

        if update_tree_selection and self.tree.exists(pos):
            self.tree.selection_set(pos)
            self.tree.focus(pos)
            self.tree.see(pos)

    def _clear_current_details(self):
        self.selected_position = ""
        self._update_visual_position(None, None)
        self.current_tray_var.set("-")
        self.current_treatment_var.set("-")
        self.current_pre_var.set("-")
        self.current_post_var.set("-")
        self.current_no_germinadas_var.set("-")

    def _update_visual_position(self, posicion, treatment_floor=None):
        posicion_activa = None

        if posicion:
            try:
                posiciones_validas = (
                    Asignador.obtener_posiciones_validas(self.current_data)
                    if self.current_data
                    else None
                )
                floor, number = Asignador.validar_posicion(
                    posicion,
                    posiciones_validas=posiciones_validas,
                )
                posicion_activa = f"{floor}{number}"
            except ValueError:
                posicion_activa = None

        for pos, cell in self.tray_cells.items():
            is_active = pos == posicion_activa
            cell.configure(
                bg=POSITION_HIGHLIGHT if is_active else PANEL_COLOR,
                fg="white" if is_active else TEXT_COLOR,
                highlightthickness=2 if is_active else 1,
                highlightbackground=ACCENT_DARK if is_active else BORDER_COLOR,
            )

        if self.treatment_floor_cell is not None:
            has_treatment_floor = treatment_floor not in (None, "", "-")
            self.treatment_floor_cell.configure(
                text=str(treatment_floor) if has_treatment_floor else "-",
                bg=ACCENT_DARK if has_treatment_floor else PANEL_COLOR,
                fg="white" if has_treatment_floor else TEXT_COLOR,
                bd=2 if has_treatment_floor else 1,
            )

    def _open_edit_dialog(self, _event=None):
        if not self.current_data:
            self._show_error("Primero cargá un experimento con bandejas.")
            return

        posicion = self.position_var.get().strip() or self.selected_position
        if not posicion:
            self._show_error("Primero seleccioná una bandeja.")
            return

        try:
            floor, number = Asignador.validar_posicion(
                posicion,
                posiciones_validas=Asignador.obtener_posiciones_validas(self.current_data),
            )
        except ValueError as exc:
            self._show_error(str(exc))
            return

        fila = Asignador.buscar_fila(self.current_data, floor, number)
        if fila is None:
            self._show_error("No encontré esa bandeja en la tabla cargada.")
            return

        dialog = tk.Toplevel(self)
        dialog.title(f"Editar {posicion}")
        dialog.configure(bg=BG_COLOR)
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        before_var = tk.StringVar(value=fila["pre_weight"])
        after_var = tk.StringVar(value=fila["post_weight"])
        no_germinadas_var = tk.StringVar(value=fila.get("no_germinadas", ""))

        container = ttk.Frame(dialog, padding=20, style="App.TFrame")
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)

        ttk.Label(container, text=f"Editar {posicion}", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            container,
            text=f"Treatment floor: {fila['treatment_floor']}",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 16))

        ttk.Label(container, text="Peso antes", style="Body.TLabel").grid(
            row=2, column=0, sticky="w"
        )
        before_entry = tk.Entry(
            container,
            textvariable=before_var,
            font=("Helvetica", 13),
            bg="white",
            fg=TEXT_COLOR,
            relief="solid",
            bd=1,
            highlightthickness=2,
            highlightbackground="#cfd8d3",
            highlightcolor=ACCENT_COLOR,
            insertbackground=TEXT_COLOR,
        )
        before_entry.grid(row=3, column=0, sticky="ew", pady=(4, 12), ipady=7)

        ttk.Label(container, text="Peso después", style="Body.TLabel").grid(
            row=4, column=0, sticky="w"
        )
        after_entry = tk.Entry(
            container,
            textvariable=after_var,
            font=("Helvetica", 13),
            bg="white",
            fg=TEXT_COLOR,
            relief="solid",
            bd=1,
            highlightthickness=2,
            highlightbackground="#cfd8d3",
            highlightcolor=ACCENT_COLOR,
            insertbackground=TEXT_COLOR,
        )
        after_entry.grid(row=5, column=0, sticky="ew", pady=(4, 16), ipady=7)

        ttk.Label(container, text="No germinadas", style="Body.TLabel").grid(
            row=6, column=0, sticky="w"
        )
        no_germinadas_entry = tk.Entry(
            container,
            textvariable=no_germinadas_var,
            font=("Helvetica", 13),
            bg="white",
            fg=TEXT_COLOR,
            relief="solid",
            bd=1,
            highlightthickness=2,
            highlightbackground="#cfd8d3",
            highlightcolor=ACCENT_COLOR,
            insertbackground=TEXT_COLOR,
        )
        no_germinadas_entry.grid(row=7, column=0, sticky="ew", pady=(4, 16), ipady=7)

        buttons = ttk.Frame(container, style="App.TFrame")
        buttons.grid(row=8, column=0, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)

        def save_and_close():
            self.position_var.set(posicion)
            self.new_pre_var.set(before_var.get())
            self.new_post_var.set(after_var.get())
            self.new_no_germinadas_var.set(no_germinadas_var.get())
            self._save_weights()
            dialog.destroy()

        ttk.Button(
            buttons,
            text="Guardar",
            command=save_and_close,
        ).grid(row=0, column=0, sticky="ew")
        ttk.Button(
            buttons,
            text="Cancelar",
            style="Secondary.TButton",
            command=dialog.destroy,
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        before_entry.focus_set()
        dialog.bind("<Return>", lambda _event: save_and_close())

    def _save_weights(self):
        try:
            exp_id = self._get_exp_id()
            posicion = self.position_var.get().strip() or self.suggestion_var.get().strip()
            nuevo_antes = self.new_pre_var.get()
            nuevo_despues = self.new_post_var.get()
            nuevo_no_germinadas = self.new_no_germinadas_var.get()

            resultado = Asignador.actualizar_pesos_en_posicion(
                exp_id,
                posicion,
                nuevo_antes=nuevo_antes,
                nuevo_despues=nuevo_despues,
                nuevo_no_germinadas=nuevo_no_germinadas,
            )
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.current_data = resultado["datos"]
        self._populate_tree()

        siguiente = resultado["siguiente_posicion"]
        posicion_guardada = f"{resultado['fila']['floor']}{resultado['fila']['position']}"

        self.suggestion_var.set(siguiente)
        self.position_var.set(posicion_guardada)
        self._sync_position_details(posicion_guardada, silent=True)

        self._set_status(
            f"Se guardó la posición {posicion_guardada}. Siguiente sugerida: {siguiente}."
        )


def main():
    app = AsignadorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
