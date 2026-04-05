from datetime import date
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import Asignador


BG_COLOR = "#f3eee6"
CARD_COLOR = "#fffaf3"
ACCENT_COLOR = "#2f6b5b"
ACCENT_DARK = "#234d42"
TEXT_COLOR = "#1f2a25"
MUTED_COLOR = "#63706b"
BORDER_COLOR = "#d7d0c6"


class CargarFotosGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Cargar Fotos")
        self.geometry("860x620")
        self.minsize(760, 520)
        self.configure(bg=BG_COLOR)

        self.exp_id_var = tk.StringVar(value=date.today().strftime("%Y%m%d"))
        self.status_var = tk.StringVar(
            value="Elegí una fecha y seleccioná las fotos para copiarlas al experimento."
        )
        self.selected_files: list[Path] = []

        self._configure_styles()
        self._build_layout()

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
            "Files.Treeview",
            background="white",
            fieldbackground="white",
            foreground=TEXT_COLOR,
            rowheight=30,
            bordercolor=BORDER_COLOR,
            font=("Helvetica", 10),
        )
        style.map(
            "Files.Treeview",
            background=[("selected", "#58708a")],
            foreground=[("selected", "white")],
        )
        style.configure(
            "Files.Treeview.Heading",
            background=ACCENT_COLOR,
            foreground="white",
            font=("Helvetica", 10, "bold"),
            relief="flat",
        )
        style.map("Files.Treeview.Heading", background=[("active", ACCENT_DARK)])

    def _build_layout(self):
        container = ttk.Frame(self, padding=24, style="App.TFrame")
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        ttk.Label(container, text="Cargar Fotos", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            container,
            text="Crea la carpeta del experimento y copia las fotos a raw/photos sin duplicar la lógica del proyecto.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 18))

        card = ttk.Frame(container, padding=20, style="Card.TFrame")
        card.grid(row=2, column=0, sticky="nsew")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(4, weight=1)

        ttk.Label(card, text="Experimento", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            card,
            text="Usá una fecha real en formato YYYYMMDD.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        controls = ttk.Frame(card, style="Card.TFrame")
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Fecha", style="Body.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )
        entry = ttk.Entry(controls, textvariable=self.exp_id_var, font=("Helvetica", 11))
        entry.grid(row=0, column=1, sticky="ew")

        buttons = ttk.Frame(card, style="Card.TFrame")
        buttons.grid(row=3, column=0, sticky="ew", pady=(18, 16))

        ttk.Button(buttons, text="Seleccionar fotos", command=self.select_files).grid(
            row=0, column=0, padx=(0, 10)
        )
        ttk.Button(
            buttons,
            text="Limpiar selección",
            command=self.clear_files,
            style="Secondary.TButton",
        ).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(buttons, text="Crear experimento y copiar", command=self.copy_files).grid(
            row=0, column=2
        )

        ttk.Label(card, text="Fotos seleccionadas", style="Section.TLabel").grid(
            row=4, column=0, sticky="w", pady=(0, 10)
        )

        table_frame = ttk.Frame(card, style="Card.TFrame")
        table_frame.grid(row=5, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            table_frame,
            columns=("name", "folder"),
            show="headings",
            style="Files.Treeview",
        )
        self.tree.heading("name", text="Archivo")
        self.tree.heading("folder", text="Carpeta de origen")
        self.tree.column("name", width=250, anchor="w")
        self.tree.column("folder", width=520, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.count_label = tk.Label(
            card,
            text="0 fotos seleccionadas",
            bg=CARD_COLOR,
            fg=MUTED_COLOR,
            anchor="w",
            font=("Helvetica", 10),
        )
        self.count_label.grid(row=6, column=0, sticky="ew", pady=(10, 0))

        status_label = tk.Label(
            container,
            textvariable=self.status_var,
            bg=BG_COLOR,
            fg=MUTED_COLOR,
            anchor="w",
            font=("Helvetica", 10),
            pady=10,
        )
        status_label.grid(row=3, column=0, sticky="ew")

    def refresh_table(self):
        self.tree.delete(*self.tree.get_children())

        for path in self.selected_files:
            self.tree.insert("", "end", values=(path.name, str(path.parent)))

        total = len(self.selected_files)
        texto = "1 foto seleccionada" if total == 1 else f"{total} fotos seleccionadas"
        self.count_label.config(text=texto)

    def select_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Seleccionar fotos",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not file_paths:
            return

        nuevos = [Path(path) for path in file_paths]
        existentes = {path.resolve() for path in self.selected_files}

        for path in nuevos:
            resolved = path.resolve()
            if resolved not in existentes:
                self.selected_files.append(path)
                existentes.add(resolved)

        self.selected_files.sort(key=lambda path: path.name.lower())
        self.refresh_table()
        self.status_var.set("Fotos listas para copiar.")

    def clear_files(self):
        self.selected_files = []
        self.refresh_table()
        self.status_var.set("Selección limpiada.")

    def copy_files(self):
        try:
            exp_id = Asignador.validar_exp_id(self.exp_id_var.get())
            copied = Asignador.copiar_fotos_a_experimento(exp_id, self.selected_files)
        except ValueError as exc:
            messagebox.showerror("Datos inválidos", str(exc))
            self.status_var.set(str(exc))
            return
        except FileNotFoundError as exc:
            messagebox.showerror("Archivo faltante", str(exc))
            self.status_var.set(str(exc))
            return
        except Exception as exc:
            messagebox.showerror("Error inesperado", str(exc))
            self.status_var.set("Ocurrió un error al copiar las fotos.")
            return

        destino = Asignador.ruta_raw_photos(exp_id)
        total = len(copied)
        texto = "1 foto copiada" if total == 1 else f"{total} fotos copiadas"
        self.status_var.set(f"{texto} en {destino}.")
        messagebox.showinfo(
            "Carga completa",
            f"{texto.capitalize()} correctamente en:\n{destino}",
        )


def main():
    app = CargarFotosGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
