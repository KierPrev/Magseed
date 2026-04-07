from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import qrcode

# =====================================
# CONFIGURACIÓN GENERAL
# =====================================

OUTPUT_FILE = Path("etiquetas_a4.png")

# A4 a 300 dpi aprox
A4_WIDTH = 2480
A4_HEIGHT = 3508

# Márgenes de página
PAGE_MARGIN_X = 120
PAGE_MARGIN_Y = 120

# Grilla: 3 columnas x 9 filas = 27 etiquetas
COLS = 3
ROWS = 9

# Separación entre etiquetas
GAP_X = 40
GAP_Y = 30

# Colores
BG_COLOR = "white"
FG_COLOR = "black"

# Borde de cada etiqueta
BORDER_WIDTH = 4

# Tamaño del texto
FONT_SIZE = 72

# Espaciado interno dentro de cada etiqueta
INNER_PADDING = 20

# =====================================
# FUENTE
# =====================================

def load_font(size: int):
    possible_fonts = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for font_path in possible_fonts:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()

font = load_font(FONT_SIZE)

# =====================================
# IDENTIFICADORES
# =====================================

def generate_identifiers():
    return [f"{letter}{number}" for number in range(1, 10) for letter in ["A", "B", "C"]]
    # Resultado: A1, B1, C1, A2, B2, C2, ... A9, B9, C9

# =====================================
# QR
# =====================================

def make_qr(data: str, target_size: int) -> Image.Image:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=1,   # borde mínimo del QR
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    img = img.resize((target_size, target_size), Image.NEAREST)
    return img

# =====================================
# DIBUJO DE UNA ETIQUETA
# =====================================

def draw_label(page, draw, x0, y0, w, h, identifier):
    # Recuadro negro único
    draw.rectangle(
        [(x0, y0), (x0 + w - 1, y0 + h - 1)],
        outline=FG_COLOR,
        width=BORDER_WIDTH
    )

    # Medir texto
    text_bbox = draw.textbbox((0, 0), identifier, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    # Espacio disponible interno
    inner_w = w - 2 * INNER_PADDING
    inner_h = h - 2 * INNER_PADDING

    # Reservar espacio para texto abajo
    text_area_h = text_h + 20

    # QR cuadrado, lo más grande posible dentro del área restante
    qr_size = min(inner_w, inner_h - text_area_h - 20)
    qr_size = max(qr_size, 50)

    qr_img = make_qr(identifier, qr_size)

    # Posición QR
    qr_x = x0 + (w - qr_size) // 2
    qr_y = y0 + INNER_PADDING + 10

    page.paste(qr_img, (qr_x, qr_y))

    # Posición texto
    text_x = x0 + (w - text_w) // 2
    text_y = qr_y + qr_size + 20

    draw.text((text_x, text_y), identifier, fill=FG_COLOR, font=font)

# =====================================
# MAIN
# =====================================

def main():
    page = Image.new("RGB", (A4_WIDTH, A4_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(page)

    identifiers = generate_identifiers()

    usable_width = A4_WIDTH - 2 * PAGE_MARGIN_X
    usable_height = A4_HEIGHT - 2 * PAGE_MARGIN_Y

    cell_w = (usable_width - (COLS - 1) * GAP_X) // COLS
    cell_h = (usable_height - (ROWS - 1) * GAP_Y) // ROWS

    for i, identifier in enumerate(identifiers):
        row = i // COLS
        col = i % COLS

        x0 = PAGE_MARGIN_X + col * (cell_w + GAP_X)
        y0 = PAGE_MARGIN_Y + row * (cell_h + GAP_Y)

        draw_label(page, draw, x0, y0, cell_w, cell_h, identifier)

    page.save(OUTPUT_FILE)
    print(f"Imagen A4 generada: {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    main()