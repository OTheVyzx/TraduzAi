"""
Script para criar um CBZ de teste com imagens sintéticas simulando páginas de mangá.
Cada imagem tem balões de texto simples com texto em inglês.
"""

import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).parent / "teste"
OUTPUT_DIR.mkdir(exist_ok=True)

PAGES = [
    {
        "filename": "001.jpg",
        "bg": (240, 240, 240),
        "balloons": [
            {"text": "Hello! Are you ready\nfor battle?", "pos": (80, 60), "size": (280, 80)},
            {"text": "I have been waiting\nfor this moment!", "pos": (80, 200), "size": (280, 80)},
        ],
    },
    {
        "filename": "002.jpg",
        "bg": (220, 220, 220),
        "balloons": [
            {"text": "You will never\ndefeat me!", "pos": (80, 80), "size": (280, 70)},
            {"text": "My power is\nbeyond your limit.", "pos": (80, 220), "size": (280, 70)},
            {"text": "Give up now.", "pos": (80, 360), "size": (200, 50)},
        ],
    },
    {
        "filename": "003.jpg",
        "bg": (230, 230, 230),
        "balloons": [
            {"text": "This is the end!", "pos": (100, 100), "size": (240, 60)},
            {"text": "No... impossible.", "pos": (100, 260), "size": (240, 60)},
        ],
    },
]

IMG_W, IMG_H = 800, 1200

try:
    font_title = ImageFont.truetype("arial.ttf", 24)
    font_text = ImageFont.truetype("arial.ttf", 20)
except OSError:
    font_title = ImageFont.load_default()
    font_text = ImageFont.load_default()

image_paths = []
for page in PAGES:
    img = Image.new("RGB", (IMG_W, IMG_H), page["bg"])
    draw = ImageDraw.Draw(img)

    # Simula painéis
    draw.rectangle([20, 20, IMG_W - 20, IMG_H - 20], outline=(0, 0, 0), width=3)
    draw.rectangle([20, 20, IMG_W // 2 - 10, IMG_H // 2 - 10], outline=(0, 0, 0), width=2)
    draw.rectangle([IMG_W // 2 + 10, 20, IMG_W - 20, IMG_H // 2 - 10], outline=(0, 0, 0), width=2)
    draw.rectangle([20, IMG_H // 2 + 10, IMG_W - 20, IMG_H - 20], outline=(0, 0, 0), width=2)

    # Balões de fala
    for b in page["balloons"]:
        x, y = b["pos"]
        w, h = b["size"]
        draw.ellipse([x, y, x + w, y + h], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        draw.text((x + 12, y + 12), b["text"], fill=(0, 0, 0), font=font_text)

    out_path = OUTPUT_DIR / page["filename"]
    img.save(out_path, "JPEG", quality=90)
    image_paths.append(out_path)
    print(f"Criada: {out_path}")

# Cria o CBZ (é um ZIP com imagens)
cbz_path = OUTPUT_DIR / "teste_capitulo.cbz"
with zipfile.ZipFile(cbz_path, "w", compression=zipfile.ZIP_STORED) as zf:
    for img_path in image_paths:
        zf.write(img_path, img_path.name)

print(f"\nCBZ criado: {cbz_path}")
print(f"Conteúdo: {[p.name for p in image_paths]}")
