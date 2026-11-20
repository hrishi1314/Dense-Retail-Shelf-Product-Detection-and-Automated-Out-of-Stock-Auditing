"""Drawing helpers for the audit overlay."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GREEN = (46, 160, 67)
RED = (220, 38, 38)


def _get_font(size: int) -> ImageFont.ImageFont:
    """Return a TrueType font if available, otherwise fall back to the built-in bitmap font."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (IOError, OSError):
        try:
            # Linux / Colab / Streamlit Cloud fallback
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except (IOError, OSError):
            return ImageFont.load_default()


def draw_audit(image: Image.Image, boxes: np.ndarray, gaps, show_boxes: bool = True) -> Image.Image:
    """Overlay product boxes (green) and out-of-stock gaps (translucent red)."""
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    lw = max(2, base.width // 400)
    font_size = max(12, base.width // 80)
    font = _get_font(font_size)

    if show_boxes:
        for x1, y1, x2, y2 in boxes:
            d.rectangle([x1, y1, x2, y2], outline=GREEN + (255,), width=lw)

    for g in gaps:
        d.rectangle([g.x1, g.y1, g.x2, g.y2], fill=RED + (90,), outline=RED + (255,), width=lw + 1)
        label = f"~{g.est_missing}"
        # Semi-transparent background behind label for readability
        bbox = d.textbbox((g.x1 + 4, g.y1 + 4), label, font=font)
        d.rectangle(bbox, fill=(0, 0, 0, 160))
        d.text((g.x1 + 4, g.y1 + 4), label, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(base, layer).convert("RGB")
