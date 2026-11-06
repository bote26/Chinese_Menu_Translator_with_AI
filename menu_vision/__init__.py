"""Menu scanning: read a Chinese menu photo, then translate it with prices and context.

Public entry point:

    from menu_vision import scan_menu
    result = scan_menu("menu.jpg")          # vision model, one call
    result = scan_menu("menu.jpg", extractor="ocr")  # easyocr + text model
    for item in result.items:
        print(item.cn, item.price_display, item.dish.en, item.dish.description)

See `pipeline.scan_menu` for the full contract and `vision.py` for why the
vision extractor is the default.
"""

from .config import Settings
from .models import DishInfo, EnrichedItem, MenuResult, OcrBox, RawItem
from .pipeline import scan_menu

__all__ = [
    "Settings",
    "DishInfo",
    "EnrichedItem",
    "MenuResult",
    "OcrBox",
    "RawItem",
    "scan_menu",
]
