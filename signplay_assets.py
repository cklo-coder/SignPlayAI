import pygame
from pathlib import Path

def load_hint_image(char: str, font=None, size: tuple[int, int] = (140, 140)) -> pygame.Surface:
    """
    動態加載手語提示圖片。
    適配 signplay_ui.py 的 3 參數調用：char, font, size
    """
    char = str(char).upper()
    
    # 預設圖片路徑
    asset_dir = Path(__file__).parent / "assets"
    image_path = asset_dir / f"{char}.png"
    
    # 1. 嘗試加載實體圖片
    if image_path.exists():
        try:
            img = pygame.image.load(str(image_path)).convert_alpha()
            return pygame.transform.smoothscale(img, size)
        except Exception:
            pass

    # 2. 【安全防禦線】如果圖片不存在，自動繪製與 UI 風格一體化的霓虹科技感佔位符
    surface = pygame.Surface(size, pygame.SRCALPHA)
    
    # 繪製深色圓角背景與藍色邊框
    pygame.draw.rect(surface, (38, 38, 48), (0, 0, size[0], size[1]), border_radius=12)
    pygame.draw.rect(surface, (59, 130, 246), (0, 0, size[0], size[1]), width=2, border_radius=12)
    
    # 渲染字元（優先使用 UI 傳過來的 font）
    try:
        use_font = font if font is not None else pygame.font.SysFont("Segoe UI", 28, bold=True)
        text_surf = use_font.render(char, True, (247, 247, 255))
        text_rect = text_surf.get_rect(center=(size[0] // 2, size[1] // 2))
        surface.blit(text_surf, text_rect)
    except Exception:
        pass
        
    return surface