"""Reusable Pygame UI: HintButton and ModalPopup."""

from __future__ import annotations

from typing import Callable, Optional

import pygame

from signplay_assets import load_hint_image

C_MINT = (78, 225, 160)
C_BLUE = (59, 130, 246)
C_BG = (30, 30, 36)
C_TEXT = (247, 247, 255)
C_PANEL = (38, 38, 48)


class HintButton:
    """Top-right '?' hint control for active game modes."""

    def __init__(self, x: int, y: int, size: int = 44) -> None:
        self.rect = pygame.Rect(x, y, size, size)
        self.visible = False
        self._font = pygame.font.SysFont("Segoe UI", 22, bold=True)

    def set_visible(self, visible: bool) -> None:
        self.visible = visible

    def handle_click(self, pos: tuple[int, int]) -> bool:
        return self.visible and self.rect.collidepoint(pos)

    def draw(self, screen: pygame.Surface) -> None:
        if not self.visible:
            return
        pygame.draw.circle(screen, C_BLUE, self.rect.center, self.rect.w // 2)
        pygame.draw.circle(screen, C_MINT, self.rect.center, self.rect.w // 2, width=2)
        label = self._font.render("?", True, C_TEXT)
        screen.blit(label, label.get_rect(center=self.rect.center))


class ModalPopup:
    """Center-screen overlay showing a hint illustration for a target character."""

    def __init__(self, screen_size: tuple[int, int], title_font: pygame.font.Font, body_font: pygame.font.Font) -> None:
        self.screen_w, self.screen_h = screen_size
        self.title_font = title_font
        self.body_font = body_font
        self.image_font = pygame.font.SysFont("Segoe UI", 48, bold=True)
        self.open = False
        self.target_chars: list[str] = ["?"]
        self._images: list[pygame.Surface] = []
        self._animation_time = 0.0
        self._card_rect = pygame.Rect(0, 0, 520, 560)
        self._card_rect.center = (self.screen_w // 2, self.screen_h // 2)
        self._close_rect = pygame.Rect(0, 0, 36, 36)
        self._ok_rect = pygame.Rect(0, 0, 120, 44)
        self._layout_buttons()

    def _layout_buttons(self) -> None:
        self._close_rect.topright = (self._card_rect.right - 12, self._card_rect.top + 12)
        self._ok_rect.centerx = self._card_rect.centerx
        self._ok_rect.bottom = self._card_rect.bottom - 20

    def toggle(self, target_chars: object) -> None:
        if self.open:
            self.close()
        else:
            self.show(target_chars)

    def show(self, target_chars: object) -> None:
        if isinstance(target_chars, (list, tuple)):
            values = [str(item) for item in target_chars if str(item).strip()]
        else:
            values = [str(target_chars)]
        self.target_chars = [value.upper() for value in values] if values else ["?"]
        self._images = [load_hint_image(char, self.image_font, (140, 140)) for char in self.target_chars]
        self._animation_time = 0.0
        self.open = True

    def close(self) -> None:
        self.open = False
        self._images = []
        self._animation_time = 0.0

    @property
    def blocks_game_input(self) -> bool:
        return self.open

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return True if the event was consumed."""
        if not self.open:
            return False

        if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
            self.close()
            return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._close_rect.collidepoint(event.pos) or self._ok_rect.collidepoint(event.pos):
                self.close()
                return True
            if not self._card_rect.collidepoint(event.pos):
                self.close()
                return True
        return True

    def draw(self, screen: pygame.Surface) -> None:
        if not self.open:
            return

        dim = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        screen.blit(dim, (0, 0))

        self._animation_time += 1 / 60
        progress = min(1.0, self._animation_time / 0.2)

        pygame.draw.rect(screen, C_PANEL, self._card_rect, border_radius=18)
        pygame.draw.rect(screen, C_MINT, self._card_rect, width=3, border_radius=18)

        title_text = "ASL Hint: " + " • ".join(self.target_chars)
        title = self.title_font.render(title_text, True, C_MINT)
        screen.blit(title, title.get_rect(midtop=(self._card_rect.centerx, self._card_rect.top + 16)))

        pygame.draw.rect(screen, (255, 100, 100), self._close_rect, border_radius=8)
        x_label = self.body_font.render("X", True, C_TEXT)
        screen.blit(x_label, x_label.get_rect(center=self._close_rect.center))

        if self._images:
            if len(self._images) == 1:
                img_rect = self._images[0].get_rect(center=(self._card_rect.centerx, self._card_rect.centery - 10))
                screen.blit(self._images[0], img_rect)
            else:
                card_w = 180
                card_h = 220
                top_y = self._card_rect.top + 110
                left_x = self._card_rect.centerx - card_w - 12
                right_x = self._card_rect.centerx + 12
                for index, image in enumerate(self._images):
                    panel_rect = pygame.Rect(0, 0, card_w, card_h)
                    panel_rect.topleft = (left_x if index == 0 else right_x, top_y)
                    panel_rect.y += int((1.0 - progress) * 24)
                    panel_rect.x += int((1.0 - progress) * (12 if index == 0 else -12))
                    alpha = int(255 * progress)
                    if alpha < 255:
                        overlay = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
                        overlay.fill((0, 0, 0, 0))
                        screen.blit(overlay, panel_rect.topleft)
                    pygame.draw.rect(screen, C_BLUE, panel_rect, width=2, border_radius=14)
                    img_rect = image.get_rect(center=panel_rect.center)
                    img_rect.y -= 8
                    screen.blit(image, img_rect)
                    label = self.body_font.render(f"Hint {index + 1}", True, C_MINT)
                    label_rect = label.get_rect(center=(panel_rect.centerx, panel_rect.bottom - 18))
                    screen.blit(label, label_rect)
                    value_label = self.body_font.render(self.target_chars[index], True, C_TEXT)
                    value_rect = value_label.get_rect(center=(panel_rect.centerx, panel_rect.bottom - 48))
                    screen.blit(value_label, value_rect)

        pygame.draw.rect(screen, C_BLUE, self._ok_rect, border_radius=10)
        ok_label = self.body_font.render("OK", True, C_TEXT)
        screen.blit(ok_label, ok_label.get_rect(center=self._ok_rect.center))
