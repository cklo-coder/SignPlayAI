"""
SignPlay ASL — Premium desktop Pygame experience.

Math Mode + Spelling Bee with combination tracker, debounced input, and mode-isolated predictions.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import random
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import joblib
import numpy as np
import pygame

from signplay_features import HandLandmarkDetector
from signplay_ui import HintButton, ModalPopup

MODEL_PATH = Path(__file__).with_name("model.pkl")
ENCODER_PATH = Path(__file__).with_name("scaler_encoder.pkl")

# Layout
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
LEFT_PANEL_W = 640
RIGHT_PANEL_X = 640
RIGHT_PANEL_W = 640
CAMERA_W = 580
CAMERA_H = 435
CAMERA_X = (LEFT_PANEL_W - CAMERA_W) // 2
CAMERA_Y = 120

# Timing
HOLD_FRAMES = 15
MATH_IDLE_SUBMIT_SEC = 2.0
MATH_CLEAR_HAND_SEC = 1.5
STATE_FADE_SEC = 0.45
COMBO_FLASH_SEC = 0.35
SUCCESS_FLASH_SEC = 0.6

SPELLING_WORDS = ["CAT", "DOG", "APPLE", "BOOK", "FISH", "TREE", "SUN", "MOON"]

# Premium palette
C_BG = (30, 30, 36)           # #1E1E24
C_PANEL = (38, 38, 48)
C_TEXT = (247, 247, 255)       # #F7F7FF
C_MINT = (78, 225, 160)        # #4EE1A0
C_BLUE = (59, 130, 246)        # #3B82F6
C_NEON = (78, 225, 160)
C_DIM = (140, 144, 168)
C_ERROR = (255, 107, 107)
C_COMBO_BG = (22, 24, 32)


class GameState(Enum):
    MAIN_MENU = auto()
    MATH_MODE = auto()
    SPELLING_BEE_MODE = auto()


class MathOperation(Enum):
    ADDITION = "+"
    SUBTRACTION = "-"
    MULTIPLICATION = "x"


@dataclass
class MathProblem:
    left: int
    right: int
    operation: MathOperation

    @property
    def answer(self) -> int:
        if self.operation == MathOperation.ADDITION:
            return self.left + self.right
        if self.operation == MathOperation.SUBTRACTION:
            return self.left - self.right
        return self.left * self.right

    @property
    def answer_str(self) -> str:
        return str(self.answer)

    def prompt(self) -> str:
        symbol = "×" if self.operation == MathOperation.MULTIPLICATION else self.operation.value
        return f"{self.left} {symbol} {self.right} = ?"


@dataclass
class HoldDebouncer:
    """Fire once after label is held stable for hold_frames consecutive frames."""

    hold_frames: int = HOLD_FRAMES
    _tracking: Optional[str] = None
    _count: int = 0

    def reset(self) -> None:
        self._tracking = None
        self._count = 0

    def update(self, label: Optional[str]) -> Optional[str]:
        if label is None:
            self._tracking = None
            self._count = 0
            return None

        if label == self._tracking:
            self._count += 1
        else:
            self._tracking = label
            self._count = 1

        if self._count >= self.hold_frames:
            self._count = 0
            self._tracking = None
            return label
        return None

    def force_lock(self) -> Optional[str]:
        if not self._tracking or self._count < 1:
            return None
        locked = self._tracking
        self.reset()
        return locked

    @property
    def progress(self) -> tuple[Optional[str], int]:
        return self._tracking, self._count


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: tuple[int, int, int]
    life: float


class SignPlayGame:
    def __init__(self) -> None:
        if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
            raise FileNotFoundError("Missing model.pkl or scaler_encoder.pkl. Run train_model.py first.")

        pygame.init()
        pygame.display.set_caption("SignPlay ASL")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("Segoe UI", 52, bold=True)
        self.font_heading = pygame.font.SysFont("Segoe UI", 30, bold=True)
        self.font_body = pygame.font.SysFont("Segoe UI", 24)
        self.font_small = pygame.font.SysFont("Segoe UI", 20)
        self.font_combo = pygame.font.SysFont("Segoe UI", 36, bold=True)
        self.font_combo_label = pygame.font.SysFont("Segoe UI", 18, bold=True)

        self.model = joblib.load(MODEL_PATH)
        self.label_encoder = joblib.load(ENCODER_PATH)
        self.detector = HandLandmarkDetector()

        self.state = GameState.MAIN_MENU
        self.prev_state = GameState.MAIN_MENU
        self.state_changed_at = time.time()
        self.state_banner = "Main Menu"

        self.math_operation = MathOperation.ADDITION
        self.math_problem: Optional[MathProblem] = None
        self.math_input_buffer = ""
        self.math_debouncer = HoldDebouncer()
        self.math_message = ""
        self.math_start = 0.0
        self.math_last_digit_time = 0.0
        self.math_no_hand_since: Optional[float] = None
        self.math_clear_notice = ""
        self.math_clear_notice_until = 0.0

        self.spelling_word = ""
        self.spelling_index = 0
        self.spelling_locked = ""
        self.spelling_debouncer = HoldDebouncer()
        self.spelling_message = ""
        self.spelling_unlocked: list[bool] = []

        self.live_sign = "--"
        self.prediction_confidence = 0.0
        self.hand_present = False

        self.combo_flash_until = 0.0
        self.success_flash_until = 0.0
        self.particles: list[Particle] = []
        self.camera_surface: Optional[pygame.Surface] = None
        self.hint_button = HintButton(WINDOW_WIDTH - 72, 24, 44)
        self.hint_popup = ModalPopup((WINDOW_WIDTH, WINDOW_HEIGHT), self.font_heading, self.font_body)

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        self.detector.open_camera()
        try:
            while self._handle_events():
                self._update()
                self._draw()
                self.clock.tick(30)
        finally:
            self.detector.release()
            pygame.quit()

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if self.hint_popup.blocks_game_input:
                if self.hint_popup.handle_event(event):
                    continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.hint_button.handle_click(event.pos):
                    self.hint_popup.toggle(self._get_hint_target_char())
                    continue

            if event.type != pygame.KEYDOWN:
                continue

            if self.state == GameState.MAIN_MENU:
                if event.key == pygame.K_1:
                    self._transition_to(self._start_math, MathOperation.ADDITION)
                elif event.key == pygame.K_2:
                    self._transition_to(self._start_math, MathOperation.SUBTRACTION)
                elif event.key == pygame.K_3:
                    self._transition_to(self._start_math, MathOperation.MULTIPLICATION)
                elif event.key == pygame.K_4:
                    self._transition_to(self._start_spelling)
            elif event.key == pygame.K_ESCAPE:
                self._transition_to(self._go_menu)
            elif event.key == pygame.K_SPACE:
                self._on_spacebar_advance()
            elif event.key == pygame.K_n and self.state == GameState.MATH_MODE:
                self._start_math(self.math_operation)
            elif event.key == pygame.K_n and self.state == GameState.SPELLING_BEE_MODE:
                self._start_spelling()
        return True

    def _transition_to(self, callback, *args) -> None:
        callback(*args)
        self.state_changed_at = time.time()

    def _refresh_hint_ui(self) -> None:
        self.hint_button.set_visible(self.state != GameState.MAIN_MENU)
        if self.state == GameState.MAIN_MENU and self.hint_popup.open:
            self.hint_popup.close()

    def _get_hint_target_chars(self) -> list[str]:
        if self.state == GameState.MATH_MODE and self.math_problem is not None:
            answer_text = self.math_problem.answer_str
            if len(answer_text) == 2 and answer_text.isdigit():
                return [answer_text[0], answer_text[1]]
            return [answer_text]
        if self.state == GameState.SPELLING_BEE_MODE and self.spelling_word:
            if self.spelling_index < len(self.spelling_word):
                return [self.spelling_word[self.spelling_index]]
            return [self.spelling_word[-1]]
        return ["?"]

    def _get_hint_target_char(self) -> str:
        chars = self._get_hint_target_chars()
        return chars[0] if chars else "?"

    def _go_menu(self) -> None:
        self.prev_state = self.state
        self.state = GameState.MAIN_MENU
        self.state_banner = "Main Menu"
        self.math_message = ""
        self.spelling_message = ""

    # ---------------------------------------------------------------- modes
    def _start_math(self, operation: MathOperation) -> None:
        self.prev_state = self.state
        self.state = GameState.MATH_MODE
        self.state_banner = f"Math · {operation.name.title()}"
        self.math_operation = operation
        self.math_problem = self._generate_math_problem(operation)
        self.math_input_buffer = ""
        self.math_debouncer.reset()
        self.math_message = ""
        self.math_start = time.time()
        self.math_last_digit_time = 0.0
        self.math_no_hand_since = None
        self.math_clear_notice = ""
        self.live_sign = "--"

    def _generate_math_problem(self, operation: MathOperation) -> MathProblem:
        if operation == MathOperation.ADDITION:
            answer = random.randint(1, 18)
            left = random.randint(1, answer - 1)
            return MathProblem(left, answer - left, operation)
        if operation == MathOperation.SUBTRACTION:
            left = random.randint(2, 18)
            right = random.randint(1, left - 1)
            return MathProblem(left, right, operation)
        left = random.randint(1, 9)
        right = random.randint(1, 9)
        while left * right > 99:
            left = random.randint(1, 9)
            right = random.randint(1, 9)
        return MathProblem(left, right, operation)

    def _start_spelling(self) -> None:
        self.prev_state = self.state
        self.state = GameState.SPELLING_BEE_MODE
        self.state_banner = "Spelling Bee"
        self.spelling_word = random.choice(SPELLING_WORDS)
        self.spelling_index = 0
        self.spelling_locked = ""
        self.spelling_debouncer.reset()
        self.spelling_message = ""
        self.spelling_unlocked = [False] * len(self.spelling_word)
        self.live_sign = "--"
        self.particles.clear()

    # ----------------------------------------------------------- prediction
    def _predict(self, feature: Optional[np.ndarray]) -> tuple[Optional[str], float]:
        if feature is None:
            return None, 0.0
        try:
            probs = self.model.predict_proba(feature.reshape(1, -1))[0]
            best_idx = int(np.argmax(probs))
            label = str(self.label_encoder.inverse_transform([best_idx])[0]).upper()
            return label, float(probs[best_idx])
        except Exception:
            return None, 0.0

    def _filter_prediction_for_mode(self, raw_label: Optional[str]) -> tuple[Optional[str], bool]:
        if raw_label is None:
            return None, False
        if self.state == GameState.MATH_MODE:
            return (raw_label, True) if raw_label.isdigit() else (None, False)
        if self.state == GameState.SPELLING_BEE_MODE:
            if raw_label.isalpha() and len(raw_label) == 1:
                return raw_label, True
            return None, False
        return raw_label, True

    # ---------------------------------------------------------------- update
    def _update(self) -> None:
        self._refresh_hint_ui()
        ok, frame = self.detector.read_frame()
        if not ok:
            self.camera_surface = None
            self.hand_present = False
            self._on_no_hand()
            return

        feature, landmarks = self.detector.process(frame)
        self.hand_present = landmarks is not None

        if landmarks is not None:
            self.detector.draw_landmarks(frame, landmarks)
            self.math_no_hand_since = None
        else:
            self._on_no_hand()

        raw_prediction, raw_confidence = self._predict(feature)
        mode_prediction, is_valid = self._filter_prediction_for_mode(raw_prediction)

        if self.state in (GameState.MATH_MODE, GameState.SPELLING_BEE_MODE):
            self.live_sign = mode_prediction if is_valid else "--"
            self.prediction_confidence = raw_confidence if is_valid else 0.0
        else:
            self.live_sign = raw_prediction or "--"
            self.prediction_confidence = raw_confidence

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.camera_surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

        if self.state == GameState.MATH_MODE and not self.hint_popup.blocks_game_input:
            self._update_math(mode_prediction if is_valid else None)
        elif self.state == GameState.SPELLING_BEE_MODE and not self.hint_popup.blocks_game_input:
            self._update_spelling(mode_prediction if is_valid else None)

        self._update_particles()

    def _on_no_hand(self) -> None:
        if self.state != GameState.MATH_MODE:
            return
        now = time.time()
        if self.math_no_hand_since is None:
            self.math_no_hand_since = now
            return

        elapsed = now - self.math_no_hand_since
        if self.math_input_buffer and elapsed >= MATH_CLEAR_HAND_SEC:
            self.math_input_buffer = ""
            self.math_debouncer.reset()
            self.math_clear_notice = "Hand dropped — buffer cleared"
            self.math_clear_notice_until = now + 2.0
            self.math_no_hand_since = None
            return

        if self.math_input_buffer and self.math_last_digit_time > 0:
            idle = now - self.math_last_digit_time
            if idle >= MATH_IDLE_SUBMIT_SEC:
                self._submit_math_answer()

    def _update_math(self, prediction: Optional[str]) -> None:
        if self.math_problem is None:
            return

        locked_digit = self.math_debouncer.update(prediction)
        if locked_digit is not None:
            if not self.math_input_buffer or self.math_input_buffer[-1] != locked_digit:
                self.math_input_buffer += locked_digit
                self.math_last_digit_time = time.time()
                self.combo_flash_until = time.time() + COMBO_FLASH_SEC

        answer_len = len(self.math_problem.answer_str)
        if self.math_input_buffer and len(self.math_input_buffer) >= answer_len:
            self._submit_math_answer()
            return

        if self.math_input_buffer and self.math_last_digit_time > 0:
            if time.time() - self.math_last_digit_time >= MATH_IDLE_SUBMIT_SEC:
                self._submit_math_answer()

    def _submit_math_answer(self) -> None:
        if self.math_problem is None or not self.math_input_buffer:
            return

        try:
            submitted = int(self.math_input_buffer)
        except ValueError:
            self.math_input_buffer = ""
            return

        expected = self.math_problem.answer
        if submitted == expected:
            elapsed = time.time() - self.math_start
            self.math_message = f"Correct! {expected} in {elapsed:.1f}s"
            self.success_flash_until = time.time() + SUCCESS_FLASH_SEC
            self._spawn_particles(RIGHT_PANEL_X + RIGHT_PANEL_W // 2, 300)
        else:
            self.math_message = "Wrong combination — try again."

        self.math_input_buffer = ""
        self.math_debouncer.reset()
        self.math_last_digit_time = 0.0

    def _on_spacebar_advance(self) -> None:
        if self.state == GameState.MATH_MODE:
            if self.math_problem is None:
                return
            self.math_input_buffer = ""
            self.math_debouncer.reset()
            self.math_last_digit_time = 0.0
            self.math_problem = self._generate_math_problem(self.math_operation)
            self.math_message = "Skipped — next question"
            self.live_sign = "--"
        elif self.state == GameState.SPELLING_BEE_MODE:
            self._start_spelling()
            self.spelling_message = "Skipped — next word"

    def _update_spelling(self, prediction: Optional[str]) -> None:
        if not self.spelling_word or self.spelling_index >= len(self.spelling_word):
            return

        target = self.spelling_word[self.spelling_index]
        candidate = prediction if prediction == target else None
        locked_letter = self.spelling_debouncer.update(candidate)

        if locked_letter is None:
            return

        self.spelling_unlocked[self.spelling_index] = True
        self.spelling_locked = " ".join(self.spelling_word[i] for i, ok in enumerate(self.spelling_unlocked) if ok)
        self.spelling_index += 1
        self.combo_flash_until = time.time() + COMBO_FLASH_SEC
        self.spelling_message = f"Locked '{locked_letter}'"

        if self.spelling_index >= len(self.spelling_word):
            self.spelling_message = f"Word complete: {self.spelling_word}!"
            self.success_flash_until = time.time() + SUCCESS_FLASH_SEC
            self._spawn_particles(RIGHT_PANEL_X + RIGHT_PANEL_W // 2, 280)

    def _spawn_particles(self, cx: int, cy: int) -> None:
        palette = [C_MINT, C_BLUE, (255, 220, 120), (255, 140, 200)]
        for _ in range(100):
            self.particles.append(
                Particle(
                    x=float(cx),
                    y=float(cy),
                    vx=random.uniform(-5, 5),
                    vy=random.uniform(-7, -1),
                    color=random.choice(palette),
                    life=random.uniform(0.8, 1.8),
                )
            )

    def _update_particles(self) -> None:
        alive: list[Particle] = []
        for p in self.particles:
            p.x += p.vx
            p.y += p.vy
            p.vy += 0.18
            p.life -= 0.025
            if p.life > 0:
                alive.append(p)
        self.particles = alive

    # ----------------------------------------------------------------- draw
    def _draw(self) -> None:
        self.screen.fill(C_BG)
        self._draw_left_camera_panel()
        self._draw_right_hud_panel()
        self._draw_state_transition_overlay()
        self._draw_success_flash()
        self._draw_particles()
        self.hint_button.draw(self.screen)
        self.hint_popup.draw(self.screen)
        pygame.display.flip()

    def _draw_left_camera_panel(self) -> None:
        title = self.font_heading.render("Live Camera", True, C_TEXT)
        self.screen.blit(title, (CAMERA_X, 36))

        outer = pygame.Rect(CAMERA_X - 8, CAMERA_Y - 8, CAMERA_W + 16, CAMERA_H + 16)
        pygame.draw.rect(self.screen, C_BLUE, outer, width=2, border_radius=18)

        inner = pygame.Rect(CAMERA_X, CAMERA_Y, CAMERA_W, CAMERA_H)
        pygame.draw.rect(self.screen, (12, 14, 20), inner, border_radius=14)

        if self.camera_surface is not None:
            scaled = pygame.transform.smoothscale(self.camera_surface, (CAMERA_W, CAMERA_H))
            self.screen.blit(scaled, inner.topleft)

        self._draw_geometric_frame(inner)
        status = "Hand tracked" if self.hand_present else "No hand detected"
        color = C_MINT if self.hand_present else C_DIM
        stat_surf = self.font_small.render(status, True, color)
        self.screen.blit(stat_surf, (CAMERA_X, CAMERA_Y + CAMERA_H + 18))

    def _draw_geometric_frame(self, rect: pygame.Rect) -> None:
        accent = C_NEON
        corner = 28
        lw = 3
        x, y, w, h = rect.x, rect.y, rect.w, rect.h
        for dx, dy in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            ox = x + (0 if dx > 0 else w)
            oy = y + (0 if dy > 0 else h)
            pygame.draw.line(self.screen, accent, (ox, oy), (ox + dx * corner, oy), lw)
            pygame.draw.line(self.screen, accent, (ox, oy), (ox, oy + dy * corner), lw)

    def _draw_right_hud_panel(self) -> None:
        panel = pygame.Rect(RIGHT_PANEL_X + 16, 16, RIGHT_PANEL_W - 32, WINDOW_HEIGHT - 32)
        pygame.draw.rect(self.screen, C_PANEL, panel, border_radius=20)

        fade = min(1.0, (time.time() - self.state_changed_at) / STATE_FADE_SEC)
        slide = int((1.0 - fade) * 30)

        brand = self.font_title.render("SignPlay", True, C_MINT)
        self.screen.blit(brand, (panel.x + 28, panel.y + 24 - slide))

        banner = self.font_body.render(self.state_banner, True, C_DIM)
        self.screen.blit(banner, (panel.x + 28, panel.y + 84 - slide))

        if self.state == GameState.MAIN_MENU:
            self._draw_main_menu(panel, slide)
        elif self.state == GameState.MATH_MODE:
            self._draw_math_hud(panel)
        elif self.state == GameState.SPELLING_BEE_MODE:
            self._draw_spelling_hud(panel)

    def _draw_main_menu(self, panel: pygame.Rect, slide: int) -> None:
        items = [
            ("1", "Addition"),
            ("2", "Subtraction"),
            ("3", "Multiplication"),
            ("4", "Spelling Bee"),
        ]
        y = panel.y + 150 - slide
        for key, label in items:
            row = self.font_body.render(f"[{key}]  {label}", True, C_TEXT)
            self.screen.blit(row, (panel.x + 36, y))
            y += 48

    def _draw_math_hud(self, panel: pygame.Rect) -> None:
        prompt = self.math_problem.prompt() if self.math_problem else "..."
        self.screen.blit(self.font_heading.render(prompt, True, C_TEXT), (panel.x + 28, panel.y + 130))

        elapsed = time.time() - self.math_start if self.math_start else 0.0
        self.screen.blit(
            self.font_small.render(f"Timer: {elapsed:.1f}s", True, C_BLUE),
            (panel.x + 28, panel.y + 175),
        )

        locked_display = " ".join(self.math_input_buffer) if self.math_input_buffer else "—"
        self._draw_combination_box(
            panel,
            pygame.Rect(panel.x + 24, panel.y + 230, panel.w - 48, 150),
            locked_display,
        )

        if self.math_message:
            color = C_MINT if self.math_message.startswith("Correct") else C_ERROR
            self.screen.blit(self.font_body.render(self.math_message, True, color), (panel.x + 28, panel.y + 400))

        if time.time() < self.math_clear_notice_until:
            self.screen.blit(
                self.font_small.render(self.math_clear_notice, True, C_BLUE),
                (panel.x + 28, panel.y + 440),
            )

        self.screen.blit(
            self.font_small.render("Drop hand 1.5s clear · Idle 2s submit · Space: next question", True, C_DIM),
            (panel.x + 28, panel.bottom - 48),
        )

    def _draw_spelling_hud(self, panel: pygame.Rect) -> None:
        y = panel.y + 130
        x = panel.x + 28
        for i, letter in enumerate(self.spelling_word):
            if self.spelling_unlocked[i]:
                color = C_MINT
            elif i == self.spelling_index:
                color = C_BLUE
            else:
                color = C_TEXT
            self.screen.blit(self.font_title.render(letter, True, color), (x, y))
            x += 46

        target = self.spelling_word[self.spelling_index] if self.spelling_index < len(self.spelling_word) else "✓"
        self.screen.blit(
            self.font_body.render(f"Target letter: {target}", True, C_BLUE),
            (panel.x + 28, panel.y + 195),
        )

        locked_display = self.spelling_locked if self.spelling_locked else "—"
        self._draw_combination_box(
            panel,
            pygame.Rect(panel.x + 24, panel.y + 240, panel.w - 48, 150),
            locked_display,
        )

        track, count = self.spelling_debouncer.progress
        if track:
            self.screen.blit(
                self.font_small.render(f"Hold: {count}/{HOLD_FRAMES}", True, C_DIM),
                (panel.x + 28, panel.y + 405),
            )

        if self.spelling_message:
            self.screen.blit(self.font_body.render(self.spelling_message, True, C_TEXT), (panel.x + 28, panel.y + 435))

    def _draw_combination_box(self, panel: pygame.Rect, rect: pygame.Rect, locked_text: str) -> None:
        flash = time.time() < self.combo_flash_until
        border_color = C_MINT if flash else C_NEON

        overlay = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        overlay.fill((C_COMBO_BG[0], C_COMBO_BG[1], C_COMBO_BG[2], 210))
        self.screen.blit(overlay, rect.topleft)
        pygame.draw.rect(self.screen, border_color, rect, width=2, border_radius=16)

        title = self.font_combo_label.render("COMBINATION TRACKER", True, C_DIM)
        self.screen.blit(title, (rect.x + 16, rect.y + 12))

        live_label = self.font_combo_label.render("Live Sign", True, C_BLUE)
        live_value = self.font_combo.render(self.live_sign, True, C_TEXT)
        self.screen.blit(live_label, (rect.x + 16, rect.y + 44))
        self.screen.blit(live_value, (rect.x + 16, rect.y + 66))

        lock_label = self.font_combo_label.render("Locked Combination", True, C_MINT)
        lock_value = self.font_combo.render(locked_text, True, C_MINT if flash else C_TEXT)
        self.screen.blit(lock_label, (rect.x + 16, rect.y + 96))
        self.screen.blit(lock_value, (rect.x + 16, rect.y + 118))

        conf = self.font_small.render(f"Confidence: {self.prediction_confidence:.2f}", True, C_DIM)
        self.screen.blit(conf, (rect.right - conf.get_width() - 16, rect.y + 14))

    def _draw_state_transition_overlay(self) -> None:
        elapsed = time.time() - self.state_changed_at
        if elapsed >= STATE_FADE_SEC:
            return
        alpha = int(180 * (1.0 - elapsed / STATE_FADE_SEC))
        fade = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        fade.fill((30, 30, 36, alpha))
        self.screen.blit(fade, (0, 0))

    def _draw_success_flash(self) -> None:
        if time.time() >= self.success_flash_until:
            return
        remaining = self.success_flash_until - time.time()
        alpha = int(120 * (remaining / SUCCESS_FLASH_SEC))
        flash = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        flash.fill((78, 225, 160, alpha))
        self.screen.blit(flash, (0, 0))

    def _draw_particles(self) -> None:
        for p in self.particles:
            radius = max(2, int(p.life * 7))
            pygame.draw.circle(self.screen, p.color, (int(p.x), int(p.y)), radius)


def main() -> None:
    try:
        SignPlayGame().run()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
