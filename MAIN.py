import pygame
import numpy as np
import soundfile as sf
import math
import random
import os
import io
import tkinter as tk
from tkinter import filedialog
from mutagen.id3 import ID3, APIC
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
try:
    import moviepy.editor as mpy  # Optional: kept for compatibility with earlier builds.
except Exception:
    mpy = None

# -------------------------------------------------------------------------
# CONSTANTS & LAYOUT
# -------------------------------------------------------------------------
SAMPLE_COUNT = 144
WIDTH, HEIGHT = 1000, 820
CENTER = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
PATH_MODES = ("LETTER", "CIRCLE", "LINE")


def get_button_rects(w, h):
    bw, bh, gap = 118, 30, 8
    panel_top = h - 182
    col0, col1, col2 = 22, 22 + bw + gap, 22 + (bw + gap) * 2

    def row(r):
        return panel_top + 54 + r * (bh + gap)

    return {
        "open": pygame.Rect(col0, row(0), bw, bh),
        "path_mode": pygame.Rect(col0, row(1), bw, bh),
        "toggle_bars": pygame.Rect(col0, row(2), bw, bh),
        "toggle_image": pygame.Rect(col0, row(3), bw, bh),
        "toggle_part": pygame.Rect(col1, row(0), bw, bh),
        "toggle_text": pygame.Rect(col1, row(1), bw, bh),
        "toggle_rgb": pygame.Rect(col1, row(2), bw, bh),
        "toggle_mblur": pygame.Rect(col1, row(3), bw, bh),
        "toggle_blur": pygame.Rect(col2, row(0), bw, bh),
        "toggle_mirror": pygame.Rect(col2, row(1), bw, bh),
        "toggle_color": pygame.Rect(col2, row(2), bw, bh),
        "toggle_glow": pygame.Rect(col2, row(3), bw, bh),
    }, pygame.Rect(w - 245, panel_top + 74, 190, 12), pygame.Rect(0, panel_top, w, h - panel_top)


BUTTON_RECTS, SLIDER_RECT, PANEL_RECT = get_button_rects(WIDTH, HEIGHT)

# -------------------------------------------------------------------------
# AUDIO DSP PIPELINE
# -------------------------------------------------------------------------
WINDOW = np.hanning(2048)
LOG_EDGES = np.logspace(np.log10(35.0), np.log10(17500.0), SAMPLE_COUNT + 1)
BEAT_BANDS = {"sub_bass": (0, 5), "kick": (5, 13), "low_mid": (14, 30), "presence": (44, 76)}
beat_energy_history = {band: [] for band in BEAT_BANDS}
_bar_lo = _bar_hi = None


def _precompute_bar_bins(rate):
    global _bar_lo, _bar_hi
    freqs = np.fft.rfftfreq(2048, d=1.0 / rate)
    _bar_lo = np.searchsorted(freqs, LOG_EDGES[:-1])
    _bar_hi = np.searchsorted(freqs, LOG_EDGES[1:])


# -------------------------------------------------------------------------
# RENDER STATE & GLOBALS
# -------------------------------------------------------------------------
class RenderState:
    def __init__(self):
        self.dt = 0.016
        self.fft = np.zeros(SAMPLE_COUNT)
        self.smooth_bars = np.zeros(SAMPLE_COUNT)
        self.peak_bars = np.zeros(SAMPLE_COUNT)
        self.peak_timer = np.zeros(SAMPLE_COUNT)
        self.is_beat = False
        self.smooth_beat = 0.0
        self.energy = 0.0
        self.shake = np.array([0.0, 0.0])
        self.rotation = 0.0
        self.time = 0.0
        self.original_cover = None
        self.path_mode_index = 0

    @property
    def path_mode(self):
        return PATH_MODES[self.path_mode_index % len(PATH_MODES)]


RS = RenderState()
data, samplerate, sound = None, None, None
title, artist = "Avee Pygame", "Pure CPU Visualizer"
volume = 0.82
show_ui = True

_font_cache = {}


def get_font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        if not pygame.font.get_init():
            pygame.font.init()
        # pygame.font.SysFont can crash on some Windows font registries when
        # scanning installed fonts. Use the bundled default font first so the
        # visualizer starts reliably on every Pygame install, then fall back to
        # SysFont only if the default loader is unavailable.
        try:
            font = pygame.font.Font(None, size)
        except Exception:
            font = pygame.font.SysFont(None, size, bold=bold)
        font.set_bold(bold)
        _font_cache[key] = font
    return _font_cache[key]


# -------------------------------------------------------------------------
# AVEE-LIKE NODE ARCHITECTURE (Element & Composition)
# -------------------------------------------------------------------------
class Element:
    def __init__(self):
        self.enabled = True

    def update(self, rs: RenderState):
        pass

    def render(self, surface: pygame.Surface, rs: RenderState):
        pass


class Composition(Element):
    def __init__(self):
        super().__init__()
        self.children = []

    def add(self, child):
        self.children.append(child)

    def update(self, rs):
        for c in self.children:
            if c.enabled:
                c.update(rs)

    def render(self, surface, rs):
        for c in self.children:
            if c.enabled:
                c.render(surface, rs)


class AudioDataProviderElement(Element):
    def __init__(self):
        super().__init__()
        self.shake_target = np.array([0.0, 0.0])

    def update(self, rs: RenderState):
        rs.time += rs.dt
        if data is not None and sound and pygame.mixer.get_busy():
            idx = int(rs.time * samplerate)
            if idx + 2048 < len(data):
                chunk = data[idx:idx + 2048]
                fft_raw = np.abs(np.fft.rfft(chunk * WINDOW))
                for b in range(SAMPLE_COUNT):
                    lo, hi = _bar_lo[b], _bar_hi[b]
                    if hi > lo:
                        rs.fft[b] = np.mean(fft_raw[lo:min(hi, len(fft_raw))])
                    elif lo < len(fft_raw):
                        rs.fft[b] = fft_raw[lo]
                max_v = np.max(rs.fft)
                if max_v > 1e-4:
                    rs.fft /= max_v
        else:
            # Idle demo signal so the UI/effects are visible before loading a track.
            phase = rs.time * 2.0
            xs = np.linspace(0, 1, SAMPLE_COUNT)
            rs.fft = np.maximum(0.0, np.sin(xs * 9.0 + phase) * 0.25 + np.sin(xs * 31.0 - phase * 1.7) * 0.18)

        beat_conf = 0.0
        for band, (lo, hi) in BEAT_BANDS.items():
            en = float(np.mean(rs.fft[lo:min(hi, len(rs.fft))]))
            hist = beat_energy_history[band]
            hist.append(en)
            if len(hist) > 46:
                hist.pop(0)
            if len(hist) > 6 and np.mean(hist[:-1]) > 1e-6:
                ratio = en / np.mean(hist[:-1])
                if ratio > 1.28:
                    beat_conf += 0.38 * ratio

        rs.energy = rs.energy * 0.82 + float(np.mean(rs.fft)) * 0.18
        rs.is_beat = beat_conf > 0.78
        if rs.is_beat:
            rs.smooth_beat = rs.smooth_beat * 0.34 + min(beat_conf / 2.0, 1.0) * 0.66
        else:
            rs.smooth_beat *= 0.88

        for i in range(SAMPLE_COUNT):
            target = rs.fft[i]
            if target > rs.smooth_bars[i]:
                rs.smooth_bars[i] = rs.smooth_bars[i] * 0.18 + target * 0.82
            else:
                rs.smooth_bars[i] = max(0.0, rs.smooth_bars[i] - (2.4 + i / SAMPLE_COUNT) * rs.dt)

            if rs.smooth_bars[i] > rs.peak_bars[i]:
                rs.peak_bars[i] = rs.smooth_bars[i]
                rs.peak_timer[i] = 0.28
            elif rs.peak_timer[i] > 0:
                rs.peak_timer[i] -= rs.dt
            else:
                rs.peak_bars[i] = max(0.0, rs.peak_bars[i] - 1.0 * rs.dt)

        if rs.is_beat:
            amt = 9.0 * rs.smooth_beat
            self.shake_target = np.array([random.uniform(-amt, amt), random.uniform(-amt, amt)])
        else:
            self.shake_target *= 0.82
        rs.shake = rs.shake * 0.62 + self.shake_target * 0.38
        rs.rotation += 0.005 + rs.smooth_beat * 0.025


class ImageElement(Element):
    def render(self, surface, rs):
        cx, cy = int(CENTER[0] + rs.shake[0]), int(CENTER[1] + rs.shake[1])
        rad = int(min(WIDTH, HEIGHT) * 0.145) + int(rs.smooth_beat * 18)
        glow_rad = rad + 24 + int(rs.energy * 55)
        glow = pygame.Surface((glow_rad * 2, glow_rad * 2), pygame.SRCALPHA)
        for r in range(glow_rad, 0, -8):
            alpha = max(0, int(42 * (r / glow_rad) ** 2))
            pygame.draw.circle(glow, (0, 210, 255, alpha), (glow_rad, glow_rad), r)
        surface.blit(glow, (cx - glow_rad, cy - glow_rad), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.circle(surface, (6, 7, 11, 180), (cx + 5, cy + 7), rad + 6)
        if rs.original_cover:
            sz = rad * 2
            pg_img = pygame.transform.smoothscale(rs.original_cover, (sz, sz))
            mask = pygame.Surface((sz, sz), pygame.SRCALPHA)
            pygame.draw.circle(mask, (255, 255, 255, 255), (rad, rad), rad)
            pg_img.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            surface.blit(pg_img, (cx - rad, cy - rad))
        else:
            pygame.draw.circle(surface, (20, 22, 30), (cx, cy), rad)
            pygame.draw.circle(surface, (0, 235, 255), (cx, cy), rad, 3)
            note = get_font(64, True).render("♫", True, (230, 250, 255))
            surface.blit(note, note.get_rect(center=(cx, cy - 4)))


class LetterPathCache:
    def __init__(self):
        self.key = None
        self.points = []

    def get_points(self, text, count):
        key = (text, count, WIDTH, HEIGHT)
        if key == self.key and self.points:
            return self.points
        font = get_font(186, True)
        mask_surf = font.render(text, True, (255, 255, 255))
        pad = 28
        canvas = pygame.Surface((mask_surf.get_width() + pad * 2, mask_surf.get_height() + pad * 2), pygame.SRCALPHA)
        canvas.blit(mask_surf, (pad, pad))
        mask = pygame.mask.from_surface(canvas)
        outlines = mask.outlines()
        raw = []
        for outline in outlines:
            if len(outline) > 22:
                raw.extend(outline[::max(1, len(outline) // 180)])
        if len(raw) < 8:
            self.points = []
            return self.points
        arr = np.array(raw, dtype=float)
        arr[:, 0] -= canvas.get_width() / 2
        arr[:, 1] -= canvas.get_height() / 2
        scale = min(WIDTH * 0.62 / max(1, canvas.get_width()), HEIGHT * 0.36 / max(1, canvas.get_height()))
        arr *= scale
        center = np.array([WIDTH / 2, HEIGHT / 2 + 10], dtype=float)
        arr += center
        angles = np.arctan2(arr[:, 1] - center[1], arr[:, 0] - center[0])
        order = np.argsort(angles)
        ordered = arr[order]
        idx = np.linspace(0, len(ordered) - 1, count).astype(int)
        pts = ordered[idx]
        self.points = [(float(x), float(y)) for x, y in pts]
        self.key = key
        return self.points


class SegmentElement(Element):
    def __init__(self):
        super().__init__()
        self.letter_cache = LetterPathCache()

    def _letter_text(self):
        clean = "".join(ch for ch in title.upper() if ch.isalnum())
        return (clean[:5] or "AVEE")

    def _path_sample(self, i, rs):
        mode = rs.path_mode
        cx, cy = CENTER[0] + rs.shake[0], CENTER[1] + rs.shake[1]
        if mode == "LETTER":
            pts = self.letter_cache.get_points(self._letter_text(), SAMPLE_COUNT)
            if pts:
                x, y = pts[i]
                x += rs.shake[0]
                y += rs.shake[1]
                x2, y2 = pts[(i + 1) % SAMPLE_COUNT]
                tx, ty = x2 - pts[i][0], y2 - pts[i][1]
                ln = math.hypot(tx, ty) or 1.0
                tx, ty = tx / ln, ty / ln
                nx, ny = -ty, tx
                if (x - cx) * nx + (y - cy) * ny < 0:
                    nx, ny = -nx, -ny
                return x, y, nx, ny, tx, ty
        if mode == "LINE":
            span = WIDTH * 0.76
            x = cx - span / 2 + span * (i / max(1, SAMPLE_COUNT - 1))
            y = cy + math.sin((i / SAMPLE_COUNT) * math.pi * 4 + rs.time * 1.6) * (22 + 28 * rs.energy)
            return x, y, 0.0, -1.0, 1.0, 0.0
        base_r = min(WIDTH, HEIGHT) * 0.23 + (rs.smooth_beat * 24)
        ang = (i / SAMPLE_COUNT) * 2 * math.pi + rs.rotation
        return cx + math.cos(ang) * base_r, cy + math.sin(ang) * base_r, math.cos(ang), math.sin(ang), -math.sin(ang), math.cos(ang)

    def render(self, surface, rs):
        max_h = min(WIDTH, HEIGHT) * (0.23 if rs.path_mode != "LETTER" else 0.15)
        bar_w = 3.6 if rs.path_mode == "LETTER" else 5.2
        for i in range(SAMPLE_COUNT):
            v, p = float(rs.smooth_bars[i]), float(rs.peak_bars[i])
            h = max(2.0, v * max_h)
            x, y, nx, ny, tx, ty = self._path_sample(i, rs)
            wobble = math.sin(rs.time * 4.5 + i * 0.17) * rs.smooth_beat * 7.0
            x += nx * wobble
            y += ny * wobble
            x0, y0 = x + nx * h, y + ny * h
            x1, y1 = x0 + tx * bar_w, y0 + ty * bar_w
            x2, y2 = x + tx * bar_w, y + ty * bar_w
            hue = i / SAMPLE_COUNT
            col = (
                int(90 + 90 * math.sin(hue * math.tau + rs.time) + 65 * v),
                int(175 + 55 * math.sin(hue * math.tau + 2.2)),
                255,
                235,
            )
            pygame.draw.polygon(surface, col, [(x0, y0), (x1, y1), (x2, y2), (x, y)])
            if p > 0.05:
                px, py = x + nx * (p * max_h + 9), y + ny * (p * max_h + 9)
                pygame.draw.circle(surface, (255, 255, 255, 210), (int(px), int(py)), 2)


class TextElement(Element):
    def render(self, surface, rs):
        if show_ui:
            return
        t_font, a_font = get_font(30, True), get_font(18, False)
        t_surf = t_font.render(title[:34], True, (255, 255, 255))
        a_surf = a_font.render(f"{artist[:40]}  •  {rs.path_mode} path  •  H hides UI", True, (125, 220, 255))
        surface.blit(t_surf, (46, HEIGHT - 112))
        surface.blit(a_surf, (48, HEIGHT - 74))


class Particle:
    __slots__ = ["x", "y", "vx", "vy", "life", "max", "size"]

    def __init__(self, x, y, vx, vy, life, size):
        self.x, self.y, self.vx, self.vy, self.life, self.max, self.size = x, y, vx, vy, life, life, size


class ParticlesElement(Element):
    def __init__(self):
        super().__init__()
        self.particles = []

    def update(self, rs):
        if rs.is_beat and rs.smooth_beat > 0.35:
            cx, cy = CENTER[0] + rs.shake[0], CENTER[1] + rs.shake[1]
            for _ in range(int(5 + rs.smooth_beat * 18)):
                a = random.uniform(0, math.tau)
                s = random.uniform(70, 260) * (0.4 + rs.smooth_beat)
                self.particles.append(Particle(cx, cy, math.cos(a) * s, math.sin(a) * s, random.uniform(0.7, 2.4), random.uniform(1.5, 4.5)))
        for p in self.particles[:]:
            p.x += p.vx * rs.dt
            p.y += p.vy * rs.dt
            p.vx *= 0.992
            p.vy *= 0.992
            p.life -= rs.dt
            if p.life <= 0:
                self.particles.remove(p)

    def render(self, surface, rs):
        for p in self.particles:
            alpha = int(210 * (p.life / p.max))
            pygame.draw.circle(surface, (90, 220, 255, alpha), (int(p.x), int(p.y)), max(1, int(p.size * (p.life / p.max))))


class GlowEffectElement(Element):
    def render(self, surface, rs):
        scale = 0.34
        tiny = pygame.transform.smoothscale(surface, (max(1, int(WIDTH * scale)), max(1, int(HEIGHT * scale))))
        glow = pygame.transform.smoothscale(tiny, (WIDTH, HEIGHT))
        glow.set_alpha(int(70 + 90 * rs.smooth_beat))
        surface.blit(glow, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)


class RgbSplitEffectElement(Element):
    def render(self, surface, rs):
        amt = int(2 + 18 * rs.smooth_beat + 7 * rs.energy)
        if amt <= 0:
            return
        red, green, blue = surface.copy(), surface.copy(), surface.copy()
        red.fill((255, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
        green.fill((0, 255, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
        blue.fill((0, 0, 255, 255), special_flags=pygame.BLEND_RGBA_MULT)
        surface.fill((0, 0, 0, 0))
        # Avee-style RGB slit: three colored taps in different directions, no shader required.
        surface.blit(green, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        surface.blit(red, (-amt, int(-amt * 0.22)), special_flags=pygame.BLEND_RGBA_ADD)
        surface.blit(blue, (amt, int(amt * 0.22)), special_flags=pygame.BLEND_RGBA_ADD)


class MotionBlurEffectElement(Element):
    def __init__(self):
        super().__init__()
        self.history = []

    def render(self, surface, rs):
        self.history.append(surface.copy())
        if len(self.history) > 6:
            self.history.pop(0)
        for idx, frame in enumerate(reversed(self.history[:-1])):
            alpha = max(14, int((92 - idx * 15) * (1.0 - rs.smooth_beat * 0.35)))
            frame.set_alpha(alpha)
            off = idx + 1
            dx = int(-rs.shake[0] * 0.18 * off)
            dy = int(-rs.shake[1] * 0.18 * off)
            surface.blit(frame, (dx, dy), special_flags=pygame.BLEND_RGBA_ADD)


class MirrorEffectElement(Element):
    def render(self, surface, rs):
        half = surface.subsurface((0, 0, WIDTH // 2, HEIGHT)).copy()
        flipped = pygame.transform.flip(half, True, False)
        surface.blit(flipped, (WIDTH // 2, 0))


class BlurEffectElement(Element):
    def render(self, surface, rs):
        scale = max(0.16, 0.46 - (rs.smooth_beat * 0.22))
        sw, sh = max(1, int(WIDTH * scale)), max(1, int(HEIGHT * scale))
        tiny = pygame.transform.smoothscale(surface, (sw, sh))
        blurred = pygame.transform.smoothscale(tiny, (WIDTH, HEIGHT))
        blurred.set_alpha(105)
        surface.blit(blurred, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)


class ColorCorrectionElement(Element):
    def render(self, surface, rs):
        tint = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        tint.fill((0, 24 + int(38 * rs.smooth_beat), 58, 36))
        surface.blit(tint, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        contrast = pygame.Surface((WIDTH, HEIGHT))
        val = int(132 + 44 * rs.smooth_beat)
        contrast.fill((val, val, val))
        surface.blit(contrast, (0, 0), special_flags=pygame.BLEND_RGB_MULT)


# -------------------------------------------------------------------------
# SCENE BUILDER
# -------------------------------------------------------------------------
Root = Composition()
CompAudio = AudioDataProviderElement()
CompParts = ParticlesElement()
CompBars = SegmentElement()
CompImage = ImageElement()
CompText = TextElement()
PPE_Glow = GlowEffectElement()
PPE_RGB = RgbSplitEffectElement()
PPE_MBlur = MotionBlurEffectElement()
PPE_Mirror = MirrorEffectElement()
PPE_Blur = BlurEffectElement()
PPE_Color = ColorCorrectionElement()

PPE_RGB.enabled = True
PPE_MBlur.enabled = True
PPE_Glow.enabled = True
PPE_Mirror.enabled = False
PPE_Blur.enabled = False
PPE_Color.enabled = True

for element in (CompAudio, CompParts, CompBars, CompImage, CompText, PPE_Glow, PPE_Blur, PPE_MBlur, PPE_Mirror, PPE_Color, PPE_RGB):
    Root.add(element)


# -------------------------------------------------------------------------
# ASSET HELPERS
# -------------------------------------------------------------------------
def extract_cover(file):
    try:
        ext = os.path.splitext(file)[1].lower()
        if ext == ".mp3":
            tags = ID3(file)
            for tag in tags.values():
                if isinstance(tag, APIC):
                    return pygame.image.load(io.BytesIO(tag.data)).convert()
        elif ext == ".flac":
            audio = FLAC(file)
            if audio.pictures:
                return pygame.image.load(io.BytesIO(audio.pictures[0].data)).convert()
        elif ext in (".m4a", ".mp4", ".aac"):
            audio = MP4(file)
            if "covr" in audio:
                return pygame.image.load(io.BytesIO(bytes(audio["covr"][0]))).convert()
    except Exception:
        pass
    return None


def reload_audio(file):
    global data, samplerate, sound, title, artist, RS
    try:
        data, samplerate = sf.read(file)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        pygame.mixer.quit()
        pygame.mixer.init(samplerate, -16, 2, 2048)
        sound = pygame.mixer.Sound(file)
        sound.set_volume(volume)
        sound.play()
        title = os.path.splitext(os.path.basename(file))[0]
        artist = "Loaded track"
        RS.time = 0.0
        RS.original_cover = extract_cover(file)
        _precompute_bar_bins(samplerate)
        for k in beat_energy_history:
            beat_energy_history[k].clear()
    except Exception as e:
        print(f"File load error: {e}")


# -------------------------------------------------------------------------
# UI DRAWING
# -------------------------------------------------------------------------
def draw_grid_background(screen, rs):
    screen.fill((4, 5, 10))
    horizon = int(HEIGHT * 0.58 + math.sin(rs.time * 0.7) * 12)
    for y in range(horizon, HEIGHT, 34):
        alpha = max(18, min(88, int((y - horizon) / max(1, HEIGHT - horizon) * 92)))
        pygame.draw.line(screen, (0, 120, 180, alpha), (0, y), (WIDTH, y), 1)
    for x in range(-WIDTH, WIDTH * 2, 54):
        skew = int(math.sin(rs.time * 0.25) * 22)
        pygame.draw.line(screen, (0, 62, 96), (WIDTH // 2, horizon), (x + skew, HEIGHT), 1)


def draw_ui(screen, clock, btn_states):
    panel = pygame.Surface((PANEL_RECT.width, PANEL_RECT.height), pygame.SRCALPHA)
    panel.fill((6, 9, 18, 218))
    pygame.draw.rect(panel, (0, 210, 255, 75), panel.get_rect(), 1, border_radius=18)
    screen.blit(panel, PANEL_RECT.topleft)

    header = get_font(21, True).render("AVEE PLAYER PY • CPU/PYGAME VISUALIZER", True, (240, 252, 255))
    screen.blit(header, (24, PANEL_RECT.y + 16))
    subtitle = get_font(13).render("Letter path bars, motion blur, RGB slit, glow, mirror, color correction — no shaders/OpenGL/Vulkan", True, (120, 205, 230))
    screen.blit(subtitle, (24, PANEL_RECT.y + 40))

    labels = {
        "open": "OPEN", "path_mode": f"PATH {RS.path_mode}", "toggle_bars": "BARS", "toggle_image": "IMAGE",
        "toggle_part": "PARTICLES", "toggle_text": "TEXT", "toggle_rgb": "RGB SLIT", "toggle_mblur": "MOTION BLUR",
        "toggle_blur": "SOFT BLUR", "toggle_mirror": "MIRROR", "toggle_color": "COLOR", "toggle_glow": "GLOW",
    }
    for name, rect in BUTTON_RECTS.items():
        if name == "open" or name == "path_mode":
            active = True
        else:
            active = btn_states[name].enabled
        bg_col = (0, 150, 170) if active else (30, 34, 48)
        hi_col = (55, 240, 255) if active else (82, 88, 108)
        pygame.draw.rect(screen, bg_col, rect, border_radius=8)
        pygame.draw.rect(screen, hi_col, rect, 1, border_radius=8)
        txt = get_font(11, True).render(labels[name], True, (245, 252, 255) if active else (176, 184, 198))
        screen.blit(txt, txt.get_rect(center=rect.center))

    pygame.draw.rect(screen, (38, 44, 58), SLIDER_RECT, border_radius=5)
    fill_w = int(volume * SLIDER_RECT.width)
    if fill_w > 0:
        pygame.draw.rect(screen, (0, 210, 255), (SLIDER_RECT.x, SLIDER_RECT.y, fill_w, SLIDER_RECT.height), border_radius=5)
    vol_label = get_font(13, True).render(f"VOLUME {int(volume * 100)}%", True, (220, 245, 255))
    screen.blit(vol_label, (SLIDER_RECT.x, SLIDER_RECT.y - 24))
    fps = get_font(13).render(f"{clock.get_fps():05.1f} FPS  •  H hide UI", True, (145, 180, 195))
    screen.blit(fps, (SLIDER_RECT.x, SLIDER_RECT.y + 20))


# -------------------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------------------
def main():
    global volume, show_ui, WIDTH, HEIGHT, CENTER, BUTTON_RECTS, SLIDER_RECT, PANEL_RECT
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Avee Player - Pure Pygame Render Tree")
    clock = pygame.time.Clock()
    _precompute_bar_bins(44100)

    btn_states = {
        "toggle_bars": CompBars,
        "toggle_image": CompImage,
        "toggle_part": CompParts,
        "toggle_text": CompText,
        "toggle_rgb": PPE_RGB,
        "toggle_blur": PPE_Blur,
        "toggle_mblur": PPE_MBlur,
        "toggle_mirror": PPE_Mirror,
        "toggle_color": PPE_Color,
        "toggle_glow": PPE_Glow,
    }

    running = True
    while running:
        RS.dt = min(0.05, clock.tick(60) / 1000.0)
        Root.update(RS)
        draw_grid_background(screen, RS)
        layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        Root.render(layer, RS)
        screen.blit(layer, (0, 0))
        if show_ui:
            draw_ui(screen, clock, btn_states)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = max(720, event.w), max(560, event.h)
                CENTER = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
                BUTTON_RECTS, SLIDER_RECT, PANEL_RECT = get_button_rects(WIDTH, HEIGHT)
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    show_ui = not show_ui
                elif event.key == pygame.K_TAB:
                    RS.path_mode_index = (RS.path_mode_index + 1) % len(PATH_MODES)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and show_ui:
                pos = event.pos
                if SLIDER_RECT.collidepoint(pos):
                    volume = max(0.0, min(1.0, (pos[0] - SLIDER_RECT.x) / SLIDER_RECT.width))
                    if sound:
                        sound.set_volume(volume)
                for name, rect in BUTTON_RECTS.items():
                    if rect.collidepoint(pos):
                        if name == "open":
                            root = tk.Tk()
                            root.withdraw()
                            f = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.flac *.m4a *.aac *.wav *.ogg")])
                            root.destroy()
                            if f:
                                reload_audio(f)
                        elif name == "path_mode":
                            RS.path_mode_index = (RS.path_mode_index + 1) % len(PATH_MODES)
                        elif name in btn_states:
                            btn_states[name].enabled = not btn_states[name].enabled

        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
