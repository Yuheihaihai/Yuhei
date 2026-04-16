"""Procedural nature-sound synthesizer (stdlib only).

Writes 16-bit mono WAV files for rain, wind, ocean, forest, and stream
using only math / random / wave from the Python standard library.
"""

from __future__ import annotations

import math
import os
import random
import struct
import wave

SAMPLE_RATE = 22050
DURATION = 10.0
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _write_wav(name: str, samples: list[float]) -> str:
    path = os.path.join(OUT_DIR, name)
    peak = max(1e-9, max(abs(s) for s in samples))
    gain = 0.9 / peak
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        frames = bytearray()
        for s in samples:
            v = int(max(-1.0, min(1.0, s * gain)) * 32767)
            frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))
    return path


def _white(n: int) -> list[float]:
    return [random.uniform(-1.0, 1.0) for _ in range(n)]


def _one_pole_lp(x: list[float], cutoff_hz: float) -> list[float]:
    # Single-pole IIR low-pass.
    dt = 1.0 / SAMPLE_RATE
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    a = dt / (rc + dt)
    y = [0.0] * len(x)
    prev = 0.0
    for i, v in enumerate(x):
        prev = prev + a * (v - prev)
        y[i] = prev
    return y


def _one_pole_hp(x: list[float], cutoff_hz: float) -> list[float]:
    dt = 1.0 / SAMPLE_RATE
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    a = rc / (rc + dt)
    y = [0.0] * len(x)
    prev_x = 0.0
    prev_y = 0.0
    for i, v in enumerate(x):
        prev_y = a * (prev_y + v - prev_x)
        prev_x = v
        y[i] = prev_y
    return y


def _pink(n: int) -> list[float]:
    # Voss-McCartney pink-noise approximation.
    num_rows = 16
    rows = [0.0] * num_rows
    out = [0.0] * n
    running = 0.0
    for i in range(n):
        idx = (i & -i).bit_length() - 1
        if 0 <= idx < num_rows:
            running -= rows[idx]
            rows[idx] = random.uniform(-1.0, 1.0)
            running += rows[idx]
        out[i] = (running + random.uniform(-1.0, 1.0)) / (num_rows + 1)
    return out


def rain() -> list[float]:
    n = int(SAMPLE_RATE * DURATION)
    bed = _one_pole_hp(_one_pole_lp(_white(n), 4000.0), 300.0)
    # Sparse droplet transients.
    drops = [0.0] * n
    i = 0
    while i < n:
        gap = random.randint(40, 300)
        i += gap
        if i >= n:
            break
        amp = random.uniform(0.2, 1.0)
        length = random.randint(30, 120)
        for k in range(length):
            if i + k >= n:
                break
            env = math.exp(-k / 20.0)
            drops[i + k] += amp * env * math.sin(2 * math.pi * random.uniform(1500, 4500) * k / SAMPLE_RATE)
    return [0.55 * bed[j] + 0.45 * drops[j] for j in range(n)]


def wind() -> list[float]:
    n = int(SAMPLE_RATE * DURATION)
    base = _one_pole_lp(_white(n), 700.0)
    # Slowly varying amplitude envelope (gusts).
    env = [0.0] * n
    phase1, phase2 = 0.0, 0.0
    for i in range(n):
        phase1 += 2 * math.pi * 0.12 / SAMPLE_RATE
        phase2 += 2 * math.pi * 0.37 / SAMPLE_RATE
        env[i] = 0.6 + 0.3 * math.sin(phase1) + 0.15 * math.sin(phase2)
    return [base[i] * env[i] for i in range(n)]


def ocean() -> list[float]:
    n = int(SAMPLE_RATE * DURATION)
    noise = _one_pole_lp(_white(n), 1200.0)
    out = [0.0] * n
    # Overlapping wave envelopes with ~8s period and variation.
    wave_period = int(SAMPLE_RATE * 7.5)
    for start in range(-wave_period, n, wave_period // 2):
        peak = start + random.randint(wave_period // 3, wave_period // 2)
        width = random.randint(wave_period // 2, wave_period)
        amp = random.uniform(0.6, 1.0)
        for i in range(max(0, start), min(n, start + width)):
            t = (i - peak) / (width / 2)
            env = math.exp(-t * t * 1.2)
            out[i] += amp * env * noise[i]
    # Low rumble.
    rumble = _one_pole_lp(_white(n), 120.0)
    return [out[i] + 0.3 * rumble[i] for i in range(n)]


def forest() -> list[float]:
    n = int(SAMPLE_RATE * DURATION)
    bed = [0.15 * v for v in _one_pole_lp(_pink(n), 3000.0)]
    # Scatter bird chirps.
    i = 0
    while i < n:
        i += random.randint(int(SAMPLE_RATE * 0.3), int(SAMPLE_RATE * 1.8))
        if i >= n:
            break
        chirp_len = random.randint(int(SAMPLE_RATE * 0.08), int(SAMPLE_RATE * 0.35))
        f_start = random.uniform(1800, 3500)
        f_end = f_start + random.uniform(-900, 1500)
        amp = random.uniform(0.15, 0.4)
        phase = 0.0
        for k in range(chirp_len):
            if i + k >= n:
                break
            t = k / chirp_len
            f = f_start + (f_end - f_start) * t
            phase += 2 * math.pi * f / SAMPLE_RATE
            env = math.sin(math.pi * t) ** 2
            # Add a second harmonic for warmth.
            bed[i + k] += amp * env * (math.sin(phase) + 0.3 * math.sin(2 * phase))
    return bed


def stream() -> list[float]:
    n = int(SAMPLE_RATE * DURATION)
    high = _one_pole_hp(_one_pole_lp(_white(n), 6000.0), 800.0)
    mid = _one_pole_lp(_white(n), 1500.0)
    # Gentle burbles via slow amplitude modulation of high band.
    bubble_env = [0.0] * n
    phase_a, phase_b = 0.0, 0.0
    for i in range(n):
        phase_a += 2 * math.pi * 3.1 / SAMPLE_RATE
        phase_b += 2 * math.pi * 5.7 / SAMPLE_RATE
        bubble_env[i] = 0.7 + 0.3 * math.sin(phase_a) * math.cos(phase_b)
    return [0.55 * mid[i] + 0.45 * high[i] * bubble_env[i] for i in range(n)]


def main() -> None:
    random.seed(20260416)
    scenes = {
        "rain.wav": rain,
        "wind.wav": wind,
        "ocean.wav": ocean,
        "forest.wav": forest,
        "stream.wav": stream,
    }
    for name, fn in scenes.items():
        path = _write_wav(name, fn())
        size = os.path.getsize(path)
        print(f"wrote {name}: {size:,} bytes")


if __name__ == "__main__":
    main()
