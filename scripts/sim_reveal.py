import os
import sys
import glob
import random
import time
import subprocess
import argparse
from typing import List, Tuple, Optional

import cv2
import numpy as np


def choose_random_video(videos_dir: str) -> str:
    patterns = ["*.mp4", "*.mov", "*.mkv", "*.webm"]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(videos_dir, p)))
    if not files:
        raise SystemExit(f"No video files found in {videos_dir}")
    return random.choice(files)


def ensure_size(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    h, w = img.shape[:2]
    # Resize with aspect fit and crop/pad to exact size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def draw_rings(frame: np.ndarray, center: Tuple[int, int], radii: List[int], base_hue: int = 90) -> None:
    # Fancy color gradient across rings for visual appeal
    n = max(1, len(radii))
    for i, r in enumerate(radii):
        hue = int((base_hue + i * 12) % 180)
        color = hsv_to_bgr(hue, 255, 230)
        cv2.circle(frame, center, int(r), color, thickness=6, lineType=cv2.LINE_AA)


def hsv_to_bgr(h: int, s: int, v: int) -> Tuple[int, int, int]:
    arr = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(arr, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def reflect(vel: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return vel - 2.0 * np.dot(vel, normal) * normal


def synth_combined_audio(bounce_times: List[float], boom_times: List[float], total_duration_sec: float, out_wav: str,
                        sample_rate: int = 44100) -> None:
    """Render short beeps at bounce_times and explosion-like booms at boom_times, mix, and save WAV."""
    sr = int(sample_rate)
    n = int(max(0.1, total_duration_sec + 0.5) * sr)
    audio = np.zeros(n, dtype=np.float32)

    rng_audio = np.random.default_rng(1337)

    # Bounce: bright short beep
    beep_dur = 0.085
    t_beep = np.linspace(0, beep_dur, int(sr * beep_dur), endpoint=False)
    env_beep = np.exp(-t_beep * 22.0) * np.clip(t_beep / 0.006, 0.0, 1.0)

    for t in bounce_times:
        if t < 0:
            continue
        start = int(t * sr)
        if start >= n:
            continue
        f0 = float(rng_audio.uniform(780.0, 1400.0))
        phase = float(rng_audio.uniform(0, 2 * np.pi))
        glide = np.linspace(0.0, float(rng_audio.uniform(-120.0, 90.0)), t_beep.shape[0], dtype=np.float32)
        sig = (np.sin(2 * np.pi * (f0 + glide) * t_beep + phase) * env_beep).astype(np.float32)
        vol = float(0.33 * rng_audio.uniform(0.85, 1.2))
        sig *= vol
        end = min(n, start + sig.shape[0])
        audio[start:end] += sig[: end - start]

    # Explosion: noisy low-mid thump with downward chirp
    boom_dur = 0.22
    t_boom = np.linspace(0, boom_dur, int(sr * boom_dur), endpoint=False)
    # envelope: quick attack then slower decay
    env_boom = np.clip(t_boom / 0.004, 0.0, 1.0) * np.exp(-t_boom * 10.0)

    for t in boom_times:
        if t < 0:
            continue
        start = int(t * sr)
        if start >= n:
            continue
        f_start = float(rng_audio.uniform(650.0, 900.0))
        f_end = float(rng_audio.uniform(120.0, 220.0))
        f = np.linspace(f_start, f_end, t_boom.shape[0], dtype=np.float32)
        phase = 2 * np.pi * np.cumsum(f) / sr
        tone = np.sin(phase)
        noise = rng_audio.normal(0.0, 0.7, size=t_boom.shape[0]).astype(np.float32)
        # bandpass-ish by mixing
        sig = (0.65 * tone + 0.35 * noise) * env_boom
        vol = float(0.60 * rng_audio.uniform(0.8, 1.1))
        sig *= vol
        end = min(n, start + sig.shape[0])
        audio[start:end] += sig[: end - start]

    # prevent clipping
    m = float(np.max(np.abs(audio))) if np.any(audio) else 0.0
    if m > 0.99:
        audio *= 0.99 / (m + 1e-9)

    # write int16 wav
    import wave
    pcm = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    with wave.open(out_wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def mux_audio_with_video(video_in: str, wav_in: str, video_out: str) -> bool:
    """Mux WAV onto video using ffmpeg. Returns True on success."""
    cmd = [
        'ffmpeg', '-y',
        '-i', video_in,
        '-i', wav_in,
        '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '192k',
        '-map', '0:v:0', '-map', '1:a:0',
        '-shortest',
        video_out,
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return res.returncode == 0 and os.path.exists(video_out) and os.path.getsize(video_out) > 0
    except Exception:
        return False


def circle_intersect_time(pos: np.ndarray, vel: np.ndarray, center: Tuple[int, int], radius: float,
                          tmax: float, eps: float = 1e-6) -> Optional[float]:
    # Solve |pos + t*vel - c| = radius for smallest t in (0, tmax]
    # Treat near-tangential contacts robustly by clamping tiny negative discriminants to zero.
    p = pos - np.array(center, dtype=np.float32)
    a = float(np.dot(vel, vel))
    if a < eps:
        return None
    b = 2.0 * float(np.dot(p, vel))
    c = float(np.dot(p, p) - radius * radius)
    disc = b * b - 4 * a * c
    # Numerical guard: allow tiny negative discriminants as tangential hits
    if disc < 0:
        if disc > -1e-7:
            disc = 0.0
        else:
            return None
    sqrt_disc = float(np.sqrt(max(0.0, disc)))
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    # We want the first positive root within (0, tmax]
    candidates = [t for t in (t1, t2) if eps < t <= tmax + eps]
    if not candidates:
        return None
    t_hit = min(candidates)
    if t_hit <= eps or t_hit > tmax + eps:
        return None
    return t_hit


def draw_trail(frame: np.ndarray, points: list, base_color=(40, 200, 255)) -> None:
    # Render a glow trail by drawing fading circles
    n = len(points)
    for i, (x, y) in enumerate(points):
        a = (i + 1) / n
        radius = int(8 + 14 * a)
        alpha = 0.12 * a
        overlay = frame.copy()
        cv2.circle(overlay, (int(x), int(y)), radius, base_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


class Particles:
    def __init__(self):
        self.p: list[np.ndarray] = []
        self.v: list[np.ndarray] = []
        self.life: list[int] = []

    def spawn(self, origin: Tuple[int, int], normal: np.ndarray, rng: np.random.Generator, count: int = 40):
        o = np.array(origin, dtype=np.float32)
        for _ in range(count):
            tangent = np.array([normal[1], -normal[0]], dtype=np.float32)
            dir_vec = normal * (0.8 + 0.6 * rng.random()) + tangent * (rng.random() - 0.5) * 0.8
            speed = 6 + 6 * rng.random()
            self.p.append(o.copy())
            self.v.append(dir_vec * speed)
            self.life.append(20 + int(rng.integers(10)))

    def step(self):
        i = 0
        while i < len(self.p):
            self.p[i] = self.p[i] + self.v[i]
            self.v[i] *= 0.96
            self.life[i] -= 1
            if self.life[i] <= 0:
                self.p.pop(i)
                self.v.pop(i)
                self.life.pop(i)
            else:
                i += 1

    def draw(self, frame: np.ndarray, color=(0, 255, 255)):
        for pt in self.p:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, color, -1, lineType=cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Simulated reveal generator")
    parser.add_argument("videos_dir", nargs="?", default="videos")
    parser.add_argument("out_dir", nargs="?", default="finished")
    parser.add_argument("--mode", choices=["bouncy", "three_hit"], default="bouncy",
                        help="bouncy: simple single-ball; three_hit: balls freeze/respawn and collide with frozen balls. Ring lives apply in both modes.")
    parser.add_argument("--ring-lives", type=int, default=3,
                        help="Number of hits required to destroy a ring in three_hit mode (default: 3)")
    parser.add_argument("--ball-life", type=float, default=2.0,
                        help="How long the active ball lives in seconds before freeze/relaunch (default: 2.0; used in three_hit to freeze and in bouncy to relaunch)")
    args = parser.parse_args()

    videos_dir = args.videos_dir
    out_dir = args.out_dir
    mode = args.mode
    ring_lives = max(1, int(args.ring_lives))
    ball_life_seconds = max(0.5, float(args.ball_life))
    os.makedirs(out_dir, exist_ok=True)

    # Pick a random source video
    src_path = choose_random_video(videos_dir)
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {src_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Target resolution (fit to 1080x1920 by default)
    target_w, target_h = (1080, 1920) if max(src_w, src_h) >= 1080 else (src_w, src_h)

    # Read first frame and create a frozen background
    ok, first = cap.read()
    if not ok:
        raise SystemExit("Could not read first frame")
    frozen_bg = ensure_size(first, (target_w, target_h))

    # Simulation config
    fps = int(round(src_fps)) or 30
    duration_pre = 0  # we simulate until reveal completes, not time-bound
    center = (target_w // 2, target_h // 2)

    # 10 rings => 10 steps of 10%
    inner_r = int(min(target_w, target_h) * 0.18)
    outer_r = int(min(target_w, target_h) * 0.43)
    ring_count = 10
    radii = np.linspace(inner_r, outer_r, ring_count).astype(float).tolist()
    rings_alive: List[float] = radii.copy()
    ring_hits: List[int] = [0 for _ in rings_alive]
    # Cooldown per ring so a single graze doesn't count multiple lives instantly
    ring_last_hit: List[float] = [-1e9 for _ in rings_alive]
    shrink_per_frame = max(0.15, min(target_w, target_h) * 0.0004)

    # Ball parameters (gravity-driven)
    rng = np.random.default_rng()
    ball_radius = 12
    # Start exactly at the center
    pos = np.array([center[0], center[1]], dtype=np.float32)
    # Initial velocity magnitude and random direction
    init_speed = max(8.0, min(target_w, target_h) * 0.01)
    angle0 = float(rng.uniform(0, 2*np.pi))
    vel = np.array([np.cos(angle0), np.sin(angle0)], dtype=np.float32) * init_speed
    g = np.array([0.0, max(0.8, target_h * 0.0012)], dtype=np.float32)  # gravity px/frame^2
    ball_color = (20, 160, 255)  # BGR

    # Chaos tuning: introduce slight randomness in dynamics and collisions
    angle_jitter_deg = 7.0             # random rotation at each collision
    restitution_range = (0.92, 1.03)   # random energy gain/loss on bounce
    tangential_factor = 0.18           # add tangential component relative to speed
    substep_noise_std = 0.02           # small per-substep velocity noise (px/frame)
    air_drag = 0.9995                  # very slight drag to tame extremes
    # Ring hit cooldown (seconds) before same ring can lose another life
    ring_hit_cooldown_sec = 0.20

    # Visual effects
    trail = []  # list of past positions
    trail_max = 28
    particles = Particles()

    # Mask alpha (1.0 = fully opaque black; 0.0 = fully transparent)
    mask_alpha = 1.0
    # Small epsilon to avoid float drift keeping loop alive forever
    REVEAL_EPS = 1e-4

    # Prepare writer
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"sim_{ts}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (target_w, target_h))

    # Track audio events (seconds) to add sound later
    bounce_events: List[float] = []
    boom_events: List[float] = []
    last_destroy_t: float = 0.0
    # 'three_hit' mode state
    frozen_balls: List[np.ndarray] = []
    # Ball life (seconds): in three_hit this is freeze time; in bouncy we relaunch after this
    freeze_seconds = float(ball_life_seconds)
    ball_elapsed = 0.0

    # Phase 1: simulate rings + ball on top of frozen first frame
    frames_written = 0
    while mask_alpha > REVEAL_EPS:
        base = frozen_bg.copy()

        # Apply black mask with current opacity
        if mask_alpha > 0:
            base = (base.astype(np.float32) * (1.0 - mask_alpha)).astype(np.uint8)

        # Slight shrink for remaining rings for tension
        rings_alive = [max(inner_r, r - shrink_per_frame) for r in rings_alive]

        # Draw remaining rings
        draw_rings(base, center, [int(r) for r in rings_alive])

        # Integrate ball with semi-implicit Euler + substeps for collisions
        # Adaptive substeps to reduce tunneling at high speed
        base_substeps = 6
        speed_now = float(np.linalg.norm(vel))
        max_disp = max(2.0, ball_radius * 0.8)
        adapt = int(np.ceil(speed_now / max_disp)) if max_disp > 0 else base_substeps
        substeps = int(max(base_substeps, min(24, adapt)))
        dt = 1.0 / substeps
        for k in range(substeps):
            # Apply gravity
            vel = vel + g * dt
            # Add small random turbulence and air drag to break regularity
            vel = (vel + rng.normal(0.0, substep_noise_std, size=2).astype(np.float32)) * air_drag

            # Plan movement within this substep
            time_left = 1.0
            while time_left > 1e-4:
                earliest_t = None
                hit_normal = None
                hit_index = -1

                # Check all rings against both surfaces rr±ball_radius
                for i, rr in enumerate(rings_alive):
                    for rad in (max(1.0, rr - ball_radius), rr + ball_radius):
                        t = circle_intersect_time(pos.astype(np.float32), (vel * dt).astype(np.float32), center, rad, time_left)
                        if t is not None and (earliest_t is None or t < earliest_t):
                            p_hit = pos + (vel * dt) * t
                            n = (p_hit - np.array(center, dtype=np.float32))
                            n /= (np.linalg.norm(n) + 1e-6)
                            earliest_t = t
                            hit_normal = n
                            hit_index = i

                # Also consider collisions with frozen balls (three_hit mode)
                if 'mode' in locals() and mode == "three_hit" and frozen_balls:
                    vdt_fb = (vel * dt).astype(np.float32)
                    for j, fb in enumerate(frozen_balls):
                        rad_fb = float(ball_radius + ball_radius)
                        t_fb = circle_intersect_time(pos.astype(np.float32), vdt_fb,
                                                      (float(fb[0]), float(fb[1])), rad_fb, time_left)
                        if t_fb is not None and (earliest_t is None or t_fb < earliest_t):
                            p_hit = pos + vdt_fb * t_fb
                            n = (p_hit - fb.astype(np.float32))
                            n /= (np.linalg.norm(n) + 1e-6)
                            earliest_t = t_fb
                            hit_normal = n
                            hit_index = -1

                if earliest_t is None:
                    # No collision in remainder of substep -> advance and do a robust penetration check
                    pos = pos + vel * dt * time_left

                    # Penetration correction against rings (for missed grazing/tunneling)
                    corrected = False
                    if len(rings_alive) > 0:
                        d_vec = np.array([pos[0] - center[0], pos[1] - center[1]], dtype=np.float32)
                        dist = float(np.linalg.norm(d_vec))
                        if dist > 1e-6:
                            n_dir = d_vec / dist
                        else:
                            n_dir = np.array([1.0, 0.0], dtype=np.float32)
                        for i, rr in enumerate(rings_alive):
                            band_in = max(1.0, rr - ball_radius)
                            band_out = rr + ball_radius
                            if band_in < dist < band_out:
                                # push to nearest boundary and reflect
                                d_out = band_out - dist
                                d_in = dist - band_in
                                if d_out <= d_in:
                                    pos = pos + n_dir * (d_out + 0.5)
                                else:
                                    pos = pos - n_dir * (d_in + 0.5)
                                vel = reflect(vel, n_dir) * 0.985
                                # Count a hit with cooldown
                                t_event = (frames_written / fps) + ((k + 1) / substeps) * (1.0 / fps)
                                # Record a bounce sound on correction
                                bounce_events.append(float(t_event))
                                if (t_event - ring_last_hit[i]) >= ring_hit_cooldown_sec:
                                    ring_last_hit[i] = t_event
                                    ring_hits[i] += 1
                                    if ring_hits[i] >= ring_lives:
                                        rings_alive.pop(i)
                                        ring_hits.pop(i)
                                        ring_last_hit.pop(i)
                                        mask_alpha = max(0.0, mask_alpha - 0.10)
                                        # Explosion sound on destruction
                                        boom_events.append(float(t_event))
                                        # Update last destruction time
                                        last_destroy_t = float(t_event)
                                        if not rings_alive or mask_alpha <= REVEAL_EPS:
                                            mask_alpha = 0.0
                                corrected = True
                                break

                    # Penetration correction against frozen balls (three_hit mode)
                    if not corrected and 'mode' in locals() and mode == "three_hit" and frozen_balls:
                        for fb in frozen_balls:
                            delta = np.array([pos[0]-fb[0], pos[1]-fb[1]], dtype=np.float32)
                            dist2 = float(np.linalg.norm(delta))
                            min_sep = float(2 * ball_radius)
                            if dist2 < min_sep - 0.25:
                                n = delta / (dist2 + 1e-6)
                                pos = pos + n * (min_sep - dist2 + 0.5)
                                vel = reflect(vel, n) * 0.985
                                break

                    break

                # Move to impact point
                pos = pos + vel * dt * earliest_t
                # Reflect velocity across normal
                vel = reflect(vel, hit_normal)

                # Chaotic tweaks: random angle jitter, tangential kick, and restitution
                speed = float(np.linalg.norm(vel) + 1e-8)
                # Small random rotation
                ang = np.deg2rad(float(rng.uniform(-angle_jitter_deg, angle_jitter_deg)))
                ca, sa = np.cos(ang), np.sin(ang)
                vx, vy = float(vel[0]), float(vel[1])
                vel = np.array([vx * ca - vy * sa, vx * sa + vy * ca], dtype=np.float32)
                # Tangential kick relative to the surface
                tangent = np.array([hit_normal[1], -hit_normal[0]], dtype=np.float32)
                vel = vel + tangent * (speed * tangential_factor * (float(rng.random()) - 0.5) * 2.0)
                # Random restitution (energy gain/loss)
                rest = float(rng.uniform(*restitution_range))
                vel = vel * rest
                # Nudge out to avoid re-colliding
                pos = pos + hit_normal * 0.5

                # Determine precise event time within the frame for audio/events
                sub_elapsed = (1.0 - time_left)  # before subtracting earliest_t
                t_frame = (frames_written / fps)
                t_event = t_frame + ((k * dt) + (sub_elapsed + earliest_t) * dt) / 1.0 / fps
                # Record a bounce sound for every collision (ring or frozen ball)
                bounce_events.append(float(t_event))

                # Ring touch handling
                if 0 <= hit_index < len(rings_alive):
                    impact = (int(pos[0]), int(pos[1]))
                    particles.spawn(impact, hit_normal, rng, count=40)

                    # t_event already computed above for this collision

                    # Apply per-ring cooldown so multiple rapid sub-collisions don't all count
                    if (t_event - ring_last_hit[hit_index]) >= ring_hit_cooldown_sec:
                        ring_last_hit[hit_index] = t_event
                        ring_hits[hit_index] += 1

                        # On destruction, remove ring and reveal 10%
                        if ring_hits[hit_index] >= ring_lives:
                            rings_alive.pop(hit_index)
                            ring_hits.pop(hit_index)
                            ring_last_hit.pop(hit_index)
                            mask_alpha = max(0.0, mask_alpha - 0.10)
                            # Explosion sound on destruction
                            boom_events.append(float(t_event))
                            # Update last destruction time
                            last_destroy_t = float(t_event)
                            # If no rings remain, consider reveal complete immediately
                            if not rings_alive or mask_alpha <= REVEAL_EPS:
                                mask_alpha = 0.0

                time_left -= earliest_t

        # Draw effects
        trail.append((pos[0], pos[1]))
        if len(trail) > trail_max:
            trail = trail[-trail_max:]
        draw_trail(base, trail)
        particles.step()
        particles.draw(base)

        # Draw frozen balls for three_hit mode
        if 'mode' in locals() and mode == "three_hit" and frozen_balls:
            for fb in frozen_balls:
                cx, cy = int(fb[0]), int(fb[1])
                cv2.circle(base, (cx, cy), ball_radius, (180, 180, 180), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(base, (cx, cy), ball_radius, (240, 240, 240), thickness=2, lineType=cv2.LINE_AA)

        # Draw ball last for crispness
        bx, by = int(pos[0]), int(pos[1])
        cv2.circle(base, (bx, by), ball_radius, ball_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(base, (bx, by), ball_radius, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # Progress bar
        reveal_pct = int(round((1.0 - mask_alpha) * 100))
        bar_w = int(target_w * 0.7)
        bar_h = 16
        x0 = (target_w - bar_w) // 2
        y0 = 40
        cv2.rectangle(base, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1, lineType=cv2.LINE_AA)
        filled = int(bar_w * reveal_pct / 100)
        cv2.rectangle(base, (x0, y0), (x0 + filled, y0 + bar_h), (60, 220, 130), -1, lineType=cv2.LINE_AA)
        cv2.putText(base, f"Reveal {reveal_pct}%", (x0, y0 + bar_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

        writer.write(base)
        frames_written += 1

        # Advance ball life timer
        ball_elapsed += (1.0 / fps)

        # In three_hit mode: freeze the ball after ball life and spawn a new one
        if 'mode' in locals() and mode == "three_hit":
            if ball_elapsed >= freeze_seconds:
                # Freeze current ball
                frozen_balls.append(pos.copy())
                # Spawn a new ball from center with random direction
                pos = np.array([center[0], center[1]], dtype=np.float32)
                angleN = float(rng.uniform(0, 2*np.pi))
                vel = np.array([np.cos(angleN), np.sin(angleN)], dtype=np.float32) * init_speed
                ball_elapsed = 0.0
                trail = []
        else:
            # In bouncy mode: relaunch after ball life
            if ball_elapsed >= freeze_seconds:
                pos = np.array([center[0], center[1]], dtype=np.float32)
                angleN = float(rng.uniform(0, 2*np.pi))
                vel = np.array([np.cos(angleN), np.sin(angleN)], dtype=np.float32) * init_speed
                ball_elapsed = 0.0
                trail = []

        # Spawn a new ball if no ring is destroyed for 10 seconds
        sim_t = frames_written / float(fps)
        if (sim_t - last_destroy_t) >= 10.0:
            if 'mode' in locals() and mode == "three_hit":
                # Freeze current ball to keep stacking and spawn a new active one
                frozen_balls.append(pos.copy())
                pos = np.array([center[0], center[1]], dtype=np.float32)
                angleR = float(rng.uniform(0, 2*np.pi))
                vel = np.array([np.cos(angleR), np.sin(angleR)], dtype=np.float32) * init_speed
                trail = []
                ball_elapsed = 0.0
            else:
                # In bouncy mode, re-center and re-launch the ball
                pos = np.array([center[0], center[1]], dtype=np.float32)
                angleR = float(rng.uniform(0, 2*np.pi))
                vel = np.array([np.cos(angleR), np.sin(angleR)], dtype=np.float32) * init_speed
                trail = []
            # Reset the timer window (so we don't spawn every frame)
            last_destroy_t = float(sim_t)

        # End Phase 1 if the ball leaves the visible frame (simulation domain)
        # Keep early-exit only in bouncy mode to avoid premature finish in three_hit
        if 'mode' in locals() and mode == "bouncy":
            exit_margin = 2 * ball_radius
            if (
                pos[0] < -exit_margin or pos[0] > target_w + exit_margin or
                pos[1] < -exit_margin or pos[1] > target_h + exit_margin
            ):
                mask_alpha = 0.0

        # Safety: avoid overly long pre-phase (fallback guard)
        if frames_written > fps * 45:  # cap phase 1 at ~45s
            mask_alpha = 0.0

    # Phase 2: play the full video (no rings/balls)
    cap.release()
    cap2 = cv2.VideoCapture(src_path)
    if not cap2.isOpened():
        writer.release()
        raise SystemExit("Failed to reopen source video")

    phase2_frames = 0
    while True:
        ok, frame = cap2.read()
        if not ok:
            break
        frame = ensure_size(frame, (target_w, target_h))
        writer.write(frame)
        phase2_frames += 1

    cap2.release()
    writer.release()

    # Render and mux combined audio (bounces + explosions)
    total_duration = (frames_written + phase2_frames) / float(fps)
    wav_path = os.path.join(out_dir, f"sim_{ts}_mix.wav")
    try:
        synth_combined_audio(bounce_events, boom_events, total_duration, wav_path, sample_rate=44100)
    except Exception as e:
        wav_path = None

    final_path = out_path
    if wav_path and os.path.exists(wav_path):
        out_with_audio = os.path.join(out_dir, f"sim_{ts}_audio.mp4")
        if mux_audio_with_video(out_path, wav_path, out_with_audio):
            final_path = out_with_audio

    print(final_path)

    # Cleanup temporary files
    try:
        # remove wav if it exists
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass
        # remove intermediate video if final has audio variant
        if final_path != out_path and os.path.exists(out_path):
            try:
                os.remove(out_path)
            except Exception:
                pass
    except Exception:
        # Swallow any cleanup errors silently
        pass


if __name__ == "__main__":
    main()
