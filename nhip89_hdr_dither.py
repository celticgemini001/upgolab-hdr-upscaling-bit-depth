#!/usr/bin/env python3
"""NHIP 89 — Experiment 5 for upgolab-hdr-upscaling-bit-depth.

The page proves: 8-bit display is a cliff, a 16-bit tone-mapped intermediate is a no-op.
It never tests the mitigation people actually reach for: DITHER.

Two questions, both measurable:
  Q1  Does TPDF dither before 8-bit quantisation recover ripple that plain rounding loses?
  Q2  Must dither be applied at OUTPUT resolution, or does dithering at the low
      (pre-upscale) resolution survive the resampler? A resampler is a low-pass filter,
      so dither added before upscaling should be re-correlated into low-frequency
      blotches rather than stay fine grain. That is testable: measure how much of the
      noise energy survives a 4x4 box average.

Same estimator as Experiment 1 (column-mean profile + lstsq on the known period),
so a difference between rows is a difference in the data, not in the fit.
"""
import json
import numpy as np
from PIL import Image

S, K, PER = 128, 4, 16.0
rng = np.random.default_rng(20261119)   # fixed seed -> rerun reproduces byte for byte


def enc(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def rein(L):
    return L / (1.0 + L)


def tm_gray(L):
    return enc(rein(np.asarray(L, dtype=float)))


def down_np(a, k=K):
    "box average 4x — what a sensor does, in linear light"
    return a.reshape(S // k, k, S // k, k).mean(axis=(1, 3))


def up(a, k=K):
    im = Image.fromarray(a.astype(np.float32), mode="F")
    return np.asarray(im.resize((a.shape[1] * k, a.shape[0] * k), Image.BICUBIC), dtype=np.float64)


def q(x, bits):
    m = 2 ** bits - 1
    return np.clip(np.round(x * m), 0, m) / m


def q_dither(x, bits=8):
    "TPDF dither of +-1 LSB, the standard choice: triangular, 2 LSB peak-to-peak"
    m = 2 ** bits - 1
    n = (rng.random(x.shape) - rng.random(x.shape)) / m
    return np.clip(np.round((x + n) * m), 0, m) / m


def prof_harm(img):
    "fundamental amplitude AND 3rd harmonic of the known period — a staircase has h3 != 0"
    p = img.mean(axis=0)
    x = np.arange(S, dtype=float)
    A = np.stack([np.sin(2 * np.pi * x / PER), np.cos(2 * np.pi * x / PER),
                  np.sin(6 * np.pi * x / PER), np.cos(6 * np.pi * x / PER),
                  np.ones(S)], 1)
    c, *_ = np.linalg.lstsq(A, p, rcond=None)
    h1 = float(np.hypot(c[0], c[1]))
    h3 = float(np.hypot(c[2], c[3]))
    return h1, (h3 / h1 if h1 > 1e-12 else 0.0)


def persistence(noise, k=K):
    "fraction of noise RMS left after a kxk box average. white -> 1/k; low-freq -> near 1"
    bl = down_np(noise, k)
    return float(np.sqrt((bl ** 2).mean()) / max(np.sqrt((noise ** 2).mean()), 1e-12))


def runs(v):
    v = np.round(np.asarray(v).ravel())
    same = np.abs(np.diff(v)) < 1e-9
    best = cur = 1
    for f in same:
        cur = cur + 1 if f else 1
        best = max(best, cur)
    return int(len(np.unique(v))), int(best)


def patch(L0, a):
    yy, xx = np.mgrid[0:S, 0:S]
    return L0 * (1.0 + a * np.sin(2 * np.pi * xx / PER))


# (L0 in linear light, relative ripple amplitude) — the last one is a plain flat control
CASES = [(0.005, 0.20), (0.01, 0.05), (0.02, 0.05), (0.05, 0.05), (0.2, 0.05), (0.02, 0.0)]

ARMS = [
    ("float (no quantisation)",      lambda d, dsp: up(dsp)),
    ("16-bit at low res",            lambda d, dsp: up(q(dsp, 16))),
    ("8-bit at low res",             lambda d, dsp: up(q(dsp, 8))),
    ("8-bit + dither at low res",    lambda d, dsp: up(q_dither(dsp, 8))),
    ("8-bit quantise after upscale", lambda d, dsp: q(up(dsp), 8)),
    ("8-bit + dither after upscale", lambda d, dsp: q_dither(up(dsp), 8)),
]

rows = []
for L0, a in CASES:
    master = patch(L0, a)
    low = down_np(master)              # the only information that exists
    disp = tm_gray(low)                # tone-mapped display code, 0..1
    ref = up(disp)                     # float reference at output resolution
    span = (disp.max() - disp.min()) * 255.0
    base, _ = prof_harm(ref)
    amps = {}
    for name, fn in ARMS:
        out = fn(low, disp)
        amp, h3 = prof_harm(out)
        # per-pixel error against the float arm, in display codes
        rmse = float(np.sqrt(((out - ref) ** 2).mean()) * 255.0)
        nz = out - ref
        uniq, mrun = runs(out[:, S // 2])
        amps[name] = amp
        rows.append({
            "L0": L0, "rel_amp": a, "display_code": round(float(disp.mean() * 255), 1),
            "ripple_span_codes": round(float(span), 2),
            "arm": name,
            "amp": round(amp, 4),
            "h3": round(h3, 4),
            "pct_of_float": round(100 * amp / base, 1) if base else None,
            "rmse_codes": round(rmse, 3),
            "noise_persistence": round(persistence(nz), 3),
            "unique_levels": uniq, "max_run": mrun,
        })

out = {"seed": 20261119, "K": K, "rows": rows}
json.dump(out, open("/tmp/ghclone/nhip89_hdr_dither.json", "w"), indent=1)

# ---- print the two tables we will publish ----
print("== Q1: ripple retention, % of the float arm ==")
hdr = "%-14s %8s %6s |" % ("L0", "code", "span")
for n, _ in ARMS:
    hdr += " %-9s" % n.split(" ")[0].replace("(", "")[:9]
print(hdr)
for L0, a in CASES:
    if a == 0.0:
        continue
    r = [x for x in rows if x["L0"] == L0 and x["rel_amp"] == a]
    line = "%-14s %8.1f %6.2f |" % (L0, r[0]["display_code"], r[0]["ripple_span_codes"])
    for n, _ in ARMS:
        v = [x for x in r if x["arm"] == n][0]["pct_of_float"]
        line += " %-9s" % v
    print(line)

print("\n== Q1b: per-pixel RMSE vs float, display codes (the cost of dither) ==")
for L0, a in CASES:
    if a == 0.0:
        continue
    r = [x for x in rows if x["L0"] == L0 and x["rel_amp"] == a]
    line = "%-14s" % L0
    for n, _ in ARMS:
        line += " %-9s" % [x for x in r if x["arm"] == n][0]["rmse_codes"]
    print(line)

print("\n== Q2: noise persistence through a 4x4 average (white=0.250) ==")
for L0, a in CASES:
    r = [x for x in rows if x["L0"] == L0 and x["rel_amp"] == a]
    line = "%-14s a=%.2f" % (L0, a)
    for n in ("8-bit + dither at low res", "8-bit + dither after upscale"):
        line += "  %s=%s" % (n.replace("8-bit + dither ", ""), [x for x in r if x["arm"] == n][0]["noise_persistence"])
    print(line)

print("\n== Q3: 3rd-harmonic ratio — a staircase is not a sine ==")
for L0, a in CASES:
    if a == 0.0:
        continue
    r = [x for x in rows if x["L0"] == L0 and x["rel_amp"] == a]
    line = "%-14s" % L0
    for n, _ in ARMS:
        line += " %-9.3f" % [x for x in r if x["arm"] == n][0]["h3"]   # 3dp, as published
    print(line)

# ---- Q4: the noise spectrum. White noise decays as 1/k under a kxk box average.
# If dither was applied BEFORE the resampler, the surviving noise is low-frequency
# and decays far slower than 1/k. This is what "blotchy" means, stated numerically.
print("\n== Q4: noise persistence vs average size  (white = 1/k) ==")
spec = []
L0, a = 0.02, 0.05
master = patch(L0, a)
low = down_np(master)
disp = tm_gray(low)
ref = up(disp)
for name, fn in ARMS:
    out = fn(low, disp)
    nz = out - ref
    row = {"arm": name, "L0": L0, "rel_amp": a}
    for k in (1, 2, 4, 8, 16):
        row["k%d" % k] = round(persistence(nz, k), 3)
    spec.append(row)
print("%-30s %6s %6s %6s %6s %6s" % ("arm", "k=1", "k=2", "k=4", "k=8", "k=16"))
print("%-30s %6s %6s %6s %6s %6s" % ("white noise, theory", "1.000", "0.500", "0.250", "0.125", "0.063"))
for row in spec:
    print("%-30s %6.3f %6.3f %6.3f %6.3f %6.3f" % (row["arm"], row["k1"], row["k2"], row["k4"], row["k8"], row["k16"]))
json.dump({"rows": rows, "spectrum": spec, "seed": 20261119},
          open("/tmp/ghclone/nhip89_hdr_dither.json", "w"), indent=1)

# ---- Q5: the case dither was invented for — a smooth gradient, not a sine.
# An 86-code ramp across 1024 px has 86 quantisation steps; each step is a visible
# band edge wherever the ramp is slow. Count the longest flat run along the ramp.
print("\n== Q5: gradient banding, 1024 px ramp over ~86 codes ==")
W = 1024
ramp_lin = np.linspace(0.02, 0.05, W)          # linear light, a sky-like range
ramp_dsp = tm_gray(ramp_lin)                    # display code, 0..1
q5 = []
for label, bits, dith in [("float", 0, False), ("8-bit round", 8, False),
                          ("8-bit + TPDF", 8, True), ("6-bit round", 6, False),
                          ("6-bit + TPDF", 6, True)]:
    v = ramp_dsp if bits == 0 else (q_dither(ramp_dsp, bits) if dith else q(ramp_dsp, bits))
    codes = np.round(np.asarray(v) * 255)
    d = np.abs(np.diff(codes))
    same = d < 0.5
    best = cur = 1
    for f in same:
        cur = cur + 1 if f else 1
        best = max(best, cur)
    nuniq = int(len(np.unique(codes)))
    rmse = float(np.sqrt(((codes - np.round(ramp_dsp * 255)) ** 2).mean()))
    q5.append({"arm": label, "bits": bits, "dither": dith,
               "distinct_codes": nuniq, "max_flat_run_px": int(best),
               "flat_pct": round(float(same.mean() * 100), 1),
               "rmse_codes": round(rmse, 3)})
    print("%-14s distinct=%3d  max_flat_run=%4d px  flat=%5.1f%%  rmse=%.3f" %
          (label, nuniq, best, same.mean() * 100, rmse))
json.dump({"rows": rows, "spectrum": spec, "gradient": q5, "seed": 20261119},
          open("/tmp/ghclone/nhip89_hdr_dither.json", "w"), indent=1)
