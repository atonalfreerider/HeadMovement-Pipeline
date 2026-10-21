r"""john_lula v2 render (2026-10-20): the realism videos re-rendered with the v2 DRIVER (realism/v2/footlock/final:
the facing-corrected refit + foot lock, John's 26.2 s patch kept) and SWINGING HAIR on Lula (dancecap.mono4d.splat_hair
on LHM++'s own avatar), plus before/after comparisons. Owner, 2026-10-20: "there is too much foot slip and also the
follow (lula) spins and shows her back, but the post track is not capturing that" / "monocular needs a lot of
improvement to be useful. do what you can" / "LHM++ quality with swinging hair".

    python -m dancecap.mono4d.jl_v2_render prep      # v2/renders/input (refs, poses of the driver, views) + check sheets
    python -m dancecap.mono4d.jl_v2_render hair      # v2/renders/hair: Lula's strands simulated over the whole take
    python -m dancecap.mono4d.jl_v2_render job       # remote/cand_lhmpp/pod_job_jl_v2_hair.json
    (DANCECAP_RUNPOD_CAP=... python -m dancecap remote run --jobs <that json> --max-cost 1 --gpus ...)
    python -m dancecap.mono4d.jl_v2_render import [--session SID]
    python -m dancecap.mono4d.jl_v2_render videos [--only front,sbs,orbit,best,ba_front,ba_phone]
    python -m dancecap.mono4d.jl_v2_render check     # decode every file, 20 frames each, repeat-frame + audio checks

WHAT IS THE SAME AS realism/videos (jl_realism_render): the avatars (the same refs byte for byte, shape ref = betas[:10]
of the fit, which the v2 driver keeps identical), LHM++'s own animation on the pod, the rasterisation, the backdrop, the
cameras: front / orbit / phone / phone_hi are the realism run's views.npz EXCEPT a smooth zoom-out where a raised hand
left the frame (front 42.5-42.7 s, orbit 14.9 s: the old framing rule ignored the hands; `hand_zoom` below), so the
before/after compares the two fits through the same camera.

WHAT IS NEW: the poses (v2 driver), Lula's free hair on simulated strands (all four groups; John's short hair stays on
LHM++'s skinning), and the clock of the phone-path videos. The phone video is 29.994 fps (14997/500); the old
side-by-side picked the nearest real frame for each 30 Hz output frame, which shows one real frame twice (83.3 s). Here
every phone-path video shows each real frame exactly ONCE, output frame n = video frame n (30/1 fps, 0.02% faster), the
splats of that frame are rendered at the frame's own pts, and the full-length audio is resampled by the same factor
15000/14997 (asetrate 44109 + aresample 44100: 0.36 ms residual drift over 88 s, pitch change 0.0035 semitone) so sound
and picture stay locked. The front / orbit videos are rendered on the fit's 30 Hz grid (k/30 s), so they need no
resampling; their audio is the source's own (copied / cut at the window start).

OUTPUTS (likeness data: work/ only, nothing committed): realism/v2/renders/{input,hair,frames,pod_out_swing}/,
realism/v2/videos/ (jl_tutorial_front.mp4 1920x1080 88 s, jl_side_by_side.mp4 1920x1080 88 s, jl_orbit.mp4 1080x1080
4.5-29.5 s, jl_best_30s.mp4, jl_before_after_front.mp4 + jl_before_after_phone.mp4 20 s, videos.json, check.json),
realism/v2/sheets/render/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE
from . import jl_realism_render as JR

ROLES = JR.ROLES
FPS = 30.0
VIDEO_FPS = 14997 / 500            # the phone video's own rate (ffprobe r_frame_rate of john_lula_jnj.mp4)
NV = 2643                          # video frames = grid frames
REAL = JR.REAL
V2 = REAL / "v2"
DRIVER = V2 / "footlock" / "final"
OLD_DRIVER = JR.FIT_DIR
OLD_INPUT = REAL / "lhmpp" / "input"
OLD_RENDERS = REAL / "renders"
R2 = V2 / "renders"
INPUT = R2 / "input"
HAIR = R2 / "hair"
FRAMES = R2 / "frames"
POD_OUT = R2 / "pod_out_swing"     # = jl_realism_render.job's <renders>.parent / "pod_out_swing"
VIDEOS = V2 / "videos"
SHEETS = V2 / "sheets" / "render"
HAIR_BUNDLE = V2 / "hair" / "bundle"   # the hair proof's final bundle (no uncover fill)
JOB_NAME = "jl_v2_render"
JOB_PATH = PACKAGE_ROOT / "remote" / "cand_lhmpp" / "pod_job_jl_v2.json"
HAIR_GROUPS = ["front", "phone_hi", "phone", "orbit"]
HAND_ZOOM = {"margin_frac_H": 0.01, "hold_s": 0.6, "sigma_s": 0.2}
WINDOWS = [  # the before/after: (title, t0 s, seconds); picked from the data (prep's "windows" in prep.json)
    ("Lula's spins (18.9 s spin; 27.2-28.3 s she faces the phone: old fit turned her ~120 deg away)", 18.5, 10.0),
    ("stance-heavy stretch: feet should stay planted (largest slide reduction of the take)", 57.5, 10.0)]
BEST = [  # 30 s best-of: (kind, t0 s, seconds, caption)
    ("front", 11.0, 10.0, "tutorial front: three spins, the ponytail swings"),
    ("orbit", 9.5, 10.0, "360 deg orbit around the couple"),
    ("sbs", 75.0, 10.0, "real phone video | splats along the same camera path")]
CAPTION = "John & Lula - Jack & Jill (3D splats from one phone)"


def log(msg: str) -> None:
    print(f"[jl_v2] {msg}", flush=True)


def _ws(p: Path) -> str:
    return JR._ws(p)


def _sha1(p: Path) -> str:
    return JR._sha1(p)


# ================================================================================================ pure helpers
def zoom_envelope(need: np.ndarray, hold: int, sigma: float) -> np.ndarray:
    """need (N,): the focal-length scale each frame needs (<= 1 shrinks; >= 1 = fine as is). -> smooth s (N,) with
    s <= need wherever need < 1 (up to the 0.3% a 3-sigma Gaussian loses), s = 1 far from every such frame: the
    deficit 1 - need is max-filtered over +-hold frames, Gaussian-smoothed (sigma frames), and never below the raw
    deficit."""
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    d = np.clip(1.0 - np.asarray(need, float), 0.0, None)
    if not (d > 0).any():
        return np.ones_like(d)
    m = maximum_filter1d(d, size=2 * int(hold) + 1, mode="nearest")
    s = gaussian_filter1d(m, float(sigma), mode="nearest") if sigma > 0 else m
    return 1.0 - np.maximum(s, d)


def focal_need(K: np.ndarray, R: np.ndarray, tvec: np.ndarray, P: np.ndarray, W: int, H: int, margin_px: float) -> np.ndarray:
    """Per frame: the largest focal scale (about the principal point) that keeps every point of P (N, M, 3) at least
    margin_px inside the W x H image (>1 = room to spare). Points behind the camera are ignored."""
    Xc = np.einsum("nij,nmj->nmi", R, P) + tvec[:, None]
    z = Xc[..., 2]
    ok = z > 0.1
    u = K[:, None, 0, 0] * Xc[..., 0] / np.where(ok, z, 1.0)
    v = K[:, None, 1, 1] * Xc[..., 1] / np.where(ok, z, 1.0)
    cx, cy = K[:, None, 0, 2], K[:, None, 1, 2]
    lim_u = np.where(u >= 0, (W - 1 - cx) - margin_px, cx - margin_px)
    lim_v = np.where(v >= 0, (H - 1 - cy) - margin_px, cy - margin_px)
    su = np.where(ok & (np.abs(u) > 1e-6), lim_u / np.maximum(np.abs(u), 1e-6), np.inf)
    sv = np.where(ok & (np.abs(v) > 1e-6), lim_v / np.maximum(np.abs(v), 1e-6), np.inf)
    return np.minimum(su, sv).min(1)


def once_index(n_out: int, n_src: int) -> np.ndarray:
    """Output frame n -> source frame n (each source frame exactly once; the clip runs at the output rate)."""
    assert n_out <= n_src, (n_out, n_src)
    return np.arange(n_out)


def nearest_index(t_out: np.ndarray, t_src: np.ndarray) -> np.ndarray:
    """The old side-by-side's mapping (nearest source pts to each output time): repeats / drops frames when the rates
    differ. Kept for the check."""
    t_src = np.asarray(t_src, float)
    i = np.clip(np.searchsorted(t_src, t_out), 1, len(t_src) - 1)
    return np.where(np.abs(t_src[i - 1] - t_out) <= np.abs(t_src[i] - t_out), i - 1, i)


def alias_rows(t_alias: np.ndarray, t_track: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alias keys (e.g. video frames at their pts) as linear blends of consecutive track rows: (row, weight of the next
    row), clamped to the track's ends (a pts after the last track key holds the last row)."""
    tt = np.asarray(t_track, float)
    ta = np.asarray(t_alias, float)
    j = np.clip(np.searchsorted(tt, ta, side="right") - 1, 0, len(tt) - 1)
    jn = np.minimum(j + 1, len(tt) - 1)
    den = np.where(jn > j, tt[jn] - tt[j], 1.0)
    w = np.where(jn > j, np.clip((ta - tt[j]) / den, 0.0, 1.0), 0.0)
    return j.astype(np.int64), w


def crop_centres(x: np.ndarray, W: int, cw: int, sigma: float) -> np.ndarray:
    """Smoothed horizontal crop centres (px) of a cw-wide crop of a W-wide frame, clamped inside the frame."""
    from scipy.ndimage import gaussian_filter1d
    c = gaussian_filter1d(np.asarray(x, float), sigma, mode="nearest") if sigma > 0 else np.asarray(x, float)
    return np.clip(np.round(c), cw // 2, W - cw + cw // 2).astype(int)


def stretch_rate(n_frames: int, src_fps: float = VIDEO_FPS, out_fps: float = FPS, sr: int = 44100) -> dict:
    """asetrate for an audio track that must follow a clip of src_fps frames shown at out_fps: the integer sample rate
    nearest sr * out/src and the drift it leaves over the clip (ms)."""
    f = out_fps / src_fps
    r = int(round(sr * f))
    dur = n_frames / out_fps
    return {"factor": f, "asetrate": r, "drift_ms": abs(r / sr - f) * dur * 1e3 / f}


# ================================================================================================ prep
def _hand_points(J: dict) -> np.ndarray:
    """(F, M, 3): both wrists + all finger joints (SMPL-X 20, 21, 25-54) of both dancers."""
    return np.concatenate([np.asarray(J[r], float)[:, [20, 21] + list(range(25, 55))] for r in ROLES], 1)


def hand_zoom(V: dict, J: dict, group: str, W: int, H: int) -> dict:
    """The group's K with the focal scaled down (smoothly) wherever a hand would leave the image (+ margin)."""
    ks = np.asarray(V[group + "_key"], int) - NV
    K = np.asarray(V[group + "_K"], float).copy()
    P = np.concatenate([JR.couple_points(J), _hand_points(J)], 1)[ks]
    need = focal_need(K, np.asarray(V[group + "_R"], float), np.asarray(V[group + "_tvec"], float), P, W, H,
                      HAND_ZOOM["margin_frac_H"] * H)
    s = zoom_envelope(need, int(round(HAND_ZOOM["hold_s"] * FPS)), HAND_ZOOM["sigma_s"] * FPS)
    K[:, 0, 0] *= s
    K[:, 1, 1] *= s
    ch = np.nonzero(s < 1 - 1e-6)[0]
    info = {"frames_changed": int(len(ch)), "min_scale": round(float(s.min()), 4),
            "need_lt_1_frames": [int(i) for i in np.nonzero(need < 1)[0]],
            "t_changed": [round(float(ks[ch[0]]) / FPS, 2), round(float(ks[ch[-1]]) / FPS, 2)] if len(ch) else None}
    return {"K": K.astype(np.float32), "scale": s, "info": info}


def prep() -> dict:
    from .. import mamma_dyn as md
    from . import cand_lhmpp as CL
    t0 = time.time()
    tr = md.load_tracks(JR.WORK, [JR.CAM])[JR.CAM]
    fits = {r: JR.load_fit(r, DRIVER) for r in ROLES}
    olds = {r: JR.load_fit(r, OLD_DRIVER) for r in ROLES}
    T = np.asarray(fits["lead"]["t"], float)
    F = len(T)
    assert F == NV and len(tr.t) == NV and np.allclose(fits["follow"]["t"], T)
    old_prep = json.loads((OLD_INPUT / "prep.json").read_text())
    if INPUT.exists():
        shutil.rmtree(INPUT)
    (INPUT / "refs").mkdir(parents=True)
    # ---- refs: the realism run's (= the preview's photofull crops), byte for byte
    refs = {}
    for r in ROLES:
        shutil.copytree(OLD_INPUT / "refs" / r, INPUT / "refs" / r)
        want = {f["file"]: f["sha1"] for f in old_prep["refs"][r]["files"]}
        got = {p.name: _sha1(p) for p in sorted((INPUT / "refs" / r).glob("*.png"))}
        assert got == want, f"refs {r} differ from the realism run"
        refs[r] = {"files": len(got), "identical_to_realism_run": True}
    # ---- poses: video frames at pts, then the 30 Hz grid (jl_realism_render.prep's layout)
    ts = np.concatenate([np.asarray(tr.t, float), T])
    poses = {"t": ts.astype(np.float64)}
    chk = {}
    for r in ROLES:
        p = JR.interp_many(fits[r], ts)
        for name, n, short in JR.ROT_KEYS:
            poses[f"{r}_{short}"] = p[name].astype(np.float32)
        poses[f"{r}_pelvis"] = p["joints"][:, 0].astype(np.float32)
        poses[f"{r}_joints22"] = p["joints"][:, :22].astype(np.float32)
        poses[f"{r}_transl"] = p["transl"].astype(np.float32)
        poses[f"{r}_betas"] = np.asarray(fits[r]["betas"], np.float32).reshape(-1)
        errs = []
        for q in np.linspace(0, len(ts) - 1, 25).round().astype(int):
            ref = CL.interp_params(fits[r], float(ts[q]))
            errs.append(max(float(np.abs(ref[name] - p[name][q]).max()) for name, _, _ in JR.ROT_KEYS))
        grid_err = float(np.abs(p["body_pose"][NV:] - np.asarray(fits[r]["body_pose"], float).reshape(F, -1)).max())
        chk[r] = {"vs_interp_params_max_rad": float(max(errs)), "grid_vs_fit_body_pose_max_rad": grid_err,
                  "betas_equal_realism_run": bool(np.array_equal(np.asarray(fits[r]["betas"]).ravel(),
                                                                 np.asarray(olds[r]["betas"]).ravel()))}
        assert chk[r]["betas_equal_realism_run"], "betas changed: the hair bundle / avatars would not match"
    np.savez_compressed(INPUT / "poses.npz", **poses)
    # ---- views: the realism run's cameras; front / orbit zoom out smoothly where a hand would leave the frame
    with np.load(OLD_INPUT / "views.npz") as z:
        V = {k: np.asarray(z[k]) for k in z.files}
    vj = json.loads((OLD_INPUT / "views.json").read_text())
    J = {r: np.asarray(fits[r]["joints"], float) for r in ROLES}
    Jo = {r: np.asarray(olds[r]["joints"], float) for r in ROLES}
    zoom = {}
    for g in ("front", "orbit"):
        gi = vj["groups"][g]
        hz = hand_zoom(V, J, g, gi["W"], gi["H"])
        V[g + "_K"] = hz["K"]
        zoom[g] = hz["info"]
        np.save(INPUT / f"{g}_zoom_scale.npy", hz["scale"].astype(np.float32))
    np.savez_compressed(INPUT / "views.npz", **V)
    (INPUT / "views.json").write_text(json.dumps(vj, indent=1))
    margins = {}
    for g in ("front", "orbit"):
        gi = vj["groups"][g]
        ks = np.asarray(V[g + "_key"], int) - NV
        for lab, JJ in (("new", J), ("old_fit", Jo)):
            P = np.concatenate([JR.couple_points(JJ), _hand_points(JJ)], 1)[ks]
            m = JR.frame_margins(V[g + "_K"], V[g + "_R"], V[g + "_tvec"], P, gi["W"], gi["H"])
            margins[f"{g}_{lab}_with_hands_min_frac_H"] = round(float(m.min()), 4)
            margins[f"{g}_{lab}_frames_outside"] = int((m < 0).sum())
    # ---- before/after windows: where the facing and the stance slide changed most (LOOK at the sheets)
    win = choose_windows(fits, olds)
    # ---- check sheets (LOOK before spending)
    SHEETS.mkdir(parents=True, exist_ok=True)
    sheets = prep_sheets(V, fits, olds, tr, zoom)
    info = {"take": JR.TAKE, "fit_dir": _ws(DRIVER), "fit_sha1": {r: _sha1(DRIVER / f"{r}.npz") for r in ROLES},
            "old_fit_dir": old_prep["fit_dir"], "old_fit_sha1": old_prep["fit_sha1"],
            "views_from": _ws(OLD_INPUT / "views.npz"), "views_from_sha1": _sha1(OLD_INPUT / "views.npz"),
            "hand_zoom": {**HAND_ZOOM, **zoom}, "margins_with_hands": margins, "transfer_check": chk, "refs": refs,
            "pose_keys": int(len(ts)), "video_frames": NV, "grid_frames": F, "front": old_prep["front"],
            "orbit": old_prep["orbit"], "phone_hi": old_prep["phone_hi"], "groups": vj["groups"], "passes": vj["passes"],
            "scene_key": vj["scene_key"], "windows": win, "sheets": sheets,
            "camera_not_ok_frames": old_prep["camera_not_ok_frames"],
            "bytes": int(sum(p.stat().st_size for p in INPUT.rglob("*") if p.is_file())), "seconds": round(time.time() - t0, 1)}
    (INPUT / "prep.json").write_text(json.dumps(info, indent=1, default=float))
    log(f"prep: zoom {json.dumps(zoom)}; margins {json.dumps(margins)}; transfer {json.dumps(chk)}")
    log(f"windows: {json.dumps(win)}")
    return info


def choose_windows(fits: dict, olds: dict, L: int = 300) -> dict:
    yaw = lambda d: np.degrees(np.arctan2(*JR.facing(d)[:, [0, 2]].T))  # noqa: E731
    dy = np.abs((yaw(fits["follow"]) - yaw(olds["follow"]) + 180) % 360 - 180)
    sl = {}
    for nm, ff in (("old", olds), ("new", fits)):
        tot = 0
        for r in ROLES:
            st = np.asarray(fits[r]["footlock_stance"], bool)[1:]
            v = np.linalg.norm(np.diff(np.asarray(ff[r]["joints"], float)[:, [10, 11]][:, :, [0, 2]], axis=0), axis=-1)
            tot = tot + np.where(st, v, 0.0).sum(1)
        sl[nm] = tot
    out = {"follow_yaw_change_gt30deg_t": [round(float(i) / FPS, 2) for i in np.nonzero(dy > 30)[0][[0, -1]]]
           if (dy > 30).any() else None}
    for (title, t0, sec) in WINDOWS:
        k0, k1 = int(round(t0 * FPS)), int(round((t0 + sec) * FPS))
        out[f"{t0:.1f}"] = {"title": title, "follow_yaw_change_deg_sum": round(float(dy[k0:k1].sum())),
                            "follow_yaw_change_max_deg": round(float(dy[k0:k1].max()), 1),
                            "stance_slide_m_old": round(float(sl["old"][k0:k1].sum()), 3),
                            "stance_slide_m_new": round(float(sl["new"][k0:k1].sum()), 3)}
    gains = [(float(sl["old"][k:k + L].sum() - sl["new"][k:k + L].sum()), k) for k in range(0, NV - L, 15)]
    out["best_stance_window_t0"] = round(max(gains)[1] / FPS, 2)
    return out


def prep_sheets(V: dict, fits: dict, olds: dict, tr, zoom: dict) -> dict:
    from . import jl_lhmpp_preview as JP
    T = np.asarray(fits["lead"]["t"], float)
    Jn = {r: np.asarray(fits[r]["joints"], float) for r in ROLES}
    Jo = {r: np.asarray(olds[r]["joints"], float) for r in ROLES}
    out = {}
    # front: zoomed frames + windows; old fit in grey, new in colour, both through the v2 camera
    tiles = []
    ks = [1277, 1279, 1281, 560, 566, 820, 833, 1800]
    for k in ks:
        img = JR.background(V["front_K"][k], V["front_R"][k], V["front_tvec"][k], 1920, 1080,
                            JR.shadow_points({r: Jn[r][k] for r in ROLES}))
        JR._draw_points(img, V["front_K"][k], V["front_R"][k], V["front_tvec"][k], {r: Jo[r][k] for r in ROLES},
                        cols=((150, 150, 150), (110, 110, 110)))
        JR._draw_points(img, V["front_K"][k], V["front_R"][k], V["front_tvec"][k], {r: Jn[r][k] for r in ROLES})
        for r, col in zip(ROLES, ((60, 200, 60), (60, 60, 230))):
            from . import cand_lhmpp as CL
            uv, z = CL.project(Jn[r][k][25:55], V["front_K"][k], V["front_R"][k], V["front_tvec"][k])
            for p_ in uv:
                cv2.circle(img, (int(p_[0]), int(p_[1])), 3, col, -1)
        JR._label(img, f"front k{k} t={T[k]:.2f}s (grey = old fit)")
        tiles.append(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
    sheet = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in (0, 4)], 0)
    p = SHEETS / "prep_front_cameras.jpg"
    cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    out["front"] = _ws(p)
    # phone_hi: real frame + both skeletons at the facing fix (27.2-28.3 s) and in the stance window
    tiles = []
    pad = int(round((1080 - 2 * 538) / 2))
    for i in (815, 825, 835, 842, 848, 858, 1740, 1900):
        real = cv2.imread(str(JP.frame_path(i)))
        hi = np.full((1920, 1080, 3), 40, np.uint8)
        hi[:, pad:pad + 2 * 538] = cv2.resize(real, (2 * 538, 1920), interpolation=cv2.INTER_LANCZOS4)
        jo = {r: JR.interp_many(olds[r], [tr.t[i]])["joints"][0] for r in ROLES}
        jn = {r: JR.interp_many(fits[r], [tr.t[i]])["joints"][0] for r in ROLES}
        JR._draw_points(hi, V["phone_hi_K"][i], V["phone_hi_R"][i], V["phone_hi_tvec"][i], jo,
                        cols=((170, 170, 170), (120, 120, 120)))
        JR._draw_points(hi, V["phone_hi_K"][i], V["phone_hi_R"][i], V["phone_hi_tvec"][i], jn)
        JR._label(hi, f"f{i} t={tr.t[i]:.2f}s", s=1.2)
        tiles.append(cv2.resize(hi, (360, 640), interpolation=cv2.INTER_AREA))
    sheet = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in (0, 4)], 0)
    p = SHEETS / "prep_phone_hi_old_grey_new_colour.jpg"
    cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    out["phone_hi"] = _ws(p)
    return out


# ================================================================================================ hair
def hair() -> dict:
    """Lula's strands simulated over the WHOLE take on the v2 driver's 30 Hz grid keys (a 45-frame static preroll at
    the first pose lets the tail settle), partner capsules from John; the video-pts keys of the phone groups are
    aliases (blend of the two grid rows around their pts, clamped at the ends). -> HAIR/{follow_hair.npz,
    follow_tracks.npz, hair.json, follow_tracks_check.json} + a strands sheet (LOOK)."""
    from . import splat_hair as SH
    HAIR.mkdir(parents=True, exist_ok=True)
    for f in ("follow_hair.npz", "follow_hair.json"):
        shutil.copy2(HAIR_BUNDLE / f, HAIR / f)
    assert _sha1(HAIR / "follow_hair.npz") == _sha1(HAIR_BUNDLE / "follow_hair.npz")
    with np.load(INPUT / "poses.npz") as z:
        Z = {k: np.asarray(z[k]) for k in z.files}
    T = np.asarray(Z["t"], float)
    NK = len(T)
    grid = NV + np.arange(NV)
    assert np.allclose(T[grid], np.arange(NV) / FPS)
    pre = int(SH.SIM["ponytail"]["preroll_frames"])
    Zc = {}
    for k, v in Z.items():          # preroll keys 0..pre-1 = copies of grid key 0, then the grid
        if v.ndim >= 1 and v.shape[0] == NK:
            Zc[k] = np.concatenate([np.repeat(v[grid[:1]], pre, 0), v[grid]], 0)
        else:
            Zc[k] = v
    b = SH.load_bundle(HAIR / "follow_hair.npz")
    sim = SH.simulate_role(b, Zc, "follow", pre + np.arange(NV), partner="lead", style="ponytail", preroll=pre)
    vid = np.arange(NV)
    rows, ws = alias_rows(T[vid], T[grid])
    np.savez_compressed(HAIR / "follow_tracks.npz", version=SH.BUNDLE_VERSION, role="follow", keys=grid.astype(np.int32),
                        tracks_local=sim["tracks_local"].astype(np.float32), head_R=sim["head_R"].astype(np.float32),
                        head_P=sim["head_P"].astype(np.float32), alias_keys=vid.astype(np.int32),
                        alias_row=rows.astype(np.int32), alias_w=ws.astype(np.float32))
    (HAIR / "hair.json").write_text(json.dumps({"version": SH.BUNDLE_VERSION, "roles": {
        "follow": {"bundle": "follow_hair.npz", "tracks": "follow_tracks.npz"}}, "groups": HAIR_GROUPS}, indent=1))
    chk = dict(sim["check"])
    pf = sim["per_frame"]
    worst = {}
    flat = {k_: v_ for k_, v_ in pf.items() if not isinstance(v_, dict)}
    for grp, sel in (("self", lambda n: not n.startswith("partner_")), ("partner", lambda n: n.startswith("partner_"))):
        dd = [np.asarray(v_, float) for n, v_ in pf.get("depth", {}).items() if sel(n)]
        if dd:
            flat[f"depth_{grp}"] = np.max(np.stack(dd), 0)
    np.savez_compressed(HAIR / "follow_tracks_per_frame.npz", **{k_: np.asarray(v_, float) for k_, v_ in flat.items()})
    for k_, v_ in flat.items():
        a = np.asarray(v_, float)
        if a.ndim == 1 and len(a) == NV:
            o = np.argsort(-a)[:8]
            worst[k_] = [[round(float(i) / FPS, 2), round(float(a[i]), 4)] for i in o]
    hv = np.linalg.norm(np.diff(sim["head_P"], axis=0), axis=1) * FPS
    chk.update({"track_keys": [int(grid[0]), int(grid[-1])], "preroll_static": pre, "aliases": int(NV),
                "alias_w_range": [round(float(ws.min()), 4), round(float(ws.max()), 4)], "groups": HAIR_GROUPS,
                "worst_frames_t_value": worst,
                "head_speed_m_s": {"p99": round(float(np.percentile(hv, 99)), 2), "max": round(float(hv.max()), 2),
                                   "frames_over_4m_s_t": [round(float(i) / FPS, 2) for i in np.nonzero(hv > 4)[0]][:30]},
                "stats": {k_: v_ for k_, v_ in sim["stats"].items() if k_ != "params"}, "params": sim["stats"]["params"]})
    (HAIR / "follow_tracks_check.json").write_text(json.dumps(chk, indent=1, default=float))
    np.save(HAIR / "follow_tracks_world_local_check.npy", sim["tracks_world"][::30].astype(np.float32))
    sheet = hair_sheet(sim, Z)
    log(f"hair: {json.dumps({k_: chk[k_] for k_ in ('max_stretch', 'deepest_capsule', 'max_speed_m_s', 'swing_tip_in_spins_m', 'swing_tip_calm_m', 'head_speed_m_s', 'sim_seconds')}, default=float)}")
    log(f"hair sheet: {sheet}")
    return chk


def hair_sheet(sim: dict, Z: dict) -> str:
    """Strands (red) over both skeletons through the front camera at the spins and the worst-check frames."""
    from . import splat_hair as SH
    with np.load(INPUT / "views.npz") as z:
        V = {k: np.asarray(z[k]) for k in z.files}
    fits = {r: JR.load_fit(r, DRIVER) for r in ROLES}
    Jn = {r: np.asarray(fits[r]["joints"], float) for r in ROLES}
    pf = sim["per_frame"]
    picks = [int(round(t * FPS)) for t in (11.97, 12.3, 15.97, 18.87, 19.2, 27.8, 44.47, 79.83)]
    tip = np.asarray(pf.get("tip", np.zeros(NV)), float)
    picks += [int(i) for i in np.argsort(-tip)[:4]]
    tiles = []
    for k in picks:
        img = JR.background(V["front_K"][k], V["front_R"][k], V["front_tvec"][k], 1920, 1080,
                            JR.shadow_points({r: Jn[r][k] for r in ROLES}))
        JR._draw_points(img, V["front_K"][k], V["front_R"][k], V["front_tvec"][k], {r: Jn[r][k] for r in ROLES})
        SH.draw_strands(img, sim["tracks_world"][k], V["front_R"][k], V["front_tvec"][k], V["front_K"][k],
                        colour=(40, 40, 255), thick=3)
        JR._label(img, f"k{k} t={k / FPS:.2f}s tip {tip[k]:.2f} m")
        # crop around Lula's head
        from . import cand_lhmpp as CL
        uv, _ = CL.project(Jn["follow"][k][[15]], V["front_K"][k], V["front_R"][k], V["front_tvec"][k])
        cx, cy = int(np.clip(uv[0, 0], 300, 1620)), int(np.clip(uv[0, 1] + 120, 300, 780))
        tiles.append(cv2.resize(img[cy - 300:cy + 300, cx - 300:cx + 300], (360, 360), interpolation=cv2.INTER_AREA))
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[i:i + 6], 1) for i in range(0, len(tiles), 6)], 0)
    SHEETS.mkdir(parents=True, exist_ok=True)
    p = SHEETS / "hair_strands_front.jpg"
    cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return _ws(p)


# ================================================================================================ job / import
def _point_jr() -> None:
    JR.INPUT, JR.RENDERS, JR.POD_OUT, JR.LH = INPUT, FRAMES, POD_OUT, R2
    JR.JOB_NAME, JR.JOB_PATH = JOB_NAME, JOB_PATH


def job(max_minutes: float = 50) -> Path:
    _point_jr()
    p = JR.job(max_minutes, hair=HAIR)
    d = json.loads(p.read_text())
    d["after_download"] = ["python -m dancecap.mono4d.jl_v2_render import --session <sid>"]
    d["notes"].append("v2 driver realism/v2/footlock/final (facing-corrected refit + foot lock, John 26.2 s patch kept); "
                      "the realism run's cameras (front / orbit: smooth zoom-out where a hand left the frame)")
    p.write_text(json.dumps(d, indent=1), newline="\n")
    log(f"job -> {p} ({d['name']})")
    return p


def import_(sid: str | None = None) -> dict:
    _point_jr()
    JR.JOB_NAME = JOB_NAME + "_hair"
    rep = JR.import_run(sid)
    podj = POD_OUT / "lhmpp_render_pod.json"
    pod = json.loads(podj.read_text()) if podj.exists() else {}
    rep["hair"] = pod.get("hair")
    rep["fit_dir"] = json.loads((INPUT / "prep.json").read_text())["fit_dir"]
    (R2 / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"hair: {json.dumps(pod.get('hair'), default=str)}")
    return rep


# ================================================================================================ videos
_AUDIO = {}


def source_audio() -> tuple[np.ndarray, int]:
    """The heat's audio as float32 (n, 2) at its own rate (decoded once)."""
    if "pcm" not in _AUDIO:
        src = JR.audio_source()
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                             "stream=sample_rate,channels", "-of", "json", str(src)], capture_output=True, text=True)
        st = json.loads(pr.stdout)["streams"][0]
        sr, ch = int(st["sample_rate"]), int(st["channels"])
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:a:0", "-f", "f32le", "-ac", str(ch),
                              "-"], capture_output=True).stdout
        _AUDIO["pcm"] = np.frombuffer(raw, np.float32).reshape(-1, ch)
        _AUDIO["sr"] = sr
    return _AUDIO["pcm"], _AUDIO["sr"]


def build_wav(path: Path, segments: list[tuple[float, float]], fade_s: float = 0.25) -> float:
    """Concatenate source-audio slices [(start s, seconds)] with short fades at every cut -> 16-bit WAV."""
    import wave
    pcm, sr = source_audio()
    parts = []
    for (a0, dur) in segments:
        i0 = int(round(a0 * sr))
        n = int(round(dur * sr))
        x = np.zeros((n, pcm.shape[1]), np.float32)
        y = pcm[max(0, i0):i0 + n]
        x[:len(y)] = y
        if len(segments) > 1 or True:
            nf = int(fade_s * sr)
            ramp = np.linspace(0, 1, nf, dtype=np.float32)[:, None]
            x[:nf] *= ramp
            x[-nf:] *= ramp[::-1]
        parts.append(x)
    a = np.concatenate(parts, 0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(a.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())
    return len(a) / sr


def _ffmpeg_cmd(out: Path, W: int, H: int, audio: dict) -> list:
    src = audio.get("src")
    if audio["kind"] == "copy":
        aud = (["-ss", f"{audio['a0']:.4f}"] if audio["a0"] > 0 else []) + ["-t", f"{audio['dur']:.4f}", "-i", str(src)]
        ac = ["-c:a", "copy"]
    elif audio["kind"] == "aac":
        aud = (["-ss", f"{audio['a0']:.4f}"] if audio["a0"] > 0 else []) + ["-t", f"{audio['dur']:.4f}", "-i", str(src)]
        ac = ["-c:a", "aac", "-b:a", "192k"]
    elif audio["kind"] == "stretch":   # whole source audio, sped up by asetrate / out rate
        aud = ["-i", str(src)]
        ac = ["-af", f"asetrate={audio['asetrate']},aresample={audio['sr']}", "-c:a", "aac", "-b:a", "192k"]
    elif audio["kind"] == "wav":
        aud = ["-i", str(audio["wav"])]
        ac = ["-c:a", "aac", "-b:a", "192k"]
    else:
        raise ValueError(audio["kind"])
    return ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-framerate",
            "30", "-i", "-", *aud, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-r", "30", *ac, "-shortest", "-movflags", "+faststart", str(out)]


STRIDE = 1   # debug: render every STRIDE-th frame only (videos --stride)


def encode(name: str, n_frames: int, make, W: int, H: int, audio: dict, workers: int = 3) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    out = VIDEOS / name
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = subprocess.Popen(_ffmpeg_cmd(out, W, H, audio), stdin=subprocess.PIPE)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, fr in enumerate(ex.map(make, range(0, n_frames, STRIDE))):
            assert fr.shape == (H, W, 3), (name, n, fr.shape)
            p.stdin.write(fr.tobytes())
            if n % 300 == 0:
                log(f"{name}: frame {n}/{n_frames} ({time.time() - t0:.0f} s)")
    p.stdin.close()
    rc = p.wait()
    info = {"file": _ws(out), "rc": rc, "frames": n_frames, "seconds": round(time.time() - t0, 1),
            "audio": {k: (str(v) if isinstance(v, Path) else v) for k, v in audio.items()},
            "bytes": out.stat().st_size if out.exists() else None, "ffprobe": ffprobe(out)}
    log(f"{name}: {json.dumps(info, default=str)}")
    return info


def ffprobe(p: Path) -> list:
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                         "stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,"
                         "sample_rate", "-of", "json", str(p)], capture_output=True, text=True)
    return json.loads(pr.stdout or "{}").get("streams")


def floor_grid(K, R, tvec, W: int, H: int, step_m: float = 0.5, width_px: float = 1.1, strength: float = 0.16,
               fade_m: float = 9.0) -> np.ndarray:
    """(H, W) float32 darkening of grid lines on the floor y = 0 (anti-aliased by the floor's pixel footprint)."""
    K, R, tvec = np.asarray(K, float), np.asarray(R, float), np.asarray(tvec, float)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    dc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1).astype(np.float32)
    dw = dc @ R.astype(np.float32)
    C = (-R.T @ tvec).astype(np.float32)
    below = dw[..., 1] < -1e-5
    s = np.where(below, -C[1] / np.where(below, dw[..., 1], -1.0), 0.0).astype(np.float32)
    x = C[0] + s * dw[..., 0]
    z = C[2] + s * dw[..., 2]
    g = np.zeros((H, W), np.float32)
    for c in (x, z):
        gy, gx = np.gradient(c)
        fp = np.maximum(np.hypot(gx, gy), 1e-6)
        d = np.abs(c / step_m - np.round(c / step_m)) * step_m / fp      # px to the nearest line
        g = np.maximum(g, np.clip(1.0 - d / width_px, 0.0, 1.0))
    dist = s * np.linalg.norm(dw, axis=-1)
    g *= below * np.exp(-dist / fade_m) * strength
    return g


def _apply_grid(img: np.ndarray, g: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) * (1.0 - g[..., None]) + 0.5).astype(np.uint8)


def _load_views() -> dict:
    with np.load(INPUT / "views.npz") as z:
        return {k: np.asarray(z[k]) for k in z.files}


def videos(only=("front", "sbs", "orbit", "best", "ba_front", "ba_phone")) -> dict:
    from . import jl_lhmpp_preview as JP
    V = _load_views()
    with np.load(OLD_INPUT / "views.npz") as z:
        Vold = {k: np.asarray(z[k]) for k in z.files}
    prep_ = json.loads((INPUT / "prep.json").read_text())
    assert prep_["fit_sha1"] == {r: _sha1(DRIVER / f"{r}.npz") for r in ROLES}, "driver changed since prep"
    fits = {r: JR.load_fit(r, DRIVER) for r in ROLES}
    olds = {r: JR.load_fit(r, OLD_DRIVER) for r in ROLES}
    T = np.asarray(fits["lead"]["t"], float)
    Jg = {r: np.asarray(fits[r]["joints"], float) for r in ROLES}
    Jo = {r: np.asarray(olds[r]["joints"], float) for r in ROLES}
    ts_v = np.asarray(V["phone_t"], float)
    Jv = {r: JR.interp_many(fits[r], ts_v)["joints"] for r in ROLES}
    Jvo = {r: JR.interp_many(olds[r], ts_v)["joints"] for r in ROLES}
    src = JR.audio_source()
    res = {}
    pad = int(round(prep_["phone_hi"]["pad_px"]))
    Wh, Hh = JR.PHONE_HI["W"], JR.PHONE_HI["H"]

    def front_frame(k, renders=FRAMES, J=Jg, Vv=V, grid=None):
        bg = JR.background(Vv["front_K"][k], Vv["front_R"][k], Vv["front_tvec"][k], 1920, 1080,
                           JR.shadow_points({r: J[r][k] for r in ROLES}))
        if grid is not None:
            bg = _apply_grid(bg, grid)
        return JR.composite(bg, JR._read_rgba(renders / "front" / f"{k:05d}.png"))

    def phone_frame(i, renders=FRAMES, J=Jv, Vv=V):
        bg = JR.background(Vv["phone_hi_K"][i], Vv["phone_hi_R"][i], Vv["phone_hi_tvec"][i], Wh, Hh,
                           JR.shadow_points({r: J[r][i] for r in ROLES}))
        return JR.composite(bg, JR._read_rgba(renders / "phone_hi" / f"{i:05d}.png"))[:, pad:Wh - pad]

    def orbit_frame(n):
        k = int(V["orbit_key"][n]) - NV
        bg = JR.background(V["orbit_K"][n], V["orbit_R"][n], V["orbit_tvec"][n], 1080, 1080,
                           JR.shadow_points({r: Jg[r][k] for r in ROLES}))
        return JR.composite(bg, JR._read_rgba(FRAMES / "orbit" / f"{n:05d}.png"))

    def real_frame(i):
        return cv2.imread(str(JP.frame_path(i)))

    if "front" in only:
        def make_front(k):
            return JR._text(front_frame(k), [(CAPTION, (28, 1080 - 52), 26)])
        res["jl_tutorial_front.mp4"] = encode("jl_tutorial_front.mp4", NV, make_front, 1920, 1080,
                                              {"kind": "copy", "src": src, "a0": 0.0, "dur": NV / FPS})
    if "orbit" in only:
        N = len(V["orbit_key"])
        ks = V["orbit_key"] - NV
        a0 = float(T[ks[0]])

        def make_orbit(n):
            return JR._text(orbit_frame(n), [(CAPTION, (22, 1080 - 46), 22),
                                             (f"360 deg orbit, {T[ks[0]]:.1f}-{T[ks[-1]]:.1f} s", (22, 18), 22)])
        res["jl_orbit.mp4"] = encode("jl_orbit.mp4", N, make_orbit, 1080, 1080,
                                     {"kind": "aac", "src": src, "a0": a0, "dur": N / FPS})
    pw = int(round(538 * 1080 / 960))
    if "sbs" in only:
        x_l, x_r = 1920 // 2 - pw - 6, 1920 // 2 + 6
        idx = once_index(NV, len(ts_v))
        sr = stretch_rate(NV)

        def make_sbs(n):
            i = int(idx[n])
            canvas = np.full((1080, 1920, 3), 22, np.uint8)
            canvas[:, x_l:x_l + pw] = cv2.resize(real_frame(i), (pw, 1080), interpolation=cv2.INTER_LANCZOS4)
            canvas[:, x_r:x_r + pw] = cv2.resize(phone_frame(i), (pw, 1080), interpolation=cv2.INTER_AREA)
            return JR._text(canvas, [("real phone video", (x_l + 12, 10), 24),
                                     ("3D splats, same camera path", (x_r + 12, 10), 24),
                                     ("John & Lula", (40, 60), 40), ("Jack & Jill", (40, 112), 30),
                                     ("3D splats from one phone", (40, 160), 26),
                                     ("LHM++ avatars posed by", (40, 230), 20), ("MAMMA + SAM 3D Body", (40, 258), 20),
                                     ("v2: facing check, foot lock,", (40, 300), 20), ("swinging ponytail", (40, 328), 20),
                                     (f"t = {ts_v[i]:5.2f} s", (x_r + pw + 30, 1080 - 60), 24)])
        res["jl_side_by_side.mp4"] = encode("jl_side_by_side.mp4", NV, make_sbs, 1920, 1080,
                                            {"kind": "stretch", "src": src, "asetrate": sr["asetrate"], "sr": 44100,
                                             "drift_ms": round(sr["drift_ms"], 3), "frame_map": "output n = video frame n"})
    if "ba_front" in only or "ba_phone" in only:
        segs = []
        for (title, t0, sec) in WINDOWS:
            k0 = int(round(t0 * FPS))
            n = int(round(sec * FPS))
            ks = np.arange(k0, k0 + n)
            assert np.allclose(V["front_K"][ks], Vold["front_K"][ks]) and np.allclose(V["front_R"][ks], Vold["front_R"][ks])
            segs.append((title, ks))
    if "ba_front" in only:
        CW = 960
        allk = np.concatenate([s[1] for s in segs])
        from . import cand_lhmpp as CL
        xs = []
        for k in allk:
            pts = np.concatenate([np.concatenate([Jg[r][k][:22], Jo[r][k][:22]]) for r in ROLES])
            uv, _ = CL.project(pts, V["front_K"][k], V["front_R"][k], V["front_tvec"][k])
            xs.append(0.5 * (uv[:, 0].min() + uv[:, 0].max()))
        cxs = np.concatenate([crop_centres(np.asarray(xs)[j:j + len(s[1])], 1920, CW, 0.5 * FPS)
                              for j, s in zip(np.cumsum([0] + [len(s[1]) for s in segs[:-1]]), segs)])
        lab = [(ti, si) for si, (ti, ks) in enumerate(segs) for _ in ks]
        wav = VIDEOS / "_ba_front.wav"
        VIDEOS.mkdir(parents=True, exist_ok=True)
        dur = build_wav(wav, [(float(ks[0]) / FPS, len(ks) / FPS) for _, ks in segs])

        def make_ba(n):
            k = int(allk[n])
            g = floor_grid(V["front_K"][k], V["front_R"][k], V["front_tvec"][k], 1920, 1080)
            a = front_frame(k, OLD_RENDERS, Jo, Vold, g)
            b = front_frame(k, FRAMES, Jg, V, g)
            c0 = int(cxs[n]) - CW // 2
            canvas = np.concatenate([a[:, c0:c0 + CW], b[:, c0:c0 + CW]], 1).copy()
            canvas[:, CW - 2:CW + 2] = 20
            ti, si = lab[n]
            return JR._text(canvas, [("BEFORE: realism run (old fit, rigid hair)", (20, 14), 24),
                                     ("AFTER: v2 (facing check + foot lock + swinging ponytail)", (CW + 20, 14), 24),
                                     (ti, (20, 1080 - 50), 21), (f"t = {k / FPS:5.2f} s   same camera, 0.5 m floor grid",
                                                                   (CW + 20, 1080 - 50), 21)])
        res["jl_before_after_front.mp4"] = encode("jl_before_after_front.mp4", len(allk), make_ba, 1920, 1080,
                                                  {"kind": "wav", "wav": wav, "seconds": dur,
                                                   "segments_s": [[float(ks[0]) / FPS, len(ks) / FPS] for _, ks in segs]})
    if "ba_phone" in only:
        vsegs = []
        for (title, t0, sec) in WINDOWS:
            i0 = int(np.searchsorted(ts_v, t0 - 0.5 / VIDEO_FPS))
            vsegs.append((title, np.arange(i0, i0 + int(round(sec * FPS)))))
        alli = np.concatenate([s[1] for s in vsegs])
        lab = [ti for ti, ii in vsegs for _ in ii]
        wav = VIDEOS / "_ba_phone.wav"
        dur = build_wav(wav, [(float(ts_v[ii[0]]), len(ii) / FPS) for _, ii in vsegs])
        gap = 10
        x0 = (1920 - 3 * pw - 2 * gap) // 2
        import textwrap
        wrapped = {ti: textwrap.wrap(ti, 50) for ti, _ in vsegs}

        def make_bap(n):
            i = int(alli[n])
            canvas = np.full((1080, 1920, 3), 22, np.uint8)
            canvas[:, x0:x0 + pw] = cv2.resize(real_frame(i), (pw, 1080), interpolation=cv2.INTER_LANCZOS4)
            canvas[:, x0 + pw + gap:x0 + 2 * pw + gap] = cv2.resize(phone_frame(i, OLD_RENDERS, Jvo, Vold), (pw, 1080),
                                                                    interpolation=cv2.INTER_AREA)
            canvas[:, x0 + 2 * (pw + gap):x0 + 3 * pw + 2 * gap] = cv2.resize(phone_frame(i), (pw, 1080),
                                                                              interpolation=cv2.INTER_AREA)
            return JR._text(canvas, [("real phone video", (x0 + 10, 10), 22), ("BEFORE (old fit)", (x0 + pw + gap + 10, 10), 22),
                                     ("AFTER (v2)", (x0 + 2 * (pw + gap) + 10, 10), 22),
                                     (f"t = {ts_v[i]:5.2f} s", (x0 + 3 * pw + 2 * gap - 150, 46), 22)] +
                           [(ln, (x0 + 10, 1080 - 30 - 26 * (len(wrapped[lab[n]]) - q)), 19)
                            for q, ln in enumerate(wrapped[lab[n]])])
        res["jl_before_after_phone.mp4"] = encode("jl_before_after_phone.mp4", len(alli), make_bap, 1920, 1080,
                                                  {"kind": "wav", "wav": wav, "seconds": dur,
                                                   "segments_s": [[float(ts_v[ii[0]]), len(ii) / FPS] for _, ii in vsegs],
                                                   "frame_map": "each real frame once"})
    if "best" in only:
        items = []
        for (kind, t0, sec, cap) in BEST:
            n = int(round(sec * FPS))
            if kind == "front":
                k0 = int(round(t0 * FPS))
                items += [(kind, k0 + j, cap, j, n) for j in range(n)]
            elif kind == "orbit":
                ok = np.asarray(V["orbit_key"]) - NV
                j0 = int(np.searchsorted(ok, int(round(t0 * FPS))))
                items += [(kind, j0 + j, cap, j, n) for j in range(n)]
            else:
                i0 = int(np.searchsorted(ts_v, t0 - 0.5 / VIDEO_FPS))
                items += [(kind, i0 + j, cap, j, n) for j in range(n)]
        segs_a = []
        for (kind, t0, sec, cap) in BEST:
            first = next(it for it in items if it[0] == kind and it[3] == 0)
            a0 = float(ts_v[first[1]]) if kind == "sbs" else (
                float(T[int(V["orbit_key"][first[1]]) - NV]) if kind == "orbit" else float(T[first[1]]))
            segs_a.append((a0, sec))
        wav = VIDEOS / "_best.wav"
        VIDEOS.mkdir(parents=True, exist_ok=True)
        dur = build_wav(wav, segs_a, fade_s=0.4)
        fade_n = int(0.4 * FPS)

        def make_best(m):
            kind, i, cap, j, n = items[m]
            if kind == "front":
                fr = front_frame(i)
                txt = [(CAPTION, (28, 1080 - 52), 26), (cap, (28, 18), 24)]
            elif kind == "orbit":
                canvas = np.full((1080, 1920, 3), 22, np.uint8)
                k = int(V["orbit_key"][i]) - NV
                vi = int(np.argmin(np.abs(ts_v - T[k])))
                canvas[:, 40:40 + pw] = cv2.resize(real_frame(vi), (pw, 1080), interpolation=cv2.INTER_LANCZOS4)
                canvas[:, 1920 - 1080 - 40:1920 - 40] = orbit_frame(i)
                fr = canvas
                txt = [("real phone video", (52, 10), 22), (cap, (1920 - 1080 - 20, 18), 24),
                       (CAPTION, (1920 - 1080 - 20, 1080 - 46), 22)]
            else:
                canvas = np.full((1080, 1920, 3), 22, np.uint8)
                x_l, x_r = 1920 // 2 - pw - 6, 1920 // 2 + 6
                canvas[:, x_l:x_l + pw] = cv2.resize(real_frame(i), (pw, 1080), interpolation=cv2.INTER_LANCZOS4)
                canvas[:, x_r:x_r + pw] = cv2.resize(phone_frame(i), (pw, 1080), interpolation=cv2.INTER_AREA)
                fr = canvas
                txt = [("real phone video", (x_l + 12, 10), 24), ("3D splats, same camera path", (x_r + 12, 10), 24),
                       ("John & Lula", (40, 60), 40), ("Jack & Jill", (40, 112), 30),
                       ("3D splats from one phone", (40, 160), 26)]
            fr = JR._text(fr, txt)
            a = min(1.0, (j + 0.5) / fade_n, (n - j - 0.5) / fade_n)
            return fr if a >= 1.0 else (fr.astype(np.float32) * a + 0.5).astype(np.uint8)
        res["jl_best_30s.mp4"] = encode("jl_best_30s.mp4", len(items), make_best, 1920, 1080,
                                        {"kind": "wav", "wav": wav, "seconds": dur,
                                         "segments": [[k_, round(a_, 3), s_] for (k_, _, s_, _), (a_, _) in zip(BEST, segs_a)]})
    for w in VIDEOS.glob("_*.wav"):
        w.unlink()
    rp = VIDEOS / "videos.json"
    old = json.loads(rp.read_text()) if rp.exists() else {}
    old.update(res)
    rp.write_text(json.dumps(old, indent=1, default=str))
    return res


# ================================================================================================ self-check
def _decode_gray_small(p: Path, w: int = 192, crop: str | None = None) -> np.ndarray:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "json", str(p)], capture_output=True, text=True)
    st = json.loads(probe.stdout)["streams"][0]
    cw, ch = st["width"], st["height"]
    if crop:
        cw, ch = int(crop.split(":")[0]), int(crop.split(":")[1])
    h = int(round(ch * w / cw / 2) * 2)
    vf = (f"crop={crop}," if crop else "") + f"scale={w}:{h},format=gray"
    pr = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vf", vf, "-f", "rawvideo", "-"], capture_output=True)
    return np.frombuffer(pr.stdout, np.uint8).reshape(-1, h, w)


def _frames_at(p: Path, idx: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(p))
    out = []
    for q in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(q))
        ok, fr = cap.read()
        out.append(fr if ok else None)
    cap.release()
    return out


def _audio_pcm(p: Path, sr: int = 8000) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-f",
                          "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def _lag_ms(a: np.ndarray, b: np.ndarray, sr: int, max_ms: float = 60) -> float:
    """Lag (ms) of b relative to a by cross-correlation (positive: b late)."""
    m = int(max_ms * sr / 1000)
    a = a - a.mean()
    b = b - b.mean()
    best, lag = -1e18, 0
    for L in range(-m, m + 1):
        if L >= 0:
            c = float(np.dot(a[:len(a) - L], b[L:len(a)]))
        else:
            c = float(np.dot(a[-L:], b[:len(b) + L]))
        if c > best:
            best, lag = c, L
    return lag * 1000.0 / sr


def panel_match(p: Path, crop: str, src_idx: np.ndarray) -> dict:
    """Decode every frame, cut the real-video panel (crop w:h:x:y) and compare it (grey MAD, full resolution) with the
    frame JPEG it should show (src_idx[n], resized like the encoder's input) and the JPEGs before / after it."""
    from . import jl_lhmpp_preview as JP
    w, h, x, y = (int(v) for v in crop.split(":"))
    cache = {}

    def jp(i):
        if i not in cache:
            im = cv2.imread(str(JP.frame_path(i)))
            cache[i] = cv2.cvtColor(cv2.resize(im, (w, h), interpolation=cv2.INTER_LANCZOS4),
                                    cv2.COLOR_BGR2GRAY).astype(np.int16)[60:h - 80]
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
        return cache[i]
    cap = cv2.VideoCapture(str(p))
    E, n = [], 0
    while n < len(src_idx):
        ok, fr = cap.read()
        if not ok:
            break
        q = cv2.cvtColor(fr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.int16)[60:h - 80]
        i = int(src_idx[n])
        E.append([float(np.abs(q - jp(j)).mean()) if 0 <= j < NV else np.inf for j in (i - 1, i, i + 1)])
        n += 1
    cap.release()
    E = np.asarray(E)
    best = E.argmin(1) - 1
    return {"frames": int(len(E)), "best_offset_counts": {str(o): int((best == o).sum()) for o in (-1, 0, 1)},
            "mad_intended_median": round(float(np.median(E[:, 1])), 2), "mad_intended_max": round(float(E[:, 1].max()), 2),
            "mad_neighbour_median": round(float(np.median(np.minimum(E[:, 0], E[:, 2]))), 2),
            "frames_not_intended": [int(v) for v in np.nonzero(best != 0)[0]][:20],
            "distinct_source_frames_shown": int(len(np.unique(np.asarray(src_idx[:len(E)])[best == 0])))}


def check() -> dict:
    from . import jl_lhmpp_preview as JP
    out = {}
    SHEETS.mkdir(parents=True, exist_ok=True)
    src = JR.audio_source()
    sa = _audio_pcm(src)
    sr = 8000
    # the source video's own successive-frame differences (to tell a repeat from a still moment)
    src_g = _decode_gray_small(src)
    src_d = np.abs(np.diff(src_g.astype(np.int16), axis=0)).mean((1, 2))
    for p in sorted(VIDEOS.glob("*.mp4")):
        info = {"ffprobe": ffprobe(p)}
        pr = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                             "stream=nb_read_frames", "-of", "json", str(p)], capture_output=True, text=True)
        info["decoded_frames"] = int(json.loads(pr.stdout)["streams"][0]["nb_read_frames"])
        dec = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-f", "null", "-"], capture_output=True, text=True)
        info["decode_errors"] = dec.stderr.strip()[:500]
        n = info["decoded_frames"]
        # whole-file successive differences: repeats (diff ~0) and the real panel vs the source
        crop, src_idx = None, None
        pw = int(round(538 * 1080 / 960))
        if p.stem.startswith("jl_side_by_side"):
            crop = f"{pw}:1080:{1920 // 2 - pw - 6}:0"
            src_idx = np.arange(NV)
        elif p.stem == "jl_before_after_phone":
            crop = f"{pw}:1080:{(1920 - 3 * pw - 20) // 2}:0"
            ts_v = np.arange(NV) / VIDEO_FPS
            src_idx = np.concatenate([np.searchsorted(ts_v, t0 - 0.5 / VIDEO_FPS) + np.arange(int(round(sec * FPS)))
                                      for (_, t0, sec) in WINDOWS])
        g = _decode_gray_small(p, crop=crop)
        d = np.abs(np.diff(g.astype(np.int16), axis=0)).mean((1, 2))
        info["successive_diff_lt_0.15_frames"] = [int(i + 1) for i in np.nonzero(d < 0.15)[0]][:40]
        if crop:   # real panel of EVERY output frame vs the frame JPEG it should show and its two neighbours (full res)
            info["real_panel_frame_match"] = panel_match(p, crop, src_idx)
            m = min(len(g), len(src_idx))
            info["source_own_repeats_in_shown_frames"] = [int(i) for i in src_idx[:m][1:][np.isin(src_idx[:m][1:] - 1,
                                                          np.nonzero(src_d < 0.15)[0])]][:40]
        # audio sync vs the source (8 kHz mono): lag at the start / middle / end
        oa = _audio_pcm(p)
        meta = json.loads((VIDEOS / "videos.json").read_text()).get(p.name, {})
        au = meta.get("audio", {})
        lags = {}
        if au.get("kind") in ("copy", "aac", "stretch"):
            a0 = float(au.get("a0", 0.0))
            f = (FPS / VIDEO_FPS) if au.get("kind") == "stretch" else 1.0
            for frac in (0.1, 0.5, 0.9):
                to = frac * len(oa) / sr
                ts_ = a0 + to * f
                i_o, i_s = int(to * sr), int(ts_ * sr)
                L = int(2.0 * sr)
                if i_s + L < len(sa) and i_o + L < len(oa):
                    lags[f"{to:.1f}s"] = round(_lag_ms(sa[i_s:i_s + L], oa[i_o:i_o + L], sr), 2)
        if au.get("kind") == "wav":     # each segment's first 2 s vs the source at the segment's start
            segs = au.get("segments_s") or [[a_, s_] for (_, a_, s_) in au.get("segments", [])]
            to = 0.0
            for (a0, sec) in segs:
                i_o, i_s, L = int((to + 0.5) * sr), int((a0 + 0.5) * sr), int(2.0 * sr)
                lags[f"seg@{a0:.2f}s"] = round(_lag_ms(sa[i_s:i_s + L], oa[i_o:i_o + L], sr), 2)
                to += sec
        info["audio_lag_ms_vs_source"] = lags
        # 20 frames, LOOK
        idx = list(np.linspace(3, n - 4, 20).round().astype(int))
        frs = _frames_at(p, idx)
        tiles = []
        for q, fr in zip(idx, frs):
            if fr is None:
                continue
            th = int(round(fr.shape[0] * 480 / fr.shape[1]))
            fr = cv2.resize(fr, (480, th), interpolation=cv2.INTER_AREA)
            JR._label(fr, f"#{q} {q / FPS:.2f}s", (6, th - 8), 0.55)
            tiles.append(fr)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
        sp = SHEETS / f"check_{p.stem}.jpg"
        cv2.imwrite(str(sp), sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])
        info["sheet"] = _ws(sp)
        info["sheet_frames"] = idx
        out[p.name] = info
        log(f"check {p.name}: {json.dumps({k: info[k] for k in info if k not in ('ffprobe', 'sheet_frames')}, default=str)}")
    (VIDEOS / "check.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.jl_v2_render")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    sub.add_parser("hair")
    p = sub.add_parser("job")
    p.add_argument("--max-minutes", type=float, default=50)
    p = sub.add_parser("import")
    p.add_argument("--session")
    p = sub.add_parser("videos")
    p.add_argument("--only", default="front,sbs,orbit,best,ba_front,ba_phone")
    p.add_argument("--stride", type=int, default=1, help="debug: every n-th frame only")
    p.add_argument("--renders", default=None, help="debug: read the new renders from this dir")
    p.add_argument("--videos", default=None, help="debug: write the videos to this dir")
    sub.add_parser("check")
    a = ap.parse_args(argv)
    global STRIDE, FRAMES, VIDEOS
    if getattr(a, "stride", None):
        STRIDE = a.stride
    if getattr(a, "renders", None):
        FRAMES = Path(a.renders).resolve()
    if getattr(a, "videos", None):
        VIDEOS = Path(a.videos).resolve()
    if a.cmd == "prep":
        prep()
    elif a.cmd == "hair":
        hair()
    elif a.cmd == "job":
        job(a.max_minutes)
    elif a.cmd == "import":
        import_(a.session)
    elif a.cmd == "videos":
        videos(tuple(x for x in a.only.split(",") if x))
    elif a.cmd == "check":
        check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
