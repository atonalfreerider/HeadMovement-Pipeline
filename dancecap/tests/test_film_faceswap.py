"""dancecap.film_faceswap: the swap weight never pops, pose proxies, the YUV paste leaves every unmasked pixel bit-exact, the sidecar
validator, the phones' frustum video surfaces as occluders, no flicker in fast turns (minimum-on per dancer, bridged dips).
Pure numpy / scipy / cv2: FaceFusion itself is not needed (that path is checked on the film excerpts, RECAP_FILM.md)."""
from __future__ import annotations

import sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

import numpy as np
import pytest

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # run from the workspace root too
from dancecap import film_faceswap as fs  # noqa: E402

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")


def test_weight_fades_inside_the_valid_span_and_drops_short_spans():
    cfg = fs.Cfg()
    g = np.zeros(80)
    g[10:50] = 1.0                  # a 40-frame qualifying stretch
    g[60:66] = 1.0                  # 6 frames: too short, must not blink
    w, raw = fs.temporal_weight(g, np.ones(80), cfg)
    assert w[:10].max() == 0 and w[50:].max() == 0      # nothing outside the stretch (ends at 0 on the last qualifying frame)
    assert w[25:35].min() == 1.0
    assert np.abs(np.diff(w)).max() <= 1.0 / (cfg.fade_frames // 2 * 2 + 1) + 1e-9   # fades, no jumps
    assert raw[60:66].max() == 1.0 and w[55:80].max() == 0


def test_hysteresis_holds_through_a_dip_and_a_phone_shot_cuts_it():
    cfg = fs.Cfg()
    g = np.full(60, 0.8)
    g[30] = 0.3                     # between off (0.25) and on (0.5): stays on
    w, _ = fs.temporal_weight(g, np.ones(60), cfg)
    assert w[30] == 1.0
    shot = np.ones(60)
    shot[40:] = 0.5                 # a flight into a phone: the phone video shows through
    w, _ = fs.temporal_weight(g, shot, cfg)
    assert w[40:].max() == 0 and w[39] < w[30]


def test_open_range_ends_do_not_fade():
    cfg = fs.Cfg()
    g = np.ones(40)
    w, _ = fs.temporal_weight(g, np.ones(40), cfg, open_start=True, open_end=True)
    assert w.min() == 1.0
    w, _ = fs.temporal_weight(g, np.ones(40), cfg)
    assert w[0] == 0 and w[-1] == 0


def test_pose_proxies_on_the_smplx_head():
    # frontal SMPL-X 5 points (x right, y down in the image), then the same head turned 30 deg to the image's right
    P = np.array([[-0.0316, 0.3107, 0.0732], [0.0316, 0.3107, 0.0732], [0.0, 0.2843, 0.1140], [-0.0237, 0.2430, 0.0847],
                  [0.0237, 0.2430, 0.0847]])

    def img(theta):
        c, s = np.cos(theta), np.sin(theta)
        X = P[:, 0] * c + P[:, 2] * s
        return np.stack([X * 1000, -P[:, 1] * 1000], 1)

    assert abs(fs.yaw5(img(0.0))) < 1.0
    assert abs(fs.pitch5(img(0.0))) < 1.0
    assert 24 < fs.yaw5(img(np.radians(30))) < 36
    assert fs.iod(img(0.0)) == pytest.approx(63.2, abs=0.1)
    assert fs.boost_for(20) == 128 and fs.boost_for(60) == 256 and fs.boost_for(100) == 512


def test_paste_changes_only_masked_pixels():
    rng = np.random.default_rng(3)
    w, h = 64, 96
    buf = rng.integers(0, 256, w * h * 3 // 2, dtype=np.uint8)
    orig = buf.copy()
    Y, U, V = fs.planes(buf, w, h)
    bgr = cv2.cvtColor(buf.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
    crop = np.full((32, 32, 3), 200, np.float32)
    mask = np.zeros((32, 32), np.float32)
    mask[8:24, 8:24] = 1.0
    M = np.array([[1.0, 0, -10], [0, 1.0, -20]])          # frame -> crop: the crop sits at (10, 20)
    box = fs.paste_yuv(Y, U, V, bgr, crop, mask, M, 1.0)
    assert box is not None
    Yo, Uo, Vo = fs.planes(orig, w, h)
    changed = Y != Yo
    assert changed.any()
    ys, xs = np.nonzero(changed)
    assert ys.min() >= 20 + 7 and ys.max() <= 20 + 24 and xs.min() >= 10 + 7 and xs.max() <= 10 + 24
    assert np.array_equal(Y[:20], Yo[:20]) and np.array_equal(U[:10], Uo[:10]) and np.array_equal(V[:, :5], Vo[:, :5])
    # zero weight: nothing at all
    buf2 = orig.copy()
    Y2, U2, V2 = fs.planes(buf2, w, h)
    assert fs.paste_yuv(Y2, U2, V2, bgr.copy(), crop, mask, M, 0.0) is None
    assert np.array_equal(buf2, orig)


def test_sidecar_validation():
    sc = {"format": fs.SIDECAR_FORMAT, "version": 1, "landmarks": list(fs.LANDMARKS5), "width": 1080, "height": 1920,
          "dancers": [{"key": "lead"}, {"key": "follow"}],
          "f": [{"t": 0.0, "d": [{"k": "lead", "in": 1, "lm": [[1, 2]] * 5}]}, {"t": 1 / 30, "d": []}]}
    info = {"width": 1080, "height": 1920, "frames": 2}
    assert fs.validate_sidecar(sc, info) == []
    bad = dict(sc, landmarks=["eye_l", "eye_r", "nose_tip", "mouth_r", "mouth_l"])
    assert any("landmarks" in p for p in fs.validate_sidecar(bad, info))
    assert any("frames" in p for p in fs.validate_sidecar(sc, dict(info, frames=3)))
    bad = dict(sc, f=[{"t": 0.0, "d": [{"k": "lead", "in": 1, "lm": [[1, 2]] * 4}]}])
    assert any("'lm'" in p for p in fs.validate_sidecar(bad, None))


# ---------------------------------------------------------------------------------------------- v3.1: frustum surfaces, no flicker
def _face(cx=500.0, cy=800.0, d=40.0):
    """5 points of a frontal face (subject's right eye first = image left), inter-ocular d px"""
    return [[cx - d / 2, cy], [cx + d / 2, cy], [cx, cy + 0.45 * d], [cx - 0.4 * d, cy + 0.9 * d], [cx + 0.4 * d, cy + 0.9 * d]]


def _sidecar(n, surfaces_at=None, z_face=3.0, layer_at=None):
    """a 3D-view sidecar: the lead's face on the splat dancers (v4.2 per-dancer layer) unless layer_at(k) says (layer, la, video_alpha, splats)"""
    fr = []
    for k in range(n):
        lay, la, va, sv = layer_at(k) if layer_at else ("splat", 1.0, 0.0, 1.0)
        d = {"k": "lead", "in": 1, "lm": _face(), "z": [z_face] * 5, "yaw": 0.0, "pitch": 0.0, "vis": 1.0}
        if lay is not None:
            d.update(layer=lay, la=la)
        e = {"t": k / 30, "view": "3d", "video_alpha": va, "ui": [], "d": [d]}
        if sv is not None:
            e["splats"] = sv
        if surfaces_at:
            sf = surfaces_at(k)
            if sf:
                e["surfaces"] = sf
        fr.append(e)
    return {"format": fs.SIDECAR_FORMAT, "version": 1, "landmarks": list(fs.LANDMARKS5), "width": 1080, "height": 1920,
            "dancers": [{"key": "lead"}, {"key": "follow"}], "f": fr}


def _surface(alpha=0.9, z=1.0, quad=((300, 600), (760, 600), (760, 1100), (300, 1100))):
    return [{"phone": "02", "alpha": alpha, "quad": [list(q) for q in quad], "z": [z] * 4, "role": "enter"}]


def _lead(plan):
    return next(p for p in plan if p["role"] == "lead")


def test_a_face_behind_a_frustum_surface_is_never_swapped_and_fades_out_before_it():
    cfg = fs.Cfg()
    n = 120
    sc = _sidecar(n, lambda k: _surface() if 60 <= k < 90 else None)
    frames = [{"dets": []} for _ in range(n)]
    # a detector hit on the surface's real face, right where the projected points are: it must not refine (or re-enable) anything
    for k in range(60, 90):
        frames[k]["dets"] = [{"lm5": (np.asarray(_face()) + 2.0).tolist()}]
    p = _lead(fs.plan_sidecar(frames, sc, 0, n, cfg))
    w = p["w"]
    assert w[:45].min() == 1.0 and w[60:90].max() == 0.0          # hidden by phone 02's surface: no swap at all
    assert w[59] == 0.0 and 0.0 < w[52] < 1.0                     # the fade reaches 0 on the last frame before the surface covers it
    assert np.abs(np.diff(w)).max() <= 1.0 / (cfg.fade_frames // 2 * 2 + 1) + 1e-9   # no pop: fades only
    assert w[-1] == 1.0 and p["behind_surface"] == {"02": 30} and p["n_det"] == 0
    assert np.allclose(p["lm"][70], _face())                      # the projected points, not the surface's face


def test_a_surface_behind_the_face_a_transparent_one_or_one_elsewhere_hides_nothing():
    cfg = fs.Cfg()
    n = 90
    frames = [{"dets": []} for _ in range(n)]
    for surf in (_surface(z=6.0),                                  # behind the dancer (the eye looks past him at the phone's frustum)
                 _surface(alpha=0.04),                             # fading in, not yet opaque enough to hide anything
                 _surface(quad=((800, 100), (1000, 100), (1000, 400), (800, 400)))):   # elsewhere on the screen
        p = _lead(fs.plan_sidecar(frames, _sidecar(n, lambda k, s=surf: s if 30 <= k < 60 else None), 0, n, cfg))
        assert p["w"].min() == 1.0 and p["behind_surface"] == {}
    # the face box counts, not only the 5 points: a surface covering the face's edge (the paste region) hides it too
    edge = _surface(quad=((530, 700), (900, 700), (900, 900), (530, 900)))          # right of the eye (520), inside the box (540)
    assert fs.surface_hit(np.asarray(_face()), [3.0] * 5, edge, cfg) == "02"
    assert fs.surface_hit(np.asarray(_face()), [3.0] * 5, [dict(edge[0], z=None)], cfg) == "02"   # no depths: in front
    assert fs.surface_hit(np.asarray(_face()), None, [dict(edge[0], full=1, quad=[[0, 0], [1, 0], [1, 1]])], cfg) == "02"


def test_point_in_poly_and_the_plane_depth_of_a_tilted_surface():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert fs.point_in_poly([[5, 5], [11, 5], [-1, -1], [9.9, 0.1]], sq).tolist() == [True, False, False, True]
    assert fs.point_in_poly([[5, 5]], sq[::-1]).tolist() == [True]                    # either winding
    # a plane tilted in depth, seen by a pinhole (f 1000 px, centre 540 / 960): its 4 corners and an inside point
    f, cx, cy = 1000.0, 540.0, 960.0
    P = np.array([[-0.5, -0.4, 1.5], [0.5, -0.4, 2.5], [0.5, 0.4, 2.5], [-0.5, 0.4, 1.5]])
    proj = lambda X: np.c_[f * X[:, 0] / X[:, 2] + cx, f * X[:, 1] / X[:, 2] + cy]
    q = proj(P)
    inner = 0.25 * P[0] + 0.25 * P[1] + 0.3 * P[2] + 0.2 * P[3]                     # a point on the plane
    assert fs.surface_depth_at(proj(inner[None]), q, P[:, 2])[0] == pytest.approx(inner[2], rel=1e-9)


def test_fast_turns_do_not_flicker_short_faces_go_and_dips_are_bridged():
    cfg = fs.Cfg()
    # a fast-turn passage: the face qualifies for 0.5-0.7 s every ~2 s (B 350-360 s): none of it is swapped
    g = np.zeros(240)
    for a, b in ((10, 25), (70, 90), (130, 148), (190, 211)):
        g[a:b] = 1.0
    w, raw = fs.temporal_weight(g, np.ones(240), cfg)
    assert raw.max() == 1.0 and w.max() == 0.0
    # a held face with a 6-frame dip where it is still a face (gate 0.1: e.g. 62 deg of yaw): one span, no dip in the weight
    g = np.ones(100)
    g[50:56] = 0.1
    w, _ = fs.temporal_weight(g, np.ones(100), cfg, open_start=True, open_end=True)
    assert w.min() == 1.0
    # the same dip with the back of the head (gate 0): the swap fades out and back in (two spans of >= 1 s), no hard step
    g[50:56] = 0.0
    w, _ = fs.temporal_weight(g, np.ones(100), cfg, open_start=True, open_end=True)
    assert w[50:56].max() == 0.0 and w[:30].min() == 1.0 and w[-20:].min() == 1.0
    # never bridged into a phone's video
    g[50:56] = 0.1
    shot = np.ones(100)
    shot[50:56] = 0.0
    w, _ = fs.temporal_weight(g, shot, cfg, open_start=True, open_end=True)
    assert w[50:56].max() == 0.0


def test_the_minimum_on_counts_a_dancer_over_all_of_its_tracks():
    cfg = fs.Cfg()
    # the tracker split one face into two tracks (frames 0-19, 20-44): together 45 frames >= min_on, both are swapped;
    # a lone 20-frame track of the other dancer is not
    plan = [{"role": "lead", "k0": 0, "k1": 19, "on": np.ones(20, bool)}, {"role": "lead", "k0": 20, "k1": 44, "on": np.ones(25, bool)},
            {"role": "follow", "k0": 60, "k1": 79, "on": np.ones(20, bool)}, {"role": None, "k0": 0, "k1": 9, "on": np.ones(10, bool)}]
    fs.min_on_per_dancer(plan, 100, cfg)
    assert plan[0]["w"].max() == 1.0 and plan[1]["w"].max() == 1.0 and plan[2]["w"].max() == 0.0 and plan[3]["w"].max() == 0.0
    spans = fs.swap_spans(plan, 30.0, 0)
    assert spans["lead"]["n"] == 1 and spans["lead"]["shortest_s"] >= cfg.min_on_frames / 30.0 - 2 * cfg.fade_frames / 30.0


def test_sidecar_surfaces_are_validated_and_the_heuristic_skips_every_frustum_window(tmp_path):
    sc = _sidecar(2, lambda k: _surface())
    assert fs.validate_sidecar(sc, None) == []
    bad = _sidecar(2, lambda k: [{"phone": "02", "alpha": 0.5, "quad": [[0, 0], [1, 1]]}])
    assert any("quad" in p for p in fs.validate_sidecar(bad, None))
    bad = _sidecar(2, lambda k: [dict(_surface()[0], z=[1.0, 2.0])])
    assert any("'z'" in p for p in fs.validate_sidecar(bad, None))
    # no sidecar: a v3 direction's frustum envelopes switch the swap off while any surface shows
    d = {"segments": [{"film": [0.0, 10.0], "camera": {"mode": "orbit"}}],
         "flow": {"frustums": [{"camera": "02", "film": [2.0, 2.5, 3.5, 3.5]}]}}
    path = tmp_path / "direction.json"
    path.write_text(__import__("json").dumps(d))
    w, _ = fs.shot_weights(path, 0, 300, 30.0, fs.Cfg())
    t = np.arange(300) / 30.0
    assert w[(t >= 2.0) & (t <= 3.5)].max() == 0.0 and w[t < 1.9].min() == 1.0 and w[t > 3.6].min() == 1.0


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None or __import__("shutil").which("ffprobe") is None,
                    reason="ffmpeg / ffprobe not on PATH")
def test_a_whole_film_keeps_the_recordings_video_start(tmp_path):
    """The recorder's MP4 starts its video at 0.033 s and its audio at 0; the swapped whole film must keep that A/V relation
    (2026-10-21: the encoded raw pipe dropped -itsoffset and the picture ran one frame early; restore_start remuxes it back)."""
    import subprocess
    base, film, out = tmp_path / "base.mp4", tmp_path / "rec.mp4", tmp_path / "out.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "nullsrc=s=64x112:r=30:d=1,geq=lum='mod(N*40+X*2+Y,256)':cb=128:cr=128", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=1.04", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(base)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-itsoffset", "0.033", "-i", str(base), "-i", str(base), "-map", "0:v:0", "-map", "1:a:0",
                    "-c", "copy", "-video_track_timescale", "30000", str(film)], check=True)
    info = fs.ffprobe(film)
    assert abs(info["v_start"] - 0.033) < 1e-3 and info["frames"] == 30
    w = fs.Writer(out, film, info, 0, info["frames"], info["width"], info["height"], fs.Cfg())
    for buf in fs.YuvReader(film, info, 0, info["frames"]):
        w.write(buf)
    w.close()
    res = fs.restore_start(out, film)
    got = fs.ffprobe(out)
    assert abs(got["v_start"] - info["v_start"]) < 5e-4 and got["frames"] == info["frames"] and res.get("ok", True)
    if res["remuxed"]:
        assert res["video_md5_unchanged"] and res["audio_md5_equals_film"]
    assert fs.stream_md5(out, "a") == fs.stream_md5(film, "a")
    assert fs.restore_start(out, film)["remuxed"] is False      # idempotent
    # frame k of the output is frame k of the film (the reader hands over exactly the coded frames: no duplicated first frame)
    src = [fs.planes(b, 64, 112)[0].astype(np.int16) for b in fs.YuvReader(film, info, 0, 30)]
    dst = [fs.planes(b, 64, 112)[0].astype(np.int16) for b in fs.YuvReader(out, fs.ffprobe(out), 0, 30)]
    assert len(src) == len(dst) == 30
    same = [float(np.abs(a - b).mean()) for a, b in zip(src, dst)]
    shifted = [float(np.abs(a - b).mean()) for a, b in zip(src[:-1], dst[1:])]
    assert max(same) < 3.0 and min(shifted) > 5 * max(same)
    # an excerpt starts on its own first frame too
    part = [fs.planes(b, 64, 112)[0].astype(np.int16) for b in fs.YuvReader(film, info, 10, 3)]
    assert all(np.array_equal(a, b) for a, b in zip(part, src[10:13]))


def test_overlay_strokes_over_a_face_are_kept_and_skin_eyes_and_hair_are_not():
    """Grey / blue skeleton strokes that cross a swapped face are restored from the recording (Larissa at 316 s, 2026-10-21);
    skin, lips, red hair, dark eyes and small bright eye whites are left to the swap."""
    h, w = 120, 110
    bgr = np.zeros((h, w, 3), np.uint8)
    bgr[:] = (126, 145, 202)                                   # splat skin (B, G, R)
    bgr[:, :18] = (36, 41, 117)                                # red hair
    bgr[40:46, 30:42] = (35, 40, 45)                           # a dark eye
    bgr[41:44, 44:47] = (205, 200, 200)                        # a small bluish eye white (9 px)
    bgr[80:86, 40:70] = (95, 110, 170)                         # lips
    stroke = np.zeros((h, w), bool)
    for x in range(w):                                         # a grey-lavender load line across the face, 6 px wide
        y0 = int(20 + 0.6 * x)
        stroke[y0:y0 + 6, x] = True
    bgr[stroke] = (158, 149, 155)
    blue = np.zeros((h, w), bool)
    blue[100:108, 60:110] = True                               # a blue arm line entering from the side
    bgr[blue] = (240, 110, 60)
    m = fs.overlay_mask(bgr)
    assert m[stroke].all() and m[blue].all()
    grown = cv2.dilate((stroke | blue).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert not (m & ~grown).any()                              # nothing beyond the strokes and their 1-px rim
    assert not m[41:44, 44:47].any() or grown[41:44, 44:47].all()
    # the restore puts the recorded pixels back exactly there (Y full res, chroma where any of the 2 x 2 is a stroke)
    Y, U, V = np.full((h, w), 50, np.uint8), np.full((h // 2, w // 2), 60, np.uint8), np.full((h // 2, w // 2), 70, np.uint8)
    Yo, Uo, Vo = np.full((h, w), 200, np.uint8), np.full((h // 2, w // 2), 210, np.uint8), np.full((h // 2, w // 2), 220, np.uint8)
    n = fs.restore_overlay(Y, U, V, Yo, Uo, Vo, bgr, [0, 0, w, h])
    assert n == int(m.sum()) and (Y[m] == 200).all() and (Y[~m] == 50).all()
    assert (U == 210).sum() > 0 and (U == 60).sum() > 0
    # a face without strokes: nothing restored
    assert not fs.overlay_mask(np.full((h, w, 3), (126, 145, 202), np.uint8)).any()


# ---------------------------------------------------------------------------------------------- v4.2: the splat dancers only
def test_only_a_splat_dancer_face_is_swapped_never_a_phone_video_avatar_or_skeleton_view():
    """owner 2026-10-22: 'apply the face fusion to the gauss splats only' - the sidecar's per-dancer layer decides"""
    cfg = fs.Cfg()
    n = 300

    def layer_at(k):
        if k < 60:
            return ("splat", 1.0, 0.0, 1.0)            # the splat dancers in a 3D view: swapped
        if k < 100:
            return ("avatar", 0.35, 0.0, 0.0)          # the mesh avatars (splats off)
        if k < 140:
            return ("skeleton", 0.0, 0.0, 0.0)         # skeleton lines only
        if k < 180:
            return ("splat", 1.0, 1.0, 1.0)            # a phone's own full-frame video over everything
        if k < 220:
            return ("splat", 0.0, 0.0, 0.0)            # the splats hidden (a phone's window)
        return ("splat", 1.0, 0.0, 1.0)
    frames = [{"dets": []} for _ in range(n)]
    for k in range(60, 220):                           # a real face detected right on the projected points: it must not re-enable anything
        frames[k]["dets"] = [{"lm5": (np.asarray(_face()) + 1.0).tolist()}]
    p = _lead(fs.plan_sidecar(frames, _sidecar(n, layer_at=layer_at), 0, n, cfg))
    w = p["w"]
    assert w[:45].min() == 1.0 and w[60:220].max() == 0.0 and w[240:].min() == 1.0
    assert np.abs(np.diff(w)).max() <= 1.0 / (cfg.fade_frames // 2 * 2 + 1) + 1e-9           # fades only
    assert p["layers"] == {"splat": 220, "avatar": 40, "skeleton": 40}
    # a half-faded splat body swaps at its own opacity; the phone video's opacity multiplies in
    assert fs.splat_share({"splats": 1.0}, {"layer": "splat", "la": 0.4}) == pytest.approx(0.4)
    assert fs.splat_share({"splats": 1.0}, {"layer": "avatar", "la": 1.0}) == 0.0
    # an older sidecar (no per-dancer layer): the frame's splats share; nothing said: no swap
    assert fs.splat_share({"splats": 0.0}, {}) == 0.0 and fs.splat_share({"splats": 1.0}, {}) == 1.0 and fs.splat_share({}, {}) == 0.0
    old = _sidecar(n, layer_at=lambda k: (None, None, 0.0, 1.0 if k < 150 else 0.0))
    w = _lead(fs.plan_sidecar([{"dets": []} for _ in range(n)], old, 0, n, cfg))["w"]
    assert w[:130].min() == 1.0 and w[150:].max() == 0.0
    # the validator knows the layers
    bad = _sidecar(2, layer_at=lambda k: ("phone", 1.0, 0.0, 1.0))
    assert any("layer" in x for x in fs.validate_sidecar(bad, None))
    assert fs.validate_sidecar(_sidecar(2), None) == []


def test_swap_refuses_to_run_without_a_sidecar_unless_asked_for_a_prototype(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "ffprobe", lambda path: {"width": 64, "height": 112, "fps": 30.0, "frames": 3, "v_start": 0.0, "fmt_start": 0.0,
                                                     "duration": 0.1, "has_audio": False})
    with pytest.raises(SystemExit, match="sidecar"):
        fs.main(["swap", "--film", str(tmp_path / "f.mp4"), "--out", str(tmp_path / "o.mp4")])
    with pytest.raises(SystemExit):
        fs.main(["swap", "--film", str(tmp_path / "f.mp4"), "--out", str(tmp_path / "o.mp4"), "--phone-shots", "swap"])


# ---------------------------------------------------------------------------------------------- v4.3: the swap on a Runpod GPU
_HAVE_FFMPEG = __import__("shutil").which("ffmpeg") is not None and __import__("shutil").which("ffprobe") is not None


def _run(*cmd):
    __import__("subprocess").run([str(c) for c in cmd], check=True)


def _recording(path, frames=30, audio=True):
    """a tiny 'recording': video (30 fps, timescale 30000 like the recorder's) + an AAC voice track, both starting at 0"""
    base = path.with_name(path.stem + "_base.mp4")
    src = ["-f", "lavfi", "-i", f"nullsrc=s=64x112:r=30:d={frames / 30},geq=lum='mod(N*40+X*2+Y,256)':cb=128:cr=128"]
    if audio:
        src += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={frames / 30 + 0.01}"]
    _run("ffmpeg", "-v", "error", "-y", *src, "-c:v", "libx264", "-pix_fmt", "yuv420p", *(["-c:a", "aac", "-shortest"] if audio else []),
         "-video_track_timescale", "30000", base)
    base.replace(path)
    return path


def test_headshots_come_from_the_flag_then_the_environment_then_the_recap_default(tmp_path, monkeypatch):
    monkeypatch.delenv(fs.HEADSHOTS_ENV, raising=False)
    assert fs.headshots() == fs.HEADSHOTS and fs.headshots("") == fs.HEADSHOTS
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(fs.HEADSHOTS_ENV, "lead=in/a.jpg, follow=in/b.png")
    got = fs.headshots()
    assert got == {"lead": (tmp_path / "in" / "a.jpg").resolve(), "follow": (tmp_path / "in" / "b.png").resolve()}
    assert list(got) == list(fs.ROLES) and all(p.is_absolute() for p in got.values())   # resolved before FF chdirs away
    flag = fs.headshots("follow=C:/x/f.jpg,lead=C:/x/l.jpg")                           # the flag wins; order free; drive letters ok
    assert flag["lead"].name == "l.jpg" and flag["follow"].name == "f.jpg"
    for bad in ("lead=a.jpg", "lead=a.jpg,partner=b.jpg", "lead=a.jpg,lead=b.jpg,follow=c.jpg", "lead:a.jpg,follow=b.jpg", "lead=,follow=b"):
        with pytest.raises(SystemExit, match="headshots"):
            fs.parse_headshots(bad)


def test_swap_refuses_a_missing_headshot_before_touching_the_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "ffprobe", lambda path: {"width": 64, "height": 112, "fps": 30.0, "frames": 3, "v_start": 0.0, "fmt_start": 0.0,
                                                     "duration": 0.1, "has_audio": False})
    monkeypatch.setattr(fs, "GpuLock", None)          # never reached
    with pytest.raises(SystemExit, match="headshot"):
        fs.main(["swap", "--film", str(tmp_path / "f.mp4"), "--out", str(tmp_path / "o.mp4"), "--sidecar", str(tmp_path / "s.json"),
                 "--headshots", f"lead={tmp_path / 'no_l.jpg'},follow={tmp_path / 'no_f.jpg'}"])


def test_progress_lines_feed_the_runners_frame_counter():
    from dancecap import progress
    line = fs.progress_line("analysis", 600, 11032, __import__("time").time() - 75)
    m = progress.R["tqdm"].search(line)
    assert m and (m.group(1), m.group(2)) == ("600", "11032") and line.startswith("[faceswap] analysis 600/11032 [01:15<")
    assert fs._clock(3725) == "1:02:05" and fs._clock(65) == "01:05"


def _job(tmp_path, start=None, end=None, frames=11032):
    film = tmp_path / "in" / "rec_video.mp4"
    film.parent.mkdir(parents=True, exist_ok=True)
    for p in ("rec_video.mp4", "rec.faces.json", "kadu.jpg", "larissa.JPG", "facefusion_57c01ed.tar.gz"):
        (tmp_path / "in" / p).write_bytes(b"x")
    heads = {"lead": tmp_path / "in" / "kadu.jpg", "follow": tmp_path / "in" / "larissa.JPG"}
    return fs.build_pod_job(film_video=film, sidecar=tmp_path / "in" / "rec.faces.json", stem="06_Rec vertical v4.2", take="larissa_kadu_recap",
                            heads=heads, archive=tmp_path / "in" / "facefusion_57c01ed.tar.gz", frames=frames, start=start, end=end,
                            name="faceswap_vertical" + fs.range_tag(start, end), ff_models="models-3.0.0/inswapper_128=7057c6ea",
                            face_points=tmp_path / "in" / "kadu.jpg")


def test_the_pod_job_is_a_valid_runner_job_that_ships_only_the_picture(tmp_path):
    from dancecap import remote_jobs as rj
    d = _job(tmp_path, 60, 120, frames=1800)
    assert "@@" not in __import__("json").dumps(d)
    job = rj.parse_job(d, "test")
    assert job.name == "faceswap_vertical_60-120" and job.take == "larissa_kadu_recap" and not job.base and job.cpu is False
    assert job.needs == ["ffmpeg"] and rj.job_features(job) == ["ffmpeg"] and rj.needed_features([job]) == ["ffmpeg"]   # no torch stack
    assert rj.job_models(job) == "none" and "mamma-turing" in job.exclude
    assert [c.name for c in job.commands] == ["install", "swap", "verify"]
    assert [c.name for _, c in rj.early_commands([job])] == ["install"]          # the venv / models start next to the upload
    stage = "dancecap/work/larissa_kadu_recap/faceswap_pod/06_Rec_vertical_v4.2"   # a name the shell needs no quotes for
    pods = [p for _, p in job.inputs]
    assert pods[:5] == [f"{stage}/in/06_Rec_vertical_v4.2_video.mp4", f"{stage}/in/06_Rec_vertical_v4.2.faces.json",
                        f"{stage}/in/headshot_lead.jpg", f"{stage}/in/headshot_follow.jpg", f"{stage}/in/facefusion_57c01ed.tar.gz"]
    assert pods[5] == "dancecap/" + fs.FACE_POINTS.relative_to(fs.ROOT).as_posix()       # where the pod's film_faceswap reads it
    assert not any(p.lower().endswith(fs.AUDIO_EXT) for p in pods)
    # the commands run from dancecap/: their paths and the headshot override name the shipped files
    wdir = stage[len("dancecap/"):]
    assert job.cwd == "dancecap" and job.env["HM_FACESWAP_HEADSHOTS"] == f"lead={wdir}/in/headshot_lead.jpg,follow={wdir}/in/headshot_follow.jpg"
    assert job.env["FACEFUSION_COMMIT"] == fs.FF_COMMIT and job.env["FACEFUSION_ARCHIVE"] == f"{wdir}/in/facefusion_57c01ed.tar.gz"
    swap = job.commands[1].cmd
    assert swap.startswith("bash remote/faceswap/run.sh swap --film " + f"{wdir}/in/06_Rec_vertical_v4.2_video.mp4 --sidecar ")
    assert swap.endswith(f"--out {wdir}/out/06_Rec_vertical_v4.2_faces_video_60-120.mp4 --start 60 --end 120")
    assert job.commands[2].cmd.endswith(f"--report {wdir}/out/06_Rec_vertical_v4.2_faces_video_60-120.report.json")
    assert job.outputs == [f"{stage}/out/"] and job.verify[0].endswith("_faces_video_60-120.mp4")
    # the time limits follow the frame count (a kill switch at twice the measured pod rate + a fixed part)
    whole = rj.parse_job(_job(tmp_path, frames=11032), "test")
    t = {c.name: c.timeout_min for c in job.commands}
    tw = {c.name: c.timeout_min for c in whole.commands}
    per = fs.POD_S_PER_FRAME["swap"]
    assert t["install"] == tw["install"] == fs.POD_INSTALL_MIN and tw["swap"] > 11032 * per / 60 > t["swap"] > 1800 * per / 60
    assert tw["swap"] > 2 * 11032 * 0.075 / 60 and tw["verify"] > 2 * 11032 * 0.009 / 60   # > 2x the L40's measured 2026-10-23 rates
    assert job.max_minutes == sum(t.values()) and whole.max_minutes == sum(tw.values()) and "--start" not in whole.commands[1].cmd
    assert whole.name == "faceswap_vertical" and whole.commands[1].cmd.endswith("_faces_video.mp4")
    # the shipped scripts exist and are the ones the commands name
    for s in ("install.sh", "run.sh", "pod_job_template.json"):
        assert (fs.ROOT / "remote" / "faceswap" / s).is_file()


def test_the_models_env_carries_the_local_copies_crc32(tmp_path):
    m = tmp_path / ".assets" / "models"
    m.mkdir(parents=True)
    (m / "inswapper_128.hash").write_text("7057c6ea")
    (m / "xseg_1.hash").write_text("not a hash")
    env = fs.ff_models_env(tmp_path).split()
    assert "models-3.0.0/inswapper_128=7057c6ea" in env and "models-3.1.0/xseg_1" in env and "models-3.3.0/nsfw_3" in env
    assert len(env) == sum(len(n) for _, n in fs.FF_MODELS)
    assert fs.range_tag(None, None) == "" and fs.range_tag(60, 120) == "_60-120" and fs.range_tag(None, 12.5) == "_0-12.5"


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg / ffprobe not on PATH")
def test_only_the_picture_leaves_and_the_recordings_own_audio_comes_back(tmp_path):
    """the pod route end to end without a pod: the video-only copy (same video bits, no audio), the audio guard, a stand-in for
    the pod's swapped output, and the remux that puts the recording's own audio back (md5, frames, A/V start)"""
    rec = _recording(tmp_path / "rec.mp4")
    copy = tmp_path / "stage" / "rec_video.mp4"
    vo = fs.video_only_copy(rec, copy)
    assert vo["ok"] and vo["streams"] == ["video"] and vo["video_md5_equals_recording"] and vo["frames"] == 30 and vo["recording_has_audio"]
    assert fs.video_only_copy(rec, copy).get("reused")                         # made once per recording
    assert fs.stream_md5(copy, "v") == fs.stream_md5(rec, "v") and not fs.ffprobe(copy)["has_audio"]
    # the guard: the recording itself, any audio file - refused; the copy, json and code pass
    (tmp_path / "voice.wav").write_bytes(b"RIFF")
    (tmp_path / "notes.json").write_text("{}")
    bad = fs.audio_problems([rec, tmp_path / "voice.wav", copy, tmp_path / "notes.json", fs.ROOT / "dancecap" / "film_faceswap.py"])
    assert len(bad) == 2 and "rec.mp4" in bad[0] and "audio" in bad[0] and "voice.wav" in bad[1]
    assert fs.audio_problems([copy, tmp_path / "notes.json"]) == []
    # the pod's whole-film output: the picture re-encoded from the copy (no audio), its start restored as the swap does
    info = fs.ffprobe(copy)
    pod_out = tmp_path / "pod_out.mp4"
    w = fs.Writer(pod_out, copy, info, 0, info["frames"], info["width"], info["height"], fs.Cfg())
    for buf in fs.YuvReader(copy, info, 0, info["frames"]):
        w.write(buf)
    w.close()
    fs.restore_start(pod_out, copy)
    assert not fs.ffprobe(pod_out)["has_audio"]
    final = tmp_path / "rec_faces.mp4"
    rm = fs.remux_audio(pod_out, rec, final)
    assert rm["ok"] and rm["whole_film"] and rm["audio_md5_equals_recording"] and rm["video_md5_equals_pod"]
    assert rm["frames"] == rm["frames_pod"] == 30 and rm["av_start_frames_apart"] == rm["av_start_frames_apart_recording"]
    assert fs.stream_md5(final, "a") == fs.stream_md5(rec, "a") and fs.stream_md5(final, "v") == fs.stream_md5(pod_out, "v")
    # an excerpt (frames 10-19): the same instant of the audio, re-cut
    part = tmp_path / "pod_part.mp4"
    w = fs.Writer(part, copy, info, 10, 10, info["width"], info["height"], fs.Cfg())
    for buf in fs.YuvReader(copy, info, 10, 10):
        w.write(buf)
    w.close()
    rx = fs.remux_audio(part, rec, tmp_path / "rec_faces_part.mp4", first=10, count=10)
    assert rx["ok"] and not rx["whole_film"] and rx["frames"] == 10 and "re-cut" in rx["audio"]
    assert fs.ffprobe(tmp_path / "rec_faces_part.mp4")["has_audio"]
    # a pod output that lost a frame never becomes the final
    short = tmp_path / "short.mp4"
    _run("ffmpeg", "-v", "error", "-y", "-i", pod_out, "-frames:v", "29", "-c", "copy", short)
    with pytest.raises(RuntimeError, match="remux"):
        fs.remux_audio(short, rec, tmp_path / "bad.mp4")
    assert not (tmp_path / "bad.mp4").exists()


def test_the_pod_report_gets_local_paths_and_its_analysis(tmp_path):
    import json
    pod = tmp_path / "out"
    pod.mkdir()
    (pod / "x_faces_video.report.json").write_text(json.dumps(
        {"film": "/workspace/HeadMovement/in/x_video.mp4", "out": "/workspace/HeadMovement/out/x_faces_video.mp4", "sidecar": "/w/s.json",
         "sources": {"lead": "/w/l.jpg", "follow": "/w/f.jpg"}, "first_frame": 0, "frames": 3, "verify": {"psnr_outside_faces_db": 50.0}}))
    (pod / "x_faces_video.analysis.json").write_text('{"frames": []}')
    heads = {"lead": tmp_path / "l.jpg", "follow": tmp_path / "f.jpg"}
    rp = fs.localize_report(pod / "x_faces_video.report.json", tmp_path / "x_faces.mp4", tmp_path / "x.mp4", tmp_path / "x.faces.json",
                            heads, {"pod": {"gpu": "RTX 4090"}, "remux": {"ok": True}})
    rep = json.loads(rp.read_text())
    assert rp == tmp_path / "x_faces.report.json" and rep["film"] == str((tmp_path / "x.mp4").resolve())
    assert rep["out"] == str((tmp_path / "x_faces.mp4").resolve()) and rep["sources"]["lead"] == str(heads["lead"])
    assert rep["pod_paths"]["out"].startswith("/workspace") and rep["pod"]["gpu"] == "RTX 4090" and rep["verify"]["psnr_outside_faces_db"] == 50.0
    assert (tmp_path / "x_faces.analysis.json").read_text() == '{"frames": []}'


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg / ffprobe not on PATH")
def test_compare_two_swaps_inside_and_outside_the_faces(tmp_path):
    """compare: A and B paint the same face box (lossless) 10 levels apart; outside the box both are the decoded recording"""
    import json
    rec = _recording(tmp_path / "rec.mp4", audio=False)
    info = fs.ffprobe(rec)
    box = [8, 16, 40, 48]

    def swapped(path, level, frames_with_face):
        w = fs.Writer(path, rec, info, 0, 30, 64, 112, fs.Cfg(lossless=True))
        for k, buf in enumerate(fs.YuvReader(rec, info, 0, 30)):
            if k in frames_with_face:
                Y, _, _ = fs.planes(buf, 64, 112)
                Y[box[1]:box[3], box[0]:box[2]] = level
            w.write(buf)
        w.close()
        pf = [{"i": k, "faces": [{"role": "follow", "w": 1.0, "box": box}] if k in frames_with_face else []} for k in range(30)]
        rp = path.with_suffix(".report.json")
        rp.write_text(json.dumps({"film": str(rec), "out": str(path), "first_frame": 0, "frames": 30, "per_frame": pf}))
        return rp
    ra = swapped(tmp_path / "a.mp4", 200, set(range(5, 15)))
    rb = swapped(tmp_path / "b.mp4", 190, set(range(5, 16)))
    out_json = tmp_path / "cmp.json"
    fs.main(["compare", "--report", str(ra), "--ref-report", str(rb), "--json", str(out_json), "--sheet", str(tmp_path / "s.jpg"),
             "--sheet-n", "3"])
    res = json.loads(out_json.read_text())
    assert res["face_frames"]["a"]["follow"] == 10 and res["face_frames"]["b"]["follow"] == 11 and res["face_frames"]["both"]["follow"] == 10
    assert res["face_frames"]["a"]["lead"] == 0 and res["weight_abs_diff"]["follow"]["max"] == 1.0
    assert res["px"]["in_a_vs_b"] == 11 * 32 * 32 and res["px"]["in_follow_a_vs_b"] == 11 * 32 * 32
    # 10 frames exactly 10 levels apart (28.13 dB alone) + 1 frame where only B painted: lower; outside: the same lossless pixels
    assert 0 < res["psnr_db"]["in_a_vs_b"] < 28.13 and res["psnr_db"]["out_a_vs_b"] == float("inf")
    assert res["psnr_db"]["out_a_vs_rec"] == res["psnr_db"]["out_b_vs_rec"]
    assert (tmp_path / "s.jpg").is_file() and cv2.imread(str(tmp_path / "s.jpg")).shape[0] == 3 * (224 + 4)
    # a window of A's frames (film 0.4-0.6 s = frames 12-17): B's extra face frame 15 is in it, the pixels line up
    fs.main(["compare", "--report", str(ra), "--ref-report", str(rb), "--json", str(out_json), "--start", "0.4", "--end", "0.6"])
    win = json.loads(out_json.read_text())
    assert win["frames"] == 6 and win["face_frames"]["a"]["follow"] == 3 and win["face_frames"]["b"]["follow"] == 4
    assert win["psnr_db"]["out_a_vs_b"] == float("inf") and win["px"]["in_a_vs_b"] == 4 * 32 * 32
