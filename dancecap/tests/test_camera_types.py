"""dancecap.camera_types + cameras/typed.py: the per-video camera type (mount handheld | tripod, zoom yes | no).

    ..\\.venv-nopo4d\\Scripts\\python -m pytest tests/test_camera_types.py -q

Covers: the TOML contract (owned [cameras."NN"] blocks: round trip on copies of both real take TOMLs, every byte
outside the blocks kept, CRLF kept, stale etag refused, keys we do not own re-emitted, backups never overwritten), the
CameraType states (unset / suggested / confirmed / conflict; the user always wins; a conflict needs newer, confident,
JUDGEABLE probe-B evidence; a weak 'tripod' pre-fills handheld; a type the stage refused shows as model_check), the
stage policies (user / auto / off; old takes = handheld + zoom), probe B rules and metrics (an attached camera's focal
is fixed by the model: no zoom verdict), probe A motion stats on synthetic frames (rotation-only vs parallax), the
type-constrained camera models on synthetic cameras (tripod -> one pose, no zoom -> one focal, tripod + zoom -> fixed
centre, a moving "tripod" is refused, the final per-frame solve uses every correspondence and a shakier track is
refused), the keyframes-only localisation (pair-noise drift check agrees with the every-frame check; a keyframes-only
cache is refused for any model but the constant pose), stage_smooth's default path unchanged, and the narration tool's
camera-type API (CSRF: cross-site / non-JSON writes refused). Nothing here writes to the real takes/ or work/.
"""
from __future__ import annotations

import json
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import camera_types as ct  # noqa: E402
from dancecap.cameras import typed  # noqa: E402

PKG = Path(__file__).resolve().parents[1]


@pytest.fixture()
def root(tmp_path):
    """A private package root (takes/ + work/) with copies of the real take TOMLs; restored afterwards."""
    (tmp_path / "takes").mkdir()
    (tmp_path / "work").mkdir()
    for p in (PKG / "takes").glob("*.toml"):
        shutil.copyfile(p, tmp_path / "takes" / p.name)
    ct.set_root(tmp_path)
    yield tmp_path
    ct.set_root(None)


def _probe_file(root: Path, take: str, which: str, cams: dict) -> None:
    name = {"a": "camera_types_suggested.json", "b": "camera_types_probe_b.json"}[which]
    p = root / "work" / take / "probe" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"version": 1, "cameras": cams}))


# ------------------------------------------------------------------------------------------------ TOML contract

@pytest.mark.parametrize("take", ["larissa_kadu", "larissa_kadu_38"])
def test_toml_round_trip_keeps_every_other_byte(root, take):
    p = root / "takes" / f"{take}.toml"
    before = p.read_bytes()
    cfg0 = tomllib.loads(before.decode("utf-8"))
    _, etag = ct.load(take, ["01"])
    t, etag2 = ct.set_type(take, "05", "tripod", False, etag)
    assert t["state"] == "confirmed" and t["effective"]["class"] == "tripod" and etag2 != etag
    ct.write_suggestion(take, "01", {"mount": "handheld", "zoom": False, "confidence": 0.93, "by": "probe_b",
                                     "evidence": {"centre_p95_m": 0.112, "focal_ratio": 1.0}})
    ct.set_type(take, "05", "handheld", True)  # rewrite an existing block in place
    after = p.read_bytes()
    assert after.startswith(before), "bytes before the camera blocks changed"
    cfg1 = tomllib.loads(after.decode("utf-8"))
    for k in cfg0:  # every table outside the camera blocks is identical, reserved [cameras] keys too
        if k == "cameras":
            assert {a: b for a, b in cfg1["cameras"].items() if not a.isdigit()} == cfg0["cameras"]
        else:
            assert cfg1[k] == cfg0[k]
    tabs = ct.cam_tables(cfg1)
    assert tabs["05"]["mount"] == "handheld" and tabs["05"]["zoom"] is True and tabs["05"]["source"] == "user"
    assert tabs["01"]["source"] == "auto" and tabs["01"]["suggested"]["confidence"] == 0.93
    assert after.count(b'[cameras."05"]') == 1
    backups = list((root / "work" / take / "monitor" / "toml_backups").glob("*.toml"))
    assert backups, "no backup of the previous file"


def test_stale_etag_and_bad_values_refused(root):
    _, etag = ct.load("larissa_kadu", ["01"])
    ct.set_type("larissa_kadu", "02", "tripod", True, etag)
    with pytest.raises(ct.StaleEtag) as e:
        ct.set_type("larissa_kadu", "03", "tripod", True, etag)
    assert e.value.etag.startswith("sha1:")
    for bad in (("1", "tripod", True), ("01", "dolly", True), ("01", "tripod", "yes")):
        with pytest.raises(ct.CameraTypeError):
            ct.set_type("larissa_kadu", *bad)
    with pytest.raises(ct.CameraTypeError):
        ct.load("../etc")


def test_owned_block_keeps_foreign_keys_crlf_and_following_comments(root):
    text = ("name = \"x\"\r\nsource_dir = \".\"\r\n\r\n[cameras]\r\nloc = { kf_by_time = true }\r\n\r\n"
            "[cameras.\"02\"]\r\nfile = \"02.mp4\"\r\nmount = \"tripod\"\r\n# my note inside the block\r\n\r\n"
            "# preamble of the next table\r\n[dance]\r\nlevel = \"pro\"\r\n")
    p = root / "takes" / "x.toml"
    p.write_bytes(text.encode())
    ct.set_type("x", "02", "handheld", False)
    out = p.read_bytes().decode()
    assert "\r\n" in out and "\n" not in out.replace("\r\n", "")
    cfg = tomllib.loads(out)
    assert cfg["cameras"]["02"]["file"] == "02.mp4" and cfg["cameras"]["02"]["mount"] == "handheld"
    assert cfg["cameras"]["loc"] == {"kf_by_time": True} and cfg["dance"] == {"level": "pro"}
    assert "# preamble of the next table\r\n[dance]" in out
    assert out.index('[cameras."02"]') < out.index("[dance]")


def test_header_parsing():
    assert ct._block_cam('[cameras."05"]') == "05"
    assert ct._block_cam("[cameras.'05']   # c") == "05"
    assert ct._block_cam("[ cameras . 05 ]") == "05"
    assert ct._block_cam("[cameras]") is None and ct._block_cam('[cameras."05".extra]') is None
    assert ct._header_parts("  [35.6, 36.4, 0.64],") is None  # an array continuation line is not a header


# ------------------------------------------------------------------------------------------------ states, policies

def test_old_takes_are_unset_and_general(root):
    types, _ = ct.load("larissa_kadu_38", ["01", "09"])
    assert all(t["state"] == "unset" and t["effective"]["class"] == "handheld_zoom" for t in types)
    for pol in ("user", "auto", "off"):
        assert all(s.general for s in ct.stage_types("larissa_kadu_38", ["01", "09"], pol).values())


def test_real_takes_load_unchanged():
    """The real take TOMLs (read-only) still load with Take.load, and every camera reads as today's model."""
    from dancecap.take import Take
    for name in ("larissa_kadu", "larissa_kadu_38"):
        tk = Take.load(name)
        assert tk.sources and tk.window
        assert all(st.cls == "handheld_zoom" for st in tk.camera_types().values())


def test_states_suggested_confirmed_conflict(root):
    take = "larissa_kadu"
    _probe_file(root, take, "a", {"05": {"mount": "tripod", "zoom": False, "confidence": 0.64, "by": "probe_a",
                                         "at": "2026-10-07T10:00:00Z", "evidence": {"parallax_px": 0.3}}})
    _probe_file(root, take, "b", {"05": {"mount": None, "zoom": False, "confidence": 0.98, "zoom_confidence": 0.98,
                                         "by": "probe_b", "at": "2026-10-07T11:00:00Z",
                                         "evidence": {"focal_ratio": 1.0, "mount_judgeable": False}},
                                  "07": {"mount": "handheld", "zoom": True, "confidence": 0.9, "mount_confidence": 0.95,
                                         "zoom_confidence": 0.9, "by": "probe_b", "at": "2099-01-01T00:00:00Z",
                                         "evidence": {"centre_p95_m": 0.178, "focal_ratio": 1.12}}})
    t5 = ct.get(take, "05")
    # mount from A (0.64 'tripod': below the bar, so the handheld prior pre-fills; the lean stays as evidence), zoom B
    assert t5["state"] == "suggested" and t5["mount"] == "handheld" and t5["zoom"] is False
    assert t5["suggestion"]["evidence"]["mount_lean"] == "tripod"
    assert t5["effective"]["class"] == "handheld_zoom"  # a suggestion is not applied under the default policy
    assert ct.needs_user(t5)  # 0.64 < 0.7
    assert ct.stage_types(take, ["05"], "auto")["05"].general  # not confident enough for 'auto' either
    # the user answers 07 = tripod, no zoom; the camera track (probe B, measured later) says handheld + zoom
    ct.set_type(take, "07", "tripod", False)
    t7 = ct.get(take, "07")
    assert t7["state"] == "conflict" and t7["conflict"]["field"] == "mount"
    assert t7["effective"]["class"] == "tripod"  # the user wins: the conflict never changes the value
    assert ct.needs_user(t7)
    # evidence older than the answer is acknowledged: no conflict
    _probe_file(root, take, "b", {"07": {"mount": "handheld", "zoom": True, "confidence": 0.9, "by": "probe_b",
                                         "at": "2020-01-01T00:00:00Z", "evidence": {}}})
    assert ct.get(take, "07")["state"] == "confirmed"
    # accept: confident suggestions become answers
    _probe_file(root, take, "b", {"02": {"mount": "handheld", "zoom": False, "confidence": 0.95, "by": "probe_b",
                                         "at": "2020-01-01T00:00:00Z", "evidence": {}}})
    types, _ = ct.accept(take, ["02", "05"], 0.7)
    by = {t["cam"]: t for t in types}
    assert by["02"]["state"] == "confirmed" and by["02"]["effective"]["class"] == "handheld"
    assert by["05"]["state"] == "suggested"
    assert ct.stage_types(take, ["02"], "user")["02"].cls == "handheld"
    assert ct.stage_types(take, ["02"], "off")["02"].general
    ct.clear(take, "02")
    assert ct.get(take, "02")["state"] == "suggested"


def test_policy_env_and_errors(monkeypatch):
    monkeypatch.setenv(ct.POLICY_ENV, "auto")
    assert ct.resolve_policy() == "auto"
    assert ct.resolve_policy("off") == "off"
    monkeypatch.setenv(ct.POLICY_ENV, "bogus")
    with pytest.raises(ct.CameraTypeError):
        ct.resolve_policy()


# ------------------------------------------------------------------------------------------------ probe B

def test_probe_b_rules():
    still = {"frames": 300, "centre_p95_m": 0.004, "axis_p95_deg": 0.1, "angvel_p50_dps": 0.05, "hf_rot_deg": 0.01,
             "focal_ratio": 1.001, "focal_step_pct": 0.1}
    s = ct.suggest_from_metrics(still)
    assert s["mount"] == "tripod" and s["zoom"] is False and s["pans"] is False and s["confidence"] >= 0.7
    pans = dict(still, axis_p95_deg=5.0)
    assert ct.suggest_from_metrics(pans)["pans"] is True
    hand = {"frames": 300, "centre_p95_m": 0.112, "axis_p95_deg": 11.3, "angvel_p50_dps": 1.5, "hf_rot_deg": 0.13,
            "focal_ratio": 1.003, "focal_step_pct": 0.2}
    s = ct.suggest_from_metrics(hand)
    assert s["mount"] == "handheld" and s["zoom"] is False and s["confidence"] >= 0.9
    zoom = dict(hand, focal_ratio=2.16, focal_step_pct=44.0)
    assert ct.suggest_from_metrics(zoom)["zoom"] is True
    between = dict(hand, centre_p95_m=0.034, hf_rot_deg=0.02, angvel_p50_dps=0.2)
    b = ct.suggest_from_metrics(between)
    assert b["mount_confidence"] < 0.7  # the user decides; the pre-fill is the handheld prior, the lean is evidence
    assert b["mount"] == "handheld" and b["evidence"]["mount_lean"] == "tripod" and b["pans"] is None
    att = ct.suggest_from_metrics(still, mount_judgeable=False)
    assert att["mount"] is None and att["zoom"] is False
    att = ct.suggest_from_metrics(still, mount_judgeable=False, zoom_judgeable=False)
    assert att is None  # nothing the track can judge


def _track(n=300, f=None, attached=False, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 30.0
    R = np.repeat(np.eye(3)[None], n, 0)
    C = np.stack([0.1 * np.sin(t), 0.02 * t, np.zeros_like(t)], 1) + rng.normal(0, 1e-3, (n, 3))
    f = np.full(n, 1000.0) if f is None else f
    return {"t": t, "R": R, "C": C, "f": f, "f_raw": f, "ok": np.ones(n, bool), "ok_raw": np.ones(n, bool),
            "attached": attached, "maxdim": 1920}


def test_probe_b_attached_focal_is_not_zoom_evidence(root):
    """An attached camera's focal is its sub-model's, fixed by the attach model (raw focal range exactly 0): probe B
    must not say 'no zoom' from it (larissa_kadu 05 zooms x1.375 at 1-2 s), never a conflict; probe A's zoom stands."""
    s = ct.suggest_from_track(_track(attached=True), 1.0)
    assert s["zoom"] is None and s["zoom_confidence"] is None and s["evidence"]["zoom_judgeable"] is False
    s2 = ct.suggest_from_track(_track(attached=False), 1.0)  # a constant raw focal is fixed by construction too
    assert s2["zoom"] is None and s2["mount"] == "handheld"
    f = 1000.0 * np.exp(0.003 * np.random.default_rng(1).standard_normal(300))  # a real per-frame focal: judged
    s3 = ct.suggest_from_track(_track(f=f), 1.0)
    assert s3["zoom"] is False and s3["evidence"]["zoom_judgeable"] is True
    take = "larissa_kadu"
    _probe_file(root, take, "a", {"05": {"mount": "handheld", "zoom": True, "confidence": 0.8, "mount_confidence": 0.85,
                                         "zoom_confidence": 0.8, "by": "probe_a", "at": "2026-10-07T10:00:00Z",
                                         "evidence": {"parallax_px": 2.0}}})
    _probe_file(root, take, "b", {"05": dict(s, at="2099-01-01T00:00:00Z")})
    t5 = ct.get(take, "05")
    assert t5["state"] == "suggested" and t5["zoom"] is True and t5["suggestion"]["zoom_confidence"] == 0.8
    ct.set_type(take, "05", "handheld", True)  # the user says zoom: the (newer) attach-fixed focal is no evidence
    assert ct.get(take, "05")["state"] == "confirmed"


def test_weak_tripod_prefills_handheld():
    a = {"mount": "handheld", "zoom": False, "confidence": 0.5, "mount_confidence": 0.5, "zoom_confidence": 0.8,
         "by": "probe_a", "at": "2026-10-07T10:00:00Z", "evidence": {"parallax_px": 0.9}}
    b = {"mount": "tripod", "zoom": False, "confidence": 0.55, "mount_confidence": 0.55, "zoom_confidence": 0.95,
         "by": "probe_b", "at": "2026-10-07T11:00:00Z", "evidence": {"centre_p95_m": 0.034}}
    m = ct.merge_suggestions(a, b)
    assert m["mount"] == "handheld" and m["evidence"]["mount_lean"] == "tripod" and m["pans"] is None
    t = ct.describe("07", None, a, b)
    assert t["mount"] == "handheld" and ct.needs_user(t)  # below the bar: the user decides, from the handheld prior
    strong = dict(b, confidence=0.9, mount_confidence=0.9)
    assert ct.merge_suggestions(a, strong)["mount"] == "tripod"  # a confident tripod stays a tripod


def test_conflict_message_wording(root):
    take = "larissa_kadu"
    _probe_file(root, take, "b", {"09": {"mount": "handheld", "zoom": True, "confidence": 0.95, "mount_confidence": 0.95,
                                         "zoom_confidence": 0.95, "by": "probe_b", "at": "2099-01-01T00:00:00Z",
                                         "evidence": {"centre_p95_m": 0.2, "focal_ratio": 2.16}}})
    ct.set_type(take, "09", "handheld", False)
    c = ct.get(take, "09")["conflict"]
    assert c["field"] == "zoom" and "you said: no zoom during the video" in c["message"]
    assert "zoomed during the video" in c["message"] and "zoom = zoom" not in c["message"]


def test_model_check_reports_a_refused_type(root):
    """report.json cams.<cam>.camera_type (what the cameras stage applied) comes back as model_check: a declared type
    the stage refused is visible next to the answer (summary 'refused'), until the answer changes."""
    take = "larissa_kadu"
    ct.set_type(take, "07", "tripod", False)
    rp = root / "work" / take / "cameras" / "report.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps({"cams": {"07": {"camera_type": {
        "class": "tripod", "source": "user", "model": "fixed_focal", "at": "2026-10-07T12:00:00Z",
        "check": {"passed": False, "why": "per-frame poses needed: the camera moves (not a locked tripod)"}}},
        "01": {"camera_type": {"class": "handheld", "model": "fixed_focal", "check": {"passed": True}}}}}))
    t7 = ct.get(take, "07")
    assert t7["state"] == "confirmed" and t7["model_check"]["refused"] and "camera moves" in t7["model_check"]["why"]
    types, _ = ct.load(take, ["01", "07"])
    assert ct.summary(types)["refused"] == 1
    ct.set_type(take, "07", "handheld", True)  # the answer changed: the old refusal is not about it any more
    assert not ct.get(take, "07")["model_check"]["refused"]


def test_backups_never_overwritten(root):
    take = "larissa_kadu"
    orig = (root / "takes" / f"{take}.toml").read_bytes()
    for k in range(3):  # three saves within the same second
        ct.set_type(take, "01", "handheld", bool(k % 2))
    backups = sorted((root / "work" / take / "monitor" / "toml_backups").glob("*.toml"))
    assert len(backups) == 3 and any(b.read_bytes() == orig for b in backups)


def test_track_metrics_static_vs_moving():
    rng = np.random.default_rng(0)
    t = np.arange(300) / 30.0
    R = np.repeat(np.eye(3)[None], 300, 0)
    C = np.zeros((300, 3)) + rng.normal(0, 1e-4, (300, 3))
    f = np.full(300, 1000.0)
    m = ct.track_metrics(t, R, C, f, np.ones(300, bool), scale=1.0)
    assert m["centre_p95_m"] < 0.001 and m["focal_ratio"] < 1.001 and m["hf_rot_deg"] < 1e-3
    C2 = np.stack([0.1 * np.sin(t), 0.02 * t, np.zeros_like(t)], 1)
    ang = 0.05 * np.sin(2 * np.pi * 5 * t)  # 5 Hz tremor, ~3 deg
    R2 = np.stack([typed.so3_exp(np.array([[0, a, 0]]))[0] for a in ang])
    f2 = 1000 * np.exp(0.4 * (t > 5))  # one zoom step
    m2 = ct.track_metrics(t, R2, C2, f2, np.ones(300, bool), scale=1.0)
    assert m2["centre_p95_m"] > 0.05 and m2["hf_rot_deg"] > 0.3 and m2["focal_ratio"] > 1.4
    assert m2["focal_step_pct"] > 30


# ------------------------------------------------------------------------------------------------ probe A (frames)

def _texture(h, w, seed):
    import cv2
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w)).astype(np.uint8), (0, 0), 2.0)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def test_probe_a_window_motion_rotation_vs_parallax():
    pytest.importorskip("cv2")
    import cv2
    h, w, hz = 360, 640, 15.0
    far, near = _texture(h + 80, w + 80, 1), _texture(h + 80, w + 80, 2)
    # tripod: the whole image moves rigidly (a slow pan), no parallax, no shake
    trip = []
    for k in range(60):
        M = np.float32([[1, 0, -40 + 0.3 * k], [0, 1, -40]])
        trip.append(cv2.warpAffine(far, M, (w, h)))
    # handheld: a near layer (lower half) moves 4x more than the far one under translation, + per-frame shake
    rng = np.random.default_rng(3)
    hand = []
    for k in range(60):
        tx = 6.0 * np.sin(2 * np.pi * k / 30.0) + rng.normal(0, 0.8)
        a = cv2.warpAffine(far, np.float32([[1, 0, -40 + tx], [0, 1, -40 + rng.normal(0, 0.8)]]), (w, h))
        b = cv2.warpAffine(near, np.float32([[1, 0, -40 + 4 * tx], [0, 1, -40]]), (w, h))
        a[h // 2:] = b[h // 2:]
        hand.append(a)
    mt, mh = ct.window_motion(trip, hz), ct.window_motion(hand, hz)
    assert mt["jitter_hf"] < ct.A_TRIPOD_JITTER and (mt["parallax_px"] or 0) < ct.A_TRIPOD_PARALLAX
    assert mh["jitter_hf"] > ct.A_HANDHELD_JITTER or (mh["parallax_px"] or 0) > ct.A_HANDHELD_PARALLAX
    assert mt["zoom_log_range"] < 0.01


# ------------------------------------------------------------------------------------------------ typed models

def _look(C, target):
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross([0, -1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])  # world -> camera rows


def synth_raw(n=90, moving=False, zoom=False, rotate=False, seed=0, w=1280, h=720, noise_px=0.5, n_pts=400,
              raw_exact=False):
    """A raw-localisation dict (loc_<cam>.npz layout) for a synthetic camera looking at a room (points 4-12 m away):
    observations = the true projection + noise; the per-frame 'raw' poses = P4Pf-like noisy solutions (focal and the
    centre along the axis trade off), as today's localiser produces (raw_exact: the true poses, a perfect solver)."""
    from dancecap.cameras.depth_guard import project
    rng = np.random.default_rng(seed)
    P = np.concatenate([rng.uniform([-6, -1, 4], [6, 3, 12], (n_pts, 3))])
    t = np.arange(n) / 30.0
    cx, cy = w / 2 - 0.5, h / 2 - 0.5
    Rs, Cs, fs, xs, Xs = [], [], [], [], []
    for i in range(n):
        C = np.array([0.0, 1.5, -2.0]) + (np.array([0.06 * np.sin(t[i] * 2), 0.02 * t[i], 0.03 * t[i]]) if moving
                                         else 0.0)
        target = np.array([0.0, 1.0, 8.0]) + (np.array([1.5 * np.sin(t[i]), 0, 0]) if rotate else 0.0)
        R = _look(C, target)
        f = 1000.0 * (1.0 + (0.5 * t[i] / t[-1] if zoom else 0.0))
        x = project(P, R, C, f, cx, cy, 0.0) + rng.normal(0, noise_px, (n_pts, 2))
        inside = (x[:, 0] > 0) & (x[:, 0] < w) & (x[:, 1] > 0) & (x[:, 1] < h)
        Xs.append(P[inside]), xs.append(x[inside])
        # noisy per-frame solution: focal +-1 %, centre slides along the axis to compensate, small rotation noise
        df = 0.0 if raw_exact else rng.normal(0, 0.01)
        Cn = C if raw_exact else C + R[2] * (-df * 6.0) + rng.normal(0, 0.002, 3)
        Rn = R if raw_exact else typed.so3_exp(rng.normal(0, 2e-4, (1, 3)))[0] @ R
        Rs.append(Rn), Cs.append(Cn), fs.append(f * (1 + df))
    Rs, Cs = np.array(Rs), np.array(Cs)
    off = np.r_[0, np.cumsum([len(a) for a in Xs])]
    d = {"t": t, "ok": np.ones(n, bool), "R": Rs, "tvec": -np.einsum("nij,nj->ni", Rs, Cs), "f": np.array(fs),
         "k1": np.zeros(n), "n_inliers": np.array([len(a) for a in Xs]), "reproj_px": np.full(n, 1.0),
         "source": np.array(["loc"] * n), "corr_off": off.astype(np.int64), "corr_xy": np.concatenate(xs),
         "corr_xyz": np.concatenate(Xs)}
    return d, w, h


def test_general_class_is_todays_path():
    d, w, h = synth_raw(30)
    for st in (None, ct.StageType("01"), ct.StageType("01", "handheld", True)):
        out, rep = typed.refit(d, w, h, st)
        assert out is d and rep["model"] == "per_frame"


def test_tripod_one_constant_pose():
    d, w, h = synth_raw(90)
    out, rep = typed.refit(d, w, h, ct.StageType("01", "tripod", False, None, "user"))
    assert rep["model"] == "constant" and rep["check"]["passed"], rep
    C = -np.einsum("fji,fj->fi", out["R"], out["tvec"])
    assert np.ptp(C, 0).max() < 1e-9 and np.ptp(out["f"]) < 1e-9 and out["type_held"].all()
    assert np.linalg.norm(C[0] - [0.0, 1.5, -2.0]) < 0.01, C[0]  # the true centre, despite +-6 cm raw scatter
    assert abs(out["f"][0] / 1000.0 - 1) < 0.002
    assert rep["check"]["median_px"] < 1.0  # the noise level
    assert str(out["type_model"]) == "constant" and str(out["camera_class"]) == "tripod"


def test_handheld_no_zoom_one_focal():
    d, w, h = synth_raw(90, moving=True, rotate=True)
    out, rep = typed.refit(d, w, h, ct.StageType("01", "handheld", False, None, "user"))
    assert rep["model"] == "fixed_focal" and rep["check"]["passed"], rep
    assert np.ptp(out["f"]) < 1e-9 and abs(out["f"][0] / 1000.0 - 1) < 0.003
    C = -np.einsum("fji,fj->fi", out["R"], out["tvec"])
    Craw = -np.einsum("fji,fj->fi", d["R"], d["tvec"])
    t = d["t"]
    truth = np.array([0.0, 1.5, -2.0]) + np.stack([0.06 * np.sin(t * 2), 0.02 * t, 0.03 * t], 1)
    assert np.median(np.linalg.norm(C - truth, axis=1)) < 0.5 * np.median(np.linalg.norm(Craw - truth, axis=1))


def test_tripod_zoom_fixed_centre():
    d, w, h = synth_raw(90, zoom=True, rotate=True)
    out, rep = typed.refit(d, w, h, ct.StageType("01", "tripod", True, None, "user"))
    assert rep["model"] == "fixed_centre" and rep["check"]["passed"], rep
    C = -np.einsum("fji,fj->fi", out["R"], out["tvec"])
    assert np.ptp(C, 0).max() < 1e-9 and np.linalg.norm(C[0] - [0.0, 1.5, -2.0]) < 0.02
    truth_f = 1000.0 * (1.0 + 0.5 * d["t"] / d["t"][-1])
    assert np.median(np.abs(out["f"] / truth_f - 1)) < 0.003  # per-frame focal follows the zoom


def test_moving_camera_declared_tripod_is_refused():
    d, w, h = synth_raw(90, moving=True, rotate=True)
    out, rep = typed.refit(d, w, h, ct.StageType("01", "tripod", False, None, "user"))
    models = {x["model"]: x["passed"] for x in rep["tried"]}
    assert models["constant"] is False and models["fixed_centre"] is False
    assert rep["model"] == "fixed_focal" and rep["check"]["passed"] is False  # kept per-frame, flagged
    assert not out["type_held"].any()
    strict, rep2 = typed.refit(d, w, h, ct.StageType("01", "tripod", False, None, "user"),
                               typed.TypedSettings(strict=True))
    assert rep2["model"] == "constant"  # strict keeps the user's model regardless


def test_zooming_camera_declared_no_zoom_keeps_todays_model():
    """A wrong 'no zoom' would turn the zoom into a slide along the axis (dolly-zoom) that still fits the static
    points (larissa_kadu cam09: 1.85 px, but centre jitter 17 -> 412 mm): the focal check refuses it."""
    d, w, h = synth_raw(90, moving=True, zoom=True, rotate=True)
    for mount in ("handheld", "tripod"):
        out, rep = typed.refit(d, w, h, ct.StageType("09", mount, False, None, "user"))
        assert out is d and rep["model"] == "per_frame" and rep["check"]["passed"] is False, rep
        assert any(x.get("focal_ok") is False and "zooms" in x["why"] for x in rep["tried"]), rep["tried"]


def test_attached_camera_keeps_attach_model_unless_tripod():
    d, w, h = synth_raw(60)
    d["source"] = np.array(["attach"] * len(d["t"]))
    out, rep = typed.refit(d, w, h, ct.StageType("05", "handheld", False, None, "user"))
    assert out is d and rep["model"] == "per_frame" and "attach" in rep["why"]
    out, rep = typed.refit(d, w, h, ct.StageType("05", "tripod", False, None, "user"))
    assert rep["model"] == "constant" and out["type_held"].all()
    assert np.ptp(out["f"]) < 1e-9 and "intrinsics" not in rep  # the sub-model's focal, not refitted


def test_final_solve_uses_every_correspondence_and_refuses_a_shakier_track():
    """The final per-frame solve of a constrained model uses all of a frame's correspondences (a 300-point subset made
    larissa_kadu 07 2.2 x shakier); the jitter check is relative: a track much shakier than today's is refused."""
    d, w, h = synth_raw(90, moving=True, rotate=True, noise_px=1.0, n_pts=1500, raw_exact=True, seed=4)
    st = ct.StageType("01", "handheld", False, None, "user")
    full, rf = typed.refit(d, w, h, st)
    few, rs = typed.refit(d, w, h, st, typed.TypedSettings(max_pts=40, final_pts=40))
    assert rf["model"] == "fixed_focal" and rf["check"]["passed"], rf["check"]
    t_all = rf["tried"][-1]["hf_centre_rel"]
    t_few = rs["tried"][-1]["hf_centre_rel"]
    assert t_all < 0.7 * t_few, (t_all, t_few)  # every correspondence: a much steadier track
    assert rs["model"] == "per_frame" and rs["check"]["passed"] is False
    assert "shakier" in rs["tried"][-1]["why"]  # today's (exact) track kept: the subset solve is refused


def test_gn_solve_frames_chunks_like_one_batch():
    d, w, h = synth_raw(12, moving=True, rotate=True, seed=2)
    cam = typed._Cam(d, w, h, typed.TypedSettings())
    fr = cam.frames
    R, C, lf, k1 = cam.states(fr, f=1000.0, k1=0.0)
    roles = {"rot": "P", "cen": "P", "lf": "F", "k1": "F"}
    one = typed.gn_solve(typed.Batch.of(d, fr, 1 << 30), R, C, lf, k1, cam.cx, cam.cy, roles)
    chunked = typed.gn_solve_frames(d, fr, 0, R, C, lf, k1, cam.cx, cam.cy, roles, chunk_pts=700)
    assert np.allclose(one["C"], chunked["C"], atol=1e-4) and np.allclose(one["R"], chunked["R"], atol=1e-6)
    with pytest.raises(ValueError):
        typed.gn_solve_frames(d, fr, 0, R, C, lf, k1, cam.cx, cam.cy, {"rot": "P", "cen": "S"})


def test_keyframe_subset():
    t = np.arange(0, 10, 1 / 30)
    idx = typed.keyframe_subset(t, 2.0, [7])
    assert idx[0] == 0 and idx[-1] == len(t) - 1 and 7 in idx and len(idx) == 7
    assert typed.keyframe_subset(np.arange(3) / 30, 2.0) == [0, 1, 2]
    pairs = typed.keyframe_subset(t, 2.0, pairs=True)
    assert len(pairs) == 12 and 1 in pairs and len(t) - 2 in pairs  # each grid keyframe + its neighbour


def _keyframes_only(d, pairs=True, step=2.0):
    """A raw dict as localize_typed leaves it: keyframes (+ next frame) localised, the loc marker 'keyframes'."""
    keep = np.zeros(len(d["t"]), bool)
    keep[typed.keyframe_subset(d["t"], step, pairs=pairs)] = True
    out = {k: np.array(v, copy=True) for k, v in d.items()}
    xy, xyz, off = [], [], [0]
    for i in range(len(d["t"])):
        a, b = d["corr_off"][i], d["corr_off"][i + 1]
        if keep[i]:
            xy.append(d["corr_xy"][a:b]), xyz.append(d["corr_xyz"][a:b])
        off.append(off[-1] + (b - a if keep[i] else 0))
    out["corr_off"] = np.array(off, np.int64)
    out["corr_xy"], out["corr_xyz"] = np.concatenate(xy), np.concatenate(xyz)
    out["ok"] = d["ok"] & keep
    out.update(typed.loc_marker("keyframes", "tripod", localised=int(keep.sum())))
    return out


@pytest.mark.parametrize("seed", [0, 1])
def test_keyframe_drift_check_agrees_with_every_frame(seed):
    """The keyframes are ~2 s apart (no 0.25 s low-pass): the pair noise is taken out of their spread, so a true
    tripod passes on keyframes as on every frame (the P4Pf raw centres scatter +-6 cm along the axis), and a moving
    camera is refused on both; without pairs (older caches) the keyframe check stays stricter (safe)."""
    trip = ct.StageType("01", "tripod", False, None, "user")
    d, w, h = synth_raw(600, seed=seed)
    assert typed.refit(d, w, h, trip)[1]["model"] == "constant"
    out, rk = typed.refit(_keyframes_only(d), w, h, trip)
    assert rk["model"] == "constant" and rk["localised"] == "keyframes" and rk["centre_drift"]["pairs"] >= 3
    assert out["type_held"].all()
    assert typed.refit(_keyframes_only(d, pairs=False), w, h, trip)[1]["model"] != "constant"
    dm, w, h = synth_raw(600, seed=seed, moving=True)
    assert typed.refit(dm, w, h, trip)[1]["model"] != "constant"
    _, rm = typed.refit(_keyframes_only(dm), w, h, trip)
    # keyframes only + the constant pose refused: no other model is fitted on the keyframes (every frame is needed)
    assert rm["model"] == "per_frame" and rm["check"]["keyframes_only"] and rm["check"]["passed"] is False


# ------------------------------------------------------------------------------------------------ stage_smooth

def _work_with_raw(tmp: Path, d: dict, w: int, h: int, cam: str = "01") -> Path:
    work = tmp / "scratchwork" / "synthtake"
    (work / "cameras" / "cache").mkdir(parents=True)
    (work / "frames" / cam).mkdir(parents=True)
    np.savez(work / "cameras" / "cache" / f"loc_{cam}.npz", **d)
    (work / "frames" / cam / "frames.json").write_text(json.dumps({"width": w, "height": h, "frames": []}))
    return work


def test_keyframes_only_cache_is_refused_for_any_other_model(tmp_path):
    """A locked tripod's keyframes-only loc_<cam>.npz is valid for the constant pose only: after a type / policy
    change (or a refused constant pose) stage_smooth does not export a track holding a few % of the frames."""
    from dancecap.cameras.pipeline import KeyframeCacheError, keyframe_mismatch, localisation_record, stage_smooth
    d, w, h = synth_raw(300, seed=3)
    work = _work_with_raw(tmp_path, _keyframes_only(d), w, h)
    trip = {"01": ct.StageType("01", "tripod", False, None, "user")}
    a = stage_smooth(work, ["01"], 0.04, 4.0, write=True, types=trip)
    z = dict(np.load(work / "cameras" / "01.npz"))
    assert str(z["type_model"]) == "constant" and z["ok"].all() and not a["01"]["keyframes_only"]
    assert keyframe_mismatch(work, ["01"], trip) == {}
    hand = {"01": ct.StageType("01", "handheld", False, None, "user")}
    assert "01" in keyframe_mismatch(work, ["01"], hand) and "01" in keyframe_mismatch(work, ["01"], {})
    with pytest.raises(KeyframeCacheError) as e:
        stage_smooth(work, ["01"], 0.04, 4.0, write=True, types=hand)
    assert "01" in e.value.cams and "--stages loc --cams 01" in str(e.value)
    assert str(np.load(work / "cameras" / "01.npz")["type_model"]) == "constant"  # the export was not replaced
    b = stage_smooth(work, ["01"], 0.04, 4.0, write=False, types={})  # probe B's re-derivation: marked, not raised
    assert b["01"]["keyframes_only"]
    rec = localisation_record(work, ["01"])
    assert rec["01"]["mode"] == "keyframes" and rec["01"]["class"] == "tripod" and rec["01"]["localised"] < 60


def test_stage_smooth_default_unchanged_and_typed_keys(tmp_path):
    from dancecap.cameras.pipeline import stage_smooth
    d, w, h = synth_raw(60)
    work = _work_with_raw(tmp_path, d, w, h)
    a = stage_smooth(work, ["01"], 0.04, 4.0, write=True)  # no take TOML for a scratch work dir: no types
    za = dict(np.load(work / "cameras" / "01.npz"))
    b = stage_smooth(work, ["01"], 0.04, 4.0, write=True, types={})
    zb = dict(np.load(work / "cameras" / "01.npz"))
    assert set(za) == set(zb) and "type_model" not in za
    for k in za:
        assert np.array_equal(za[k], zb[k], equal_nan=True), k
    assert "camera_type" not in a["01"] and "camera_type" not in b["01"]
    c = stage_smooth(work, ["01"], 0.04, 4.0, write=True,
                     types={"01": ct.StageType("01", "tripod", False, None, "user")})
    zc = dict(np.load(work / "cameras" / "01.npz"))
    assert str(zc["type_model"]) == "constant" and str(zc["camera_class"]) == "tripod" and zc["type_held"].all()
    assert zc["ok"].all() and np.ptp(zc["K"][:, 0, 0]) < 1e-6
    assert c["01"]["camera_type"]["model"] == "constant"
    assert c["01"]["depth_guard"]["skipped"].startswith("camera type")


# ------------------------------------------------------------------------------------------------ narration tool API

def test_annotate_camera_type_api(root):
    pytest.importorskip("flask")
    from dancecap.moves.annotate.server import create_app
    from dancecap.moves.annotate.session import Camera, Session
    take = "larissa_kadu"
    cams = [Camera(c, root / f"{c}.mp4") for c in ("01", "05")]
    s = Session(take, "take", cams, work_root=root / "work", graph_path=root / "g.json",
                additions_path=root / "a.toml", vocabulary_path=root / "v", zouk1_path=root / "z", annotator="t")
    app = create_app(s, check_asr=False)
    app.testing = True
    c = app.test_client()
    r = c.get("/api/cameras/types")
    assert r.status_code == 200 and r.json["enabled"] and {t["cam"] for t in r.json["types"]} == {"01", "05"}
    etag = r.json["toml_etag"]
    r = c.put("/api/cameras/types/05?preview=1", json={"mount": "tripod", "zoom": False})
    assert r.status_code == 200 and r.json["rerun_needed"] is False
    r = c.put("/api/cameras/types/05", json={"mount": "tripod", "zoom": False}, headers={"If-Match": etag})
    assert r.status_code == 200 and r.json["type"]["state"] == "confirmed"
    assert r.json["type"]["effective"]["class"] == "tripod"
    r2 = c.put("/api/cameras/types/01", json={"mount": "tripod", "zoom": False}, headers={"If-Match": etag})
    assert r2.status_code == 412 and r2.json["toml_etag"] == r.json["toml_etag"]
    assert c.put("/api/cameras/types/01", json={"mount": "crane", "zoom": False}).status_code == 422
    assert c.put("/api/cameras/types/09", json={"mount": "tripod", "zoom": False}).status_code == 404
    assert c.put("/api/cameras/types/01", data="mount=tripod").status_code == 415
    assert tomllib.loads((root / "takes" / f"{take}.toml").read_text())["cameras"]["05"]["mount"] == "tripod"
    r = c.delete("/api/cameras/types/05")
    assert r.status_code == 200 and r.json["type"]["state"] == "unset"
    _probe_file(root, take, "b", {"01": {"mount": "handheld", "zoom": False, "confidence": 0.95, "by": "probe_b",
                                         "at": "2026-10-07T11:00:00Z", "evidence": {}}})
    # CSRF: a cross-site page can target 127.0.0.1 directly; non-JSON / cross-origin writes are refused
    assert c.post("/api/cameras/types/accept", data="{}", content_type="text/plain").status_code == 415
    assert c.post("/api/cameras/types/accept", json={}, headers={"Origin": "http://evil.example"}).status_code == 403
    assert c.post("/api/cameras/types/accept", json={}, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert c.put("/api/cameras/types/01", json={"mount": "tripod", "zoom": False},
                 headers={"Origin": "null"}).status_code == 403
    assert ct.get(take, "01")["state"] != "confirmed"  # none of the refused writes reached the take TOML
    r = c.post("/api/cameras/types/accept", json={}, headers={"Origin": "http://localhost"})
    assert r.status_code == 200 and {t["cam"]: t["state"] for t in r.json["types"]}["01"] == "confirmed"
    # the re-run warning compares with the class the export was made with (cameras/<cam>.npz camera_class)
    (root / "work" / take / "cameras").mkdir(parents=True, exist_ok=True)
    np.savez(root / "work" / take / "cameras" / "05.npz", t=np.zeros(1), camera_class=np.str_("tripod"))
    assert c.put("/api/cameras/types/05?preview=1", json={"mount": "tripod", "zoom": False}).json["rerun_needed"] is False
    assert c.put("/api/cameras/types/05?preview=1", json={"mount": "handheld", "zoom": True}).json["rerun_needed"] is True
    np.savez(root / "work" / take / "cameras" / "05.npz", t=np.zeros(1))  # a legacy export: the general model
    assert c.put("/api/cameras/types/05?preview=1", json={"mount": "handheld", "zoom": True}).json["rerun_needed"] is False
    # a lone video has no take: the panel is off
    v = Session("clip", "video", [Camera("video", root / "clip.mp4")], video=root / "clip.mp4",
                work_root=root / "work", graph_path=root / "g.json", additions_path=root / "a.toml",
                vocabulary_path=root / "v", zouk1_path=root / "z", annotator="t")
    cv = create_app(v, check_asr=False).test_client()
    assert cv.get("/api/cameras/types").json["enabled"] is False
    assert cv.put("/api/cameras/types/video", json={"mount": "tripod", "zoom": False}).status_code == 404


def test_corr_frames_skip_a_frame_with_a_non_positive_focal():
    import numpy as np
    from dancecap.cameras import typed as T
    d = {"corr_off": np.array([0, 10, 20, 30, 40]), "ok": np.array([True, True, True, False]), "f": np.array([1500.0, -3.0, 1510.0, 1505.0])}
    assert T._corr_frames(d).tolist() == [0, 2]


def test_zoom_test_reads_the_rolling_median_of_the_per_frame_focal():
    """lucas_luiza's no-zoom Steadicam: P4Pf's per-frame focal scatters on weak frames (p90 9.1 % frame by frame) - the
    fixed focal was refused as 'the camera zooms'. A zoom is a level that holds, even where its frames are weak."""
    from dancecap.cameras.typed import TypedSettings, _Cam
    rng = np.random.default_rng(0)
    n = 1500
    cam = _Cam.__new__(_Cam)
    cam.s, cam.frames = TypedSettings(), np.arange(n)
    t = np.arange(n) / 29.0
    weak = np.arange(n) % 5 == 0  # a fifth of the frames weak
    cam.d = {"t": t, "n_inliers": np.where(weak, 40, 300)}
    cam.f = 1731.0 * np.exp(np.where(weak, rng.normal(0, 0.12, n), rng.normal(0, 0.005, n)))
    assert np.exp(cam.focal_dev(1731.0)) - 1 < 0.02  # no zoom: passes zoom_ratio 1.05
    assert np.exp(np.percentile(np.abs(np.log(cam.f / 1731.0)), 90)) - 1 > 0.05  # frame by frame it would not
    cam.f = np.where(t < 25, 2124.0, 2440.0) * np.exp(rng.normal(0, 0.005, n))  # a real +15 % zoom
    assert np.exp(cam.focal_dev(2124.0)) - 1 > 0.05
    zoomed = t >= 25  # ... whose zoomed frames are all weak (fewer map features): still a zoom
    cam.d = {"t": t, "n_inliers": np.where(zoomed, 40, 300)}
    cam.f = np.where(zoomed, 2440.0 * np.exp(rng.normal(0, 0.08, n)), 2124.0)
    assert np.exp(cam.focal_dev(2124.0)) - 1 > 0.05
