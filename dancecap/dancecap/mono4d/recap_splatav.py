r"""LHM++ splat avatars of a take's two dancers in the clothes of that take, exported in the shared splat-avatar format
(docs/SPLAT_AVATAR_FORMAT.md) for the Unity splat layer. First written for the class recap's teachers, Kadu (lead) and
Larissa (follow) (owner 2026-10-20: "I need the larissa kadu recap to look LHM++ quality with swinging hair"); since
2026-10-23 a TAKE PROFILE (`--take`) runs the same route on any take with a multi-view world fit (owner: "let's also get
gauss splats and make them default for Carlos Aline"). No --take = the class recap, with exactly its outputs.

    python -m dancecap.mono4d.recap_splatav survey  [--take T]   # local CPU: every camera x 8th frame x dancer -> lhmpp/survey.json
    python -m dancecap.mono4d.recap_splatav frames  [--take T]   # pick SAM candidates, decode their frames -> lhmpp/sam_in/
    python -m dancecap.mono4d.recap_splatav sam-job [--take T]   # the SAM pod job json (recap: remote/cand_lhmpp/pod_job_recap_sam.json,
                                                                 #  any other take: work/<take>/lhmpp/pod_job_sam.json)
    python -m dancecap.mono4d.recap_splatav select  [--take T]   # masks back: white-patch checks, crops per dancer, sheets
    python -m dancecap.mono4d.recap_splatav pick    [--take T] [--colour auto|none|ref:<cam>|wb|wb_common]   # re-pick
    python -m dancecap.mono4d.recap_splatav jev     [--take T] [--what crops|turntables]   # Jev: an independent vet (advisory)
    python -m dancecap.mono4d.recap_splatav job     [--take T]   # the LHM++ pod job json (build + export + A2 + preview)
    python -m dancecap.mono4d.recap_splatav import  [--take T] [--session SID]   # .splatav + .json, preview videos, sheets
    (pod jobs: python -m dancecap remote run --jobs <job json> --max-cost X, from dancecap/)

Data (all git-ignored likeness data): work/<take>/lhmpp/ and head-movement/Assets/StreamingAssets/<capture>/splats/ (the
StreamingAssets capture folders are ignored by head-movement/.gitignore line `Assets/StreamingAssets/*`).

PROFILES (PROFILES below): larissa_kadu_recap (six phones, piecewise sync offsets, students from world_people.npz, the
red-hair first pass) and carlos_aline (two tripod cameras 01 tele / 02 wide; the `_world.npz` tracks get pts /
frame_index from cameras/<cam>.npz; constant sync offsets from the take's offsets.json (02 = -62 ms); no other people in
the world frame (the judges sit BEHIND the couple: no occluder gate); survey window 14.82-196.5 s (the mono span before
and camera 01's zoom-out after are excluded); camera 01's burnt-in watermark rectangle rejects a candidate whose body box
or SAM mask touches it; colour harmonisation chosen by the white-shirt test (colour_check); no hair at export: her top is
red, so the recap's red-hue rule must not run; the hair stage (recap_splat_hair --take) selects and binds her black hair).

POSE / TIME. The bodies are the capture's motion (smplx_rf2_noslip/, what Unity plays), on the fit's 30 Hz grid in
sync-corrected reference time. A camera frame with audio-clock reference time t (cameras/<cam>_world.npz "t") shows the
bodies at t + delta_cam(t), delta from camera_video_offsets.json (the recap: piecewise linear between the chunk centres,
flat outside: the same correction the Unity camera videos use) or the take's sync offsets.json (delta + rate (t - t_ref)).
Cameras: cameras/<cam>_world.npz (room world frame, Y up, metres, x_cam = R X + tvec, K in display pixels).

CROPS. survey: per camera every 8th frame (0.27 s at 30 fps) and dancer, from the fit alone: the dancer's projected sparse
SMPL-X surface (every 13th vertex) inside the image, its height in pixels, the part of it behind the partner (splatted
depth), behind a nearer student (world_people.npz, all other people, 17 keypoints, as boxes), the body yaw seen from the
camera, the body speed and the fit's quality flags (suspect or absent). frames: per dancer and camera the best candidates
over 8 yaw bins, >= 3 s apart, decoded with the take's own ingest decoder (ffmpeg showinfo pts, autorotation = display
pixels). SAM 2.1 on a pod (box + projected joint points per dancer, the partner's joints as negative points) gives both
dancers' masks. select: cand_lhmpp's white-patch test (both bodies rasterised, ~1 cm cells: body not covered by its own
mask <= 1.5 %, enclosed mask holes on the body <= 0.2 %), sharpness, then a greedy over yaw (cand_lhmpp._greedy: most
frontal first, then quality x yaw distance; at most CAP per camera).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE

ROLES = ("lead", "follow")
JOB_DIR = PACKAGE_ROOT / "remote" / "cand_lhmpp"        # the pod scripts (and the recap's tracked job jsons)
LHMPP_COMMIT = "906b5d9fb967ab42efb92f6fa55bf22cac86b653"
MODEL_NAME = "LHMPP-700M-PixelShuffle"
STRIDE = 8
VSTEP = 13
ROT_KEYS = (("global_orient", 1, "root"), ("body_pose", 21, "body"), ("jaw_pose", 1, "jaw"), ("leye_pose", 1, "leye"),
            ("reye_pose", 1, "reye"), ("left_hand_pose", 15, "lhand"), ("right_hand_pose", 15, "rhand"))
GATE = {"inimg": 1.0, "occ_partner": 0.01, "occ_student": 0.005, "v_med": 0.45, "min_h_frac": 0.22}
SEL = {"n": 12, "per_cam": 3, "min_dt": 2.0, "white_frac": 0.015, "bad_hole_frac": 0.002, "outside_frac": 0.15,
       "sam_score": 0.80, "front_deg": 40.0}


# ================================================================================================ take profiles
@dataclass(frozen=True)
class Profile:
    """Everything that differs between takes. Paths derive from `take` (work/<take>/...) and `capture`."""
    take: str
    capture: str
    cams: tuple
    hires: tuple                       # cameras allowed for crop 00 (LHM++'s frontal reference)
    names: dict                        # role -> display name (provenance in the .json only)
    hires_in_order: bool = False       # True = crop 00 from the first hires camera that has a frontal crop (False: any)
    portrait_cams: tuple = ()          # stored landscape, displayed portrait (-90 tag): W, H swapped to portrait
    offsets_file: str | None = None    # None = work/<take>/camera_video_offsets.json (piecewise); else a sync offsets.json
    students: bool = True              # people/world_people.npz occluders (other people in the world frame)
    window: tuple | None = None        # survey rows only for body time inside [t0, t1]
    ignore_rects: dict = field(default_factory=dict)   # cam -> [(x0, y0, x1, y1)] screen-fixed overlays (watermarks)
    ignore_pad: float = 0.05           # the body box is grown by this share of its height before the rectangle test
    gate: dict | None = None           # survey gate of the SAM candidates (None = GATE)
    sel: dict | None = None            # crop selection rules (None = SEL)
    per_cam: int = 10                  # SAM candidates per camera and dancer
    front_extra: int = 0               # + this many FRONTAL (|yaw| <= FRONT_YAW) candidates per camera and dancer
    front_gate: dict | None = None     # ... passing this gate (None = the gate), >= FRONT_DT apart
    caps: dict = field(default_factory=dict)   # crops per camera in the final pick
    cap_default: int = 2
    min_sharp: float = 200.0
    preview_window: tuple = (345.0, 355.0)
    a2_cams: tuple = ("01", "06")
    prev_cams: tuple = ("01", "06")
    a2_span: tuple = (30.0, 330.0)
    film: str | None = None            # avatar film (head-movement/Recordings/...) for the preview's middle panel
    hair_rule: str = "red"             # "red" = recap_splatav.hair_selection on the follow at export; "none" = hair byte 0
    hair_notes: dict = field(default_factory=dict)     # role -> note when no hair is selected at export
    colour: str = "ref:01"             # crop colour harmonisation: ref:<cam> | none | wb | auto (white-shirt test)
    white_role: str | None = None      # the role whose top is white (the colour test's neutral reference)
    veto: dict = field(default_factory=dict)
    face_ok: dict = field(default_factory=dict)
    legacy_jobs: bool = False          # True = the recap's tracked job jsons under remote/cand_lhmpp/ (never for a new take)
    sam_job_name: str = ""
    lhm_job_name: str = ""
    sam_notes: str = ""
    lhm_notes: str = ""
    preview_label: str = "LHM++ splats"
    out_dir: str | None = None         # override of work/<take>/lhmpp (regression runs into a scratch folder)

    # ---------------------------------------------------------------- paths
    @property
    def work(self) -> Path:
        return PACKAGE_ROOT / "work" / self.take

    @property
    def fit_dir(self) -> Path:
        return self.work / "smplx_rf2_noslip"

    @property
    def out(self) -> Path:
        return Path(self.out_dir) if self.out_dir else self.work / "lhmpp"

    @property
    def sam_in(self) -> Path:
        return self.out / "sam_in"

    @property
    def sam_out(self) -> Path:
        return self.out / "sam_out"

    @property
    def input(self) -> Path:
        return self.out / "input"

    @property
    def pod_out(self) -> Path:
        return self.out / "pod_out"

    @property
    def preview_dir(self) -> Path:
        return self.out / "preview"

    @property
    def unity_dir(self) -> Path:
        return WORKSPACE / "head-movement" / "Assets" / "StreamingAssets" / self.capture / "splats"

    @property
    def sam_job(self) -> Path:
        return JOB_DIR / "pod_job_recap_sam.json" if self.legacy_jobs else self.out / "pod_job_sam.json"

    @property
    def lhm_job(self) -> Path:
        return JOB_DIR / "pod_job_recap_lhmpp.json" if self.legacy_jobs else self.out / "pod_job_lhmpp.json"

    @property
    def film_path(self) -> Path | None:
        return WORKSPACE / self.film if self.film else None

    def cli(self, cmd: str) -> str:
        """The local command line of a step (the jobs' after_download hints)."""
        return f"python -m dancecap.mono4d.recap_splatav {cmd}" + ("" if self.legacy_jobs else f" --take {self.take}")


RECAP = Profile(
    take="larissa_kadu_recap", capture="06_LarissaKaduRecap", cams=("00", "01", "02", "03", "05", "06"), hires=("01", "06"),
    names={"lead": "Kadu", "follow": "Larissa"}, portrait_cams=("05", "06"), offsets_file=None, students=True, window=None,
    per_cam=10, caps={"01": 4, "06": 4, "00": 2, "02": 2, "03": 2, "05": 2}, cap_default=2, min_sharp=200.0,
    preview_window=(345.0, 355.0), a2_cams=("01", "06"), prev_cams=("01", "06"), a2_span=(30.0, 330.0),
    film="head-movement/Recordings/06_LarissaKaduRecap_vertical_2026-10-11b.mp4", hair_rule="red",
    hair_notes={"lead": "Kadu: shaved head, no hair Gaussians (by design)"}, colour="ref:01", white_role=None,
    # by eye (2026-10-20, review of refs_<role>.jpg / masks_<role>.jpg of the first pick): crops the automatic tests passed
    veto={
        ("lead", "03_04680"): "red patch at his neck in the crop (her hair / a red shirt behind him)",
        ("lead", "05_00368"): "dark and blurry (sharpness 173), low resolution",
        ("follow", "03_05552"): "raised hand outside the mask, a detached dark blob",
        ("follow", "03_01992"): "face motion-blurred and smeared, top of the head cut",
        ("follow", "05_03520"): "dark and blurry (sharpness 109)",
        ("lead", "06_02824"): "her hand inside his mask at his hip",
    },
    face_ok={("follow", "01_03872"): "her raised hand is beside / behind her head, the face is visible (by eye)"},
    legacy_jobs=True, sam_job_name="recap_lhmpp_sam", lhm_job_name="recap_lhmpp",
    sam_notes="SAM 2.1 masks (three candidates each) of both recap teachers on the frames chosen by recap_splatav frames "
              "(phones of the class recap); likeness data goes to our own pod only (owner consent 2026-10-19)",
    lhm_notes="LHM++ (LHMPP-700M-PixelShuffle @ 906b5d9, CC BY-NC 4.0 weights) avatars of Kadu (lead) and Larissa (follow) in "
              "their class-recap clothes from 12 crops each of 5-6 recap phones, shape ref (recap fit betas[:10]); export of "
              "the canonical zero-pose Gaussians + LHM++'s own LBS weights (SPLAT_AVATAR_FORMAT.md), the A2 check (LHM++ "
              "animation vs our plain LBS) on phones 01 / 06, and a 10 s preview on phones 01 / 06; weights stay on the pod; "
              "crops go to our own pod only (owner consent 2026-10-19)",
    preview_label="LHM++ splats (recap avatars)")


def _toml_names(take: str) -> dict:
    import tomllib
    p = PACKAGE_ROOT / "takes" / f"{take}.toml"
    d = tomllib.loads(p.read_text(encoding="utf-8")).get("dancers") or {}
    return {r: str((d.get(r) or {}).get("name") or r) for r in ROLES}


def _carlos_aline() -> Profile:
    names = _toml_names("carlos_aline")
    return Profile(
        take="carlos_aline", capture="07_CarlosAline", cams=("01", "02"), hires=("01", "02"), hires_in_order=True,
        names=names, portrait_cams=(),
        # the take's own sync offsets (two cameras: fixed by the two-view joint reprojection, people/video_offset.json;
        # every chunk take carries the same file): 01 = 0, 02 = -62 ms
        offsets_file="work/carlos_aline_cag0000/sync/offsets.json",
        students=False, window=(14.82, 196.5),
        ignore_rects={"01": [(1440.0, 800.0, 1920.0, 1080.0)]},     # dancecap/room_two_cams.py RECTS: camera 01's watermark
        # a competition heat in close hold: the recap's gate (partner occlusion <= 1 %, median speed <= 0.45 m/s) keeps
        # 4-28 frames per camera and dancer (2026-10-23 survey); <= 5 % / <= 0.8 m/s keeps 42-124 with every yaw bin. SAM +
        # the white-patch test (partner in front <= 1.5 % of the body, hands excluded) and the sharpness floor decide.
        gate={**GATE, "occ_partner": 0.05, "v_med": 0.8},
        # SAM's own IoU estimate is lower on these smaller (550-850 px), softer tele / wide frames in front of white-shirted
        # judges: 26 of 36 lead crops had score < 0.80 with recall >= 0.92, nothing outside the body and no white patch
        # (2026-10-23 select). The geometric tests (recall, outside, white patch, holes), the sharpness floor, the LOOK at
        # masks_*.jpg / refs_*.jpg and the Jev vet decide; the score only drops SAM's clear failures.
        sel={**SEL, "sam_score": 0.60},
        # the yaw round robin gave 2-3 frontal frames per camera and dancer, and the first LOOK vetoed the frontal crops of
        # camera 01 (his arm and her hair over his face; her hair over her face): more frontal candidates for crop 00
        # (frontal views of a dancer in close hold mostly have the partner in front: 3-8 frontal frames per camera pass the
        # gate; partner <= 15 % / speed <= 1.2 m/s gives 12-24 distinct ones; the white-patch test then rejects the crops
        # where the partner really covers the body)
        per_cam=20, front_extra=8, front_gate={**GATE, "occ_partner": 0.15, "v_med": 1.2},
        caps={"01": 8, "02": 6}, cap_default=0, min_sharp=200.0,
        preview_window=(100.0, 110.0), a2_cams=("01", "02"), prev_cams=("01", "02"), a2_span=(25.0, 185.0),
        film="head-movement/Recordings/07_CarlosAline_vertical_2026-10-11b.mp4", hair_rule="none",
        hair_notes={"lead": f"{names['lead']}: shaved / close-cropped head, no hair Gaussians (by design)",
                    "follow": f"{names['follow']}: none at export (her top is red: the recap's red-hue rule must not run); "
                              "the hair stage (recap_splat_hair --take carlos_aline) selects and binds her black hair"},
        colour="auto", white_role="lead",
        # by eye (2026-10-23, refs_<role>.jpg / masks_<role>.jpg / raw-frame head crops of the first pick)
        veto={
            ("lead", "01_04456"): "his raised arm and her hair over his face (crop 00 must show the face)",
            ("lead", "02_03136"): "a white mask gap across his right sleeve",
            ("follow", "01_01856"): "her face hidden by her hair, head turned down",
            ("follow", "02_01760"): "hair flying sideways in a spin, face motion-blurred",
            ("follow", "02_01896"): "her hair over her face",
            ("follow", "02_04632"): "his white sleeve inside her mask across her back",
            ("follow", "01_02720"): "head thrown back out of the crop: no head",
            # second pick (with the extra frontal candidates)
            ("follow", "01_01800"): "her hair over her face (crop 00 must show the face)",
            ("follow", "02_02472"): "her hair whipped across her face and chest",
            ("follow", "02_03184"): "head thrown back, face turned up",
            ("follow", "02_03104"): "her forearm across her face, loose hair specks beside her head",
            ("follow", "02_02296"): "her hair flying out sideways in a spin",
        },
        face_ok={}, legacy_jobs=False,
        sam_job_name="carlos_aline_lhmpp_sam", lhm_job_name="carlos_aline_lhmpp",
        sam_notes=f"SAM 2.1 masks (three candidates each) of both dancers ({names['lead']} lead, {names['follow']} follow) "
                  "on the frames chosen by recap_splatav frames --take carlos_aline (cameras 01 / 02 of the heat); likeness "
                  "data goes to our own pod only (owner consent 2026-10-19, open collaboration)",
        lhm_notes=f"LHM++ (LHMPP-700M-PixelShuffle @ 906b5d9, CC BY-NC 4.0 weights) avatars of {names['lead']} (lead) and "
                  f"{names['follow']} (follow) in their heat clothes from 12 crops each of cameras 01 / 02, shape ref (the "
                  "take's fit betas[:10]); export of the canonical zero-pose Gaussians + LHM++'s own LBS weights "
                  "(SPLAT_AVATAR_FORMAT.md), the A2 check (LHM++ animation vs our plain LBS) on cameras 01 / 02, and a 10 s "
                  "preview on cameras 01 / 02; weights stay on the pod; crops go to our own pod only (owner consent 2026-10-19)",
        preview_label="LHM++ splats (take avatars)")


PROFILES = {"larissa_kadu_recap": lambda: RECAP, "recap": lambda: RECAP, "carlos_aline": _carlos_aline}
P: Profile = RECAP          # the active profile (use())


def profile(take: str | None = None, **over) -> Profile:
    """The profile of a take (None = the class recap), optionally with fields replaced (e.g. out_dir)."""
    key = take or "larissa_kadu_recap"
    if key not in PROFILES:
        raise SystemExit(f"recap_splatav: no take profile {key!r} (known: {', '.join(sorted(PROFILES))})")
    p = PROFILES[key]()
    return dataclasses.replace(p, **over) if over else p


def use(take: str | None = None, **over) -> Profile:
    """Make a take's profile the active one for every step of this module."""
    global P
    P = profile(take, **over)
    return P


def log(msg: str) -> None:
    print(f"[recap_splatav] {msg}", flush=True)


def _ws(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return Path(p).as_posix()


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ================================================================================================ fit, cameras, time
def load_fit(role: str) -> dict:
    with np.load(P.fit_dir / f"{role}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def merge_cam_rows(world: dict, plain: dict, cam: str = "?") -> dict:
    """A `_world.npz` track without per-frame `pts` / `frame_index` (two-camera takes) gets them from the camera's own
    cameras/<cam>.npz: the SAME rows (count and reference times t must agree to 1 us, else ValueError)."""
    d = dict(world)
    tw, tp = np.asarray(world["t"], float), np.asarray(plain["t"], float)
    if len(tw) != len(tp) or (len(tw) and float(np.abs(tw - tp).max()) > 1e-6):
        raise ValueError(f"camera {cam}: cameras/{cam}.npz rows ({len(tp)}) do not match {cam}_world.npz ({len(tw)}) "
                         "row for row (t)")
    for k in ("pts", "frame_index"):
        if k not in d and k in plain:
            d[k] = np.asarray(plain[k])
    return d


def load_cam(cam: str) -> dict:
    cdir = P.work / "cameras"
    with np.load(cdir / f"{cam}_world.npz") as z:
        d = {k: z[k] for k in z.files}
    if "pts" not in d:
        with np.load(cdir / f"{cam}.npz") as z:
            d = merge_cam_rows(d, {k: z[k] for k in ("t", "pts", "frame_index") if k in z.files}, cam)
    meta = json.loads((cdir / "recap_cams.json").read_text())["cams"][cam]
    W, H = int(meta["width"]), int(meta["height"])
    if abs(int(meta.get("rotation", 0))) % 180 == 90:
        W, H = H, W
    if cam in P.portrait_cams:  # stored landscape, displayed portrait (-90 tag): K is in display pixels
        W, H = min(W, H), max(W, H)
    d["W"], d["H"], d["file"] = W, H, meta["file"]
    return d


def read_offsets(path: Path) -> dict:
    """A sync offsets.json (dancecap.sync layout {"offsets": {cam: {"delta", "use", "use_rate", "rate", "t_ref"}}} or
    plain {cam: seconds}) -> {cam: {"delta", "rate", "t_ref"}} (mamma_dyn.read_sync_offsets' reading: "use": false = 0,
    the rate only when "use_rate")."""
    raw = json.loads(Path(path).read_text())
    raw = raw.get("offsets", raw)
    out = {}
    for cam, v in raw.items():
        if isinstance(v, dict):
            if v.get("use", True) is False:
                out[str(cam)] = {"delta": 0.0, "rate": 0.0, "t_ref": 0.0}
                continue
            d = v.get("delta", v.get("delta_c", v.get("offset", 0.0)))
            rate = (v.get("rate") or 0.0) if v.get("use_rate", False) else 0.0
            out[str(cam)] = {"delta": float(d), "rate": float(rate), "t_ref": float(v.get("t_ref", 0.0) or 0.0)}
        elif isinstance(v, (int, float)):
            out[str(cam)] = {"delta": float(v), "rate": 0.0, "t_ref": 0.0}
    return out


def video_delta(cam: str, t: np.ndarray) -> np.ndarray:
    """Sync correction delta(t) of a camera: body time = t + delta (camera_video_offsets.json piecewise linear, or the
    take's sync offsets.json: delta + rate (t - t_ref))."""
    if P.offsets_file is None:
        off = json.loads((P.work / "camera_video_offsets.json").read_text())["offsets"][cam]
        pw = np.asarray(off["piecewise"], float)
        return np.interp(np.asarray(t, float), pw[:, 0], pw[:, 1])
    o = read_offsets(PACKAGE_ROOT / P.offsets_file).get(cam, {"delta": 0.0, "rate": 0.0, "t_ref": 0.0})
    t = np.asarray(t, float)
    return o["delta"] + o["rate"] * (t - o["t_ref"]) + 0.0 * t


def grid_pos(T: np.ndarray, tb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = np.clip(np.searchsorted(T, tb) - 1, 0, len(T) - 2)
    a = np.clip((tb - T[k]) / (T[k + 1] - T[k]), 0.0, 1.0)
    return k, a


def interp_params(d: dict, ts: np.ndarray) -> dict:
    """Fit parameters at many times: per-joint slerp of the rotations, linear transl / joints (cand_lhmpp's rule)."""
    from scipy.spatial.transform import Rotation as Rot
    T = np.asarray(d["t"], float)
    ts = np.atleast_1d(np.asarray(ts, float))
    k, a = grid_pos(T, ts)
    N = len(ts)
    out = {"t": ts, "k": k, "a": a}
    for name, n, _ in ROT_KEYS:
        A = np.asarray(d[name], float).reshape(len(T), n, 3)
        r0 = Rot.from_rotvec(A[k].reshape(-1, 3))
        r1 = Rot.from_rotvec(A[k + 1].reshape(-1, 3))
        rel = (r0.inv() * r1).as_rotvec()
        out[name] = (r0 * Rot.from_rotvec(rel * np.repeat(a, n)[:, None])).as_rotvec().reshape(N, n * 3)
    for name in ("transl", "joints"):
        X = np.asarray(d[name], float)
        w = a.reshape((-1,) + (1,) * (X.ndim - 1))
        out[name] = (1 - w) * X[k] + w * X[k + 1]
    return out


def project(P_: np.ndarray, K, R, t) -> tuple[np.ndarray, np.ndarray]:
    Xc = P_ @ np.asarray(R, float).T + np.asarray(t, float)
    z = Xc[..., 2]
    uv = (Xc[..., :2] / np.maximum(z, 1e-6)[..., None]) * np.diag(K)[:2] + np.asarray(K)[:2, 2]
    return uv, z


def body_yaw_deg(global_orient: np.ndarray, pelvis: np.ndarray, R, t) -> float:
    """Signed yaw of the camera seen from the body: 0 = the dancer faces the camera, +-180 = back to it."""
    from scipy.spatial.transform import Rotation as Rot
    fwd = Rot.from_rotvec(global_orient).apply([0.0, 0.0, 1.0])
    C = -np.asarray(R, float).T @ np.asarray(t, float)
    dv = C - pelvis
    f2, d2 = np.array([fwd[0], fwd[2]]), np.array([dv[0], dv[2]])
    return float(np.degrees(math.atan2(f2[0] * d2[1] - f2[1] * d2[0], f2 @ d2)))


def smplx_model(d: dict, batch: int):
    import smplx
    from ..avatar_texture import model_dir_for
    return smplx.create(str(model_dir_for(d)), model_type="smplx", gender="neutral", ext="npz", use_pca=False,
                        flat_hand_mean=bool(d.get("flat_hand_mean", True)),
                        num_betas=int(d.get("num_betas", len(d["betas"]))), batch_size=batch)


def pose_vertices(d: dict, p: dict, vidx: np.ndarray | None = None, batch: int = 256) -> np.ndarray:
    """(N, V', 3) posed vertices of the fit model at the parameter sets p (interp_params output)."""
    import torch
    N = len(p["t"])
    nb = int(d.get("num_betas", len(d["betas"])))
    out = []
    m = None
    for s in range(0, N, batch):
        e = min(N, s + batch)
        if m is None or e - s != m.batch_size:
            m = smplx_model(d, e - s)
        f = lambda k: torch.tensor(np.asarray(p[k][s:e], np.float32))  # noqa: E731
        with torch.no_grad():
            o = m(betas=torch.tensor(np.asarray(d["betas"], np.float32).reshape(-1)[:nb])[None].expand(e - s, -1),
                  global_orient=f("global_orient"), body_pose=f("body_pose"), jaw_pose=f("jaw_pose"),
                  leye_pose=f("leye_pose"), reye_pose=f("reye_pose"), left_hand_pose=f("left_hand_pose"),
                  right_hand_pose=f("right_hand_pose"), transl=f("transl"))
        v = o.vertices.numpy()
        out.append(v if vidx is None else v[:, vidx])
    return np.concatenate(out, 0).astype(np.float32)


def flagged(d: dict, k: int) -> bool:
    """The fit's suspect flag (or `absent`: a stitched take's dancer out of view, parked by the export) at grid frame
    k or k + 1."""
    s = bool(d["suspect"][k] or d["suspect"][k + 1])
    if "absent" in d:
        s = s or bool(d["absent"][k] or d["absent"][k + 1])
    return s


# ================================================================================================ survey
def _students(wp: dict, k: int, pel: list[np.ndarray]) -> np.ndarray:
    """(n, 17, 3) keypoints of every other person at grid frame k (hip centre > 0.6 m from both teachers' pelvis)."""
    o0, o1 = int(wp["off"][k]), int(wp["off"][k + 1])
    X = wp["X"][o0:o1].astype(float)
    if not len(X):
        return X.reshape(0, 17, 3)
    hip = np.nanmean(X[:, 11:13], axis=1)
    far = np.ones(len(X), bool)
    for p in pel:
        far &= np.hypot(hip[:, 0] - p[0], hip[:, 2] - p[2]) > 0.6
    far &= np.isfinite(hip).all(1)
    return X[far]


def survey() -> Path:
    from scipy.ndimage import minimum_filter
    t0 = time.time()
    fits = {r: load_fit(r) for r in ROLES}
    T = np.asarray(fits["lead"]["t"], float)
    wp = None
    if P.students:
        with np.load(P.work / "people" / "world_people.npz") as z:
            wp = {k: z[k] for k in ("off", "X")}
    vidx = None
    sparse, joints, speed = {}, {}, {}
    for r in ROLES:
        d = fits[r]
        nV = 10475
        vidx = np.arange(0, nV, VSTEP)
        p_all = {k: np.asarray(d[k], float) for k, _, _ in ROT_KEYS}
        p_all.update({"t": T, "transl": np.asarray(d["transl"], float)})
        sparse[r] = pose_vertices(d, p_all, vidx, batch=512)
        joints[r] = np.asarray(d["joints"], float)
        v = np.gradient(sparse[r], T, axis=0)
        sp = np.linalg.norm(v, axis=-1)
        speed[r] = np.stack([np.median(sp, 1), np.percentile(sp, 90, 1)], 1)
        log(f"{r}: sparse surface {sparse[r].shape} in {time.time() - t0:.0f} s")
    rows = []
    for cam in P.cams:
        c = load_cam(cam)
        W, H = c["W"], c["H"]
        idx = np.arange(0, len(c["t"]), STRIDE)
        idx = idx[c["ok"][idx]]
        tb = c["t"][idx] + video_delta(cam, c["t"][idx])
        sel = (tb >= T[0]) & (tb <= T[-1])
        if P.window is not None:
            sel &= (tb >= P.window[0]) & (tb <= P.window[1])
        idx, tb = idx[sel], tb[sel]
        kk, aa = grid_pos(T, tb)
        for n, i in enumerate(idx):
            k, a = int(kk[n]), float(aa[n])
            K, R, tv = c["K"][i], c["R"][i], c["tvec"][i]
            P_ = {r: (1 - a) * sparse[r][k] + a * sparse[r][k + 1] for r in ROLES}
            J = {r: (1 - a) * joints[r][k] + a * joints[r][k + 1] for r in ROLES}
            uvz = {r: project(P_[r], K, R, tv) for r in ROLES}
            zc = min(float(np.median(uvz[r][1])) for r in ROLES)
            if zc <= 0.3:
                continue
            # depth splat grid ~2 cm cells at the nearer dancer
            s = float(np.clip(zc / (float(K[0, 0]) * 0.02), 0.05, 1.0))
            Ws, Hs = int(math.ceil(W * s)), int(math.ceil(H * s))
            zmap = {}
            for r in ROLES:
                uv, z = uvz[r]
                u = np.round(uv[:, 0] * s).astype(int)
                v = np.round(uv[:, 1] * s).astype(int)
                okp = (u >= 0) & (u < Ws) & (v >= 0) & (v < Hs) & (z > 0)
                zm = np.full((Hs, Ws), np.inf, np.float32)
                np.minimum.at(zm, (v[okp], u[okp]), z[okp])
                zmap[r] = minimum_filter(zm, size=3)
            studs = _students(wp, int(round(k + a)), [J[r][0] for r in ROLES]) if wp is not None else []
            sboxes = []
            for X in studs:
                g = np.isfinite(X).all(1)
                if g.sum() < 3:
                    continue
                uv, z = project(X[g], K, R, tv)
                if np.median(z) <= 0.2:
                    continue
                x0, y0 = uv.min(0)
                x1, y1 = uv.max(0)
                w_, h_ = x1 - x0, y1 - y0
                hpad = max(0.25 * h_, 0.12 * K[0, 0] / max(np.median(z), 0.3))  # head above the nose + body width
                sboxes.append((x0 - 0.3 * w_ - 0.5 * hpad, y0 - hpad, x1 + 0.3 * w_ + 0.5 * hpad, y1 + 0.1 * h_,
                               float(np.median(z))))
            for r in ROLES:
                o = ROLES[1 - ROLES.index(r)]
                uv, z = uvz[r]
                inside = (uv[:, 0] >= 2) & (uv[:, 0] < W - 3) & (uv[:, 1] >= 2) & (uv[:, 1] < H - 3) & (z > 0)
                u = np.clip(np.round(uv[:, 0] * s).astype(int), 0, Ws - 1)
                v = np.clip(np.round(uv[:, 1] * s).astype(int), 0, Hs - 1)
                occ_p = (zmap[o][v, u] < z - 0.05) & inside
                occ_s = np.zeros(len(z), bool)
                for (x0, y0, x1, y1, zs) in sboxes:
                    if zs < np.median(z) - 0.3:
                        occ_s |= (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
                # 2D box overlap with the partner (any depth): a hand on the partner is fine, an arm across is not
                bx = [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]
                pi_ = interp_params(fits[r], np.array([tb[n]]))
                rows.append({"cam": cam, "i": int(i), "t": round(float(c["t"][i]), 4), "tb": round(float(tb[n]), 4),
                             "pts": round(float(c["pts"][i]), 6), "role": r,
                             "inimg": float(inside.mean()), "h_px": round(bx[3] - bx[1], 1), "box": [round(x, 1) for x in bx],
                             "occ_partner": round(float(occ_p.mean()), 4), "occ_student": round(float(occ_s.mean()), 4),
                             "yaw_deg": round(body_yaw_deg(pi_["global_orient"][0], pi_["joints"][0][0], R, tv), 1),
                             "v_med": round(float((1 - a) * speed[r][k, 0] + a * speed[r][k + 1, 0]), 3),
                             "v_p90": round(float((1 - a) * speed[r][k, 1] + a * speed[r][k + 1, 1]), 3),
                             "dist_m": round(float(np.median(z)), 3),
                             "suspect": flagged(fits[r], k),
                             "quality": round(float(min(fits[r]["quality"][k], fits[r]["quality"][k + 1])), 3),
                             "reproj_px": round(float(max(fits[r]["reproj_px"][k], fits[r]["reproj_px"][k + 1])), 2),
                             "W": W, "H": H})
        log(f"camera {cam}: {len(idx)} frames ({time.time() - t0:.0f} s)")
    P.out.mkdir(parents=True, exist_ok=True)
    p = P.out / "survey.json"
    p.write_text(json.dumps({"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "fit": _ws(P.fit_dir), "stride": STRIDE,
                             "vstep": VSTEP, "seconds": round(time.time() - t0, 1), "rows": rows}))
    log(f"{len(rows)} dancer-frames -> {p} ({time.time() - t0:.0f} s)")
    return p


# ================================================================================================ SAM candidates
YAW_BINS = 8
MIN_DT = 3.0
FRONT_YAW = 45.0
FRONT_DT = 2.0
POS_JOINTS = (0, 3, 6, 9, 12, 15, 16, 17, 1, 2, 4, 5)
NEG_JOINTS = (0, 3, 9, 12, 15, 4, 5, 18, 19)


def box_touches(box, rects, pad: float = 0.0) -> bool:
    """Does an image box (x0, y0, x1, y1), grown by `pad` x its height on every side, touch any rectangle?"""
    x0, y0, x1, y1 = (float(v) for v in box)
    g = pad * max(0.0, y1 - y0)
    x0, y0, x1, y1 = x0 - g, y0 - g, x1 + g, y1 + g
    return any(x1 >= rx0 and x0 <= rx1 and y1 >= ry0 and y0 <= ry1 for rx0, ry0, rx1, ry1 in rects)


def mask_touches(mask: np.ndarray, rects) -> bool:
    """Does a (H, W) mask have any pixel inside a rectangle (x0, y0, x1, y1; pixel centres x + 0.5 inside)?"""
    H, W = mask.shape
    for rx0, ry0, rx1, ry1 in rects:
        a, b = max(0, int(math.floor(rx0))), min(W, int(math.ceil(rx1)))
        c, d = max(0, int(math.floor(ry0))), min(H, int(math.ceil(ry1)))
        if a < b and c < d and mask[c:d, a:b].any():
            return True
    return False


def ignored(r: dict) -> bool:
    """The survey row's projected body box touches one of its camera's ignore rectangles (a burnt-in watermark)."""
    rects = P.ignore_rects.get(r["cam"], [])
    return bool(rects) and box_touches(r["box"], rects, P.ignore_pad)


def _gate(r: dict, G: dict | None = None) -> bool:
    G = G or P.gate or GATE
    return (r["inimg"] >= G["inimg"] and r["occ_partner"] <= G["occ_partner"]
            and r["occ_student"] <= G["occ_student"] and r["v_med"] <= G["v_med"] and not r["suspect"]
            and r["h_px"] >= G["min_h_frac"] * r["H"] and not ignored(r))


def _q(r: dict, h_med: float) -> float:
    return min(1.0, r["h_px"] / h_med) ** 0.5 * math.exp(-r["v_med"] / 0.25) * max(0.2, r["quality"])


def pick_sam_candidates(rows: list[dict]) -> list[dict]:
    out = []
    for role in ROLES:
        for cam in P.cams:
            rr = [r for r in rows if r["role"] == role and r["cam"] == cam and _gate(r)]
            if not rr:
                continue
            h_med = float(np.median([r["h_px"] for r in rr]))
            for r in rr:
                r["q"] = _q(r, h_med)
            bins = {}
            for r in rr:
                b = int(((r["yaw_deg"] + 180.0 + 180.0 / YAW_BINS) % 360.0) // (360.0 / YAW_BINS))
                bins.setdefault(b, []).append(r)
            picked = []
            # round robin over the yaw bins, best first in each, MIN_DT apart on this camera
            order = {b: sorted(v, key=lambda r: -r["q"]) for b, v in bins.items()}
            while len(picked) < P.per_cam and any(order.values()):
                for b in sorted(order, key=lambda b: -(order[b][0]["q"] if order[b] else -1)):
                    while order[b]:
                        r = order[b].pop(0)
                        if all(abs(r["t"] - p["t"]) >= MIN_DT for p in picked):
                            picked.append(r)
                            break
                    if len(picked) >= P.per_cam:
                        break
            n_rr = len(picked)
            if P.front_extra:
                # extra frontal candidates (crop 00 is LHM++'s frontal reference: the face must show), best q first
                fr = [r for r in rows if r["role"] == role and r["cam"] == cam and abs(r["yaw_deg"]) <= FRONT_YAW
                      and r not in picked and _gate(r, P.front_gate)]
                for r in fr:
                    r["q"] = _q(r, h_med)
                for r in sorted(fr, key=lambda r: -r["q"]):
                    if len(picked) >= n_rr + P.front_extra:
                        break
                    if all(abs(r["t"] - p["t"]) >= FRONT_DT for p in picked):
                        picked.append(r)
            out += picked
            log(f"{role} phone {cam}: {len(rr)} pass the gate, {len(picked)} SAM candidates"
                + (f" ({len(picked) - n_rr} extra frontal)" if P.front_extra else "") + ", yaw "
                f"{sorted(int(p['yaw_deg']) for p in picked)}")
    return out


def decode_frames(cam: str, pts_list: list[float], dest: Path, names: list[str]) -> list[dict]:
    """Decode the frames of one camera at the given source pts (the take's ingest decoder: ffmpeg showinfo pts,
    autorotation) -> dest/<name>.jpg (quality 95, as ingest). Returns per frame the decoded pts."""
    from ..ingest import decode_range
    from ..take import Take
    take = Take.load(PACKAGE_ROOT / "takes" / f"{P.take}.toml")
    src = take.source(cam).path
    recs = []
    tmp = dest / f"_tmp_{cam}"
    for pts, name in sorted(zip(pts_list, names)):
        if tmp.exists():
            shutil.rmtree(tmp)
        got = decode_range(src, max(0.0, pts - 0.02), pts + 0.02, tmp, quality=95)
        if not got:
            recs.append({"name": name, "pts": None})
            continue
        best = min(got, key=lambda g: abs(g["pts"] - pts))
        shutil.move(str(tmp / best["file"]), str(dest / f"{name}.jpg"))
        recs.append({"name": name, "pts": best["pts"], "dpts": round(best["pts"] - pts, 6)})
    if tmp.exists():
        shutil.rmtree(tmp)
    return recs


def sharpness(img_gray: np.ndarray, box) -> float:
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H, W = img_gray.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    c = img_gray[y0:y1, x0:x1]
    if c.size < 100:
        return 0.0
    s = 400.0 / max(1, c.shape[0])
    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return float(cv2.Laplacian(c, cv2.CV_32F).var())


def frames() -> Path:
    rows = json.loads((P.out / "survey.json").read_text())["rows"]
    cands = pick_sam_candidates(rows)
    fits = {r: load_fit(r) for r in ROLES}
    if P.sam_in.exists():
        shutil.rmtree(P.sam_in)
    (P.sam_in / "frames").mkdir(parents=True)
    keys = sorted({(r["cam"], r["i"]) for r in cands})
    cams = {c: load_cam(c) for c in P.cams}
    vidx = np.arange(0, 10475, VSTEP)
    frames_out = []
    t0 = time.time()
    for cam in P.cams:
        ks = [k for k in keys if k[0] == cam]
        if not ks:
            continue
        c = cams[cam]
        names = [f"{cam}_{i:05d}" for _, i in ks]
        recs = decode_frames(cam, [float(c["pts"][i]) for _, i in ks], P.sam_in / "frames", names)
        tb = np.array([c["t"][i] + video_delta(cam, c["t"][i]) for _, i in ks])
        P_, J = {}, {}
        for r in ROLES:
            p = interp_params(fits[r], tb)
            P_[r] = pose_vertices(fits[r], p, vidx)
            J[r] = p["joints"]
        for n, ((_, i), rec) in enumerate(zip(ks, recs)):
            if rec["pts"] is None:
                log(f"{cam}/{i}: decode failed")
                continue
            K, R, tv = c["K"][i], c["R"][i], c["tvec"][i]
            W, H = c["W"], c["H"]
            img = cv2.imread(str(P.sam_in / "frames" / f"{rec['name']}.jpg"), cv2.IMREAD_GRAYSCALE)
            if img is None or img.shape != (H, W):
                log(f"{cam}/{i}: decoded frame {None if img is None else img.shape} != {(H, W)}")
                continue
            prompts, meta = {}, {}
            uv_s = {r: project(P_[r][n], K, R, tv)[0] for r in ROLES}
            for r in ROLES:
                o = ROLES[1 - ROLES.index(r)]
                uv = uv_s[r]
                x0, y0 = uv.min(0)
                x1, y1 = uv.max(0)
                h = y1 - y0
                box = [max(0.0, x0 - 0.10 * h * 0.5), max(0.0, y0 - 0.07 * h), min(W - 1.0, x1 + 0.10 * h * 0.5),
                       min(H - 1.0, y1 + 0.03 * h)]
                ju, _ = project(J[r][n][list(POS_JOINTS)], K, R, tv)
                pos = [[float(u), float(v)] for u, v in ju if 0 <= u < W and 0 <= v < H]
                jo, _ = project(J[o][n][list(NEG_JOINTS)], K, R, tv)
                neg = []
                for u, v in jo:
                    if not (box[0] <= u <= box[2] and box[1] <= v <= box[3]):
                        continue
                    if np.min(np.hypot(uv[:, 0] - u, uv[:, 1] - v)) > 0.06 * h:
                        neg.append([float(u), float(v)])
                prompts[r] = {"box": [round(x, 1) for x in box], "pos": pos, "neg": neg}
                meta[r] = {"sharp": round(sharpness(img, box), 2), "box": prompts[r]["box"]}
            frames_out.append({"id": rec["name"], "cam": cam, "i": int(i), "t": round(float(c["t"][i]), 4),
                               "tb": round(float(tb[n]), 4), "pts": rec["pts"], "dpts": rec["dpts"], "W": W, "H": H,
                               "file": f"frames/{rec['name']}.jpg", "prompts": prompts, "meta": meta,
                               "cand_roles": sorted({r["role"] for r in cands if r["cam"] == cam and r["i"] == i})})
        log(f"phone {cam}: {len(ks)} frames decoded ({time.time() - t0:.0f} s)")
    (P.sam_in / "prompts.json").write_text(json.dumps({"frames": frames_out}, indent=1))
    (P.out / "sam_candidates.json").write_text(json.dumps({"gate": P.gate or GATE, "per_cam": P.per_cam, "yaw_bins": YAW_BINS,
                                                           "min_dt": MIN_DT, "candidates": cands}, indent=1))
    log(f"{len(frames_out)} frames, {len(cands)} dancer candidates -> {P.sam_in}")
    return P.sam_in


def sam_job_spec() -> dict:
    rel_in = _ws(P.sam_in)
    rel_out = _ws(P.sam_out)
    return {"schema": "dancecap.pod_job/1", "name": P.sam_job_name, "take": P.take, "max_minutes": 12.0,
            "needs": ["dancers"], "inputs": [rel_in + "/"],
            "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
            "env": {}, "commands": [
                {"name": "sam", "timeout_min": 10,
                 "cmd": f"python remote/cand_lhmpp/recap_sam_pod.py --input {rel_in.removeprefix('dancecap/')} "
                        f"--out {rel_out.removeprefix('dancecap/')}"}],
            "outputs": [rel_out + "/"], "verify": [rel_out + "/sam.json"],
            "after_download": [P.cli("select")],
            "notes": [P.sam_notes]}


def sam_job(dest: Path | None = None) -> Path:
    dest = Path(dest) if dest else P.sam_job
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(sam_job_spec(), indent=1))
    log(f"-> {dest}")
    return dest


# ================================================================================================ select crops
class _FrameBodies:
    """bodies[r].at(t) for cand_lhmpp.occlusion_stats: the full posed fit meshes of ONE frame."""

    def __init__(self, V: dict):
        self.V = V

    def __getitem__(self, r):
        v = self.V[r]

        class _B:
            def at(self, t):
                return v, None, 1.0
        return _B()


def _choose_mask(cands: list[np.ndarray], body_vis: np.ndarray, body_dil: np.ndarray, Ws: int, Hs: int):
    """SAM candidate that best covers the visible body (recall) without spilling far outside it."""
    best, best_s, best_info = None, -9.0, None
    for k, m in enumerate(cands):
        ms = cv2.resize(m.astype(np.float32), (Ws, Hs), interpolation=cv2.INTER_AREA) > 0.5
        if not ms.any():
            continue
        rec = float((ms & body_vis).sum() / max(1, body_vis.sum()))
        out = float((ms & ~body_dil).sum() / max(1, ms.sum()))
        s = rec - 1.5 * out
        if s > best_s:
            best, best_s, best_info = k, s, {"recall": round(rec, 4), "outside_frac": round(out, 4)}
    return best, best_info


def _clean_mask(m: np.ndarray, body_full: np.ndarray) -> np.ndarray:
    """Keep the components that touch the dilated projected body; fill tiny enclosed holes (SAM speckle)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    for k in range(1, n):
        keep[k] = bool(body_full[lab == k].any()) and st[k, cv2.CC_STAT_AREA] >= 30
    out = keep[lab]
    from .cand_lhmpp import _holes
    tiny = _holes(out, 1) & ~_holes(out, max(20, int(0.0005 * out.sum())))
    return out | tiny


def select() -> dict:
    import torch
    from types import SimpleNamespace
    from . import cand_lhmpp as CL
    from ..avatar_texture import model_dir_for
    import smplx
    spec = json.loads((P.sam_in / "prompts.json").read_text())
    sam = json.loads((P.sam_out / "sam.json").read_text())
    fits = {r: load_fit(r) for r in ROLES}
    d = fits["lead"]
    sm = smplx.create(str(model_dir_for(d)), model_type="smplx", gender="neutral", ext="npz", use_pca=False,
                      flat_hand_mean=True, num_betas=int(d.get("num_betas", 16)))
    faces_np = np.asarray(sm.faces, np.int64)
    faces = torch.tensor(faces_np)
    hand_faces = CL.hand_face_mask(faces_np, sm.lbs_weights.detach().numpy())
    cams = {c: load_cam(c) for c in P.cams}
    tb_all = np.array([fr["tb"] for fr in spec["frames"]])
    Vfull = {r: pose_vertices(fits[r], interp_params(fits[r], tb_all)) for r in ROLES}
    rows = []
    S = P.sel or SEL
    MASKS = P.out / "masks"
    if MASKS.exists():
        shutil.rmtree(MASKS)
    MASKS.mkdir(parents=True)
    t0 = time.time()
    for n, fr in enumerate(spec["frames"]):
        fid, cam, i = fr["id"], fr["cam"], fr["i"]
        c = cams[cam]
        K, R, tv, W, H = c["K"][i], c["R"][i], c["tvec"][i], fr["W"], fr["H"]
        posed = {r: Vfull[r][n] for r in ROLES}
        zc = min(float(np.median(posed[r] @ np.asarray(R).T[:, 2] + tv[2])) for r in ROLES)
        s = float(np.clip(zc / (float(K[0, 0]) * CL.OCC_CELL_M), 0.1, 1.0))
        Ws, Hs = int(math.ceil(W * s)), int(math.ceil(H * s))
        zb = {r: CL._body_zbuffer(torch, posed[r], faces, K, R, tv, s, Ws, Hs) for r in ROLES}
        cell_m = zc / (float(K[0, 0]) * s)
        dil = max(1, int(round(0.12 / cell_m)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1))
        chosen = {}
        info = {}
        for r in ROLES:
            o = ROLES[1 - ROLES.index(r)]
            ps, po = np.isfinite(zb[r]), np.isfinite(zb[o])
            vis = ps & ~(po & (zb[o] < zb[r]))
            body_dil = cv2.dilate(ps.astype(np.uint8), ker) > 0
            cands = [cv2.imread(str(P.sam_out / f"{fid}_{r}_{k}.png"), cv2.IMREAD_GRAYSCALE) for k in range(3)]
            cands = [m > 127 for m in cands if m is not None]
            k, ci = _choose_mask(cands, vis, body_dil, Ws, Hs)
            if k is None:
                info[r] = {"error": "no mask"}
                continue
            bd_full = cv2.resize(body_dil.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            m = _clean_mask(cands[k], bd_full)
            chosen[r] = m
            cv2.imwrite(str(MASKS / f"{fid}_{r}.png"), m.astype(np.uint8) * 255)
            ys, xs = np.nonzero(m)
            border = bool(len(xs) and (xs.min() <= 2 or ys.min() <= 2 or xs.max() >= W - 3 or ys.max() >= H - 3))
            info[r] = {"sam_k": int(k), "sam_score": sam["frames"].get(fid, {}).get(r, {}).get("scores", [0, 0, 0])[k],
                       **ci, "border": border, "area": int(m.sum()),
                       "mask_h_px": int(ys.max() - ys.min() + 1) if len(ys) else 0}
        tr = SimpleNamespace(K={i: K}, R={i: R}, tvec={i: tv})
        occ = CL.occlusion_stats(torch, cam, i, fr["tb"], tr, _FrameBodies(posed), faces,
                                 {r: chosen.get(r) for r in ROLES}, W, H, hand_faces) if len(chosen) == 2 else {}
        img = cv2.imread(str(P.sam_in / fr["file"]), cv2.IMREAD_GRAYSCALE)
        for r in fr["cand_roles"]:
            if r not in chosen:
                continue
            p = interp_params(fits[r], np.array([fr["tb"]]))
            row = {"id": fid, "cam": cam, "i": i, "t": fr["t"], "tb": fr["tb"], "role": r, "W": W, "H": H,
                   "yaw_deg": round(body_yaw_deg(p["global_orient"][0], p["joints"][0][0], R, tv), 1),
                   "sharp": round(sharpness(img, fr["meta"][r]["box"]), 2), **info[r], "occ": occ.get(r)}
            ys, xs = np.nonzero(chosen[r])
            row["box"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            drop = []
            o_ = occ.get(r) or {}
            if o_.get("white_frac") is None or o_["white_frac"] > S["white_frac"]:
                drop.append(f"white {o_.get('white_frac')}")
            if o_.get("bad_hole_frac") is not None and o_["bad_hole_frac"] > S["bad_hole_frac"]:
                drop.append(f"hole {o_['bad_hole_frac']}")
            if row["outside_frac"] > S["outside_frac"]:
                drop.append(f"outside {row['outside_frac']}")
            if row["sam_score"] < S["sam_score"]:
                drop.append(f"score {row['sam_score']}")
            if row["border"]:
                drop.append("border")
            if P.ignore_rects.get(cam) and mask_touches(chosen[r], P.ignore_rects[cam]):
                drop.append("watermark")
            if P.preview_window[0] - 1 <= fr["tb"] <= P.preview_window[1] + 1:
                drop.append("preview window")
            row["drop"] = drop
            rows.append(row)
        if n % 20 == 0:
            log(f"{n}/{len(spec['frames'])} frames ({time.time() - t0:.0f} s)")
    (P.out / "select.json").write_text(json.dumps({"rules": S, "rows": rows}, indent=1, default=float))
    return pick(rows)


def face_covered(fits: dict, r: dict, cams: dict) -> bool:
    """A wrist or finger joint in front of the face in the image: within 0.08 m of the eye midpoint or the jaw joint
    (image distance at that depth) and at least 2 cm nearer to the camera (frontal half only)."""
    if abs(r["yaw_deg"]) > 90:
        return False
    p = interp_params(fits[r["role"]], np.array([r["tb"]]))
    J = p["joints"][0]
    c = cams[r["cam"]]
    R, tv = c["R"][r["i"]], c["tvec"][r["i"]]
    P_ = np.concatenate([J, (0.5 * (J[23] + J[24]))[None]], 0)  # + the eye midpoint
    uv, z = project(P_, c["K"][r["i"]], R, tv)
    f = float(c["K"][r["i"]][0, 0])
    for face in (55, 22):  # eyes, jaw
        for j in (20, 21, 27, 28, 42, 43):
            if np.linalg.norm(uv[j] - uv[face]) * z[face] / f < 0.08 and z[j] < z[face] - 0.02:
                return True
    return False


# ------------------------------------------------------------------------------------------------ colour
def _masked_pixels(r: dict, role: str) -> np.ndarray:
    img = cv2.imread(str(P.sam_in / "frames" / f"{r['id']}.jpg"))[:, :, ::-1]
    m = cv2.imread(str(P.out / "masks" / f"{r['id']}_{role}.png"), cv2.IMREAD_GRAYSCALE) > 127
    return img[m]


def colour_maps(rows: list[dict], ref_cam: str = "01", n: int = 60000) -> dict:
    """Per camera and channel a = slope, b = offset (sRGB 0..1) mapping its dancer pixels onto the reference camera's:
    least squares between the 2..98 % quantiles of the pooled masked pixels of both dancers (equal samples per dancer)
    over every clean candidate crop of the camera (same clothes in every camera)."""
    rng = np.random.default_rng(0)
    qs = np.linspace(2, 98, 25)
    Q = {}
    for cam in P.cams:
        pools = []
        for role in ROLES:
            px = []
            for r in rows:
                if r["cam"] != cam or r["role"] != role or r["drop"]:
                    continue
                px.append(_masked_pixels(r, role))
            if px:
                p = np.concatenate(px).astype(float) / 255.0
                pools.append(p[rng.choice(len(p), min(n, len(p)), replace=False)])
        if len(pools) == 2:
            Q[cam] = np.percentile(np.concatenate(pools), qs, axis=0)  # (25, 3)
    maps = {}
    for cam, q in Q.items():
        ab = []
        for ch in range(3):
            A = np.stack([q[:, ch], np.ones(len(qs))], 1)
            a_, b_ = np.linalg.lstsq(A, Q[ref_cam][:, ch], rcond=None)[0]
            a_ = float(np.clip(a_, 0.6, 1.6))
            b_ = float(np.mean(Q[ref_cam][:, ch] - a_ * q[:, ch]))
            ab.append([round(a_, 4), round(b_, 4)])
        maps[cam] = ab
    return maps


def apply_colour(img_rgb_u8: np.ndarray, ab) -> np.ndarray:
    x = img_rgb_u8.astype(np.float32) / 255.0
    for ch in range(3):
        x[..., ch] = ab[ch][0] * x[..., ch] + ab[ch][1]
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def white_pixels(img_rgb_u8: np.ndarray, mask: np.ndarray, band=(0.18, 0.50), keep: float = 0.5,
                 min_l: float = 35.0) -> np.ndarray:
    """(n, 3) uint8 pixels of a white TOP in a dancer crop: the mask rows between band[0] and band[1] of the mask's
    height (shoulders to hips), the brightest `keep` share of them (CIE L*), and L* >= min_l (skin and shadows out)."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return np.zeros((0, 3), np.uint8)
    y0, y1 = ys.min(), ys.max()
    h = y1 - y0 + 1
    sel = (ys >= y0 + band[0] * h) & (ys <= y0 + band[1] * h)
    px = img_rgb_u8[ys[sel], xs[sel]]
    if not len(px):
        return px
    L = cv2.cvtColor(px[None].astype(np.uint8), cv2.COLOR_RGB2LAB)[0][:, 0].astype(float) * 100.0 / 255.0
    thr = max(min_l, float(np.percentile(L, 100 * (1 - keep))))
    return px[L >= thr]


def lab_mean(px_u8: np.ndarray) -> np.ndarray:
    """Mean CIE Lab (L* 0..100, a*, b*) of sRGB uint8 pixels (OpenCV's float conversion)."""
    if not len(px_u8):
        return np.full(3, np.nan)
    return cv2.cvtColor((px_u8.astype(np.float32) / 255.0)[None], cv2.COLOR_RGB2LAB)[0].mean(0).astype(float)


def wb_maps(rows: list[dict], role: str, common: bool = False) -> dict:
    """White balance per camera on a white top: channel gains a_c = level / mean_c(white px) (b = 0) over every clean
    candidate crop of `role` on the camera, so the top's mean colour becomes grey: level = the camera's own mean
    (common False: each camera keeps its brightness) or the mean level over the cameras (common True: the same grey in
    every camera)."""
    mus = {}
    for cam in P.cams:
        px = []
        for r in rows:
            if r["cam"] != cam or r["role"] != role or r["drop"]:
                continue
            img = cv2.imread(str(P.sam_in / "frames" / f"{r['id']}.jpg"))[:, :, ::-1]
            m = cv2.imread(str(P.out / "masks" / f"{r['id']}_{role}.png"), cv2.IMREAD_GRAYSCALE) > 127
            px.append(white_pixels(img, m))
        px = np.concatenate(px) if px else np.zeros((0, 3), np.uint8)
        if len(px) < 200:
            continue
        mus[cam] = px
    level = float(np.mean([np.asarray(px, float).mean() / 255.0 for px in mus.values()])) if (common and mus) else None
    return {cam: wb_gains(px, level) for cam, px in mus.items()}


def wb_gains(px_u8: np.ndarray, level: float | None = None) -> list:
    """[[a_r, 0], [a_g, 0], [a_b, 0]] (apply_colour maps) that turn the mean of these pixels grey at `level` (sRGB 0..1;
    None = their own mean level); gains clipped to 0.6..1.6 like colour_maps' slopes."""
    mu = np.asarray(px_u8, float).reshape(-1, 3).mean(0) / 255.0
    g = (mu.mean() if level is None else level) / np.maximum(mu, 1e-6)
    return [[round(float(np.clip(gc, 0.6, 1.6)), 4), 0.0] for gc in g]


def colour_options(rows: list[dict]) -> dict:
    """The candidate harmonisations: none, every camera as the reference of colour_maps, wb (white top)."""
    opts = {"none": {}}
    for cam in P.cams:
        opts[f"ref:{cam}"] = colour_maps(rows, cam)
    if P.white_role:
        opts["wb"] = wb_maps(rows, P.white_role)
        opts["wb_common"] = wb_maps(rows, P.white_role, common=True)
    return opts


def colour_check(rows: list[dict], picks_by_role: dict | None = None) -> dict:
    """The white-top test of every harmonisation (owner 2026-10-23: the one that makes the white shirt most neutral):
    per option and camera the mean CIE Lab of the white top over the clean crops of P.white_role after the mapping, its
    chroma C*ab and the spread between cameras (Delta E of the means). The winner = the lowest mean chroma + 0.5 x the
    camera spread. Writes colour_check.jpg (rows = options; per camera one crop of each dancer)."""
    opts = colour_options(rows)
    role = P.white_role
    res = {}
    for name, maps in opts.items():
        per = {}
        for cam in P.cams:
            px = []
            for r in rows:
                if r["cam"] != cam or r["role"] != role or r["drop"]:
                    continue
                img = cv2.imread(str(P.sam_in / "frames" / f"{r['id']}.jpg"))[:, :, ::-1]
                if cam in maps:
                    img = apply_colour(img, maps[cam])
                m = cv2.imread(str(P.out / "masks" / f"{r['id']}_{role}.png"), cv2.IMREAD_GRAYSCALE) > 127
                px.append(white_pixels(img, m))
            px = np.concatenate(px) if px else np.zeros((0, 3), np.uint8)
            if len(px):
                lab = lab_mean(px)
                per[cam] = {"lab": [round(float(v), 2) for v in lab], "chroma": round(float(np.hypot(lab[1], lab[2])), 2),
                            "pixels": int(len(px))}
        cs = [v["chroma"] for v in per.values()]
        labs = [np.asarray(v["lab"]) for v in per.values()]
        spread = max((float(np.linalg.norm(a - b)) for i, a in enumerate(labs) for b in labs[i + 1:]), default=0.0)
        res[name] = {"maps": maps, "cams": per, "chroma_mean": round(float(np.mean(cs)), 2) if cs else None,
                     "camera_spread_dE": round(spread, 2),
                     "score": round(float(np.mean(cs)) + 0.5 * spread, 2) if cs else None}
    best = min((k for k in res if res[k]["score"] is not None), key=lambda k: res[k]["score"])
    # the sheet: one clean crop per camera and dancer (the best q), every option
    tiles_rows = []
    for name in res:
        row = []
        for cam in P.cams:
            for rl in ROLES:
                cand = [r for r in rows if r["cam"] == cam and r["role"] == rl and not r["drop"]]
                if picks_by_role:
                    ids = [x["id"] for x in picks_by_role.get(rl, []) if x["cam"] == cam]
                    cand = [r for r in cand if r["id"] in ids] or cand
                if not cand:
                    continue
                r = max(cand, key=lambda r: r.get("q", 0.0))
                img = cv2.imread(str(P.sam_in / "frames" / f"{r['id']}.jpg"))[:, :, ::-1]
                if cam in res[name]["maps"]:
                    img = apply_colour(img, res[name]["maps"][cam])
                m = cv2.imread(str(P.out / "masks" / f"{r['id']}_{rl}.png"), cv2.IMREAD_GRAYSCALE) > 127
                from . import cand_lhmpp as CL
                crop = CL.lhm_crop(np.ascontiguousarray(img), m)
                u8 = (np.clip(crop, 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]
                u8 = cv2.resize(u8, (168, 280), interpolation=cv2.INTER_AREA)
                ch = res[name]["cams"].get(cam, {}).get("chroma")
                lab = [f"{name} cam {cam}"] + ([f"top C* {ch:.1f}"] if rl == role and ch is not None else [])
                row.append(_label(u8, lab, colour=(0, 0, 200)))
        if row:
            tiles_rows.append(np.concatenate(row, 1))
    if tiles_rows:
        Wm = max(t.shape[1] for t in tiles_rows)
        sheet = np.concatenate([np.pad(t, ((0, 4), (0, Wm - t.shape[1]), (0, 0)), constant_values=255) for t in tiles_rows], 0)
        cv2.imwrite(str(P.out / "colour_check.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    out = {"white_role": role, "options": res, "best": best,
           "rule": "lowest (mean white-top chroma C*ab over the cameras + 0.5 x the largest camera-to-camera Delta E)"}
    (P.out / "colour_check.json").write_text(json.dumps(out, indent=1))
    log("colour check: " + ", ".join(f"{k} C* {v['chroma_mean']} spread {v['camera_spread_dE']}" for k, v in res.items())
        + f" -> {best}")
    return out


def _greedy_caps(ok: list[dict], n: int, min_dt: float, seed: list[dict]) -> list[dict]:
    picked = list(seed)

    def allowed(r):
        if any(r["cam"] == p["cam"] and abs(r["t"] - p["t"]) < min_dt for p in picked):
            return False
        return sum(p["cam"] == r["cam"] for p in picked) < P.caps.get(r["cam"], P.cap_default)

    while len(picked) < n:
        best, best_s = None, -1.0
        for r in ok:
            if r in picked or not allowed(r):
                continue
            dy = min(abs((r["yaw_deg"] - p["yaw_deg"] + 180) % 360 - 180) for p in picked) if picked else 180.0
            sc = r["q"] * (dy / 180.0 + 0.15)
            if sc > best_s:
                best, best_s = r, sc
        if best is None:
            break
        picked.append(best)
    return picked


def pick(rows: list[dict] | None = None, colour: bool | str = True) -> dict:
    """The final crops per dancer. colour: True = the profile's harmonisation (P.colour), False = none, or an option name
    (none | ref:<cam> | wb | auto: the colour_check winner)."""
    from . import cand_lhmpp as CL
    if rows is None:
        rows = json.loads((P.out / "select.json").read_text())["rows"]
    mode = P.colour if colour is True else ("none" if colour is False else str(colour))
    fits = {r: load_fit(r) for r in ROLES}
    cams = {c: load_cam(c) for c in P.cams}
    for r in rows:
        r["veto"] = P.veto.get((r["role"], r["id"]))
        r["face_covered"] = face_covered(fits, r, cams) and (r["role"], r["id"]) not in P.face_ok
        if r["sharp"] < P.min_sharp and "soft" not in " ".join(r["drop"]):
            r["drop"] = r["drop"] + [f"soft {r['sharp']:.0f}"]
    for cam in P.cams:
        rr = [r for r in rows if r["cam"] == cam]
        if not rr:
            continue
        a_med = float(np.median([r["area"] for r in rr]))
        s_med = float(np.median([r["sharp"] for r in rr]))
        for r in rr:
            res = min(1.0, r["mask_h_px"] / 700.0) ** 0.5
            r["q"] = round(min(1.0, r["area"] / a_med) * min(1.0, r["sharp"] / s_med) ** 0.5 * res *
                           max(0.0, r["recall"]) ** 2 * (0.5 if r["face_covered"] else 1.0), 4)
    if mode == "ref:01" and P.legacy_jobs:
        # the recap's exact record (refs.json unchanged)
        maps = colour_maps(rows)
        colour_rec = {"reference_phone": "01", "maps": maps, "applied": True,
                      "method": "per phone and channel x -> a x + b (sRGB 0..1), least squares on the 2..98 % quantiles "
                                "of both dancers' masked pixels pooled over the phone's clean candidate crops"}
    elif mode == "none" and P.legacy_jobs:
        maps = {}
        colour_rec = {"reference_phone": "01", "maps": maps, "applied": False,
                      "method": "per phone and channel x -> a x + b (sRGB 0..1), least squares on the 2..98 % quantiles "
                                "of both dancers' masked pixels pooled over the phone's clean candidate crops"}
    else:
        chk = colour_check(rows) if P.white_role else None
        if mode == "auto":
            if chk is None:
                raise SystemExit("pick --colour auto needs a profile with white_role (the white-top test)")
            mode = chk["best"]
        opts = chk["options"] if chk else colour_options(rows)
        if mode not in opts:
            raise SystemExit(f"pick: colour option {mode!r} unknown ({', '.join(opts)})")
        maps = opts[mode]["maps"]
        colour_rec = {"option": mode, "requested": P.colour if colour is True else str(colour), "maps": maps,
                      "applied": bool(maps),
                      "white_top_test": ({k: {kk: v[kk] for kk in ("chroma_mean", "camera_spread_dE", "score", "cams")}
                                          for k, v in chk["options"].items()} if chk else None),
                      "method": {"none": "no mapping (raw camera colours)",
                                 "wb": "per camera channel gains making the white top's mean grey (white_pixels)",
                                 "wb_common": "per camera channel gains making the white top's mean the SAME grey in every "
                                              "camera (white_pixels; level = the cameras' mean)"}.get(
                          mode, "per camera and channel x -> a x + b (sRGB 0..1), least squares on the 2..98 % quantiles "
                                "of both dancers' masked pixels pooled over the camera's clean candidate crops, onto "
                                f"camera {mode.removeprefix('ref:')}")}
    S = P.sel or SEL
    out = {"rules": S, "caps": P.caps, "min_sharp": P.min_sharp, "preview_window": P.preview_window,
           "face_ok": {f"{k[0]}:{k[1]}": v for k, v in P.face_ok.items()},
           "veto": {f"{k[0]}:{k[1]}": v for k, v in P.veto.items()},
           "colour": colour_rec,
           "roles": {}}
    if (P.input / "refs").exists():
        shutil.rmtree(P.input / "refs")
    for role in ROLES:
        ok = [r for r in rows if r["role"] == role and not r["drop"] and not r["veto"]]
        # slot 00 = LHM++'s frontal reference: a hi-res camera, face not covered, best q x cos(yaw) within 60 deg
        front = [r for r in ok if abs(r["yaw_deg"]) <= 60 and r["cam"] in P.hires and not r["face_covered"]]
        if P.hires_in_order:     # the sharper camera's face first (the tele), the next camera only when it has none
            for cam in P.hires:
                if any(r["cam"] == cam for r in front):
                    front = [r for r in front if r["cam"] == cam]
                    break
        seed = [max(front, key=lambda r: r["q"] * math.cos(math.radians(r["yaw_deg"])))] if front else []
        picks = _greedy_caps(ok, S["n"], S["min_dt"], seed)
        sd = P.input / "refs" / role
        sd.mkdir(parents=True, exist_ok=True)
        recs = []
        for k, r in enumerate(picks):
            img = cv2.imread(str(P.sam_in / "frames" / f"{r['id']}.jpg"))[:, :, ::-1]
            if r["cam"] in maps:
                img = apply_colour(img, maps[r["cam"]])
            m = cv2.imread(str(P.out / "masks" / f"{r['id']}_{role}.png"), cv2.IMREAD_GRAYSCALE) > 127
            crop = CL.lhm_crop(np.ascontiguousarray(img), m)
            u8 = (np.clip(crop, 0, 1) * 255 + 0.5).astype(np.uint8)
            cv2.imwrite(str(sd / f"{k:02d}.png"), u8[:, :, ::-1])
            recs.append({"slot": k, **{kk: r[kk] for kk in ("id", "cam", "i", "t", "tb", "yaw_deg", "sharp", "area",
                                                           "mask_h_px", "recall", "outside_frac", "sam_score", "q",
                                                           "face_covered")},
                         "white_frac": (r["occ"] or {}).get("white_frac"),
                         "bad_hole_frac": (r["occ"] or {}).get("bad_hole_frac")})
        out["roles"][role] = {"candidates": sum(r["role"] == role for r in rows),
                              "clean": sum(r["role"] == role and not r["drop"] for r in rows),
                              "after_veto": len(ok), "picks": recs}
        log(f"{role}: {len(ok)} clean after veto; picks "
            + ", ".join(f"{x['cam']}/{x['i']} {x['yaw_deg']:.0f}d" for x in recs))
    (P.out / "refs.json").write_text(json.dumps(out, indent=1, default=float))
    (P.out / "select.json").write_text(json.dumps({"rules": S, "rows": rows}, indent=1, default=float))
    sheets(out, rows)
    return out


def _label(img: np.ndarray, lines: list[str], colour=(255, 255, 255)) -> np.ndarray:
    img = np.ascontiguousarray(img)
    for k, txt in enumerate(lines):
        y = 16 + 16 * k
        cv2.putText(img, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
    return img


def sheets(sel: dict, rows: list[dict]) -> None:
    """refs_<role>.jpg (the LHM++ inputs as fed, 504 x 840 canvases) and masks_<role>.jpg (every candidate crop with its
    mask outline, kept = green, dropped = red + why)."""
    for role in ROLES:
        tiles = []
        for x in sel["roles"][role]["picks"]:
            im = cv2.imread(str(P.input / "refs" / role / f"{x['slot']:02d}.png"))
            im = cv2.resize(im, (252, 420), interpolation=cv2.INTER_AREA)
            tiles.append(_label(im, [f"{x['slot']:02d} ph{x['cam']} t{x['tb']:.1f}", f"yaw {x['yaw_deg']:.0f} h{x['mask_h_px']}",
                                     f"sh {x['sharp']:.0f} w{(x['white_frac'] or 0) * 100:.1f}%"]))
        if tiles:
            per = 8
            rws = [np.concatenate(tiles[k:k + per] + [np.full_like(tiles[0], 255)] * (per - len(tiles[k:k + per])), 1)
                   for k in range(0, len(tiles), per)]
            cv2.imwrite(str(P.out / f"refs_{role}.jpg"), np.concatenate(rws, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
        tiles = []
        for r in [r for r in rows if r["role"] == role]:
            img = cv2.imread(str(P.sam_in / "frames" / f"{r['id']}.jpg"))
            m = cv2.imread(str(P.out / "masks" / f"{r['id']}_{role}.png"), cv2.IMREAD_GRAYSCALE) > 127
            x0, y0, x1, y1 = r["box"]
            pad = int(0.06 * (y1 - y0))
            X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(r["W"], x1 + pad), min(r["H"], y1 + pad)
            col = (0, 200, 0) if not r["drop"] else (0, 0, 255)
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnt, -1, col, max(1, (y1 - y0) // 250))
            c = img[Y0:Y1, X0:X1]
            s = 300 / max(1, c.shape[0])
            c = cv2.resize(c, (max(1, int(c.shape[1] * s)), 300))
            canvas = np.full((300, 170, 3), 40, np.uint8)
            w = min(170, c.shape[1])
            canvas[:, (170 - w) // 2:(170 - w) // 2 + w] = c[:, (c.shape[1] - w) // 2:(c.shape[1] - w) // 2 + w]
            tiles.append(_label(canvas, [f"{r['id']} y{r['yaw_deg']:.0f}", ("ok q%.2f" % r["q"]) if not r["drop"]
                                         else r["drop"][0][:22]]))
        per = 14
        rws = [np.concatenate(tiles[k:k + per] + [np.zeros_like(tiles[0])] * (per - len(tiles[k:k + per])), 1)
               for k in range(0, len(tiles), per)]
        cv2.imwrite(str(P.out / f"masks_{role}.jpg"), np.concatenate(rws, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ================================================================================================ pod job (LHM++)
PREV_SCALE = 0.5            # preview renders at half the camera's size (recap: 540 x 960 panels of a 1620 x 960 side-by-side)


def _scaled_K(K: np.ndarray, s: float) -> np.ndarray:
    """K for an image scaled by s (pixel centres at integers: c' = (c + 0.5) s - 0.5)."""
    K2 = np.array(K, float).copy()
    K2[0, 0] *= s
    K2[1, 1] *= s
    K2[0, 2] = (K2[0, 2] + 0.5) * s - 0.5
    K2[1, 2] = (K2[1, 2] + 0.5) * s - 0.5
    return K2


def a2_instants(n: int = 6) -> list[dict]:
    """n instants spread over the dance where both dancers are fully inside BOTH A2 cameras (survey rows), one of
    them inside the preview window; per camera the frame nearest in body time."""
    rows = json.loads((P.out / "survey.json").read_text())["rows"]
    c0, c1 = P.a2_cams
    by = {}
    for r in rows:
        by.setdefault((r["cam"], r["i"]), {})[r["role"]] = r
    good = {c: sorted([(v["lead"]["tb"], i) for (cc, i), v in by.items() if cc == c and len(v) == 2
                       and v["lead"]["inimg"] >= 1.0 and v["follow"]["inimg"] >= 1.0]) for c in P.a2_cams}
    t6 = np.array([t for t, _ in good[c1]])
    picks = []
    targets = list(np.linspace(P.a2_span[0], P.a2_span[1], n - 1)) + [0.5 * (P.preview_window[0] + P.preview_window[1])]
    for tt in targets:
        cand = [(abs(t - tt), t, i) for t, i in good[c0]]
        cand = [c for c in cand if len(t6) and np.min(np.abs(t6 - c[1])) < 0.2]
        if not cand:
            continue
        _, t1, i1 = min(cand)
        if any(p[c0] == i1 for p in picks):          # the preview instant can fall on a spread one: render it once
            continue
        j = int(np.argmin(np.abs(t6 - t1)))
        picks.append({c0: i1, c1: good[c1][j][1], f"tb_{c0}": t1, f"tb_{c1}": float(t6[j])})
    return picks


def lhm_job_spec(max_minutes: float = 40.0) -> dict:
    rel_in, rel_out, rel_ren = (_ws(P.input).removeprefix("dancecap/"), _ws(P.pod_out).removeprefix("dancecap/"),
                                _ws(P.out / "renders").removeprefix("dancecap/"))
    return {"schema": "dancecap.pod_job/1", "name": P.lhm_job_name, "take": P.take, "max_minutes": max_minutes,
            "needs": [], "inputs": [_ws(P.input) + "/"],
            "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
            "env": {"LHMPP_COMMIT": LHMPP_COMMIT, "LHMPP_ROOT": "/workspace/lhmpp_cand/LHM-plusplus",
                    "LHMPP_VENV": "/workspace/lhmpp_cand/venv", "LHMPP_MODEL": MODEL_NAME,
                    "TORCH_HOME": "/workspace/lhmpp_cand/torch_home", "HF_HOME": "/workspace/lhmpp_cand/hf_home"},
            "commands": [{"name": "install", "timeout_min": 25, "cmd": "bash remote/cand_lhmpp/install.sh"},
                         {"name": "build", "timeout_min": 25,
                          "cmd": f"/workspace/lhmpp_cand/venv/bin/python remote/cand_lhmpp/recap_lhmpp_pod.py "
                                 f"--input {rel_in} --out {rel_out} --renders {rel_ren}"}],
            "outputs": [_ws(P.pod_out) + "/", _ws(P.out / "renders") + "/"],
            "verify": [_ws(P.pod_out) + "/recap_lhmpp_pod.json", _ws(P.pod_out) + "/export/lead_export.npz",
                       _ws(P.pod_out) + "/export/follow_export.npz", _ws(P.pod_out) + "/a2.json"],
            "after_download": [P.cli("import --session <sid>") if P.legacy_jobs
                               else f"python -m dancecap.mono4d.recap_splatav import --take {P.take} --session <sid>"],
            "notes": [P.lhm_notes]}


def job(max_minutes: float = 40.0, dest: Path | None = None) -> Path:
    from types import SimpleNamespace  # noqa: F401
    fits = {r: load_fit(r) for r in ROLES}
    cams = {c: load_cam(c) for c in set(P.a2_cams) | set(P.prev_cams)}
    for role in ROLES:
        n = len(list((P.input / "refs" / role).glob("*.png")))
        if n < 8:
            raise SystemExit(f"{role}: only {n} reference crops in {P.input / 'refs' / role}: run pick first")
    keys_t = []   # body times, one pose key each
    groups, gmeta = {}, {}

    def add_group(name, cam, idx, scale):
        c = cams[cam]
        idx = np.asarray(idx, int)
        tb = c["t"][idx] + video_delta(cam, c["t"][idx])
        k0 = len(keys_t)
        keys_t.extend(tb.tolist())
        Ks = np.stack([_scaled_K(c["K"][i], scale) for i in idx])
        groups[name] = {"key": np.arange(k0, k0 + len(idx)), "K": Ks, "R": c["R"][idx], "tvec": c["tvec"][idx],
                        "index": idx}
        gmeta[name] = {"cam": cam, "W": int(round(c["W"] * scale)), "H": int(round(c["H"] * scale)), "scale": scale,
                       "frames": [{"i": int(i), "t": round(float(c["t"][i]), 5), "tb": round(float(t), 5),
                                   "pts": round(float(c["pts"][i]), 6)} for i, t in zip(idx, tb)]}

    inst = a2_instants()
    for cam in P.a2_cams:
        add_group(f"a2_{cam}", cam, [p[cam] for p in inst], 1.0)
    for cam in P.prev_cams:
        c = cams[cam]
        tb = c["t"] + video_delta(cam, c["t"])
        idx = np.nonzero((tb >= P.preview_window[0]) & (tb < P.preview_window[1]) & c["ok"])[0]
        add_group(f"prev_{cam}", cam, idx, PREV_SCALE)
    T = np.asarray(keys_t, float)
    Z = {"t": T}
    for role in ROLES:
        p = interp_params(fits[role], T)
        for name, n, short in ROT_KEYS:
            Z[f"{role}_{short}"] = p[name].astype(np.float32)
        Z[f"{role}_pelvis"] = p["joints"][:, 0].astype(np.float64)
        Z[f"{role}_joints22"] = p["joints"][:, :22].astype(np.float32)
        Z[f"{role}_betas"] = np.asarray(fits[role]["betas"], np.float64)
    np.savez(P.input / "poses.npz", **Z)
    V = {}
    for g, d in groups.items():
        for k, v in d.items():
            V[f"{g}_{k}"] = v
    np.savez(P.input / "views.npz", **V)
    (P.input / "views.json").write_text(json.dumps({"groups": gmeta, "passes": [[f"prev_{c}" for c in P.prev_cams]],
                                                    "a2_instants": inst, "preview_window": P.preview_window,
                                                    "fit": _ws(P.fit_dir)}, indent=1))
    dest = Path(dest) if dest else P.lhm_job
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(lhm_job_spec(max_minutes), indent=1))
    log(f"{len(T)} pose keys; groups " + ", ".join(f"{g}: {len(d['key'])}" for g, d in groups.items())
        + f"; A2 instants {[round(p[f'tb_{P.a2_cams[0]}'], 2) for p in inst]} -> {dest}")
    return dest


# ================================================================================================ import: .splatav, checks, preview
SPLATAV_MAGIC = b"SPAV"
RECORD = [("mean", "f4", 3), ("rotation", "f4", 4), ("log_scale", "f4", 3), ("opacity_logit", "f4", 1),
          ("sh0", "f4", 3), ("joints", "u2", 4), ("weights", "f4", 4), ("hair", "u1", 1), ("reserved", "u1", 3),
          ("strand", "i4", 1), ("segment", "i4", 1), ("local_offset", "f4", 3), ("local_rotation", "f4", 4)]
REC_DTYPE = np.dtype([(n, "<" + t, (c,)) if c > 1 else (n, "<" + t) for n, t, c in RECORD])
SH_C0 = 0.28209479177387814
MIRROR = np.array([1.0, 1.0, -1.0])
HAIR_JOINTS = (3, 6, 9, 12, 13, 14, 15, 16, 17)


def hair_selection(ex: dict) -> tuple[np.ndarray, dict]:
    """First-pass hair flag (canonical zero pose, world frame): crimson / red colour (HSV hue >= 330 or <= 18 deg,
    saturation >= 0.40, value >= 0.12) on Gaussians skinned mainly to the head, neck, spine or shoulders, within
    0.70 m of the head joint, not on the front of the chest below the neck (her top's red print); then only voxel
    clusters (1.5 cm cells, 26-connected) holding >= 2 % of the candidates (drops red specks)."""
    from scipy import ndimage
    rgb = np.clip(ex["rgb"], 0, 1).astype(np.float32)
    hsv = cv2.cvtColor(rgb[None], cv2.COLOR_RGB2HSV)[0]
    H, S, Vv = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    red = ((H >= 330) | (H <= 18)) & (S >= 0.40) & (Vv >= 0.12)
    top = ex["top4_idx"][:, 0].astype(int)
    J = ex["rest_joints"]
    x = ex["mean0"].astype(float)
    region = np.isin(top, HAIR_JOINTS) & (np.linalg.norm(x - J[15], axis=1) < 0.70)
    chest = (x[:, 2] > J[9][2] + 0.04) & (x[:, 1] < J[12][1] - 0.06)
    cand = red & region & ~chest
    info = {"red": int(red.sum()), "region": int(region.sum()), "front_chest_excluded": int((red & region & chest).sum()),
            "candidates": int(cand.sum())}
    if cand.sum() == 0:
        info.update({"hair": 0, "clusters_kept": 0})
        return cand, info
    cell = 0.015
    g = np.floor((x[cand] - x[cand].min(0)) / cell).astype(int)
    occ = np.zeros(g.max(0) + 1, bool)
    occ[tuple(g.T)] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    lg = lab[tuple(g.T)]
    sizes = np.bincount(lg, minlength=n + 1)
    keep_lab = np.nonzero(sizes >= max(20, 0.02 * cand.sum()))[0]
    keep_lab = keep_lab[keep_lab > 0]
    ok = np.isin(lg, keep_lab)
    hair = np.zeros(len(x), bool)
    hair[np.nonzero(cand)[0][ok]] = True
    info.update({"hair": int(hair.sum()), "clusters": int(n), "clusters_kept": int(len(keep_lab)),
                 "hair_y_range_m": [round(float(x[hair, 1].min()), 3), round(float(x[hair, 1].max()), 3)] if hair.any() else None,
                 "head_joint_y_m": round(float(J[15][1]), 3)})
    return hair, info


def export_hair(role: str, ex: dict) -> tuple[np.ndarray, dict]:
    """The hair byte at export: the recap's red-hue first pass on the follow (hair_rule "red"), else 0 with the
    profile's note (the hair stage selects it)."""
    if role == "follow" and P.hair_rule == "red":
        return hair_selection(ex)
    return np.zeros(len(ex["mean0"]), bool), {"hair": 0, "note": P.hair_notes.get(role, "no hair selected at export")}


def splat_view(xyz, rgb, scale, opac, R, t, K, W, H, tint=None, bg=255) -> np.ndarray:
    """Quick CPU check render: every Gaussian with opacity >= 0.3 as an opaque disc of 1.5 sigma (largest axis),
    painter's order far to near. Not the real splat look; for the hair-selection check only."""
    Xc = xyz @ R.T + t
    z = Xc[:, 2]
    keep = (z > 0.05) & (opac >= 0.3)
    uv = Xc[:, :2] / z[:, None] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    rad = np.maximum(1, np.round(1.5 * scale.max(1) * K[0, 0] / np.maximum(z, 1e-3))).astype(int)
    img = np.full((H, W, 3), bg, np.uint8)
    col = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if tint is not None:
        col = col.copy()
        col[tint] = (0.35 * col[tint] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    for k in np.nonzero(keep)[0][np.argsort(-z[keep])]:
        c = col[k]
        cv2.circle(img, (int(uv[k, 0]), int(uv[k, 1])), int(min(rad[k], 12)), (int(c[2]), int(c[1]), int(c[0])), -1)
    return img


def hair_sheet(ex: dict, hair: np.ndarray, path: Path) -> None:
    import sys
    sys.path.insert(0, str(JOB_DIR))
    from recap_lhmpp_pod import look_at
    x = ex["mean0"].astype(float)
    ctr = (x.max(0) + x.min(0)) / 2
    W, H = 420, 700
    K = np.array([[700.0, 0, W / 2], [0, 700.0, H / 2], [0, 0, 1]])
    tiles = []
    for tint in (None, hair):
        for az in (0, 90, 180, 270):
            C = ctr + 2.9 * np.array([math.sin(math.radians(az)), 0.0, math.cos(math.radians(az))])
            R, t = look_at(C, ctr)
            tiles.append(splat_view(x, ex["rgb"], ex["scale"], ex["opacity"], R, t, K, W, H, tint))
    cv2.imwrite(str(path), np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def write_splatav(role: str, ex: dict, hair: np.ndarray, hair_info: dict, pod: dict, a2: dict, refs: dict,
                  dest: Path, *, name: str | None = None, capture: str | None = None, take: str | None = None,
                  inputs: list | None = None, pod_report: Path | None = None, hair_note: str | None = None,
                  likeness: str | None = None) -> dict:
    """<dest>/<role>.splatav + <role>.json from LHM++'s export npz (SPLAT_AVATAR_FORMAT.md). The keyword arguments make it
    usable for any take (mono4d.take_splatav); left out, they are the active profile's (P, use())."""
    from ..export_unity import read_skin
    N = len(ex["mean0"])
    rec = np.zeros(N, REC_DTYPE)
    rec["mean"] = ex["mean0"].astype(np.float64) * MIRROR
    q = ex["rot0"].astype(np.float64)                        # wxyz, world
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    q_xyzw = np.concatenate([q[:, 1:], q[:, :1]], 1)
    rec["rotation"] = q_xyzw * np.array([-1.0, -1.0, 1.0, 1.0])
    rec["log_scale"] = np.log(np.maximum(ex["scale"].astype(np.float64), 1e-9))
    o = np.clip(ex["opacity"].astype(np.float64), 1e-6, 1 - 1e-6)
    rec["opacity_logit"] = np.log(o / (1 - o))
    rec["sh0"] = (ex["rgb"].astype(np.float64) - 0.5) / SH_C0
    rec["joints"] = ex["top4_idx"].astype(np.uint16)
    rec["weights"] = ex["top4_w"]
    rec["hair"] = hair.astype(np.uint8)
    rec["strand"] = -1
    rec["segment"] = -1
    rec["local_rotation"] = np.array([0.0, 0.0, 0.0, 1.0])
    hdr = np.zeros(16, "<u4")
    hdr[1], hdr[2], hdr[3], hdr[4], hdr[5], hdr[6] = 1, N, REC_DTYPE.itemsize, 0, 1, 64
    assert REC_DTYPE.itemsize == 120, REC_DTYPE.itemsize
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / f"{role}.splatav"
    with open(p, "wb") as f:
        f.write(SPLATAV_MAGIC)
        f.write(hdr[1:].tobytes())
        f.write(rec.tobytes())
    Ju = ex["rest_joints"] * MIRROR
    cjw = []
    for j in range(55):
        M = np.eye(4)
        M[:3, 3] = Ju[j]
        cjw.append([round(float(v), 7) for v in M.reshape(-1)])
    nb = np.zeros((55, 3))
    nb[1:22] = np.asarray(ex["neutral_body_pose"], float).reshape(21, 3)
    W = ex["weights"].astype(np.float64)
    w4 = np.take_along_axis(W, ex["top4_idx"].astype(int), 1).sum(1) / np.maximum(W.sum(1), 1e-12)
    skin_cmp = None
    # the capture's skin file: next to the profile's Unity dir, or (a take written by mono4d.take_splatav) next to dest
    sk = (Path(dest).parent if take is not None else P.unity_dir.parent) / f"smplx_{role}_skin.bin"
    if sk.exists():
        _, skin = read_skin(sk)
        dj = np.linalg.norm(np.asarray(skin.joints, float) - Ju, axis=1) * 1e3
        skin_cmp = {"file": _ws(sk), "max_mm": round(float(dj.max()), 2), "median_mm": round(float(np.median(dj)), 2),
                    "pelvis_mm": round(float(dj[0]), 2), "per_joint_mm": [round(float(v), 2) for v in dj]}
    sc = ex["scale"].astype(np.float64)
    dn = pod["dancers"][role]
    if P.hair_rule == "red":
        hsel = "recap_splatav.hair_selection: " + (hair_selection.__doc__ or "").split("\n")[0] + \
               " ... (first pass, by colour + region + clusters)"
    else:
        hsel = "none at export (hair byte 0; profile hair_rule 'none')"
    js = {"format": "dancecap splat avatar", "version": 1, "role": role, "name": name or P.names[role], "capture": capture or P.capture,
          "take": take or P.take, "data": p.name, "data_bytes": p.stat().st_size, "data_sha256": _sha256(p), "count": int(N),
          "hair_count": int(hair.sum()), "record": [[n, t, c] for n, t, c in RECORD], "record_bytes": 120,
          "coords": "unity (x, y, -z) of the Y-up right-handed dancecap world", "units": "m",
          "conversion": {"points": "(x, y, z) -> (x, y, -z)", "quaternions_xyzw": "(qx, qy, qz, qw) -> (-qx, -qy, qz, qw)",
                         "source": "dancecap.export_unity.to_unity_points / to_unity_quats"},
          "colour": "sh0 DC; colour = 0.5 + 0.28209479 * sh0 (clamp 0..1); sRGB-encoded display colour (LHM++ use_rgb)",
          "canonical_pose": {"name": "smplx_zero_pose", "axis_angle": np.zeros((55, 3)).tolist(),
                             "note": "all 55 joint rotations identity: the shaped body's SMPL-X rest pose (T-pose)"},
          "lhmpp_neutral_pose": {"name": "lhmpp_neutral", "axis_angle": nb.round(6).tolist(),
                                 "cano_pose_type": (pod.get("renderer") or {}).get("cano_pose_type"),
                                 "note": "the pose LHM++ predicts the Gaussians in (template shape, betas 0); provenance only"},
          "betas": [round(float(b), 6) for b in ex["betas10"]], "fit_betas": [round(float(b), 6) for b in ex["betas16"]],
          "body_model": "SMPL-X neutral (SMPLX_NEUTRAL.npz of LHMPP-Prior), 10 betas, flat_hand_mean, no pose correctives",
          "rest_joints": Ju.round(7).tolist(), "parents": [int(v) for v in ex["parents"]],
          "canonical_joint_world": cjw,
          "skinning": {"joints_per_gaussian": 4,
                       "source": "LHM++'s own weights (diffused voxel field sampled at query point + offset; template "
                                 "SMPL-X weights for hand and face query points), top 4 renormalised",
                       "truncation": {"kept_weight_mass_mean": round(float(w4.mean()), 5),
                                      "kept_weight_mass_min": round(float(w4.min()), 5),
                                      "kept_weight_mass_p01": round(float(np.percentile(w4, 1)), 5),
                                      "gaussians_kept_mass_below_0.95": int((w4 < 0.95).sum())}},
          "posing": "SPLAT_AVATAR_FORMAT.md section 4: S_j = G_j T(-J_j), G_0 = [R_0 | pelvis], "
                    "G_j = G_parent [R_j | J_j - J_parent]; centre = (sum_k w_k S_k) mean; rotation = "
                    "q(blended 3x3) * rotation; scale / opacity / colour fixed",
          "hair": {"selection": hair_note or hsel, **hair_info, "binding": "unbound (strand = -1): filled by the hair stage"},
          "provenance": {"lhmpp_commit": pod.get("commit"), "lhmpp_model": pod.get("model"),
                         "weights_licence": "CC BY-NC 4.0 (LHM++ LICENSE_WEIGHT); code Apache-2.0 + NVIDIA EG3D parts",
                         "smplx_licence": "SMPL-X model licence (non-commercial)",
                         "inputs": inputs if inputs is not None else refs["roles"][role]["picks"], "colour_normalisation": refs.get("colour"),
                         "pod_report": _ws(pod_report or P.pod_out / "recap_lhmpp_pod.json"), "gpu": (pod.get("versions") or {}).get("gpu"),
                         "infer_s": dn.get("infer_s"), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "likeness": likeness or ("photographic likeness of a real person (owner consent 2026-10-19, research use); "
                                                  "never commit, never ship in an APK")},
          "checks": {"a2": a2.get("summary"), "rest_joints_vs_avatar_skin_mm": skin_cmp,
                     "rest_joints_vs_std_smplx_mm": dn.get("rest_joints_vs_std_smplx_mm"),
                     "weights_nonzero_per_gaussian": dn.get("weights_nonzero_per_gaussian")},
          "stats": {"opacity_median": round(float(np.median(ex["opacity"])), 4),
                    "opacity_below_0.05": int((ex["opacity"] < 0.05).sum()),
                    "scale_max_mm_p50_p90_max": [round(float(np.percentile(sc.max(1), q_) * 1e3), 2) for q_ in (50, 90, 100)],
                    "bounds_min": (ex["mean0"] * MIRROR).min(0).round(4).tolist(),
                    "bounds_max": (ex["mean0"] * MIRROR).max(0).round(4).tolist()}}
    (dest / f"{role}.json").write_text(json.dumps(js, indent=1))
    return js


def read_splatav(p: Path) -> tuple[np.ndarray, dict]:
    raw = p.read_bytes()
    assert raw[:4] == SPLATAV_MAGIC, raw[:4]
    hdr = np.frombuffer(raw[4:64], "<u4")
    N, rb, hb = int(hdr[1]), int(hdr[2]), int(hdr[5])
    rec = np.frombuffer(raw[hb:hb + N * rb], REC_DTYPE)
    return rec, {"version": int(hdr[0]), "count": N, "record_bytes": rb, "sh_rest": int(hdr[3]), "flags": int(hdr[4])}


def file_selfcheck(role: str, dest: Path, Z: dict, key: int, ex: dict) -> dict:
    """Pose the WRITTEN Unity-frame file by the format's rule (mirrored rotations) and compare with the world-frame
    pose of the exported data, mirrored: checks the file, the conversion and the documented rule."""
    import sys
    sys.path.insert(0, str(JOB_DIR))
    import recap_lhmpp_pod as RP
    rec, h = read_splatav(dest / f"{role}.splatav")
    js = json.loads((dest / f"{role}.json").read_text())
    aa = np.concatenate([Z[f"{role}_root"][key].reshape(1, 3), Z[f"{role}_body"][key].reshape(21, 3),
                         Z[f"{role}_jaw"][key].reshape(1, 3), Z[f"{role}_leye"][key].reshape(1, 3),
                         Z[f"{role}_reye"][key].reshape(1, 3), Z[f"{role}_lhand"][key].reshape(15, 3),
                         Z[f"{role}_rhand"][key].reshape(15, 3)], 0).astype(float)
    pel = np.asarray(Z[f"{role}_pelvis"][key], float)
    Sw = RP.skin_mats(aa, ex["rest_joints"], ex["parents"], pel)
    pw, qw = RP.lbs_pose(ex, Sw, top4=True)
    # Unity side: local rotations mirrored (M R M), rest joints / pelvis mirrored
    Rm = RP.rotmats(aa)
    Mm = np.diag(MIRROR)
    Ru = Mm @ Rm @ Mm
    from scipy.spatial.transform import Rotation as Rot
    aa_u = Rot.from_matrix(Ru).as_rotvec()
    Ju = np.asarray(js["rest_joints"], float)
    Su = RP.skin_mats(aa_u, Ju, np.asarray(js["parents"]), pel * MIRROR)
    qx = rec["rotation"].astype(np.float64)
    exu = {"mean0": rec["mean"].astype(np.float64), "top4_idx": rec["joints"].astype(int),
           "top4_w": rec["weights"].astype(np.float64), "rot0": np.concatenate([qx[:, 3:], qx[:, :3]], 1)}
    pu, qu = RP.lbs_pose(exu, Su, top4=True)
    dpos = np.linalg.norm(pu - pw * MIRROR, axis=1) * 1e3
    qw_u = np.concatenate([qw[:, :1], -qw[:, 1:2], -qw[:, 2:3], qw[:, 3:]], 1)  # wxyz mirror: (w, -x, -y, z)
    dang = RP.ang_deg(qu, qw_u)
    sha_ok = _sha256(dest / f"{role}.splatav") == js["data_sha256"]
    return {"header": h, "sha256_ok": bool(sha_ok), "key": int(key), "pos_mm_max": round(float(dpos.max()), 4),
            "rot_deg_max": round(float(dang.max()), 4), "weights_sum_err_max": float(np.abs(rec["weights"].sum(1) - 1).max())}


def _over(rgba: np.ndarray, bg) -> np.ndarray:
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return (rgba[..., :3].astype(np.float32) * a + np.asarray(bg, np.float32) * (1 - a) + 0.5).astype(np.uint8)


def _film_time(tb: float, segs: list[dict]) -> float | None:
    for s in segs:
        d0, d1 = s["dance"]
        if d0 - 1e-6 <= tb <= d1 + 1e-6 and s.get("film"):
            return float(s["film"][0] + (tb - d0) / float(s.get("speed") or 1.0))
    return None


def _fit_into(img: np.ndarray, W: int, H: int, bg: int = 30) -> np.ndarray:
    """img scaled into a W x H panel: a plain resize when the aspect ratios agree within 1 %, else letterboxed on bg."""
    h, w = img.shape[:2]
    if abs((w / h) / (W / H) - 1.0) < 0.01:
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    s = min(W / w, H / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    out = np.full((H, W, 3), bg, np.uint8)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def a2_sheet(vj: dict, a2: dict) -> Path:
    tiles_rows = []
    by = {(v["group"], v["j"]): v for v in a2["views"]}
    for g in [g for g in vj["groups"] if g.startswith("a2_")]:
        cam = vj["groups"][g]["cam"]
        fr = vj["groups"][g]["frames"]
        tmp = P.out / "a2_real"
        tmp.mkdir(exist_ok=True)
        names = [f"{cam}_{f['i']:05d}" for f in fr]
        missing = [(f, n) for f, n in zip(fr, names) if not (tmp / f"{n}.jpg").exists()]
        if missing:
            decode_frames(cam, [f["pts"] for f, _ in missing], tmp, [n for _, n in missing])
        for j, (f, n) in enumerate(zip(fr, names)):
            real = cv2.imread(str(tmp / f"{n}.jpg"))
            lh = cv2.imread(str(P.pod_out / "a2" / f"{g}_{j}_lhmpp.png"), cv2.IMREAD_UNCHANGED)
            ou = cv2.imread(str(P.pod_out / "a2" / f"{g}_{j}_ours4.png"), cv2.IMREAD_UNCHANGED)
            df = cv2.imread(str(P.pod_out / "a2" / f"{g}_{j}_diff.png"))
            if real is None or lh is None or ou is None:
                continue
            al = np.maximum(lh[..., 3], ou[..., 3]) > 8
            ys, xs = np.nonzero(al)
            if not len(xs):
                continue
            h_ = ys.max() - ys.min()
            x0, y0 = max(0, xs.min() - int(0.08 * h_)), max(0, ys.min() - int(0.06 * h_))
            x1, y1 = min(real.shape[1], xs.max() + int(0.08 * h_)), min(real.shape[0], ys.max() + int(0.06 * h_))
            v = by.get((g, j), {})
            panels = [real, _over(lh, (128, 128, 128)), _over(ou, (128, 128, 128)), df]
            labs = [f"real ph{cam} t{f['tb']:.2f}", "LHM++ own animation", "ours: .splatav LBS (4 joints)",
                    f"|diff| x4  PSNR {v.get('psnr_ours4', 0):.1f} / in {v.get('psnr_ours4_inside', 0):.1f} dB"]
            row = []
            for P_, lab in zip(panels, labs):
                c = P_[y0:y1, x0:x1]
                s = 420 / max(1, c.shape[0])
                c = cv2.resize(c, (max(1, int(c.shape[1] * s)), 420), interpolation=cv2.INTER_AREA)
                row.append(_label(c, [lab]))
            r_ = np.concatenate([np.concatenate([t, np.full((420, 4, 3), 255, np.uint8)], 1) for t in row], 1)
            tiles_rows.append(r_)
    Wm = max(r.shape[1] for r in tiles_rows)
    sheet = np.concatenate([np.pad(r, ((0, 4), (0, Wm - r.shape[1]), (0, 0)), constant_values=255) for r in tiles_rows], 0)
    p = P.out / "a2_sheet.jpg"
    cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return p


def preview(vj: dict) -> dict:
    """preview/preview_<cam>.mp4 (3 panels of the preview group's size, 30 fps, silent): real camera | the avatar film at
    the same dance time (P.film, film/direction.json film = dance mapping) | LHM++ splats on grey."""
    P.preview_dir.mkdir(parents=True, exist_ok=True)
    dirn = P.work / "film" / "direction.json"
    segs = json.loads(dirn.read_text())["segments"] if dirn.exists() else []
    out = {}
    film_frames = None
    tmpf = P.preview_dir / "_film"
    FILM = P.film_path
    if FILM is not None and FILM.exists() and segs:
        ft = [_film_time(P.preview_window[0], segs), _film_time(P.preview_window[1] - 1e-3, segs)]
        if None not in ft:
            from ..ingest import decode_range
            if tmpf.exists():
                shutil.rmtree(tmpf)
            film_frames = decode_range(FILM, ft[0] - 0.1, ft[1] + 0.1, tmpf, quality=92)
    for g in [g for g in vj["groups"] if g.startswith("prev_")]:
        cam = vj["groups"][g]["cam"]
        fr = vj["groups"][g]["frames"]
        tmp = P.preview_dir / f"_real_{cam}"
        from ..ingest import decode_range
        from ..take import Take
        take = Take.load(PACKAGE_ROOT / "takes" / f"{P.take}.toml")
        if tmp.exists():
            shutil.rmtree(tmp)
        recs = decode_range(take.source(cam).path, fr[0]["pts"] - 0.05, fr[-1]["pts"] + 0.05, tmp, quality=92)
        rp = np.array([r["pts"] for r in recs])
        W2, H2 = int(vj["groups"][g]["W"]), int(vj["groups"][g]["H"])
        He = H2 + H2 % 2                              # yuv420p needs an even height (a 906-px camera at half size: 453)
        mp4 = P.preview_dir / f"preview_{cam}.mp4"
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{3 * W2}x{He}",
                                 "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(mp4)],
                                stdin=subprocess.PIPE)
        stills = []
        n_film = 0
        for k, f in enumerate(fr):
            real = cv2.imread(str(tmp / recs[int(np.argmin(np.abs(rp - f["pts"])))]["file"]))
            real = cv2.resize(real, (W2, H2), interpolation=cv2.INTER_AREA)
            sp = cv2.imread(str(P.out / "renders" / g / f"{f['i']:05d}.png"), cv2.IMREAD_UNCHANGED)
            spl = _over(sp, (118, 118, 118)) if sp is not None else np.zeros((H2, W2, 3), np.uint8)
            film = np.full((H2, W2, 3), 30, np.uint8)
            flab = "avatar film: not available"
            if film_frames:
                ftt = _film_time(f["tb"], segs)
                if ftt is not None:
                    fp = np.array([r["pts"] for r in film_frames])
                    j = int(np.argmin(np.abs(fp - ftt)))
                    if abs(fp[j] - ftt) < 0.05:
                        im = cv2.imread(str(tmpf / film_frames[j]["file"]))
                        film = _fit_into(im, W2, H2)
                        flab = f"avatar film {ftt:.2f} s (other camera)"
                        n_film += 1
            panel = np.concatenate([_label(real, [f"real phone {cam}", f"t {f['tb']:.2f} s"]), _label(film, [flab]),
                                    _label(spl, [P.preview_label, "same phone camera"])], 1)
            if He != H2:
                panel = np.pad(panel, ((0, He - H2), (0, 0), (0, 0)))
            proc.stdin.write(panel.tobytes())
            if k in (0, len(fr) // 3, 2 * len(fr) // 3, len(fr) - 1):
                stills.append(panel)
        proc.stdin.close()
        proc.wait()
        cv2.imwrite(str(P.preview_dir / f"preview_{cam}_stills.jpg"), np.concatenate(stills, 0), [cv2.IMWRITE_JPEG_QUALITY, 86])
        shutil.rmtree(tmp, ignore_errors=True)
        out[cam] = {"mp4": _ws(mp4), "frames": len(fr), "film_frames": n_film, "size": [3 * W2, He], "fps": 30}
        log(f"preview {cam}: {len(fr)} frames, {n_film} with the film -> {mp4}")
    shutil.rmtree(tmpf, ignore_errors=True)
    return out


def turntable_sheet() -> Path | None:
    rows = []
    for role in ROLES:
        for name in ("zero", "apose"):
            im = cv2.imread(str(P.pod_out / "turntable" / f"{role}_{name}.png"))
            if im is not None:
                rows.append(_label(cv2.resize(im, (im.shape[1] * 2 // 3, im.shape[0] * 2 // 3)), [f"{P.names[role]} {name}"]))
    if not rows:
        return None
    p = P.out / "turntables.jpg"
    cv2.imwrite(str(p), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return p


def import_run(sid: str | None = None, skip_preview: bool = False) -> dict:
    pod = json.loads((P.pod_out / "recap_lhmpp_pod.json").read_text())
    a2 = json.loads((P.pod_out / "a2.json").read_text())
    refs = json.loads((P.out / "refs.json").read_text())
    vj = json.loads((P.input / "views.json").read_text())
    with np.load(P.input / "poses.npz") as z:
        Z = {k: z[k] for k in z.files}
    rep = {"session": sid, "take": P.take, "capture": P.capture,
           "pod": {k: pod.get(k) for k in ("commit", "model", "versions", "renderer", "timings", "errors",
                                           "peak_gpu_gb", "written")},
           "dancers": {}, "a2": a2.get("summary")}
    UNITY_DIR = P.unity_dir
    for role in ROLES:
        with np.load(P.pod_out / "export" / f"{role}_export.npz") as z:
            ex = {k: z[k] for k in z.files}
        hair, hinfo = export_hair(role, ex)
        hair_sheet(ex, hair, P.out / f"hair_{role}.jpg")
        js = write_splatav(role, ex, hair, hinfo, pod, a2, refs, UNITY_DIR)
        for f in (f"{role}.splatav", f"{role}.json"):
            shutil.copy2(UNITY_DIR / f, P.out / f)
        a2key = int(np.load(P.input / "views.npz")[f"a2_{P.a2_cams[0]}_key"][0])
        chk = file_selfcheck(role, UNITY_DIR, Z, a2key, ex)
        js["checks"]["file_selfcheck"] = chk
        (UNITY_DIR / f"{role}.json").write_text(json.dumps(js, indent=1))
        shutil.copy2(UNITY_DIR / f"{role}.json", P.out / f"{role}.json")
        rep["dancers"][role] = {"count": js["count"], "hair_count": js["hair_count"], "hair": hinfo,
                                "truncation": js["skinning"]["truncation"], "selfcheck": chk,
                                "rest_joints_vs_avatar_skin_mm": (js["checks"]["rest_joints_vs_avatar_skin_mm"] or {}).get("max_mm"),
                                "splatav": _ws(UNITY_DIR / f"{role}.splatav"), "bytes": js["data_bytes"]}
        log(f"{role}: {js['count']} Gaussians, hair {js['hair_count']}, self-check {chk}")
    ign = subprocess.run(["git", "-C", str(WORKSPACE / "head-movement"), "check-ignore", "-v",
                          *[str(UNITY_DIR / f"{r}.{e}") for r in ROLES for e in ("splatav", "json")]],
                         capture_output=True, text=True)
    rep["git_ignored"] = {"head_movement": ign.stdout.strip().splitlines(), "rc": ign.returncode}
    ign2 = subprocess.run(["git", "-C", str(PACKAGE_ROOT), "check-ignore", "-v", str(P.out / "lead.splatav")],
                          capture_output=True, text=True)
    rep["git_ignored"]["dancecap"] = ign2.stdout.strip()
    rep["a2_sheet"] = _ws(a2_sheet(vj, a2))
    tt = turntable_sheet()
    rep["turntables"] = _ws(tt) if tt else None
    if not skip_preview:
        rep["preview"] = preview(vj)
    (P.out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"report -> {P.out / 'report.json'}")
    return rep


# ================================================================================================ Jev (advisory)
def _dancer_text(role: str) -> str:
    import tomllib
    d = (tomllib.loads((PACKAGE_ROOT / "takes" / f"{P.take}.toml").read_text(encoding="utf-8")).get("dancers") or {}).get(role) or {}
    return str(d.get("description") or role)


def _jev_client(client=None):
    if client is None:
        from ..judgments import client as _client
        client = _client()
    return client


def jev_crops(client=None) -> dict:
    """Jev (TypeSafe System One with images; owner consent 2026-10-19) on every picked LHM++ crop (input/refs/<role>/NN.png),
    next to the white-patch test: P(yes) of "exactly one whole person, head to feet, not cut off, not overlapped by another
    person", "the face is visible" and "this is the dancer described in the take TOML" -> lhmpp/jev_crops.json.
    Advisory: a low value is a reason to LOOK at that crop again, never a pick by itself."""
    from typesafe_sdk import Noul
    from .judge import IMAGE_HEADERS, MODEL, _asset, prepare_image
    client = _jev_client(client)
    refs = json.loads((P.out / "refs.json").read_text())
    about = ("`crop` is a cut-out of one dancer from a frame of a dance competition video (white background outside the "
             "person's mask), the input of a 3D avatar model that needs exactly one whole person.")
    out = {"model": MODEL, "rows": [], "tokens": 0}
    for role in ROLES:
        desc = _dancer_text(role)
        qs = {"one_whole_person": Noul(
                  instructions="Does `crop` show exactly one whole person from head to feet, not cut off and not overlapped "
                               "by another person?",
                  criteria={"true": "One person; the head and both feet are inside the picture; no part of a second "
                                    "person (an arm, a hand, a sleeve, hair) is in the cut-out.",
                            "false": "The head or the feet are cut off, a large part of the body is missing, or a part of "
                                     "another person is inside the cut-out."}),
              "face_visible": Noul(instructions="Is the person's face visible in `crop` (not hidden by hair, an arm or a "
                                                "hand, and not turned fully away)?"),
              "is_described_dancer": Noul(instructions=f"Is the person in `crop` this dancer: {desc}?")}
        for x in refs["roles"][role]["picks"]:
            img = P.input / "refs" / role / f"{x['slot']:02d}.png"
            k, a = _asset(prepare_image(img))
            r = client.system_one(state={"about": about, "crop": k}, questions=qs, model=MODEL,
                                  extra_headers=IMAGE_HEADERS, extra_body={"assets": {k: a}})
            row = {"role": role, "slot": x["slot"], "id": x["id"], "yaw_deg": x["yaw_deg"],
                   "white_frac": x.get("white_frac"), **{q: round(float(v.noul), 3) for q, v in r.answers.items()}}
            out["tokens"] += int(getattr(r.usage, "input_tokens", 0) or 0)
            out["rows"].append(row)
            log(f"jev {role} {x['slot']:02d} {x['id']}: " + ", ".join(f"{q} {row[q]}" for q in qs))
    (P.out / "jev_crops.json").write_text(json.dumps(out, indent=1))
    return out


def jev_turntables(client=None) -> dict:
    """Jev on the built avatars: each turntable render (pod_out/turntable/<role>_{zero,apose}.png) against the crop 00 and
    the most-backward crop of the same dancer: P(yes) "same clothes", "same person", "matches the TOML description",
    "render defects" -> lhmpp/jev_turntables.json. Advisory, next to the LOOK at turntables.jpg."""
    from typesafe_sdk import Noul
    from .judge import IMAGE_HEADERS, MODEL, _asset, prepare_image
    client = _jev_client(client)
    refs = json.loads((P.out / "refs.json").read_text())
    out = {"model": MODEL, "rows": [], "tokens": 0}
    about = ("`render` is a turntable of a 3D Gaussian-splat avatar (several views of one figure in a T-pose or an A-pose on "
             "a plain background) built from photos of a real dancer; `photo_front` and `photo_back` are two of those input "
             "photos (cut-outs on white).")
    for role in ROLES:
        desc = _dancer_text(role)
        picks = refs["roles"][role]["picks"]
        back = max(picks, key=lambda x: abs(x["yaw_deg"]))
        qs = {"same_clothes": Noul(instructions="Does the figure in `render` wear the same clothes as the person in "
                                                "`photo_front` / `photo_back` (same garments, colours and lengths)?"),
              "same_person": Noul(instructions="Does the figure in `render` look like the same person as in the photos "
                                               "(skin tone, hair, build)?"),
              "matches_description": Noul(instructions=f"Does the figure in `render` match this description: {desc}?"),
              "render_defects": Noul(instructions="Are there rendering defects on the figure in `render`: floating "
                                                  "blobs, holes through the body, smeared or missing parts?")}
        for name in ("zero", "apose"):
            tt = P.pod_out / "turntable" / f"{role}_{name}.png"
            if not tt.exists():
                continue
            state, assets = {"about": about}, {}
            for key, path in (("render", tt), ("photo_front", P.input / "refs" / role / "00.png"),
                              ("photo_back", P.input / "refs" / role / f"{back['slot']:02d}.png")):
                k, a = _asset(prepare_image(path))
                state[key], assets[k] = k, a
            r = client.system_one(state=state, questions=qs, model=MODEL, extra_headers=IMAGE_HEADERS,
                                  extra_body={"assets": assets})
            row = {"role": role, "turntable": name, **{q: round(float(v.noul), 3) for q, v in r.answers.items()}}
            out["tokens"] += int(getattr(r.usage, "input_tokens", 0) or 0)
            out["rows"].append(row)
            log(f"jev turntable {role} {name}: " + ", ".join(f"{q} {row[q]}" for q in qs))
    (P.out / "jev_turntables.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.recap_splatav")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--take", default=None, help="take profile (default: the class recap, larissa_kadu_recap)")
    common.add_argument("--out-dir", default=None, help="write work/<take>/lhmpp outputs here instead (regression runs)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("survey", parents=[common])
    sub.add_parser("frames", parents=[common])
    sub.add_parser("sam-job", parents=[common])
    sub.add_parser("select", parents=[common])
    pk = sub.add_parser("pick", parents=[common])
    pk.add_argument("--no-colour", action="store_true")
    pk.add_argument("--colour", default=None, help="none | ref:<cam> | wb | wb_common | auto (default: the profile's)")
    jv = sub.add_parser("jev", parents=[common])
    jv.add_argument("--what", choices=("crops", "turntables"), default="crops")
    jb = sub.add_parser("job", parents=[common])
    jb.add_argument("--max-minutes", type=float, default=40.0)
    im = sub.add_parser("import", parents=[common])
    im.add_argument("--session", default=None)
    im.add_argument("--no-preview", action="store_true")
    a = ap.parse_args(argv)
    use(a.take, **({"out_dir": a.out_dir} if a.out_dir else {}))
    if a.cmd == "survey":
        survey()
    elif a.cmd == "frames":
        frames()
    elif a.cmd == "sam-job":
        sam_job()
    elif a.cmd == "select":
        select()
    elif a.cmd == "pick":
        pick(colour=False if a.no_colour else (a.colour or True))
    if a.cmd == "jev":
        if a.what == "crops":
            jev_crops()
        else:
            jev_turntables()
    if a.cmd == "job":
        job(a.max_minutes)
    if a.cmd == "import":
        import_run(a.session, a.no_preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
