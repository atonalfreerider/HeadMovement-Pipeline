r"""mono4d track "closeapp" (2026-10-19): CloseApp on ONE handheld phone (06) of larissa_kadu_38, 58-62 s.

CloseApp = "Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning" (Huang et al., CVPR 2025),
github.com/boycehbz/CloseApp @ 516ed6f (2026-07-27), MIT code; SMPL body model (non-commercial, gated, the owner's
download in models/smpl + models/smplify). Its demo is a per-video OPTIMISATION of two SMPL bodies:
    train.py -s data/preprocess_data/<seq> -m output/<seq> --train_stage=1 --save_render --use_appearance --save_params
model/avatar_model.AvatarModel.optimization() minimises, over all frames of the clip, 0.5 reprojection (Halpe-26 joints
regressed from the SMPL mesh vs the 2D keypoints, conf < 0.7 ignored) + 1.0 rotation prior to the initial poses + 0.5
shape prior + 0.1 smoothness (2nd + 0.2 x 1st order) + 0.0 x penetration (BVH distance field: computed, weight 0 in the
release) [+ 0.5 contact, only with contact labels], plus - with --use_appearance - the photometric branch: one 3D-Gaussian
avatar per person (POP-style UV U-Net) rasterised over 16-frame chunks against the masked frames (L1 + D-SSIM), whose
gradients reach the poses through the skinning; 20 epochs, SparseAdam lr 5e-3 on the per-frame poses.

What the RELEASE does not do (read from the code, not run): the proxemics diffusion prior (the paper's "motion branch",
data/best_reconstruction.pkl) is loaded but used only with --use_prior, which the demo command omits and which cannot run
as released (optimization() passes an undefined `step`, and with the prior the 16-frame batch is mixed with the
all-frame smoothness / init-prior terms: shape mismatch). The penetration weight is 0.000. So what is tested here is the
released demo configuration: keypoints + init prior + smoothness + 3D-Gaussian appearance.

Inputs (feeding OUR data instead of its wild preprocessing, whose YOLOX / HMR2 / SAMURAI / CLIFF / ViTPose / ScoreHMR
weights are on Baidu Netdisk only):
    images     the 120 canonical frames of phone 06 inside the window (1080 x 1920 portrait, unchanged; SIMPLE_RADIAL
               k1 = -0.0023 -> < 1 px where the dancers are, < 2.5 px in the corners: not undistorted)
    masks      our SAM 2.1 dancer masks (mask_<frame>_01 = lead -> CloseApp person 0, _02 = follow -> person 1)
    keypoints  COCO-17 rows of people/06.npz matched to the roles.json lead / follow boxes (max IoU assignment) as the
               first 17 Halpe-26 joints (same definitions); Halpe 17-25 (head top, neck, mid-hip, toes, heels) conf 0
    camera     static identity extrinsics (CloseApp's own camera model: a fixed camera), our focal 1147.09 px, principal
               point = image centre (ours is 539.5 / 959.5, CloseApp uses w/2, h/2: 0.5 px); so the result is in each
               frame's phone-06 camera frame (OpenCV axes) and goes to the world through the phone's per-frame track
    init       fit B (candidate mamma_mono06: MAMMA single-camera SMPL-X on phone 06 only) converted to SMPL by
               surface_fit(): its posed SMPL-X meshes moved into each frame's camera frame, SMPL-X body pose copied
               joint-for-joint as the start (SMPL joints 1-21 = SMPL-X body joints 1-21, SMPL hands 22/23 = 0), then a
               part-restricted point-to-point ICP of the SMPLify-neutral SMPL surface onto fit B's surface (shared 10
               betas, per-frame pose + translation; 2026-10-19: nearest-surface distance median 8.3 / 6.8 mm lead /
               follow). No official SMPL-X -> SMPL converter was used: smplx/transfer_model needs SMPL-X's
               deformation-transfer file, which we do not hold.
               features (the prior's HMR2 image features) = 0: unused without --use_prior. contact = None (as the
               official wild pipeline when no annotation exists).
Variants (one pod session): closeapp = the release demo config (appearance on); closeapp_noapp = the same without the
appearance branch (the paper's "w/o appearance" ablation); closeapp_init = the converted fit B before any CloseApp
step (what CloseApp started from: the yardstick for what CloseApp itself changed).
Joints: SMPL body order = SMPL-X body order (0 pelvis ... 21 right wrist), but the joint DEFINITIONS differ a lot: on
the surface-matched bodies SMPL's FK joints sit a median 19-20 mm from MAMMA SMPL-X's (neck / collars / head 37-60 mm).
The candidates therefore carry SMPL FK joints + a joint-definition adapter: per joint a constant offset in the frame of
the segment it sits on, calibrated once on the init pair (fit B joints vs the surface-fitted SMPL; residual median
3.2 / 3.1 mm) and applied unchanged to every CloseApp output (adapt()). Scores with the raw SMPL joints (no adapter) are
reported too (closeapp_summary.json diagnostics.raw_smpl_joints).
Clock: candidate t = phone-06 frame time + the take's sync delta (-30.04 ms) = the reference fit's own instants of these
observations (the harness matches by nearest time within 1/60 s and picks the camera track by nearest corrected time).

    python -m dancecap.mono4d.cand_closeapp smpl     # models/smpl/nochumpy_closeapp/SMPL_{NEUTRAL,MALE,FEMALE}.pkl
    python -m dancecap.mono4d.cand_closeapp prep     # frames + masks + keypoints + fit-B->SMPL init -> mono4d/closeapp/input/
    python -m dancecap.mono4d.cand_closeapp job      # dancecap/remote/cand_closeapp/pod_job.json
    (DANCECAP_RUNPOD_CAP=... python -m dancecap remote run --jobs <abs path of that json> --max-cost 5 --gpus ...)
    python -m dancecap.mono4d.cand_closeapp import [--session SID]   # candidates + cand-score + interaction metrics
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE
from . import candidates as C

TAKE, CAM = "larissa_kadu_38", "06"
WINDOW = (58.0, 62.0)
WORK = PACKAGE_ROOT / "work" / TAKE
OUT = WORK / "mono4d" / "closeapp"
INPUT = OUT / "input"
POD_OUT = OUT / "pod_out"
JOB_DIR = PACKAGE_ROOT / "remote" / "cand_closeapp"
FITB = PACKAGE_ROOT / "work" / "larissa_kadu_38_mono06" / "mamma" / "fits" / "m06" / "smplx"
SMPL_PLAIN = WORKSPACE / "models" / "smpl" / "nochumpy_closeapp"
SMPL_SRC = {"NEUTRAL": WORKSPACE / "models" / "smplify" / "smplify_public" / "code" / "models"
                       / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
            "MALE": WORKSPACE / "models" / "smpl" / "SMPL_python_v.1.1.0" / "smpl" / "models"
                    / "basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
            "FEMALE": WORKSPACE / "models" / "smpl" / "SMPL_python_v.1.1.0" / "smpl" / "models"
                      / "basicmodel_f_lbs_10_207_0_v1.1.0.pkl"}
CLOSEAPP_COMMIT = "516ed6f8f8bd7bf5a358ce762e51845227d8aa85"
DGR_COMMIT = "59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d"      # graphdeco-inria/diff-gaussian-rasterization (glm 5c46b9c)
SDF_COMMIT = "8016aef2e8cd56f1b9d5481ddc0036543ad984c7"      # penincillin/SDF_ihmr
ISECT_COMMIT = "85b30177821a1527e3fe62fcf8ce65262d7c1879"    # vchoutas/torch-mesh-isect
DATA_GDRIVE_ID = "1kTMD6t-GSJ4tHnC2R19RD7dyMWMXwqsv"         # README "Download data ... Google Drive": data.zip
DATA_BYTES = 228421916
ROLES = ("lead", "follow")
SEQ = "lk38_06"
VARIANTS = ("closeapp", "closeapp_noapp")
LICENCE = ("CloseApp code: MIT (boycehbz/CloseApp); its data.zip (proxemics prior, UV maps, Halpe regressor) has no "
           "stated licence; SMPL / SMPLify body models: MPI non-commercial research licence (gated, never "
           "redistributed); diff-gaussian-rasterization: Inria/MPII non-commercial research licence; SDF_ihmr: MIT; "
           "torch-mesh-isect: MPI non-commercial. Research use only; outputs stay local.")
HALPE = 26
PEN_MAX_M = 0.15   # penetration test only for vertices within 15 cm of the partner surface (nearest-normal sign test)


def log(msg: str) -> None:
    print(f"[cand_closeapp {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ws(p: Path) -> str:
    return str(Path(p).resolve().relative_to(WORKSPACE.resolve())).replace("\\", "/")


# ------------------------------------------------------------------------------------------------ SMPL files


def convert_smpl() -> list[Path]:
    """The owner's chumpy SMPL pickles -> plain numpy/scipy pickles CloseApp's two loaders read (its SMPLModel needs a
    scipy-sparse J_regressor and 10-PC shapedirs; smplx.SMPL truncates by itself). NEUTRAL = SMPLify's
    basicModel_neutral v1.0.0 (what CloseApp's README asks for: 'SMPL and SMPLify website'), MALE / FEMALE = SMPL
    v1.1.0 (300 PCs, the first 10 kept: CloseApp only loads them for its GT evaluation, which we do not have)."""
    from ..hand_mano_worker import _Stub, load_mano_plain
    SMPL_PLAIN.mkdir(parents=True, exist_ok=True)
    done = []
    for g, src in SMPL_SRC.items():
        if not src.exists():
            raise SystemExit(f"missing {src} (owner download)")
        d = load_mano_plain(src)
        d.pop("pose_training_info", None)
        bad = [k for k, v in d.items() if isinstance(v, _Stub)]
        if bad:
            raise ValueError(f"{src}: unconverted keys {bad}")
        d["shapedirs"] = np.asarray(d["shapedirs"])[:, :, :10]
        for k in ("v_template", "shapedirs", "posedirs", "weights", "weights_prior", "J"):
            if k in d:
                d[k] = np.ascontiguousarray(np.asarray(d[k]), dtype=np.float32)
        dst = SMPL_PLAIN / f"SMPL_{g}.pkl"
        with open(dst, "wb") as fh:
            pickle.dump(d, fh, protocol=4)
        done.append(dst)
    (SMPL_PLAIN / "README.txt").write_text(
        "SMPL model files converted to plain numpy (no chumpy, 10 shape PCs, float32) by dancecap.mono4d.cand_closeapp smpl\n"
        "for the CloseApp pod run. NEUTRAL from SMPLify (basicModel_neutral_lbs_10_207_0_v1.0.0), MALE/FEMALE from SMPL\n"
        "v1.1.0. Derived from the owner's licensed downloads (smpl.is.tue.mpg.de, smplify.is.tue.mpg.de): non-commercial\n"
        "research only, never commit, never copy anywhere public; only our own Runpod pods receive them.\n")
    return done


def load_smpl_plain(gender: str = "NEUTRAL") -> dict:
    with open(SMPL_PLAIN / f"SMPL_{gender}.pkl", "rb") as fh:
        return pickle.load(fh, encoding="latin1")


# ------------------------------------------------------------------------------------------------ SMPL FK / IK


def _rodrigues(aa):
    import torch
    th = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / th
    K = torch.zeros(*aa.shape[:-1], 3, 3, dtype=aa.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, dtype=aa.dtype) + s * K + (1 - c) * (K @ K)


class SmplJoints:
    """SMPL FK joints (24) from axis-angle pose, betas, transl (torch, CPU float64)."""

    def __init__(self, model: dict):
        import torch
        Jr = np.asarray(model["J_regressor"].todense(), float)
        self.Jt = torch.tensor(Jr @ np.asarray(model["v_template"], float))                       # (24, 3)
        self.Jd = torch.tensor(np.einsum("jv,vcb->jcb", Jr, np.asarray(model["shapedirs"], float)[:, :, :10]))
        self.parents = [int(p) for p in np.asarray(model["kintree_table"])[0]]
        self.parents[0] = -1

    def __call__(self, pose, betas, transl, rotations: bool = False):
        import torch
        B = pose.shape[0]
        J = self.Jt[None] + torch.einsum("jcb,nb->njc", self.Jd, betas.expand(B, -1))
        R = _rodrigues(pose.reshape(B, 24, 3))
        Rg, pos = [R[:, 0]], [J[:, 0]]
        for i in range(1, 24):
            p = self.parents[i]
            pos.append(pos[p] + torch.einsum("nij,nj->ni", Rg[p], J[:, i] - J[:, p]))
            Rg.append(Rg[p] @ R[:, i])
        out = torch.stack(pos, 1) + transl[:, None]
        return (out, torch.stack(Rg, 1)) if rotations else out

    def seg_rot(self, Rg):
        """(B,24,3,3) global joint rotations -> the rotation of the segment each joint 0-21 sits on (its parent's;
        the pelvis its own)."""
        return Rg[:, [0] + self.parents[1:22]]


def adapt(fk: SmplJoints, pose, betas, transl, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SMPL FK joints 0-21 (T,22,3) and the same joints re-defined as MAMMA SMPL-X joints: + the segment-local offsets
    calibrated in surface_fit (J'_j = J_j + R_seg(j) o_j)."""
    import torch
    t = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
    with torch.no_grad():
        J, Rg = fk(t(pose), t(betas).reshape(1, -1), t(transl), rotations=True)
        Ja = J[:, :22] + torch.einsum("njab,jb->nja", fk.seg_rot(Rg), t(offsets))
    return J[:, :22].numpy(), Ja.numpy()


def surface_fit(Vx: np.ndarray, x_weights: np.ndarray, joints_x: np.ndarray, pose_copy: np.ndarray, model: dict,
                rounds: int = 10, steps: int = 60) -> dict:
    """fit B's SMPL-X SURFACE (T,10475,3, camera frame) -> SMPL (pose (T,72), transl (T,3), shared betas (10,)).

    Part-restricted point-to-point ICP: every round each SMPL body vertex of part j (skinning-weight argmax, body parts
    0-19; wrists, hands, fingers, eyes left out) is matched to the nearest fit-B surface vertex of the SAME part (SMPL and
    SMPL-X share joints 0-21; SMPL-X's jaw counts as head), then `steps` Adam steps on pose + betas + transl. This is what SMPL-X's
    transfer_model does with its fixed SMPL<->SMPL-X correspondences (which need SMPL-X's deformation-transfer file,
    not held here), with nearest-neighbour correspondences instead. Afterwards the joint-DEFINITION adapter: per joint
    a constant offset in the frame of the segment it sits on, o_j = median_t R_seg(j)^T (X_j - J_j) between fit B's
    joints X and the fitted SMPL's FK joints J (the surface fixes the segments, so the offsets are definitional, not
    pose bias). adapt() applies the same offsets to every CloseApp output."""
    import smplx
    import torch
    from scipy.spatial import cKDTree
    torch.manual_seed(0)
    T = len(Vx)
    m = smplx.SMPL(model_path=str(SMPL_PLAIN / "SMPL_NEUTRAL.pkl"), batch_size=T)
    wa = np.argmax(np.asarray(model["weights"]), axis=1)
    xa = np.argmax(x_weights, axis=1)
    xa = np.where(xa == 22, 15, xa)                                              # SMPL-X jaw belongs to the SMPL head
    parts = list(range(20))                                                      # body parts 0-19 (SMPL = SMPL-X joints)
    src = {j: np.nonzero(wa == j)[0] for j in parts}                             # wrists / hands / eyes / fingers left out
    tgt = {j: np.nonzero(xa == j)[0] for j in parts}
    fk = SmplJoints(model)
    p0 = torch.tensor(pose_copy, dtype=torch.float32)
    pose = p0.clone().requires_grad_(True)
    betas = torch.zeros(1, 10, requires_grad=True)
    X = torch.tensor(joints_x, dtype=torch.float32)
    with torch.no_grad():
        o = m(betas=betas.expand(T, -1), global_orient=p0[:, :3], body_pose=p0[:, 3:])
        transl0 = X[:, 0] - o.joints[:, 0]
    transl = transl0.clone().requires_grad_(True)
    opt = torch.optim.Adam([pose, betas, transl], lr=5e-3)
    trees = [{j: cKDTree(Vx[k][tgt[j]]) for j in parts} for k in range(T)]
    src_all = np.concatenate([src[j] for j in parts])

    def match(v):
        nn = np.zeros((T, len(src_all)), np.int64)
        dist = np.zeros((T, len(src_all)))
        for k in range(T):
            o = 0
            for j in parts:
                dd, ii = trees[k][j].query(v[k][src[j]])
                nn[k, o:o + len(ii)], dist[k, o:o + len(ii)] = tgt[j][ii], dd
                o += len(ii)
        return nn, dist

    hist = []
    for r in range(rounds):
        with torch.no_grad():
            v = m(betas=betas.expand(T, -1), global_orient=pose[:, :3], body_pose=pose[:, 3:], transl=transl).vertices.numpy()
        nn, _ = match(v)
        Y = torch.tensor(np.take_along_axis(Vx, nn[..., None], axis=1), dtype=torch.float32)
        if r == rounds - 3:
            for g in opt.param_groups:
                g["lr"] = 1e-3
        for _ in range(steps):
            opt.zero_grad()
            v = m(betas=betas.expand(T, -1), global_orient=pose[:, :3], body_pose=pose[:, 3:], transl=transl).vertices
            e = ((v[:, src_all] - Y) ** 2).sum(-1).mean()
            reg = 1e-4 * ((pose[:, 3:] - p0[:, 3:]) ** 2).sum(-1).mean() + 1e-4 * (betas ** 2).sum()
            (e + reg).backward()
            opt.step()
        hist.append(float(np.sqrt(e.item()) * 1e3))
    with torch.no_grad():
        v = m(betas=betas.expand(T, -1), global_orient=pose[:, :3], body_pose=pose[:, 3:], transl=transl).vertices.numpy()
    d = match(v)[1].ravel()
    P, B, Tr = pose.detach().numpy(), betas.detach().numpy()[0], transl.detach().numpy()
    J, Rg = fk(torch.tensor(P, dtype=torch.float64), torch.tensor(B, dtype=torch.float64)[None],
               torch.tensor(Tr, dtype=torch.float64), rotations=True)
    J, Rs = J[:, :22].numpy(), fk.seg_rot(Rg).numpy()
    offs = np.median(np.einsum("njba,njb->nja", Rs, joints_x - J), axis=0)          # R^T (X - J), median over frames
    Ja = J + np.einsum("njab,jb->nja", Rs, offs)
    return {"pose": P, "transl": Tr, "betas": B, "offsets": offs, "rms_mm_per_round": [round(h, 1) for h in hist],
            "surface_nn_mm_median": float(np.median(d) * 1e3), "surface_nn_mm_p95": float(np.percentile(d, 95) * 1e3),
            "residual_mm": 1e3 * np.linalg.norm(Ja - joints_x, axis=-1), "residual_raw_mm": 1e3 * np.linalg.norm(J - joints_x, axis=-1),
            "pose_change_deg": np.degrees(np.linalg.norm((P - pose_copy).reshape(-1, 24, 3), axis=-1))}


# ------------------------------------------------------------------------------------------------ prep


def _assign_people(boxes_role: dict, p_boxes: np.ndarray, p_valid: np.ndarray) -> dict:
    from .holdout import box_iou
    cand = [j for j in range(len(p_boxes)) if p_valid[j]]
    best, best_s = {}, -1.0
    for a in cand + [None]:
        for b in cand + [None]:
            if a is not None and a == b:
                continue
            s, ok = 0.0, True
            sel = {}
            for role, j in (("lead", a), ("follow", b)):
                if j is None or boxes_role[role] is None:
                    sel[role] = (None, 0.0)
                    continue
                iou = box_iou(np.array(boxes_role[role], float), p_boxes[j])
                sel[role] = (j, iou)
                s += iou
            if ok and s > best_s:
                best, best_s = sel, s
    return best


def _halpe_from_coco(k17: np.ndarray) -> np.ndarray:
    out = np.zeros((HALPE, 3), np.float32)
    out[:17] = k17
    return out


def _bbox_center_scale(b) -> tuple[list, float]:
    """HMR2 / CloseApp bbox_from_detector: centre and scale = max(w * 256 / 192, h) / 200 * 1.1."""
    x0, y0, x1, y1 = [float(v) for v in b]
    return [(x0 + x1) / 2, (y0 + y1) / 2], max((x1 - x0) * 256 / 192.0, y1 - y0) / 200.0 * 1.1


def prep() -> dict:
    from .. import mamma_dyn as md
    if not (SMPL_PLAIN / "SMPL_NEUTRAL.pkl").exists():
        convert_smpl()
    fj = json.loads((WORK / "frames" / CAM / "frames.json").read_text())
    frames = fj["frames"]
    ft = np.array([f["t"] for f in frames])
    cam = np.load(WORK / "cameras" / f"{CAM}.npz")
    if len(cam["t"]) != len(frames) or np.abs(cam["t"] - ft).max() > 1e-6:
        raise SystemExit("cameras/06.npz and frames.json are not index-aligned")
    tr = md.load_tracks(WORK, [CAM])[CAM]
    off = md.read_sync_offsets(WORK)[CAM]
    t_all = md.apply_offset(ft, off)          # reference seconds: the window is on the reference clock
    sel = np.nonzero((t_all >= WINDOW[0]) & (t_all <= WINDOW[1]))[0]
    t_ref = t_all[sel]
    if np.abs(tr.t[sel] - t_ref).max() > 1e-6:
        raise SystemExit("camera track t != frame t + sync delta")
    roles = json.loads((WORK / "dancers" / "roles.json").read_text())["cams"][CAM]["frames"]
    ppl = np.load(WORK / "people" / f"{CAM}.npz")
    if np.abs(ppl["t"] - ft).max() > 1e-3:
        raise SystemExit("people/06.npz is not index-aligned with frames.json")
    for d in ("images", "masks/0", "masks/1"):
        (INPUT / d).mkdir(parents=True, exist_ok=True)
    T = len(sel)
    kp = np.zeros((T, 2, HALPE, 3), np.float32)
    boxes = np.zeros((T, 2, 4), np.float32)
    centers = np.zeros((T, 2, 2), np.float32)
    scales = np.zeros((T, 2), np.float32)
    match_iou = np.zeros((T, 2), np.float32)
    names = []
    for k, i in enumerate(sel):
        name = f"{k:06d}"
        names.append(name)
        shutil.copy2(WORK / "frames" / CAM / frames[i]["file"], INPUT / "images" / f"{name}.jpg")
        for a, pid in enumerate(("01", "02")):
            shutil.copy2(WORK / "dancers" / CAM / "masks" / f"mask_{i:04d}_{pid}.png", INPUT / "masks" / str(a) / f"{name}.png")
        r = roles[i]
        if r["i"] != i or abs(r["t"] - ft[i]) > 1e-3:
            raise SystemExit(f"roles.json frame {i} misaligned")
        rb = {role: (r.get(role) or {}).get("box") for role in ROLES}
        asg = _assign_people(rb, ppl["boxes"][i], ppl["valid"][i])
        for a, role in enumerate(ROLES):
            if rb[role] is None:
                raise SystemExit(f"frame {i}: no {role} box")
            boxes[k, a] = rb[role]
            centers[k, a], scales[k, a] = _bbox_center_scale(rb[role])
            j, iou = asg.get(role, (None, 0.0))
            match_iou[k, a] = iou
            if j is not None and iou >= 0.3:
                kp[k, a] = _halpe_from_coco(ppl["kpts"][i, j])
    # fit B (SMPL-X, world) -> each frame's camera frame -> SMPL IK
    model = load_smpl_plain("NEUTRAL")
    init = {"pose": np.zeros((T, 2, 72), np.float32), "transl": np.zeros((T, 2, 3), np.float32),
            "betas": np.zeros((2, 10), np.float32), "offsets": np.zeros((2, 22, 3)), "fitb_cam": np.zeros((2, T, 22, 3))}
    ik_rep = {}
    R, tv = tr.R[sel], tr.tvec[sel]
    from scipy.spatial.transform import Rotation as Rot
    for a, role in enumerate(ROLES):
        z = np.load(FITB / f"{role}.npz", allow_pickle=True)
        zt = np.asarray(z["t"], float)
        idx = np.array([int(np.argmin(np.abs(zt - x))) for x in t_ref])
        dt = np.abs(zt[idx] - t_ref)
        if dt.max() > 5e-3:
            raise SystemExit(f"fit B {role}: no sample within 5 ms of frame {int(np.argmax(dt))} (max {dt.max() * 1e3:.1f} ms)")
        Jw = np.asarray(z["joints"], float)[idx, :22]
        Jc = np.einsum("fij,fnj->fni", R, Jw) + tv[:, None, :]
        go = Rot.from_matrix(np.einsum("fij,fjk->fik", R, Rot.from_rotvec(np.asarray(z["global_orient"], float)[idx]).as_matrix()))
        pc = np.zeros((T, 72))
        pc[:, :3] = go.as_rotvec()
        pc[:, 3:66] = np.asarray(z["body_pose"], float)[idx]
        from ..avatar_uv import Body
        from ..mamma_dyn import SMPLX_MODELS
        zd = {kk: z[kk] for kk in z.files}
        zd = {kk: (v[idx] if getattr(v, "ndim", 0) and len(v) == len(zt) else v) for kk, v in zd.items()}
        body = Body(zd, SMPLX_MODELS)
        if body.fk_err_m > 1e-3:
            raise SystemExit(f"fit B {role}: SMPL-X FK misses its stored joints by {body.fk_err_m * 1e3:.1f} mm")
        Vc = np.einsum("fij,fnj->fni", R, body.verts.astype(float)) + tv[:, None, :]
        xw = np.load(SMPLX_MODELS / "smplx" / "SMPLX_NEUTRAL.npz", allow_pickle=True)["weights"]
        f = surface_fit(Vc, np.asarray(xw), Jc, pc, model)
        init["pose"][:, a], init["transl"][:, a], init["betas"][a] = f["pose"], f["transl"], f["betas"]
        init["offsets"][a], init["fitb_cam"][a] = f["offsets"], Jc
        ik_rep[role] = {"fitb_dt_ms_max": round(float(dt.max() * 1e3), 3),
                        "residual_mm_median": round(float(np.median(f["residual_mm"])), 2),
                        "residual_mm_p95": round(float(np.percentile(f["residual_mm"], 95)), 2),
                        "residual_mm_max": round(float(f["residual_mm"].max()), 2),
                        "residual_raw_smpl_joints_mm_median": round(float(np.median(f["residual_raw_mm"])), 2),
                        "residual_raw_smpl_joints_mm_per_joint": np.round(np.median(f["residual_raw_mm"], 0), 1).tolist(),
                        "residual_adapted_mm_per_joint": np.round(np.median(f["residual_mm"], 0), 1).tolist(),
                        "surface_nn_mm_median": round(f["surface_nn_mm_median"], 2),
                        "surface_nn_mm_p95": round(f["surface_nn_mm_p95"], 2), "icp_rms_mm": f["rms_mm_per_round"],
                        "offsets_mm": {j: np.round(1e3 * f["offsets"][j], 1).tolist() for j in range(22)},
                        "pose_change_deg_median_body": round(float(np.median(f["pose_change_deg"][:, 1:22])), 2),
                        "betas": np.round(f["betas"], 3).tolist(),
                        "depth_m_median": round(float(np.median(Jc[:, 0, 2])), 3)}
        log(f"{role}: fit B surface -> SMPL: surface nn {ik_rep[role]['surface_nn_mm_median']} mm; joints + adapter residual median {ik_rep[role]['residual_mm_median']} mm, raw SMPL "
            f"joints {ik_rep[role]['residual_raw_smpl_joints_mm_median']} mm "
            f"(p95 {ik_rep[role]['residual_mm_p95']}), body rotation change median {ik_rep[role]['pose_change_deg_median_body']} deg")
    K = np.median(cam["K"][sel], axis=0)
    np.savez(INPUT / "closeapp_input.npz", names=np.array(names), frame_index=sel, t_frame=ft[sel], t_ref=t_ref,
             K=K, img_w=np.int64(fj["width"]), img_h=np.int64(fj["height"]), keypoints=kp, boxes=boxes,
             centers=centers, scales=scales, init_pose=init["pose"], init_transl=init["transl"],
             init_betas=init["betas"], adapter_offsets=init["offsets"], fitb_joints_cam=init["fitb_cam"],
             match_iou=match_iou, sync_delta_s=np.float64(off["delta"]))
    conf = kp[..., :17, 2]
    rep = {"frames": T, "frame_index": [int(sel[0]), int(sel[-1])], "t_frame": [float(ft[sel[0]]), float(ft[sel[-1]])],
           "t_ref": [float(t_ref[0]), float(t_ref[-1])], "sync_delta_ms": round(off["delta"] * 1e3, 3),
           "K": np.round(K, 3).tolist(), "k1": float(np.median(cam["dist"][sel])),
           "kp_match_iou_min": np.round(match_iou.min(0), 3).tolist(),
           "kp_frac_conf_ge_0.7": {role: round(float((conf[:, a] >= 0.7).mean()), 3) for a, role in enumerate(ROLES)},
           "kp_frames_without_match": {role: int((match_iou[:, a] < 0.3).sum()) for a, role in enumerate(ROLES)},
           "ik": ik_rep, "fitb": str(FITB)}
    (INPUT / "prep.json").write_text(json.dumps(rep, indent=1))
    log(f"prep -> {INPUT}: {T} frames; keypoint conf>=0.7 {rep['kp_frac_conf_ge_0.7']}")
    return rep


# ------------------------------------------------------------------------------------------------ pod job


def write_job(max_minutes: float = 46) -> Path:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    inp = _ws(INPUT)
    job = {
        "schema": "dancecap.pod_job/1", "name": "cand_closeapp", "take": TAKE, "max_minutes": max_minutes, "needs": [],
        "inputs": [inp + "/", {"local": _ws(SMPL_PLAIN) + "/", "pod": _ws(SMPL_PLAIN) + "/"}],
        "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
        "env": {"CLOSEAPP_ROOT": "/workspace/closeapp_cand", "CLOSEAPP_COMMIT": CLOSEAPP_COMMIT, "DGR_COMMIT": DGR_COMMIT,
                "SDF_COMMIT": SDF_COMMIT, "ISECT_COMMIT": ISECT_COMMIT, "CLOSEAPP_DATA_ID": DATA_GDRIVE_ID,
                "CLOSEAPP_DATA_BYTES": str(DATA_BYTES)},
        "commands": [
            {"name": "install", "timeout_min": 20, "cmd": "bash remote/cand_closeapp/install.sh"},
            {"name": "run", "timeout_min": 25,
             "cmd": f"python remote/cand_closeapp/closeapp_pod.py run --input {inp.removeprefix('dancecap/')} "
                    f"--smpl /workspace/HeadMovement/{_ws(SMPL_PLAIN)} --out {_ws(POD_OUT).removeprefix('dancecap/')} "
                    f"--variants {','.join(VARIANTS)}"},
        ],
        "outputs": [_ws(POD_OUT) + "/"],
        "verify": [_ws(POD_OUT) + "/closeapp_pod.json", _ws(POD_OUT) + "/closeapp_out.npz"],
        "after_download": ["python -m dancecap.mono4d.cand_closeapp import --session <sid>"],
        "notes": ["CloseApp (MIT code) @ " + CLOSEAPP_COMMIT[:7] + " on the 120 canonical frames of phone 06, 58-62 s, our "
                  "masks / COCO-17 keypoints / focal, init = fit B converted to SMPL; variants " + ", ".join(VARIANTS) +
                  ". The bundle holds the owner's SMPL / SMPLify model files (plain-numpy copies): our own pod only, "
                  "deleted with it. data.zip (Google Drive, 228 MB) is fetched on the pod; nothing licensed comes back."],
    }
    p = JOB_DIR / "pod_job.json"
    p.write_text(json.dumps(job, indent=1))
    log(f"wrote {p}")
    return p


# ------------------------------------------------------------------------------------------------ import + metrics


def _smpl_layer(T: int):
    import smplx
    return smplx.SMPL(model_path=str(SMPL_PLAIN / "SMPL_NEUTRAL.pkl"), batch_size=T)


def smpl_bodies(pose: np.ndarray, transl: np.ndarray, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T,72), (T,3), (10,) -> vertices (T,6890,3), FK joints (T,24,3), faces."""
    import torch
    T = len(pose)
    m = _smpl_layer(T)
    f = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float32)  # noqa: E731
    with torch.no_grad():
        o = m(betas=f(betas).reshape(1, -1).expand(T, -1), global_orient=f(pose[:, :3]), body_pose=f(pose[:, 3:]),
              transl=f(transl))
    return o.vertices.numpy(), o.joints.numpy()[:, :24], np.asarray(m.faces, np.int64)


def _hand_sets(weights: np.ndarray, left: list[int], right: list[int]) -> dict:
    a = np.argmax(weights, axis=1)
    return {"L": np.nonzero(np.isin(a, left))[0], "R": np.nonzero(np.isin(a, right))[0]}


def _normals(v: np.ndarray, faces: np.ndarray) -> np.ndarray:
    from ..avatar_uv import vertex_normals
    return vertex_normals(v, faces)


def interaction(VA: np.ndarray, VB: np.ndarray, faces_a, faces_b, hands_a: dict, hands_b: dict) -> dict:
    """Per frame: signed distance of each dancer's hands to the partner (nearest partner vertex, negative = inside,
    by the partner's vertex normal) and the penetration of every vertex into the partner (same test)."""
    from scipy.spatial import cKDTree
    T = len(VA)
    hand = {f"{w}_{h}": np.full(T, np.nan) for w in ("lead", "follow") for h in ("L", "R")}
    pen_max, pen_n = np.zeros(T), np.zeros(T)
    for k in range(T):
        for who, V, P, fP, hs in (("lead", VA[k], VB[k], faces_b, hands_a), ("follow", VB[k], VA[k], faces_a, hands_b)):
            tree = cKDTree(P)
            nP = _normals(P, fP)
            d, j = tree.query(V)
            sgn = np.sign(np.einsum("ij,ij->i", V - P[j], nP[j]))
            sd = d * np.where(sgn == 0, 1, sgn)
            for h, ids in hs.items():
                hand[f"{who}_{h}"][k] = float(sd[ids].min())
            inside = (sd < 0) & (d < PEN_MAX_M)
            pen_n[k] += inside.sum()
            pen_max[k] = max(pen_max[k], float(-sd[inside].min()) if inside.any() else 0.0)
    return {"hand_sd_m": hand, "pen_max_m": pen_max, "pen_nverts": pen_n}


def _interaction_summary(cur: dict, ref: dict | None, contact_m: float = 0.03) -> dict:
    out = {"pen_max_mm_median": round(float(np.median(cur["pen_max_m"]) * 1e3), 1),
           "pen_max_mm_p95": round(float(np.percentile(cur["pen_max_m"], 95) * 1e3), 1),
           "pen_verts_median": int(np.median(cur["pen_nverts"])),
           "hand_to_partner_mm_median": {k: round(float(np.median(v) * 1e3), 1) for k, v in cur["hand_sd_m"].items()}}
    if ref is not None:
        errs, tp, fp, fn = [], 0, 0, 0
        for k, v in cur["hand_sd_m"].items():
            r = ref["hand_sd_m"][k]
            near = r < 0.15
            errs += list(np.abs(v[near] - r[near]))
            c, cr = v < contact_m, r < contact_m
            tp += int((c & cr).sum()); fp += int((c & ~cr).sum()); fn += int((~c & cr).sum())
        out["hand_dist_err_mm_median_ref_within_15cm"] = round(float(np.median(errs) * 1e3), 1) if errs else None
        out["hand_contact_3cm"] = {"tp": tp, "fp": fp, "fn": fn,
                                   "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                                   "recall": round(tp / (tp + fn), 3) if tp + fn else None}
    return out


def _session_info(sid: str | None) -> dict:
    root = PACKAGE_ROOT / "work" / "runpod" / "sessions"
    if sid is None:
        cands = sorted([p for p in root.iterdir() if (p / "result.json").exists() and
                        "cand_closeapp" in (p / "result.json").read_text()], key=lambda p: p.stat().st_mtime) if root.exists() else []
        if not cands:
            return {}
        sid = cands[-1].name
    p = root / sid / "result.json"
    res = {"session": sid, **(json.loads(p.read_text()) if p.exists() else {})}
    pid = (res.get("pod") or {}).get("id")
    led = PACKAGE_ROOT / "work" / "runpod" / "ledger.json"
    if pid and led.exists():
        rows = json.loads(led.read_text())
        rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
        for r in rows:
            if isinstance(r, dict) and r.get("pod_id") == pid:
                res["ledger_row"] = r
    return res


def _medians(sc: dict) -> dict:
    out = {}
    for k in ("frames", "pa22_mm", "pa14_mm", "rr_mm", "pelvis_mm", "limb_deg"):
        v = sc.get(k)
        out[k] = (round(v["median"], 1) if v.get("median") is not None else None) if isinstance(v, dict) else v
    return out


def _window_dict(path: Path, t_ref: np.ndarray) -> dict:
    zz = dict(np.load(path, allow_pickle=True))
    t = np.asarray(zz["t"], float)
    k = np.array([int(np.argmin(np.abs(t - x))) for x in t_ref])
    if np.abs(t[k] - t_ref).max() > 5e-3:
        raise SystemExit(f"{path}: no sample within 5 ms of every phone-06 instant")
    return {kk: (v[k] if getattr(v, "ndim", 0) and len(v) == len(t) else v) for kk, v in zz.items()}


METHODS = {"closeapp": "CloseApp (CVPR 2025) release demo optimisation (--train_stage=1 --use_appearance)",
           "closeapp_noapp": "CloseApp without the appearance branch (--train_stage=1; the paper's w/o-appearance ablation)",
           "closeapp_init": "CloseApp's input initialisation: fit B (mamma_mono06) surface-fitted to SMPL, before any "
                            "CloseApp step"}


def import_(sid: str | None = None) -> dict:
    from .. import mamma_dyn as md
    from ..avatar_uv import Body
    from ..mamma_dyn import SMPLX_MODELS
    from .holdout import load_fit
    z = np.load(POD_OUT / "closeapp_out.npz", allow_pickle=True)
    pj = json.loads((POD_OUT / "closeapp_pod.json").read_text())
    inp = np.load(INPUT / "closeapp_input.npz", allow_pickle=True)
    prep_rep = json.loads((INPUT / "prep.json").read_text())
    t_ref, t_frame = np.asarray(inp["t_ref"], float), np.asarray(inp["t_frame"], float)
    sel, offsets = np.asarray(inp["frame_index"], int), np.asarray(inp["adapter_offsets"], float)
    ses = _session_info(sid)
    pod = ses.get("pod", {}) if isinstance(ses.get("pod"), dict) else {}
    pod = {**pod, "cloud": (ses.get("ledger_row") or {}).get("cloud")}
    tr = md.load_tracks(WORK, [CAM])[CAM]
    R, tv = tr.R[sel], tr.tvec[sel]
    kf = np.array([int(np.argmin(np.abs(tr.t - x))) for x in t_frame])   # the harness's camera pick for frame-clock t
    model = load_smpl_plain("NEUTRAL")
    fk = SmplJoints(model)
    # reference (8 phones) and fit B surfaces (world, MAMMA SMPL-X locked head) on the same instants
    xnpz = np.load(SMPLX_MODELS / "smplx" / "SMPLX_NEUTRAL.npz", allow_pickle=True)
    fx = np.asarray(xnpz["f"], np.int64)
    hx = _hand_sets(np.asarray(xnpz["weights"]), [20] + list(range(25, 40)), [21] + list(range(40, 55)))
    ref_v = {r: Body(_window_dict(WORK / "smplx" / f"{r}.npz", t_ref), SMPLX_MODELS).verts for r in ROLES}
    fb_v = {r: Body(_window_dict(FITB / f"{r}.npz", t_ref), SMPLX_MODELS).verts for r in ROLES}
    ref_int = interaction(ref_v["lead"], ref_v["follow"], fx, fx, hx, hx)
    fb_int = interaction(fb_v["lead"], fb_v["follow"], fx, fx, hx, hx)
    hs = _hand_sets(np.asarray(model["weights"]), [20, 22], [21, 23])
    refj = {r: load_fit(WORK / "smplx", r) for r in ROLES}
    out = {"pod": {k: pj.get(k) for k in ("versions", "stubs", "builds", "timings_s", "gpu", "sequence")},
           "session": {"sid": ses.get("session"), "gpu": pod.get("gpu"), "cloud": pod.get("cloud"),
                       "cost_per_hr": pod.get("cost_per_hr"), "ledger_cost_usd": pod.get("ledger_cost"),
                       "pod_lifetime_min": round(pod["lifetime_s"] / 60, 1) if pod.get("lifetime_s") else None},
           "init_conversion": {r: {k: prep_rep["ik"][r][k] for k in ("surface_nn_mm_median", "residual_mm_median",
                                                                       "residual_raw_smpl_joints_mm_median")}
                               for r in ROLES},
           "interaction_reference_8phone": _interaction_summary(ref_int, None),
           "interaction_mamma_mono06_fitB": _interaction_summary(fb_int, ref_int), "variants": {}}
    names = ["closeapp_init"] + [v for v in VARIANTS if f"{v}_pose" in z.files]
    for name in names:
        P, Tr, B = z[f"{name}_pose"], z[f"{name}_transl"], z[f"{name}_betas"]        # (2,T,72) (2,T,3) (2,10)
        vpod = (pj.get("variants") or {}).get(name, {}) or {}
        raw, ad, Vw, chk = {}, {}, {}, {}
        faces = None
        for a, role in enumerate(ROLES):
            raw[role], ad[role] = adapt(fk, P[a], B[a], Tr[a], offsets[a])
            v, j24, faces = smpl_bodies(P[a], Tr[a], B[a])
            Vw[role] = np.einsum("fji,fnj->fni", R, v - tv[:, None, :])
            pj_joints = z[f"{name}_joints"][a][:, :22] if f"{name}_joints" in z.files else None
            chk[role] = {"local_fk_vs_smplx_mm": round(float(np.abs(j24[:, :22] - raw[role]).max() * 1e3), 4),
                         "pod_vs_local_mm": round(float(np.abs(pj_joints - raw[role]).max() * 1e3), 4)
                         if pj_joints is not None else None}
            meta = {"method": METHODS[name], "input_cam": CAM, "version": f"boycehbz/CloseApp @ {CLOSEAPP_COMMIT}",
                    "licence": LICENCE, "flags": vpod.get("flags"), "gpu": pod.get("gpu") or pj.get("gpu"),
                    "cloud": pod.get("cloud"), "cost_per_hr": pod.get("cost_per_hr"), "session": ses.get("session"),
                    "cost_usd": pod.get("ledger_cost"),
                    "wall_min": round(vpod["wall_s"] / 60, 2) if vpod.get("wall_s") else None,
                    "body_model": "SMPL neutral (SMPLify basicModel_neutral v1.0.0)",
                    "joints": "SMPL FK joints 0-21 + the fit-B-calibrated segment-local joint-definition offsets "
                              "(cand_closeapp.surface_fit / adapt): SMPL -> MAMMA SMPL-X joint definitions",
                    "check": chk[role], "clock": "t = frame t + sync delta (reference seconds)",
                    "notes": "camera frame of phone 06 (CloseApp static identity camera, our focal); masks = our SAM 2.1 "
                             "dancer masks; keypoints = COCO-17 (people/06.npz) as Halpe 0-16; no proxemics prior "
                             "(release default; --use_prior cannot run as released); penetration weight 0 (release)"}
            C.save(TAKE, name, role, t_ref, ad[role], "camera", meta)
        sc = C.score(TAKE, name, WINDOW)
        diag = {"raw_smpl_joints": {}, "frame_clock_t": {}}
        for role in ROLES:
            r = refj[role]
            diag["raw_smpl_joints"][role] = _medians(C.score_arrays(t_ref, C.to_world(raw[role], R, tv), "world",
                                                                    r["t"], r["joints"], WINDOW))
            diag["frame_clock_t"][role] = _medians(C.score_arrays(t_frame, C.to_world(ad[role], tr.R[kf], tr.tvec[kf]),
                                                                  "world", r["t"], r["joints"], WINDOW))
        it = interaction(Vw["lead"], Vw["follow"], faces, faces, hs, hs)
        out["variants"][name] = {
            "score": {r: _medians(s) for r, s in sc["roles"].items()}, "diagnostics": diag,
            "interaction": _interaction_summary(it, ref_int),
            "closeapp_A_PD": vpod.get("A_PD"), "closeapp_A_PD_per_frame_median": vpod.get("A_PD_per_frame_median"),
            "bvh_inter_collisions_median": vpod.get("bvh_inter_collisions_median"),
            "bvh_frames_with_collisions": vpod.get("bvh_inter_collisions_frames_gt0"),
            "train": {k: vpod.get(k) for k in ("rc", "wall_s", "epochs", "epoch_s_median", "total_loss_first_last")},
            "check": chk}
        log(f"{name}: " + json.dumps(out["variants"][name]["score"]))
    (OUT / "closeapp_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(C.leaderboard(TAKE, WINDOW))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.cand_closeapp")
    ap.add_argument("action", choices=["smpl", "prep", "job", "import"])
    ap.add_argument("--session")
    ap.add_argument("--max-minutes", type=float, default=46)
    a = ap.parse_args(argv)
    if a.action == "smpl":
        for p in convert_smpl():
            log(f"wrote {p}")
    elif a.action == "prep":
        prep()
    elif a.action == "job":
        write_job(a.max_minutes)
    else:
        import_(a.session)


if __name__ == "__main__":
    main()
