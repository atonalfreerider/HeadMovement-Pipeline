"""AGS: avatar-guided dancer Gaussian splats (docs/DANCER_SPLATS_RESEARCH.md, approved 2026-10-08 as a prototype).

    python -m dancecap.dancer_splats larissa_kadu prepare            # local CPU: work/<take>/dancer_splats/bundle/
    python -m dancecap.dancer_splats larissa_kadu write-pod-job      # dancer_splats/pod_job.json (dancecap.pod_job/1)
    python -m dancecap remote run --take larissa_kadu --jobs dancer_splats --max-cost 1.5 --max-minutes 55
    python -m dancecap.dancer_splats larissa_kadu train  --preset pod # (on the pod) S1 -> S2 -> S3
    python -m dancecap.dancer_splats larissa_kadu eval   --preset pod # held-out metrics vs the avatar, review sheets
    python -m dancecap.dancer_splats larissa_kadu export             # local: canonical + per-frame PLYs for Unity
    python -m dancecap.dancer_splats larissa_kadu render --cam 03 --t 59.18   # any camera / time -> JPEG

Every Gaussian is ATTACHED: a body Gaussian to one SMPL-X triangle of its dancer (barycentric anchor + offset along the
triangle normal + a small tangent offset + a local rotation / log-scale, all in the triangle's frame, so the fit skins
it every frame), a hair Gaussian to one segment of a groom guide strand (avatar/hair_groom.json: 160 guides x 16
segments rooted on the follow's scalp; the strand nodes get free per-frame offsets in the head frame, kept physical by
inextensibility, a fixed root, an acceleration penalty and the groom's colliders). There are no free Gaussians: the
splats show the two dancers only (no room, floor, spectators or mirror - VIEWER_SPEC 3.2a; the viewer layer starts off).

Per frame the trainer refines the fit with small regularised pose corrections (<= 3 deg per body joint, <= 2 cm root
translation, an acceleration penalty) written to dancer_splats/pose_delta.npz - never to smplx/. The loss is a masked
L1 + D-SSIM against the photo inside the union of the dancers' masks after a per-image colour transform (room.py's
Appearance), a silhouette term on a random background, per-role ownership (one-hot role channels rendered in the same
gsplat call, so mutual occlusion is sorted) vs the SAM masks, hair vs the hair masks, a hinge against one dancer's
Gaussians entering the other's body, MCMC opacity/scale regularisers and an avatar colour prior on face and hands.
Densification is gsplat MCMC relocation restricted to the anchors (a dead Gaussian moves onto the triangle / strand
segment of a live one, never off the body). Camera 03 is held out, plus every 10th instant of the other cameras.

Outputs (work/<take>/dancer_splats/): model.pt (canonical Gaussians + bindings + residual coefficients + hair tracks +
pose deltas + colour transforms), pose_delta.npz, metrics.json (held-out PSNR / SSIM / LPIPS / IoU / ghosts vs the
textured avatar rendered from the same held-out camera, gates), report.json, progress.json, train.log; review sheets in
work/<take>/review/splats/; after `export`: canonical_<role>.ply, frames/frame_<k>.ply (+ splat_sequence.json).
Splats are likeness data: work/ and the git-ignored StreamingAssets only, never the head-movement repo.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

ROLES = ("lead", "follow")
PIDS = {"lead": 1, "follow": 2}
GROUPS = ("lead", "follow", "hair")          # Gaussian groups; hair belongs to the groom's role (follow)
BUNDLE_VERSION = 1
N_BODY_JOINTS = 22                            # pelvis + 21 body joints carry pose deltas (jaw/eyes/fingers follow)
HEAD_JOINT = 15

# label image of a training crop (labels/<i>.png)
L_BG, L_LEAD, L_FOLLOW, L_HAIR, L_BOTH, L_IGNORE = 0, 1, 2, 3, 4, 255

# per-Gaussian region bits (init_<role>.npz) -> offset / residual caps and the avatar colour prior
R_HEAD, R_FACE, R_EYE, R_HAND, R_SKIN, R_FOOT = 1, 2, 4, 8, 16, 32

STAGES = ("S1", "S2", "S3")


@dataclass
class SplatConfig:
    # ---- prep (fixed when the bundle is written)
    window: tuple[float, float] = (58.0, 61.5)  # sync-corrected reference seconds (the bodies' clock)
    cams: tuple[str, ...] = ()                  # () = every camera with cameras/<cam>.npz
    holdout_cams: tuple[str, ...] = ("03",)
    holdout_every: int = 10                     # every 10th instant (grid frame) of the other cameras is held out
    crop_pad: float = 0.10                      # crop = union of the dancers' masks + projected bodies, +10 %
    max_side: int = 0                           # 0 = native resolution crops; > 0 downsizes (smoke test)
    init_samples: int = 160_000                 # surface samples stored per dancer (the trainer uses a prefix)
    face_iod_px: float = 15.0                   # views where a face is >= this inter-ocular distance train the face
    eval_train_views: int = 40                  # in-sample views also scored (fitting metric)
    novel_pair: tuple[str, str] = ("03", "04")  # fixed novel camera halfway between these (temporal check)
    # ---- model
    n_lead: int = 70_000
    n_follow: int = 60_000
    n_hair_per_seg: int = 4                     # Gaussians along each guide segment (x (1 + groom children))
    cap_factor: float = 1.15                    # MCMC growth cap = cap_factor x the initial count per group
    sh_degree: int = 2
    resid_k: int = 12                           # cosine (DCT) basis of the per-Gaussian residual over the window
    resid_k_skin: int = 8
    resid_cap_skin: float = 0.03                # m
    resid_cap_cloth: float = 0.06
    normal_range_skin: tuple[float, float] = (-0.01, 0.015)
    normal_range_cloth: tuple[float, float] = (-0.015, 0.06)
    normal_range_foot: tuple[float, float] = (-0.01, 0.05)
    scale_cap: float = 0.03                     # m, body Gaussians
    pose_cap_deg: float = 3.0
    pose_cap_m: float = 0.02
    hair_off_cap: float = 0.25                  # m, per-frame node offset in the head frame (soft, tanh)
    # ---- training
    iters: tuple[int, int, int] = (6000, 15000, 4000)  # S1 canonical, S2 dynamic, S3 polish
    views_per_step: int = 4
    time_budget_min: float = 0.0                # > 0: the schedule shrinks so training ends inside it
    kill_reserve_min: float = 15.0              # on a pod: training also ends this long before the kill switch
                                                # (/workspace/.dancecap_kill_at): eval + pack + download + margin
    lr_off: float = 2e-3
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opac: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    lr_resid: float = 1e-3
    lr_pose: float = 2e-3
    lr_hair_nodes: float = 2e-4
    lr_app: float = 1e-3
    ssim_lambda: float = 0.2
    w_sil: float = 0.5
    w_role: float = 0.2
    w_hair: float = 0.2
    w_pen: float = 1.0                          # interpenetration hinge
    w_floor: float = 1.0                        # hinge on Gaussians below the floor (world y < floor_m)
    floor_m: float = -0.01
    w_prior: float = 0.01                       # face / hand colour stays near the avatar albedo
    w_pose: float = 0.05                        # pose deltas: L2 (pull to MAMMA) ...
    w_pose_acc: float = 1.0                     # ... + second difference over frames
    w_resid: float = 0.1
    w_hair_len: float = 10.0
    w_hair_acc: float = 1.0
    w_hair_col: float = 10.0
    w_hair_l2: float = 0.01
    app_reg: float = 1e-2
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    swap_max: float = 0.2                       # roles.json swap_prob above this: no role-ownership loss
    refine_every: int = 100
    refine_start: int = 500
    refine_stop_frac: float = 0.84              # relocation stops at this fraction of the schedule (S3 = polish)
    min_opacity: float = 0.005
    log_every: int = 250
    ckpt_every: int = 2000
    seed: int = 0
    device: str = "cuda"
    note: str = field(default="", repr=False)


PREP_FIELDS = ("window", "cams", "holdout_cams", "holdout_every", "crop_pad", "max_side", "init_samples",
               "face_iod_px", "eval_train_views", "novel_pair")

PRESETS: dict[str, dict] = {
    # 80 GB A100 pod: the approved prototype (25k steps, ~170k Gaussians, 4 views per step)
    "pod": dict(time_budget_min=24.0),
    # local code-path check on the shared 2080 Ti: a few hundred steps, tiny model (< 2 min GPU)
    "smoke": dict(n_lead=12_000, n_follow=10_000, n_hair_per_seg=1, iters=(120, 120, 60), views_per_step=2,
                  log_every=20, ckpt_every=10_000, refine_start=60, refine_every=40, time_budget_min=1.5,
                  sh_degree=1),
    # 2080 Ti, longer (not used by the pipeline)
    "local": dict(n_lead=30_000, n_follow=25_000, n_hair_per_seg=2, iters=(1500, 3000, 500), views_per_step=2),
}


def preset(name: str, **over) -> SplatConfig:
    if name not in PRESETS:
        raise ValueError(f"dancer_splats: unknown preset {name!r}; choose from {sorted(PRESETS)}")
    cfg = SplatConfig(**PRESETS[name])
    for k, v in over.items():
        if v is not None:
            setattr(cfg, k, v)
    cfg.note = f"preset {name}"
    return cfg


def config_dict(cfg: SplatConfig) -> dict:
    d = asdict(cfg)
    return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}


def prep_dict(cfg: SplatConfig) -> dict:
    d = config_dict(cfg)
    return {k: d[k] for k in PREP_FIELDS}


# the user-approved prototype window per take (2026-10-08: larissa_kadu 58.0-61.5 s = her three fastest head turns, the
# worst body-fit stretch, camera 09 zooming); takes/<take>.toml [dancer_splats] window = [a, b] overrides
PROTOTYPE_WINDOWS = {"larissa_kadu": (58.0, 61.5)}
DEFAULT_WINDOW_S = 3.5


def default_window(take: str) -> tuple[float, float]:
    """[dancer_splats] window of the take TOML, else the approved prototype window, else DEFAULT_WINDOW_S centred in
    the take's analysis window."""
    import tomllib

    from ..take import PACKAGE_ROOT
    p = PACKAGE_ROOT / "takes" / f"{take}.toml"
    cfg = tomllib.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    w = (cfg.get("dancer_splats") or {}).get("window")
    if w and len(w) == 2:
        return float(w[0]), float(w[1])
    if take in PROTOTYPE_WINDOWS:
        return PROTOTYPE_WINDOWS[take]
    tw = cfg.get("window")
    if tw and len(tw) == 2:
        mid, half = 0.5 * (float(tw[0]) + float(tw[1])), min(DEFAULT_WINDOW_S, float(tw[1]) - float(tw[0])) / 2
        return round(mid - half, 3), round(mid + half, 3)
    return SplatConfig().window


def total_iters(cfg: SplatConfig) -> int:
    return int(sum(cfg.iters))
