"""SidesCfg: every threshold of the "sides" step (DESIGN sections 2.1-2.4, 3.1, 3.4), and its loader.

No take constant lives here. A take overrides any field in an optional `[sides]` table of takes/<take>.toml, e.g.

    [sides]
    blind_prior_nats = 0.5          # owner decision D1 (a): the prior never decides, even where no camera can tell
    [sides.priors.R4.face]
    back = 0.5

Values marked [I] in DESIGN.md are starting values (tunable with evidence); the others are measured there.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from ..take import PACKAGE_ROOT

HZ = 30.0


def _default_priors() -> dict:
    # R4 classes per couple-facing class (face / side / away): the owner's "his right hand usually rests on her upper-left back"
    # (and the zouk holds of zouk_contacts.HOLD_PRIOR: face 0.63 on the back, 0.07 chest), as signed nats before the cap.
    return {"R4": {"face": {"back": 0.5, "flank": 0.0, "front": -0.5},
                   "side": {"back": 0.25, "flank": 0.0, "front": -0.25},
                   "away": {"back": 0.5, "flank": 0.0, "front": -0.5}}}


@dataclass
class SidesCfg:
    # ---------------------------------------------------------------- inputs (layout conventions, searched in order; no take names)
    base_fit: str = "smplx_rf2_noslip"                      # work/<take>/<base_fit>/{lead,follow}.npz
    obs_dirs: tuple = ("sides/obs", "v3/hand_contact/obs")   # consolidated grid observations (hc.build_obs layout), first found wins
    obs_upload_dirs: tuple = ("sides/obs", "v4/contacts/pod_inputs/obs", "v3/hand_contact/obs")   # job_evidence: compressed first
    hz: float = HZ
    roles: tuple = ("lead", "follow")
    # ---------------------------------------------------------------- COCO detections on the grid (hc.CocoJudge rules, all 17 joints)
    det_conf: float = 0.5           # person confidence of the teacher row
    kp_conf: float = 0.5            # key point confidence for triangulation
    max_dt_s: float = 1.0 / 50      # nearest phone frame within this of the grid time
    # ---------------------------------------------------------------- triangulation (DESIGN 2.2)
    inlier_px: float = 14.0         # x f/1000 (TriCfg)
    min_angle_deg: float = 8.0      # seed pair ray angle (6 phones)
    min_angle_deg_two: float = 15.0  # two cameras
    identity_gate_m: float = 0.60   # within this of the base joint: catches role swaps only (never the 0.30 m TriCfg gate)
    sigma_px: float = 4.0           # x f/1000: covariance of a view's 2D
    sigma_floor_m: float = 0.02     # eigenvalue floor of each joint covariance
    min_views_real: int = 3         # "real" joint targets need >= this many inlier views
    vis_consistent: bool = True     # drop detections in views where the joint is predicted hidden, then re-triangulate
    vis_point: str = "tri"          # ... predicted at the triangulated point of all detections ("tri") or at the base joint ("base")
    occl_margin_m: float = 0.035    # the ray to a joint stops this short of it (zc.joint_visible)
    occl_r_scale: float = 0.92
    # ---------------------------------------------------------------- COCO -> SMPL-X bias (DESIGN 2.2)
    bias_free_m: float = 0.6        # free frames: pelvises at least this far apart horizontally
    bias_min_views: int = 3
    bias_max_frames: int = 3000
    # ---------------------------------------------------------------- held-out phone (DESIGN 1, "Held-out phone")
    heldout_block_s: float = 10.0
    heldout_min_cams: int = 5
    heldout_min_share: float = 0.2  # a phone "sees" the dancer in a block when its detection is there on >= this share of frames
    # ---------------------------------------------------------------- role-swap self-check (DESIGN 2.2)
    role_check_min: float = 0.90
    role_check_strict: bool = True  # fail loudly (an exception) when a camera's roles look swapped
    # ---------------------------------------------------------------- per-frame log-odds (DESIGN 2.3)
    clip_3d: float = 4.0
    clip_vis: float = 3.0
    clip_crop: float = 3.0
    prior_nats: float = 0.5         # the prior cap everywhere ...
    blind_prior_nats: float = 1.5   # ... and in camera-blind R4 runs (owner decision D1 = (b); 0.5 = decision (a))
    blind_vis_nats: float = 0.3     # camera-blind: the R4 visibility test spreads less than this over the three classes ...
    blind_3d_nats: float = 0.1      # ... and the 3D term is below this
    q_min: float = 0.55             # a view's ordinal cue below this reliability gets weight 0
    q_label_sigma: float = 3.0      # q_c labels: |m_tri| > this x sigma_m
    vote_conf_gap: float = 0.05     # COCO confidence order: the front limb's mean confidence higher by at least this
    crop_min_score: float = 0.8     # Jev questions used as evidence only when they scored >= this on the calibration frames
    crop_halfwidth_s: float = 0.5
    # a crop look holds along the entangled / contact run that contains it ("hold while entangled"): full weight within
    # crop_hold_full_s, falling linearly to 0 at crop_hold_s, never outside the run (0 = DESIGN's +-0.5 s only)  [I, Stage 2]
    crop_hold_s: float = 2.5
    crop_hold_full_s: float = 1.0
    # R4 cue calibration on the crop looks (the Jev rule applied to the other R4 cues): a cue (3D palm class, visibility test) whose
    # arg-max agrees with the agent's looks on < r4_cue_min_acc of the labelled frames where it votes is check-only (weight 0)
    r4_cue_calibrate: bool = True
    r4_cue_min_acc: float = 0.55
    r4_cue_min_n: int = 10
    r4_cue_label_p: float = 0.6     # a look labels a frame when its class probability is >= this
    r4_cue_mode: str = "confusion"  # "confusion": the cue's vote through its calibrated P(vote | truth); "weight": 0 / 1 by r4_cue_min_acc
    r4_cue_alpha_diag: float = 2.0  # [I] Dirichlet prior of the confusion (off-diagonal 1)
    # camera-blind R4 frames (where D1's blind_prior_nats applies): "hand_unseen" = no view's MAMMA sees his right hand (majority over
    # r4_blind_window_s) and no camera evidence favours another class over the prior's by >= blind_vis_nats [Stage 2];
    # "no_evidence" = the Stage-1 rule (visibility test, 3D and crops all silent)
    r4_blind_rule: str = "hand_unseen"
    r4_blind_window_s: float = 1.0
    side_cost: float = 0.7          # emission cost of a side state against "undecided" (nats)
    # ---------------------------------------------------------------- HMM (DESIGN 2.3)
    p_stay: float = 0.98
    p_undecided: float = 0.015
    p_flip: float = 0.005
    p_flip_band: float = 1e-6       # a direct flip inside the entangled band (~14 nats)
    p_enter: float = 0.01           # undecided -> a side
    post_decide: float = 0.9
    post_exit: float = 0.7          # Schmitt exit
    min_run_s: float = 0.5
    band_gli: float = 0.02
    band_dist_m: float = 0.10
    r4_band_m: float = 0.05         # R4: palm within this of her surface = in contact (front <-> back directly = through her)
    # ---------------------------------------------------------------- R4 (DESIGN 2.1)
    r4_gate_m: float = 0.15         # palm within this of her torso surface
    r4_fb_dz_m: float = 0.03
    r4_fl_dz_m: float = 0.02
    r4_inside_m: float = 0.03
    palm_from_wrist_m: float = 0.08  # the palm centre along the forearm beyond the triangulated wrist
    palm_lift_m: float = 0.015
    p_vis_floor: float = 0.05       # detection rates clipped to [floor, 1 - floor]
    vis_sigma_px: float = 30.0      # x f/1000: a detected visible wrist around its predicted pixel (the hypotheses' wrist is a coarse
                                    # placement: 30 px ~ 8-10 cm at 3-4 m) ...  [I, tuned from 20]
    hall_sigma_px: float = 100.0    # ... a hallucinated one (broad, near her outline)
    hand_vis_thr: float = 0.15      # MAMMA "sees his right hand": the mean visibility of its points above this (recap medians 0.11 behind
                                    # her, 0.20 clear: relations.md 2.1 d)
    # ---------------------------------------------------------------- yaw HMM (DESIGN 2.4)
    yaw_bins: int = 72
    yaw_max_dps: float = 1080.0
    yaw_trans_sigma_deg: float = 10.0     # per frame, around the base's own angular velocity ...
    yaw_trans_wide_deg: float = 25.0      # ... and a wide component around 0
    yaw_trans_wide_w: float = 0.03        # [I, tuned from 0.15: a single view bit flipped the track by 120 deg at 327.7 s]
    yaw_line_sigma_deg: float = 25.0      # triangulated hip line, debiased in its own frame (recap: p90 20-22 deg from the base, >= 3 views)
    yaw_shoulder_sigma_deg: float = 20.0  # triangulated shoulder line (recap: p90 13.5-14.1 deg with >= 3 views)
    yaw_line_min_views: int = 3
    yaw_line_width: tuple = (0.7, 1.4)    # the triangulated line's horizontal width / the base's (mislocated joints collapse it)
    yaw_base_sigma_deg: float = 60.0      # weak anchor to the base heading (0 = off) [I, tuned from 90]
    yaw_soft_deg: float = 10.0            # softness of a face / L-R half-plane
    yaw_twist_clip_deg: float = 90.0
    face_nose: float = 0.7
    face_eye: float = 0.6
    face_hidden: float = 0.2
    lr_dead: float = 0.1
    spin_err_deg: float = 45.0
    spin_min_frames: int = 3
    spin_cand_dps: float = 150.0
    # ---------------------------------------------------------------- violations (DESIGN 3.1)
    side_min_frames: int = 3
    inside_min_frames: int = 6
    real_win: int = 10
    real_min: int = 5
    real_tau_m: float = 0.06
    real_tau_contact_m: float = 0.08
    contact_joints: dict = field(default_factory=lambda: {"lead": ("elR", "wrR"), "follow": ("elL", "wrL")})
    swing_win_s: float = 2.0
    swing_floor_m: float = 0.005
    swing_slack_m: float = 0.003
    swing_band_hz: tuple = (1.0, 5.0)
    thru_jump: float = 0.05
    limit_min_frames: int = 3
    # gate thresholds of the swing-twist limits (DESIGN 3.4, deg)
    twist_gate: dict = field(default_factory=lambda: {"spine": 35.0, "spine_total": 70.0, "neck_head": 70.0, "hip": 50.0, "knee": 45.0,
                                                      "ankle": 40.0})
    wrap_deg: float = 300.0
    # ---------------------------------------------------------------- R16 unwrap + spine split (DESIGN 2.4 "Wraps"; sides.unwrap)
    unwrap_candy_deg: float = 90.0        # [I] a single local twist beyond this is never anatomical: re-expressed like a wrap
    unwrap_guard_deg: float = 30.0        # Sigma* within this of the measured relative heading = a pure wrap
    unwrap_guard_total_deg: float = 90.0  # [I] and |Sigma*| <= this
    unwrap_ramp_frames: int = 6           # [I]
    unwrap_split_joint_max_deg: float = 35.0
    unwrap_split_move_max_m: float = 0.03
    unwrap_split_ramp_frames: int = 4     # [I]
    unwrap_guard_mode: str = "flag"       # [I] "flag" (re-express the event, list guard failures as not-pure) | "gate" (DESIGN-strict)
    unwrap_chains: tuple = ("spine", "neck", "leg_l", "leg_r")
    # ---------------------------------------------------------------- spin splice (DESIGN 2.4 "Splice"; sides.splice)
    splice_defect_min_views: int = 2
    splice_merge_gap: int = 2             # [I] frames
    splice_entry_frames: int = 7
    splice_exit_p95_m: float = 0.060
    splice_exit_seg_deg: float = 15.0
    splice_exit_hold_s: float = 0.3
    splice_exit_slerp_frames: int = 6
    splice_exit_relative: bool = True     # spine3 / feet as (segment - pelvis) winding (F1, S/final/f1_exit.py)
    splice_foot_min_h_m: float = 0.03     # [I]
    splice_switch_nats: float = 3.0       # [I] an alternative must beat the base by this over the defect run
    splice_face_q: float = 0.9            # [I] face-bit reliability in the alternative scores (s6: 89.3 % agree)
    splice_hip_sigma_deg: float = 25.0    # [I] COCO-3D hip line (calm-frame bias removed) in the alternative scores
    splice_w2d_nats_per_px: float = 0.1   # [I] MAMMA-512 2D cost in the alternative scores
    # ---------------------------------------------------------------- windows (DESIGN 3.1; sides.plan)
    plan_pad_s: float = 0.5
    plan_merge_gap_s: float = 1.0
    plan_core_max_s: float = 6.0
    plan_overlap_s: float = 1.0
    plan_ramp_s: float = 0.75             # [I] cosine ramps 0.5-1.0 s
    plan_real_min_s: float = 0.5          # [I] a `real` item opens a window when it lasts this long ...
    plan_real_big_m: float = 0.15         # ... or is this far off
    plan_owner_windows: tuple = ()        # (dance s) spans the report lists first / tier 1 (the take's [sides] table)
    plan_real_scope: str = "owner"        # [I] "owner": `real` alone opens a window only inside plan_owner_windows (elsewhere it joins
                                          # windows opened by another reason, and is listed as deferred); "all": everywhere
    # ---------------------------------------------------------------- trust region (DESIGN 1.1 / 3.2), deg unless noted
    trust_arm_deg: float = 25.0
    trust_collar_deg: float = 10.0
    trust_spine3_arm_deg: float = 5.0
    trust_spine_deg: float = 15.0
    trust_neck_deg: float = 20.0
    trust_leg_deg: float = 20.0
    trust_global_deg: float = 45.0
    trust_root_m: float = 0.03
    # ---------------------------------------------------------------- the solve (DESIGN 3.3 / 3.5; sides.solve), weights [I]
    s_mamma_px: float = 10.0              # x f/1000
    s_coco_px: float = 14.0
    w_2d: float = 2.0                     # [I] round 3: G1 failed on most windows at 1
    w_g1: float = 100.0                   # [I] round 4: E_g1 = the gate's per-camera G1 statistic as a hinge (px^2)
    g1_margin_px: float = 0.5             # [I] ... aiming this far inside the gate's tolerance
    w_tri: float = 2.0
    w_prior: float = 1.0
    w_side: float = 8.0
    w_sep: float = 20.0                   # [I] round 3 (G4)
    w_gli: float = 5.0
    w_inside: float = 4.0
    w_contact: float = 4.0
    w_lim: float = 0.05                   # per deg^2
    w_acc: float = 1.0
    w_swivel: float = 2.0
    w_hinge: float = 5.0
    w_hinge_soft: float = 1.0             # elbow off-hinge beyond elbow_off_soft_deg (inside w_lim)
    w_band: float = 300.0                 # [I] per cm^2 per 2-s window (report.band_rms_windows' measure, as G5)
    w_swivel_acc: float = 10.0            # [I] the swivel's acceleration beyond the base's ...
    swivel_acc_slack_deg: float = 0.15    # ... by more than this per frame^2
    w_jerk: float = 1.0                   # [I] the arm correction's jerk ...
    jerk_slack_m: float = 0.0005          # ... beyond 0.5 mm per frame^3 (~13 m/s^3)
    w_yaw: float = 3.0
    w_wind: float = 3.0
    w_face: float = 2.0
    w_floor: float = 2.0
    prior_sigma0_deg: float = 4.0
    prior_sigma_q_deg: float = 16.0
    prior_root_m: float = 0.01
    acc_slack_mps2: float = 2.0
    w_add: float = 10.0                   # [I] the correction's own speed beyond add_mps (per (0.1 m/s)^2)
    add_mps: float = 0.35                 # [I] below zouk_contacts.GATE add_mps 0.55 (G7)
    w_trust: float = 20.0                 # [I] barrier before the trust region's edge ...
    trust_soft: float = 0.85              # ... from this tanh on
    prior_sigma_trust_frac: float = 0.5   # [I] the prior's sigma <= this x the joint's trust radius
    w_smooth: float = 1.0                 # [I] second difference of the per-frame correction ...
    smooth_deg: float = 0.5               # ... per frame^2 at this scale
    w_hold_base: float = 1.0              # [I] undecided but entangled arms keep the base's own GLI sign (no new pass-through, G4)
    elbow_off_soft_deg: float = 15.0
    sep_d0_m: float = 0.02               # [I] F2's d0 = 1 cm let Adam's momentum carry a bone through (window 4): 2 cm
    sep_dist_m: float = 0.10
    contact_m: float = 0.02
    solve_lr: float = 0.02
    solve_iters1: int = 120
    solve_iters2: int = 120
    solve_iters3: int = 40
    solve_seed: int = 0
    solve_free_ramps: bool = True        # [I] the ramp frames carry their own knots (x the ramp weight)
    solve_workers: int = 1                # processes per device (J2: 4 on a 24-GB GPU)
    solve_threads: int = 2                # torch CPU threads per worker
    solve_r4_vis: bool = False            # [I] build C: E_2D / E_g1 / E_tri drop his right hand / wrist / distal-forearm evidence on decided-R4
    #                                       frames in the views where the DECIDED class hides the hand (a proxy palm on her surface of that
    #                                       class, two-body ray test on the base poses) - the gate's own rule (gate_vis_r4) on the solution
    w_r4_mult: float = 1.0                # [I] build C: E_side's R4 class part and E_contact x this ...
    r4_keep_scale: bool = False           # [I] ... and not softened by an attempt's side_scale (the decided class stays enforced)
    solve_attempts: tuple = ({"name": "planned"}, {"name": "prior_x2_soft", "lam_prior": 2.0, "side_scale": 0.25},
                             {"name": "worst_chain_minimal", "lam_prior": 2.0, "side_scale": 0.0, "chain": True},
                             {"name": "worst_chain_prior_x4", "lam_prior": 4.0, "side_scale": 0.0, "chain": True})
    # [I] DESIGN 3.5's ladder; the last attempt keeps the holds, inside, the triangulated joints and the 2D but no longer pulls toward a
    # decided side / class (the palm stays as detected where only the class is wrong: flagged as a missed target)
    # [I] round 4: a fourth, gentler chain attempt (prior x 4) for windows whose minimal correction still moves too fast (G5 / G7)
    # ---------------------------------------------------------------- gates (DESIGN 7.1; sides.gate)
    gate_moved_m: float = 0.01            # G1 / G2 / G10 measure the points the candidate moved more than this somewhere in the window
    gate_g1_min_points: int = 5           # a camera "sees" the moved part in a frame with >= this many visible MAMMA points of it
    gate_g1_min_share: float = 0.2        # ... on >= this share of the window frames
    gate_g1_med_px: float = 1.0
    gate_g1_p90_px: float = 3.0
    gate_g1_rel: float = 0.10             # [I] ... or this share of the base's own error (a camera where MAMMA is 15-28 px off)
    gate_per_chain: bool = True           # [I] fix round 1 (verify_r0 major 2): G1 / G2 also per moved chain (each arm, each leg, the
    #                                       torso + head), as well as pooled: a spine3 move pools the torso and both arms, which hid one arm
    #                                       moving away from every phone (window 38)
    gate_per_chain_any: bool = True       # [I] ... also when a single chain moved (the pooled check reads every window frame; the chain
    #                                       check only the (frame, point) pairs that moved)
    gate_vis_r4: bool = True              # [I] build C (owner rule 2026-10-24 + D1): on decided-R4 frames with the candidate's palm on the
    #                                       decided class, G1 / G10 drop his right hand / wrist / distal-forearm detections of the views where
    #                                       the candidate predicts them hidden (two-body ray test; hallucinations of a hidden hand), G2 his
    #                                       triangulated wrist with such a view among its inliers; recorded per window (G1 "vis_r4")
    gate_chain_min_cams: int = 2          # [I] ... a chain fails G1 when its median error rises in >= this many cameras, by more than
    gate_chain_med_px: float = 2.0        # [I] ... max(this, gate_g1_rel x its base median) on the (frame, point) pairs that moved.
    #                                       Calibrated (gate_calibration.json): the 199-s splice, better in 3D on every chain, rises by
    #                                       1.3-2.7 px in one camera per chain (cam 03 / 05 / 06, the detectors disagreeing); with a 1-px
    #                                       floor or one camera it is rejected. Window 38's right arm rises in 4 phones (up to +15 px)
    gate_g2_chain_min_n: int = 10         # a chain's G2 counts only with >= this many (frame, joint) samples with >= 3 views
    gate_g2_m: float = 0.005
    gate_g2_real_gain_m: float = 0.02
    gate_g2_real_gain_rel: float = 0.2
    gate_g2_real_end_m: float = 0.08
    gate_g3_share: float = 0.02
    gate_g5_band_m: float = 0.003
    gate_g5_ratio: float = 1.2
    gate_g5_rev_margin_m: float = 0.005    # [I] the reversal count's hysteresis margin
    gate_g5_ratio_spin: float = 2.0       # [I] a spin window's dancer (the turn changes every joint's motion): swivel / jerk p99 ratio
    gate_g5_swivel_floor: float = 300.0   # [I] deg/s^2 added to 1.2 x the base's p99 (a still arm has ~0)
    gate_g5_jerk_floor: float = 5.0       # [I] m/s^3 added to 1.2 x the base's p99
    gate_g8_p90_deg: float = 25.0
    gate_g8_rev_deg: float = 90.0
    gate_g8_torso_deg: float = 70.0       # shoulder line - hip line (spine total, DESIGN 3.4)
    gate_g8_foot_deg: float = 60.0        # [I] foot - hip line (E_wind's stance-foot hinge)
    gate_g9_m: float = 0.03
    gate_g9_hard_m: float = 0.05
    gate_g9_share: float = 0.05
    gate_g9_allow_m: float = 0.015        # her upper arms: the h4 allowance
    gate_g10_min_frames: int = 10
    gate_g10_med_px: float = 0.5
    gate_g10_p90_px: float = 2.0
    gate_take_2d_px: float = 0.2          # whole take: per-camera MAMMA median <= base + this
    gate_take_slip_mm: float = 2.0
    gate_take_floor_m: float = 0.01
    # ---------------------------------------------------------------- export / install (sides.install)
    install_input_export_dir: str = ""     # work/<take>-relative folder of the INPUT fit's export (the proof's reference)
    install_capture: str = ""              # the Unity capture the motion is for (the side export is <capture>_sides)
    gate_trust_frac: float = 0.98         # a joint "hits" its trust region at this share of the radius
    gate_trust_share: float = 0.10
    gate_exact_device: str = "cpu"        # where the gate measures the exact mesh depth when the solve did not
    gate_calib_swing_t: float = 57.0      # [I] dance s: the synthetic swing of the calibration (the take's [sides] / cfg.json)
    # ---------------------------------------------------------------- priors
    priors: dict = field(default_factory=_default_priors)
    # ---------------------------------------------------------------- compute
    workers: int = 1

    def to_json(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=list))

    def unwrap_cfg(self):
        """the sides.unwrap.UnwrapCfg of these values (one source of truth: SidesCfg / the take's [sides] table)"""
        from .unwrap import UnwrapCfg
        return UnwrapCfg(wrap_deg=self.wrap_deg, candy_deg=self.unwrap_candy_deg, guard_deg=self.unwrap_guard_deg,
                         guard_total_deg=self.unwrap_guard_total_deg, ramp_frames=self.unwrap_ramp_frames,
                         split_joint_max_deg=self.unwrap_split_joint_max_deg, split_move_max_m=self.unwrap_split_move_max_m,
                         split_ramp_frames=self.unwrap_split_ramp_frames, guard_mode=self.unwrap_guard_mode,
                         chains=tuple(self.unwrap_chains))

    def splice_cfg(self):
        """the sides.splice.SpliceCfg of these values"""
        from .splice import SpliceCfg
        return SpliceCfg(hz=self.hz, defect_deg=self.spin_err_deg, defect_min_frames=self.spin_min_frames,
                         defect_min_views=self.splice_defect_min_views, defect_merge_gap=self.splice_merge_gap,
                         entry_frames=self.splice_entry_frames, exit_p95_m=self.splice_exit_p95_m, exit_seg_deg=self.splice_exit_seg_deg,
                         exit_hold_s=self.splice_exit_hold_s, exit_slerp_frames=self.splice_exit_slerp_frames,
                         exit_relative=self.splice_exit_relative, foot_min_h_m=self.splice_foot_min_h_m)


def toml_path(take: str) -> Path:
    return PACKAGE_ROOT / "takes" / f"{take}.toml"


def _coerce(cur, val):
    if isinstance(cur, tuple):
        return tuple(val)
    if isinstance(cur, float) and isinstance(val, int):
        return float(val)
    return val


def apply_overrides(cfg: SidesCfg, over: dict) -> SidesCfg:
    """a new SidesCfg with `over` ({field: value}) applied; unknown keys are an error; dict fields are merged one level deep"""
    names = {f.name for f in fields(SidesCfg)}
    bad = sorted(set(over) - names)
    if bad:
        raise KeyError(f"[sides] unknown keys: {bad}")
    kw = {}
    for k, v in over.items():
        cur = getattr(cfg, k)
        if isinstance(cur, dict) and isinstance(v, dict):
            m = json.loads(json.dumps(cur, default=list))
            for kk, vv in v.items():
                if isinstance(m.get(kk), dict) and isinstance(vv, dict):
                    m[kk] = {**m[kk], **vv}
                else:
                    m[kk] = vv
            kw[k] = m
        else:
            kw[k] = _coerce(cur, v)
    return replace(cfg, **kw)


def parse_toml_text(text: str) -> dict:
    """the [sides] table of a take TOML text ({} when absent)"""
    return dict(tomllib.loads(text).get("sides", {}) or {})


def work_cfg_path(take: str) -> Path:
    """the take's work-data overrides (git-ignored): work/<take>/sides/cfg.json {field: value} - for values that are data of one take
    (its owner windows), kept out of the tracked TOML"""
    return PACKAGE_ROOT / "work" / take / "sides" / "cfg.json"


def load_cfg(take: str | None = None, overrides: dict | None = None, toml_text: str | None = None, work_overrides: bool = True) -> SidesCfg:
    """SidesCfg defaults <- takes/<take>.toml [sides] (optional) <- work/<take>/sides/cfg.json (optional) <- overrides"""
    cfg = SidesCfg()
    tbl = {}
    if toml_text is not None:
        tbl = parse_toml_text(toml_text)
    elif take:
        p = toml_path(take)
        if p.exists():
            tbl = parse_toml_text(p.read_text(encoding="utf-8"))
    if tbl:
        cfg = apply_overrides(cfg, tbl)
    if take and work_overrides and toml_text is None:
        wp = work_cfg_path(take)
        if wp.exists():
            d = json.loads(wp.read_text(encoding="utf-8"))
            d = {k: v for k, v in d.items() if not k.startswith("_")}
            if d:
                cfg = apply_overrides(cfg, d)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg
