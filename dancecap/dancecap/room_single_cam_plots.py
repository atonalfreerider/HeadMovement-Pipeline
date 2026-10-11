"""Evidence and review plots of the single-camera room analysis (john_lula): camera motion, the focal / distance / height table, plan view, heights.

    python -m dancecap.room_single_cam evidence       -> review/room/{camera_motion,focal_table,plan_view,heights}.png + cameras/motion_stats.json
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import room_single_cam as RS  # noqa: E402


def motion(arr: dict) -> dict:
    """Pan / tilt / roll of the camera against the world frame, zoom steps, speeds: from the per-frame R of cameras/01.npz."""
    R = arr["R"]
    fwd = np.einsum("nba,b->na", R, np.array([0.0, 0.0, 1.0]))             # camera forward in the world (R^T z)
    rgt = np.einsum("nba,b->na", R, np.array([1.0, 0.0, 0.0]))
    yaw = np.degrees(np.arctan2(fwd[:, 0], -fwd[:, 2]))                    # 0 = looking along -Z, positive towards +X
    pitch = np.degrees(np.arcsin(np.clip(-fwd[:, 1], -1, 1)))               # positive = looking down
    roll = np.degrees(np.arcsin(np.clip(rgt[:, 1], -1, 1)))                 # positive = right side of the camera up
    t = arr["frame_index"] / RS.FPS
    f = arr["focal_px"]
    two = arr["zoom_2x"]
    steps = np.where(np.diff(two.astype(int)) != 0)[0]
    vel = np.degrees(np.linalg.norm(np.diff(np.stack([yaw, pitch], 1), axis=0), axis=1)) * 0 + np.linalg.norm(np.diff(np.stack([yaw, pitch], 1), axis=0), axis=1) * RS.FPS
    return {"t": t, "yaw": yaw, "pitch": pitch, "roll": roll, "f": f, "two": two, "steps_s": (t[steps + 1]).tolist(), "vel": vel,
            "stats": {"yaw_range_deg": [float(np.percentile(yaw, 1)), float(np.percentile(yaw, 99))], "pitch_down_range_deg": [float(np.percentile(pitch, 1)), float(np.percentile(pitch, 99))],
                      "roll_range_deg": [float(np.percentile(roll, 1)), float(np.percentile(roll, 99))], "roll_median_deg": float(np.median(roll)),
                      "pan_tilt_speed_median_deg_s": float(np.median(vel)), "pan_tilt_speed_p90_deg_s": float(np.percentile(vel, 90)),
                      "pan_tilt_speed_max_deg_s": float(vel.max()), "zoom_step_times_s": (t[steps + 1]).tolist(),
                      "frames_1x": int((~two).sum()), "frames_2x": int(two.sum())}}


def bootstrap(dw: dict, n: int = 400, block_s: float = 8.0, seed: int = 0) -> dict:
    """Block bootstrap (8 s blocks) of the dancers' pooled camera height: the statistical error of the height."""
    rng = np.random.default_rng(seed)
    t = dw["frame"] / RS.FPS
    ok = (dw["role"] > 0) & dw["upright"] & np.isfinite(dw["h_sample"]) & (dw["h_sample"] > 0.6) & (dw["h_sample"] < 2.6)
    blk = (t // block_s).astype(int)
    ub = np.unique(blk[ok])
    out = []
    for _ in range(n):
        pick = rng.choice(ub, len(ub), replace=True)
        meds = []
        for role in (1, 2):
            v = np.concatenate([dw["h_sample"][ok & (blk == b) & (dw["role"] == role)] for b in pick])
            if len(v) > 20:
                meds.append(np.median(v))
        if len(meds) == 2:
            out.append(float(np.mean(meds)))
    return {"h_boot_mean": float(np.mean(out)), "h_boot_sd": float(np.std(out)), "n_blocks": int(len(ub)), "n_boot": len(out)}


def main_plots() -> dict:
    RS.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    arr = dict(np.load(RS.CAM_DIR / "01.npz", allow_pickle=True))
    rig = json.loads((RS.CAM_DIR / "rig.json").read_text())
    dw = dict(np.load(RS.CAM_DIR / "dancers_world.npz"))
    mo = motion(arr)
    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(mo["t"], mo["f"], c="k")
    ax[0].set(ylabel="focal (px)", title="john_lula: the handheld camera over the 88 s (relative to the world frame)")
    for s in mo["steps_s"]:
        ax[0].axvline(s, c="tab:red", lw=0.8)
    ax[1].plot(mo["t"], mo["yaw"], c="tab:blue", label="yaw (0 = looking along -Z)")
    ax[1].plot(mo["t"], mo["pitch"], c="tab:orange", label="pitch (+ = down)")
    ax[1].legend(loc="upper right")
    ax[1].set(ylabel="deg")
    ax[2].plot(mo["t"], mo["roll"], c="tab:green")
    ax[2].set(ylabel="roll (deg)")
    rms = arr["reproj_px"]
    ax[3].plot(mo["t"], rms, c="tab:purple", lw=0.7)
    ax[3].plot(mo["t"][~arr["ok"]], rms[~arr["ok"]], "r.", ms=3, label="not ok")
    ax[3].set(ylabel="track rms (px)", xlabel="time (s)", ylim=(0, 8))
    ax[3].legend()
    fig.savefig(RS.REVIEW_DIR / "camera_motion.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    # focal table
    vp = RS.CAM_DIR / "uncertainty_variants.json"
    if vp.exists():
        var = json.loads(vp.read_text())["variants"]
        f1 = [v["f1x_px"] for v in var]
        fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
        ax[0].plot(f1, [v["rotation_fit_median_px"] for v in var], "o-")
        ax[0].set(title="pure-rotation fit (median px)", xlabel="1x focal (px)")
        ax[1].plot(f1, [v["lens_k1"] for v in var], "o-")
        ax[1].axhline(0, c="gray")
        ax[1].set(title="lens k1 at that focal (~0 expected)", xlabel="1x focal (px)")
        ax[2].plot(f1, [v["camera_height_m"] for v in var], "o-", label="dancers")
        ax[2].plot(f1, [v["population_height_m"] for v in var], "s--", label="everyone else (1.70 m)")
        ax[2].set(title="camera height (m)", xlabel="1x focal (px)", ylim=(1.0, 1.8))
        ax[2].legend()
        ax[3].plot(f1, [v["distance_to_dancers_m"] for v in var], "o-")
        ax[3].set(title="distance to the dancers (m)", xlabel="1x focal (px)")
        for a_ in ax:
            for x, lab, c in ((rig["moge"]["focal_model_px_1x"], "MoGe-2", "tab:green"), (912, "VGGT", "tab:red"), (rig["focal_1x_px"], "adopted", "k")):
                a_.axvline(x, c=c, lw=0.8, ls=":")
        fig.savefig(RS.REVIEW_DIR / "focal_table.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
    # plan view
    mp = RS.ROOM_DIR / "room_model.json"
    model = json.loads(mp.read_text()) if mp.exists() else None
    fig, ax = plt.subplots(figsize=(10, 9))
    sel = (dw["role"] > 0) & dw["upright"] | (dw["role"] > 0)
    ok1 = ~np.isin(dw["frame"], arr["frame_index"][arr["zoom_2x"]])
    for role, col, name in ((1, "tab:blue", "John's feet"), (2, "tab:red", "Lula's feet")):
        m = (dw["role"] == role) & ok1 & dw["trusted"]
        ax.scatter(dw["X"][m, 0], dw["X"][m, 2], s=3, c=col, alpha=0.4, label=name)
        m2 = (dw["role"] == role) & ok1 & ~dw["trusted"]
        ax.scatter(dw["X"][m2, 0], dw["X"][m2, 2], s=3, c="lightgray", alpha=0.5, label="picked, outside the cluster (wrong couple?)" if role == 1 else None)
    C = np.array(rig["centre_world_m"])
    ax.scatter([C[0]], [C[2]], marker="^", s=120, c="k", label=f"camera ({C[0]:.1f}, {C[1]:.2f} m high, {C[2]:.1f})")
    half = np.degrees(np.arctan(480 / rig["focal_1x_px"]))
    yaw = mo["yaw"]
    for y0 in (np.percentile(yaw, 2), np.percentile(yaw, 98)):
        for sg in (-1, 1):
            a_ = np.radians(y0 + sg * half * 0.56)           # horizontal half field of view of the 538 px width at 1x
            ax.plot([C[0], C[0] + 20 * np.sin(a_)], [C[2], C[2] - 20 * np.cos(a_)], c="gray", lw=0.8)
    if model:
        zf, xl = model["far_wall"]["plane"]["z"], model["left_wall"]["plane"]["x"]
        ax.plot([xl, model["floor"]["x"][1]], [zf, zf], c="tab:orange", lw=3, label=f"far wall z = {zf:.1f} m")
        ax.plot([xl, xl], [zf, model["floor"]["z"][1]], c="m", lw=3, label=f"left wall x = {xl:.1f} m")
    ax.scatter([0], [0], c="k", marker="x")
    ax.set_aspect("equal")
    ax.set_xlim(-14, 9)
    ax.set_ylim(7, -17)
    ax.set(xlabel="X (m)  [along the floor boards]", ylabel="Z (m)  [towards the camera]", title="john_lula hall, plan view (world frame, Y up). Wedges: the pan range at 1x")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(RS.REVIEW_DIR / "plan_view.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    # heights
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    hh = dw["h_sample"]
    good = np.isfinite(hh) & dw["upright"] & (hh > 0.6) & (hh < 2.6)
    for name, m, c in (("John", dw["role"] == 1, "tab:blue"), ("Lula", dw["role"] == 2, "tab:red"), ("everyone else (1.70 m)", dw["role"] == 0, "gray")):
        ax[0].hist(hh[good & m], bins=np.arange(0.9, 2.0, 0.02), alpha=0.5, color=c, label=f"{name} (n={int((good & m).sum())}, median {np.median(hh[good & m]):.2f})", density=True)
    ax[0].axvline(rig["camera_height_m"], c="k")
    ax[0].set(title="camera height from each upright person (m)", xlabel="m")
    ax[0].legend(fontsize=8)
    mg = rig["moge"]
    ax[1].bar(["people (adopted)", "MoGe-2's own metric"], [rig["camera_height_m"], mg["camera_height_model_m_median"]], color=["k", "tab:green"])
    ax[1].set(title="camera height: the people vs the model's own scale (m)")
    fig.savefig(RS.REVIEW_DIR / "heights.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    stats = {"motion": mo["stats"], "bootstrap_height": bootstrap(dw)}
    (RS.CAM_DIR / "motion_stats.json").write_text(json.dumps(RS._jsonable(stats), indent=1))
    return stats
