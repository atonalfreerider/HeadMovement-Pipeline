"""Pod side of the Kimodo gap-fill test (dancecap.mono4d.cand_kimodo; runs in /workspace/kimodo_cand/venv).

Input  (prep): per role the 3-s window 58-61 s of the 8-phone reference fit retargeted onto Kimodo's somaskel30
       (<role>_pos (T,30,3) world metres, <role>_grot (T,30,3,3) global rotations, <role>_hips), the frame index
       sets gap / ctx / sparse and soma_rest (the rest joints the local conversion used).
Output (--out): <role>_<variant>_<raw|pp>.npz in Kimodo's canonical frame (posed_joints (S,T,30,3), global_rot_mats,
       local_rot_mats, root_positions, foot_contacts, target_pos (T,30,3) = the constrained positions, canon_R / canon_o:
       X_canon = canon_R @ (X_world - canon_o)) + gapfill.json (versions, timings, checks, errors).

Constraint-only: ZeroTextEncoder stands in for LLM2Vec (gated Meta-Llama-3-8B); the prompt is '' so Kimodo zeroes the
text features and their mask anyway (kimodo_model._generate), i.e. the classifier-free 'unconditional text' case.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

VARIANTS = {  # name: (keyframe set, dense oracle root path through the whole window incl. the gap)
    "dense": ("ctx", False),
    "dense_root": ("ctx", True),
    "sparse": ("sparse", False),
    "sparse_root": ("sparse", True),
}


def log(*a):
    print("[kimodo-pod]", time.strftime("%H:%M:%S"), *a, flush=True)


class ZeroTextEncoder:
    """Returns (zeros [B, 1, 4096], lengths); with prompt '' Kimodo sets the features and their mask to zero itself."""

    llm_dim = 4096

    def __call__(self, texts):
        return torch.zeros(len(texts), 1, self.llm_dim), [1] * len(texts)

    def to(self, *a, **k):
        return self

    def eval(self):
        return self


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Kimodo-SOMA-RP-v1.2")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", type=float, nargs=2, default=[2.0, 2.0])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = {"args": vars(a), "errors": [], "timings": {}, "checks": {}}
    t_all = time.time()

    def save_info():
        info["timings"]["total_s"] = round(time.time() - t_all, 1)
        (out / "gapfill.json").write_text(json.dumps({"info": info, "variants": sorted(done)}, indent=1, default=str))

    done: set[str] = set()
    torch.backends.cuda.matmul.allow_tf32 = False   # Kimodo issue #52: TF32 changes results
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for f in ("/workspace/kimodo_cand/pip_freeze.txt", "/workspace/kimodo_cand/pip_full.log"):
        if os.path.exists(f):
            shutil.copy(f, out / Path(f).name)
    try:
        import kimodo
        from kimodo import load_model
        from kimodo.constraints import FullBodyConstraintSet, Root2DConstraintSet
        from kimodo.model.registry import AVAILABLE_MODELS
        from kimodo.motion_rep.feature_utils import compute_heading_angle
        from kimodo.motion_rep.smooth_root import get_smooth_root_pos
        from kimodo.postprocess import post_process_motion
        info["versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                            "kimodo_file": kimodo.__file__, "commit": os.environ.get("KIMODO_COMMIT"),
                            "hf_repo": os.environ.get("KIMODO_HF"), "hf_rev": os.environ.get("KIMODO_HF_REV"),
                            "available_models": list(AVAILABLE_MODELS)}
        try:
            import motion_correction._motion_correction  # noqa: F401
            have_pp = True
        except Exception as e:
            have_pp = False
            info["errors"].append(f"motion_correction import: {e}")
        info["post_processing_available"] = have_pp
        t0 = time.time()
        model = load_model(a.model, device="cuda", text_encoder=ZeroTextEncoder())
        info["timings"]["load_s"] = round(time.time() - t0, 1)
        skel = model.skeleton
        info["skeleton"] = {"class": type(skel).__name__, "names": list(skel.bone_order_names), "fps": model.fps}
        log("model", a.model, type(skel).__name__, "fps", model.fps)
    except Exception:
        info["errors"].append("setup: " + traceback.format_exc())
        save_info()
        log(info["errors"][-1])
        return 2

    d = np.load(a.inputs, allow_pickle=True)
    rest = torch.as_tensor(d["soma_rest"], dtype=torch.float32)
    nj = skel.neutral_joints.detach().cpu().float()
    info["checks"]["soma_rest_max_diff_m"] = float((nj - rest).abs().max())
    info["checks"]["names_match"] = list(skel.bone_order_names) == [
        "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye", "LeftShoulder",
        "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd", "RightShoulder", "RightArm",
        "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd", "LeftLeg", "LeftShin", "LeftFoot",
        "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase"]
    log("checks", info["checks"])
    gap, ctx, sparse = (np.asarray(d[k], int) for k in ("gap", "ctx", "sparse"))
    sel = {"ctx": ctx, "sparse": sparse}
    s30 = skel.get_skel_slice(skel.somaskel77)
    dev = "cuda"
    for role in ("lead", "follow"):
        P = torch.as_tensor(d[f"{role}_pos"], dtype=torch.float32)
        G = torch.as_tensor(d[f"{role}_grot"], dtype=torch.float32)
        T = len(P)
        # input check: FK of Kimodo's own skeleton on our local rotations reproduces our positions
        try:
            L = torch.as_tensor(d[f"{role}_lrot"], dtype=torch.float32)
            _, pk, _ = skel.fk(L.to(dev), P[:, 0].to(dev))
            info["checks"][f"{role}_fk_vs_prep_max_m"] = float((pk.cpu() - P).abs().max())
        except Exception as e:
            info["errors"].append(f"{role} fk check: {e}")
        hips = P[:, 0].clone()
        a0, b0 = int(gap[0] - 1), int(gap[-1] + 1)
        w = torch.as_tensor((gap - a0) / (b0 - a0), dtype=torch.float32)[:, None]
        hips_noleak = hips.clone()
        hips_noleak[gap] = (1 - w) * hips[a0] + w * hips[b0]   # the gap's pelvis is never seen by 'no root' variants
        smooth = {False: get_smooth_root_pos(hips_noleak[None])[0].float(), True: get_smooth_root_pos(hips[None])[0].float()}
        h0 = float(compute_heading_angle(P[None], skel)[0, 0])
        for vname, (kset, oracle) in VARIANTS.items():
            try:
                sm = smooth[oracle]
                o = np.array([float(sm[0, 0]), 0.0, float(sm[0, 2])])
                Rc = ry(-h0)
                Pc = torch.as_tensor(np.einsum("ij,tnj->tni", Rc, P.numpy() - o), dtype=torch.float32)
                Gc = torch.as_tensor(np.einsum("ij,tnjk->tnik", Rc, G.numpy()), dtype=torch.float32)
                smc = torch.as_tensor(np.einsum("ij,tj->ti", Rc, sm.numpy() - o), dtype=torch.float32)
                hc = float(compute_heading_angle(Pc[None], skel)[0, 0])
                if abs(hc) > 1e-3:
                    raise RuntimeError(f"canonical heading {hc:.4f} rad != 0 (rotation sign)")
                idx = sel[kset]
                cons = [FullBodyConstraintSet(skel, torch.as_tensor(idx), Pc[idx].to(dev), Gc[idx].to(dev),
                                              smooth_root_2d=smc[idx][:, [0, 2]].to(dev))]
                if oracle:
                    cons.append(Root2DConstraintSet(skel, torch.arange(T), smc[:, [0, 2]].to(dev)))
                cons = [c.to(dev) for c in cons]
                torch.manual_seed(a.seed)
                t0 = time.time()
                res = model("", num_frames=T, num_denoising_steps=a.steps, constraint_lst=cons, cfg_weight=list(a.cfg),
                            num_samples=a.samples, return_numpy=False, first_heading_angle=0.0, post_processing=False)
                torch.cuda.synchronize()
                info["timings"][f"{role}_{vname}_gen_s"] = round(time.time() - t0, 2)
                l30 = res["local_rot_mats"][:, :, s30]
                root = res["root_positions"]
                if not (torch.isfinite(l30).all() and torch.isfinite(root).all()):
                    raise RuntimeError("non-finite Kimodo output (NaN/inf)")
                S = l30.shape[0]
                g30, p30, _ = skel.fk(l30.reshape(S * T, 30, 3, 3), root.reshape(S * T, 3))
                fc6 = res["foot_contacts"]
                fc4 = fc6[..., [0, 1, 3, 4]] if fc6.shape[-1] == 6 else fc6
                common = {"canon_R": Rc, "canon_o": o, "heading0": h0, "target_pos": Pc.numpy(), "keyframes": idx,
                          "oracle_root": oracle, "smooth_root_canon": smc.numpy()}
                np.savez_compressed(out / f"{role}_{vname}_raw.npz", posed_joints=p30.reshape(S, T, 30, 3).cpu().numpy(),
                                    global_rot_mats=g30.reshape(S, T, 30, 3, 3).cpu().numpy(),
                                    local_rot_mats=l30.cpu().numpy(), root_positions=root.cpu().numpy(),
                                    foot_contacts=fc4.cpu().numpy(), **common)
                done.add(vname)
                kf = float((p30.reshape(S, T, 30, 3)[:, idx].cpu() - Pc[idx]).norm(dim=-1).mean())
                info["checks"][f"{role}_{vname}_raw_keyframe_err_m"] = kf
                log(role, vname, "raw ok", f"{info['timings'][f'{role}_{vname}_gen_s']} s, keyframe err {kf * 1e3:.1f} mm")
                if have_pp:
                    t0 = time.time()
                    cor = post_process_motion(l30, root, fc4, skel, cons)
                    info["timings"][f"{role}_{vname}_pp_s"] = round(time.time() - t0, 2)
                    np.savez_compressed(out / f"{role}_{vname}_pp.npz", posed_joints=cor["posed_joints"].cpu().numpy(),
                                        global_rot_mats=cor["global_rot_mats"].cpu().numpy(),
                                        local_rot_mats=cor["local_rot_mats"].cpu().numpy(),
                                        root_positions=cor["root_positions"].cpu().numpy(), foot_contacts=fc4.cpu().numpy(),
                                        **common)
                    kf = float((cor["posed_joints"][:, idx].cpu() - Pc[idx]).norm(dim=-1).mean())
                    info["checks"][f"{role}_{vname}_pp_keyframe_err_m"] = kf
                    log(role, vname, "pp ok", f"keyframe err {kf * 1e3:.1f} mm")
            except Exception:
                info["errors"].append(f"{role} {vname}: " + traceback.format_exc())
                log(info["errors"][-1])
            save_info()
    info["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    save_info()
    log("done", len(done), "variants;", len(info["errors"]), "errors")
    return 0 if done else 3


if __name__ == "__main__":
    raise SystemExit(main())
