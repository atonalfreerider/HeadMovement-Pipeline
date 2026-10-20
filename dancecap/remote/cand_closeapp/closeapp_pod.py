#!/usr/bin/env python
"""Pod side of mono4d track "closeapp" (dancecap.mono4d.cand_closeapp): CloseApp (boycehbz/CloseApp, CVPR 2025) on our
120 frames of phone 06.

    python remote/cand_closeapp/closeapp_pod.py run --input <dir> --smpl <dir> --out <dir> [--variants closeapp,closeapp_noapp]

run:
  1. data      the owner's SMPL files (plain-numpy copies) -> CloseApp/data/smpl/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl;
               our sequence -> CloseApp/data/preprocess_data/lk38_06/train/ in the layout of the release's demo sequence
               04305 (images/, masks/0, masks/1, smpl_parms.pth with the demo's keys, cam_parms.npz, and the UV / joint
               files the official wild preprocessing copies into every sequence, taken from the demo's train/)
  2. train     per variant, CloseApp's own train.py in a fresh process (train-entry): the release demo flags
               (--train_stage=1 [--use_appearance]) minus --save_render / --save_params, which only drive the moderngl
               mesh renders after the last epoch (the epoch checkpoints hold every optimised parameter)
  3. export    (export-entry, fresh process) per variant + the initialisation: the optimised SMPL parameters, FK joints
               (CloseApp's bundled smplx.SMPL, the model it optimises with), CloseApp's own A-PD metric (SDFLoss, as in
               AvatarModel.eval_smpl) and the BVH inter-person triangle collisions per frame
  4. results   closeapp_out.npz + closeapp_pod.json (+ logs/, renders/: a few of the epoch-20 appearance renders)

Import stand-ins (only when the real import fails; every one is listed in closeapp_pod.json "stubs"): modules that the
optimisation path imports but never calls - open3d (train.py), skvideo (utils.video_processing), scipy.misc
(utils.imutils), aitviewer_core (utils.renderer_moderngl: mesh visualisation), pytorch3d.transforms (arguments/:
replaced by CloseApp's own utils.rotation_conversions, a copy of pytorch3d's) - and, only if their CUDA builds failed,
bvh_cuda (no-collision result: the penetration term has weight 0.000 in the release, so the optimisation is unchanged)
and sdf (SDFLoss constructed but only evaluated with ground truth; the A-PD metric is then reported as missing).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("CLOSEAPP_ROOT", "/workspace/closeapp_cand"))
REPO = ROOT / "CloseApp"
SEQ = "lk38_06"
FLAGS = {"closeapp": ["--train_stage=1", "--use_appearance"], "closeapp_noapp": ["--train_stage=1"]}
TRAIN_TIMEOUT_MIN = {"closeapp": 13.0, "closeapp_noapp": 6.0}   # paper: 3-5 min per 128 frames on an RTX 3090


def log(msg: str) -> None:
    print(f"[closeapp_pod {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ import stand-ins


def _mod(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    m.__closeapp_stub__ = True
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, m)
    return m


class _Dummy:
    def __init__(self, *a, **k):
        raise RuntimeError("stand-in class: visualisation module not installed on this pod")


def install_stubs() -> dict:
    """Stand-ins for modules whose real import fails (see the module docstring). Returns {module: reason}."""
    import torch  # noqa: F401  (torch first: the CUDA extensions link against its libraries)
    out = {}

    def ok(name):
        try:
            importlib.import_module(name)
            return None
        except Exception as e:  # noqa: BLE001
            for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
                sys.modules.pop(k, None)
            return f"{type(e).__name__}: {e}"[:160]

    for name in ("open3d", "skvideo"):
        r = ok(name)
        if r:
            _mod(name)
            if name == "skvideo":
                _mod("skvideo.io")
            out[name] = r
    r = ok("scipy.misc")
    if r:
        import scipy
        _mod("scipy.misc")
        scipy.misc = sys.modules["scipy.misc"]
        out["scipy.misc"] = r
    r = ok("pytorch3d.transforms")
    if r:
        rc = importlib.import_module("utils.rotation_conversions")
        _mod("pytorch3d")
        _mod("pytorch3d.transforms", **{k: v for k, v in rc.__dict__.items() if not k.startswith("__")})
        out["pytorch3d.transforms"] = r + " -> CloseApp utils.rotation_conversions"
    r = ok("aitviewer_core.aitviewer.viewer")
    if r:
        for k in [k for k in sys.modules if k.startswith("aitviewer_core")]:
            sys.modules.pop(k, None)
        _mod("aitviewer_core")
        _mod("aitviewer_core.aitviewer")
        _mod("aitviewer_core.aitviewer.viewer", Viewer=_Dummy)
        _mod("aitviewer_core.aitviewer.scene")
        _mod("aitviewer_core.aitviewer.scene.camera", OpenCVCamera=_Dummy)
        _mod("aitviewer_core.aitviewer.scene.material", Material=_Dummy)
        _mod("aitviewer_core.aitviewer.renderables")
        _mod("aitviewer_core.aitviewer.renderables.meshes", Meshes=_Dummy)
        _mod("aitviewer_core.aitviewer.renderables.billboard", Billboard=_Dummy)
        out["aitviewer_core"] = r
    r = ok("bvh_cuda")
    if r:
        def forward(triangles, max_collisions=16):
            return -torch.ones(triangles.shape[0], triangles.shape[1] * max_collisions, 2, dtype=torch.long,
                               device=triangles.device)
        _mod("bvh_cuda", forward=forward)
        out["bvh_cuda"] = r + " -> no-collision stand-in (penetration weight is 0 in the release)"
    r = ok("sdf")
    if r:
        import torch.nn as nn

        class SDFLoss(nn.Module):
            def __init__(self, *a, **k):
                super().__init__()

            def forward(self, *a, **k):
                raise RuntimeError("sdf extension not built")
        _mod("sdf", SDFLoss=SDFLoss)
        out["sdf"] = r + " -> stand-in (A-PD metric unavailable)"
    return out


def _enter_repo() -> dict:
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return install_stubs()


def train_entry(args: list[str]) -> None:
    stubs = _enter_repo()
    print("[closeapp_pod] stubs " + json.dumps(stubs), flush=True)
    import runpy
    sys.argv = ["train.py"] + args
    runpy.run_path(str(REPO / "train.py"), run_name="__main__")


# ------------------------------------------------------------------------------------------------ data


def build_sequence(inp: Path, smpl_dir: Path) -> dict:
    import torch
    d = REPO / "data" / "smpl" / "smpl"
    d.mkdir(parents=True, exist_ok=True)
    for g in ("NEUTRAL", "MALE", "FEMALE"):
        shutil.copy2(smpl_dir / f"SMPL_{g}.pkl", d / f"SMPL_{g}.pkl")
    z = np.load(inp / "closeapp_input.npz", allow_pickle=True)
    names = [str(n) for n in z["names"]]
    T = len(names)
    seq = REPO / "data" / "preprocess_data" / SEQ / "train"
    if seq.exists():
        shutil.rmtree(seq)
    for sub in ("images", "masks/0", "masks/1"):
        (seq / sub).mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copy2(inp / "images" / f"{n}.jpg", seq / "images" / f"{n}.jpg")
        for a in ("0", "1"):
            shutil.copy2(inp / "masks" / a / f"{n}.png", seq / "masks" / a / f"{n}.png")
    K = np.asarray(z["K"], np.float32)
    W, H = int(z["img_w"]), int(z["img_h"])
    f32 = lambda a: torch.tensor(np.asarray(a, np.float32))  # noqa: E731
    pose = np.asarray(z["init_pose"], np.float32)                                   # (T, 2, 72)
    betas = np.repeat(np.asarray(z["init_betas"], np.float32)[None], T, 0)          # (T, 2, 10)
    data = {"pred_bbox": f32(z["boxes"]), "pred_pose": f32(pose), "pred_trans": f32(z["init_transl"]),
            "pred_shape": f32(betas), "gender": ["neutral", "neutral"], "features": torch.zeros(T, 2, 1024),
            "init_pose": f32(pose), "img_w": torch.full((T, 2), float(W)), "img_h": torch.full((T, 2), float(H)),
            "focal_length": torch.full((T, 2), float(K[0, 0])), "center": f32(z["centers"]), "scale": f32(z["scales"]),
            "keypoints": f32(z["keypoints"]), "pred_keypoints": f32(z["keypoints"]), "valid": torch.ones(T, 2),
            "contact": None}
    torch.save(data, seq / "smpl_parms.pth")
    np.savez(seq / "cam_parms.npz", intrinsic=K, extrinsic=np.eye(4, dtype=np.float32))
    demo = REPO / "data" / "preprocess_data" / "04305" / "train"
    for f in ("query_posemap_128_cano_smpl.npz", "query_posemap_256_cano_smpl.npz", "query_posemap_512_cano_smpl.npz",
              "smpl_cano_joint_mat.pth"):
        shutil.copy2(demo / f, seq / f)
    return {"frames": T, "K": K.tolist(), "img": [W, H], "seq": str(seq)}


# ------------------------------------------------------------------------------------------------ export


def _latest_ckpt(model_dir: Path) -> Path | None:
    best, best_ep = None, -1
    for p in model_dir.rglob("net/*.pth"):
        m = re.search(r"_net_(\d+)\.pth$", p.name)
        if m and int(m.group(1)) > best_ep:
            best, best_ep = p, int(m.group(1))
    return best


def export_entry(inp: Path, out: Path, variants: list[str]) -> None:
    stubs = _enter_repo()
    import torch
    from smplx import SMPL  # CloseApp's bundled smplx (the model AvatarModel optimises with)
    z = np.load(inp / "closeapp_input.npz", allow_pickle=True)
    T = len(z["names"])
    res, arrays = {"stubs": stubs, "variants": {}}, {}
    params = {"closeapp_init": (np.asarray(z["init_pose"], np.float32).transpose(1, 0, 2),
                                np.asarray(z["init_transl"], np.float32).transpose(1, 0, 2),
                                np.asarray(z["init_betas"], np.float32))}
    for v in variants:
        ck = _latest_ckpt(REPO / "output" / v)
        if ck is None:
            res["variants"][v] = {"error": "no checkpoint"}
            continue
        st = torch.load(ck, map_location="cpu", weights_only=False)
        P = np.stack([st[f"pose_{a}"]["weight"].numpy() for a in (0, 1)])
        Tr = np.stack([st[f"transl_{a}"]["weight"].numpy() for a in (0, 1)])
        B = np.stack([st[f"betas_{a}"]["weight"].numpy()[0] for a in (0, 1)])
        params[v] = (P, Tr, B)
        res["variants"][v] = {"checkpoint": str(ck.relative_to(REPO))}
    model = SMPL(model_path="data/smpl/smpl", gender="neutral", batch_size=T)
    faces = np.asarray(model.faces, np.int64)
    sdf_loss, bvh = None, None
    try:
        from sdf import SDFLoss
        sdf_loss = SDFLoss(faces, faces, robustifier=None).cuda()
    except Exception as e:  # noqa: BLE001
        res["sdf_error"] = f"{type(e).__name__}: {e}"[:200]
    try:
        from utils.mesh_intersection.bvh_search_tree import BVH
        bvh = BVH(max_collisions=8)
    except Exception as e:  # noqa: BLE001
        res["bvh_error"] = f"{type(e).__name__}: {e}"[:200]
    nf = len(faces)
    for name, (P, Tr, B) in params.items():
        f = lambda a: torch.tensor(np.asarray(a, np.float32))  # noqa: E731
        verts, joints = [], []
        with torch.no_grad():
            for a in (0, 1):
                o = model(betas=f(B[a]).reshape(1, -1).expand(T, -1), global_orient=f(P[a][:, :3]),
                          body_pose=f(P[a][:, 3:]), transl=f(Tr[a]))
                verts.append(o.vertices)
                joints.append(o.joints[:, :24].numpy())
        arrays[f"{name}_pose"], arrays[f"{name}_transl"], arrays[f"{name}_betas"] = P, Tr, B
        arrays[f"{name}_joints"] = np.stack(joints)
        r = res["variants"].setdefault(name, {})
        if sdf_loss is not None and "sdf" not in stubs:
            try:
                hv = torch.stack(verts, 1).cuda()
                _, _, los = sdf_loss(hv, return_per_vert_loss=True, return_origin_scale_loss=True)
                per = 1000 * los.detach().cpu().numpy().mean(axis=1)
                r["A_PD"] = float(per.mean())
                r["A_PD_per_frame_median"] = float(np.median(per))
                arrays[f"{name}_apd"] = per
            except Exception as e:  # noqa: BLE001
                r["A_PD_error"] = f"{type(e).__name__}: {e}"[:200]
        if bvh is not None and "bvh_cuda" not in stubs:
            try:
                Fc = torch.tensor(np.concatenate([faces, faces + 6890]), device="cuda")
                cnt = []
                for k in range(T):
                    V = torch.cat([verts[0][k], verts[1][k]], 0).cuda()
                    c = bvh(V[Fc][None].contiguous())[0]
                    c = c[c[:, 0] >= 0]
                    cnt.append(int(((c[:, 0] < nf) != (c[:, 1] < nf)).sum()))
                arrays[f"{name}_bvh_inter"] = np.array(cnt)
                r["bvh_inter_collisions_median"] = float(np.median(cnt))
                r["bvh_inter_collisions_frames_gt0"] = int((np.array(cnt) > 0).sum())
            except Exception as e:  # noqa: BLE001
                r["bvh_error"] = f"{type(e).__name__}: {e}"[:200]
    np.savez(out / "closeapp_out.npz", names=z["names"], t_ref=z["t_ref"], **arrays)
    (out / "export.json").write_text(json.dumps(res, indent=1))
    log(f"export -> {out / 'closeapp_out.npz'}: {sorted(params)}")


# ------------------------------------------------------------------------------------------------ run


def _parse_train_log(text: str) -> dict:
    tot = [float(m.group(1)) for m in re.finditer(r"Total: ([-\d.eE+]+), Motion: ([-\d.eE+]+)", text)]
    ep = [float(m.group(1)) for m in re.finditer(r"Epoch Time ([\d.]+)s", text)]
    return {"epochs": len(ep), "epoch_s_median": float(np.median(ep)) if ep else None,
            "total_loss_first_last": [tot[0], tot[-1]] if tot else None}


def run(inp: Path, smpl_dir: Path, out: Path, variants: list[str], train_timeout_min: float | None) -> int:
    import torch
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "renders").mkdir(exist_ok=True)
    t0 = time.time()
    rep = {"versions": {"torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0],
                        "closeapp": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                                   text=True).stdout.strip()},
           "gpu": torch.cuda.get_device_name(0), "timings_s": {}, "variants": {}}
    st = ROOT / "install_status.json"
    rep["builds"] = json.loads(st.read_text()) if st.exists() else None
    rep["sequence"] = build_sequence(inp, smpl_dir)
    rep["timings_s"]["data"] = round(time.time() - t0, 1)
    me = str(Path(__file__).resolve())
    for v in variants:
        md = REPO / "output" / v
        if md.exists():
            shutil.rmtree(md)
        args = ["-s", f"data/preprocess_data/{SEQ}", "-m", f"output/{v}"] + FLAGS[v]
        tv = time.time()
        logf = out / "logs" / f"train_{v}.log"
        log(f"train {v}: {' '.join(args)}")
        with open(logf, "w") as fh:
            try:
                p = subprocess.run([sys.executable, me, "train-entry", "--"] + args, stdout=fh, stderr=subprocess.STDOUT,
                                   timeout=(train_timeout_min or TRAIN_TIMEOUT_MIN[v]) * 60, cwd=str(REPO))
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        text = logf.read_text(errors="replace")
        r = {"flags": FLAGS[v], "rc": rc, "wall_s": round(time.time() - tv, 1), **_parse_train_log(text)}
        m = re.search(r"\[closeapp_pod\] stubs (\{.*\})", text)
        r["stubs"] = json.loads(m.group(1)) if m else None
        if rc != 0:
            r["log_tail"] = text[-3000:]
        mem = re.findall(r"CUDA out of memory.*", text)
        if mem:
            r["oom"] = mem[0][:300]
        rep["variants"][v] = r
        log(f"train {v}: rc {rc}, {r['wall_s']} s, epochs {r['epochs']}, loss {r['total_loss_first_last']}")
        rd = next(iter(sorted(md.rglob("log/epoch00020"))), None) if md.exists() else None
        if rd is not None:
            import cv2
            files = sorted(rd.glob("*.jpg"))
            for fpath in [files[i] for i in sorted({0, len(files) // 3, 2 * len(files) // 3, len(files) - 1})] if files else []:
                im = cv2.imread(str(fpath))
                cv2.imwrite(str(out / "renders" / f"{v}_{fpath.stem}.jpg"), cv2.resize(im, None, fx=0.5, fy=0.5),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
            for tex in sorted(rd.parent.glob("epoch00020_c0*_f*.jpg")):
                shutil.copy2(tex, out / "renders" / f"{v}_texture_{tex.name}")
        for lg in md.rglob("log.txt") if md.exists() else []:
            shutil.copy2(lg, out / "logs" / f"closeapp_{v}_log.txt")
    rep["timings_s"]["train"] = round(time.time() - t0 - rep["timings_s"]["data"], 1)
    te = time.time()
    with open(out / "logs" / "export.log", "w") as fh:
        p = subprocess.run([sys.executable, me, "export-entry", "--input", str(inp.resolve()), "--out", str(out.resolve()),
                            "--variants", ",".join(variants)], stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO),
                           timeout=15 * 60)
    rep["timings_s"]["export"] = round(time.time() - te, 1)
    ex = out / "export.json"
    if ex.exists():
        e = json.loads(ex.read_text())
        rep["stubs"] = e.get("stubs")
        for k in ("sdf_error", "bvh_error"):
            if k in e:
                rep[k] = e[k]
        for v, r in e["variants"].items():
            rep["variants"].setdefault(v, {}).update(r)
    else:
        rep["export_error"] = (out / "logs" / "export.log").read_text(errors="replace")[-3000:]
    rep["export_rc"] = p.returncode
    rep["timings_s"]["total"] = round(time.time() - t0, 1)
    (out / "closeapp_pod.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"done in {rep['timings_s']['total']} s")
    ok = p.returncode == 0 and any(rep["variants"].get(v, {}).get("rc") == 0 for v in variants)
    return 0 if ok else 1


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "train-entry":
        train_entry(sys.argv[3:] if len(sys.argv) > 2 and sys.argv[2] == "--" else sys.argv[2:])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "export-entry"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--smpl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="closeapp,closeapp_noapp")
    ap.add_argument("--train-timeout-min", type=float, help="per variant (default TRAIN_TIMEOUT_MIN)")
    a = ap.parse_args()
    variants = [v for v in a.variants.split(",") if v]
    if a.action == "export-entry":
        export_entry(Path(a.input), Path(a.out), variants)
        return
    sys.exit(run(Path(a.input).resolve(), Path(a.smpl), Path(a.out).resolve(), variants, a.train_timeout_min))


if __name__ == "__main__":
    main()
