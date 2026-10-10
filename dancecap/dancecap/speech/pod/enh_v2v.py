"""Speech enhancers on the pod for the V2 re-recording (agent V2): one tool per call, each from its own venv (/opt/enh/<venv>).

    /opt/enh/dfn/bin/python enh_v2v.py dfn3        IN_DIR OUT_DIR       DeepFilterNet3 (48 kHz native)
    /opt/enh/clearvoice/bin/python enh_v2v.py mossformer48 IN_DIR OUT_DIR   ClearerVoice-Studio MossFormer2_SE_48K  (also: frcrn16, mfgan16)
    /opt/enh/fixer/bin/python enh_v2v.py voicefixer IN_DIR OUT_DIR      VoiceFixer (restoration vocoder; mode 0 / 1 / 2 -> voicefixer0..2)
    /opt/enh/sep/bin/python enh_v2v.py uvr:<model file> IN_DIR OUT_DIR  audio-separator (UVR / roformer models, e.g. UVR-DeEcho-DeReverb.pth)

IN_DIR holds 48 kHz mono wavs; OUT_DIR gets a 48 kHz mono 16-bit wav of the same name for each. A failing file is reported and skipped.
"""
from __future__ import annotations

import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def to48(x: np.ndarray, sr: int) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=1) if x.shape[1] <= 8 else x.mean(axis=0)
    if sr != 48000:
        g = gcd(48000, sr)
        x = signal.resample_poly(x, 48000 // g, sr // g)
    return x.astype(np.float32)


def save(path: Path, x: np.ndarray, sr: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = to48(x, sr)
    pk = float(np.abs(x).max())
    if pk > 0.99:
        x = x * (0.99 / pk)
    sf.write(str(path), x, 48000, subtype="PCM_16")


def run_dfn3(files, out):
    from df.enhance import enhance, init_df
    import torch
    model, state, _ = init_df()
    for f in files:
        x, sr = sf.read(str(f), dtype="float32")
        assert sr == 48000
        y = enhance(model, state, torch.from_numpy(x)[None, :])
        save(out / f.name, y.squeeze().cpu().numpy())


def run_clearvoice(files, out, model):
    from clearvoice import ClearVoice
    cv = ClearVoice(task="speech_enhancement", model_names=[model])
    for f in files:
        y = cv(input_path=str(f), online_write=False)
        y = np.asarray(y)
        sr = 48000 if "48K" in model else 16000
        save(out / f.name, y.squeeze(), sr)


def run_voicefixer(files, out, mode):
    from voicefixer import VoiceFixer
    vf = VoiceFixer()
    for f in files:
        tmp = out / ("_" + f.name)
        out.mkdir(parents=True, exist_ok=True)
        vf.restore(input=str(f), output=str(tmp), cuda=True, mode=mode)
        x, sr = sf.read(str(tmp), dtype="float32")
        save(out / f.name, x, sr)
        tmp.unlink()


def run_uvr(files, out, model):
    from audio_separator.separator import Separator
    tmp = out.parent / (out.name + "_raw")
    tmp.mkdir(parents=True, exist_ok=True)
    sep = Separator(output_dir=str(tmp), output_format="WAV", model_file_dir="/opt/enh/models")
    sep.load_model(model_filename=model)
    for f in files:
        names = sep.separate(str(f))
        # the wanted stem is named in parentheses: "(No Reverb)", "(dry)", "(No Echo)", "(Vocals)", "(No Noise)" (never "(Reverb)", "(No Dry)", "(Noise)", "(other)")
        low = [n.lower() for n in names]
        pick = next((names[i] for key in ("(no reverb)", "(noreverb)", "(dry)", "(no echo)", "(no noise)", "(vocals)") for i, n in enumerate(low) if key in n), names[0])
        x, sr = sf.read(str(tmp / pick), dtype="float32")
        save(out / f.name, x, sr)


def main(argv=None) -> int:
    tool, indir, outdir = sys.argv[1:4] if argv is None else argv[:3]
    files = sorted(Path(indir).glob("*.wav"))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if tool == "dfn3":
        run_dfn3(files, out)
    elif tool == "mossformer48":
        run_clearvoice(files, out, "MossFormer2_SE_48K")
    elif tool == "frcrn16":
        run_clearvoice(files, out, "FRCRN_SE_16K")
    elif tool == "mfgan16":
        run_clearvoice(files, out, "MossFormerGAN_SE_16K")
    elif tool.startswith("voicefixer"):
        run_voicefixer(files, out, int(tool[len("voicefixer"):] or 0))
    elif tool.startswith("uvr:"):
        run_uvr(files, out, tool[4:])
    else:
        raise SystemExit(f"unknown tool {tool}")
    print(f"{tool}: {len(list(out.glob('*.wav')))} files in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
