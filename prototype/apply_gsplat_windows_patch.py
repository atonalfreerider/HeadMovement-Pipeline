"""Idempotent MSVC compiler-flag correction for gsplat 1.5.3's JIT backend."""
from pathlib import Path
import gsplat

path=Path(gsplat.__file__).parent/'cuda/_backend.py'
old='extra_cflags = [opt_level, "-Wno-attributes"]'
new='extra_cflags = (["/Od" if FAST_COMPILE else "/O2"] if os.name == "nt"\n                        else [opt_level, "-Wno-attributes"])'
text=path.read_text()
if old in text:
    path.write_text(text.replace(old,new))
    print('Applied Windows compiler flags:',path)
elif new in text:
    print('Windows compiler flags already patched')
else:
    raise RuntimeError('Unrecognized gsplat backend: review before changing it')
