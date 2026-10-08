"""Native file dialog for the monitor's picker, run as a short subprocess (tkinter needs its own main thread).

    python -m dancecap.pipeline.monitor.pick --kind videos|soundtrack|portrait|folder [--initial DIR]

Prints one JSON line {"files": [...absolute paths...]} (empty on cancel). The browser never sees a dropped file's
real path, so picking goes through here; the server turns the paths into handles.
"""
from __future__ import annotations

import argparse
import json
import sys

KINDS = {
    "videos": ("Choose the source videos", [("Videos", "*.mp4 *.mov *.m4v *.mkv *.avi *.webm *.3gp"), ("All", "*.*")], True),
    "soundtrack": ("Choose the soundtrack", [("Audio", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus"), ("All", "*.*")], False),
    "portrait": ("Choose a portrait", [("Images", "*.jpg *.jpeg *.png *.webp"), ("All", "*.*")], False),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=[*KINDS, "folder"], required=True)
    ap.add_argument("--initial")
    a = ap.parse_args(argv)
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if a.kind == "folder":
            d = filedialog.askdirectory(title="Choose the folder with the videos", initialdir=a.initial or None,
                                        parent=root)
            files = [d] if d else []
        else:
            title, types, multi = KINDS[a.kind]
            if multi:
                files = list(filedialog.askopenfilenames(title=title, filetypes=types, initialdir=a.initial or None,
                                                         parent=root))
            else:
                f = filedialog.askopenfilename(title=title, filetypes=types, initialdir=a.initial or None, parent=root)
                files = [f] if f else []
    finally:
        root.destroy()
    sys.stdout.write(json.dumps({"files": files}, ensure_ascii=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
