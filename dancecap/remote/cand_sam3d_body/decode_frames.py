"""Decode the frames a SAM 3D Body input lists (input/meta.json "frames" [{file, i}], in decode order) from the source video, on
the pod: the job uploads the video (MBs) instead of the take's JPEGs (hundreds of MB; PIPELINE_EFFICIENCY.md rank 4). Frame i of
meta.json = the i-th decoded frame of the video (the one-camera clock: cameras/01.npz frame_index = video frame index).

    python decode_frames.py VIDEO META_JSON OUT_DIR [--quality 95]
"""
import argparse
import json
import os
import sys
import time


def main(argv=None) -> int:
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("meta")
    ap.add_argument("out")
    ap.add_argument("--quality", type=int, default=95)
    a = ap.parse_args(argv)
    t0 = time.time()
    rows = json.load(open(a.meta))["frames"]
    os.makedirs(a.out, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    n = 0
    for k, r in enumerate(rows):
        if int(r.get("i", k)) != k:
            raise SystemExit(f"meta.json row {k} has i={r.get('i')}: the frames must be the video's frames 0..F-1 in order")
        ok, im = cap.read()
        if not ok:
            raise SystemExit(f"the video ended after {n} frames; meta.json lists {len(rows)}")
        cv2.imwrite(os.path.join(a.out, r["file"]), im, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        n += 1
    cap.release()
    print(f"decoded {n} frames in {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
