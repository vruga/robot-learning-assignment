"""Assemble the demo video from the evaluator's own clips (advaitic-eval --video-episodes N).

    python submission/scripts/make_video.py --clips results/video_cr_all --episodes 1 2 4 0 \
        --ffmpeg <path to ffmpeg> --font /System/Library/Fonts/Supplemental/Arial.ttf --out cr_all_demo.mp4

Each evaluator frame is the policy's three camera views (head, right wrist, left wrist; 84x84 each) side by side,
recorded every 5th simulation step at 10 fps (real time). Clips are enlarged 4x (nearest neighbour), played at 2x speed,
and captioned from report.json (seed, starting slot, outcome, duration): nothing is typed by hand.
"""

import argparse
import json
import subprocess
from pathlib import Path

W, H_VIDEO, BAR = 1008, 336, 100  # 3 x 84 x 4 wide, 84 x 4 high, caption bar on top
SPEED = 2


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def text(font, line, x, y, size, color="white"):
    line = line.replace(":", " -").replace("'", "")  # ':' separates ffmpeg filter options
    return f"drawtext=fontfile={font}:text='{line}':x={x}:y={y}:fontsize={size}:fontcolor={color}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--font", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", nargs="*", default=[])
    args = ap.parse_args()
    clips = Path(args.clips)
    rep = json.loads((clips / "report.json").read_text())
    eps = {e["episode"]: e for e in rep["episodes"]}
    work = Path(args.out).with_suffix("").parent / "_video_parts"
    work.mkdir(parents=True, exist_ok=True)
    parts = []

    title = args.title or [
        "cr_all - Unitree G1 plate transfer",
        "imitation-learning policy: camera images + proprioception only",
        f"evaluator run from seed {rep['seed_start']}: {rep['successes']} of {rep['completed_episodes']} successes",
        "views, left to right - head, right wrist, left wrist (84x84 each, as the policy sees them)",
        "clips at 2x speed: three successes from different start slots, then the one failure",
    ]
    vf = ",".join(
        text(args.font, t, 30, 60 + 60 * i, 30 if i == 0 else 22) for i, t in enumerate(title)
    )
    p = work / "title.mp4"
    run(
        [
            args.ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={W}x{H_VIDEO + BAR}:d=6:r=20",
            "-vf",
            vf,
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            str(p),
        ]
    )
    parts.append(p)

    for n in args.episodes:
        e = eps[n]
        secs = e["steps"] / 50
        ok = e["success"]
        head = f"Episode {n} - seed {e['seed']} - start slot {e['source_slot'] + 1} - " + (
            f"SUCCESS in {secs:.1f} s" if ok else f"FAILURE - timeout at {secs:.0f} s"
        )
        vf = (
            f"scale=iw*4:ih*4:flags=neighbor,setpts=PTS/{SPEED},pad={W}:{H_VIDEO + BAR}:0:{BAR}:black,"
            + text(args.font, head, 20, 18, 28, "lightgreen" if ok else "tomato")
            + ","
            + text(args.font, "head", 20, 64, 20, "gray")
            + ","
            + text(args.font, "right wrist", 356, 64, 20, "gray")
            + ","
            + text(args.font, "left wrist", 692, 64, 20, "gray")
        )
        p = work / f"ep{n}.mp4"
        run(
            [
                args.ffmpeg,
                "-y",
                "-i",
                str(clips / f"episode_{n:03}.mp4"),
                "-vf",
                vf,
                "-r",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                str(p),
            ]
        )
        parts.append(p)

    lst = work / "list.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    run([args.ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", args.out])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
