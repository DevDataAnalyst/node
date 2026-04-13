#!/usr/bin/env python3
"""Upscale an input video to 4K with a smooth 30 FPS output.

Designed for old footage (including black-and-white videos), with optional
restoration filters and motion interpolation.

Examples:
  python video_upscale_4k.py old_clip.mp4 old_clip_4k_30fps.mp4

  python video_upscale_4k.py old_clip.mp4 old_clip_4k_30fps.mp4 \
      --method realesrgan --realesrgan-bin /path/to/realesrgan-ncnn-vulkan
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

FOUR_K_WIDTH = 3840
FOUR_K_HEIGHT = 2160
DEFAULT_FPS = 30


class CommandError(RuntimeError):
    """Raised when a subprocess command fails."""


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command and raise CommandError on failure."""
    process = subprocess.run(cmd, text=True, capture_output=True)
    if process.returncode != 0:
        raise CommandError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n\n"
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )
    return process


def require_binary(name: str, override: str | None = None) -> str:
    """Return absolute binary path or raise a helpful error."""
    candidate = override or name
    path = shutil.which(candidate)
    if not path:
        raise FileNotFoundError(
            f"Required binary '{candidate}' was not found in PATH."
        )
    return path


def probe_video_fps(ffprobe_bin: str, input_video: Path) -> float | None:
    """Probe source FPS using ffprobe; return None if unavailable."""
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "json",
        str(input_video),
    ]
    process = subprocess.run(cmd, text=True, capture_output=True)
    if process.returncode != 0:
        return None

    try:
        payload = json.loads(process.stdout)
        rate = payload["streams"][0]["r_frame_rate"]
        fps = float(Fraction(rate))
        return fps if fps > 0 else None
    except (KeyError, IndexError, ValueError, ZeroDivisionError, json.JSONDecodeError):
        return None


def build_filter_chain(
    keep_aspect: bool,
    smooth_motion: bool,
    target_fps: int,
    restore_old_footage: bool,
) -> str:
    """Build ffmpeg filter chain for restoration + smooth 4K output."""
    filters: list[str] = []

    if restore_old_footage:
        # Mild denoise + contrast/saturation + edge sharpening.
        # Safe for grayscale footage as saturation gain has no adverse effect.
        filters.extend(
            [
                "hqdn3d=1.5:1.5:6:6",
                "eq=contrast=1.08:brightness=0.01:saturation=1.05",
                "unsharp=5:5:0.7:5:5:0.0",
            ]
        )

    if smooth_motion:
        filters.append(
            "minterpolate="
            f"fps={target_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
        )
    else:
        filters.append(f"fps={target_fps}")

    if keep_aspect:
        filters.append(
            f"scale={FOUR_K_WIDTH}:{FOUR_K_HEIGHT}:"
            "force_original_aspect_ratio=decrease:flags=lanczos"
        )
        filters.append(
            f"pad={FOUR_K_WIDTH}:{FOUR_K_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"
        )
    else:
        filters.append(f"scale={FOUR_K_WIDTH}:{FOUR_K_HEIGHT}:flags=lanczos")

    filters.append("format=yuv420p")
    return ",".join(filters)


def ffmpeg_upscale(
    ffmpeg_bin: str,
    ffprobe_bin: str,
    input_video: Path,
    output_video: Path,
    crf: int,
    preset: str,
    keep_aspect: bool,
    target_fps: int,
    smooth_motion: bool,
    restore_old_footage: bool,
) -> None:
    """Upscale to 4K and output a smooth 30 FPS video."""
    source_fps = probe_video_fps(ffprobe_bin, input_video)
    apply_smoothing = smooth_motion and (source_fps is None or source_fps < target_fps)

    vf = build_filter_chain(
        keep_aspect=keep_aspect,
        smooth_motion=apply_smoothing,
        target_fps=target_fps,
        restore_old_footage=restore_old_footage,
    )

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_video),
        "-vf",
        vf,
        "-r",
        str(target_fps),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_video),
    ]
    run_command(cmd)


def realesrgan_upscale(
    ffmpeg_bin: str,
    realesrgan_bin: str,
    input_video: Path,
    output_video: Path,
    model: str,
    scale: int,
    target_fps: int,
    crf: int,
    preset: str,
    restore_old_footage: bool,
) -> None:
    """Upscale video via frame extraction + Real-ESRGAN + remux at 30 FPS."""
    with tempfile.TemporaryDirectory(prefix="upscale_4k_") as tmp:
        tmp_path = Path(tmp)
        frames_in = tmp_path / "frames_in"
        frames_out = tmp_path / "frames_out"
        audio_file = tmp_path / "audio.m4a"

        frames_in.mkdir()
        frames_out.mkdir()

        run_command(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(input_video),
                str(frames_in / "frame_%08d.png"),
            ]
        )

        run_command(
            [
                realesrgan_bin,
                "-i",
                str(frames_in),
                "-o",
                str(frames_out),
                "-n",
                model,
                "-s",
                str(scale),
                "-f",
                "png",
            ]
        )

        audio_extract = subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(input_video),
                "-vn",
                "-c:a",
                "aac",
                str(audio_file),
            ],
            text=True,
            capture_output=True,
        )

        filters = []
        if restore_old_footage:
            filters.extend(
                [
                    "hqdn3d=1.2:1.2:5:5",
                    "eq=contrast=1.06:brightness=0.01:saturation=1.04",
                    "unsharp=5:5:0.5:5:5:0.0",
                ]
            )
        filters.extend(
            [
                f"fps={target_fps}",
                f"scale={FOUR_K_WIDTH}:{FOUR_K_HEIGHT}:flags=lanczos",
                "format=yuv420p",
            ]
        )

        video_only = tmp_path / "video_only.mp4"
        run_command(
            [
                ffmpeg_bin,
                "-y",
                "-framerate",
                str(target_fps),
                "-i",
                str(frames_out / "frame_%08d.png"),
                "-vf",
                ",".join(filters),
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                str(video_only),
            ]
        )

        if audio_extract.returncode == 0 and audio_file.exists():
            run_command(
                [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    str(video_only),
                    "-i",
                    str(audio_file),
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    str(output_video),
                ]
            )
        else:
            video_only.replace(output_video)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upscale an old video to 4K with smooth 30 FPS output."
    )
    parser.add_argument("input", type=Path, help="Path to input video.")
    parser.add_argument("output", type=Path, help="Path for output 4K video.")
    parser.add_argument(
        "--method",
        choices=["ffmpeg", "realesrgan"],
        default="ffmpeg",
        help="Upscaling method (default: ffmpeg).",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg binary name/path.")
    parser.add_argument("--ffprobe-bin", default="ffprobe", help="ffprobe binary name/path.")
    parser.add_argument(
        "--realesrgan-bin",
        default="realesrgan-ncnn-vulkan",
        help="Real-ESRGAN binary name/path (used when --method realesrgan).",
    )
    parser.add_argument(
        "--realesrgan-model",
        default="realesr-animevideov3",
        help="Real-ESRGAN model name.",
    )
    parser.add_argument(
        "--realesrgan-scale",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Real-ESRGAN scale factor.",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=DEFAULT_FPS,
        help="Target output FPS (default: 30).",
    )
    parser.add_argument(
        "--disable-smooth-motion",
        action="store_true",
        help="Disable motion interpolation and use simple FPS conversion.",
    )
    parser.add_argument(
        "--disable-restore",
        action="store_true",
        help="Disable denoise/contrast/sharpen restoration filters.",
    )
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF quality (lower is better).")
    parser.add_argument(
        "--preset",
        default="slow",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        help="x264 encoding preset.",
    )
    parser.add_argument(
        "--stretch",
        action="store_true",
        help="Stretch to exactly 3840x2160 instead of preserving aspect ratio.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    ffmpeg_bin = require_binary("ffmpeg", args.ffmpeg_bin)
    ffprobe_bin = require_binary("ffprobe", args.ffprobe_bin)

    try:
        if args.method == "ffmpeg":
            ffmpeg_upscale(
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
                input_video=args.input,
                output_video=args.output,
                crf=args.crf,
                preset=args.preset,
                keep_aspect=not args.stretch,
                target_fps=args.target_fps,
                smooth_motion=not args.disable_smooth_motion,
                restore_old_footage=not args.disable_restore,
            )
        else:
            realesrgan_bin = require_binary(
                "realesrgan-ncnn-vulkan", args.realesrgan_bin
            )
            realesrgan_upscale(
                ffmpeg_bin=ffmpeg_bin,
                realesrgan_bin=realesrgan_bin,
                input_video=args.input,
                output_video=args.output,
                model=args.realesrgan_model,
                scale=args.realesrgan_scale,
                target_fps=args.target_fps,
                crf=args.crf,
                preset=args.preset,
                restore_old_footage=not args.disable_restore,
            )
    except (CommandError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1

    print(
        f"Done. 4K {args.target_fps} FPS video written to: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
