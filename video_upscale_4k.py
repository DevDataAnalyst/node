#!/usr/bin/env python3
"""Upscale an input video to 4K (3840x2160).

This script supports two workflows:
1) ffmpeg-only (fast, broadly available): high-quality Lanczos scaling.
2) AI-assisted (optional): Real-ESRGAN frame upscaling, then ffmpeg muxing.

Examples:
  # ffmpeg-only upscale
  python video_upscale_4k.py input.mp4 output_4k.mp4

  # AI-assisted upscale using realesrgan-ncnn-vulkan executable
  python video_upscale_4k.py input.mp4 output_4k.mp4 \
      --method realesrgan --realesrgan-bin /path/to/realesrgan-ncnn-vulkan
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


FOUR_K_WIDTH = 3840
FOUR_K_HEIGHT = 2160


class CommandError(RuntimeError):
    """Raised when a subprocess command fails."""


def run_command(cmd: list[str]) -> None:
    """Run a command and raise CommandError on failure."""
    process = subprocess.run(cmd, text=True, capture_output=True)
    if process.returncode != 0:
        raise CommandError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n\n"
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )


def require_binary(name: str, override: str | None = None) -> str:
    """Return absolute binary path or raise a helpful error."""
    candidate = override or name
    path = shutil.which(candidate)
    if not path:
        raise FileNotFoundError(
            f"Required binary '{candidate}' was not found in PATH."
        )
    return path


def ffmpeg_upscale(
    ffmpeg_bin: str,
    input_video: Path,
    output_video: Path,
    crf: int,
    preset: str,
    keep_aspect: bool,
) -> None:
    """Upscale to 4K with ffmpeg's Lanczos scaler."""
    if keep_aspect:
        vf = (
            f"scale={FOUR_K_WIDTH}:{FOUR_K_HEIGHT}:"
            "force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={FOUR_K_WIDTH}:{FOUR_K_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
        )
    else:
        vf = f"scale={FOUR_K_WIDTH}:{FOUR_K_HEIGHT}:flags=lanczos"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_video),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
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
    fps: str | None,
    crf: int,
    preset: str,
) -> None:
    """Upscale video via frame extraction + Real-ESRGAN + remux."""
    with tempfile.TemporaryDirectory(prefix="upscale_4k_") as tmp:
        tmp_path = Path(tmp)
        frames_in = tmp_path / "frames_in"
        frames_out = tmp_path / "frames_out"
        audio_file = tmp_path / "audio.m4a"

        frames_in.mkdir()
        frames_out.mkdir()

        # 1) Extract frames.
        run_command(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(input_video),
                str(frames_in / "frame_%08d.png"),
            ]
        )

        # 2) AI upscale frames.
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

        # 3) Extract audio (if present).
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

        # 4) Build final video at 4K.
        input_fps = ["-r", fps] if fps else []
        video_only = tmp_path / "video_only.mp4"

        run_command(
            [
                ffmpeg_bin,
                "-y",
                *input_fps,
                "-i",
                str(frames_out / "frame_%08d.png"),
                "-vf",
                f"scale={FOUR_K_WIDTH}:{FOUR_K_HEIGHT}:flags=lanczos",
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

        # 5) Mux audio if available.
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
                    str(output_video),
                ]
            )
        else:
            video_only.replace(output_video)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upscale an old video to 4K (3840x2160)."
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
        "--fps",
        default=None,
        help="Optional FPS override for frame sequence encoding (e.g. 23.976).",
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

    try:
        if args.method == "ffmpeg":
            ffmpeg_upscale(
                ffmpeg_bin=ffmpeg_bin,
                input_video=args.input,
                output_video=args.output,
                crf=args.crf,
                preset=args.preset,
                keep_aspect=not args.stretch,
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
                fps=args.fps,
                crf=args.crf,
                preset=args.preset,
            )
    except (CommandError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1

    print(f"Done. 4K video written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
