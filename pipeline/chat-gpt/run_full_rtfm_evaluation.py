#!/usr/bin/env python3
"""
End-to-end RTFM evaluation pipeline.

Runs all three stages sequentially:
  1. run_rtfm_pipeline.py     → RTFM inference, gating, segment detection, frame extraction
  2. generate_explanations_rtfm.py → GPT-4o VLM explanation generation
  3. judge_explanations_rtfm.py    → GPT-4o-as-judge scoring vs human ground truth

Usage:
    export OPENAI_API_KEY="sk-..."
    python run_full_rtfm_evaluation.py
    python run_full_rtfm_evaluation.py --video 01_0015
    python run_full_rtfm_evaluation.py --skip-pipeline   # only re-run explanation + judge
    python run_full_rtfm_evaluation.py --skip-explain     # only re-run judge
"""

import subprocess
import sys
import argparse
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent


def run_step(name, cmd):
    print(f"\n{'='*65}")
    print(f"  STAGE: {name}")
    print(f"  CMD  : {' '.join(cmd)}")
    print(f"{'='*65}\n")

    result = subprocess.run(cmd, cwd=str(PIPELINE_DIR))
    if result.returncode != 0:
        print(f"\nERROR: {name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n  {name} completed successfully.\n")


def main():
    parser = argparse.ArgumentParser(description="Full RTFM evaluation pipeline")
    parser.add_argument("--video", type=str, default=None,
                        help="Single video ID (omit for all test videos)")
    parser.add_argument("--gate-threshold", type=float, default=0.2)
    parser.add_argument("--segment-threshold", type=float, default=0.3)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--min-gap", type=int, default=2)
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip RTFM pipeline (reuse existing outputs)")
    parser.add_argument("--skip-explain", action="store_true",
                        help="Skip explanation generation (reuse existing)")
    args = parser.parse_args()

    py = sys.executable

    # Stage 1: RTFM Pipeline
    if not args.skip_pipeline:
        cmd = [
            py, "run_rtfm_pipeline.py",
            "--gate-threshold", str(args.gate_threshold),
            "--segment-threshold", str(args.segment_threshold),
            "--frame-budget", str(args.frame_budget),
            "--min-gap", str(args.min_gap),
        ]
        if args.video:
            cmd += ["--video", args.video]
        run_step("RTFM Pipeline (inference + gating + frame extraction)", cmd)

    # Stage 2: Explanation Generation
    if not args.skip_explain:
        cmd = [py, "generate_explanations_rtfm.py"]
        if args.video:
            cmd += ["--video", args.video]
        else:
            cmd += ["--batch"]
        run_step("VLM Explanation Generation (GPT-4o)", cmd)

    # Stage 3: Judge
    cmd = [py, "judge_explanations_rtfm.py"]
    if args.video:
        cmd += ["--video", args.video]
    else:
        cmd += ["--batch"]
    run_step("Judge Explanations (GPT-4o-as-judge)", cmd)

    print(f"\n{'='*65}")
    print(f"  ALL STAGES COMPLETE")
    print(f"  Results in: pipeline/rtfm_outputs/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
