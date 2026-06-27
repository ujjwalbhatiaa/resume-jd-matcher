#!/usr/bin/env python3
"""
main.py — command-line interface for the Resume ↔ Job-Description Matcher.

Examples
--------
    # Compare two text files, human-readable report:
    python main.py --resume samples/resume_sample.txt --jd samples/jd_sample.txt

    # Machine-readable JSON (pipe into jq, store, etc.):
    python main.py -r samples/resume_sample.txt -j samples/jd_sample.txt --json

    # Read the resume from a file and paste the JD on stdin:
    cat job.txt | python main.py -r my_resume.txt --jd -

No third-party dependencies — Python 3.8+ standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys

from analyzer import analyze, format_report


def _read_source(path: str, label: str) -> str:
    """Read text from a file path, or from stdin when path == '-'."""
    if path == "-":
        data = sys.stdin.read()
        if not data.strip():
            sys.exit(f"error: no {label} text received on stdin")
        return data
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        sys.exit(f"error: {label} file not found: {path}")
    except OSError as exc:
        sys.exit(f"error: could not read {label} file {path}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="resume-jd-matcher",
        description="Compare a resume against a job description: match score, "
                    "skill-gap analysis, and tailored suggestions.",
    )
    p.add_argument("-r", "--resume", required=True,
                   help="path to the resume text file ('-' for stdin)")
    p.add_argument("-j", "--jd", required=True,
                   help="path to the job-description text file ('-' for stdin)")
    p.add_argument("--json", action="store_true",
                   help="emit the report as JSON instead of formatted text")
    p.add_argument("--max-suggestions", type=int, default=6,
                   help="cap the number of suggestions (default: 6)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.resume == "-" and args.jd == "-":
        sys.exit("error: only one of --resume / --jd can read from stdin")

    resume_text = _read_source(args.resume, "resume")
    jd_text = _read_source(args.jd, "job-description")

    report = analyze(resume_text, jd_text, max_suggestions=args.max_suggestions)

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
