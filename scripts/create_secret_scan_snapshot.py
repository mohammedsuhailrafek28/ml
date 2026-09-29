"""Copy candidate text/source files into a bounded Gitleaks scan snapshot.

The file list is Git-aware: it includes tracked files present in the worktree
and unignored untracked files, while excluding only generated/vendor/cache,
raw-data, and binary artifact locations that are not source text.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".next",
    ".nyc_output",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "coverage",
    "datasets",
    "htmlcov",
    "node_modules",
    "playwright-report",
    "test-results",
    "venv",
}
EXCLUDED_FILE_NAMES = {
    ".coverage",
    "coverage.xml",
    "coverage-final.json",
}
EXCLUDED_SUFFIXES = {
    ".7z",
    ".arrow",
    ".db",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".joblib",
    ".npz",
    ".npy",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".pyc",
    ".pyo",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tgz",
    ".webp",
    ".zip",
}
DEFAULT_ALLOWLISTED_LOCKFILES = {
    "deno.lock",
    "npm-shrinkwrap.json",
    "package-lock.json",
    "pipfile.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "yarn.lock",
}


def git_candidate_paths(repository: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [os.fsdecode(item) for item in result.stdout.split(b"\0") if item]


def exclusion_reason(relative: Path) -> str | None:
    parts = {part.lower() for part in relative.parts}
    if parts & EXCLUDED_DIRECTORY_NAMES:
        return "generated_or_vendor_directory"
    if relative.name.lower() in EXCLUDED_FILE_NAMES:
        return "coverage_output"
    if relative.suffix.lower() in EXCLUDED_SUFFIXES:
        return "binary_or_compiled_artifact"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == repository or repository in output.parents:
        parser.error("--output must be outside the repository")
    if output.exists():
        parser.error("--output must not already exist")
    output.mkdir(parents=True)

    candidates = git_candidate_paths(repository)
    excluded: Counter[str] = Counter()
    included = 0
    missing = 0
    renamed_lockfiles: list[str] = []

    for candidate in candidates:
        relative = Path(candidate)
        reason = exclusion_reason(relative)
        if reason:
            excluded[reason] += 1
            continue

        source = repository / relative
        try:
            if not source.is_file() and not source.is_symlink():
                missing += 1
                continue
            if source.is_symlink():
                contents = os.fsencode(os.readlink(source))
            else:
                contents = source.read_bytes()
        except OSError as error:
            print(f"Unable to read candidate file: {relative.as_posix()} ({error})", file=sys.stderr)
            return 2

        if b"\0" in contents:
            excluded["binary_content"] += 1
            continue

        destination_relative = relative
        if relative.name.lower() in DEFAULT_ALLOWLISTED_LOCKFILES:
            # Gitleaks' default config allowlists common lockfile basenames.
            # A temporary suffix keeps their exact contents in-scope without
            # changing detector rules or adding a finding allowlist.
            destination_relative = Path(f"{relative.as_posix()}.gitleaks-scan-input")
            renamed_lockfiles.append(relative.as_posix())

        destination = output / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
        included += 1

    if missing:
        print(f"Candidate file list had {missing} missing worktree paths.", file=sys.stderr)
        return 2
    if not included:
        print("No source files were selected for the Gitleaks scan.", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "candidate_paths": len(candidates),
                "included_text_files": included,
                "excluded": dict(sorted(excluded.items())),
                "default_allowlisted_lockfile_aliases": renamed_lockfiles,
                "snapshot": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
