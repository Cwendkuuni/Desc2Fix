#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFECTS4J_REPO = "https://github.com/rjust/defects4j.git"


@dataclass
class Defects4JStatus:
    root: Path
    exists: bool
    has_framework_projects: bool
    has_checkout_script: bool
    has_defects4j_bin: bool
    git_available: bool
    perl_available: bool
    java_available: bool

    @property
    def ready_for_corpus_build(self) -> bool:
        return self.exists and self.has_framework_projects

    def as_dict(self) -> dict:
        return {
            "root": str(self.root),
            "exists": self.exists,
            "has_framework_projects": self.has_framework_projects,
            "has_checkout_script": self.has_checkout_script,
            "has_defects4j_bin": self.has_defects4j_bin,
            "git_available": self.git_available,
            "perl_available": self.perl_available,
            "java_available": self.java_available,
            "ready_for_corpus_build": self.ready_for_corpus_build,
        }


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_cmd(cmd: list[str], cwd: Optional[Path] = None) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def inspect_defects4j(root: Path) -> Defects4JStatus:
    root = root.expanduser().resolve()

    return Defects4JStatus(
        root=root,
        exists=root.exists(),
        has_framework_projects=(root / "framework" / "projects").is_dir(),
        has_checkout_script=(root / "framework" / "bin" / "defects4j").exists(),
        has_defects4j_bin=(root / "framework" / "bin" / "defects4j").exists(),
        git_available=command_exists("git"),
        perl_available=command_exists("perl"),
        java_available=command_exists("java"),
    )


def clone_defects4j(root: Path, *, force: bool = False) -> None:
    root = root.expanduser().resolve()

    if root.exists():
        if not force:
            print(f"[SKIP] Defects4J already exists: {root}")
            return
        print(f"[WARN] Removing existing Defects4J directory: {root}")
        shutil.rmtree(root)

    root.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["git", "clone", DEFECTS4J_REPO, str(root)])


def initialize_defects4j(root: Path, *, skip_init: bool = False) -> None:
    """
    Initializes Defects4J if possible.

    Defects4J usually requires:
      cpanm --installdeps .
      ./init.sh

    We do not force this automatically unless the user asks through setup script.
    """
    root = root.expanduser().resolve()

    if skip_init:
        print("[SKIP] Defects4J initialization skipped.")
        return

    init_script = root / "init.sh"
    if not init_script.exists():
        print(f"[WARN] Missing init.sh in {root}. Cannot initialize automatically.")
        return

    run_cmd(["bash", "init.sh"], cwd=root)


def setup_defects4j(
    *,
    root: Path,
    clone_if_missing: bool = True,
    initialize: bool = False,
    force_clone: bool = False,
) -> Defects4JStatus:
    root = root.expanduser().resolve()

    status = inspect_defects4j(root)
    if not status.exists and clone_if_missing:
        clone_defects4j(root, force=force_clone)

    if initialize:
        initialize_defects4j(root)

    status = inspect_defects4j(root)

    print("[Desc2Fix] Defects4J status:")
    for k, v in status.as_dict().items():
        print(f"  {k}: {v}")

    if not status.git_available:
        print("[WARN] git not found.")
    if not status.perl_available:
        print("[WARN] perl not found.")
    if not status.java_available:
        print("[WARN] java not found.")

    if not status.ready_for_corpus_build:
        print(
            "[WARN] Defects4J is not fully ready for raw corpus build. "
            "Expected external/defects4j/framework/projects."
        )

    return status
