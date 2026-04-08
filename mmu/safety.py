"""Output-path safety guardrails for the MMU v2 HATS pipeline.

The shared Flatiron mount at ``/mnt/ceph/users/polymathic/`` contains datasets
that took months of compute to produce, AND the read-only raw-data mirror
under ``external_data/`` that every build script reads from. Writing to the
wrong path on that mount could destroy months of work or corrupt the raw
inputs every future build depends on.

This module centralizes the validation logic so every code path that accepts
a user-configurable output directory (Snakefile, CLI scripts, tests) can
reuse the same rules. It's the single source of truth for "which output
paths are allowed under /mnt/ceph/users/polymathic/".
"""

from __future__ import annotations

import os


# The Flatiron shared research mount.
POLYMATHIC_ROOT = "/mnt/ceph/users/polymathic"

# The only subdirectories of POLYMATHIC_ROOT that our pipeline is permitted to
# write into. Everything else under POLYMATHIC_ROOT is off-limits — especially
# ``external_data/`` (read-only raw data) and any neighbor project that's not
# part of MMU v2 HATS work.
ALLOWED_HATS_ROOT_PREFIXES: tuple[str, ...] = (
    f"{POLYMATHIC_ROOT}/MultimodalUniverse_v2_hats",
    f"{POLYMATHIC_ROOT}/MultimodalUniverse_v2_hats_cosmos",
)

# Explicit denylist substrings that ABSOLUTELY must not appear in any output
# path. Adds a second layer of defense on top of the allowlist: even if a new
# allowlist entry is added in the future, these will still reject writes that
# touch critical infrastructure.
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "external_data",  # read-only raw data mirror — writing here corrupts every future build
)


class UnsafeOutputPathError(ValueError):
    """Raised when the requested output directory is not on the safe list.

    This is a loud, explicit error type so grep / stack traces make it obvious
    that the safety check (not a random misconfiguration) blocked the run.
    """


def validate_hats_root(hats_root: str) -> None:
    """Raise ``UnsafeOutputPathError`` if ``hats_root`` is not safe to write into.

    Rules:

    1. ``hats_root`` must be a non-empty string.
    2. If ``hats_root`` starts with ``/mnt/ceph/users/polymathic/`` it must
       also match one of :data:`ALLOWED_HATS_ROOT_PREFIXES`.
    3. ``hats_root`` must not contain any substring in :data:`FORBIDDEN_SUBSTRINGS`
       anywhere in the path (belt-and-suspenders check against
       ``external_data/`` etc.).
    4. Paths outside ``/mnt/ceph/users/polymathic/`` are allowed without
       further checks (user home dirs, local scratch, test fixtures, etc.).

    This function is a pure string check with no filesystem side effects, so
    it's safe to call from test code and from Snakefile parse time.
    """
    if not hats_root or not isinstance(hats_root, str):
        raise UnsafeOutputPathError(
            f"hats_root is empty or not a string: {hats_root!r}. "
            "Refusing to run without a known-good output directory."
        )

    normalized = os.path.normpath(hats_root).rstrip("/")

    # Rule 3: forbidden substrings, checked against BOTH the raw and normalized
    # paths so we catch things like ``/foo/../polymathic/external_data`` too.
    for forbidden in FORBIDDEN_SUBSTRINGS:
        if forbidden in normalized or forbidden in hats_root:
            raise UnsafeOutputPathError(
                f"Refusing to use hats_root={hats_root!r}: path contains "
                f"{forbidden!r}, which is on the forbidden substring list. "
                "This check exists specifically to prevent overwriting the "
                "read-only raw data mirror at "
                f"{POLYMATHIC_ROOT}/external_data/."
            )

    # Rule 2: anything under POLYMATHIC_ROOT must match an allowed prefix.
    if normalized == POLYMATHIC_ROOT or normalized.startswith(POLYMATHIC_ROOT + "/"):
        is_allowed = any(
            normalized == prefix or normalized.startswith(prefix + "/")
            for prefix in ALLOWED_HATS_ROOT_PREFIXES
        )
        if not is_allowed:
            allowed = "\n  - ".join(ALLOWED_HATS_ROOT_PREFIXES)
            raise UnsafeOutputPathError(
                f"Refusing to use hats_root={hats_root!r}. "
                f"Writes under {POLYMATHIC_ROOT}/ are only allowed under:\n"
                f"  - {allowed}\n"
                f"If you really want to write somewhere else on this mount, "
                f"update ALLOWED_HATS_ROOT_PREFIXES in mmu/safety.py explicitly."
            )
    # Non-polymathic paths are fine — the user asked for that location knowing
    # it's on their local disk or a personal scratch area.
