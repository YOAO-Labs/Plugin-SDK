"""Explicit environment migration at dedicated backend boundaries."""

from __future__ import annotations

from collections.abc import Mapping


def migrated_value(
    environ: Mapping[str, str], name: str, retired_name: str
) -> str | None:
    """Return ``name`` while refusing an unmigrated ``retired_name``.

    The old key must leave the environment at the controlled deployment switch;
    silently preferring one of two sources would make the effective credential or
    GPU identity ambiguous. Errors name both keys but never include their values.
    """

    if retired_name in environ:
        if name in environ:
            raise RuntimeError(
                f"{name} and retired {retired_name} are both set; "
                f"configure only {name}"
            )
        raise RuntimeError(f"{retired_name} is retired; configure {name}")
    return environ.get(name)
