"""Environment for isolated provider child processes.

Route control and plugin-registration variables belong to the platform and
operator boundary. A provider started for local checks or development must not
inherit them and become its own routing controller.
"""

from __future__ import annotations

from collections.abc import Mapping

_EXCLUDED_PREFIXES = ("AAP_ROUTE_", "AAP_PLUGIN_")
_EXCLUDED_NAMES = frozenset({"ROUTE_CONTROL_TOKEN", "ROUTE_CONTROL_TOKEN_FILE"})


def isolated_provider_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Return ``environ`` without platform authority or plugin registration keys."""

    return {
        key: value
        for key, value in environ.items()
        if key not in _EXCLUDED_NAMES and not key.startswith(_EXCLUDED_PREFIXES)
    }
