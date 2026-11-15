"""Shared command-line startup.

Known problems (a missing folder, a bad config value) print as one Error
line. Unexpected bugs still show a normal Python traceback.
"""

from __future__ import annotations

from collections.abc import Callable


def run_main(main: Callable[[], None]) -> None:
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
