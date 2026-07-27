"""Allow ``python -m ebl`` to use the same entry point as ``ebl``."""

from ebl.cli import main


raise SystemExit(main())
