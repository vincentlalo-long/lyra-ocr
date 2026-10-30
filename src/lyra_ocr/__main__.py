"""Module entry point for lyra-ocr."""

import sys

from .cli import main as cli_main
from .web import main as web_main


def main() -> None:
    # If any CLI arguments are provided, route to CLI
    if len(sys.argv) > 1:
        sys.exit(cli_main())
    # Default without arguments: launch web UI
    web_main()


if __name__ == "__main__":
    main()
