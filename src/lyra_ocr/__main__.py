"""Module entry point for lyra-ocr."""

import sys

from .cli import main as cli_main


def main() -> None:
    # Without arguments, launch the web review UI
    if len(sys.argv) > 1:
        sys.exit(cli_main())
    sys.exit(cli_main(["review-ui"]))


if __name__ == "__main__":
    main()
