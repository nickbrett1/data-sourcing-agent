"""Default module entry point (python -m data_sourcing_agent).

Override the container command via the genproj docker-container "command"
configuration option (or "entrypoint") when your application needs a custom
entry point.
"""

import sys


def main() -> int:
    print(f"{__package__} is installed and importable.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
