"""Command-line native build entry point."""

from ._build import main
from ._build import matching_library as matching_library
from ._build import support_library as support_library

if __name__ == "__main__":
    main()
