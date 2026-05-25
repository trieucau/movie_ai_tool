"""
Movie AI Tool - Main Entry Point
Automatically converts YouTube movie videos into viral TikTok review clips.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.gui import MovieAIApp


def main():
    """Launch the Movie AI Tool GUI."""
    app = MovieAIApp()
    app.mainloop()


if __name__ == "__main__":
    main()
