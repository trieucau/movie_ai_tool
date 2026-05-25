"""
Movie AI Tool - Main Entry Point
Automatically converts YouTube movie videos into viral TikTok review clips.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ── Inject NVIDIA CUDA DLL dirs into PATH BEFORE any import ──────────────────
# ctranslate2 uses LoadLibrary() internally to find cublas64_12.dll.
# Setting PATH (not just os.add_dll_directory) ensures Windows DLL loader
# finds the libraries packaged with the pip-installed nvidia-* packages.
if sys.platform == "win32":
    _search_roots = list(sys.path)
    try:
        import site as _site
        _search_roots += _site.getsitepackages()
        try:
            _search_roots.append(_site.getusersitepackages())
        except Exception:
            pass
    except Exception:
        pass

    _nvidia_subdirs = [
        Path("nvidia") / "cublas"       / "bin",
        Path("nvidia") / "cuda_runtime" / "bin",
        Path("nvidia") / "cuda_cupti"   / "bin",
        Path("nvidia") / "cuda_nvrtc"   / "bin",
        Path("nvidia") / "cudnn"        / "bin",
        Path("nvidia") / "cufft"        / "bin",
        Path("nvidia") / "curand"       / "bin",
        Path("nvidia") / "cusolver"     / "bin",
        Path("nvidia") / "cusparse"     / "bin",
        Path("nvidia") / "nvjpeg"       / "bin",
    ]

    _extra_paths = []
    for _root in set(_search_roots):
        for _sub in _nvidia_subdirs:
            _candidate = Path(_root) / _sub
            if _candidate.exists() and str(_candidate) not in _extra_paths:
                _extra_paths.append(str(_candidate))
                # Both approaches: PATH env var + DLL dir registry
                try:
                    os.add_dll_directory(str(_candidate))
                except Exception:
                    pass

    if _extra_paths:
        os.environ["PATH"] = os.pathsep.join(_extra_paths) + os.pathsep + os.environ.get("PATH", "")
# ─────────────────────────────────────────────────────────────────────────────

from app.gui import MovieAIApp


def main():
    """Launch the Movie AI Tool GUI."""
    app = MovieAIApp()
    app.mainloop()


if __name__ == "__main__":
    main()
