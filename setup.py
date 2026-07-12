"""
Hunyuan3D 2 Mini — extension setup script.

Creates an isolated venv and installs all required dependencies.
Called by Modly at extension install time with:

    python setup.py <json_args>

where json_args contains:
    python_exe  — path to Modly's embedded Python (used to create the venv)
    ext_dir     — absolute path to this extension directory
    gpu_sm      — GPU compute capability as integer (e.g. 61 for Pascal, 86 for Ampere; 0 on macOS)
    accelerator — "mps" | "cuda" | "cpu"  (passed by Electron since Modly 1.x)
    platform    — Electron's process.platform string ("win32", "darwin", "linux")

Example (manual test):
    python setup.py '{"python_exe":"C:/…/python.exe","ext_dir":"C:/…/hunyuan3d-2-mini","gpu_sm":86}'
"""
import json
import platform
import subprocess
import sys
from pathlib import Path


def pip(venv: Path, *args: str) -> None:
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def setup(
    python_exe:    str,
    ext_dir:       Path,
    gpu_sm:        int,
    accelerator:   str = "",
    platform_name: str = "",
) -> None:
    venv   = ext_dir / "venv"
    is_win = platform.system() == "Windows"

    # Resolve accelerator when not supplied by Electron (manual invocation)
    if not accelerator:
        sys_name = platform.system()
        if sys_name == "Darwin":
            accelerator = "mps" if platform.machine() == "arm64" else "cpu"
        elif gpu_sm > 0:
            accelerator = "cuda"
        else:
            accelerator = "cpu"

    print(f"[setup] accelerator={accelerator}  gpu_sm={gpu_sm}")
    print(f"[setup] Creating venv at {venv} …")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ------------------------------------------------------------------ #
    # PyTorch — version depends on accelerator
    # ------------------------------------------------------------------ #
    if accelerator == "mps":
        # Apple Silicon — standard PyPI wheel includes the Metal (MPS) backend
        print("[setup] Apple Silicon (MPS) -> PyTorch from standard PyPI")
        pip(venv, "install", "torch", "torchvision")
    elif accelerator == "cuda":
        if gpu_sm >= 100:
            # Blackwell (RTX 50xx) — PyTorch 2.7+ + CUDA 12.8
            torch_index = "https://download.pytorch.org/whl/cu128"
            torch_pkgs  = ["torch>=2.7.0", "torchvision>=0.22.0"]
            print(f"[setup] GPU SM {gpu_sm} (Blackwell) -> PyTorch 2.7 + CUDA 12.8")
        elif gpu_sm >= 70:
            # Volta to Ada (RTX 20/30/40) — PyTorch 2.6 + CUDA 12.4
            torch_index = "https://download.pytorch.org/whl/cu124"
            torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
            print(f"[setup] GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.4")
        else:
            # Pascal (SM 6.x) — PyTorch 2.5 + CUDA 11.8 (last version with SM 6.1)
            torch_index = "https://download.pytorch.org/whl/cu118"
            torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
            print(f"[setup] GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8")
        print("[setup] Installing PyTorch …")
        pip(venv, "install", *torch_pkgs, "--index-url", torch_index)
    else:
        # CPU-only (Intel Mac or no GPU on any platform)
        print("[setup] CPU-only -> PyTorch from standard PyPI")
        pip(venv, "install", "torch", "torchvision")

    # ------------------------------------------------------------------ #
    # Core dependencies
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies …")
    pip(venv, "install",
        "Pillow",
        "numpy",
        "trimesh",
        "pymeshlab",
        "opencv-python-headless",
        "huggingface_hub",
        "hf_xet",  # fast Xet-backed HF downloads; without it HF falls back to slow plain HTTP
        "diffusers>=0.31.0",
        "transformers>=4.46.0,<4.54",
        "accelerate",
        "einops",
        "scipy",
        "scikit-image",
    )

    # ------------------------------------------------------------------ #
    # rembg (background removal) + ONNX runtime
    # ------------------------------------------------------------------ #
    print("[setup] Installing rembg …")
    pip(venv, "install", "rembg")
    if accelerator == "cuda" and gpu_sm >= 70:
        # Volta+ CUDA: prefer GPU-accelerated ONNX runtime, fall back to CPU
        try:
            pip(venv, "install", "onnxruntime-gpu")
        except subprocess.CalledProcessError:
            pip(venv, "install", "onnxruntime")
    else:
        # macOS (MPS/CPU) and Pascal CUDA: CPU ONNX runtime
        # - onnxruntime-gpu is not available for macOS
        # - Pascal has cuDNN FE compatibility issues with onnxruntime-gpu
        pip(venv, "install", "onnxruntime")

    # ------------------------------------------------------------------ #
    # Texture generation dependencies (optional — heavy)
    # Skipped here; will be installed on first texture request if needed.
    # Requires custom C++ extensions (custom_rasterizer, differentiable_renderer)
    # built via separate wheel distribution.
    # ------------------------------------------------------------------ #

    print("[setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    # Accepts either JSON (from Electron) or positional args (for manual testing)
    # Positional: python setup.py <python_exe> <ext_dir> <gpu_sm>
    # JSON:       python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86}'
    if len(sys.argv) >= 4:
        setup(
            python_exe = sys.argv[1],
            ext_dir    = Path(sys.argv[2]),
            gpu_sm     = int(sys.argv[3]),
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe    = args["python_exe"],
            ext_dir       = Path(args["ext_dir"]),
            gpu_sm        = int(args["gpu_sm"]),
            accelerator   = args.get("accelerator", ""),
            platform_name = args.get("platform", ""),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86}\'')
        sys.exit(1)