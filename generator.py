"""
Reference : https://huggingface.co/tencent/Hunyuan3D-2mini
Turbo variant — hunyuan3d-dit-v2-mini-turbo subfolder + hunyuan3d-vae-v2-mini-turbo.
"""
import io
import os
import random
import sys
import tempfile
import time
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Optional

# Must be set before the first `import torch` in this process so Metal (MPS)
# operators without a kernel transparently fall back to CPU instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from PIL import Image

from services.generators.base import (
    BaseGenerator,
    smooth_progress,
    GenerationCancelled,
    select_device,
    select_dtype,
)

_HF_REPO_ID      = "tencent/Hunyuan3D-2mini"
_SUBFOLDER       = "hunyuan3d-dit-v2-mini-turbo"
_TURBO_VAE       = "hunyuan3d-vae-v2-mini-turbo"
_GITHUB_ZIP      = "https://github.com/Tencent/Hunyuan3D-2/archive/refs/heads/main.zip"
_PAINT_HF_REPO   = "tencent/Hunyuan3D-2"
_PAINT_SUBFOLDER = "hunyuan3d-paint-v2-0-turbo"


class Hunyuan3DMiniTurboGenerator(BaseGenerator):
    MODEL_ID     = "hunyuan3d-mini-turbo"
    DISPLAY_NAME = "Hunyuan3D 2 Mini Turbo"
    VRAM_GB      = 6

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        subfolder = self.download_check if self.download_check else _SUBFOLDER
        model_dir = self.model_dir / subfolder
        return model_dir.exists() and (model_dir / "model.fp16.safetensors").exists()

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_weights()

        self._ensure_hy3dgen()

        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        device = select_device()
        dtype  = select_dtype(device)
        self._device = device

        self._patch_sdp_backend_for_non_cuda(device)

        subfolder = self.download_check if self.download_check else _SUBFOLDER
        print(f"[Hunyuan3DMiniTurboGenerator] Loading pipeline from {self.model_dir} (subfolder={subfolder})…")
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(self.model_dir),
            subfolder=subfolder,
            use_safetensors=True,
            device=device,
            dtype=dtype,
        )

        self._enable_flashvdm(pipeline, device, dtype)

        self._model = pipeline
        print(f"[Hunyuan3DMiniTurboGenerator] Loaded on {device}.")

    @staticmethod
    def _patch_sdp_backend_for_non_cuda(device: str) -> None:
        """Neutralise hy3dgen's CUDA-only `torch.backends.cuda.sdp_kernel(...)`.

        The DiT blocks request `enable_flash=True, enable_math=False` which is
        only satisfiable on CUDA. On MPS/CPU there is no fused kernel, so
        forbidding the math backend makes SDPA raise "No available kernel".
        """
        if device == "cuda":
            return
        import contextlib
        import torch

        if getattr(torch.backends.cuda, "_modly_sdp_patched", False):
            return

        @contextlib.contextmanager
        def _sdpa_kernel(**_kwargs):
            yield

        try:
            torch.backends.cuda._modly_sdp_original = torch.backends.cuda.sdp_kernel
            torch.backends.cuda.sdp_kernel = _sdpa_kernel
            torch.backends.cuda._modly_sdp_patched = True
            print("[Hunyuan3DMiniTurboGenerator] Relaxed sdp_kernel for non-CUDA device.")
        except Exception as exc:
            print(f"[Hunyuan3DMiniTurboGenerator] Could not relax sdp_kernel: {exc}")

    @staticmethod
    def _unpatch_sdp_backend() -> None:
        """Restore `torch.backends.cuda.sdp_kernel` if this generator patched it.

        Scopes the patch to this generator's loaded lifetime so it doesn't
        silently neuter kernel selection for other CUDA models running in the
        same process after this generator unloads.
        """
        import torch

        original = getattr(torch.backends.cuda, "_modly_sdp_original", None)
        if original is not None:
            torch.backends.cuda.sdp_kernel = original
            del torch.backends.cuda._modly_sdp_original
        if getattr(torch.backends.cuda, "_modly_sdp_patched", False):
            torch.backends.cuda._modly_sdp_patched = False

    def _enable_flashvdm(self, pipeline, device: str, dtype) -> None:
        """Enable FlashVDM: the adaptive-KV volume decoder used by the turbo model.

        The turbo checkpoint embeds the standard mini VAE, so the dedicated
        turbo VAE is swapped in before enabling the decoder. Surface extraction
        uses marching cubes ('mc'); the DMC extractor depends on the CUDA-only
        `diso` package and has no Metal (MPS) or CPU backend.
        """
        try:
            from hy3dgen.shapegen.models import ShapeVAE

            vae_dir = self.model_dir / _TURBO_VAE
            if vae_dir.exists():
                pipeline.vae = ShapeVAE.from_pretrained(
                    str(self.model_dir),
                    subfolder=_TURBO_VAE,
                    use_safetensors=True,
                    device=device,
                    dtype=dtype,
                )

            pipeline.vae.enable_flashvdm_decoder(
                enabled=True,
                adaptive_kv_selection=True,
                topk_mode="mean",
                mc_algo="mc",
            )
            print("[Hunyuan3DMiniTurboGenerator] FlashVDM decoder enabled (mc surface extraction).")
        except Exception as exc:
            print(f"[Hunyuan3DMiniTurboGenerator] FlashVDM unavailable ({exc}); using vanilla decoder.")

    def unload(self) -> None:
        super().unload()
        try:
            import torch
            self._unpatch_sdp_backend()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch

        num_steps      = int(params.get("num_inference_steps", 10))
        vert_count     = int(params.get("vertex_count", 0))
        enable_texture = bool(params.get("enable_texture", False))
        octree_res     = int(params.get("octree_resolution", 380))
        guidance_scale = float(params.get("guidance_scale", 5.5))
        seed           = int(params.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        self._report(progress_cb, 5, "Removing background…")
        image = self._preprocess(image_bytes)
        self._check_cancelled(cancel_event)

        shape_end = 70 if enable_texture else 82
        self._report(progress_cb, 12, "Generating 3D shape…")
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 12, shape_end, "Generating 3D shape…", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            with torch.no_grad():
                generator = torch.Generator().manual_seed(seed)
                outputs = self._model(
                    image=image,
                    num_inference_steps=num_steps,
                    octree_resolution=octree_res,
                    guidance_scale=guidance_scale,
                    num_chunks=4000,
                    generator=generator,
                    output_type="trimesh",
                )
            mesh = outputs[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        if enable_texture:
            self._report(progress_cb, 72, "Freeing VRAM for texture model…")
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

            self._check_cancelled(cancel_event)
            mesh = self._run_texture(mesh, image, progress_cb)
            self.load()  # restore shape model so next generation doesn't crash
        else:
            if vert_count > 0 and hasattr(mesh, "vertices") and len(mesh.vertices) > vert_count:
                self._report(progress_cb, 85, "Optimizing mesh…")
                mesh = self._decimate(mesh, vert_count)

        self._report(progress_cb, 96, "Exporting GLB…")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        mesh.export(str(path))

        self._report(progress_cb, 100, "Done")
        return path

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _preprocess(self, image_bytes: bytes) -> Image.Image:
        import rembg
        img = Image.open(io.BytesIO(image_bytes))
        try:
            return rembg.remove(img).convert("RGBA")
        except Exception:
            session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
            return rembg.remove(img, session=session).convert("RGBA")

    def _run_texture(self, mesh, image: "Image.Image", progress_cb=None):
        import torch

        if getattr(self, "_device", None) != "cuda":
            raise RuntimeError(
                "Texture generation requires an NVIDIA GPU: the custom rasterizer / "
                "differentiable renderer have no Metal (MPS) or CPU implementation. "
                "Disable the texture option on macOS."
            )

        self._check_texgen_extensions()

        self._report(progress_cb, 73, "Preparing texture model…")
        self._ensure_paint_weights()

        self._report(progress_cb, 78, "Loading texture model…")
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        paint_dir = self.model_dir / "_paint_weights"
        paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            str(paint_dir), subfolder=_PAINT_SUBFOLDER
        )

        from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
        paint_pipeline.config.render_size  = 1024
        paint_pipeline.config.texture_size = 1024
        paint_pipeline.render = MeshRender(default_resolution=1024, texture_size=1024)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            image.save(tmp.name)
            tmp.close()

            self._report(progress_cb, 83, "Generating textures…")
            with torch.no_grad():
                result = paint_pipeline(mesh, image=tmp.name)
        finally:
            os.unlink(tmp.name)
            del paint_pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return result[0] if isinstance(result, (list, tuple)) else result

    def _check_texgen_extensions(self) -> None:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline  # noqa: F401
        except (ImportError, OSError) as exc:
            base = self.model_dir / "_hy3dgen" / "hy3dgen" / "texgen"
            raise RuntimeError(
                "C++ extensions for texture generation are not compiled.\n"
                "Build them with:\n\n"
                f"  cd \"{base / 'custom_rasterizer'}\"\n"
                f"  python setup.py install\n\n"
                f"  cd \"{base / 'differentiable_renderer'}\"\n"
                f"  python setup.py install\n\n"
                f"Original error: {exc}"
            ) from exc

    def _ensure_paint_weights(self) -> None:
        paint_dir = self.model_dir / "_paint_weights"
        if (paint_dir / _PAINT_SUBFOLDER).exists() and (paint_dir / "hunyuan3d-delight-v2-0").exists():
            return

        from huggingface_hub import snapshot_download
        print(f"[Hunyuan3DMiniTurboGenerator] Downloading paint model ({_PAINT_HF_REPO})…")
        snapshot_download(
            repo_id=_PAINT_HF_REPO,
            local_dir=str(paint_dir),
            ignore_patterns=[
                "hunyuan3d-dit-v2-0/**",
                "hunyuan3d-dit-v2-0-fast/**",
                "hunyuan3d-dit-v2-0-turbo/**",
                "hunyuan3d-vae-v2-0/**",
                "hunyuan3d-vae-v2-0-turbo/**",
                "hunyuan3d-vae-v2-0-withencoder/**",
                "hunyuan3d-paint-v2-0/**",
                "assets/**",
                "*.md", "LICENSE", "NOTICE", ".gitattributes",
            ],
        )
        print("[Hunyuan3DMiniTurboGenerator] Paint model downloaded.")

    def _decimate(self, mesh, target_vertices: int):
        target_faces = max(4, target_vertices * 2)
        try:
            return mesh.simplify_quadric_decimation(target_faces)
        except Exception as exc:
            print(f"[Hunyuan3DMiniTurboGenerator] Decimation skipped: {exc}")
            return mesh

    def _download_weights(self) -> None:
        from huggingface_hub import snapshot_download
        print(f"[Hunyuan3DMiniTurboGenerator] Downloading {_HF_REPO_ID} (turbo variant)…")
        snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=str(self.model_dir),
            ignore_patterns=[
                "hunyuan3d-dit-v2-mini/**",
                "hunyuan3d-dit-v2-mini-fast/**",
                "hunyuan3d-vae-v2-mini-withencoder/**",
                "*.md", "LICENSE", "NOTICE", ".gitattributes",
            ],
        )
        print("[Hunyuan3DMiniTurboGenerator] Download complete.")

    def _ensure_hy3dgen(self) -> None:
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
            return
        except ImportError:
            pass

        src_dir = self.model_dir / "_hy3dgen"
        if not (src_dir / "hy3dgen").exists():
            self._download_hy3dgen(src_dir)

        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"hy3dgen still not importable after extraction to {src_dir}.\n"
                f"Check the folder contents.\n{exc}"
            ) from exc

    def _download_hy3dgen(self, dest: Path) -> None:
        import urllib.request

        dest.mkdir(parents=True, exist_ok=True)
        print("[Hunyuan3DMiniTurboGenerator] Downloading hy3dgen source from GitHub…")
        with urllib.request.urlopen(_GITHUB_ZIP, timeout=180) as resp:
            data = resp.read()
        print("[Hunyuan3DMiniTurboGenerator] Extracting hy3dgen…")

        prefix = "Hunyuan3D-2-main/hy3dgen/"
        strip  = "Hunyuan3D-2-main/"

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if not member.startswith(prefix):
                    continue
                rel    = member[len(strip):]
                target = dest / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))

        print(f"[Hunyuan3DMiniTurboGenerator] hy3dgen extracted to {dest}.")

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id":      "num_inference_steps",
                "label":   "Quality",
                "type":    "select",
                "default": 5,
                "options": [
                    {"value": 5,  "label": "Fast"},
                    {"value": 10, "label": "Balanced"},
                    {"value": 20, "label": "High"},
                ],
                "tooltip": "Number of diffusion steps. More steps = better quality but slower.",
            },
            {
                "id":      "octree_resolution",
                "label":   "Mesh Resolution",
                "type":    "select",
                "default": 380,
                "options": [
                    {"value": 256, "label": "Low"},
                    {"value": 380, "label": "Medium"},
                    {"value": 512, "label": "High"},
                ],
                "tooltip": "Octree resolution for mesh reconstruction. Higher = more detail but slower and more VRAM.",
            },
            {
                "id":      "guidance_scale",
                "label":   "Guidance Scale",
                "type":    "float",
                "default": 5.5,
                "min":     1.0,
                "max":     10.0,
                "step":    0.5,
                "tooltip": "Classifier-free guidance strength. Higher = closer to the input image.",
            },
            {
                "id":      "seed",
                "label":   "Seed",
                "type":    "int",
                "default": -1,
                "min":     -1,
                "max":     4294967295,
                "tooltip": "Seed for reproducibility. Set to -1 for a random seed.",
            },
        ]
