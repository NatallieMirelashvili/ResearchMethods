# rgb_viewer.py
from pathlib import Path
from typing import Union

import numpy as np
import rasterio
import matplotlib.pyplot as plt


class RGBViewer:
    """
    Minimal RGB viewer for a BigEarthNet-S2 patch folder.
    Expects files ending with: *_B04.tif, *_B03.tif, *_B02.tif
    """

    def __init__(self, patch_path: Union[str, Path]):
        self.patch_dir = Path(patch_path)

    def _read(self, band: str) -> np.ndarray:
        tif = next(self.patch_dir.glob(f"*_{band}.tif"))
        with rasterio.open(tif) as src:
            return src.read(1).astype(np.float32)

    def _stretch(self, rgb: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
        lo = np.percentile(rgb, p_low, axis=(0, 1), keepdims=True)
        hi = np.percentile(rgb, p_high, axis=(0, 1), keepdims=True)
        rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
        return (rgb * 255).astype(np.uint8)

    def rgb_uint8(self) -> np.ndarray:
        r = self._read("B04")  # Red
        g = self._read("B03")  # Green
        b = self._read("B02")  # Blue
        rgb = np.dstack([r, g, b])  # (H, W, 3)
        return self._stretch(rgb)

    def save(self, out_path: Union[str, Path]) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rgb8 = self.rgb_uint8()

        plt.figure()
        plt.imshow(rgb8)
        plt.title(self.patch_dir.name)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved: {out_path.resolve()}")
        return out_path


if __name__ == "__main__":
    PATH_TO_PATCH = (
        "data/BigEarthNet-S2-toy/"
        "S2A_MSIL2A_20170701T093031_N9999_R136_T35VPK/"
        "S2A_MSIL2A_20170701T093031_N9999_R136_T35VPK_01_04"
    )

    # Change this output path to whatever you want
    OUT_PATH = "/home/avivyuv/bigearthnet_v2/ResearchMethods/main_logs/debug_outputs/rgb_patch.png"

    RGBViewer(PATH_TO_PATCH).save(OUT_PATH)
