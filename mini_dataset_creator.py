# mini_dataset_creator.py
from __future__ import annotations

import random
import shutil
from pathlib import Path


class MiniBigEarthNetCreator:
    """
    Create a tiny BigEarthNet-S2 subset by sampling:
      - N tiles (top-level folders under src_root)
      - M patches per tile (subfolders that contain .tif files)

    Output structure is identical:
      dst_root/<tile>/<patch>/*.tif
    """

    def __init__(self, src_root: str | Path, dst_root: str | Path, seed: int = 0) -> None:
        self.src_root = Path(src_root)
        self.dst_root = Path(dst_root)
        self.rng = random.Random(seed)

    def _list_tiles(self) -> list[Path]:
        return sorted([p for p in self.src_root.iterdir() if p.is_dir()])

    def _list_patches(self, tile_dir: Path) -> list[Path]:
        # Patch folder = directory that contains at least one .tif
        patches = []
        for d in tile_dir.iterdir():
            if d.is_dir() and any(f.is_file() and f.suffix.lower() == ".tif" for f in d.iterdir()):
                patches.append(d)
        return sorted(patches)

    def create(self, n_tiles: int, m_patches_per_tile: int) -> None:
        tiles = self._list_tiles()
        if n_tiles > len(tiles):
            raise ValueError(f"Requested n_tiles={n_tiles}, but only {len(tiles)} tiles exist in {self.src_root}")

        chosen_tiles = self.rng.sample(tiles, n_tiles)

        self.dst_root.mkdir(parents=True, exist_ok=True)

        for tile_dir in chosen_tiles:
            patches = self._list_patches(tile_dir)
            if m_patches_per_tile > len(patches):
                raise ValueError(
                    f"Tile {tile_dir.name} has only {len(patches)} patches; "
                    f"cannot sample m_patches_per_tile={m_patches_per_tile}"
                )

            chosen_patches = self.rng.sample(patches, m_patches_per_tile)

            for patch_dir in chosen_patches:
                dst_patch_dir = self.dst_root / tile_dir.name / patch_dir.name
                dst_patch_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(patch_dir, dst_patch_dir, dirs_exist_ok=True)


def main() -> None:
    # Example usage (edit paths + N/M)
    src_root = "/dt/shabtaia/DT_Satellite/satellite_image_data/BigEarthNet-S2"
    dst_root = "/home/avivyuv/bigearthnet_v2/ResearchMethods/data/BigEarthNet-S2-toy"

    N = 5  # number of tiles - big geographical areas, 100km x 100km
    M = 50  # number of patches per tile - small areas in each tile, not overlapping, each tile divided to about 8000+ patches
    creator = MiniBigEarthNetCreator(src_root, dst_root, seed=42)
    creator.create(n_tiles=N, m_patches_per_tile=M)


if __name__ == "__main__":
    main()
