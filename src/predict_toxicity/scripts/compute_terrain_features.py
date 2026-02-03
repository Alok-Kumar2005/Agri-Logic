import rasterio
import numpy as np
from rasterio.windows import Window
from pathlib import Path


BLOCK = 512  # safer for laptops


def _read_block(src, row, col, block):
    """Read block with 1-pixel overlap"""
    win = Window(
        max(col - 1, 0),
        max(row - 1, 0),
        min(block + 2, src.width - col + 1),
        min(block + 2, src.height - row + 1)
    )
    data = src.read(1, window=win).astype(np.float32)
    return data, win


def compute_slope(dem_path: Path, output_path: Path):
    print(f"🗻 Computing slope: {dem_path}")

    with rasterio.open(dem_path) as src:
        meta = src.meta.copy()
        dx = abs(src.transform.a)
        dy = abs(src.transform.e)

        meta.update(dtype="float32", nodata=np.nan, compress="LZW")

        with rasterio.open(output_path, "w", **meta) as dst:
            for row in range(0, src.height, BLOCK):
                for col in range(0, src.width, BLOCK):

                    dem, win = _read_block(src, row, col, BLOCK)
                    dem[dem == src.nodata] = np.nan

                    dzdx = (dem[1:-1, 2:] - dem[1:-1, :-2]) / (2 * dx)
                    dzdy = (dem[2:, 1:-1] - dem[:-2, 1:-1]) / (2 * dy)

                    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

                    dst.write(
                        slope.astype(np.float32),
                        1,
                        window=Window(col, row, slope.shape[1], slope.shape[0])
                    )

    print(f"✅ Slope saved → {output_path}")


def compute_roughness(dem_path: Path, output_path: Path):
    print(f"🌄 Computing roughness: {dem_path}")

    with rasterio.open(dem_path) as src:
        meta = src.meta.copy()
        meta.update(dtype="float32", nodata=np.nan, compress="LZW")

        with rasterio.open(output_path, "w", **meta) as dst:
            for row in range(0, src.height, BLOCK):
                for col in range(0, src.width, BLOCK):

                    dem, win = _read_block(src, row, col, BLOCK)
                    dem[dem == src.nodata] = np.nan

                    neighbors = np.stack([
                        dem[:-2, :-2], dem[:-2, 1:-1], dem[:-2, 2:],
                        dem[1:-1, :-2], dem[1:-1, 1:-1], dem[1:-1, 2:],
                        dem[2:, :-2], dem[2:, 1:-1], dem[2:, 2:]
                    ])

                    rough = np.nanmax(neighbors, axis=0) - np.nanmin(neighbors, axis=0)

                    dst.write(
                        rough.astype(np.float32),
                        1,
                        window=Window(col, row, rough.shape[1], rough.shape[0])
                    )

    print(f"✅ Roughness saved → {output_path}")


def compute_flow_direction(dem_path: Path, output_path: Path):
    print(f"💧 Computing flow direction (D8): {dem_path}")

    # ESRI D8
    directions = np.array([
        (0, 1, 1), (1, 1, 2), (1, 0, 4), (1, -1, 8),
        (0, -1, 16), (-1, -1, 32), (-1, 0, 64), (-1, 1, 128)
    ])

    with rasterio.open(dem_path) as src:
        meta = src.meta.copy()
        meta.update(dtype="uint8", nodata=0, compress="LZW")

        with rasterio.open(output_path, "w", **meta) as dst:
            for row in range(0, src.height, BLOCK):
                for col in range(0, src.width, BLOCK):

                    dem, win = _read_block(src, row, col, BLOCK)
                    dem[dem == src.nodata] = np.nan

                    center = dem[1:-1, 1:-1]
                    drops = []

                    for dr, dc, _ in directions:
                        neigh = dem[1+dr:1+dr+center.shape[0],
                                    1+dc:1+dc+center.shape[1]]
                        drops.append(center - neigh)

                    drops = np.stack(drops)
                    idx = np.argmax(drops, axis=0)
                    max_drop = np.max(drops, axis=0)

                    fd = np.zeros(center.shape, dtype=np.uint8)
                    for i, (_, _, code) in enumerate(directions):
                        fd[(idx == i) & (max_drop > 0)] = code

                    dst.write(
                        fd,
                        1,
                        window=Window(col, row, fd.shape[1], fd.shape[0])
                    )

    print(f"✅ Flow direction saved → {output_path}")


if __name__ == "__main__":
    raw = Path("data/raw/terrain/elevation.tif")
    out = Path("data/processed/terrain")
    out.mkdir(parents=True, exist_ok=True)

    compute_slope(raw, out / "slope.tif")
    compute_roughness(raw, out / "roughness.tif")
    compute_flow_direction(raw, out / "flow_direction.tif")
