import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from scipy.spatial import KDTree
from sklearn.linear_model import LinearRegression

from src.DisplacementDetector import config


class DataProcessor:
    def __init__(self):
        self.egms_dir: Path = config.EGMS_DATA_DIR
        self.kdtree: Optional[KDTree] = None
        self.coord_index = []   # [(file, row_idx)]
        self.coords = []

        if not self.egms_dir.exists():
            raise FileNotFoundError(f"EGMS directory not found: {self.egms_dir}")

    def load_egms_data(self):
        print("Building spatial index (streaming CSVs)...")

        for csv_file in self.egms_dir.glob("*.csv"):
            print(f"Indexing {csv_file.name}")

            for chunk in pd.read_csv(csv_file, chunksize=200_000):
                for idx, row in chunk.iterrows():
                    self.coords.append((row["easting"], row["northing"]))
                    self.coord_index.append((csv_file, idx))

        self.kdtree = KDTree(np.array(self.coords))
        print(f"Indexed {len(self.coords)} EGMS points")

    def find_nearest_point(
        self, easting: float, northing: float, radius_m: float = 100
    ) -> Optional[Dict]:

        if self.kdtree is None:
            self.load_egms_data()

        dist, idx = self.kdtree.query([(easting, northing)], k=1)

        if dist[0] > radius_m:
            return None

        csv_file, row_idx = self.coord_index[idx[0]]

        df = pd.read_csv(csv_file)
        return df.iloc[row_idx].to_dict()

    def extract_time_series(self, point: Dict) -> pd.DataFrame:
        ts = []

        for col, val in point.items():
            if col.isdigit() and len(col) == 8 and pd.notna(val):
                ts.append({
                    "date": pd.to_datetime(col, format="%Y%m%d"),
                    "displacement": float(val),
                })

        return pd.DataFrame(ts).sort_values("date")

    def compute_velocity(self, ts: pd.DataFrame) -> Dict:
        if len(ts) < 5:
            return {"mean_velocity_mm_year": 0.0}

        ts = ts.copy()
        ts["days"] = (ts["date"] - ts["date"].min()).dt.days

        X = ts[["days"]].values
        y = ts["displacement"].values

        model = LinearRegression()
        model.fit(X, y)

        vel_mm_day = model.coef_[0]
        vel_mm_year = vel_mm_day * 365.25

        return {"mean_velocity_mm_year": round(vel_mm_year, 2)}


if __name__ == "__main__":
    processor = DataProcessor()
    processor.load_egms_data()

    sample_easting = 3786327.44
    sample_northing = 2383447.15

    point = processor.find_nearest_point(sample_easting, sample_northing)

    if point:
        print(f"\nPoint ID: {point['pid']}")
        print(f"Coherence: {point['temporal_coherence']}")

        ts = processor.extract_time_series(point)
        print(f"Time series points: {len(ts)}")

        vel = processor.compute_velocity(ts)
        print(f"Mean velocity: {vel['mean_velocity_mm_year']} mm/year")
