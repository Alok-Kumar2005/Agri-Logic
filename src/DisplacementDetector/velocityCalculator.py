import pandas as pd
import numpy as np
from typing import Dict
from sklearn.linear_model import LinearRegression


class VelocityCalculator:
    HAZARD_THRESHOLDS = {
        "STABLE": 2,
        "MODERATE": 8,
        "HIGH": 15,
    }

    @staticmethod
    def calculate_mean_velocity(time_series: pd.DataFrame) -> float:
        if len(time_series) < 5:
            return 0.0

        ts = time_series.copy()
        ts["days"] = (ts["date"] - ts["date"].min()).dt.days

        X = ts[["days"]].values
        y = ts["displacement"].values

        model = LinearRegression()
        model.fit(X, y)

        velocity_mm_day = model.coef_[0]
        velocity_mm_year = velocity_mm_day * 365.25

        return round(float(velocity_mm_year), 2)

    @staticmethod
    def calculate_acceleration(time_series: pd.DataFrame) -> float:
        if len(time_series) < 10:
            return 0.0

        ts = time_series.copy()
        ts["days"] = (ts["date"] - ts["date"].min()).dt.days

        days = ts["days"].values
        disp = ts["displacement"].values
        coeffs = np.polyfit(days, disp, 2)
        accel_mm_day2 = 2 * coeffs[0]
        accel_mm_year2 = accel_mm_day2 * (365.25 ** 2)

        return round(float(accel_mm_year2), 2)

    @staticmethod
    def calculate_seasonality(time_series: pd.DataFrame) -> float:
        if len(time_series) < 12:
            return 0.0

        ts = time_series.copy()
        ts["month"] = ts["date"].dt.month

        monthly_mean = ts.groupby("month")["displacement"].mean()
        seasonality = monthly_mean.max() - monthly_mean.min()

        return round(float(seasonality), 2)

    @classmethod
    def determine_hazard_level(cls, velocity: float) -> str:
        v = abs(velocity)

        if v <= cls.HAZARD_THRESHOLDS["STABLE"]:
            return "STABLE"
        elif v <= cls.HAZARD_THRESHOLDS["MODERATE"]:
            return "MODERATE"
        elif v <= cls.HAZARD_THRESHOLDS["HIGH"]:
            return "HIGH"
        else:
            return "CRITICAL"

    @staticmethod
    def get_trend_direction(velocity: float) -> str:
        if velocity > 2:
            return "UPLIFTING"
        elif velocity < -2:
            return "SUBSIDING"
        else:
            return "STABLE"

    @classmethod
    def analyze_point(cls, time_series: pd.DataFrame, point_data: Dict | None = None) -> Dict:
        mean_velocity = cls.calculate_mean_velocity(time_series)
        acceleration = cls.calculate_acceleration(time_series)
        seasonality = cls.calculate_seasonality(time_series)

        analysis = {
            "mean_velocity_mm_year": mean_velocity,
            "acceleration_mm_year2": acceleration,
            "seasonality_mm": seasonality,
            "hazard_level": cls.determine_hazard_level(mean_velocity),
            "trend_direction": cls.get_trend_direction(mean_velocity),
            "measurement_count": len(time_series),
            "time_span_days": (
                time_series["date"].max() - time_series["date"].min()
            ).days,
        }

        if point_data:
            analysis.update({
                "point_id": point_data.get("pid"),
                "temporal_coherence": point_data.get("temporal_coherence"),
                "easting": point_data.get("easting"),
                "northing": point_data.get("northing"),
            })

        return analysis

    @staticmethod
    def format_time_series(time_series: pd.DataFrame) -> list:
        return [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "displacement_mm": round(float(row["displacement"]), 2),
            }
            for _, row in time_series.iterrows()
        ]


if __name__ == "__main__":
    dates = pd.date_range("2019-01-01", "2023-12-31", freq="12D")
    displacements = np.cumsum(np.random.randn(len(dates))) - 0.2 * np.arange(len(dates))

    ts = pd.DataFrame({
        "date": dates,
        "displacement": displacements,
    })

    analysis = VelocityCalculator.analyze_point(ts)

    print("\nVelocity Analysis")
    for k, v in analysis.items():
        print(f"{k}: {v}")
