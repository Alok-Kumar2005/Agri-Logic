import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from src.logging import logging as logger
from src.predict_toxicity.config.settings import settings


class MeteorologicalService:
    """Service for meteorological data access and processing"""
    
    def __init__(self):
        self.era5_path = settings.ERA5_DATA_PATH
        self.dataset = None
        self._load_dataset()
    
    def _load_dataset(self):
        """Load ERA5 NetCDF dataset"""
        try:
            if self.era5_path.exists():
                self.dataset = xr.open_dataset(self.era5_path)
                logger.info(f"Loaded ERA5 dataset from {self.era5_path}")
                
                # Log available variables and dimensions for debugging
                logger.info(f"ERA5 dimensions: {list(self.dataset.dims.keys())}")
                logger.info(f"ERA5 variables: {list(self.dataset.data_vars.keys())}")
            else:
                logger.warning(f"ERA5 dataset not found at {self.era5_path}")
        except Exception as e:
            logger.error(f"Failed to load ERA5 dataset: {e}")
    
    def get_current_weather(self, lat: float, lon: float) -> Dict:
        """
        Get current meteorological conditions
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Current weather data
        """
        
        if self.dataset is None:
            # Return synthetic data if dataset not available
            return self._generate_synthetic_weather(lat, lon)
        
        try:
            # ERA5 uses 'valid_time' not 'time' - check which dimension exists
            time_dim = None
            if 'time' in self.dataset.dims:
                time_dim = 'time'
            elif 'valid_time' in self.dataset.dims:
                time_dim = 'valid_time'
            else:
                logger.warning(f"No time dimension found in dataset. Available dims: {list(self.dataset.dims.keys())}")
                return self._generate_synthetic_weather(lat, lon)
            
            # Select nearest point and most recent time
            data = self.dataset.sel(
                latitude=lat,
                longitude=lon,
                method='nearest'
            )
            
            # Get latest timestamp
            latest_time = data[time_dim].values[-1]
            latest_data = data.sel({time_dim: latest_time})
            
            # Extract weather parameters with safe defaults
            u10 = float(latest_data.get('u10', 0).values) if 'u10' in latest_data else 0
            v10 = float(latest_data.get('v10', 0).values) if 'v10' in latest_data else 0
            
            return {
                "timestamp": pd.Timestamp(latest_time).isoformat(),
                "location": {"lat": lat, "lon": lon},
                "temperature_c": float(latest_data.get('t2m', 288.15).values) - 273.15 if 't2m' in latest_data else 15.0,
                "wind_speed_ms": float(np.sqrt(u10**2 + v10**2)),
                "wind_direction_deg": float(np.arctan2(v10, u10) * 180 / np.pi) if (u10 != 0 or v10 != 0) else 0,
                "pressure_hpa": float(latest_data.get('sp', 101325).values / 100) if 'sp' in latest_data else 1013.25,
                "boundary_layer_height_m": float(latest_data.get('blh', 1000).values) if 'blh' in latest_data else 1000.0
            }
            
        except Exception as e:
            logger.error(f"Error fetching current weather: {e}")
            return self._generate_synthetic_weather(lat, lon)
    
    def get_historical_weather(
        self,
        lat: float,
        lon: float,
        start_date: datetime,
        end_date: datetime,
        parameters: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get historical weather data for time period
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start datetime
            end_date: End datetime
            parameters: List of parameter names to retrieve
            
        Returns:
            List of weather data records
        """
        
        if self.dataset is None:
            return []
        
        try:
            # Determine time dimension
            time_dim = 'valid_time' if 'valid_time' in self.dataset.dims else 'time'
            
            # Select location and time range
            data = self.dataset.sel(
                latitude=lat,
                longitude=lon,
                **{time_dim: slice(start_date, end_date)},
                method='nearest'
            )
            
            results = []
            for time in data[time_dim].values:
                time_data = data.sel({time_dim: time})
                
                u10 = float(time_data.get('u10', 0).values) if 'u10' in time_data else 0
                v10 = float(time_data.get('v10', 0).values) if 'v10' in time_data else 0
                
                record = {
                    "timestamp": pd.Timestamp(time).isoformat(),
                    "temperature_c": float(time_data.get('t2m', 288.15).values) - 273.15 if 't2m' in time_data else 15.0,
                    "wind_speed_ms": float(np.sqrt(u10**2 + v10**2)),
                    "pressure_hpa": float(time_data.get('sp', 101325).values / 100) if 'sp' in time_data else 1013.25,
                }
                
                if parameters:
                    record = {k: v for k, v in record.items() if k in parameters or k == 'timestamp'}
                
                results.append(record)
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching historical weather: {e}")
            return []
    
    def get_dispersion_parameters(
        self,
        lat: float,
        lon: float,
        timestamp: datetime
    ) -> Dict:
        """
        Get atmospheric dispersion parameters for pollution modeling
        
        Args:
            lat: Latitude
            lon: Longitude
            timestamp: Time of interest
            
        Returns:
            Dispersion parameters
        """
        
        weather = self.get_current_weather(lat, lon)
        
        # Calculate stability class
        stability = self._calculate_stability_class(
            weather['wind_speed_ms'],
            timestamp.hour
        )
        
        return {
            "stability_class": stability,
            "mixing_height_m": weather['boundary_layer_height_m'],
            "wind_u_component": weather['wind_speed_ms'] * np.cos(
                weather['wind_direction_deg'] * np.pi / 180
            ),
            "wind_v_component": weather['wind_speed_ms'] * np.sin(
                weather['wind_direction_deg'] * np.pi / 180
            ),
            "temperature_c": weather['temperature_c'],
            "pressure_hpa": weather['pressure_hpa']
        }
    
    def get_wind_field(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        timestamp: datetime,
        resolution: float = 0.1
    ) -> List[Dict]:
        """
        Get wind field data for visualization
        
        Args:
            min_lat, min_lon, max_lat, max_lon: Bounding box
            timestamp: Time of interest
            resolution: Grid resolution in degrees
            
        Returns:
            List of wind vectors
        """
        
        wind_field = []
        
        lat_points = np.arange(min_lat, max_lat, resolution)
        lon_points = np.arange(min_lon, max_lon, resolution)
        
        for lat in lat_points:
            for lon in lon_points:
                weather = self.get_current_weather(lat, lon)
                
                wind_field.append({
                    "lat": lat,
                    "lon": lon,
                    "u": weather['wind_speed_ms'] * np.cos(
                        weather['wind_direction_deg'] * np.pi / 180
                    ),
                    "v": weather['wind_speed_ms'] * np.sin(
                        weather['wind_direction_deg'] * np.pi / 180
                    ),
                    "speed": weather['wind_speed_ms'],
                    "direction": weather['wind_direction_deg']
                })
        
        return wind_field
    
    def get_forecast(
        self,
        lat: float,
        lon: float,
        hours: int
    ) -> List[Dict]:
        """
        Get weather forecast (simplified - uses current + trend)
        
        Args:
            lat: Latitude
            lon: Longitude
            hours: Number of hours ahead
            
        Returns:
            Forecast data
        """
        
        current = self.get_current_weather(lat, lon)
        forecast = []
        
        for hour in range(hours):
            forecast_time = datetime.now() + timedelta(hours=hour)
            
            # Simple forecast: add small random variations
            forecast.append({
                "timestamp": forecast_time.isoformat(),
                "temperature_c": current['temperature_c'] + np.random.uniform(-2, 2),
                "wind_speed_ms": max(0, current['wind_speed_ms'] + np.random.uniform(-1, 1)),
                "wind_direction_deg": (current['wind_direction_deg'] + np.random.uniform(-15, 15)) % 360,
                "pressure_hpa": current['pressure_hpa'] + np.random.uniform(-2, 2)
            })
        
        return forecast
    
    def _calculate_stability_class(
        self,
        wind_speed: float,
        hour: int
    ) -> str:
        """
        Calculate Pasquill-Gifford stability class
        
        Args:
            wind_speed: Wind speed (m/s)
            hour: Hour of day (0-23)
            
        Returns:
            Stability class (A-F)
        """
        
        # Simplified stability classification
        # Day: 6-18, Night: 18-6
        is_day = 6 <= hour < 18
        
        if wind_speed < 2:
            return 'A' if is_day else 'F'
        elif wind_speed < 3:
            return 'B' if is_day else 'E'
        elif wind_speed < 5:
            return 'C' if is_day else 'D'
        elif wind_speed < 6:
            return 'D'
        else:
            return 'D'
    
    def _generate_synthetic_weather(self, lat: float, lon: float) -> Dict:
        """Generate synthetic weather data when ERA5 unavailable"""
        
        # Generate realistic synthetic data based on location
        base_temp = 15.0 + (lat - 40) * 0.5
        
        return {
            "timestamp": datetime.now().isoformat(),
            "location": {"lat": lat, "lon": lon},
            "temperature_c": base_temp + np.random.uniform(-5, 5),
            "wind_speed_ms": np.random.uniform(2, 8),
            "wind_direction_deg": np.random.uniform(0, 360),
            "pressure_hpa": 1013.25 + np.random.uniform(-10, 10),
            "boundary_layer_height_m": np.random.uniform(500, 2000)
        }
    
    def calculate_mixing_height(
        self,
        temperature_c: float,
        wind_speed_ms: float,
        surface_roughness_m: float = 0.1
    ) -> float:
        """
        Calculate atmospheric mixing height
        
        Args:
            temperature_c: Surface temperature
            wind_speed_ms: Wind speed
            surface_roughness_m: Surface roughness length
            
        Returns:
            Mixing height in meters
        """
        
        # Simplified mixing height calculation
        # Based on mechanical turbulence
        friction_velocity = 0.4 * wind_speed_ms / np.log(10 / surface_roughness_m)
        
        # Coriolis parameter (approximate for mid-latitudes)
        f = 1e-4
        
        mixing_height = 0.3 * friction_velocity / f
        
        return max(100, min(mixing_height, 3000))