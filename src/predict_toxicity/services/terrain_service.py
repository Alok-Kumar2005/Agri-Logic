import rasterio
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from shapely.geometry import Point
from src.logging import logging as logger
from src.predict_toxicity.config.settings import settings


class TerrainService:
    """Service for terrain data access and analysis"""
    
    def __init__(self):
        self.dem_path = settings.DEM_PATH
        self.slope_path = settings.SLOPE_PATH
        self.roughness_path = settings.ROUGHNESS_PATH
        self.flow_direction_path = settings.FLOW_DIRECTION_PATH
        self.flow_accumulation_path = settings.FLOW_ACCUMULATION_PATH
    
    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """
        Get elevation at a point
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Elevation in meters
        """
        
        return self._sample_raster(self.dem_path, lat, lon)
    
    def get_slope(self, lat: float, lon: float) -> Optional[float]:
        """
        Get terrain slope at a point
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Slope in degrees
        """
        
        return self._sample_raster(self.slope_path, lat, lon)
    
    def get_roughness(self, lat: float, lon: float) -> Optional[float]:
        """
        Get terrain roughness at a point
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Roughness in meters
        """
        
        return self._sample_raster(self.roughness_path, lat, lon)
    
    def get_flow_direction(self, lat: float, lon: float) -> Optional[int]:
        """
        Get D8 flow direction at a point
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Flow direction code (1-8)
        """
        
        return self._sample_raster(self.flow_direction_path, lat, lon, dtype=int)
    
    def get_flow_accumulation(self, lat: float, lon: float) -> Optional[float]:
        """
        Get flow accumulation at a point
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Flow accumulation (number of upstream cells)
        """
        
        return self._sample_raster(self.flow_accumulation_path, lat, lon)
    
    def get_terrain_profile(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        num_points: int = 100
    ) -> Dict:
        """
        Get elevation profile between two points
        
        Args:
            start_lat, start_lon: Start coordinates
            end_lat, end_lon: End coordinates
            num_points: Number of sample points
            
        Returns:
            Terrain profile data
        """
        
        if not self.dem_path.exists():
            return self._generate_synthetic_profile(
                start_lat, start_lon, end_lat, end_lon, num_points
            )
        
        points = []
        elevations = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            lat = start_lat + t * (end_lat - start_lat)
            lon = start_lon + t * (end_lon - start_lon)
            
            elev = self.get_elevation(lat, lon)
            if elev is not None:
                points.append({"lat": lat, "lon": lon, "elevation_m": elev})
                elevations.append(elev)
        
        # Calculate distance
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371000  # Earth radius in meters
        lat1, lon1 = radians(start_lat), radians(start_lon)
        lat2, lon2 = radians(end_lat), radians(end_lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        total_distance = R * c
        
        # Calculate elevation gain/loss
        elevation_gain = sum(max(0, elevations[i+1] - elevations[i]) 
                           for i in range(len(elevations)-1))
        elevation_loss = sum(max(0, elevations[i] - elevations[i+1]) 
                           for i in range(len(elevations)-1))
        
        return {
            "points": points,
            "total_distance_m": total_distance,
            "elevation_gain_m": elevation_gain,
            "elevation_loss_m": elevation_loss
        }
    
    def delineate_watershed(
        self,
        lat: float,
        lon: float,
        threshold: float = 1000
    ) -> Dict:
        """
        Delineate watershed for outlet point
        
        Args:
            lat: Outlet latitude
            lon: Outlet longitude
            threshold: Flow accumulation threshold
            
        Returns:
            Watershed geometry
        """
        
        # Simplified watershed delineation
        # In production, use actual flow direction tracing
        
        flow_acc = self.get_flow_accumulation(lat, lon)
        
        if flow_acc is None:
            flow_acc = 5000
        
        # Estimate watershed area from flow accumulation
        cell_area_km2 = 0.09  # 30m resolution ≈ 0.09 km²
        watershed_area_km2 = flow_acc * cell_area_km2
        
        # Generate approximate circular watershed
        radius_km = np.sqrt(watershed_area_km2 / np.pi)
        
        coords = []
        num_points = 32
        km_per_degree = 111.0
        
        for i in range(num_points + 1):
            angle = (i / num_points) * 2 * np.pi
            lat_offset = (radius_km / km_per_degree) * np.sin(angle)
            lon_offset = (radius_km / km_per_degree) * np.cos(angle)
            coords.append([lon + lon_offset, lat + lat_offset])
        
        return {
            "type": "Polygon",
            "coordinates": [coords],
            "area_km2": watershed_area_km2,
            "outlet": {"lat": lat, "lon": lon}
        }
    
    def get_aspect(self, lat: float, lon: float) -> Optional[float]:
        """
        Get terrain aspect (direction slope faces)
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Aspect in degrees (0-360)
        """
        
        if not self.dem_path.exists():
            return None
        
        try:
            with rasterio.open(self.dem_path) as src:
                row, col = src.index(lon, lat)
                
                # Read 3x3 window
                window = rasterio.windows.Window(col-1, row-1, 3, 3)
                dem = src.read(1, window=window)
                
                if dem.shape != (3, 3):
                    return None
                
                # Calculate aspect using gradient
                dzdx = ((dem[0, 2] + 2*dem[1, 2] + dem[2, 2]) - 
                        (dem[0, 0] + 2*dem[1, 0] + dem[2, 0])) / 8
                dzdy = ((dem[2, 0] + 2*dem[2, 1] + dem[2, 2]) - 
                        (dem[0, 0] + 2*dem[0, 1] + dem[0, 2])) / 8
                
                aspect_rad = np.arctan2(dzdy, -dzdx)
                aspect_deg = aspect_rad * 180 / np.pi
                
                # Convert to compass bearing (0-360)
                aspect_deg = (90 - aspect_deg) % 360
                
                return float(aspect_deg)
                
        except Exception as e:
            logger.error(f"Error calculating aspect: {e}")
            return None
    
    def _sample_raster(
        self,
        raster_path: Path,
        lat: float,
        lon: float,
        dtype=float
    ) -> Optional[float]:
        """
        Sample value from raster at point
        
        Args:
            raster_path: Path to raster file
            lat: Latitude
            lon: Longitude
            dtype: Data type for return value
            
        Returns:
            Sampled value or None
        """
        
        if not raster_path.exists():
            logger.warning(f"Raster not found: {raster_path}")
            return None
        
        try:
            with rasterio.open(raster_path) as src:
                # Convert lat/lon to raster coordinates
                row, col = src.index(lon, lat)
                
                # Check bounds
                if (0 <= row < src.height) and (0 <= col < src.width):
                    value = src.read(1)[row, col]
                    
                    # Check for nodata
                    if src.nodata is not None and value == src.nodata:
                        return None
                    
                    return dtype(value)
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error sampling raster {raster_path}: {e}")
            return None
    
    def _generate_synthetic_profile(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        num_points: int
    ) -> Dict:
        """Generate synthetic elevation profile"""
        
        points = []
        
        # Simple sinusoidal profile
        for i in range(num_points):
            t = i / (num_points - 1)
            lat = start_lat + t * (end_lat - start_lat)
            lon = start_lon + t * (end_lon - start_lon)
            
            # Synthetic elevation: base + variation
            base_elevation = 200
            variation = 50 * np.sin(t * np.pi * 3)
            elevation = base_elevation + variation
            
            points.append({
                "lat": lat,
                "lon": lon,
                "elevation_m": elevation
            })
        
        elevations = [p['elevation_m'] for p in points]
        elevation_gain = sum(max(0, elevations[i+1] - elevations[i]) 
                           for i in range(len(elevations)-1))
        elevation_loss = sum(max(0, elevations[i] - elevations[i+1]) 
                           for i in range(len(elevations)-1))
        
        return {
            "points": points,
            "total_distance_m": 10000,
            "elevation_gain_m": elevation_gain,
            "elevation_loss_m": elevation_loss
        }
    
    def calculate_terrain_ruggedness(
        self,
        lat: float,
        lon: float,
        radius_m: float = 1000
    ) -> float:
        """
        Calculate terrain ruggedness index in area around point
        
        Args:
            lat: Center latitude
            lon: Center longitude
            radius_m: Analysis radius
            
        Returns:
            Ruggedness index
        """
        
        if not self.dem_path.exists():
            return 0.0
        
        try:
            with rasterio.open(self.dem_path) as src:
                # Convert radius to pixels
                pixel_size_m = abs(src.transform.a) * 111000
                radius_pixels = int(radius_m / pixel_size_m)
                
                row, col = src.index(lon, lat)
                
                # Read window
                window = rasterio.windows.Window(
                    col - radius_pixels,
                    row - radius_pixels,
                    2 * radius_pixels + 1,
                    2 * radius_pixels + 1
                )
                
                dem = src.read(1, window=window)
                
                # Calculate TRI (Terrain Ruggedness Index)
                # Standard deviation of elevations
                tri = float(np.std(dem[dem != src.nodata]))
                
                return tri
                
        except Exception as e:
            logger.error(f"Error calculating ruggedness: {e}")
            return 0.0