import numpy as np
import rasterio
from typing import Dict, Tuple, List, Optional
from pathlib import Path
from shapely.geometry import Point, Polygon
import geopandas as gpd
from src.logging import logging as logger
from src.predict_toxicity.config.settings import settings


class HydrologicalService:
    """Service for hydrological flow modeling and flood simulation"""
    
    def __init__(self):
        self.flow_direction_path = settings.FLOW_DIRECTION_PATH
        self.flow_accumulation_path = settings.FLOW_ACCUMULATION_PATH
        self.dem_path = settings.DEM_PATH
        
    def simulate_flood(
        self,
        site_id: str,
        magnitude: float,
        facility_lat: float = None,
        facility_lon: float = None,
        pollutant_concentration: float = 100.0
    ) -> Dict:
        """
        Simulate flood-induced toxic runoff from industrial site
        
        Args:
            site_id: Facility identifier
            magnitude: Flood depth in meters above base
            facility_lat: Facility latitude
            facility_lon: Facility longitude
            pollutant_concentration: Initial concentration (ppm)
            
        Returns:
            Simulation results with affected area and toxicity
        """
        
        logger.info(f"Simulating flood for site {site_id}, magnitude {magnitude}m")
        
        try:
            # Calculate affected radius based on flood magnitude
            affected_radius_km = self._calculate_flood_radius(magnitude)
            
            # Calculate flow paths and accumulation areas
            flow_paths = self._trace_flow_paths(facility_lat, facility_lon, affected_radius_km)
            
            # Model pollutant transport
            toxicity_map = self._model_pollutant_transport(
                flow_paths,
                pollutant_concentration,
                magnitude
            )
            
            # Calculate impact metrics
            impact_metrics = self._calculate_impact_metrics(
                toxicity_map,
                affected_radius_km
            )
            
            # Generate fallout geometry
            fallout_geometry = self._generate_fallout_polygon(
                facility_lat,
                facility_lon,
                flow_paths,
                affected_radius_km
            )
            
            return {
                "simulation_type": "flood",
                "magnitude_m": magnitude,
                "critical_radius_km": affected_radius_km,
                "affected_metrics": impact_metrics,
                "fallout_geometry": fallout_geometry,
                "toxicity_concentration_ppm": toxicity_map,
                "flow_paths": flow_paths,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Flood simulation failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _calculate_flood_radius(self, magnitude: float) -> float:
        """Calculate affected radius from flood magnitude"""
        # Empirical relationship: radius increases with flood depth
        base_radius = 2.0  # km
        radius = base_radius * (1 + magnitude * 0.5)
        return min(radius, settings.MAX_SIMULATION_RADIUS_KM)
    
    def _trace_flow_paths(
        self,
        start_lat: float,
        start_lon: float,
        max_distance_km: float
    ) -> List[Dict]:
        """
        Trace hydrological flow paths from source point
        
        Returns:
            List of flow path segments with coordinates and properties
        """
        
        # D8 flow direction mapping
        d8_offsets = {
            1: (0, 1),    # E
            2: (1, 1),    # SE
            3: (1, 0),    # S
            4: (1, -1),   # SW
            5: (0, -1),   # W
            6: (-1, -1),  # NW
            7: (-1, 0),   # N
            8: (-1, 1)    # NE
        }
        
        flow_paths = []
        
        # Generate multiple flow paths in different directions
        for direction in range(1, 9):
            path = self._generate_flow_path(
                start_lat,
                start_lon,
                direction,
                max_distance_km
            )
            if path:
                flow_paths.append(path)
        
        return flow_paths
    
    def _generate_flow_path(
        self,
        start_lat: float,
        start_lon: float,
        direction: int,
        max_distance_km: float
    ) -> Dict:
        """Generate a single flow path in specified direction"""
        
        # Approximate conversion: 1 degree ≈ 111 km
        km_per_degree = 111.0
        
        # Direction vectors
        direction_map = {
            1: (0, 1),    # E
            2: (0.707, 0.707),   # SE
            3: (1, 0),    # S
            4: (0.707, -0.707),  # SW
            5: (0, -1),   # W
            6: (-0.707, -0.707), # NW
            7: (-1, 0),   # N
            8: (-0.707, 0.707)   # NE
        }
        
        dx, dy = direction_map.get(direction, (0, 1))
        
        # Generate path points
        num_points = 20
        coords = []
        
        for i in range(num_points):
            distance = (i / num_points) * max_distance_km
            lat = start_lat + (dx * distance / km_per_degree)
            lon = start_lon + (dy * distance / km_per_degree)
            coords.append([lon, lat])
        
        return {
            "direction": direction,
            "coordinates": coords,
            "length_km": max_distance_km,
            "pollutant_retention": 1.0 - (0.1 * (direction % 3))  # Decay factor
        }
    
    def _model_pollutant_transport(
        self,
        flow_paths: List[Dict],
        initial_concentration: float,
        flood_magnitude: float
    ) -> Dict:
        """
        Model pollutant concentration along flow paths
        
        Returns:
            Dictionary of distance: concentration pairs
        """
        
        toxicity_map = {}
        
        # Decay model: concentration decreases with distance
        distances = [0.5, 1.0, 2.0, 5.0, 10.0]
        
        for dist_km in distances:
            # Exponential decay with distance
            decay_factor = np.exp(-0.3 * dist_km)
            
            # Dilution factor from flood volume
            dilution = 1.0 / (1.0 + flood_magnitude * 0.5)
            
            concentration = initial_concentration * decay_factor * dilution
            
            toxicity_map[f"{dist_km}_km"] = round(concentration, 2)
        
        return toxicity_map
    
    def _calculate_impact_metrics(
        self,
        toxicity_map: Dict,
        radius_km: float
    ) -> Dict:
        """Calculate population and land use impacts"""
        
        # Approximate affected area
        affected_area_km2 = np.pi * (radius_km ** 2)
        
        # Estimate population (assume 500 people/km2 average)
        population_density = 500
        est_population = int(affected_area_km2 * population_density)
        
        # Estimate agricultural land (assume 40% of area)
        agri_land_km2 = affected_area_km2 * 0.4
        agri_land_acres = agri_land_km2 * 247.105  # Convert to acres
        
        # Determine health risks based on concentration
        avg_concentration = np.mean(list(toxicity_map.values()))
        
        health_risks = []
        if avg_concentration > 50:
            health_risks.append("Severe contamination risk")
            health_risks.append("Groundwater contamination")
        if avg_concentration > 20:
            health_risks.append("Neurological issues")
            health_risks.append("Soil contamination")
        if avg_concentration > 10:
            health_risks.append("Respiratory stress")
        
        if not health_risks:
            health_risks.append("Low contamination risk")
        
        return {
            "est_population": est_population,
            "affected_area_km2": round(affected_area_km2, 2),
            "agri_land_acres": round(agri_land_acres, 1),
            "primary_toxins": ["Heavy metals", "Industrial effluents"],
            "health_risks": health_risks,
            "avg_toxicity_ppm": round(avg_concentration, 2)
        }
    
    def _generate_fallout_polygon(
        self,
        center_lat: float,
        center_lon: float,
        flow_paths: List[Dict],
        radius_km: float
    ) -> Dict:
        """Generate GeoJSON polygon for affected area"""
        
        # Create circular approximation
        num_points = 32
        km_per_degree = 111.0
        
        coords = []
        for i in range(num_points + 1):
            angle = (i / num_points) * 2 * np.pi
            lat = center_lat + (radius_km / km_per_degree) * np.sin(angle)
            lon = center_lon + (radius_km / km_per_degree) * np.cos(angle)
            coords.append([lon, lat])
        
        return {
            "type": "Polygon",
            "coordinates": [coords]
        }
    
    def calculate_watershed_impact(
        self,
        facility_lat: float,
        facility_lon: float
    ) -> Dict:
        """Calculate watershed delineation for facility location"""
        
        logger.info(f"Calculating watershed for location ({facility_lat}, {facility_lon})")
        
        # Simplified watershed calculation
        # In production, use actual DEM and flow direction data
        
        watershed_area_km2 = np.random.uniform(10, 100)
        
        return {
            "watershed_area_km2": round(watershed_area_km2, 2),
            "outlet_location": {
                "lat": facility_lat,
                "lon": facility_lon
            },
            "stream_order": 3,
            "total_stream_length_km": round(watershed_area_km2 * 0.8, 2)
        }
    
    def estimate_runoff_volume(
        self,
        rainfall_mm: float,
        area_km2: float,
        curve_number: int = 75
    ) -> float:
        """
        Estimate runoff volume using SCS Curve Number method
        
        Args:
            rainfall_mm: Rainfall depth
            area_km2: Catchment area
            curve_number: SCS curve number (0-100)
            
        Returns:
            Runoff volume in cubic meters
        """
        
        # Convert rainfall to inches for SCS method
        rainfall_in = rainfall_mm / 25.4
        
        # Calculate potential maximum retention
        S = (1000 / curve_number) - 10
        
        # Calculate runoff depth
        if rainfall_in > 0.2 * S:
            runoff_in = ((rainfall_in - 0.2 * S) ** 2) / (rainfall_in + 0.8 * S)
        else:
            runoff_in = 0
        
        # Convert to volume
        runoff_mm = runoff_in * 25.4
        runoff_m3 = runoff_mm * area_km2 * 1e6 / 1000
        
        return runoff_m3