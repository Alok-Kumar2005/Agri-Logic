import numpy as np
from typing import Dict, Tuple
from src.logging import logging as logger

class DispersionService:
    def __init__(self):
        # Pasquill-Gifford stability classes
        self.stability_classes = {
            'A': {'name': 'Very unstable', 'sigma_y': 0.22, 'sigma_z': 0.20},
            'B': {'name': 'Moderately unstable', 'sigma_y': 0.16, 'sigma_z': 0.12},
            'C': {'name': 'Slightly unstable', 'sigma_y': 0.11, 'sigma_z': 0.08},
            'D': {'name': 'Neutral', 'sigma_y': 0.08, 'sigma_z': 0.06},
            'E': {'name': 'Slightly stable', 'sigma_y': 0.06, 'sigma_z': 0.03},
            'F': {'name': 'Moderately stable', 'sigma_y': 0.04, 'sigma_z': 0.016}
        }
    
    def simulate_dispersion(
        self,
        site_id: str,
        calamity_type: str,
        magnitude: float,
        wind_speed: float = 5.0,
        wind_direction: float = 0.0,
        stability_class: str = 'D',
        release_height: float = 10.0
    ) -> Dict:
        """
        Simulate atmospheric dispersion of pollutants
        
        Args:
            site_id: Facility identifier
            calamity_type: Type of release (fire, explosion)
            magnitude: Release magnitude (kg)
            wind_speed: Wind speed (m/s)
            wind_direction: Wind direction (degrees)
            stability_class: Atmospheric stability class
            release_height: Effective release height (m)
            
        Returns:
            Dispersion results with concentrations
        """
        
        logger.info(f"Simulating {calamity_type} dispersion for site {site_id}")
        
        try:
            # Emission rate calculation based on calamity type
            if calamity_type == "fire":
                emission_rate = magnitude * 10  # kg/s
                duration_hours = 4
            elif calamity_type == "explosion":
                emission_rate = magnitude * 100  # kg/s
                duration_hours = 0.5
            else:
                emission_rate = magnitude
                duration_hours = 1
            
            # Calculate maximum downwind distance where concentration exceeds threshold
            max_distance_km = self._calculate_max_distance(
                emission_rate,
                wind_speed,
                stability_class,
                release_height
            )
            
            # Calculate concentration at various distances
            distances = [0.5, 1.0, 2.0, 5.0, 10.0]
            concentrations = []
            
            for dist_km in distances:
                conc = self._gaussian_plume_concentration(
                    emission_rate,
                    dist_km * 1000,  # convert to meters
                    wind_speed,
                    stability_class,
                    release_height,
                    0  # ground level
                )
                concentrations.append({
                    "distance_km": dist_km,
                    "concentration_mg_m3": round(conc, 4)
                })
            
            # Calculate affected area (approximate)
            plume_width_factor = 0.3  # plume width as fraction of length
            affected_area_km2 = max_distance_km * max_distance_km * plume_width_factor
            
            return {
                "calamity_type": calamity_type,
                "emission_rate_kg_s": emission_rate,
                "duration_hours": duration_hours,
                "total_release_kg": emission_rate * duration_hours * 3600,
                "max_distance_km": max_distance_km,
                "affected_area_km2": affected_area_km2,
                "concentrations": concentrations,
                "wind_speed_ms": wind_speed,
                "wind_direction_deg": wind_direction,
                "stability_class": stability_class,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Dispersion simulation failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _gaussian_plume_concentration(
        self,
        emission_rate: float,
        distance_m: float,
        wind_speed: float,
        stability_class: str,
        release_height: float,
        receptor_height: float = 0.0
    ) -> float:
        """
        Calculate concentration using Gaussian plume model
        
        Args:
            emission_rate: Emission rate (kg/s)
            distance_m: Downwind distance (m)
            wind_speed: Wind speed (m/s)
            stability_class: Pasquill-Gifford stability class
            release_height: Effective release height (m)
            receptor_height: Receptor height above ground (m)
            
        Returns:
            Concentration (mg/m³)
        """
        
        if distance_m <= 0:
            return 0.0
        
        # Get dispersion parameters
        params = self.stability_classes.get(stability_class, self.stability_classes['D'])
        
        # Calculate dispersion coefficients (simplified power law)
        sigma_y = params['sigma_y'] * (distance_m ** 0.894)
        sigma_z = params['sigma_z'] * (distance_m ** 0.894)
        
        # Avoid division by zero
        sigma_y = max(sigma_y, 1.0)
        sigma_z = max(sigma_z, 1.0)
        
        # Gaussian plume equation
        Q = emission_rate * 1e6  # kg/s to mg/s
        u = max(wind_speed, 0.5)
        H = release_height
        z = receptor_height
        
        # Vertical term (with ground reflection)
        vertical_term = (
            np.exp(-0.5 * ((z - H) / sigma_z) ** 2) +
            np.exp(-0.5 * ((z + H) / sigma_z) ** 2)
        )
        
        # Concentration at plume centerline (y=0)
        C = (Q / (2 * np.pi * u * sigma_y * sigma_z)) * vertical_term
        
        return float(C)
    
    def _calculate_max_distance(
        self,
        emission_rate: float,
        wind_speed: float,
        stability_class: str,
        release_height: float,
        threshold_mg_m3: float = 1.0
    ) -> float:
        """
        Calculate maximum distance where concentration exceeds threshold
        
        Returns:
            Maximum distance in kilometers
        """
        
        # Iterative search for maximum distance
        distances = np.logspace(1, 5, 100)  # 10m to 100km
        
        for dist in distances:
            conc = self._gaussian_plume_concentration(
                emission_rate,
                dist,
                wind_speed,
                stability_class,
                release_height
            )
            
            if conc < threshold_mg_m3:
                return dist / 1000  # convert to km
        
        return 50.0  # default maximum
    
    def calculate_dosage(
        self,
        concentration_mg_m3: float,
        exposure_time_hours: float,
        breathing_rate_m3_hr: float = 1.0
    ) -> float:
        """
        Calculate inhaled dose
        
        Args:
            concentration_mg_m3: Air concentration (mg/m³)
            exposure_time_hours: Exposure duration (hours)
            breathing_rate_m3_hr: Breathing rate (m³/hr)
            
        Returns:
            Dose (mg)
        """
        
        dose = concentration_mg_m3 * exposure_time_hours * breathing_rate_m3_hr
        return dose
    
    def determine_stability_class(
        self,
        wind_speed_ms: float,
        solar_radiation: str = "moderate",
        cloud_cover: int = 5
    ) -> str:
        """
        Determine Pasquill-Gifford stability class from meteorological conditions
        
        Args:
            wind_speed_ms: Wind speed (m/s)
            solar_radiation: Solar radiation level (strong/moderate/weak)
            cloud_cover: Cloud cover (0-10 scale)
            
        Returns:
            Stability class (A-F)
        """
        
        # Simplified determination
        # In practice, use Turner's method or similar
        
        if wind_speed_ms < 2:
            if solar_radiation == "strong":
                return 'A'
            elif solar_radiation == "moderate":
                return 'B'
            else:
                return 'E'
        elif wind_speed_ms < 3:
            if solar_radiation == "strong":
                return 'B'
            elif solar_radiation == "moderate":
                return 'C'
            else:
                return 'E'
        elif wind_speed_ms < 5:
            if solar_radiation == "strong":
                return 'C'
            else:
                return 'D'
        else:
            return 'D'  # Neutral for high winds