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
                # Fire: continuous release over hours
                emission_rate = magnitude * 0.5  # kg/s (slower burn)
                duration_hours = 4
                effective_height = release_height + 50  # Buoyant plume rise
            elif calamity_type == "explosion":
                # Explosion: instantaneous/rapid release
                # For explosions, use TNT equivalency for blast radius
                # Then model atmospheric dispersion of debris/toxins
                
                # Initial blast effects (mechanical damage)
                blast_radius_km = self._calculate_blast_radius(magnitude)
                
                # Atmospheric dispersion of explosion products
                emission_rate = magnitude * 0.1  # kg/s (rapid but not instant for modeling)
                duration_hours = 0.5
                effective_height = release_height + 100  # Mushroom cloud effect
            else:
                emission_rate = magnitude * 0.01
                duration_hours = 1
                effective_height = release_height
            
            # Ensure minimum wind speed for model stability
            wind_speed = max(wind_speed, 0.5)
            
            # For explosions, use blast radius as minimum critical radius
            if calamity_type == "explosion":
                # Calculate atmospheric dispersion radius
                dispersion_radius = self._calculate_max_distance(
                    emission_rate,
                    wind_speed,
                    stability_class,
                    effective_height,
                    threshold_mg_m3=10.0  # Lower threshold for explosions
                )
                
                # Critical radius is the maximum of blast and dispersion
                max_distance_km = max(blast_radius_km, dispersion_radius)
            else:
                max_distance_km = self._calculate_max_distance(
                    emission_rate,
                    wind_speed,
                    stability_class,
                    effective_height
                )
            
            # Calculate concentration at various distances
            distances = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
            # Filter distances to only those within max_distance
            relevant_distances = [d for d in distances if d <= max_distance_km]
            if not relevant_distances or len(relevant_distances) < 3:
                # Ensure we have at least 3 distance points
                relevant_distances = [
                    max_distance_km * 0.1,
                    max_distance_km * 0.3,
                    max_distance_km * 0.5,
                    max_distance_km * 0.7,
                    max_distance_km * 1.0
                ]
            
            concentrations = []
            
            for dist_km in relevant_distances:
                conc = self._gaussian_plume_concentration(
                    emission_rate,
                    dist_km * 1000,  # convert to meters
                    wind_speed,
                    stability_class,
                    effective_height,
                    0  # ground level
                )
                concentrations.append({
                    "distance_km": round(dist_km, 2),
                    "concentration_mg_m3": round(conc, 4)
                })
            
            # Calculate affected area
            if calamity_type == "explosion":
                # Elliptical plume shape for explosions
                plume_width_factor = 0.5
            else:
                # Narrower plume for fires
                plume_width_factor = 0.3
                
            affected_area_km2 = max_distance_km * max_distance_km * plume_width_factor
            
            # Total release amount
            total_release_kg = emission_rate * duration_hours * 3600
            
            return {
                "calamity_type": calamity_type,
                "emission_rate_kg_s": emission_rate,
                "duration_hours": duration_hours,
                "total_release_kg": total_release_kg,
                "max_distance_km": round(max_distance_km, 2),
                "affected_area_km2": round(affected_area_km2, 2),
                "effective_release_height_m": effective_height,
                "concentrations": concentrations,
                "wind_speed_ms": wind_speed,
                "wind_direction_deg": wind_direction,
                "stability_class": stability_class,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Dispersion simulation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _calculate_blast_radius(self, tnt_equivalent_kg: float) -> float:
        """
        Calculate blast damage radius from TNT equivalent
        
        Uses scaled distance approach:
        R = W^(1/3) * K
        
        Where:
        - R is radius in meters
        - W is TNT equivalent in kg
        - K is scaling factor based on overpressure level
        
        Args:
            tnt_equivalent_kg: TNT equivalent mass in kg
            
        Returns:
            Blast radius in kilometers (severe damage threshold)
        """
        
        # Cube root scaling law for blast radius
        # K factors for different overpressure levels:
        # - 20 psi (138 kPa): severe structural damage, K ≈ 10
        # - 5 psi (34 kPa): moderate damage, K ≈ 18
        # - 1 psi (7 kPa): glass breakage, K ≈ 40
        
        # Use 5 psi (moderate damage) as critical radius
        K = 18  # meters per kg^(1/3)
        
        radius_m = K * (tnt_equivalent_kg ** (1/3))
        radius_km = radius_m / 1000
        
        # Minimum 0.1 km for any explosion
        return max(0.1, radius_km)
    
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
            # At source, use very high concentration
            return emission_rate * 1e6 / (4/3 * np.pi * 10**3)  # Assume 10m radius sphere
        
        # Get dispersion parameters
        params = self.stability_classes.get(stability_class, self.stability_classes['D'])
        
        # Calculate dispersion coefficients (simplified power law)
        # More accurate formulas based on distance ranges
        if distance_m < 1000:
            sigma_y = params['sigma_y'] * (distance_m ** 0.894)
            sigma_z = params['sigma_z'] * (distance_m ** 0.894)
        else:
            # For longer distances, use different scaling
            sigma_y = params['sigma_y'] * 1000**0.894 * (distance_m/1000) ** 0.5
            sigma_z = params['sigma_z'] * 1000**0.894 * (distance_m/1000) ** 0.5
        
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
        try:
            C = (Q / (2 * np.pi * u * sigma_y * sigma_z)) * vertical_term
        except (FloatingPointError, ZeroDivisionError):
            C = 0.0
        
        return float(max(0, C))
    
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
        
        # Adjust threshold based on emission rate for realistic results
        # Higher emissions = use higher threshold to avoid tiny radii
        if emission_rate > 100:  # kg/s
            max_search_km = 50
            # For very high emissions, use higher threshold (10 mg/m³)
            threshold_mg_m3 = max(threshold_mg_m3, 10.0)
        elif emission_rate > 50:
            max_search_km = 40
            threshold_mg_m3 = max(threshold_mg_m3, 5.0)
        elif emission_rate > 10:
            max_search_km = 30
            threshold_mg_m3 = max(threshold_mg_m3, 2.0)
        else:
            max_search_km = 20
        
        # Low wind speed significantly extends impact radius
        if wind_speed < 2.0:
            max_search_km = max_search_km * 1.5
        
        # Logarithmic search for maximum distance
        # Start from 100m (not 10m) for more realistic modeling
        distances = np.logspace(2, np.log10(max_search_km * 1000), 100)  # 100m to max_km
        
        last_valid_distance = 0.5  # Minimum 500m radius
        
        for dist in distances:
            conc = self._gaussian_plume_concentration(
                emission_rate,
                dist,
                wind_speed,
                stability_class,
                release_height
            )
            
            if conc >= threshold_mg_m3:
                last_valid_distance = dist / 1000  # convert to km
            elif last_valid_distance > 0:
                # Found the crossover point
                return last_valid_distance
        
        # If we never dropped below threshold, return max search distance
        return max(last_valid_distance, max_search_km * 0.5)
    
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