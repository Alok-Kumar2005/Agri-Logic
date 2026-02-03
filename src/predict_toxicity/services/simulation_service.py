from typing import Dict, Optional
from src.logging import logging as logger
from src.predict_toxicity.services.facilities_service import FacilitiesService
from src.predict_toxicity.services.hydrological_service import HydrologicalService
from src.predict_toxicity.services.dispersion_service import DispersionService
from src.predict_toxicity.services.meteorological_service import MeteorologicalService
from src.predict_toxicity.services.terrain_service import TerrainService


class SimulationService:
    """
    Main simulation orchestration service
    Coordinates all sub-services for comprehensive disaster modeling
    """
    
    def __init__(self):
        self.facilities_service = FacilitiesService()
        self.hydro_service = HydrologicalService()
        self.dispersion_service = DispersionService()
        self.meteo_service = MeteorologicalService()
        self.terrain_service = TerrainService()
        
        logger.info("SimulationService initialized")
    
    def run_simulation(
        self,
        site_id: str,
        calamity_type: str,
        magnitude: float,
        meteorological_override: Optional[Dict] = None
    ) -> Dict:
        """
        Run complete disaster simulation
        
        Args:
            site_id: Industrial facility identifier
            calamity_type: Type of disaster (flood, fire, earthquake, explosion)
            magnitude: Disaster magnitude
            meteorological_override: Optional weather conditions
            
        Returns:
            Complete simulation results
        """
        
        logger.info(f"Running simulation for {site_id}: {calamity_type} (magnitude: {magnitude})")
        
        try:
            # Step 1: Get facility information
            facility = self.facilities_service.get_by_id(site_id)
            
            if not facility:
                logger.warning(f"Facility {site_id} not found, using synthetic location")
                facility = {
                    "facility_id": site_id,
                    "latitude": 45.0,
                    "longitude": 10.0,
                    "facility_name": "Unknown Facility",
                    "pollutants": [{"name": "Mixed contaminants", "release_amount": 1000}]
                }
            
            lat = facility['latitude']
            lon = facility['longitude']
            
            # Step 2: Get meteorological conditions with proper override handling
            default_weather = self.meteo_service.get_current_weather(lat, lon)
            
            if meteorological_override and isinstance(meteorological_override, dict):
                # Filter out empty nested dicts like {"additionalProp1": {}}
                filtered_override = {
                    k: v for k, v in meteorological_override.items() 
                    if v and not (isinstance(v, dict) and not v)
                }
                # Merge override with default weather to ensure all required keys exist
                weather = {**default_weather, **filtered_override}
            else:
                weather = default_weather
            
            logger.info(f"Weather conditions: {weather.get('wind_speed_ms', 5.0):.1f} m/s wind")
            
            # Step 3: Get terrain data
            elevation = self.terrain_service.get_elevation(lat, lon) or 100.0
            slope = self.terrain_service.get_slope(lat, lon) or 5.0
            
            logger.info(f"Terrain: elevation {elevation}m, slope {slope}°")
            
            # Step 4: Run appropriate simulation model
            if calamity_type.lower() == "flood":
                results = self.hydro_service.simulate_flood(
                    site_id=site_id,
                    magnitude=magnitude,
                    facility_lat=lat,
                    facility_lon=lon,
                    pollutant_concentration=100.0
                )
            
            elif calamity_type.lower() in ["fire", "explosion"]:
                results = self.dispersion_service.simulate_dispersion(
                    site_id=site_id,
                    calamity_type=calamity_type,
                    magnitude=magnitude,
                    wind_speed=weather.get('wind_speed_ms', 5.0),
                    wind_direction=weather.get('wind_direction_deg', 0.0),
                    stability_class='D',
                    release_height=10.0
                )
            
            elif calamity_type.lower() == "earthquake":
                results = self._simulate_earthquake(
                    site_id=site_id,
                    magnitude=magnitude,
                    lat=lat,
                    lon=lon,
                    facility=facility
                )
            
            else:
                results = {
                    "error": f"Unknown calamity type: {calamity_type}",
                    "status": "failed"
                }
            
            # Step 5: Enrich results with additional context
            if results.get('status') == 'completed':
                results['facility_info'] = {
                    "id": facility['facility_id'],
                    "name": facility['facility_name'],
                    "location": {"lat": lat, "lon": lon},
                    "elevation_m": elevation,
                    "slope_deg": slope
                }
                
                results['meteorological_conditions'] = weather
                
                # Add terrain-modified impact assessment
                results = self._apply_terrain_corrections(results, slope)
            
            logger.info(f"Simulation completed: {results.get('status')}")
            return results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def get_results(self, simulation_id: str) -> Dict:
        """
        Retrieve results for a completed simulation
        
        Args:
            simulation_id: Simulation identifier
            
        Returns:
            Simulation results
        """
        
        # In production, retrieve from database
        # For now, return example structure
        
        return {
            "simulation_id": simulation_id,
            "critical_radius_km": 3.8,
            "affected_metrics": {
                "est_population": 14500,
                "affected_area_km2": 45.2,
                "agri_land_acres": 1200,
                "primary_toxins": ["Lead", "Chromium VI"],
                "health_risks": ["Neurological issues", "Groundwater contamination"]
            },
            "fallout_geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [10.0, 45.0],
                    [10.05, 45.0],
                    [10.05, 45.05],
                    [10.0, 45.05],
                    [10.0, 45.0]
                ]]
            },
            "status": "completed"
        }
    
    def _simulate_earthquake(
        self,
        site_id: str,
        magnitude: float,
        lat: float,
        lon: float,
        facility: Dict
    ) -> Dict:
        """
        Simulate earthquake impact on industrial facility
        
        Args:
            site_id: Facility identifier
            magnitude: Earthquake magnitude (Richter scale)
            lat, lon: Facility coordinates
            facility: Facility data
            
        Returns:
            Earthquake simulation results
        """
        
        logger.info(f"Simulating earthquake: magnitude {magnitude}")
        
        # Calculate affected radius based on magnitude
        # Empirical: radius increases exponentially with magnitude
        radius_km = 10 * (10 ** (magnitude / 3))
        radius_km = min(radius_km, 50.0)
        
        # Calculate structural damage probability
        if magnitude < 5.0:
            damage_prob = 0.1
            damage_level = "Minor"
        elif magnitude < 6.0:
            damage_prob = 0.3
            damage_level = "Moderate"
        elif magnitude < 7.0:
            damage_prob = 0.6
            damage_level = "Severe"
        else:
            damage_prob = 0.9
            damage_level = "Critical"
        
        # Estimate pollutant release based on damage
        total_pollutants = sum(
            p.get('release_amount', 0) 
            for p in facility.get('pollutants', [])
        )
        
        released_amount = total_pollutants * damage_prob * 0.2  # 20% of inventory
        
        # Calculate population exposure
        affected_area_km2 = 3.14159 * (radius_km ** 2)
        est_population = int(affected_area_km2 * 500)  # 500 people/km²
        
        # Health risks based on magnitude
        health_risks = ["Structural collapse risk"]
        if magnitude > 5.5:
            health_risks.append("Hazardous material release")
        if magnitude > 6.0:
            health_risks.append("Secondary fires")
            health_risks.append("Water contamination")
        
        # Generate affected area polygon
        import numpy as np
        coords = []
        num_points = 32
        km_per_degree = 111.0
        
        for i in range(num_points + 1):
            angle = (i / num_points) * 2 * np.pi
            lat_offset = (radius_km / km_per_degree) * np.sin(angle)
            lon_offset = (radius_km / km_per_degree) * np.cos(angle)
            coords.append([lon + lon_offset, lat + lat_offset])
        
        return {
            "simulation_type": "earthquake",
            "magnitude_richter": magnitude,
            "critical_radius_km": radius_km,
            "damage_level": damage_level,
            "damage_probability": damage_prob,
            "released_pollutants_kg": released_amount,
            "affected_metrics": {
                "est_population": est_population,
                "affected_area_km2": round(affected_area_km2, 2),
                "primary_toxins": [p['name'] for p in facility.get('pollutants', [])[:3]],
                "health_risks": health_risks
            },
            "fallout_geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            },
            "status": "completed"
        }
    
    def _apply_terrain_corrections(self, results: Dict, slope: float) -> Dict:
        """
        Apply terrain-based corrections to simulation results
        
        Args:
            results: Raw simulation results
            slope: Terrain slope in degrees
            
        Returns:
            Corrected results
        """
        
        # Steep slopes increase runoff and extend impact radius
        if slope > 15:
            slope_factor = 1.3
        elif slope > 8:
            slope_factor = 1.15
        else:
            slope_factor = 1.0
        
        # Apply correction to critical radius
        if 'critical_radius_km' in results:
            original_radius = results['critical_radius_km']
            results['critical_radius_km'] = round(original_radius * slope_factor, 2)
            
            # Recalculate affected area
            if 'affected_metrics' in results:
                area = 3.14159 * (results['critical_radius_km'] ** 2)
                results['affected_metrics']['affected_area_km2'] = round(area, 2)
                
                # Recalculate population
                results['affected_metrics']['est_population'] = int(area * 500)
        
        results['terrain_correction_applied'] = True
        results['terrain_slope_factor'] = slope_factor
        
        return results
    
    def calculate_cumulative_impact(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10.0
    ) -> Dict:
        """
        Calculate cumulative pollution impact for an area
        
        Args:
            lat: Center latitude
            lon: Center longitude
            radius_km: Analysis radius
            
        Returns:
            Cumulative impact assessment
        """
        
        # Get all facilities in radius
        facilities = self.facilities_service.get_nearby(lat, lon, radius_km, limit=100)
        
        total_emissions = 0
        pollutant_types = set()
        
        for facility in facilities:
            for pollutant in facility.get('pollutants', []):
                total_emissions += pollutant.get('release_amount', 0)
                pollutant_types.add(pollutant.get('name', 'Unknown'))
        
        # Calculate risk score (0-100)
        risk_score = min(100, total_emissions / 1000)
        
        if risk_score < 20:
            risk_level = "Low"
        elif risk_score < 50:
            risk_level = "Moderate"
        elif risk_score < 75:
            risk_level = "High"
        else:
            risk_level = "Critical"
        
        return {
            "location": {"lat": lat, "lon": lon},
            "analysis_radius_km": radius_km,
            "total_facilities": len(facilities),
            "total_emissions_kg": total_emissions,
            "unique_pollutants": len(pollutant_types),
            "pollutant_types": list(pollutant_types),
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level
        }