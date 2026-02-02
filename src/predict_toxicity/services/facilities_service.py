import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import List, Dict, Optional
from src.logging import logging as logger

from config.settings import settings

class FacilitiesService:
    """
    Service for managing industrial facility data from E-PRTR
    """
    
    def __init__(self):
        self.air_releases_df = None
        self.water_releases_df = None
        self._load_data()
    
    def _load_data(self):
        """Load industrial facility data"""
        try:
            # Load air releases
            if settings.INDUSTRIAL_AIR_RELEASES_PATH.exists():
                self.air_releases_df = pd.read_csv(settings.INDUSTRIAL_AIR_RELEASES_PATH)
                logger.info(f"Loaded {len(self.air_releases_df)} air release records")
            
            # Load water releases
            if settings.INDUSTRIAL_WATER_RELEASES_PATH.exists():
                self.water_releases_df = pd.read_csv(settings.INDUSTRIAL_WATER_RELEASES_PATH)
                logger.info(f"Loaded {len(self.water_releases_df)} water release records")
                
        except Exception as e:
            logger.warning(f"Could not load facility data: {e}")
    
    def search(
        self,
        filters: Dict,
        limit: int = 100
    ) -> List[Dict]:
        """
        Search facilities with various filters
        
        Args:
            filters: Dictionary of filter criteria
            limit: Maximum number of results
            
        Returns:
            List of facility dictionaries
        """
        
        # Combine air and water releases
        df = self._get_combined_dataframe()
        
        if df is None or len(df) == 0:
            return []
        
        # Apply filters
        if 'country' in filters:
            df = df[df['countryName'].str.contains(filters['country'], case=False, na=False)]
        
        if 'sector' in filters:
            df = df[df['EPRTR_SectorName'].str.contains(filters['sector'], case=False, na=False)]
        
        if 'pollutant' in filters:
            df = df[df['Pollutant'].str.contains(filters['pollutant'], case=False, na=False)]
        
        if 'year' in filters:
            df = df[df['reportingYear'] == filters['year']]
        
        if 'bbox' in filters:
            # Parse bounding box: min_lon,min_lat,max_lon,max_lat
            bbox = [float(x) for x in filters['bbox'].split(',')]
            df = df[
                (df['Longitude'] >= bbox[0]) &
                (df['Longitude'] <= bbox[2]) &
                (df['Latitude'] >= bbox[1]) &
                (df['Latitude'] <= bbox[3])
            ]
        
        # Group by facility
        facilities = []
        grouped = df.groupby('FacilityInspireId')
        
        for facility_id, group in list(grouped)[:limit]:
            facility = self._format_facility(group)
            facilities.append(facility)
        
        return facilities
    
    def get_by_id(self, facility_id: str) -> Optional[Dict]:
        """
        Get facility by ID
        
        Args:
            facility_id: Facility identifier
            
        Returns:
            Facility dictionary or None
        """
        
        df = self._get_combined_dataframe()
        
        if df is None:
            return None
        
        facility_data = df[df['FacilityInspireId'] == facility_id]
        
        if len(facility_data) == 0:
            return None
        
        return self._format_facility(facility_data)
    
    def get_pollutants(self, facility_id: str) -> List[Dict]:
        """
        Get list of pollutants for a facility
        
        Args:
            facility_id: Facility identifier
            
        Returns:
            List of pollutant dictionaries
        """
        
        df = self._get_combined_dataframe()
        
        if df is None:
            return []
        
        facility_data = df[df['FacilityInspireId'] == facility_id]
        
        pollutants = []
        for _, row in facility_data.iterrows():
            pollutants.append({
                "name": row['Pollutant'],
                "release_amount": float(row['Releases']),
                "target": row['TargetRelease'],
                "year": int(row['reportingYear'])
            })
        
        return pollutants
    
    def get_emissions_history(
        self,
        facility_id: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> List[Dict]:
        """
        Get historical emissions for a facility
        
        Args:
            facility_id: Facility identifier
            start_year: Start year
            end_year: End year
            
        Returns:
            List of emission records by year
        """
        
        df = self._get_combined_dataframe()
        
        if df is None:
            return []
        
        facility_data = df[df['FacilityInspireId'] == facility_id]
        
        if start_year:
            facility_data = facility_data[facility_data['reportingYear'] >= start_year]
        if end_year:
            facility_data = facility_data[facility_data['reportingYear'] <= end_year]
        
        # Group by year and pollutant
        history = []
        for (year, pollutant), group in facility_data.groupby(['reportingYear', 'Pollutant']):
            total_release = group['Releases'].sum()
            history.append({
                "year": int(year),
                "pollutant": pollutant,
                "total_release": float(total_release),
                "targets": group['TargetRelease'].unique().tolist()
            })
        
        return sorted(history, key=lambda x: x['year'], reverse=True)
    
    def get_nearby(
        self,
        lat: float,
        lon: float,
        radius_km: float,
        limit: int = 50
    ) -> List[Dict]:
        """
        Find facilities near a point
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers
            limit: Maximum results
            
        Returns:
            List of nearby facilities
        """
        
        df = self._get_combined_dataframe()
        
        if df is None:
            return []
        
        # Calculate distances (simplified using lat/lon degrees)
        # More accurate would use geodesic distance
        df['distance_deg'] = (
            (df['Longitude'] - lon) ** 2 +
            (df['Latitude'] - lat) ** 2
        ) ** 0.5
        
        # Approximate: 1 degree ≈ 111 km
        radius_deg = radius_km / 111.0
        
        nearby = df[df['distance_deg'] <= radius_deg]
        nearby = nearby.sort_values('distance_deg')
        
        # Group by facility
        facilities = []
        for facility_id, group in list(nearby.groupby('FacilityInspireId'))[:limit]:
            facility = self._format_facility(group)
            facility['distance_km'] = round(group['distance_deg'].min() * 111.0, 2)
            facilities.append(facility)
        
        return facilities
    
    def get_statistics(
        self,
        country: Optional[str] = None,
        sector: Optional[str] = None,
        year: Optional[int] = None
    ) -> Dict:
        """
        Get statistical summary
        
        Args:
            country: Filter by country
            sector: Filter by sector
            year: Filter by year
            
        Returns:
            Statistics dictionary
        """
        
        df = self._get_combined_dataframe()
        
        if df is None:
            return {}
        
        # Apply filters
        if country:
            df = df[df['countryName'] == country]
        if sector:
            df = df[df['EPRTR_SectorName'] == sector]
        if year:
            df = df[df['reportingYear'] == year]
        
        stats = {
            "total_facilities": df['FacilityInspireId'].nunique(),
            "total_records": len(df),
            "countries": df['countryName'].nunique(),
            "sectors": df['EPRTR_SectorName'].nunique(),
            "pollutants": df['Pollutant'].nunique(),
            "total_releases": float(df['Releases'].sum()),
            "top_pollutants": df.groupby('Pollutant')['Releases'].sum()\
                .sort_values(ascending=False).head(10).to_dict(),
            "top_sectors": df.groupby('EPRTR_SectorName')['Releases'].sum()\
                .sort_values(ascending=False).head(5).to_dict()
        }
        
        return stats
    
    def _get_combined_dataframe(self) -> Optional[pd.DataFrame]:
        """Combine air and water releases into single dataframe"""
        
        dfs = []
        
        if self.air_releases_df is not None:
            dfs.append(self.air_releases_df)
        
        if self.water_releases_df is not None:
            dfs.append(self.water_releases_df)
        
        if len(dfs) == 0:
            return None
        
        return pd.concat(dfs, ignore_index=True)
    
    def _format_facility(self, facility_data: pd.DataFrame) -> Dict:
        """Format facility data into dictionary"""
        
        first_row = facility_data.iloc[0]
        
        # Get all pollutants
        pollutants = []
        for _, row in facility_data.iterrows():
            pollutants.append({
                "name": row['Pollutant'],
                "release_amount": float(row['Releases']),
                "target": row['TargetRelease'],
                "year": int(row['reportingYear'])
            })
        
        return {
            "facility_id": first_row['FacilityInspireId'],
            "facility_name": first_row['facilityName'],
            "city": first_row['city'],
            "country": first_row['countryName'],
            "longitude": float(first_row['Longitude']),
            "latitude": float(first_row['Latitude']),
            "sector": first_row['EPRTR_SectorName'],
            "sector_code": int(first_row['EPRTR_SectorCode']),
            "activity": first_row['EPRTRAnnexIMainActivity'],
            "reporting_year": int(first_row['reportingYear']),
            "pollutants": pollutants
        }