import os
import sys
import ee
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from src.logging import logging
from src.exception import CustomException



class EarthEngineDataFetcher:
    def __init__(self, project_id: str = "gee-hackathon-485713"):
        self.project_id = project_id
        self.S2_BANDS = ["B2","B3","B4","B8","B11","B12"]
        self._initialize_ee()
    
    def _initialize_ee(self):
        try:
            try:
                ee.Initialize(project=self.project_id)
                logging.info(f"Earth Engine initialized with project: {self.project_id}")
            except Exception:
                logging.info("Authenticating Earth Engine...")
                ee.Authenticate()
                ee.Initialize(project=self.project_id)
        except Exception as e:
            logging.error("Earth Engine Init failed")
            raise CustomException(e, sys)
    
    def _mask_s2_sr(self, image):
        scl = image.select("SCL")
        mask = (
            scl.neq(3)   # Cloud shadows
            .And(scl.neq(7))  # Cloud medium probability
            .And(scl.neq(8))  # Cloud high probability
            .And(scl.neq(9))  # Cirrus
            .And(scl.neq(10)) # Snow/ice
            .And(scl.neq(11)) # Saturated/defective
        )
        
        return ( image.updateMask(mask).select(self.S2_BANDS).divide(10000) )
    
    def fetch_satellite_data( self, geometry: ee.Geometry, start_date: str = None, end_date: str = None, cloud_threshold: int = 20 ) -> ee.Image:
        try:
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
            
            logging.info(f"Fetching Sentinel-2 data from {start_date} to {end_date}")
            
            s2_collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterDate(start_date, end_date)
                .filterBounds(geometry)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
                .map(self._mask_s2_sr)
                .median()  # Take median to reduce noise
                .resample("bilinear")
                .reproject(crs="EPSG:4326", scale=100)
            )
            
            ndvi = s2_collection.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ndwi = s2_collection.normalizedDifference(["B3", "B8"]).rename("NDWI")
            savi = s2_collection.expression(
                "((NIR - RED) / (NIR + RED + 0.5)) * 1.5",
                {
                    "NIR": s2_collection.select("B8"),
                    "RED": s2_collection.select("B4")
                }
            ).rename("SAVI")
            
            s2_with_indices = s2_collection.addBands([ndvi, ndwi, savi])
            
            logging.info("Satellite data fetched")
            return s2_with_indices
        
        except Exception as e:
            logging.error("Error fetching satellite data")
            raise CustomException(e, sys)
    
    def fetch_terrain_data(self, geometry: ee.Geometry) -> ee.Image:
        try:
            logging.info("Fetching terrain data ........")
            dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
            slope = ee.Terrain.slope(dem).rename("slope")
            terrain = dem.addBands(slope)
            return terrain
        except Exception as e:
            logging.error("Error fetching terrain data")
            raise CustomException(e, sys)
    
    def fetch_climate_data( self, geometry: ee.Geometry, start_date: str = None, end_date: str = None ) -> ee.Image:
        try:
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
            
            logging.info(f"Fetching GLDAS climate data from {start_date} to {end_date}")
            gldas = (
                ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
                .filterDate(start_date, end_date)
                .select(["Rainf_tavg","SoilMoi0_10cm_inst","Tair_f_inst","Evap_tavg"])
                .mean()
                .resample("bilinear")
            )
            return gldas
        except Exception as e:
            logging.error("Error fetching climate data")
            raise CustomException(e, sys)
    
    def fetch_all_features( self, geometry: ee.Geometry, start_date: str = None, end_date: str = None, scale: int = 100, max_pixels: int = 10000 ) -> ee.Image:
        try:
            logging.info("Fetching all features for AOI .......")
            
            satellite = self.fetch_satellite_data(geometry, start_date, end_date)
            terrain = self.fetch_terrain_data(geometry)
            climate = self.fetch_climate_data(geometry, start_date, end_date)
            combined = (
                satellite
                .addBands(terrain)
                .addBands(climate)
                .clip(geometry)
            )
            return combined
        except Exception as e:
            logging.error("Error fetching all features")
            raise CustomException(e, sys)
    
    def sample_to_dataframe( self, image: ee.Image, geometry: ee.Geometry, scale: int = 100, num_pixels: int = 1000, seed: int = 42 ) -> pd.DataFrame:
        try:
            logging.info(f"Sampling {num_pixels} pixels from image")
            samples = image.sample(
                region=geometry,
                scale=scale,
                numPixels=num_pixels,
                seed=seed,
                geometries=False
            )
            
            features = samples.getInfo()['features']
            data = [feature['properties'] for feature in features]
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            logging.error("Error sampling image to DataFrame")
            raise CustomException(e, sys)
    
    def get_mean_values( self, image: ee.Image, geometry: ee.Geometry, scale: int = 100 ) -> Dict[str, float]:
        try:
            logging.info("Computing mean values for AOI")
            
            mean_dict = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=scale,
                maxPixels=1e9
            ).getInfo()
            return mean_dict
        except Exception as e:
            logging.error("Error computing mean values")
            raise CustomException(e, sys)
    
    @staticmethod
    def create_geometry_from_coords(coordinates: List[List[float]]) -> ee.Geometry:
        return ee.Geometry.Polygon(coordinates)
    
    @staticmethod
    def create_geometry_from_bbox( min_lon: float, min_lat: float, max_lon: float, max_lat: float ) -> ee.Geometry:

        return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])


if __name__ == "__main__":
    try:
        fetcher = EarthEngineDataFetcher()
        test_geometry = fetcher.create_geometry_from_bbox(
            min_lon=75.8,
            min_lat=30.9,
            max_lon=75.9,
            max_lat=31.0
        )
        
        print("\n📡 Fetching Earth Engine data...")
        combined_image = fetcher.fetch_all_features(
            geometry=test_geometry,
            scale=100
        )
        
        print("\n📊 Computing mean values...")
        mean_values = fetcher.get_mean_values(
            image=combined_image,
            geometry=test_geometry,
            scale=100
        )
        
        print("\n✅ Mean feature values:")
        for key, value in mean_values.items():
            if value is not None:
                print(f"  {key}: {value:.6f}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise CustomException(e, sys)




