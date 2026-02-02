import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point

def process_industrial_data(input_dir: Path, output_dir: Path):
    """
    Process E-PRTR industrial facility data
    
    Args:
        input_dir: Directory containing raw CSV files
        output_dir: Directory for processed data
    """
    
    print("Processing industrial facility data...")
    
    # Define expected columns
    air_columns = [
        'PublicationDate', 'countryName', 'reportingYear', 'EPRTR_SectorCode',
        'EPRTR_SectorName', 'EPRTRAnnexIMainActivity', 'FacilityInspireId',
        'facilityName', 'city', 'Longitude', 'Latitude', 'TargetRelease',
        'Pollutant', 'Releases', 'confidentialityReason'
    ]
    
    water_columns = air_columns.copy()
    
    # Process air releases
    air_files = list(input_dir.glob('*air*.csv'))
    if air_files:
        print(f"Found {len(air_files)} air release files")
        
        dfs = []
        for file in air_files:
            try:
                df = pd.read_csv(file, low_memory=False)
                df['TargetRelease'] = 'AIR'
                dfs.append(df)
                print(f"  Loaded: {file.name} ({len(df)} records)")
            except Exception as e:
                print(f"  Error loading {file.name}: {e}")
        
        if dfs:
            air_df = pd.concat(dfs, ignore_index=True)
            
            # Clean data
            air_df = clean_facility_data(air_df)
            
            # Save processed data
            output_file = output_dir / 'air_releases.csv'
            air_df.to_csv(output_file, index=False)
            print(f"✓ Saved processed air releases: {output_file}")
            print(f"  Total records: {len(air_df)}")
            print(f"  Unique facilities: {air_df['FacilityInspireId'].nunique()}")
    
    # Process water releases
    water_files = list(input_dir.glob('*water*.csv'))
    if water_files:
        print(f"\nFound {len(water_files)} water release files")
        
        dfs = []
        for file in water_files:
            try:
                df = pd.read_csv(file, low_memory=False)
                df['TargetRelease'] = 'WATER'
                dfs.append(df)
                print(f"  Loaded: {file.name} ({len(df)} records)")
            except Exception as e:
                print(f"  Error loading {file.name}: {e}")
        
        if dfs:
            water_df = pd.concat(dfs, ignore_index=True)
            
            # Clean data
            water_df = clean_facility_data(water_df)
            
            # Save processed data
            output_file = output_dir / 'water_releases.csv'
            water_df.to_csv(output_file, index=False)
            print(f"✓ Saved processed water releases: {output_file}")
            print(f"  Total records: {len(water_df)}")
            print(f"  Unique facilities: {water_df['FacilityInspireId'].nunique()}")
    
    print("\n✅ Industrial data processing complete!")

def clean_facility_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate facility data
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['FacilityInspireId', 'Longitude', 'Latitude', 'Pollutant'])
    
    # Convert data types
    df['reportingYear'] = pd.to_numeric(df['reportingYear'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Releases'] = pd.to_numeric(df['Releases'], errors='coerce')
    df['EPRTR_SectorCode'] = pd.to_numeric(df['EPRTR_SectorCode'], errors='coerce')
    
    # Remove invalid coordinates
    df = df[
        (df['Longitude'] >= -180) & (df['Longitude'] <= 180) &
        (df['Latitude'] >= -90) & (df['Latitude'] <= 90)
    ]
    
    # Remove negative releases
    df = df[df['Releases'] >= 0]
    
    # Fill missing values
    df['city'] = df['city'].fillna('Unknown')
    df['facilityName'] = df['facilityName'].fillna('Unnamed Facility')
    
    return df

def create_facility_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Create GeoDataFrame from facility data
    
    Args:
        df: Facility DataFrame
        
    Returns:
        GeoDataFrame
    """
    
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    return gdf

def generate_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for facility data
    
    Args:
        df: Facility DataFrame
        
    Returns:
        Dictionary of statistics
    """
    
    stats = {
        'total_records': len(df),
        'unique_facilities': df['FacilityInspireId'].nunique(),
        'countries': df['countryName'].nunique(),
        'sectors': df['EPRTR_SectorName'].nunique(),
        'pollutants': df['Pollutant'].nunique(),
        'year_range': (int(df['reportingYear'].min()), int(df['reportingYear'].max())),
        'total_releases': float(df['Releases'].sum()),
        'top_countries': df['countryName'].value_counts().head(10).to_dict(),
        'top_pollutants': df.groupby('Pollutant')['Releases'].sum()\
            .sort_values(ascending=False).head(10).to_dict(),
        'top_sectors': df['EPRTR_SectorName'].value_counts().head(5).to_dict()
    }
    
    return stats

if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / 'data' / 'raw' / 'industrial'
    output_dir = base_dir / 'data' / 'processed' / 'industrial'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process data
    process_industrial_data(input_dir, output_dir)
    
    # Generate statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    # Load processed data
    air_file = output_dir / 'air_releases.csv'
    water_file = output_dir / 'water_releases.csv'
    
    if air_file.exists():
        air_df = pd.read_csv(air_file)
        print("\n📊 AIR RELEASES:")
        stats = generate_summary_statistics(air_df)
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    if water_file.exists():
        water_df = pd.read_csv(water_file)
        print("\n📊 WATER RELEASES:")
        stats = generate_summary_statistics(water_df)
        for key, value in stats.items():
            print(f"  {key}: {value}")