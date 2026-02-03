import pandas as pd
import os
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point

def process_industrial_data(input_dir: Path, output_dir: Path):
    print("Processing industrial facility data...")
    air_columns = [
        'PublicationDate', 'countryName', 'reportingYear', 'EPRTR_SectorCode',
        'EPRTR_SectorName', 'EPRTRAnnexIMainActivity', 'FacilityInspireId',
        'facilityName', 'city', 'Longitude', 'Latitude', 'TargetRelease',
        'Pollutant', 'Releases', 'confidentialityReason'
    ]
    
    water_columns = air_columns.copy()
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
            air_df = clean_facility_data(air_df)
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
            water_df = clean_facility_data(water_df)
            output_file = output_dir / 'water_releases.csv'
            water_df.to_csv(output_file, index=False)
            print(f"✓ Saved processed water releases: {output_file}")
            print(f"  Total records: {len(water_df)}")
            print(f"  Unique facilities: {water_df['FacilityInspireId'].nunique()}")
    
    print("\nIndustrial data processing complete!")

def clean_facility_data(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df
        .dropna(subset=['FacilityInspireId', 'Longitude', 'Latitude', 'Pollutant'])
        .copy()
    )

    df.loc[:, 'reportingYear'] = pd.to_numeric(df['reportingYear'], errors='coerce')
    df.loc[:, 'Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df.loc[:, 'Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df.loc[:, 'Releases'] = pd.to_numeric(df['Releases'], errors='coerce')
    df.loc[:, 'EPRTR_SectorCode'] = pd.to_numeric(df['EPRTR_SectorCode'], errors='coerce')

    df = df.loc[
        (df['Longitude'].between(-180, 180)) &
        (df['Latitude'].between(-90, 90)) &
        (df['Releases'] >= 0)
    ].copy()

    df.loc[:, 'city'] = df['city'].fillna('Unknown')
    df.loc[:, 'facilityName'] = df['facilityName'].fillna('Unnamed Facility')

    return df


def create_facility_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    return gdf

def generate_summary_statistics(df: pd.DataFrame) -> dict:
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
    input_dir = Path("data/raw/industrial")
    output_dir = Path("data/processed/industrial")

    output_dir.mkdir(parents=True, exist_ok=True)

    process_industrial_data(input_dir, output_dir)

    # Generate statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    air_file = output_dir / "air_releases.csv"
    water_file = output_dir / "water_releases.csv"

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
