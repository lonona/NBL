import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon

# --- 1. Define directories and load data ---
hmdir = "/home/anarh/PROJECTs/caro/caleb/anly_downscal/"

ghana_xy = pd.read_csv(f"{hmdir}/Data/Basins/ghana.csv")
volta_xy = pd.read_csv(f"{hmdir}/Data/Basins/volta_basin.csv")

gws = xr.open_dataset(hmdir + "Data/gws.nc")

# --- 2. Time selection ---
gws = gws.sel(time=slice('2004-01-01', '2024-12-31'))

# --- 3. Create polygons from boundary points ---
# Fix deprecation warning by using Polygon() directly
ghana_polygon = Polygon(zip(ghana_xy['lon'], ghana_xy['lat']))
volta_polygon = Polygon(zip(volta_xy['lon'], volta_xy['lat']))

# Create GeoDataFrames
ghana_gdf = gpd.GeoDataFrame(geometry=[ghana_polygon], crs="EPSG:4326")
volta_gdf = gpd.GeoDataFrame(geometry=[volta_polygon], crs="EPSG:4326")

# --- 4. Extract GWS data within boundaries using spatial join ---
# Convert GWS data to GeoDataFrame
gws_df = gws['gws'].to_dataframe().reset_index()
gws_points = gpd.GeoDataFrame(
    gws_df, 
    geometry=gpd.points_from_xy(gws_df.lon, gws_df.lat),
    crs="EPSG:4326"
)

# Spatial join to find points within each polygon
gws_ghana_points = gpd.sjoin(gws_points, ghana_gdf, how='inner', predicate='within')
gws_volta_points = gpd.sjoin(gws_points, volta_gdf, how='inner', predicate='within')

print(f"Ghana points found: {len(gws_ghana_points)}")
print(f"Volta points found: {len(gws_volta_points)}")

# Convert back to xarray Datasets
if not gws_ghana_points.empty:
    gws_ghana_ds = gws_ghana_points.set_index(['time', 'lat', 'lon']).to_xarray()
else:
    # Create empty dataset with proper structure
    gws_ghana_ds = xr.Dataset({'gws': (['time', 'lat', 'lon'], np.empty((0,0,0)))})
    
if not gws_volta_points.empty:
    gws_volta_ds = gws_volta_points.set_index(['time', 'lat', 'lon']).to_xarray()
else:
    gws_volta_ds = xr.Dataset({'gws': (['time', 'lat', 'lon'], np.empty((0,0,0)))})

# --- 5. Extract data for 2020 ---
gws_ghana_2020 = gws_ghana_ds.sel(time='2020')
gws_volta_2020 = gws_volta_ds.sel(time='2020')

# Check if we have data for 2020
has_ghana_data = len(gws_ghana_points) > 0 and 'time' in gws_ghana_2020.dims and gws_ghana_2020.dims['time'] == 12
has_volta_data = len(gws_volta_points) > 0 and 'time' in gws_volta_2020.dims and gws_volta_2020.dims['time'] == 12

print(f"Ghana 2020 data available: {has_ghana_data}")
print(f"Volta 2020 data available: {has_volta_data}")

# --- 6. Plot Ghana Monthly GWS for 2020 ---
if has_ghana_data:
    fig, axes = plt.subplots(4, 3, figsize=(12, 15), dpi=300)
    axes = axes.flatten()
    
    # Get color limits for Ghana
    vmin_ghana = float(gws_ghana_2020['gws'].min().values)
    vmax_ghana = float(gws_ghana_2020['gws'].max().values)
    
    for i, month in enumerate(range(1, 13)):
        ax = axes[i]
        
        # Select month data
        ghana_month = gws_ghana_2020.isel(time=i)
        ghana_df = ghana_month.to_dataframe().reset_index()
        
        # Plot Ghana data
        im = ax.scatter(ghana_df['lon'], ghana_df['lat'], 
                       c=ghana_df['gws'], cmap='RdBu_r', vmin=vmin_ghana, vmax=vmax_ghana, s=30)
        
        # Plot Ghana boundary
        ax.plot(ghana_xy['lon'], ghana_xy['lat'], 'k', linewidth=2, label='Ghana')
        
        ax.set_title(f'Ghana - 2020-{month:02d}', fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()
        
        # Add colorbar to the last subplot
        if i == 11:
            cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
            cbar.set_label('GWS (mm)')
    
    plt.suptitle('Monthly Groundwater Storage (GWS) - Ghana - 2020', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Ghana statistics
    print(f"\nGhana GWS Statistics for 2020:")
    print(f"Min: {vmin_ghana:.2f} mm")
    print(f"Max: {vmax_ghana:.2f} mm")
    print(f"Mean: {float(gws_ghana_2020['gws'].mean().values):.2f} mm")
else:
    print("No Ghana data available for 2020 plotting")

# --- 7. Plot Volta Monthly GWS for 2020 ---
if has_volta_data:
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    # Get color limits for Volta
    vmin_volta = float(gws_volta_2020['gws'].min().values)
    vmax_volta = float(gws_volta_2020['gws'].max().values)
    
    for i, month in enumerate(range(1, 13)):
        ax = axes[i]
        
        # Select month data
        volta_month = gws_volta_2020.isel(time=i)
        volta_df = volta_month.to_dataframe().reset_index()
        
        # Plot Volta data
        im = ax.scatter(volta_df['lon'], volta_df['lat'], 
                       c=volta_df['gws'], cmap='RdBu_r', vmin=vmin_volta, vmax=vmax_volta, s=30)
        
        # Plot Volta boundary
        ax.plot(volta_xy['lon'], volta_xy['lat'], 'r', linewidth=2, label='Volta Basin')
        
        ax.set_title(f'Volta Basin - 2020-{month:02d}', fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()
        
        # Add colorbar to the last subplot
        if i == 11:
            cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, extend='both')
            cbar.set_label('GWS (mm)')
    
    plt.suptitle('Monthly Groundwater Storage (GWS) - Volta Basin - 2020', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Volta statistics
    print(f"\nVolta Basin GWS Statistics for 2020:")
    print(f"Min: {vmin_volta:.2f} mm")
    print(f"Max: {vmax_volta:.2f} mm")
    print(f"Mean: {float(gws_volta_2020['gws'].mean().values):.2f} mm")
else:
    print("No Volta data available for 2020 plotting")

# --- 8. Save extracted datasets for future analysis ---
print("\n=== Data Extraction Summary ===")
print(f"Full Ghana GWS dataset shape: {gws_ghana_ds['gws'].shape if has_ghana_data else 'No data'}")
print(f"Full Volta GWS dataset shape: {gws_volta_ds['gws'].shape if has_volta_data else 'No data'}")

# You can save these datasets if needed:
# gws_ghana_ds.to_netcdf(f"{hmdir}/Data/gws_ghana.nc")
# gws_volta_ds.to_netcdf(f"{hmdir}/Data/gws_volta.nc")

# --- 9. Optional: Time series analysis for specific points ---
if has_ghana_data:
    # Example: Extract time series for a specific point in Ghana
    try:
        # Get first valid point
        ghana_time_series = gws_ghana_ds.mean(dim=['lat', 'lon'], skipna=True)
        plt.figure(figsize=(12, 6))
        plt.plot(ghana_time_series.time, ghana_time_series.gws, 'b-', linewidth=2)
        plt.title('Mean GWS Time Series - Ghana', fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('GWS (mm)')
        plt.grid(True, alpha=0.3)
        plt.show()
    except:
        print("Could not create Ghana time series plot")

if has_volta_data:
    # Example: Extract time series for a specific point in Volta
    try:
        # Get first valid point
        volta_time_series = gws_volta_ds.mean(dim=['lat', 'lon'], skipna=True)
        plt.figure(figsize=(12, 6))
        plt.plot(volta_time_series.time, volta_time_series.gws, 'r-', linewidth=2)
        plt.title('Mean GWS Time Series - Volta Basin', fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('GWS (mm)')
        plt.grid(True, alpha=0.3)
        plt.show()
    except:
        print("Could not create Volta time series plot")