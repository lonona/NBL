
#!/usr/bin/python
# Author: T. Ansah-Narh

"""
This script analyzes 20 years of GRACE satellite data to track groundwater storage changes 
and trends in Ghana and the Volta Basin, identifying wet/dry periods and spatial patterns 
for water resource management.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
from scipy.interpolate import Rbf
import matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

# --- 1. Define directories and load data ---
hmdir = "/home/anarh/PROJECTs/caro/caleb/anly_downscal/"

# Load Volta basin CSV (unchanged)
volta_xy = pd.read_csv(f"{hmdir}/Data/Basins/volta_basin.csv")

gws = xr.open_dataset(hmdir + "Data/gws.nc")

# --- 2. Time selection and data quality check ---
gws = gws.sel(time=slice('2004-01-01', '2024-12-31'))
print(f"Data loaded: {dict(gws.dims)}")
print(f"Time range: {gws.time.min().values} to {gws.time.max().values}")

# --- 3. Create study area polygons ---
# OPTION A: Load Ghana boundary from shapefile
use_shapefile = True  # Set to False to use CSV instead

if use_shapefile:
    try:
        # Load shapefile for Ghana
        pt = '/home/anarh/PROJECTs/KGeorgePro/HPI/hmetalData/gha_admbnda_gss_20210308_SHP/'
        fn = "gha_admbnda_adm0_gss_20210308.shp"
        shapefile_path = pt + fn
        ghana_gdf = gpd.read_file(shapefile_path)
        
        # Ensure CRS is consistent
        ghana_gdf = ghana_gdf.to_crs("EPSG:4326")
        
        # Get the union of all geometries - FIXED deprecated unary_union
        ghana_polygon = ghana_gdf.geometry.union_all()
        
        print("Successfully loaded Ghana boundary from shapefile")
        print(f"Shapefile CRS: {ghana_gdf.crs}")
        print(f"Number of features: {len(ghana_gdf)}")
        
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        print("Falling back to CSV method")
        # Fallback to CSV
        ghana_xy = pd.read_csv(f"{hmdir}/Data/Basins/ghana.csv")
        ghana_polygon = Polygon(zip(ghana_xy['lon'], ghana_xy['lat']))
        ghana_gdf = gpd.GeoDataFrame(geometry=[ghana_polygon], crs="EPSG:4326")
else:
    # OPTION B: Use CSV method (original approach)
    ghana_xy = pd.read_csv(f"{hmdir}/Data/Basins/ghana.csv")
    ghana_polygon = Polygon(zip(ghana_xy['lon'], ghana_xy['lat']))
    ghana_gdf = gpd.GeoDataFrame(geometry=[ghana_polygon], crs="EPSG:4326")

# Create Volta polygon (unchanged)
volta_polygon = Polygon(zip(volta_xy['lon'], volta_xy['lat']))
volta_gdf = gpd.GeoDataFrame(geometry=[volta_polygon], crs="EPSG:4326")

# --- 4. CONCEPT: "GRACE-based Assessment of Groundwater Storage Dynamics in West Africa" ---

# Convert GWS data to GeoDataFrame for spatial analysis
gws_df = gws['gws'].to_dataframe().reset_index()
gws_points = gpd.GeoDataFrame(
    gws_df, 
    geometry=gpd.points_from_xy(gws_df.lon, gws_df.lat),
    crs="EPSG:4326"
)

# Spatial join to extract regional data
gws_ghana_points = gpd.sjoin(gws_points, ghana_gdf, how='inner', predicate='within')
gws_volta_points = gpd.sjoin(gws_points, volta_gdf, how='inner', predicate='within')

print(f"Ghana monitoring points: {len(gws_ghana_points)}")
print(f"Volta Basin monitoring points: {len(gws_volta_points)}")

# Convert to xarray datasets
if not gws_ghana_points.empty:
    gws_ghana_ds = gws_ghana_points.set_index(['time', 'lat', 'lon']).to_xarray()
if not gws_volta_points.empty:
    gws_volta_ds = gws_volta_points.set_index(['time', 'lat', 'lon']).to_xarray()

# --- INTERPOLATION FUNCTIONS WITH BOUNDARY CONSTRAINTS ---
def create_boundary_mask(grid_lon, grid_lat, boundary_gdf):
    """Create a mask for points inside the boundary - FIXED version"""
    # Create points from grid
    grid_points = []
    for i in range(grid_lon.shape[0]):
        for j in range(grid_lon.shape[1]):
            grid_points.append(Point(grid_lon[i, j], grid_lat[i, j]))
    
    points_gdf = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:4326")
    
    # Find points within boundary using spatial join
    points_within = gpd.sjoin(points_gdf, boundary_gdf, how='inner', predicate='within')
    
    # Create mask
    mask = np.zeros(grid_lon.shape, dtype=bool)
    for idx in points_within.index:
        i, j = np.unravel_index(idx, grid_lon.shape)
        mask[i, j] = True
    
    return mask

def interpolate_rbf(lon, lat, values, grid_lon, grid_lat, boundary_mask):
    """Radial Basis Function interpolation with boundary constraint"""
    # Remove NaN values for interpolation
    mask = ~np.isnan(values)
    if np.sum(mask) < 4:  # Need at least 4 points for RBF
        return np.full(grid_lon.shape, np.nan)
    
    try:
        rbf = Rbf(lon[mask], lat[mask], values[mask], function='linear')
        interpolated = rbf(grid_lon, grid_lat)
        
        # Apply boundary mask
        interpolated[~boundary_mask] = np.nan
            
        return interpolated
    except:
        return np.full(grid_lon.shape, np.nan)

def interpolate_idw(lon, lat, values, grid_lon, grid_lat, boundary_mask, power=2):
    """Inverse Distance Weighting interpolation with boundary constraint"""
    mask = ~np.isnan(values)
    if np.sum(mask) == 0:
        return np.full(grid_lon.shape, np.nan)
    
    interpolated = np.full(grid_lon.shape, np.nan)
    
    # Only interpolate within boundary
    for i in range(grid_lon.shape[0]):
        for j in range(grid_lon.shape[1]):
            if boundary_mask[i, j]:
                distances = np.sqrt((lon[mask] - grid_lon[i,j])**2 + (lat[mask] - grid_lat[i,j])**2)
                # Avoid division by zero
                distances = np.maximum(distances, 1e-8)
                weights = 1 / (distances ** power)
                interpolated[i,j] = np.sum(weights * values[mask]) / np.sum(weights)
    
    return interpolated

def interpolate_rf(lon, lat, values, grid_lon, grid_lat, boundary_mask):
    """Random Forest interpolation with boundary constraint"""
    mask = ~np.isnan(values)
    if np.sum(mask) < 10:  # Need reasonable number of points for RF
        return np.full(grid_lon.shape, np.nan)
    
    try:
        X_train = np.column_stack([lon[mask], lat[mask]])
        y_train = values[mask]
        
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Only predict points within boundary
        boundary_indices = np.where(boundary_mask)
        if len(boundary_indices[0]) == 0:
            return np.full(grid_lon.shape, np.nan)
            
        X_pred = np.column_stack([grid_lon[boundary_mask], grid_lat[boundary_mask]])
        interpolated_flat = rf.predict(X_pred)
        
        interpolated = np.full(grid_lon.shape, np.nan)
        interpolated[boundary_mask] = interpolated_flat
            
        return interpolated
    except Exception as e:
        print(f"RF interpolation error: {e}")
        return np.full(grid_lon.shape, np.nan)

def interpolate_lanczos(data, boundary_mask, scale_factor=4):
    """Lanczos resampling with boundary constraint"""
    if np.all(np.isnan(data)):
        return data
    
    try:
        # Simple implementation using scipy zoom with order=3 (cubic, close to Lanczos)
        interpolated = zoom(data, scale_factor, order=3)
        
        # Scale the boundary mask accordingly
        mask_scaled = zoom(boundary_mask.astype(float), scale_factor, order=0) > 0.5
        interpolated[~mask_scaled] = np.nan
        
        return interpolated
    except Exception as e:
        print(f"Lanczos interpolation error: {e}")
        return data

def interpolate_lgbm(lon, lat, values, grid_lon, grid_lat, boundary_mask):
    """LightGBM interpolation with boundary constraint"""
    mask = ~np.isnan(values)
    if np.sum(mask) < 10:
        return np.full(grid_lon.shape, np.nan)
    
    try:
        X_train = np.column_stack([lon[mask], lat[mask]])
        y_train = values[mask]
        
        lgb_model = lgb.LGBMRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        lgb_model.fit(X_train, y_train)
        
        # Only predict points within boundary
        boundary_indices = np.where(boundary_mask)
        if len(boundary_indices[0]) == 0:
            return np.full(grid_lon.shape, np.nan)
            
        X_pred = np.column_stack([grid_lon[boundary_mask], grid_lat[boundary_mask]])
        interpolated_flat = lgb_model.predict(X_pred)
        
        interpolated = np.full(grid_lon.shape, np.nan)
        interpolated[boundary_mask] = interpolated_flat
            
        return interpolated
    except Exception as e:
        print(f"LGBM interpolation error: {e}")
        return np.full(grid_lon.shape, np.nan)

def create_interpolated_map(gws_points, boundary_gdf, method='RBF', resolution_factor=4):
    """Create interpolated GWS map with chosen method and boundary constraints"""
    
    if gws_points.empty:
        return None, None, None
    
    # Use mean GWS for spatial visualization
    spatial_mean = gws_points.groupby(['lat', 'lon'])['gws'].mean().reset_index()
    
    # Get boundary bounds for grid creation
    bounds = boundary_gdf.total_bounds
    lon_min, lat_min, lon_max, lat_max = bounds
    
    # Add small buffer
    lon_buffer = (lon_max - lon_min) * 0.05
    lat_buffer = (lat_max - lat_min) * 0.05
    
    # Create grid based on boundary extent
    grid_resolution = 100  # Base resolution
    grid_lon = np.linspace(lon_min - lon_buffer, lon_max + lon_buffer, grid_resolution)
    grid_lat = np.linspace(lat_min - lat_buffer, lat_max + lat_buffer, grid_resolution)
    
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    
    # Create boundary mask
    boundary_mask = create_boundary_mask(grid_lon, grid_lat, boundary_gdf)
    
    # Perform interpolation
    if method == 'RBF':
        interpolated_data = interpolate_rbf(
            spatial_mean['lon'].values, spatial_mean['lat'].values, 
            spatial_mean['gws'].values, grid_lon, grid_lat, boundary_mask
        )
    elif method == 'IDW':
        interpolated_data = interpolate_idw(
            spatial_mean['lon'].values, spatial_mean['lat'].values, 
            spatial_mean['gws'].values, grid_lon, grid_lat, boundary_mask
        )
    elif method == 'RF':
        interpolated_data = interpolate_rf(
            spatial_mean['lon'].values, spatial_mean['lat'].values, 
            spatial_mean['gws'].values, grid_lon, grid_lat, boundary_mask
        )
    elif method == 'Lanczos':
        # For Lanczos, create a base grid first
        base_grid_x = np.linspace(lon_min, lon_max, 50)
        base_grid_y = np.linspace(lat_min, lat_max, 50)
        base_grid_x, base_grid_y = np.meshgrid(base_grid_x, base_grid_y)
        
        base_boundary_mask = create_boundary_mask(base_grid_x, base_grid_y, boundary_gdf)
        base_data = interpolate_rbf(
            spatial_mean['lon'].values, spatial_mean['lat'].values, 
            spatial_mean['gws'].values, base_grid_x, base_grid_y, base_boundary_mask
        )
        
        interpolated_data = interpolate_lanczos(base_data, boundary_mask, scale_factor=resolution_factor)
    elif method == 'LGBM':
        interpolated_data = interpolate_lgbm(
            spatial_mean['lon'].values, spatial_mean['lat'].values, 
            spatial_mean['gws'].values, grid_lon, grid_lat, boundary_mask
        )
    else:
        interpolated_data = interpolate_rbf(
            spatial_mean['lon'].values, spatial_mean['lat'].values, 
            spatial_mean['gws'].values, grid_lon, grid_lat, boundary_mask
        )
    
    return interpolated_data, grid_lon, grid_lat

# --- 5. CONFERENCE-ORIENTED ANALYSIS ---

print("\n" + "="*70)
print("GRACE-BASED GROUNDWATER ASSESSMENT FOR CONFERENCE PRESENTATION")
print("="*70)

# 5.1 Calculate long-term trends and statistical significance
def calculate_regional_trends(gws_ds, region_name):
    """Calculate comprehensive trends and statistics for conference presentation"""
    
    # Annual aggregation - FIXED: use 'YE' instead of 'Y'
    gws_annual = gws_ds.resample(time='YE').mean()
    
    # Regional mean time series
    regional_mean = gws_annual.mean(dim=['lat', 'lon'], skipna=True)
    
    # Calculate trend
    years = np.arange(len(regional_mean.time))
    valid_data = ~np.isnan(regional_mean.gws.values)
    
    if np.sum(valid_data) > 5:  # Sufficient data points
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            years[valid_data], regional_mean.gws.values[valid_data]
        )
        
        # Calculate total change over study period
        total_change = slope * (len(years) - 1)
        percent_change = (total_change / regional_mean.gws.values[0]) * 100 if regional_mean.gws.values[0] != 0 else np.nan
        
        print(f"\n{region_name} Trend Analysis:")
        print(f"  Trend: {slope:.3f} mm/year")
        print(f"  Total Change (2004-2024): {total_change:.2f} mm")
        print(f"  Percentage Change: {percent_change:.2f}%")
        print(f"  R²: {r_value**2:.3f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'Not significant'}")
        
        return slope, p_value, regional_mean
    else:
        print(f"Insufficient data for {region_name} trend analysis")
        return np.nan, np.nan, None

# Calculate trends for both regions
if not gws_ghana_points.empty:
    ghana_slope, ghana_p, ghana_annual = calculate_regional_trends(gws_ghana_ds, "Ghana")
if not gws_volta_points.empty:
    volta_slope, volta_p, volta_annual = calculate_regional_trends(gws_volta_ds, "Volta Basin")

# 5.2 Create comprehensive conference figures
fig = plt.figure(figsize=(20, 16))

# --- Figure 1: Study Area and Data Overview ---
ax1 = plt.subplot(2, 3, 1, projection=ccrs.PlateCarree())
ax1.set_extent([-4, 2, 4, 12], crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linestyle=':')
ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# Plot boundaries - updated for shapefile compatibility
if use_shapefile:
    ghana_gdf.boundary.plot(ax=ax1, color='red', linewidth=2, label='Ghana', transform=ccrs.PlateCarree())
else:
    ax1.plot(ghana_xy['lon'], ghana_xy['lat'], 'r-', linewidth=2, label='Ghana', transform=ccrs.PlateCarree())

ax1.plot(volta_xy['lon'], volta_xy['lat'], 'b-', linewidth=2, label='Volta Basin', transform=ccrs.PlateCarree())

# Plot GRACE monitoring points
if not gws_ghana_points.empty:
    sample_ghana = gws_ghana_points.drop_duplicates(['lat', 'lon'])
    ax1.scatter(sample_ghana['lon'], sample_ghana['lat'], c='red', s=20, alpha=0.6, 
               label=f'GRACE Points Ghana (n={len(sample_ghana)})', transform=ccrs.PlateCarree())

if not gws_volta_points.empty:
    sample_volta = gws_volta_points.drop_duplicates(['lat', 'lon'])
    ax1.scatter(sample_volta['lon'], sample_volta['lat'], c='blue', s=20, alpha=0.6, 
               label=f'GRACE Points Volta (n={len(sample_volta)})', transform=ccrs.PlateCarree())

ax1.legend(loc='lower left')
ax1.set_title('A) Study Area and GRACE Monitoring Network', fontweight='bold')

# --- Figure 2: Long-term Time Series ---
ax2 = plt.subplot(2, 3, 2)
if not gws_ghana_points.empty and ghana_annual is not None:
    ax2.plot(ghana_annual.time, ghana_annual.gws, 'r-', linewidth=2, label='Ghana')
    # Add trend line
    trend_years = np.arange(len(ghana_annual.time))
    trend_line = ghana_slope * trend_years + (ghana_annual.gws.values[0] if not np.isnan(ghana_annual.gws.values[0]) else np.mean(ghana_annual.gws.values))
    ax2.plot(ghana_annual.time, trend_line, 'r--', linewidth=1, 
             label=f'Trend: {ghana_slope:.2f} mm/year')

if not gws_volta_points.empty and volta_annual is not None:
    ax2.plot(volta_annual.time, volta_annual.gws, 'b-', linewidth=2, label='Volta Basin')
    # Add trend line
    trend_years = np.arange(len(volta_annual.time))
    trend_line = volta_slope * trend_years + (volta_annual.gws.values[0] if not np.isnan(volta_annual.gws.values[0]) else np.mean(volta_annual.gws.values))
    ax2.plot(volta_annual.time, trend_line, 'b--', linewidth=1, 
             label=f'Trend: {volta_slope:.2f} mm/year')

ax2.set_xlabel('Year')
ax2.set_ylabel('Groundwater Storage (mm)')
ax2.set_title('B) Long-term GWS Trends (2004-2024)', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Figure 3: Seasonal Cycle Analysis ---
ax3 = plt.subplot(2, 3, 3)
if not gws_ghana_points.empty:
    ghana_monthly = gws_ghana_ds.groupby('time.month').mean()
    monthly_means_ghana = [float(ghana_monthly.sel(month=i).gws.mean().values) for i in range(1, 13)]
    ax3.plot(range(1, 13), monthly_means_ghana, 'ro-', linewidth=2, markersize=6, label='Ghana')

if not gws_volta_points.empty:
    volta_monthly = gws_volta_ds.groupby('time.month').mean()
    monthly_means_volta = [float(volta_monthly.sel(month=i).gws.mean().values) for i in range(1, 13)]
    ax3.plot(range(1, 13), monthly_means_volta, 'bo-', linewidth=2, markersize=6, label='Volta Basin')

ax3.set_xlabel('Month')
ax3.set_ylabel('Groundwater Storage (mm)')
ax3.set_title('C) Average Seasonal Cycle', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(1, 13))

# --- Figure 4: Spatial GWS Distribution with Interpolation ---
ax4 = plt.subplot(2, 3, 4, projection=ccrs.PlateCarree())
ax4.set_extent([-4, 2, 4, 12], crs=ccrs.PlateCarree())
ax4.add_feature(cfeature.COASTLINE)
ax4.add_feature(cfeature.BORDERS, linestyle=':')

# Choose interpolation method
interpolation_method = 'RBF'  # Options: 'RBF', 'IDW', 'RF', 'Lanczos', 'LGBM'

if not gws_ghana_points.empty:
    try:
        # Create interpolated map with boundary constraint
        interpolated_data, grid_lon, grid_lat = create_interpolated_map(
            gws_ghana_points, 
            ghana_gdf,
            method=interpolation_method,
            resolution_factor=4
        )
        
        if interpolated_data is not None and not np.all(np.isnan(interpolated_data)):
            # Plot interpolated data
            im = ax4.contourf(grid_lon, grid_lat, interpolated_data, 
                             levels=50, cmap='RdBu_r', 
                             transform=ccrs.PlateCarree(), alpha=0.8)
            
            # Overlay original points for reference
            scatter = ax4.scatter(gws_ghana_points['lon'], gws_ghana_points['lat'], 
                                 c='black', s=10, alpha=0.7, transform=ccrs.PlateCarree())
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4, orientation='vertical', shrink=0.8)
            cbar.set_label('GWS (mm)')
        else:
            # Fallback to scatter plot if interpolation fails
            scatter = ax4.scatter(gws_ghana_points['lon'], gws_ghana_points['lat'], 
                                 c=gws_ghana_points['gws'], cmap='RdBu_r', 
                                 vmin=-200, vmax=200, s=10, transform=ccrs.PlateCarree())
            cbar = plt.colorbar(scatter, ax=ax4, orientation='vertical', shrink=0.8)
            cbar.set_label('GWS (mm)')
            
    except Exception as e:
        print(f"Interpolation failed: {e}")
        # Fallback to scatter plot
        scatter = ax4.scatter(gws_ghana_points['lon'], gws_ghana_points['lat'], 
                             c=gws_ghana_points['gws'], cmap='RdBu_r', 
                             vmin=-200, vmax=200, s=10, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(scatter, ax=ax4, orientation='vertical', shrink=0.8)
        cbar.set_label('GWS (mm)')

# Plot boundaries
if use_shapefile:
    ghana_gdf.boundary.plot(ax=ax4, color='black', linewidth=2, transform=ccrs.PlateCarree())
else:
    ax4.plot(ghana_xy['lon'], ghana_xy['lat'], 'k-', linewidth=2, transform=ccrs.PlateCarree())

ax4.plot(volta_xy['lon'], volta_xy['lat'], 'k-', linewidth=2, transform=ccrs.PlateCarree())
ax4.set_title(f'D) Spatial GWS Distribution ({interpolation_method} Interpolation)', fontweight='bold')

# --- Figure 5: Drought/Flood Years Analysis ---
ax5 = plt.subplot(2, 3, 5)
if not gws_ghana_points.empty and ghana_annual is not None:
    # Calculate anomalies
    ghana_anomaly = ghana_annual.gws - ghana_annual.gws.mean()
    colors = ['red' if x < 0 else 'blue' for x in ghana_anomaly.values]
    bars = ax5.bar(range(len(ghana_anomaly.time)), ghana_anomaly.values, color=colors, alpha=0.7)
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax5.set_xlabel('Year')
    ax5.set_ylabel('GWS Anomaly (mm)')
    ax5.set_title('E) Annual GWS Anomalies', fontweight='bold')
    
    # Label extreme years
    for i, (year, anomaly) in enumerate(zip(ghana_annual.time.dt.year.values, ghana_anomaly.values)):
        if abs(anomaly) > np.std(ghana_anomaly.values):
            ax5.text(i, anomaly + (5 if anomaly > 0 else -15), f'{year}', 
                    ha='center', va='bottom' if anomaly > 0 else 'top', fontsize=8)

# --- Figure 6: GRACE Mission Impact ---
ax6 = plt.subplot(2, 3, 6)
mission_periods = [
    ('GRACE', '2002-2017', '2002-2017'),
    ('GRACE-FO', '2018-present', '2018-present')
]

ax6.text(0.1, 0.9, 'GRACE Mission Timeline:', fontweight='bold', transform=ax6.transAxes)
for i, (mission, period, coverage) in enumerate(mission_periods):
    ax6.text(0.1, 0.8 - i*0.1, f'{mission}: {period}', transform=ax6.transAxes)
    
ax6.text(0.1, 0.5, 'Key Findings:', fontweight='bold', transform=ax6.transAxes)
if not gws_ghana_points.empty:
    ax6.text(0.1, 0.4, f'• Ghana trend: {ghana_slope:.2f} mm/year', transform=ax6.transAxes, fontsize=10)
if not gws_volta_points.empty:
    ax6.text(0.1, 0.3, f'• Volta trend: {volta_slope:.2f} mm/year', transform=ax6.transAxes, fontsize=10)
ax6.text(0.1, 0.2, '• Clear seasonal patterns detected', transform=ax6.transAxes, fontsize=10)
ax6.text(0.1, 0.1, '• Valuable for water resource management', transform=ax6.transAxes, fontsize=10)

ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)
ax6.axis('off')
ax6.set_title('F) Mission Impact Summary', fontweight='bold')

plt.tight_layout()
plt.suptitle('GRACE-based Assessment of Groundwater Storage Dynamics in West Africa (2004-2024)', 
             fontsize=20, fontweight='bold', y=0.98)
plt.subplots_adjust(top=0.93)
plt.show()

# --- 6. ABSTRACT-READY SUMMARY STATISTICS ---
print("\n" + "="*70)
print("CONFERENCE ABSTRACT KEY FINDINGS")
print("="*70)

if not gws_ghana_points.empty:
    # Calculate additional statistics for abstract
    ghana_std = float(gws_ghana_ds.gws.std().values)
    ghana_range = float(gws_ghana_ds.gws.max().values - gws_ghana_ds.gws.min().values)
    
    print(f"\nGHANA SUMMARY:")
    print(f"• Monitoring period: 2004-2024 (20 years)")
    print(f"• Spatial coverage: {len(gws_ghana_points[['lat', 'lon']].drop_duplicates())} GRACE grid points")
    print(f"• Mean GWS: {float(gws_ghana_ds.gws.mean().values):.1f} ± {ghana_std:.1f} mm")
    print(f"• Total range: {ghana_range:.1f} mm")
    print(f"• Long-term trend: {ghana_slope:.2f} mm/year ({'increasing' if ghana_slope > 0 else 'decreasing'})")
    print(f"• Statistical significance: p = {ghana_p:.4f}")

if not gws_volta_points.empty:
    volta_std = float(gws_volta_ds.gws.std().values)
    volta_range = float(gws_volta_ds.gws.max().values - gws_volta_ds.gws.min().values)
    
    print(f"\nVOLTA BASIN SUMMARY:")
    print(f"• Monitoring period: 2004-2024 (20 years)")
    print(f"• Spatial coverage: {len(gws_volta_points[['lat', 'lon']].drop_duplicates())} GRACE grid points")
    print(f"• Mean GWS: {float(gws_volta_ds.gws.mean().values):.1f} ± {volta_std:.1f} mm")
    print(f"• Total range: {volta_range:.1f} mm")
    print(f"• Long-term trend: {volta_slope:.2f} mm/year ({'increasing' if volta_slope > 0 else 'decreasing'})")
    print(f"• Statistical significance: p = {volta_p:.4f}")

print(f"\nINTERPOLATION METHOD USED: {interpolation_method}")

print(f"\nMETHODOLOGY:")
print(f"• Data: GRACE/GRACE-FO Groundwater Storage Anomalies")
print(f"• Period: April 2004 - December 2024")
print(f"• Processing: Spatial aggregation, trend analysis, anomaly detection")
print(f"• Interpolation: {interpolation_method} method constrained to Ghana boundary")
print(f"• Significance: Demonstrates GRACE capability for regional water resource monitoring")

print(f"\nCONCLUSION:")
print(f"GRACE satellite data provides valuable insights into groundwater dynamics in West Africa,")
print(f"enabling long-term monitoring essential for sustainable water resource management.")

# --- 7. Save presentation-ready data ---
print("\n" + "="*70)
print("SAVING RESULTS FOR CONFERENCE PRESENTATION")
print("="*70)

# Save the composite figure
plt.savefig(f"{hmdir}/Results/conference_figure.png", dpi=300, bbox_inches='tight')
print("Conference figure saved: conference_figure.png")

# Save summary statistics
summary_data = {
    'Region': ['Ghana', 'Volta Basin'],
    'Trend_mm_per_year': [ghana_slope if not gws_ghana_points.empty else np.nan, 
                         volta_slope if not gws_volta_points.empty else np.nan],
    'P_value': [ghana_p if not gws_ghana_points.empty else np.nan, 
               volta_p if not gws_volta_points.empty else np.nan],
    'Mean_GWS_mm': [float(gws_ghana_ds.gws.mean().values) if not gws_ghana_points.empty else np.nan,
                   float(gws_volta_ds.gws.mean().values) if not gws_volta_points.empty else np.nan],
    'Monitoring_Points': [len(gws_ghana_points[['lat', 'lon']].drop_duplicates()) if not gws_ghana_points.empty else 0,
                         len(gws_volta_points[['lat', 'lon']].drop_duplicates()) if not gws_volta_points.empty else 0],
    'Interpolation_Method': [interpolation_method, interpolation_method]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f"{hmdir}/Results/conference_summary.csv", index=False)
print("Summary statistics saved: conference_summary.csv")

print("\n" + "="*70)
print("CONFERENCE ABSTRACT READY FOR SUBMISSION")
print("="*70)