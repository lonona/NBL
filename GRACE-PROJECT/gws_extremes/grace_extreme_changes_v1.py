import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.validation import explain_validity
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.ensemble import IsolationForest

# ===============================================================
# SAVE FIGURE UTILITY (saves current figure)
# ===============================================================
def save_figure(prefix, fig=None, dpi=300, results_dir="IEEE_figures"):
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    if fig is None:
        fig = plt.gcf()

    for ext in ["pdf", "png"]:
        out = results_dir / f"{prefix}.{ext}"
        fig.savefig(
            out,
            bbox_inches='tight',
            pad_inches=0.1,
            dpi=dpi if ext == "png" else None,
            transparent=True
        )
    print(f"[SAVED] {prefix}.pdf/.png")

# ===============================================================
# ANOMALY DETECTION (Isolation Forest per pixel)
# ===============================================================
def detect_anomalies_isolation_forest(ds, contamination=0.05, random_state=42):
    """
    Detect monthly extreme events using per-pixel Isolation Forest.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Must have dimensions ('time', 'lat', 'lon') and variable 'gws'.
    contamination : float, default=0.05
        Expected proportion of anomalies.
    random_state : int, default=42
    
    Returns:
    --------
    anomaly_scores : xarray.DataArray
        Same shape as ds['gws']; higher score = more anomalous.
    ranking : pd.DataFrame
        Months ranked by basin‑average anomaly score.
    """
    # Stack spatial dimensions into a single 'pixel' dimension
    stacked = ds['gws'].stack(pixel=('lat', 'lon'))
    # Remove pixels that are all NaN (if any)
    not_all_nan = ~stacked.isnull().all(dim='time')
    stacked_clean = stacked.where(not_all_nan, drop=True)
    
    n_pixels = stacked_clean.pixel.size
    n_times = stacked_clean.time.size
    
    # Array to store anomaly scores (negative of Isolation Forest score_samples)
    anomaly_scores_pixel = np.full((n_times, n_pixels), np.nan)
    
    for i in range(n_pixels):
        series = stacked_clean.isel(pixel=i).values
        if np.isnan(series).any():
            continue  # should not happen after drop, but safety
        X = series.reshape(-1, 1)
        model = IsolationForest(contamination=contamination, random_state=random_state)
        model.fit(X)
        # score_samples: larger = more normal -> invert for anomaly intensity
        scores = model.score_samples(X)
        anomaly_scores_pixel[:, i] = -scores   # now larger = more anomalous
    
    # Convert back to xarray with original lat/lon grid (NaN for dropped pixels)
    anomaly_da = stacked.copy()
    anomaly_da.values[:] = np.nan
    anomaly_da.loc[dict(pixel=stacked_clean.pixel)] = xr.DataArray(
        anomaly_scores_pixel, 
        dims=('time', 'pixel'),
        coords={'time': stacked.time, 'pixel': stacked_clean.pixel}
    )
    anomaly_da = anomaly_da.unstack('pixel')
    
    # Ranking: mean anomaly score over all pixels for each month
    monthly_mean = anomaly_da.mean(dim=['lat', 'lon'], skipna=True)
    ranking = monthly_mean.to_dataframe(name='mean_anomaly_score')
    ranking = ranking.sort_values('mean_anomaly_score', ascending=False)
    ranking['rank'] = range(1, len(ranking)+1)
    
    return anomaly_da, ranking

def plot_anomaly_map(anomaly_da, month_idx, title, boundary_xy, boundary_color, filename_prefix, vmin=None, vmax=None):
    """
    Scatter plot of anomaly scores for a specific month.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    month_data = anomaly_da.isel(time=month_idx).to_dataframe(name='anomaly_score').reset_index()
    
    if vmin is None:
        vmin = float(anomaly_da.min().values)
    if vmax is None:
        vmax = float(anomaly_da.max().values)
    
    sc = ax.scatter(month_data['lon'], month_data['lat'],
                    c=month_data['anomaly_score'], cmap='jet', #Reds
                    vmin=vmin, vmax=vmax, s=30, edgecolor='k', linewidth=0.3)
    
    ax.plot(boundary_xy['lon'], boundary_xy['lat'], boundary_color, linewidth=2, label='Basin')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower left')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    cbar = plt.colorbar(sc, ax=ax, extend='both')
    cbar.set_label('Anomaly intensity (higher = more extreme)', fontsize=10)
    
    plt.tight_layout()
    save_figure(filename_prefix, fig=fig, dpi=300)
    plt.show()

# ===============================================================
# MAIN PIPELINE (your original code + anomaly detection)
# ===============================================================

# --- 1. Define directories and load data ---
hmdir = "/Users/theonarh/Desktop/TopAgenda/cnference/cr/anly_downscal/"

ghana_xy = pd.read_csv(f"{hmdir}/Data/Basins/ghana.csv")
volta_xy = pd.read_csv(f"{hmdir}/Data/Basins/volta_basin.csv")

gws = xr.open_dataset(hmdir + "Data/gws.nc")

# --- 2. Time selection ---
gws = gws.sel(time=slice('2004-01-01', '2024-12-31'))

# --- 3. Create polygons from boundary points (with validation) ---
def make_valid_polygon(xy_df, x_col='lon', y_col='lat'):
    coords = list(zip(xy_df[x_col], xy_df[y_col]))
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    poly = Polygon(coords)
    if not poly.is_valid:
        print(f"Invalid polygon detected. Validity explanation: {explain_validity(poly)}")
        poly = poly.buffer(0)
    return poly

ghana_polygon = make_valid_polygon(ghana_xy)
volta_polygon = make_valid_polygon(volta_xy)

ghana_gdf = gpd.GeoDataFrame(geometry=[ghana_polygon], crs="EPSG:4326")
volta_gdf = gpd.GeoDataFrame(geometry=[volta_polygon], crs="EPSG:4326")

# --- 4. Extract GWS data within boundaries ---
gws_df = gws['gws'].to_dataframe().reset_index()
gws_points = gpd.GeoDataFrame(
    gws_df, 
    geometry=gpd.points_from_xy(gws_df.lon, gws_df.lat),
    crs="EPSG:4326"
)

gws_ghana_points = gpd.sjoin(gws_points, ghana_gdf, how='inner', predicate='within')
gws_volta_points = gpd.sjoin(gws_points, volta_gdf, how='inner', predicate='within')

# --- 5. Convert back to xarray Datasets ---
if not gws_ghana_points.empty:
    gws_ghana_points = gws_ghana_points.drop(columns='index_right')
    gws_ghana_ds = gws_ghana_points.set_index(['time', 'lat', 'lon']).to_xarray()
else:
    gws_ghana_ds = xr.Dataset({'gws': (['time', 'lat', 'lon'], np.empty((0, 0, 0)))})

if not gws_volta_points.empty:
    gws_volta_points = gws_volta_points.drop(columns='index_right')
    gws_volta_ds = gws_volta_points.set_index(['time', 'lat', 'lon']).to_xarray()
else:
    gws_volta_ds = xr.Dataset({'gws': (['time', 'lat', 'lon'], np.empty((0, 0, 0)))})

# --- 6. Extract 2020 data for original monthly plots (keep unchanged) ---
def extract_year(ds, year):
    if 'time' in ds.dims and ds.dims['time'] > 0:
        try:
            return ds.sel(time=str(year))
        except KeyError:
            return xr.Dataset()
    return xr.Dataset()

gws_ghana_2020 = extract_year(gws_ghana_ds, 2020)
gws_volta_2020 = extract_year(gws_volta_ds, 2020)

has_ghana_data = ('time' in gws_ghana_2020.dims and gws_ghana_2020.dims['time'] == 12)
has_volta_data = ('time' in gws_volta_2020.dims and gws_volta_2020.dims['time'] == 12)

# --- 7. Original monthly plotting function (unchanged) ---
def plot_monthly_gws(ds, boundary_xy, title_prefix, boundary_color, filename_prefix, vmin=None, vmax=None):
    fig, axes = plt.subplots(4, 3, figsize=(14, 16), dpi=300)
    axes = axes.flatten()
    
    if vmin is None:
        vmin = float(ds['gws'].min().values)
    if vmax is None:
        vmax = float(ds['gws'].max().values)
    
    for i, month in enumerate(range(1, 13)):
        ax = axes[i]
        month_data = ds.isel(time=i)
        df = month_data.to_dataframe().reset_index()
        sc = ax.scatter(df['lon'], df['lat'], 
                        c=df['gws'], cmap='RdBu_r', 
                        vmin=vmin, vmax=vmax, s=30)
        ax.plot(boundary_xy['lon'], boundary_xy['lat'], boundary_color, linewidth=2, label=title_prefix)
        ax.set_title(f'2020-{month:02d}', fontweight='bold')
        ax.legend(loc='lower left')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    fig.supxlabel('Longitude', fontsize=12)
    fig.supylabel('Latitude', fontsize=12)
    fig.suptitle(f'Monthly Groundwater Storage (GWS) - {title_prefix} - 2020', fontsize=16, fontweight='bold')
    
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
    cbar = fig.colorbar(sc, cax=cax, orientation='vertical', extend='both')
    cbar.set_label('GWS (mm)', fontsize=11)
    
    plt.subplots_adjust(left=0.07, right=0.90, top=0.92, bottom=0.08, wspace=0.25, hspace=0.3)
    save_figure(filename_prefix, fig=fig, dpi=300)
    plt.show()

# --- 8. Plot original 2020 maps ---
if has_ghana_data:
    plot_monthly_gws(gws_ghana_2020, ghana_xy, 'Ghana', 'k', 'Ghana_GWS_2020_monthly')
if has_volta_data:
    plot_monthly_gws(gws_volta_2020, volta_xy, 'Volta Basin', 'r', 'Volta_GWS_2020_monthly')

# ===============================================================
# NEW: ANOMALY DETECTION ON FULL TIME SERIES (2004-2024)
# ===============================================================

def run_anomaly_detection(ds, region_name, boundary_xy, boundary_color):
    """
    Run Isolation Forest anomaly detection and produce:
    - Ranking of most extreme months
    - Top 3 anomaly maps
    - Marked time series
    """
    if ds.dims['time'] == 0:
        print(f"No data for {region_name}, skipping anomaly detection.")
        return
    
    print(f"\n=== Running anomaly detection for {region_name} ===")
    anomaly_da, ranking = detect_anomalies_isolation_forest(ds, contamination=0.05)
    
    # Save ranking to CSV
    ranking_file = f"{region_name}_anomaly_ranking.csv"
    ranking.to_csv(ranking_file)
    print(f"Saved ranking to {ranking_file}")
    print("Top 10 most extreme months:")
    print(ranking.head(10))
    
    # Plot top 3 anomalous months
    top_months = ranking.head(3).index
    for rank, month in enumerate(top_months, 1):
        # Find the time index (position) for this month
        time_idx = np.where(ds.time.values == month)[0][0]
        title = f"{region_name} - Rank {rank} anomaly\n{month.strftime('%Y-%b')}"
        filename = f"{region_name}_anomaly_rank{rank}_{month.strftime('%Y%m')}"
        plot_anomaly_map(anomaly_da, time_idx, title, boundary_xy, boundary_color, filename,
                         vmin=None, vmax=None)
    
    # Mark anomalies on the time series plot
    try:
        timeseries = ds.mean(dim=['lat', 'lon'], skipna=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timeseries.time, timeseries.gws, 'b-', linewidth=2, label='Mean GWS')
        
        # Highlight top 10 anomaly months (red dots)
        top10_months = ranking.head(10).index
        top10_values = timeseries.sel(time=top10_months, method='nearest')['gws']
        ax.scatter(top10_values.time, top10_values.values, color='red', s=80, zorder=5,
                   label='Top 10 extreme events')
        
        ax.set_title(f'Mean GWS Time Series with Extreme Events - {region_name}', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('GWS (mm)')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        save_figure(f"{region_name}_GWS_time_series_with_anomalies", fig=fig, dpi=300)
        plt.show()
    except Exception as e:
        print(f"Could not create marked time series for {region_name}: {e}")

# Run anomaly detection for both regions
run_anomaly_detection(gws_ghana_ds, "Ghana", ghana_xy, 'k')
run_anomaly_detection(gws_volta_ds, "Volta_Basin", volta_xy, 'r')

print("\n=== All anomaly detection completed ===")