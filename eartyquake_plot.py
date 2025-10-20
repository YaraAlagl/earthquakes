from datetime import date
import requests
import json
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

def get_data():
    """Retrieve the data we will be working with."""
    response = requests.get(
        "http://earthquake.usgs.gov/fdsnws/event/1/query.geojson",
        params={
            'starttime': "2000-01-01",
            "maxlatitude": "58.723",
            "minlatitude": "50.008",
            "maxlongitude": "1.67",
            "minlongitude": "-9.756",
            "minmagnitude": "1",
            "endtime": "2018-10-11",
            "orderby": "time-asc"}
   )
    text = response.text
    # Save the raw text to file (not double-encoded)
    with open("earthquakes-response.json", "w", encoding="utf-8") as f:
        f.write(text)

    # To understand the structure of this text, you may want to save it
    # to a file and open it in VS Code or a browser.
    # See the README file for more information.

    # We need to interpret the text to get values that we can work with.
    # What format is the text in? How can we load the values?
    return json.loads(text)

def get_year(earthquake):
    """Extract the year in which an earthquake happened."""
    timestamp = earthquake['properties']['time']
    # The time is given in a strange-looking but commonly-used format.
    # To understand it, we can look at the documentation of the source data:
    # https://earthquake.usgs.gov/data/comcat/index.php#time
    # Fortunately, Python provides a way of interpreting this timestamp:
    # (Question for discussion: Why do we divide by 1000?)
    year = date.fromtimestamp(timestamp/1000).year
    return year


def get_magnitude(earthquake):
    """Retrive the magnitude of an earthquake item."""
    return earthquake['properties']['mag']


# This is function you may want to create to break down the computations,
# although it is not necessary. You may also change it to something different.
def get_magnitudes_per_year(earthquakes):
    """Retrieve the magnitudes of all the earthquakes in a given year.
    
    Returns a dictionary with years as keys, and lists of magnitudes as values.
    """
    magnitudes_per_year = {}
    for earthquake in earthquakes:
        year = get_year(earthquake)
        magnitude = get_magnitude(earthquake)
        if magnitude is None:
            continue
        if year not in magnitudes_per_year:
            magnitudes_per_year[year] = []
        magnitudes_per_year[year].append(magnitude)
    return magnitudes_per_year


def plot_average_magnitude_per_year(earthquakes):
    """Plot the average magnitude of earthquakes per year."""
    magnitudes_per_year = get_magnitudes_per_year(earthquakes)
    years = []
    average_magnitudes = []
    for year in sorted(magnitudes_per_year.keys()):
        years.append(year)
        magnitudes = magnitudes_per_year[year]
        average_magnitude = sum(magnitudes) / len(magnitudes)
        average_magnitudes.append(average_magnitude)
    
    plt.plot(years, average_magnitudes)
    plt.xlim(2000,2018)
    plt.xlabel("Year")
    plt.ylabel("Average Magnitude")
    plt.title("Average Earthquake Magnitude per Year")
    plt.grid(True)
    plt.show()



def plot_number_per_year(earthquakes):
    """Plot the number of earthquakes per year."""
    counts_per_year = {}
    for earthquake in earthquakes:
        year = get_year(earthquake)
        if year not in counts_per_year:
            counts_per_year[year] = 0
        counts_per_year[year] += 1

    years = sorted(counts_per_year.keys())
    counts = [counts_per_year[year] for year in years]

    plt.plot(years, counts)
    plt.xlim(2000,2018)
    plt.xlabel("Year")
    plt.ylabel("Number of Earthquakes")
    plt.title("Number of Earthquakes per Year")
    plt.grid(True)
    plt.show()
    

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.basemap import Basemap
from datetime import datetime, date

def plot_longitude_latitude_with_histograms(
    earthquakes,
    binsize=1,
    year=None,
    *,
    animate=False,
    freq='Y',                # 'Y' for yearly, 'D' for daily
    interval=500,            # ms between frames
    save_path=None,          # e.g., 'quakes.mp4' or 'quakes.gif'
    date_extractor=None      # callable(eq) -> datetime.date (only needed for daily if your data isn't USGS-style)
):
    """
    Plot earthquake locations with histograms on top and right, or animate over time.

    Parameters:
        earthquakes : list of earthquake dicts
        binsize : float, histogram bin size (degrees)
        year : int or None, only show earthquakes from this year if provided (ignored when animate=True)
        animate : bool, if True create an animation over time
        freq : 'Y' (yearly) or 'D' (daily) when animate=True
        interval : int, milliseconds between animation frames
        save_path : str or None, path to save animation (.mp4 uses ffmpeg, .gif uses Pillow)
        date_extractor : callable or None, returns datetime.date from an earthquake dict (needed for daily if your data lacks USGS 'time')
    """
    # --- Bounds (UK box given) ---
    MAX_LAT = 58.723
    MIN_LAT = 50.008
    MAX_LON = 1.67
    MIN_LON = -9.756

    # Helper: robust date extraction for daily animation
    def _default_date_extractor(eq):
        # Try common USGS format: milliseconds since epoch at properties['time']
        try:
            ms = eq.get('properties', {}).get('time', None)
            if ms is not None:
                return datetime.utcfromtimestamp(ms / 1000.0).date()
        except Exception:
            pass
        # If user provided get_date() elsewhere, you can swap this in.
        raise ValueError(
            "Could not infer a date from earthquake record. "
            "Provide date_extractor=callable(eq)->datetime.date for daily animation."
        )

    if animate and freq.upper() == 'D' and date_extractor is None:
        date_extractor = _default_date_extractor

    # ---- Pre-filter and optionally group data ----
    def in_box(lon, lat):
        return (MIN_LON <= lon <= MAX_LON) and (MIN_LAT <= lat <= MAX_LAT)

    # Static mode: collect once
    if not animate:
        longs, lats, mags = [], [], []
        for eq in earthquakes:
            if year is not None and get_year(eq) != year:
                continue
            lon, lat = eq['geometry']['coordinates'][:2]
            mag = get_magnitude(eq)
            if mag is None or not in_box(lon, lat):
                continue
            longs.append(lon); lats.append(lat); mags.append(mag * 10)

        if not longs:
            print(f"No earthquakes found for year {year}.")
            return

        longs = np.array(longs)
        lats  = np.array(lats)
        mags  = np.array(mags)

        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                              wspace=0.05, hspace=0.05)
        ax_scatter = fig.add_subplot(gs[1, 0])
        ax_histx   = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
        ax_histy   = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

        pad_lon = (MAX_LON - MIN_LON) * 0.03
        pad_lat = (MAX_LAT - MIN_LAT) * 0.03

        m = Basemap(
            projection='cyl',
            llcrnrlon=MIN_LON - pad_lon, urcrnrlon=MAX_LON + pad_lon,
            llcrnrlat=MIN_LAT - pad_lat, urcrnrlat=MAX_LAT + pad_lat,
            resolution='i',
            ax=ax_scatter
        )
        m.drawcoastlines()
        m.drawcountries(linewidth=0.5)
        m.drawmapboundary(fill_color='lightblue')
        m.fillcontinents(color='lightgray', lake_color='lightblue')

        dlon = max(1, int((MAX_LON - MIN_LON) // 2) or 1)
        dlat = max(1, int((MAX_LAT - MIN_LAT) // 2) or 1)
        m.drawparallels(range(int(MIN_LAT), int(MAX_LAT)+1, dlat), labels=[1,0,0,0], linewidth=0.2)
        m.drawmeridians(range(int(MIN_LON), int(MAX_LON)+1, dlon), labels=[0,0,0,1], linewidth=0.2)

        x, y = m(longs, lats)
        m.scatter(x, y, s=mags, c='red', alpha=0.6, edgecolors='k', zorder=5)

        # Histograms
        ax_histx.hist(longs, bins=np.arange(MIN_LON, MAX_LON + binsize, binsize),
                      color='gray', edgecolor='black')
        ax_histx.axis('off')

        ax_histy.hist(lats, bins=np.arange(MIN_LAT, MAX_LAT + binsize, binsize),
                      orientation='horizontal', color='gray', edgecolor='black')
        ax_histy.axis('off')

        ax_scatter.set_xlabel("Longitude")
        ax_scatter.set_ylabel("Latitude")
        title = "Earthquakes with Longitude/Latitude Histograms"
        if year is not None:
            title += f" ({year})"
        ax_scatter.set_title(title)
        plt.show()
        return

    # --- Animation mode ---
    # Build frames keyed by year or day
    buckets = {}  # key -> (longs[], lats[], mags[])
    if freq.upper() == 'Y':
        key_func = lambda eq: get_year(eq)
        key_format = lambda k: str(k)
    elif freq.upper() == 'D':
        key_func = lambda eq: date_extractor(eq)
        key_format = lambda k: k.isoformat() if isinstance(k, date) else str(k)
    else:
        raise ValueError("freq must be 'Y' or 'D'")

    for eq in earthquakes:
        lon, lat = eq['geometry']['coordinates'][:2]
        mag = get_magnitude(eq)
        if mag is None or not in_box(lon, lat):
            continue
        k = key_func(eq)
        if k is None:
            continue
        b = buckets.setdefault(k, ([], [], []))
        b[0].append(lon); b[1].append(lat); b[2].append(mag * 10)

    if not buckets:
        print("No earthquakes found in the specified bounds for animation.")
        return

    # Sort keys chronologically
    if freq.upper() == 'Y':
        keys = sorted(buckets.keys())
    else:  # daily
        keys = sorted(buckets.keys(), key=lambda d: (d.toordinal() if isinstance(d, date) else d))

    # Create figure & axes layout once
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          wspace=0.05, hspace=0.05)
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histx   = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_histy   = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    pad_lon = (MAX_LON - MIN_LON) * 0.03
    pad_lat = (MAX_LAT - MIN_LAT) * 0.03
    m = Basemap(
        projection='cyl',
        llcrnrlon=MIN_LON - pad_lon, urcrnrlon=MAX_LON + pad_lon,
        llcrnrlat=MIN_LAT - pad_lat, urcrnrlat=MAX_LAT + pad_lat,
        resolution='i',
        ax=ax_scatter
    )
    m.drawcoastlines()
    m.drawcountries(linewidth=0.5)
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='lightgray', lake_color='lightblue')

    dlon = max(1, int((MAX_LON - MIN_LON) // 2) or 1)
    dlat = max(1, int((MAX_LAT - MIN_LAT) // 2) or 1)
    m.drawparallels(range(int(MIN_LAT), int(MAX_LAT)+1, dlat), labels=[1,0,0,0], linewidth=0.2)
    m.drawmeridians(range(int(MIN_LON), int(MAX_LON)+1, dlon), labels=[0,0,0,1], linewidth=0.2)

    # Empty artists to update
    scat = m.scatter([], [], s=[], c='red', alpha=0.6, edgecolors='k', zorder=5)

    # Prepare histogram bin edges
    lon_bins = np.arange(MIN_LON, MAX_LON + binsize, binsize)
    lat_bins = np.arange(MIN_LAT, MAX_LAT + binsize, binsize)

    def update(frame_key):
        longs, lats, mags = buckets.get(frame_key, ([], [], []))
        longs = np.asarray(longs); lats = np.asarray(lats); mags = np.asarray(mags)

        # Update scatter
        if longs.size:
            x, y = m(longs, lats)
            offsets = np.column_stack([x, y])
        else:
            offsets = np.empty((0, 2))
        scat.set_offsets(offsets)
        scat.set_sizes(mags if mags.size else np.array([]))

        # Update histograms (clear & redraw for simplicity)
        ax_histx.clear()
        ax_histy.clear()
        ax_histx.hist(longs, bins=lon_bins, color='gray', edgecolor='black')
        ax_histy.hist(lats, bins=lat_bins, orientation='horizontal', color='gray', edgecolor='black')
        ax_histx.axis('off'); ax_histy.axis('off')

        # Axes labels & title
        ax_scatter.set_xlabel("Longitude")
        ax_scatter.set_ylabel("Latitude")
        label = "Year" if freq.upper() == 'Y' else "Day"
        ax_scatter.set_title(f"Earthquakes with Longitude/Latitude Histograms — {label}: {key_format(frame_key)}")

        return scat,

    anim = FuncAnimation(fig, update, frames=keys, interval=interval, blit=False, repeat=False)

    if save_path:
        if save_path.lower().endswith('.gif'):
            from matplotlib.animation import PillowWriter
            anim.save(save_path, writer=PillowWriter(fps=max(1, int(1000/interval))))
        else:
            # mp4: requires ffmpeg installed on your system
            anim.save(save_path, writer='ffmpeg', fps=max(1, int(1000/interval)))
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return anim




    
    

# Get the data we will work with
quakes = get_data()['features']

# Plot the results - this is not perfect since the x axis is shown as real
# numbers rather than integers, which is what we would prefer!
plot_number_per_year(quakes)
plt.clf()  # This clears the figure, so that we don't overlay the two plots
plot_average_magnitude_per_year(quakes)
#plot_longitude_latitude(quakes)
plot_longitude_latitude_with_histograms(quakes, binsize=1)
# Year-by-year animation on screen
plot_longitude_latitude_with_histograms(quakes, animate=True, freq='Y', interval=400)
# Day-by-day animation, USGS-style data 
plot_longitude_latitude_with_histograms(quakes, animate=True, freq='D', interval=200)