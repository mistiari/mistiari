import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Prakiraan Cuaca Wilayah Indonesia", layout="wide")

st.title("📡 Global Forecast System Viewer Wilayah Sumatera (Realtime via NOMADS)")
st.header("Web Hasil Pembelajaran Pengelolaan Informasi Meteorologi")
st.subheader("UAS a.n Mistiari NPT. 14.24.0007 😎")

@st.cache_data
def load_dataset(run_date, run_hour):
    base_url = f"https://nomads.ncep.noaa.gov/dods/gfs_0p25_1hr/gfs{run_date}/gfs_0p25_1hr_{run_hour}z"
    ds = xr.open_dataset(base_url)
    return ds

st.sidebar.title("⚙️ Pengaturan")

# Input pengguna
today = datetime.utcnow()
run_date = st.sidebar.date_input("Tanggal Run GFS (UTC)", today.date())
run_hour = st.sidebar.selectbox("Jam Run GFS (UTC)", ["00", "06", "12", "18"])
forecast_hour = st.sidebar.slider("Jam ke depan", 0, 240, 0, step=1)
parameter = st.sidebar.selectbox("Parameter", [
    "Curah Hujan per jam (pratesfc)",
    "Suhu Permukaan (tmp2m)",
    "Angin Permukaan (ugrd10m & vgrd10m)",
    "Tekanan Permukaan Laut (prmslmsl)"
])

if st.sidebar.button("🔎 Tampilkan Visualisasi"):
    try:
        ds = load_dataset(run_date.strftime("%Y%m%d"), run_hour)
        st.success("Dataset berhasil dimuat.")
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.stop()

    is_contour = False
    is_vector = False

    # Ambil parameter sesuai pilihan
    if "pratesfc" in parameter:
        var = ds["pratesfc"][forecast_hour, :, :] * 3600
        label = "Curah Hujan (mm/jam)"
        cmap = "Blues"
    elif "tmp2m" in parameter:
        var = ds["tmp2m"][forecast_hour, :, :] - 273.15
        label = "Suhu (°C)"
        cmap = "coolwarm"
    elif "ugrd10m" in parameter:
        u = ds["ugrd10m"][forecast_hour, :, :]
        v = ds["vgrd10m"][forecast_hour, :, :]
        speed = (u**2 + v**2)**0.5 * 1.94384  # konversi ke knot
        var = speed
        label = "Kecepatan Angin (knot)"
        cmap = plt.cm.get_cmap("RdYlGn_r", 10)
        is_vector = True
    elif "prmsl" in parameter:
        var = ds["prmslmsl"][forecast_hour, :, :] / 100
        label = "Tekanan Permukaan Laut (hPa)"
        cmap = "cool"
        is_contour = True
    else:
        st.warning("Parameter tidak dikenali.")
        st.stop()

    # Potong wilayah Sumatera dan urutkan (lat, lon)
    var = var.sel(lat=slice(5, 0), lon=slice(95, 106)).transpose('lat', 'lon')
    if is_vector:
        u = u.sel(lat=slice(5, 0), lon=slice(95, 106)).transpose('lat', 'lon')
        v = v.sel(lat=slice(5, 0), lon=slice(95, 106)).transpose('lat', 'lon')

    # Konversi ke numpy
    C = var.to_numpy()
    lon2d, lat2d = np.meshgrid(var.lon.to_numpy(), var.lat.to_numpy())

    # Buat peta
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([95, 106, 0, 5], crs=ccrs.PlateCarree())

    valid_time = ds.time[forecast_hour].values
    valid_dt = pd.to_datetime(str(valid_time))
    valid_str = valid_dt.strftime("%HUTC %a %d %b %Y")
    tstr = f"t+{forecast_hour:03d}"

    ax.set_title(f"{label} Valid {valid_str}", loc="left", fontsize=10, fontweight="bold")
    ax.set_title(f"GFS {tstr}", loc="right", fontsize=10, fontweight="bold")

    if is_contour:
        cs = ax.contour(lon2d, lat2d, C,
                        levels=15, colors='black',
                        linewidths=0.8, transform=ccrs.PlateCarree())
        ax.clabel(cs, fmt="%d", colors='black', fontsize=8)
    else:
        im = ax.pcolormesh(lon2d, lat2d, C,
                           cmap=cmap, vmin=0, vmax=50,
                           transform=ccrs.PlateCarree(),
                           shading="auto")
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label(label)

        if is_vector:
            ax.quiver(lon2d[::5, ::5], lat2d[::5, ::5],
                      u.to_numpy()[::5, ::5], v.to_numpy()[::5, ::5],
                      transform=ccrs.PlateCarree(),
                      scale=700, width=0.002, color='black')

    ax.coastlines(resolution='10m', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    st.pyplot(fig)
