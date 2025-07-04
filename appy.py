import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(page_title="GFS Viewer Sumatera", layout="wide")
st.title("📡 Global Forecast System Viewer Wilayah Sumatera")
st.header("Web Hasil Pembelajaran Pengelolaan Informasi Meteorologi")
st.subheader("UAS a.n Mistiari NPT. 14.24.0007 😎")

# Fungsi memuat dataset
@st.cache_data
def load_dataset(run_date, run_hour):
    url = f"https://nomads.ncep.noaa.gov/dods/gfs_0p25_1hr/gfs{run_date}/gfs_0p25_1hr_{run_hour}z"
    return xr.open_dataset(url)

# Sidebar pengaturan
st.sidebar.title("⚙️ Pengaturan")
today = datetime.utcnow()
run_date = st.sidebar.date_input("Tanggal Run GFS (UTC)", today.date())
run_hour = st.sidebar.selectbox("Jam Run GFS (UTC)", ["00", "06", "12", "18"])
forecast_hour = st.sidebar.slider("Jam ke depan (forecast lead)", 0, 240, 0, step=1)
parameter = st.sidebar.selectbox("Parameter Cuaca", [
    "Curah Hujan per jam (pratesfc)",
    "Suhu Permukaan (tmp2m)",
    "Angin Permukaan (ugrd10m & vgrd10m)",
    "Tekanan Permukaan Laut (prmslmsl)"
])

if st.sidebar.button("🔎 Tampilkan Visualisasi"):
    try:
        ds = load_dataset(run_date.strftime("%Y%m%d"), run_hour)
        st.success("✅ Dataset berhasil dimuat.")
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {e}")
        st.stop()

    # Inisialisasi
    is_vector = False
    is_contour = False

    # Pemilihan variabel
    try:
        if "pratesfc" in parameter:
            var = ds["pratesfc"][forecast_hour] * 3600
            label = "Curah Hujan (mm/jam)"
            cmap = "Blues"
        elif "tmp2m" in parameter:
            var = ds["tmp2m"][forecast_hour] - 273.15
            label = "Suhu (°C)"
            cmap = "coolwarm"
        elif "ugrd10m" in parameter:
            u = ds["ugrd10m"][forecast_hour]
            v = ds["vgrd10m"][forecast_hour]
            var = np.sqrt(u**2 + v**2) * 1.94384  # Kecepatan angin (knot)
            label = "Kecepatan Angin (knot)"
            cmap = "YlGnBu"
            is_vector = True
        elif "prmsl" in parameter:
            var = ds["prmslmsl"][forecast_hour] / 100  # dari Pa ke hPa
            label = "Tekanan Permukaan Laut (hPa)"
            cmap = "cool"
            is_contour = True
        else:
            st.error("❌ Parameter tidak tersedia.")
            st.stop()
    except KeyError:
        st.error("❌ Parameter tidak ditemukan dalam dataset.")
        st.write("Data tersedia:", list(ds.data_vars))
        st.stop()

    # Potong wilayah Sumatera
    try:
        lat_slice = slice(5, 0) if var.lat[0] > var.lat[-1] else slice(0, 5)
        var = var.sel(lat=lat_slice, lon=slice(95, 106)).transpose("lat", "lon")
        if is_vector:
            u = u.sel(lat=lat_slice, lon=slice(95, 106)).transpose("lat", "lon")
            v = v.sel(lat=lat_slice, lon=slice(95, 106)).transpose("lat", "lon")
    except Exception as e:
        st.error(f"❌ Gagal slicing lat/lon: {e}")
        st.stop()

    # Cek apakah data valid
    if np.isnan(var.to_numpy()).all():
        st.error("❌ Semua data bernilai NaN. Coba ubah jam atau parameter.")
        st.stop()

    # Waktu validasi
    valid_time = ds.time[forecast_hour].values
    valid_dt = pd.to_datetime(str(valid_time))
    valid_str = valid_dt.strftime("%HUTC %a %d %b %Y")
    tstr = f"t+{forecast_hour:03d}"

    # Plotting
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([95, 106, 0, 5], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    # Judul rapi
    plt.suptitle(f"{label} - GFS {tstr}", fontsize=12, fontweight="bold")
    ax.set_title(f"Valid: {valid_str}", fontsize=9, loc="center", pad=10)

    # Plot isi data
    try:
        if is_contour:
            cs = var.plot.contour(ax=ax, transform=ccrs.PlateCarree(),
                                  levels=15, colors='black', linewidths=0.8,
                                  add_colorbar=False)
            ax.clabel(cs, inline=1, fontsize=8)
        else:
            pc = var.plot.pcolormesh(ax=ax, cmap=cmap, transform=ccrs.PlateCarree(),
                                     cbar_kwargs={"label": label}, add_colorbar=True)

            # Plot angin vektor
            if is_vector:
                ax.quiver(u.lon.values[::5], u.lat.values[::5],
                          u.values[::5, ::5], v.values[::5, ::5],
                          transform=ccrs.PlateCarree(), scale=700, width=0.002, color='black')
    except Exception as e:
        st.error(f"❌ Gagal membuat plot: {e}")
        st.stop()

    st.pyplot(fig)
