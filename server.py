import io
import logging
import os
import sys
import threading
import time
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory

import anvil.media
import anvil.server

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import requests
import sounderpy as spy


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("sounderpy-webapp-server")


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

UPLINK_KEY = os.environ.get("ANVIL_API_KEY")

if not UPLINK_KEY:
    LOGGER.critical("ANVIL_API_KEY environment variable not found.")
    sys.exit(1)

try:
    SOUNDERPY_VERSION = version("sounderpy")
except PackageNotFoundError:
    SOUNDERPY_VERSION = "unknown"

LOGGER.info("Starting backend with SounderPy %s", SOUNDERPY_VERSION)
anvil.server.connect(UPLINK_KEY)


# -----------------------------------------------------------------------------
# LOCKS / CACHES
# -----------------------------------------------------------------------------

# Retrieval functions are not locked. Only Matplotlib plotting is serialized.
PLOT_LOCK = threading.Lock()

LATEST_RUN_CACHE_LOCK = threading.Lock()
LATEST_RUN_CACHE = {"expires": 0.0, "data": None}
LATEST_RUN_CACHE_SECONDS = 300

AIRPORTS_CSV_URL = ("https://raw.githubusercontent.com/kylejgillett/sounderpy/main/src/AIRPORTS.csv")


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_airports_df():
    """Load the airport metadata once per server process."""
    response = requests.get(AIRPORTS_CSV_URL, timeout=15)
    response.raise_for_status()
    return pd.read_csv(
        io.StringIO(response.text),
        skiprows=7,
        skipinitialspace=True,
    )


def _airport_description(iata):
    airports = _get_airports_df()
    code = str(iata).strip().upper()

    iata_col = airports["IATA"].astype(str).str.strip().str.upper()
    matches = airports.loc[iata_col == code]

    if matches.empty:
        return "Unknown Airport"

    row = matches.iloc[0]
    return f"{row['Name']}, {row['City']}"


def _make_file_media(clean_data, file_type):
    """Export SounderPy data using a request-specific temporary file."""
    file_type = str(file_type).lower()

    suffixes = {
        "csv": ".csv",
        "cm1": ".snd",
        "sharppy": ".txt",
    }
    content_types = {
        "csv": "text/csv",
        "cm1": "text/plain",
        "sharppy": "text/plain",
    }

    if file_type not in suffixes:
        raise ValueError(
            f"Unsupported export type '{file_type}'. "
            "Expected csv, cm1, or sharppy."
        )

    with TemporaryDirectory(prefix="sounderpy_export_") as tmpdir:
        output = Path(tmpdir) / f"sounderpy_data{suffixes[file_type]}"

        spy.to_file(
            file_type,
            clean_data,
            filename=str(output),
        )

        return anvil.media.from_file(
            str(output),
            content_types[file_type],
            name=output.name,
        )


def _make_plot_media(
    clean_data,
    *,
    hodo,
    color_blind,
    dark_mode,
    storm_motion,
    modify_sfc,
    special_parcels,
    map_zoom,
    radar,
    radar_time,
    hodo_boundary,
):
    """Render a sounding or hodograph to a unique temporary PNG."""
    with TemporaryDirectory(prefix="sounderpy_plot_") as tmpdir:
        output = Path(tmpdir) / (
            "sounderpy_hodograph.png" if hodo else "sounderpy_sounding.png"
        )

        with PLOT_LOCK:
            try:
                if hodo:
                    spy.build_hodograph(
                        clean_data,
                        dark_mode=dark_mode,
                        storm_motion=storm_motion,
                        modify_sfc=modify_sfc,
                        radar=radar,
                        radar_time=radar_time,
                        map_zoom=map_zoom,
                        hodo_boundary=hodo_boundary,
                        save=True,
                        filename=str(output),
                    )
                else:
                    spy.build_sounding(
                        clean_data,
                        color_blind=color_blind,
                        dark_mode=dark_mode,
                        storm_motion=storm_motion,
                        special_parcels=special_parcels,
                        radar=radar,
                        radar_time=radar_time,
                        map_zoom=map_zoom,
                        modify_sfc=modify_sfc,
                        hodo_boundary=hodo_boundary,
                        save=True,
                        filename=str(output),
                    )

                return anvil.media.from_file(
                    str(output),
                    "image/png",
                    name=output.name,
                )

            finally:
                plt.close("all")


def _finish_request(
    clean_data,
    *,
    label_txt,
    color_blind,
    dark_mode,
    hodo,
    storm_motion,
    modify_sfc,
    special_parcels,
    map_zoom,
    radar,
    radar_time,
    hodo_boundary,
    file,
    file_type,
):
    """Shared render/export path for all sounding sources."""
    if file:
        return _make_file_media(clean_data, file_type)

    image = _make_plot_media(
        clean_data,
        hodo=hodo,
        color_blind=color_blind,
        dark_mode=dark_mode,
        storm_motion=storm_motion,
        modify_sfc=modify_sfc,
        special_parcels=special_parcels,
        map_zoom=map_zoom,
        radar=radar,
        radar_time=radar_time,
        hodo_boundary=hodo_boundary,
    )

    return image, label_txt


def _site_label(clean_data, descriptor):
    info = clean_data["site_info"]
    valid = info["valid-time"]
    return (
        f"{valid[3]}Z {descriptor} "
        f"{info['site-id']}, {info['site-name']} at "
        f"{valid[1]}-{valid[2]}-{valid[0]}"
    )


# -----------------------------------------------------------------------------
# BACKEND STATUS
# -----------------------------------------------------------------------------

@anvil.server.callable
def get_backend_info():
    return {
        "online": True,
        "sounderpy_version": SOUNDERPY_VERSION,
    }


# -----------------------------------------------------------------------------
# RAOB
# -----------------------------------------------------------------------------

@anvil.server.callable
def get_raob_sounding(
    site_id, year, month, day, hour,
    color_blind, dark_mode, hodo, storm_motion, modify_sfc,
    special_parcels, map_zoom, radar, radar_time, hodo_boundary,
    file, file_type,
):
    clean_data = spy.get_obs_data(
        str(site_id), str(year), str(month), str(day), str(hour),
        hush=True,
    )

    return _finish_request(
        clean_data,
        label_txt=_site_label(clean_data, "RAOB for"),
        color_blind=color_blind,
        dark_mode=dark_mode,
        hodo=hodo,
        storm_motion=storm_motion,
        modify_sfc=modify_sfc,
        special_parcels=special_parcels,
        map_zoom=map_zoom,
        radar=radar,
        radar_time=radar_time,
        hodo_boundary=hodo_boundary,
        file=file,
        file_type=file_type,
    )


# -----------------------------------------------------------------------------
# ACARS LISTS
# -----------------------------------------------------------------------------

@anvil.server.callable
def get_acars_all_profile_list(year, month, day, hour):
    try:
        profiles = spy.acars_data(
            str(year), str(month), str(day), str(hour)
        ).list_profiles()
    except Exception:
        LOGGER.exception(
            "Unable to retrieve ACARS profiles for %s-%s-%s %sZ",
            year, month, day, hour,
        )
        return ["No profiles found for this date or time"]

    output = []

    for profile in profiles:
        airport = profile[0:3]
        try:
            airport_info = _airport_description(airport)
        except Exception:
            LOGGER.exception("Unable to resolve airport %s", airport)
            airport_info = "Unknown Airport"

        output.append(f"{profile} | {airport_info}")

    return output


@anvil.server.callable
def get_acars_airport_profile_list(year, month, day, airport):
    profiles_list = []
    dates_list = []
    airport = str(airport).upper()

    for hour in range(24):
        hour_str = f"{hour:02d}"

        try:
            acars_list = spy.acars_data(
                str(year), str(month), str(day), hour_str
            ).list_profiles()

            matches = [item for item in acars_list if airport in item]

            if matches:
                profiles_list.extend(matches)
                dates_list.extend(
                    [[str(year), str(month), str(day), hour_str]
                     for _ in matches]
                )
        except Exception:
            LOGGER.debug(
                "No usable ACARS result for %s-%s-%s %sZ",
                year, month, day, hour_str,
                exc_info=True,
            )

    if profiles_list:
        profile_ids = profiles_list
    else:
        profile_ids = ["No profiles found for given date & airport"]

    return profile_ids, profiles_list, dates_list


# -----------------------------------------------------------------------------
# ACARS SOUNDING
# -----------------------------------------------------------------------------

@anvil.server.callable
def get_acars_sounding(
    profile_id, year, month, day, hour,
    color_blind, dark_mode, hodo, storm_motion, modify_sfc,
    special_parcels, map_zoom, radar, radar_time, hodo_boundary,
    file, file_type,
):
    clean_data = spy.acars_data(
        str(year), str(month), str(day), str(hour)
    ).get_profile(profile_id)

    return _finish_request(
        clean_data,
        label_txt=_site_label(clean_data, "flight from"),
        color_blind=color_blind,
        dark_mode=dark_mode,
        hodo=hodo,
        storm_motion=storm_motion,
        modify_sfc=modify_sfc,
        special_parcels=special_parcels,
        map_zoom=map_zoom,
        radar=radar,
        radar_time=radar_time,
        hodo_boundary=hodo_boundary,
        file=file,
        file_type=file_type,
    )


# -----------------------------------------------------------------------------
# BUFKIT — SPECIFIC RUN
# -----------------------------------------------------------------------------

@anvil.server.callable
def get_bufkit_sounding(
    model, bufkit_site, fcst_hr,
    run_year, run_month, run_day, run_hour,
    color_blind, dark_mode, hodo, storm_motion, modify_sfc,
    special_parcels, map_zoom, radar, radar_time, hodo_boundary,
    file, file_type,
):
    clean_data = spy.get_bufkit_data(
        str(model),
        str(bufkit_site),
        int(fcst_hr),
        str(run_year),
        str(run_month),
        str(run_day),
        str(run_hour),
        hush=True,
    )

    return _finish_request(
        clean_data,
        label_txt=_site_label(clean_data, "forecast for"),
        color_blind=color_blind,
        dark_mode=dark_mode,
        hodo=hodo,
        storm_motion=storm_motion,
        modify_sfc=modify_sfc,
        special_parcels=special_parcels,
        map_zoom=map_zoom,
        radar=radar,
        radar_time=radar_time,
        hodo_boundary=hodo_boundary,
        file=file,
        file_type=file_type,
    )


# -----------------------------------------------------------------------------
# BUFKIT — LATEST RUN
# -----------------------------------------------------------------------------

@anvil.server.callable
def get_latest_bufkit_sounding(
    model, bufkit_site, fcst_hr,
    color_blind, dark_mode, hodo, storm_motion, modify_sfc,
    special_parcels, map_zoom, radar, radar_time, hodo_boundary,
    file, file_type,
):
    clean_data = spy.get_bufkit_data(
        str(model), str(bufkit_site), int(fcst_hr),
        hush=True,
    )

    return _finish_request(
        clean_data,
        label_txt=_site_label(clean_data, "forecast for"),
        color_blind=color_blind,
        dark_mode=dark_mode,
        hodo=hodo,
        storm_motion=storm_motion,
        modify_sfc=modify_sfc,
        special_parcels=special_parcels,
        map_zoom=map_zoom,
        radar=radar,
        radar_time=radar_time,
        hodo_boundary=hodo_boundary,
        file=file,
        file_type=file_type,
    )


# -----------------------------------------------------------------------------
# RAP/RUC REANALYSIS
# -----------------------------------------------------------------------------

@anvil.server.callable
def get_reanl_sounding(
    latlon, year, month, day, hour,
    color_blind, dark_mode, hodo, storm_motion, modify_sfc,
    special_parcels, map_zoom, radar, radar_time, hodo_boundary,
    file, file_type,
):
    clean_data = spy.get_model_data(
        "rap-ruc",
        latlon,
        str(year),
        str(month),
        str(day),
        str(hour),
        hush=True,
    )

    info = clean_data["site_info"]
    valid = info["valid-time"]
    label_txt = (
        f"{valid[3]}Z reanalysis for {info['site-latlon']} at "
        f"{valid[1]}-{valid[2]}-{valid[0]}"
    )

    return _finish_request(
        clean_data,
        label_txt=label_txt,
        color_blind=color_blind,
        dark_mode=dark_mode,
        hodo=hodo,
        storm_motion=storm_motion,
        modify_sfc=modify_sfc,
        special_parcels=special_parcels,
        map_zoom=map_zoom,
        radar=radar,
        radar_time=radar_time,
        hodo_boundary=hodo_boundary,
        file=file,
        file_type=file_type,
    )


# -----------------------------------------------------------------------------
# LATEST BUFKIT MODEL RUNS
# -----------------------------------------------------------------------------

def _fetch_latest_runs():
    text_list = []
    station = "KMOP"

    for model in ["gfs", "nam", "namnest", "rap", "hrrr", "sref", "hiresw"]:
        model3 = "gfs3" if model == "gfs" else model

        data_url = (
            f"http://www.meteo.psu.edu/bufkit/data/"
            f"{model.upper()}/{model3}_{station.lower()}.buf"
        )

        try:
            response = requests.get(data_url, timeout=10)
            response.raise_for_status()
            lines = response.text.splitlines()

            if len(lines) < 5 or "TIME" not in lines[4]:
                raise ValueError(
                    "BUFKIT header does not contain expected TIME field"
                )

            run_line = lines[4]
            start = run_line.index("TIME") + 7
            run_time = run_line[start:start + 11]

            text_list.append(
                f"{model.upper()} | Latest run: "
                f"{run_time[2:4]}/{run_time[4:6]} "
                f"{run_time[7:9]}Z"
            )

        except Exception:
            LOGGER.exception(
                "Unable to determine latest %s BUFKIT run",
                model,
            )
            text_list.append(
                f"{model.upper()} | Latest run unavailable"
            )

    return text_list


@anvil.server.callable
def get_latest_run():
    now = time.time()

    with LATEST_RUN_CACHE_LOCK:
        cached = LATEST_RUN_CACHE["data"]

        if cached is not None and now < LATEST_RUN_CACHE["expires"]:
            return list(cached)

        latest = _fetch_latest_runs()
        LATEST_RUN_CACHE["data"] = latest
        LATEST_RUN_CACHE["expires"] = now + LATEST_RUN_CACHE_SECONDS

        return list(latest)


# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------

LOGGER.info("Anvil Uplink connected; waiting for requests.")
anvil.server.wait_forever()
