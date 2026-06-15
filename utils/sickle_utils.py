"""
SICKLE utils functions: apply per-band scaling, and build a cloud/shadow mask from QA_PIXEL.
"""
 
import numpy as np
 
 
# --- Landsat C2 L2 scale/offset constants ---------------------------------
# Surface Reflectance (SR_B1..SR_B7): reflectance = DN * 2.75e-5 - 0.2 
# source: https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-reflectance
SR_SCALE, SR_OFFSET = 2.75e-5, -0.2
 
# Surface Temperature (ST_B10) in Kelvin
# source: https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature
ST_SCALE, ST_OFFSET = 0.00341802, 149.0
 
# Other ST_* ancillary products each have their own scale/offset
# (values per USGS Landsat C2 L2 Data Format Control Book)
ST_AUX_SCALES = {
    "ST_ATRAN": (0.0001, 0.0),     # transmittance, unitless
    "ST_CDIST": (0.01,   0.0),     # cloud distance, km
    "ST_DRAD":  (0.001,  0.0),     # downwelling radiance, W/(m^2 sr um)
    "ST_URAD":  (0.001,  0.0),     # upwelling radiance
    "ST_TRAD":  (0.001,  0.0),     # thermal radiance at sensor
    "ST_EMIS":  (0.0001, 0.0),     # emissivity, unitless (0-1)
    "T_EMIS":   (0.0001, 0.0),     # same as ST_EMIS in SICKLE
    "ST_EMSD":  (0.0001, 0.0),     # emissivity std dev
    "ST_QA":    (0.01,   0.0),     # ST uncertainty, Kelvin
}
 
 
# --- Canonical spectrum band name -> Landsat C2 L2 raster key in the .npz ---
# The YAML / spectrum_specs use canonical names (aerosol, blue_2, ...), while the
# .npz archives are keyed by Landsat band names (SR_B1, ST_B10, ...).
BAND_NAME_TO_NPZ = {
    "aerosol":               "SR_B1",   # coastal / aerosol
    "blue_2":                "SR_B2",
    "green_2":               "SR_B3",
    "red_2":                 "SR_B4",
    "near_infrared_3":       "SR_B5",
    "short_wave_infrared_3": "SR_B6",   # SWIR 1
    "short_wave_infrared_4": "SR_B7",   # SWIR 2
    "thermal_infrared_1":    "ST_B10",
}


# --- QA_PIXEL bit definitions (Landsat C2 L2) -----------------------------
# Bit 0: fill, 1: dilated cloud, 2: cirrus, 3: cloud,
# Bit 4: cloud shadow, 5: snow, 6: clear, 7: water
# source: https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands
QA_BITS = {
    "fill":          0,
    "dilated_cloud": 1,
    "cirrus":        2,
    "cloud":         3,
    "cloud_shadow":  4,
    "snow":          5,
    "clear":         6,
    "water":         7,
}

 
def scale_band(name: str, dn: np.ndarray) -> np.ndarray:
    """Convert raw DN to physical units for a given band name."""
    arr = dn.astype(np.float32)
 
    if name.startswith("SR_B"):
        out = arr * SR_SCALE + SR_OFFSET
        # valid SR range is roughly [0, 1]; mask out invalid values (including fill=0)
        
        return out
 
    if name == "ST_B10":
        out = arr * ST_SCALE + ST_OFFSET
        out[(arr == 0)] = np.nan
        return out
 
    if name in ST_AUX_SCALES:
        s, o = ST_AUX_SCALES[name]
        return arr * s + o
 
    # QA bands and SR_QA_AEROSOL: leave as-is, they're bit-packed
    return dn
 
 
def cloud_mask_from_qa(qa_pixel: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask where True = good pixel.
    Masks out fill, cloud, dilated cloud, cirrus, and cloud shadow.
    """
    # cloud_shadow` is intentionally NOT in bad_bits because Landsat C2 QA
    # over-flags dark/water-saturated surfaces as cloud shadow, and we don't want to mask those out in SICKLE.
    bad_bits = ["fill", "dilated_cloud", "cirrus", "cloud"]
    bad = np.zeros_like(qa_pixel, dtype=bool)
    for b in bad_bits:
        bad |= ((qa_pixel >> QA_BITS[b]) & 1).astype(bool)
    return ~bad

