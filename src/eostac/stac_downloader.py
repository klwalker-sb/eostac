import contextlib
from pathlib import Path
import re
import enum
import math
import typing as T
from abc import ABC
import rasterio as rio
import xml.etree.ElementTree as ET

from .errors import CorruptedError, DownloadError
from .stac_utils import (
    LandsatBands,
    Sentinel2Bands,
    Sentinel2Bands_v0,
    SensorPaths,
    pickle_scene_info,
    rechunk,
    NODATA,
    NODOWNLOAD
)
from .utils import TqdmParallel, set_color_logger

import geowombat as gw
from geowombat.core.stac import open_stac
from geowombat.radiometry import QABits

import pandas as pd
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
import xarray as xr
import pyproj
from shapely.geometry import CAP_STYLE, JOIN_STYLE
# from s2cloudless import S2PixelCloudDetector
import attr
from tqdm.auto import tqdm
from joblib import delayed, parallel_backend
import time
from wrapt_timeout_decorator import timeout
from datetime import datetime
import os
import dask

logger = set_color_logger(__name__)

import geowombat.core.stac as _gwstac

_original_download_worker = _gwstac._download_worker

def _safe_download_worker(item, extra, out_path):
    try:
        return _original_download_worker(item, extra, out_path)
    except Exception as e:
        logger.warning(f"Skipping metadata asset '{extra}' for {item.id}: {e}")
        return {'id': item.id, extra: None}

_gwstac._download_worker = _safe_download_worker

@contextlib.contextmanager
def _no_stac_scaling():
    """open_stac rescales every band to reflectance and clips to 0-1
    if the qa mask is desired, it will turn into garbage without this patch"""
    original = _gwstac.STAC_SCALING
    _gwstac.STAC_SCALING = {}
    try:
        yield
    finally:
        _gwstac.STAC_SCALING = original
        
@attr.s
class Parameters(object):
    """General parameters
    """
    proj_ver: str = attr.ib(default=None)
    bounds: T.Union[str, Path, T.Sequence[float]] = attr.ib(default=None)
    bounds_query: str = attr.ib(default=None)
    bounds_buffer: float = attr.ib(default=0, converter=float)
    out_path: Path = attr.ib(default=None, converter=Path)
    start_date: str = attr.ib(default=None)
    end_date: str = attr.ib(default=None)
    epsg: T.Union[int, None] = attr.ib(default=None)
    resolution: float = attr.ib(default=10, converter=float)
    chunksize: int = attr.ib(default=512, converter=int)
    cloud_cover_perc: int = attr.ib(default=90)
    n_workers: int = attr.ib(default=1, converter=int)
    threads_per_worker: int = attr.ib(default=1, converter=int)
    view_asset_keys: bool = attr.ib(default=False)
    max_items: T.Union[int, None] = attr.ib(default=100)
    tqdm_item_position: int = attr.ib(default=0)
    tqdm_extra_position: int = attr.ib(default=1)
    errors_only: bool = attr.ib(default=False)
    metadata_only: bool = attr.ib(default=False)
    sensors: str = attr.ib(default='LS2')
    l7_stop_year: int = attr.ib(default=2020)
    stac_cat: str = attr.ib(default=None)
    not_masked: bool = attr.ib(default=False)
    get_masks: bool = attr.ib(default=False)
    mask_suffix: str = attr.ib(default='_mask')
    mask_dir: str = attr.ib(default=None)
    buffer_extra: int = attr.ib(default=0)
    mask_resolution: float = attr.ib(default=0, converter=float)

    def __attrs_post_init__(self):
        # Open a file if passed as a path
        if not isinstance(self.bounds, gpd.GeoDataFrame):
            self.bounds = gpd.read_file(self.bounds)
        # Query a geometry feature subset
        if self.bounds_query is not None:
            self.bounds = self.bounds.query(self.bounds_query)
        # Buffer the bounds
        if self.bounds_buffer != 0:
            buffer_geom = self.bounds.buffer(
                self.bounds_buffer, cap_style=CAP_STYLE.square, join_style=JOIN_STYLE.mitre
            )
            self.bounds = self.bounds.assign(geometry=buffer_geom)
        self.original_crs = self.bounds.crs
        self.proj_bounds = self.bounds.total_bounds.flatten().tolist()

        # Translate to WGS 84 lat/lon
        if self.bounds.crs != pyproj.CRS.from_epsg(4326):
            self.bounds = self.bounds.to_crs('epsg:4326')


class SensorParams(ABC):
    """Sensor parameters meta class
    """
    bands: T.List[str]
    collection: str
    stac_catalog: str
    mask_data: bool
    epsg: T.Union[int, None]
    resolution: T.Union[float, int]
    resampling: Resampling
    extra_assets: T.List[str]
    out_path: Path
    date_pos: int


@attr.s
class LandsatParams(SensorParams):
    """Landsat-specific parameters
    """
    bands: T.List[str] = attr.ib(default=[
        LandsatBands.blue.value,
        LandsatBands.green.value,
        LandsatBands.red.value,
        LandsatBands.nir.value,
        LandsatBands.swir1.value,
        LandsatBands.swir2.value,
        LandsatBands.qa.value
    ])
    collection: str = attr.ib(default='landsat_c2_l2')
    stac_catalog: str = attr.ib(default='microsoft_v1')
    mask_data: bool = attr.ib(default=True)
    epsg: T.Union[int, None] = attr.ib(default=None)
    resolution: T.Union[float, int] = attr.ib(default=10.0)
    resampling: Resampling = attr.ib(default=Resampling.cubic)
    extra_assets: T.List[str] = attr.ib(default=['ang', 'mtl.txt', 'mtl.xml'])
    out_path: Path = attr.ib(default='.')
    date_pos: int = attr.ib(default=3)

    def __attrs_post_init__(self):
        self.out_path.mkdir(parents=True, exist_ok=True)


@attr.s
class Sentinel2Params(SensorParams):
    """Sentinel-2-specific parameters
    """
    bands: T.List[str] = attr.ib(default=[
        Sentinel2Bands.blue.value,
        Sentinel2Bands.green.value,
        Sentinel2Bands.red.value,
        Sentinel2Bands.nir.value,
        Sentinel2Bands.swir1.value,
        Sentinel2Bands.swir2.value,
        Sentinel2Bands.scl.value
    ])
    collection: str = attr.ib(default='sentinel_s2_l2a')
    stac_catalog: str = attr.ib(default='element84_v1')
    mask_data: bool = attr.ib(default=False)
    epsg: T.Union[int, None] = attr.ib(default=None)
    resolution: T.Union[float, int] = attr.ib(default=10.0)
    resampling: Resampling = attr.ib(default=Resampling.cubic)
    extra_assets: T.List[str] = attr.ib(default=['granule_metadata'])
    out_path: Path = attr.ib(default='.')
    date_pos: int = attr.ib(default=2)

    def __attrs_post_init__(self):
        self.out_path.mkdir(parents=True, exist_ok=True)

@attr.s
class DEMParams(SensorParams):
    """Copernicus DEM GLO-30 parameters
    """
    bands: T.List[str] = attr.ib(default=['data'])
    collection: str = attr.ib(default='cop_dem_glo_30')
    stac_catalog: str = attr.ib(default='element84_v1')
    mask_data: bool = attr.ib(default=False)
    epsg: T.Union[int, None] = attr.ib(default=None)
    resolution: T.Union[float, int] = attr.ib(default=10.0)
    resampling: Resampling = attr.ib(default=Resampling.bilinear)
    extra_assets: T.List[str] = attr.ib(default=None)
    out_path: Path = attr.ib(default='.')
    start_date: str = attr.ib(default='2000-01-01')
    end_date: str = attr.ib(default='2030-01-01')

    def __attrs_post_init__(self):
        self.out_path.mkdir(parents=True, exist_ok=True)


class S2SceneClassificationLegend(enum.Enum):
    """Sentinel-2 Scene Classification Layer legend
    """
    saturated_or_defective = 1
    dark_area_pixels = 2
    cloud_shadows = 3
    vegetation = 4
    bare_soils = 5
    water = 6
    unclassified = 7
    cloud_medium_probability = 8
    cloud_high_probability = 9
    thin_cirrus = 10
    snow_ice = 11


#S2CLOUDLESS_BAND_NAMES = ['coastal', 'blue', 'red', 'nir1', 'nir', 'rededge', 'water', 'cirrus', 'swir1', 'swir2']
#S2CLOUDLESS_BANDS = [getattr(Sentinel2Bands, band).value for band in S2CLOUDLESS_BAND_NAMES]
S2_BAD_SCL = [
    S2SceneClassificationLegend.saturated_or_defective.value,
    S2SceneClassificationLegend.dark_area_pixels.value,
    S2SceneClassificationLegend.cloud_shadows.value,
    S2SceneClassificationLegend.cloud_medium_probability.value,
    S2SceneClassificationLegend.cloud_high_probability.value,
    S2SceneClassificationLegend.thin_cirrus.value,
    S2SceneClassificationLegend.snow_ice.value,
]

S2_VIS_BAND_NAMES = ['blue', 'green', 'red']
S2_VIS_BANDS = [getattr(Sentinel2Bands, band).value for band in S2_VIS_BAND_NAMES]
S2_BAND_ORDER = ('blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'scl')       # scl must stay last (download_masks)
S2_CATALOGS = {   # --stac-cat value: (band enum, extra metadata asset)
    'element84_v1': (Sentinel2Bands, ['granule_metadata']),
    'element84_v0': (Sentinel2Bands_v0, ['metadata']),
    'microsoft_v1': (Sentinel2Bands_v0, ['granule-metadata']),
}

LANDSAT_MASK_ITEMS = ['fill', 'dilated_cloud', 'cirrus', 'cloud', 'cloud_shadow', 'snow']  ## can drop what we don't want here
#LANDSAT_QA_BITS = {'fill': 0, 'dilated_cloud': 1, 'cirrus': 2, 'cloud': 3, 'cloud_shadow': 4, 'snow': 5, 'clear': 6, 'water': 7}
LANDSAT_QA_BITS = QABits.landsat_c2_l2.value

def s2_l2a_clouds(data: xr.DataArray) -> xr.DataArray:
    """Data masker for Sentinel-2 Level 2A
    """
    replace = {
        old_value: NODATA for old_value in [
            S2SceneClassificationLegend.saturated_or_defective.value,
            S2SceneClassificationLegend.dark_area_pixels.value,
            S2SceneClassificationLegend.cloud_shadows.value,
            S2SceneClassificationLegend.cloud_medium_probability.value,
            S2SceneClassificationLegend.cloud_high_probability.value,
            S2SceneClassificationLegend.thin_cirrus.value,
            S2SceneClassificationLegend.snow_ice.value
        ]
    }

    return data.sel(band='scl').gw.replace(replace)


# def s2_l1c_clouds(data: xr.DataArray):
#     """Data masker for Sentinel-2 Level 1C
#     """
#     data = data.assign_coords(
#         band=S2CLOUDLESS_BAND_NAMES
#     )
#
#     kwargs = dict(
#         threshold=0.4,
#         average_over=4,
#         dilation_size=5,
#         all_bands=False
#     )
#     cloud_detector = S2PixelCloudDetector(**kwargs)
#     # Get the S2Cloudless bands'
#     # B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12
#     data_cloudless = data.sel(band=S2CLOUDLESS_BAND_NAMES)
#
#     X = Stackerizer(
#         stack_dims=('y', 'x', 'time'), direction='stack'
#     ).fit_transform(data_cloudless)

SENTINEL2_ID_PATTERN = re.compile(r'^(?P<base>.+)_(?P<idx>\d+)_(?P<suffix>L2A)$')

def _sibling_scene_id(scene_id: str, new_idx: str = '0') -> T.Optional[str]:
    '''
    Sentinel-2 scene ids here follow {base}_{idx}_L2A, where idx
    disambiguates multiple catalog entries for the same tile/date
    (e.g. a legacy-baseline item alongside a Collection-1 reprocessed
    one). Returns the same id with idx replaced by new_idx, or None
    if scene_id doesn't match that pattern or is already new_idx.
    '''
    m = SENTINEL2_ID_PATTERN.match(scene_id)
    if not m or m.group('idx') == new_idx:
        return None
    return f"{m.group('base')}_{new_idx}_{m.group('suffix')}"
    
def get_landsat_sun_angles(mtl_xml_path: T.Union[str, Path]) -> T.Tuple[float, float]:
    """Returns (sun_zenith, sun_azimuth) in degrees from a Landsat Collection 2 MTL.xml file.
    """
    tree = ET.parse(mtl_xml_path)
    root = tree.getroot()
    # Search anywhere in the tree rather than assuming exact nesting,
    # since group structure can vary slightly across sensor generations.
    azimuth = float(root.find('.//SUN_AZIMUTH').text)
    elevation = float(root.find('.//SUN_ELEVATION').text)
    zenith = 90.0 - elevation
    return zenith, azimuth

def get_sentinel2_sun_angles(mtd_tl_path: T.Union[str, Path]) -> T.Tuple[float, float]:
    """Returns (sun_zenith, sun_azimuth) in degrees from a Sentinel-2 MTD_TL.xml (granule metadata) file.
        Note, this is the global scene value. Sentinel-2 MTD_TL.xml does have a Sun_Angles_Grid item, 
        with more precise values (roughly a 23×23 grid at ~5 km spacing) if slight variation (up to 0.5deg) matters
    """
    tree = ET.parse(mtd_tl_path)
    root = tree.getroot()
    zenith = float(root.find('.//Mean_Sun_Angle/ZENITH_ANGLE').text)
    azimuth = float(root.find('.//Mean_Sun_Angle/AZIMUTH_ANGLE').text)
    return zenith, azimuth

def extract_sun_angles(
    processing_info_path: Path,
    scene_info_path: Path,
    sensor: str,
    metadata_col: str,
) -> pd.DataFrame:
    """Parses sun zenith/azimuth from each scene's metadata file (referenced in scene_info) and writes 
    them into processing_info, keyed by scene id.
    """
    processing_info = pd.read_pickle(processing_info_path)
    scene_info = pd.read_pickle(scene_info_path)

    if metadata_col not in scene_info.columns:
        alt = next((c for c in ('granule_metadata', 'granule-metadata', 'metadata', 'mtl.xml') if c in scene_info.columns), None)
        if alt is None:
            raise KeyError(f'no metadata column ({metadata_col!r} or an alternative) in {scene_info_path}')
        metadata_col = alt
      
    for col in ('sun_zenith', 'sun_azimuth'):
        if col not in processing_info.columns:
            processing_info[col] = pd.NA

    parser = get_sentinel2_sun_angles if sensor == 'S2' else get_landsat_sun_angles

    n_ok, n_missing, n_failed = 0, 0, 0
    for _, row in scene_info.iterrows():
        scene_id = row['id']
        meta_path = row[metadata_col]

        if scene_id not in processing_info.index:
            logger.warning(f"{scene_id} not in processing.info; skipping")
            n_missing += 1
            continue
        if pd.isna(meta_path) or not Path(meta_path).is_file():
            logger.warning(f"Metadata file missing on disk for {scene_id}; skipping")
            n_missing += 1
            continue

        try:
            zenith, azimuth = parser(meta_path)
        except Exception as e:
            ## can get angles from sibling file (same acquisition from a different collection)
            sibling_id = _sibling_scene_id(scene_id, new_idx='0') if sensor == 'S2' else None
            sibling_match = scene_info.loc[scene_info['id'] == sibling_id] if sibling_id else None

            if sibling_match is not None and not sibling_match.empty:
                sibling_meta_path = sibling_match.iloc[0][metadata_col]
                try:
                    zenith, azimuth = parser(sibling_meta_path)
                    logger.info(
                        f"{scene_id} metadata failed to parse ({e}); "
                        f"used sibling {sibling_id}'s sun angles instead"
                    )
                    processing_info.loc[scene_id, 'sun_zenith'] = zenith
                    processing_info.loc[scene_id, 'sun_azimuth'] = azimuth
                    #processing_info.loc[scene_id, 'sun_angle_source'] = sibling_id
                    n_ok += 1
                    continue
                except Exception as e2:
                    logger.warning(f"Sibling {sibling_id} also failed to parse: {e2}")
            logger.warning(f"Failed to parse sun angles for {scene_id}: {e}")
            n_failed += 1
            continue

        processing_info.loc[scene_id, 'sun_zenith'] = zenith
        processing_info.loc[scene_id, 'sun_azimuth'] = azimuth
        n_ok += 1

    logger.info(
        f"Sun angles: {n_ok} extracted, {n_missing} missing, {n_failed} failed "
        f"-> {processing_info_path}"
    )
    pd.to_pickle(processing_info, processing_info_path)
    return processing_info

def write_stack(
    row_id: str,
    data: xr.DataArray,
    out_path: Path,
    threads_per_worker: int,
    tqdm_extra_position: int,
    l7_stop_year: int
):
    """Writes a single time band stack to file
        returns a dictionary entry to log in processing database if image not
        already in database
    """
    processing_db = pd.read_pickle(Path(out_path.parent/'processing.info'))
    out_name = out_path / f"{row_id}.tif"
    error_file = out_path.parent.parent / 'errors' / f'{Path(out_name).parent.parent.name}_{Path(out_name).stem}_ERROR.txt'
    error_file.parent.mkdir(parents=True, exist_ok=True)
    if error_file.is_file():
        if out_name.is_file():
            out_name.unlink()
        error_file.unlink()
    # only download images that don't already exist in directory/database:
    process_this = True
    if row_id in processing_db.index:
        if (processing_db.at[row_id,'redownload'] in ('True', 'TRUE')) or (processing_db.at[row_id,'redownload'] is True):
            logger.info(f'reprocessing {row_id}')
            #pass
        else:
            process_this = False
    if process_this == True:
        if row_id.split('_')[0][:4] == 'LE07' and int(row_id.split('_')[3][:4]) >= l7_stop_year:
            new_dict_entry={row_id:{'dl':'None','skip':True,'redownload':False,'skip_reason':'L7 after l7_stop_year'}}
            #logger.info('ignoring image because it is L7 and past l7_stop_year')
        else:
            # Index by scene id because it is always unique
            scene_idx = data.id.values.tolist().index(row_id)
            try:
                try:
                    img = (
                            data
                            .isel(time=scene_idx)
                            .squeeze()
                            .transpose('band', 'y','x')
                    )
                    aband = (img.sel(band='blue')
                             .squeeze()
                             .where(lambda x:(x!=img.nodatavals[0]) & (x > 0))
                             .astype('uint8')
                    )
                    valid_pix = dask.array.count_nonzero(aband).compute(num_workers=4)

                    img.gw.save(
                            out_name,
                            overwrite=True,
                            compression='lzw',
                            num_workers=threads_per_worker,
                            log_progress=False,
                            tqdm_kwargs={
                                'position': tqdm_extra_position+2,
                                'desc': 'Chunks'
                            }
                        )
                except (RasterioIOError, RuntimeError, ValueError) as e:
                    raise DownloadError(e)
                if out_name.is_file():
                    try:
                        # Ensure the file is not corrupted
                        with gw.open(out_name):
                            pass
                        # Successful open at this point
                        # If there was previously an error file, remove it
                        if error_file.is_file():
                            error_file.unlink()
                    except Exception as e:
                        raise CorruptedError(f'The downloaded file is corrupted because of {e}.')
                        new_dict_entry = {row_id:{'dl':'None','redownload':True,'error':'download corrupted','dldate':datetime.today().strftime('%Y-%m-%d')}}
                else:
                    raise DownloadError('No file was created')
                    new_dict_entry={row_id:{'dl':'None','redownload':True,'error':'download failed','dldate':datetime.today().strftime('%Y-%m-%d')}}

                new_dict_entry={row_id:
                                {'dl':out_name,
                                 'beforeDB':False,
                                 'redownload':False,
                                 'skip':False,
                                 'dldate':datetime.today().strftime('%Y-%m-%d'),
                                 'numpix':valid_pix}
                                }

            except (CorruptedError, DownloadError) as e:
                if out_name.is_file():
                    out_name.unlink()
                if e.message.endswith('The image is empty.'):
                    if error_file.is_file():
                        error_file.unlink()
                    new_dict_entry={row_id:{'dl':'None','skip':True,'redownload':False,'skip_reason':'image is empty','dldate':datetime.today().strftime('%Y-%m-%d')}}
                else:
                    with open(error_file, mode='w') as f:
                        f.write(str(e))
                    new_dict_entry={row_id:{'dl':'None','redownload':True,'error':'download failed','dldate':datetime.today().strftime('%Y-%m-%d')}}
        #logger.info('adding entry to new download dict: {}'.format(new_dict_entry))

    else:
        new_dict_entry = {}
    return new_dict_entry

@timeout(600)
def write_time_stack(
    data: xr.DataArray,
    out_path: Path,
    n_workers: int,
    threads_per_worker: int,
    df: pd.DataFrame,
    tqdm_extra_position: int,
    desc: str,
    chunks: int,
    errors_only: bool,
    l7_stop_year: int
) -> None:
    """Writes each DataArray time step to file
    """
    # Check for chunk sizes in intervals of 16
    data = rechunk(data, chunks)

    if errors_only:
        df['error_file'] = out_path.parent / 'errors' / (f'{out_path.parent.name}_' + df.id.map(str) + '_ERROR.txt')
        df['error_file_exists'] = df.error_file.apply(lambda x: x.is_file())
        df = df.loc[df.error_file_exists]
        if df.empty:
            return []

    data = (
        data
        .gw.assign_nodata_attrs(NODATA)
        .assign_attrs(**{
            'time': pd.DatetimeIndex(data.time.values).round('D'),
            'crs': data.gw.crs_to_pyproj.to_wkt()
        })
    )

    # from tqdm.dask import TqdmCallback
    # with TqdmCallback(desc=desc):
    #     # Download the data into memory
    #     data.load(num_workers=threads_per_worker)

    if n_workers == 1:
        results = []
        for row in tqdm(
            df.sort_values('date').itertuples(),
            position=tqdm_extra_position,
            desc=desc
        ):
            pdic = write_stack(
                row.id,
                data,
                out_path,
                threads_per_worker,
                tqdm_extra_position,
                l7_stop_year
            )
            results.append(pdic)

    else:
        data_gen = (
            (
                row.id,
                data,
                out_path,
                threads_per_worker,
                tqdm_extra_position,
                l7_stop_year
            ) for row in df.sort_values('date').itertuples()
        )

        results = []
        try:
            with parallel_backend(
                backend='loky',
                n_jobs=n_workers
            ):
                with TqdmParallel(
                    tqdm_kwargs={
                        'total': len(df.index),
                        'desc': desc
                    }
                ) as pool:
                    results = pool(delayed(write_stack)(*args) for args in data_gen)
                    #__ = pool(
                    #    delayed(write_stack)(*args) for args in data_gen
                    #)

        except (CorruptedError, DownloadError) as e:
            logger.warning(e)
            pass

    return results

def fetch_mask_stack(params, sensor_params, mask_band, nodata, dtype, buffer_extra=0.0, resolution=None):
    """
    Nearest-neighbour, unscaled QA/SCL stack on the same grid as the images if desired (<buffer_extra> = 0)
        If <buffer_extra> > 0: the extent grows by buffer_extra metres on every side, so it no longer matches the image grid
            This is so that clouds just off the edge of the grid cell can have their shadows cast on the cell when masking
            (if used, the shadow step needs to resample onto the cell's grid, which it does in tstuyau). 
        <resolution> (m) can be used to override the sensor's resolution
    """
    
    bounds = params.bounds                       # lat/lon search bounds (left, bottom, right, top), degrees
    proj_bounds = None if params.original_crs == pyproj.CRS.from_epsg(4326) else params.proj_bounds   # metres
    if buffer_extra > 0:
        crs = params.original_crs
        dist = buffer_extra / 111320.0 if crs.is_geographic else float(buffer_extra)   # metres -> CRS units
        # same square, mitred buffer Parameters uses, done in the original CRS and brought back to lat/lon
        grown = bounds.to_crs(crs).buffer(dist, cap_style=CAP_STYLE.square, join_style=JOIN_STYLE.mitre)
        bounds = bounds.assign(geometry=grown.to_crs('epsg:4326'))
        if proj_bounds is not None:
            proj_bounds = [proj_bounds[0] - dist, proj_bounds[1] - dist, proj_bounds[2] + dist, proj_bounds[3] + dist]
            
    with _no_stac_scaling():
        data, _ = open_stac(
            stac_catalog=sensor_params.stac_catalog,
            bounds=bounds,
            proj_bounds=proj_bounds,
            collection=sensor_params.collection,
            bands=[mask_band],
            cloud_cover_perc=params.cloud_cover_perc,
            chunksize=params.chunksize,
            start_date=params.start_date,
            end_date=params.end_date,
            bounds_query=params.bounds_query,
            mask_data=False,
            resolution=sensor_params.resolution if resolution is None else resolution,
            epsg=sensor_params.epsg,
            resampling=Resampling.nearest,
            max_items=params.max_items,
        )
    if data is None:
        return None
    return (
        data.fillna(nodata).astype(dtype)
        .gw.assign_nodata_attrs(nodata)
        .assign_attrs(crs=data.gw.crs_to_pyproj.to_wkt())
    )

def _bad_to_dataarray(bad, data):
    return xr.DataArray(bad, dims=('time', 'y', 'x'),
                        coords={'time': data.time, 'y': data.y, 'x': data.x})

def _align_to_images(mask, data):
    """Reorder the mask's time axis to match the image stack, by scene id."""
    mask_ids = mask.id.values.tolist()
    order = [mask_ids.index(i) for i in data.id.values.tolist()]
    mask = mask.isel(time=order, band=0)
    if mask.shape != (data.sizes['time'], data.sizes['y'], data.sizes['x']):
        raise RuntimeError('Mask grid does not match image grid')
    return mask

def apply_landsat_qa_mask(data, params, landsat_params, items=LANDSAT_MASK_ITEMS):
    qa = fetch_mask_stack(params, landsat_params, LandsatBands.qa.value, nodata=1, dtype='uint16')
    if qa is None:
        raise RuntimeError('Could not fetch Landsat QA for masking')
    bitmask = sum(1 << LANDSAT_QA_BITS[i] for i in items)
    qa = _align_to_images(qa, data)
    bad = _bad_to_dataarray((qa.data & bitmask) != 0, data)
    return data.where(~bad).assign_attrs(**data.attrs)


def apply_s2_scl_mask(data, params, s2_params):
    scl = fetch_mask_stack(params, s2_params, s2_params.bands[-1], nodata=0, dtype='uint8')
    if scl is None:
        raise RuntimeError('Could not fetch Sentinel-2 SCL for masking')
    scl = _align_to_images(scl, data)
    arr = scl.data
    bad = arr == S2_BAD_SCL[0]
    for v in S2_BAD_SCL[1:]:
        bad = bad | (arr == v)
    bad = _bad_to_dataarray(bad, data)
    return xr.where(bad, NODATA, data).assign_attrs(**data.attrs)

def download_landsat(params: Parameters, landsat_params: LandsatParams) -> int:
    """Downloads Landsat data
    """
    ### Note for legacy: Py crop maps version 0 & 1 used option to apply qa mask directly to data in geowombat.
    ##   this incorrectly masks clear pixels because cubic convolution is incorrectly applied to the mask.
    ##   for other versions and projects, we apply the masks separately using the methods here, so the mask_data parameter in
    ##   geowombat open_stac is set to False.
    apply_masks = False
    if params.proj_ver in ['Py_0','Py_1','Py0','Py1']:
        apply_masks = True
    data, df = open_stac(
        stac_catalog=landsat_params.stac_catalog,
        bounds=params.bounds,
        proj_bounds=None if params.original_crs == pyproj.CRS.from_epsg(4326) else params.proj_bounds,
        collection=landsat_params.collection,
        bands=landsat_params.bands,
        cloud_cover_perc=params.cloud_cover_perc,
        chunksize=params.chunksize,
        start_date=params.start_date,
        end_date=params.end_date,
        mask_data=apply_masks,
        resolution=landsat_params.resolution,
        epsg=landsat_params.epsg,
        view_asset_keys=params.view_asset_keys,
        resampling=landsat_params.resampling,
        extra_assets=landsat_params.extra_assets,
        out_path=landsat_params.out_path,
        max_items=params.max_items
    )
    if data is None:
        return 0,[]
    if landsat_params.mask_data:
        data = apply_landsat_qa_mask(data, params, landsat_params)
    data = data.assign_coords(band=[LandsatBands(band).name for band in data.band.values])
    data = data.assign_attrs(scene_ids=df.id.values.tolist())

    # Save the scene info
    scene_info = landsat_params.out_path / 'scene.info'
    df, df_download = pickle_scene_info(df, date_pos=3, scene_info=scene_info)

    # Write each date to file
    ld = write_time_stack(
        data,
        landsat_params.out_path,
        n_workers=params.n_workers,
        threads_per_worker=params.threads_per_worker,
        df=df_download,
        tqdm_extra_position=params.tqdm_extra_position,
        desc='Landsat scenes',
        chunks=params.chunksize,
        errors_only=params.errors_only,
        l7_stop_year=params.l7_stop_year
    )

    return int(data.epsg.values), ld


def download_sentinel2(params: Parameters, sentinel2_params: Sentinel2Params):
    """Downloads Sentinel-2 data
    """
    mask_band = sentinel2_params.bands[-1]
    bands = ([b for b in sentinel2_params.bands if b != mask_band]
         if (params.not_masked) else sentinel2_params.bands)

    data, df = open_stac(
        stac_catalog=sentinel2_params.stac_catalog,
        bounds=params.bounds,
        proj_bounds=None if params.original_crs == pyproj.CRS.from_epsg(4326) else params.proj_bounds,
        collection=sentinel2_params.collection,
        bands=bands,
        cloud_cover_perc=params.cloud_cover_perc,
        chunksize=params.chunksize,
        start_date=params.start_date,
        end_date=params.end_date,
        bounds_query=params.bounds_query,
        resolution=sentinel2_params.resolution,
        epsg=sentinel2_params.epsg,
        nodata_fill=NODATA,
        resampling=sentinel2_params.resampling,
        extra_assets=sentinel2_params.extra_assets,
        out_path=sentinel2_params.out_path,
        max_items=params.max_items
    )
    if data is None:
        return []

    band_enum = S2_CATALOGS[sentinel2_params.stac_catalog][0]
    s2_bands = [band_enum(band).name for band in data.band.values]
    data = data.assign_coords(band=s2_bands)
    data = data.assign_attrs(scene_ids=df.id.values.tolist())

    ## Mask clouds using the ESA sen2cor scene classification layer -- only for legacy because this incorrectly applies
    ##  cubic convolution to mask. Correct method applies masks outside geowombat
    if params.proj_ver in ['Py_0','Py_1','Py0','Py1']:
        data_s2_cloud_mask = s2_l2a_clouds(data)

        data = (
            (
                xr.where(data_s2_cloud_mask == NODATA, NODATA, data)
            )
            .sel(band=[band for band in s2_bands if band != 'scl'])
            .assign_attrs(**data.attrs)
        )

    if not params.not_masked:
        data = apply_s2_scl_mask(data, params, sentinel2_params)
    
    # Save the scene info
    scene_info = sentinel2_params.out_path / 'scene.info'
    df, df_download = pickle_scene_info(df, date_pos=2, scene_info=scene_info)

    # Write each date to file
    sd = write_time_stack(
        data,
        sentinel2_params.out_path,
        n_workers=params.n_workers,
        threads_per_worker=params.threads_per_worker,
        df=df_download,
        tqdm_extra_position=params.tqdm_extra_position,
        desc='Sentinel-2 scenes',
        chunks=params.chunksize,
        errors_only=params.errors_only,
        l7_stop_year=params.l7_stop_year
    )

    return sd

def download_dem(params: Parameters, dem_params: DEMParams):
    """Downloads and mosaics Copernicus DEM GLO-30 for the current bounds

    Unlike download_landsat/download_sentinel2, this has no time
    dimension to iterate -- it produces a single static mosaic clipped
    and resampled to the cell's grid, so it bypasses write_time_stack.
    """
    with rio.Env(AWS_NO_SIGN_REQUEST='YES'):
        data, _ = open_stac(
            stac_catalog=dem_params.stac_catalog,
            bounds=params.bounds,
            proj_bounds=None if params.original_crs == pyproj.CRS.from_epsg(4326) else params.proj_bounds,
            collection=dem_params.collection,
            start_date=dem_params.start_date, 
            end_date=dem_params.end_date,
            bands=dem_params.bands,
            bounds_query=params.bounds_query,
            resolution=dem_params.resolution,
            epsg=dem_params.epsg,
            resampling=dem_params.resampling,
            out_path=dem_params.out_path,
            max_items=None
        )

        if data is None:
            logger.warning('No Copernicus DEM tiles were found for this cell')
            return None

        stack_dim = 'time' if 'time' in data.dims else None  ## if None throws an error, just pass in generous time bounds
        if stack_dim and data.sizes.get(stack_dim, 1) > 1:
            logger.info(f'Cell spans {data.sizes[stack_dim]} DEM tiles; mosaicking')
            attrs = data.attrs.copy()   ## .max() / .mean(), etd drop attrs, so will need this later 
            ## skip na should work because Copernicus DEM tiles are seamless and non-overlapping
            data = data.max(dim=stack_dim, skipna=True)
            data = data.assign_attrs(attrs)
        
        elif stack_dim:
            data = data.squeeze(stack_dim, drop=True)

        data = data.fillna(-9999)
        data = data.gw.assign_nodata_attrs(-9999)
        dem_out = dem_params.out_path / 'dem.tif'
        data.squeeze().gw.save(dem_out, overwrite=True, compression='lzw')
    
    logger.info(f'Saved DEM mosaic to {dem_out}')

    return dem_out

def download_metadata_only(params: Parameters, sensor_params: SensorParams, id_col_date_pos: int) -> pd.DataFrame:
    """Downloads only the extra metadata assets (no pixel data) for scenes matching the current bounds/date range.
    """
    kwargs = dict(
        stac_catalog=sensor_params.stac_catalog,
        bounds=params.bounds,
        proj_bounds=None if params.original_crs == pyproj.CRS.from_epsg(4326) else params.proj_bounds,
        collection=sensor_params.collection,
        bounds_query=params.bounds_query,
        start_date=params.start_date,
        end_date=params.end_date,
        cloud_cover_perc=params.cloud_cover_perc,
        chunksize=params.chunksize,
        resolution=sensor_params.resolution,
        epsg=sensor_params.epsg,
        resampling=sensor_params.resampling,
        extra_assets=sensor_params.extra_assets,
        out_path=sensor_params.out_path,
        max_items=params.max_items,
    )

    try:
        # Preferred: never fetch pixel data at all
        data, df = open_stac(bands=[sensor_params.bands[0]], compute=False, **kwargs)
    except TypeError:
        # Fallback for geowombat versions without `compute`
        logger.warning("open_stac() in this geowombat version has no 'compute' arg; "
                        "falling back to a single minimal band instead of a lazy skip.")
        logger.info(f'getting metadata from {sensor_params.stac_catalog}' )
        data, df = open_stac(bands=[sensor_params.bands[0]], **kwargs)

    if df is None or df.empty:
        logger.warning('No metadata found for this cell/date range')
        return pd.DataFrame()

    extract_sun_angles(
        processing_info_path=params.out_path / 'processing.info',
        scene_info_path=SensorPaths(out_path=params.out_path).sentinel2 / 'scene.info',
        sensor='S2',
        metadata_col='granule_metadata',
    )
    extract_sun_angles(
        processing_info_path=params.out_path / 'processing.info',
        scene_info_path=SensorPaths(out_path=params.out_path).landsat / 'scene.info',
        sensor='L',
        metadata_col='mtl.xml',
    )

    logger.info(f"updated processing info: {pd.read_pickle(params.out_path / 'processing.info').head()}")

def download_masks(params: Parameters, sensor_params:SensorParams, mask_band=None, nodata=None, dtype=None,
                   img_dir=None, mask_path=None, buffer_extra=0, mask_suffix='_mask', mask_resolution=0.0) -> None:
    """Saves the QA / SCL layer for every already-downloaded image for date range as <image_stem><mask_suffix>.tif in mask dir 
    Uses nearest-neighbour resampling and no reflectance scaling, so class/bit values are preserved.
    <mask_suffix> allows for different <mask_suffix> vals for different masks (e.g. wide and clipped), but all 
        masks should contain _mask in the name if downloaded to the img directory to not interfere with image sorting
    <buffer_extra>(m) allows extra buffer space (beyond normal grid buffer) so that clouds off edge can be projected 
        onto shadows. NOTE: With <buffer_extra> > 0, the mask no longer sits on the image grid and alignment is needed in 
        downstream methods. Likewise, <mask_resolution>(m) can be used to override the sensor's resolution for download 
        to reduce file size (but output won't be aligned with grid). The default <mask_resolution>= 0.0 sets the mask resolution
        to <resolution> (the main project setting) to align with other products. 
    """
    img_dir = Path(img_dir if img_dir is not None else sensor_params.out_path)
    logger.info(f' masking images in {img_dir}')
    mask_dir = img_dir if mask_path in [None, 'None', ''] else Path(mask_path)
    logger.info(f'saving mask into {mask_dir}')
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    start, end = str(params.start_date).replace('-', ''), str(params.end_date).replace('-', '')
    date_pos = sensor_params.date_pos
    missing = sorted(
        p for p in img_dir.glob('*.tif')
        if '_mask' not in p.stem
        and (date_pos is None or start <= p.stem.split('_')[date_pos][:8] <= end)
        and not (mask_dir / f'{p.stem}{mask_suffix}.tif').is_file()
    )
    if not missing:
        logger.info(f'No missing masks in {img_dir}')
        return
    
    logger.info('downloading mask layers for existing images...')
    missing_ids = {p.stem: p for p in missing}

    data = fetch_mask_stack(params, sensor_params, mask_band, nodata, dtype, buffer_extra=buffer_extra,
                            resolution=mask_resolution if mask_resolution > 0 else None)
    
    if data is None:
        logger.warning(f'No scenes found when fetching masks for {img_dir} ({params.start_date} to {params.end_date})')
        return

    scene_ids = data.id.values.tolist()

    for scene_id, img_path in tqdm(missing_ids.items(), desc='Masks',
                                   position=params.tqdm_extra_position):
        if scene_id not in scene_ids:
            logger.warning(f'{scene_id} not returned by the mask search; skipping')
            continue
            
        out_file = mask_dir / f'{scene_id}{mask_suffix}.tif'
        
        try:
            m = (data.isel(time=scene_ids.index(scene_id))
                     .transpose('band', 'y', 'x'))
            with rio.open(img_path) as src:
                if buffer_extra <= 0:
                    ## sanity check: must sit on the same grid as the image
                    if (src.height, src.width) != (m.sizes['y'], m.sizes['x']):
                        logger.warning(f'{scene_id}: mask shape differs from image; skipping')
                        continue
                else:
                    ## sanity check: the mask must contain the image (within one mask pixel)
                    tol = abs(float(m.x[1] - m.x[0]))
                    b = src.bounds
                    if (float(m.x.min()) - tol > b.left or float(m.x.max()) + tol < b.right
                        or float(m.y.min()) - tol > b.bottom or float(m.y.max()) + tol < b.top):
                        logger.warning(f'{scene_id}: mask does not cover the image extent; skipping')
                        continue

            m.gw.save(
                out_file,
                overwrite=True,
                compression='lzw',
                num_workers=params.threads_per_worker,
                log_progress=False,
            )
        except Exception as e:
            logger.warning(f'Failed to write mask for {scene_id}: {e}')
            if out_file.is_file():
                out_file.unlink()
                
                
def download_data(args):
    params = Parameters(
        proj_ver=args.proj_ver,
        bounds=args.bounds,
        bounds_query=args.bounds_query,
        bounds_buffer=args.bounds_buffer,
        out_path=args.out_path,
        start_date=args.start_date,
        end_date=args.end_date,
        epsg=args.epsg,
        resolution=args.resolution,
        cloud_cover_perc=args.cloud_cover_perc,
        chunksize=args.chunksize,
        n_workers=args.workers,
        threads_per_worker=args.threads,
        view_asset_keys=args.view_asset_keys,
        max_items=None if args.max_items == -1 else args.max_items,
        errors_only=args.errors_only,
        metadata_only=args.metadata_only,
        sensors=args.sensors,
        stac_cat=args.stac_cat,
        l7_stop_year=args.l7_stop_year,
        not_masked=args.not_masked,
        get_masks=args.get_masks,
        mask_suffix=args.mask_suffix,
        mask_dir=args.mask_dir,
        buffer_extra=args.buffer_extra,
        mask_resolution = args.mask_resolution
    )

    if 'L' in params.sensors:
        ## Set Landsat Settings:
        default_bands = attr.fields(LandsatParams).bands.default
        if params.proj_ver in ['Py_0','Py_1','Py0','Py1']:
            lbands = default_bands
        else:
            lbands=[b for b in default_bands if b != LandsatBands.qa.value]
        ## by default, Landsat data has mask applied in geowombat open stac.
        if params.not_masked:
            masked = False
        landsat_params = LandsatParams(
            out_path=SensorPaths(out_path=params.out_path).landsat,
            epsg=params.epsg,
            resolution=params.resolution,
            bands=lbands,
            mask_data=not(params.get_masks or params.not_masked),
        )

    if 'S2' in params.sensors:
        ## Set Sentinel-2 Settings (catalog chosen with --stac-cat; element84_v1 if not given)
        stac_catalog = params.stac_cat or 'element84_v1'
        if stac_catalog not in S2_CATALOGS:
            raise ValueError(f'--stac-cat must be one of {list(S2_CATALOGS)}, got {stac_catalog!r}')
        band_enum, extra_assets = S2_CATALOGS[stac_catalog]
        logger.info(f'Sentinel-2 catalog: {stac_catalog}')
 
        sentinel2_params = Sentinel2Params(
            out_path=SensorPaths(out_path=params.out_path).sentinel2,
            epsg=params.epsg,
            resolution=params.resolution,
            stac_catalog=stac_catalog,
            extra_assets=extra_assets,
            bands=[getattr(band_enum, name).value for name in S2_BAND_ORDER],
        )

    if params.metadata_only:
        logger.info('downloading only metadata')
        if 'L' in params.sensors:
            download_metadata_only(params, landsat_params, id_col_date_pos=3)
            
        if 'S2' in params.sensors:
            download_metadata_only(params, sentinel2_params, id_col_date_pos=2)
        return  # skip the rest of the imagery download path
    
    if 'DEM' in params.sensors:
        dem_out_path = SensorPaths(out_path=params.out_path).dem
        dem_path = dem_out_path / 'dem.tif'
        if not dem_path.is_file():
            dem_params = DEMParams(
                out_path=dem_out_path,
                epsg=params.epsg,
                resolution=params.resolution
            )
            download_dem(params, dem_params)
        else:
            logger.info('using existing DEM file')
            
    if 'L' in params.sensors or 'S2' in params.sensors:
        processing_info_path = params.out_path/ 'processing.info'
        if processing_info_path.is_file():
            processing_dict = pd.read_pickle(processing_info_path).to_dict(orient='index')

        else:
            processing_dict = {}
            if os.path.exists(params.out_path/'landsat'):
                landsat_files = [fi for fi in os.listdir(params.out_path/'landsat') 
                                 if fi.endswith('.tif') and '_mask' not in fi]
                if len(landsat_files) > 0:
                    for f in landsat_files:
                        processing_dict[os.path.splitext(f)[0]]={
                            'dl':params.out_path/'landsat/{}'.format(f),
                            'beforeDB':True,
                            'redownload':False,
                            'skip':False
                            }
            if os.path.exists(params.out_path/'sentinel2'):
                sentinel2_files = [fi for fi in os.listdir(params.out_path/'sentinel2') 
                                   if fi.endswith('.tif') and '_mask' not in fi]
                if len(sentinel2_files) > 0:
                    for s in sentinel2_files:
                        processing_dict[os.path.splitext(s)[0]]={
                            'dl':params.out_path/'sentinel2/{}'.format(s),
                            'beforeDB':True,
                            'redownload':False,
                            'skip':False}
            # Write new processing database
            if not os.path.exists(params.out_path):
                os.makedirs(params.out_path)
            new_processing_info = pd.DataFrame.from_dict(processing_dict,orient='index')
            new_processing_info.rename_axis('id', axis=1, inplace=True)
            pd.to_pickle(new_processing_info, processing_info_path)
        
        logger.info(f'\n {len(processing_dict)} images already downloaded (or looked at and skipped) \n')

    if 'L' in params.sensors:
        ## Download Landsat
        landsatdl = download_landsat(params, landsat_params)

        if landsatdl[0] == 0:
            logger.warning('There are no Landsat images for this period')
            num_new_landsat = 0

        else:    
            num_new_landsat= sum(1 for e in landsatdl[1] if e)
            if num_new_landsat > 0:
                if len(processing_dict) ==  0:
                    processing_dict = next((e for e in landsatdl[1] if e),{})
                for d in filter(None,landsatdl[1]):
                    processing_dict.update(d)
    else:
        num_new_landsat = 0
        

    if 'S2' in params.sensors:
        ## Download Sentinel-2
        sentdl = download_sentinel2(params, sentinel2_params)
        num_new_sentinel = sum(1 for e in sentdl if e)
        if num_new_sentinel > 0:
            for d in filter(None,sentdl):
                processing_dict.update(d)
    else:
        num_new_sentinel = 0

    if(num_new_landsat + num_new_sentinel)>0:
        new_processing_info = pd.DataFrame.from_dict(processing_dict,orient='index')
        new_processing_info.rename_axis('id', axis=1, inplace=True)
        pd.to_pickle(new_processing_info, processing_info_path)
        logger.info(f'{num_new_landsat} Landsat images and {num_new_sentinel} Sentinel-2 images downloaded and added to database.')
        
    if params.get_masks:
        if 'L' in params.sensors:
            download_masks(
                params,
                landsat_params,
                mask_band=LandsatBands.qa.value,
                nodata=1,
                dtype='uint16',
                mask_suffix=params.mask_suffix,
                img_dir=params.out_path / 'landsat',
                mask_path= params.out_path / 'landsat' if not params.mask_dir else params.out_path/params.mask_dir,
                buffer_extra= 0 if params.buffer_extra<=0 else params.buffer_extra,
                mask_resolution= 30.0 if params.mask_resolution<=0.0 else params.mask_resolution
            )
        if 'S2' in params.sensors:
            download_masks(
                params,
                sentinel2_params,
                mask_band=sentinel2_params.bands[-1],   # scl (also correct for the v0 catalogs)
                nodata=0,
                dtype='uint8',
                mask_suffix=params.mask_suffix,
                img_dir=params.out_path / 'sentinel2',
                mask_path= params.out_path / 'sentinel2' if not params.mask_dir else params.out_path/params.mask_dir,
                buffer_extra = 0 if params.buffer_extra<=0 else params.buffer_extra,
                mask_resolution=20.0 if params.mask_resolution <=0.0 else params.mask_resolution
            )
        
        
