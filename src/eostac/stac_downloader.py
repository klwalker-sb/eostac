from pathlib import Path
import enum
import typing as T
from abc import ABC

from .stac_utils import (
    LandsatBands,
    Sentinel2Bands,
    SensorPaths,
    pickle_scene_info,
    NODATA,
    NODOWNLOAD
)

from geowombat.core.stac import open_stac
# from geowombat.ml.transformers import Stackerizer

import pandas as pd
from rasterio.enums import Resampling
import xarray as xr
from s2cloudless import S2PixelCloudDetector
import attr
from tqdm.auto import tqdm


@attr.s
class Parameters(object):
    """General parameters
    """
    bounds: T.Union[str, Path] = attr.ib(default=None)
    out_path: Path = attr.ib(default=None, converter=Path)
    start_date: str = attr.ib(default=None)
    end_date: str = attr.ib(default=None)
    bounds_query: str = attr.ib(default=None)
    epsg: int = attr.ib(default=None)
    chunksize: int = attr.ib(default=512, converter=int)
    cloud_cover_perc: int = attr.ib(default=90)
    n_workers: int = attr.ib(default=1, converter=int)
    threads_per_worker: int = attr.ib(default=1, converter=int)
    view_asset_keys: bool = attr.ib(default=False)
    max_items: T.Union[int, None] = attr.ib(default=100)
    tqdm_item_position: int = attr.ib(default=0)
    tqdm_extra_position: int = attr.ib(default=1)


class SensorParams(ABC):
    """Sensor parameters meta class
    """
    bands: T.List[str]
    collection: str
    stac_catalog: str
    mask_data: bool
    epsg: int
    resolution: T.Union[float, int]
    resampling: Resampling
    extra_assets: T.List[str]
    out_path: Path


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
    stac_catalog: str = attr.ib(default='microsoft')
    mask_data: bool = attr.ib(default=True)
    epsg: int = attr.ib(default=None)
    resolution: T.Union[float, int] = attr.ib(default=10.0)
    resampling: Resampling = attr.ib(default=Resampling.cubic)
    extra_assets: T.List[str] = attr.ib(default=['ang', 'mtl.txt', 'mtl.xml'])
    out_path: Path = attr.ib(default='.')

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
    stac_catalog: str = attr.ib(default='element84')
    mask_data: bool = attr.ib(default=False)
    epsg: int = attr.ib(default=None)
    resolution: T.Union[float, int] = attr.ib(default=10.0)
    resampling: Resampling = attr.ib(default=Resampling.cubic)
    extra_assets: T.List[str] = attr.ib(default=['metadata'])
    out_path: Path = attr.ib(default='.')

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


S2CLOUDLESS_BAND_NAMES = ['coastal', 'blue', 'red', 'nir1', 'nir', 'rededge', 'water', 'cirrus', 'swir1', 'swir2']
S2CLOUDLESS_BANDS = [getattr(Sentinel2Bands, band).value for band in S2CLOUDLESS_BAND_NAMES]
S2_VIS_BAND_NAMES = ['blue', 'green', 'red']
S2_VIS_BANDS = [getattr(Sentinel2Bands, band).value for band in S2_VIS_BAND_NAMES]


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

    return data.sel(band=Sentinel2Bands('SCL').name).gw.replace(replace)


def s2_l1c_clouds(data: xr.DataArray):
    """Data masker for Sentinel-2 Level 2A
    """
    data = data.assign_coords(
        band=S2CLOUDLESS_BAND_NAMES
    )

    kwargs = dict(
        threshold=0.4,
        average_over=4,
        dilation_size=5,
        all_bands=False
    )
    cloud_detector = S2PixelCloudDetector(**kwargs)
    # Get the S2Cloudless bands
    # B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12
    data_cloudless = data.sel(band=S2CLOUDLESS_BAND_NAMES)

    X = Stackerizer(
        stack_dims=('y', 'x', 'time'), direction='stack'
    ).fit_transform(data_cloudless)


def write_time_stack(
    data: xr.DataArray,
    out_path: Path,
    n_workers: int,
    threads_per_worker: int,
    df: pd.DataFrame,
    tqdm_extra_position: int
) -> None:
    """Writes each DataArray time step to file
    """
    for row in tqdm(
        df.sort_values('date').itertuples(),
        position=tqdm_extra_position+1,
        desc='Scene id'
    ):
        out_name = out_path / f"{row.id}.tif"
        if not out_name.is_file():
            # Index by scene id because it is always unique
            scene_idx = data.scene_ids.index(row.id)
            (
                data
                .assign_coords(time=pd.DatetimeIndex(data.time.values).round('D'))
                .isel(time=scene_idx)
                .squeeze()
                .transpose('band', 'y', 'x')
                .gw.to_raster(
                    out_name,
                    n_workers=n_workers,
                    n_threads=threads_per_worker,
                    compress='lzw',
                    overwrite=True,
                    tqdm_kwargs={'position': tqdm_extra_position+2, 'desc': 'Chunks'}
                )
            )


def download_landsat(params: Parameters, landsat_params: LandsatParams) -> int:
    """Downloads Landsat data
    """
    data, df = open_stac(
        stac_catalog=landsat_params.stac_catalog,
        bounds=params.bounds,
        collection=landsat_params.collection,
        bands=landsat_params.bands,
        cloud_cover_perc=params.cloud_cover_perc,
        chunksize=params.chunksize,
        start_date=params.start_date,
        end_date=params.end_date,
        bounds_query=params.bounds_query,
        mask_data=landsat_params.mask_data,
        resolution=landsat_params.resolution,
        view_asset_keys=params.view_asset_keys,
        resampling=landsat_params.resampling,
        extra_assets=landsat_params.extra_assets,
        out_path=landsat_params.out_path,
        max_items=params.max_items,
        tqdm_item_position=params.tqdm_item_position,
        tqdm_extra_position=params.tqdm_extra_position
    )
    if data is None:
        return NODOWNLOAD
    else:
        data = data.assign_coords(band=[LandsatBands(band).name for band in data.band.values])
        data = data.assign_attrs(scene_ids=df.id.values.tolist())

        # Save the scene info
        scene_info = landsat_params.out_path / 'scene.info'
        df, df_download = pickle_scene_info(df, date_pos=3, scene_info=scene_info)

        # Write each date to file
        write_time_stack(
            data,
            landsat_params.out_path,
            n_workers=params.n_workers,
            threads_per_worker=params.threads_per_worker,
            df=df_download,
            tqdm_extra_position=params.tqdm_extra_position
        )

        return int(data.epsg.values)


def download_sentinel2(params: Parameters, sentinel2_params: Sentinel2Params):
    """Downloads Sentinel-2 data
    """
    data, df = open_stac(
        stac_catalog=sentinel2_params.stac_catalog,
        bounds=params.bounds,
        collection=sentinel2_params.collection,
        bands=sentinel2_params.bands,
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
        max_items=params.max_items,
        tqdm_item_position=params.tqdm_item_position,
        tqdm_extra_position=params.tqdm_extra_position
    )
    if data is not None:
        s2_bands = [Sentinel2Bands(band).name for band in data.band.values]
        data = data.assign_coords(band=s2_bands)
        data = data.assign_attrs(scene_ids=df.id.values.tolist())

        # Mask clouds using the ESA sen2cor scene classification layer
        data_s2_cloud_mask = s2_l2a_clouds(data)
        data = (
            (
                xr.where(data_s2_cloud_mask == NODATA, NODATA, data)
            )
            .sel(band=[band for band in s2_bands if band != Sentinel2Bands('SCL').name])
            .assign_attrs(**data.attrs)
        )

        # Save the scene info
        scene_info = sentinel2_params.out_path / 'scene.info'
        df, df_download = pickle_scene_info(df, date_pos=2, scene_info=scene_info)

        # Write each date to file
        write_time_stack(
            data,
            sentinel2_params.out_path,
            n_workers=params.n_workers,
            threads_per_worker=params.threads_per_worker,
            df=df_download,
            tqdm_extra_position=params.tqdm_extra_position
        )


def download_data(args):
    params = Parameters(
        bounds=args.bounds,
        out_path=args.out_path,
        start_date=args.start_date,
        end_date=args.end_date,
        bounds_query=args.bounds_query,
        epsg=args.epsg,
        cloud_cover_perc=args.cloud_cover_perc,
        chunksize=args.chunksize,
        n_workers=args.workers,
        threads_per_worker=args.threads,
        view_asset_keys=args.view_asset_keys,
        max_items=None if args.max_items == -1 else args.max_items,
        tqdm_item_position=args.tqdm_item_position,
        tqdm_extra_position=args.tqdm_extra_position
    )
    # Download Landsat
    landsat_params = LandsatParams(
        out_path=SensorPaths(out_path=params.out_path).landsat,
        epsg=params.epsg
    )
    epsg = download_landsat(params, landsat_params)

    if epsg != NODOWNLOAD:
        # Download Sentinel-2
        sentinel2_params = Sentinel2Params(
            out_path=SensorPaths(out_path=params.out_path).sentinel2,
            epsg=epsg
        )
        download_sentinel2(params, sentinel2_params)
