import os
from pathlib import Path
import rasterio as rio
import logging
import attr
import typing as T
from .utils import set_color_logger

logger = set_color_logger(__name__)

@attr.s
class Parameters(object):
    file_type: str  = attr.ib(default='tif')
    out_path: T.Union[str, Path, None] = attr.ib(default=None)

def rm_corrupt_files(args):
    out_path=args.out_path
    file_type=args.file_type

    if os.path.exists(out_path):
        logger.info(f'checking {out_path} for corrupted {file_type} files')
        for f in os.listdir(out_path):
            if f.endswith(file_type):
                #TODO: if filetype is '.nc' use geowombat or rioxarray to open
                try:
                    with rio.open(os.path.join(out_path,f)) as src:
                        ras=src.read(1)
                    pass
                except Exception as e:
                    logger.warning('file {} is corrupt: {}. \n'.format(f,e))
                    rmList=[fl for fl in os.listdir(out_path) if f.strip(file_type) in fl]
                    for item in rmList:
                        logger.warning('removing {}'.format(item))
                        os.remove(os.path.join(out_path,item))
