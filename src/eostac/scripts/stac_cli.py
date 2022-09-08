import argparse
import enum

from ..stac_downloader import download_data
from ..stac_post import load_data, FileFormat
from ..stac_adjust import adjust_data
from ..stac_smooth import smooth_data


class Processes(enum.Enum):
    download = 'download'
    brdf = 'brdf'
    adjust = 'adjust'
    smooth = 'smooth'


def main():
    parser = argparse.ArgumentParser(description='STAC CLI')
    subparsers = parser.add_subparsers(dest='process')

    for process in Processes:
        subparser = subparsers.add_parser(process.value)

        subparser.add_argument(
            '-sdt', '--start-date', dest='start_date',
            help='The search start date (YYYY-MM-DD) (default: %(default)s)'
        )
        subparser.add_argument(
            '-edt', '--end-date', dest='end_date', help='The search end date (YYYY-MM-DD) (default: %(default)s)'
        )
        subparser.add_argument(
            '-o', '--out-path', dest='out_path', help='The output path (default: %(default)s)'
        )
        subparser.add_argument(
            '-r', '--resolution', dest='resolution', help='The cell resolution (default: %(default)s)',
            default=10.0, type=float
        )
        subparser.add_argument(
            '-w', '--workers', dest='workers', help='The number of process workers (default: %(default)s)',
            default=1, type=int
        )
        subparser.add_argument(
            '-t', '--threads', dest='threads', help='The threads per worker (default: %(default)s)',
            default=1, type=int
        )

        if process == Processes.download:
            subparser.add_argument(
                '-b', '--bounds', dest='bounds', help='The download bounding box (default: %(default)s)'
            )
            subparser.add_argument(
                '-q', '--bounds-query', dest='bounds_query',
                help='A query to select bounds from a file (default: %(default)s)'
            )
            subparser.add_argument(
                '-bb', '--bounds-buffer', dest='bounds_buffer',
                help='A buffer to apply around the bounds (default: %(default)s)',
                default=0, type=int
            )
            subparser.add_argument(
                '--epsg', dest='epsg',
                help='An EPSG code to transform the data to (default: %(default)s)', default=None
            )
            subparser.add_argument(
                '-cc', '--cloud-cover', dest='cloud_cover_perc',
                help='The maximum percentage cloud cover (default: %(default)s)',
                default=90, type=int
            )
            subparser.add_argument(
                '-c', '--chunksize', dest='chunksize', help='The dask chunk size (default: %(default)s)',
                default=512, type=int
            )
            subparser.add_argument(
                '--view-asset-keys', dest='view_asset_keys', help='Whether to show asset keys (default: %(default)s)',
                action='store_true'
            )
            subparser.add_argument(
                '--max-items', dest='max_items',
                help='The maximum number of items to return from the search (default: %(default)s)',
                type=int, default=100
            )
            subparser.add_argument(
                '--tqdm-item-position', dest='tqdm_item_position',
                help='The position of the item progress bar (default: %(default)s)',
                type=int, default=0
            )
            subparser.add_argument(
                '--tqdm-extra-position', dest='tqdm_extra_position',
                help='The position of the extra progress bar (default: %(default)s)',
                type=int, default=1
            )
            subparser.add_argument(
                '--errors-only', dest='errors_only',
                help='Whether to only check and download errors (default: %(default)s)',
                action='store_true'
            )
        elif process == Processes.brdf:
            subparser.add_argument(
                '-p', '--project-path', dest='project_path', help='The input project path (default: %(default)s)'
            )
            subparser.add_argument(
                '-f', '--file-format', dest='file_format', help='The output file format (default: %(default)s)',
                choices=[file_format.value for file_format in FileFormat],
                default=FileFormat.netcdf.value
            )
            subparser.add_argument(
                '--apply-bandpass', dest='apply_bandpass',
                help='Whether to apply bandpass adjustments (default: %(default)s)',
                action='store_true'
            )
            subparser.add_argument(
                '--get-coeffs', dest='get_coeffs',
                help='Whether to get bandpass coefficients (default: %(default)s)',
                action='store_true'
            )
            subparser.add_argument(
                '--coeffs-path', dest='coeffs_path',
                help='The path where coefficient files are stored (default: %(default)s)'
            )
            subparser.add_argument(
                '--coeffs-prefix', dest='coeffs_prefix', default='pry_',
                help='The coefficient prefix (default: %(default)s)'
            )
            subparser.add_argument(
                '--show-plot', dest='show_plot',
                help='Whether to show the regression plots (default: %(default)s)',
                action='store_true'
            )
            subparser.add_argument(
                '--pool-size', dest='pool_size',
                help='The processing pool size (default: %(default)s)',
                type=int, default=10
            )
        elif process == Processes.adjust:
            subparser.add_argument(
                '--epsg', dest='epsg',
                help='An EPSG code to transform the data to (default: %(default)s)', type=int
            )
            subparser.add_argument(
                '-g', '--geometry', dest='geometry', help='The geometry (default: %(default)s)'
            )
            subparser.add_argument(
                '-n', '--n-samples', dest='n_samples', help='The number of random samples (default: %(default)s)',
                type=int, default=20
            )
            subparser.add_argument(
                '-d', '--dist', dest='dist', help='The point offset distance (default: %(default)s)',
                type=float, default=0.001
            )
            subparser.add_argument(
                '-f', '--freq', dest='freq', help='The date range frequency (default: %(default)s)',
                default=None, choices=['Y', 'M']
            )
            subparser.add_argument(
                '--freq-repeat', dest='freq_repeat', help='The number of frequency repetitions (default: %(default)s)',
                default=1, type=int
            )
            subparser.add_argument(
                '--regress', dest='regress', help='Whether to apply linear regression (default: %(default)s)',
                action='store_true'
            )
            subparser.add_argument(
                '--max-day-diff', dest='max_day_diff',
                help='The maximum day difference for Landsat (default: %(default)s)',
                type=int, default=3
            )
            subparser.add_argument(
                '--errors-only', dest='errors_only',
                help='Whether to only check and download errors (default: %(default)s)',
                action='store_true'
            )
        elif process == Processes.smooth:
            subparser.add_argument(
                '-p', '--project-path', dest='project_path', help='The input project path (default: %(default)s)'
            )
            subparser.add_argument(
                '--vi', dest='vi', help='The vegetation index (default: %(default)s)', default='evi2'
            )
            subparser.add_argument(
                '--bands', dest='bands', help='The bands to open (default: %(default)s)', nargs='+'
            )
            subparser.add_argument(
                '--read-chunks', dest='read_chunks', help='The read chunk size (default: %(default)s)',
                default=512, type=int
            )
            subparser.add_argument(
                '--proc-chunks', dest='proc_chunks', help='The processing chunk size (default: %(default)s)',
                default=128, type=int
            )
            subparser.add_argument(
                '--freq', dest='freq', help='The interpolation frequency (default: %(default)s)',
                default='SMS', choices=['M', 'MS', 'SM', 'SMS', 'W', 'D']
            )
            subparser.add_argument(
                '--target-day', dest='target_day', help='The target day (m-d) (default: %(default)s)',
                default='7-1'
            )
            subparser.add_argument(
                '--memory-limit', dest='memory_limit', help='The memory limit per worker (default: %(default)s)',
                default='auto'
            )

    args = parser.parse_args()

    if args.process == Processes.download.value:
        download_data(args)
    elif args.process == Processes.brdf.value:
        load_data(args)
    elif args.process == Processes.adjust.value:
        adjust_data(args)
    elif args.process == Processes.smooth.value:
        smooth_data(args)


if __name__ == '__main__':
    main()
