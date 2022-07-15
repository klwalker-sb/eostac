## Examples

### Download a time series of Landsat and Sentinel-2 data

#### Use a single file for the search bounding box

```commandline
eostac download \
    --start-date 2020-01-01 \
    --end-date 2021-01-01 \
    --out-path /downloads/000001 \
    --bounds bounds.geojson \
    --cloud-cover 90 \
    --threads 8 \
    --max-items -1
```

#### Use a feature row within the geometry file for the search bounding box

```commandline
eostac download \
    --start-date 2020-01-01 \
    --end-date 2021-01-01 \
    --out-path /downloads/000001 \
    --bounds bounds.geojson \
    --bounds-query 'grid==1' \
    --cloud-cover 90 \
    --threads 8 \
    --max-items -1
```

#### Download 4 scenes in parallel, with 4 concurrent threads per band/chunk

> NOTE: this example would require 16 cores

```commandline
eostac download \
    --start-date 2020-01-01 \
    --end-date 2021-01-01 \
    --out-path /downloads/000001 \
    --bounds bounds.geojson \
    --bounds-query 'grid==1' \
    --cloud-cover 90 \
    --workers 4 \
    --threads 4 \
    --max-items -1
```

### Adjust for sensor radiometric differences

#### Download data

```commandline
eostac adjust \
    --start-date 2013-01-01 \
    --end-date 2013-14-01 \
    --out-path /coefficients \
    --geometry /PRY_adm.zip \
    --n-samples 10 \
    --freq Y \
    --freq-repeat 9 \
    --threads 8 \
    --epsg 8858
```

#### Get linear coefficients

```commandline
eostac adjust \
    --start-date 2013-01-01 \
    --end-date 2013-14-01 \
    --out-path /coefficients \
    --geometry /PRY_adm.zip \
    --n-samples 10 \
    --freq Y \
    --freq-repeat 9 \
    --threads 8 \
    --epsg 8858 \
    --regress
```

### Apply BRDF normalization to downloaded surface reflectance

> NOTE: use `--workers` and not `--threads` with the BRDF process, as dask handles all workers

```commandline
eostac brdf \
    --start-date 2020-01-01 \
    --end-date 2021-01-01 \
    --project-path /downloads/000001 \
    --out-path /brdf/000001 \
    --coeffs-path /eostac/files/coefficients/ \
    --file-format geotiff \
    --apply-bandpass \
    --workers 8
```

> Output BRDF names

```text
L<processing level>_<sensor>_<image acquisition date>_<file creation date>_C<collection><extension>
```

where,

* processing level
  * 3A
    * BRDF
  * 3B
    * Bandpass
* sensor
  * S2A
  * S2B
  * LT05
  * LE07
  * LC08
  * LC09
* image acquisition date
  * YYYYMMDD
* file creation date
  * YYYYMMDD
* collection
  * 01
* extension
  * .tif
  * .nc

Example:

```text
L3B_S2A_20210104_20220711_C01.tif
```

### View scene file information

#### Read the `scene.info` file

```python
from eostac.stac_utils import read_df

df = read_df('/brdf/000001/scene.info')
```

#### Get a specific time slice

```python
df_slice = df['2021-01-01':'2022-01-01']
```

#### Get a list of scene ids

```python
df_slice.id.values.tolist()
```

#### Get Sentinel-2 scene ids

```python
df_s2a = df.query("id == 's2a'")
```

#### Open time series

```python
from pathlib import Path
import geowombat as gw
from eostac.stac_utils import read_df
import matplotlib.pyplot as plt


brdf_path = Path('/brdf/00001')
df = read_df(brdf_path / 'scene.info')
file_list = (str(brdf_path) + '/' + df.id + '.tif').values.tolist()
time_names = df.date.dt.strftime('%Y-%m-%d').values.tolist()
band_names = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

# We use overlap='min' because the nodata values are 32768.
with gw.open(
    file_list,
    time_names=time_names,
    band_names=band_names,
    overlap='min'
) as src:
    # View RGB of first date
    (
        src
        .isel(time=0)
        .sel(band=['red', 'green', 'blue'])
        .squeeze()
        .where(lambda x: x != x.nodatavals[0])
        .gw.imshow(robust=True)
    )
    plt.show()
```