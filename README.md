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

```commandline
eostac brdf \
    --start-date 2020-01-01 \
    --end-date 2021-01-01 \
    --project-path /downloads/000001 \
    --out-path /brdf/000001 \
    --coeffs-path /eostac/files/coefficients/ \
    --file-format geotiff \
    --apply-bandpass \
    --threads 8
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