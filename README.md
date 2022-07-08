## Examples

### Download a time series of Landsat and Sentinel-2 data

#### Use a single file for the search bounding box

```python
eostac download \
    --start-date 2020-01-01 \
    --end-date 2021-01-01 \
    --out-path /downloads \
    --bounds bounds.geojson \
    --cloud-cover 90 \
    --threads 8 \
    --max-items -1
```

#### Use a feature row within the geometry file for the search bounding box

```python
eostac download \
    --start-date 2020-01-01 \
    --end-date 2021-01-01 \
    --out-path /downloads \
    --bounds bounds.geojson \
    --bounds-query 'grid==1' \
    --cloud-cover 90 \
    --threads 8 \
    --max-items -1
```

#### Download 4 scenes in parallel, with 4 concurrent threads per band/chunk

> NOTE: this example would require 16 cores

```python
eostac download \
    --start-date 2020-01-01 \
    --end-date 2021-01-01 \
    --out-path /downloads \
    --bounds bounds.geojson \
    --bounds-query 'grid==1' \
    --cloud-cover 90 \
    --workers 4 \
    --threads 4 \
    --max-items -1
```

### Adjust for sensor radiometric differences

#### Download data

```python
eostac adjust \
    --start-date 2020-01-01 \
    --end-date 2020-14-01 \
    --out-path /coefficients \
    --geometry /paraguay.zip \
    --n-samples 20 \
    --freq Y \
    --freq-repeat 10 \
    --threads 8 \
```

#### Get linear coefficients

```python
eostac adjust \
    --start-date 2020-01-01 \
    --end-date 2020-14-01 \
    --out-path /coefficients \
    --geometry /paraguay.zip \
    --n-samples 20 \
    --freq Y \
    --freq-repeat 10 \
    --threads 8 \
    --regress
```

### Apply BRDF normalization to downloaded surface reflectance

```python
eostac brdf \
    --start-date 2020-01-01 \
    --end-date 2021-01-01 \
    --project-path /downloads \
    --out-path /brdf \
    --file-format geotiff \
    --threads 8
```