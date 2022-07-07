## Examples

### Download a time series of Landsat and Sentinel-2 data

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