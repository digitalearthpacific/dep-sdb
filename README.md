## Satellite Derived Bathymetry Product for the Pacific Region

In development.


### Testing

```bash
python src/run_task.py \
    --model-zip-uri=https://dep-public-staging.s3.us-west-2.amazonaws.com/dep_s2_sdb/models/2025_04_16d_nn.zip \
    --tile-id=64,20 \
    --version=0.1.0 \
    --include-scaler
    --parallelism=8
```