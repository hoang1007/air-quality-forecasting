# air-quality-forecasting

## Notes:

    - Các config và chạy train, test được mô tả trong file config/config.yaml
    - Chỉnh sửa các config của model ở trong folder config/model
    - Model daqff có số lượng tham số nhỏ hơn daqff-large (model được sử dụng để submit)

## Train

> Chạy full config
```bash
docker run --rm airqualityforecasting ++mode=train ++device=cpu model=daqff-large
```

> Chạy với config có sẵn trong config.yaml
```
docker run --rm airqualityforecasting
```
> Chỉnh sửa config trong config.yaml


## Test
> Command line
```bash
docker run --rm airqualityforecasting ++mode=test
```

> Chỉnh sửa config:
```bash
mode=test
```