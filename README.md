# air-quality-forecasting

## Notes:

    - Các config và chạy train, test được mô tả trong file config/config.yaml
    - Chỉnh sửa các config của model ở trong folder config/model
    - Sau khi chạy xong test có thể tải folder `submit` trong volume

## Build docker image
```bash
docker build -t airqualityforecasting .
```

## Khởi tạo volume
    docker volume create hblh

## Chuẩn bị dữ liệu
> **_NOTE:_** Chỉ cần chạy lần đầu tiên để giải nén dữ liệu vào volume
```bash
docker run -v hblh:/app --rm airqualityforecasting mode=makedata
```
## Train

> Chạy full config
```bash
docker run -v hblh:/app --rm airqualityforecasting mode=train device=cpu
```

> Chạy với config có sẵn trong config.yaml
```
docker run -v hblh:/app --rm airqualityforecasting
```
> Chỉnh sửa config trong config.yaml


## Test
> Command line
```bash
docker run -v hblh:/app --rm airqualityforecasting ++mode=test
```

> Chỉnh sửa config:
```bash
mode=test
```

## Copy submit folder to local
```bash
CID=$(docker run -d -v hblh:/app busybox true)
docker cp $CID:/app/submit ./
```