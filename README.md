# 大兴PM2.5预报空间辅助数据下载与处理

本仓库通过 GitHub Actions 自动下载、裁剪、核验并打包大兴研究区空间辅助数据。

## 核心公开数据

- Copernicus DEM GLO-30及坡度；
- ESA WorldCover 2021 10 m土地覆盖、建成区和耕地；
- WorldPop 2025约100 m约束型人口栅格；
- OpenStreetMap道路与建筑轮廓；
- MODIS MOD13Q1 V6.1，2024-05-01至2026-06-15的250 m、16天NDVI及质量层；
- 22站空间特征表和数据清单。

## 输出

工作流成功后会产生：

`daxing_pm25_auxiliary_data_core.zip`

压缩包内的 `07_docs/DATA_MANIFEST.csv` 记录每项数据是否真实下载成功、来源、输出路径及限制。

## 说明

CAMS需要ADS/CDS凭证；本地工业源、交通流量、扬尘治理记录需要人工整理或部门申请。EDGAR 0.1度排放只适合作为对照，暂不进入核心包，避免在22站小研究区内形成块状伪影。
