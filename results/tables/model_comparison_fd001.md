# FD001 Model Comparison

Validation metrics from saved experiment result files.

| Model | Target | Train RMSE | Train MAE | Val RMSE | Val MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| mean baseline | raw | 63.7555 | 51.9936 | 53.3266 | 45.0388 |
| median baseline | raw | 64.0799 | 51.7534 | 52.2061 | 44.0990 |
| cycle-only | raw | 48.1256 | 38.0464 | 36.6858 | 30.2421 |
| ridge | raw | 36.1414 | 26.8972 | 31.2521 | 25.5538 |
| MLP | raw | 32.9016 | 22.8513 | 25.7360 | 19.2739 |
| MLP | capped_125 | 16.7483 | 12.5427 | 17.8267 | 13.1033 |
| GRU | raw | 36.1100 | 20.9116 | 24.6257 | 17.0478 |
| GRU | capped_125 | 13.4156 | 10.8200 | 15.6378 | 12.2138 |
