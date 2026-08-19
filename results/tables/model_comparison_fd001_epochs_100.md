# FD001 Model Comparison

Validation metrics from saved experiment result files. Deep models use best-epoch validation metrics when available.

| Model | Target | Train RMSE | Train MAE | Val RMSE | Val MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| mean baseline | raw | 63.7555 | 51.9936 | 53.3266 | 45.0388 |
| median baseline | raw | 64.0799 | 51.7534 | 52.2061 | 44.0990 |
| cycle-only | raw | 48.1256 | 38.0464 | 36.6858 | 30.2421 |
| ridge | raw | 36.1414 | 26.8972 | 31.2521 | 25.5538 |
| MLP | raw | 30.8972 | 21.0173 | 23.8568 | 18.0148 |
| GRU | raw | 30.8147 | 17.5907 | 23.6389 | 16.6905 |
| LSTM | raw | 33.5718 | 18.5196 | 24.7112 | 16.6915 |
| CNN | raw | 29.4004 | 19.8436 | 25.2088 | 18.5996 |
