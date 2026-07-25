# FD001 LSTM/GRU Hidden-Dim Comparison

Validation metrics from saved LSTM and GRU result files. Uses best-epoch validation metrics when available.

| Model | Hidden Dim | Target | Best Epoch | Trained Epochs | Train RMSE | Train MAE | Val RMSE | Val MAE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gru | 64 | raw | 61 | 72 | 30.8147 | 17.5907 | 23.6389 | 16.6905 |
| gru | 128 | raw | 33 | 48 | 32.2895 | 19.3011 | 22.8826 | 16.0703 |
| gru | 256 | raw | 30 | 32 | 29.6828 | 18.8880 | 22.0890 | 15.3056 |
| lstm | 64 | raw | 55 | 70 | 33.5718 | 18.5196 | 24.7112 | 16.6915 |
| lstm | 128 | raw | 44 | 59 | 32.0483 | 19.0648 | 23.3039 | 16.0995 |
| lstm | 256 | raw | 17 | 32 | 33.4678 | 20.8005 | 22.4455 | 15.4877 |
