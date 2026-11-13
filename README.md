# FedSCR: federated shift aware conformal calibration of sensor reliability

Code for the experiments on federated calibration of per sensor reliability tests for multisensor fault
diagnosis. Each client scores every sensor group with a cross sensor reconstruction residual and a class
typicality score, shares only quantile sketches of these scores on healthy data, and receives client specific
thresholds from a mixture of sketches weighted by the distance between healthy score distributions. An adaptive
partner selection test gives sensor alarms and fusion weights. A new site calibrates from unlabelled windows by
trimming with a tail shape taken from the other clients.

## Requirements

Python 3.9 or later and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

Everything runs on a CPU. The neural networks are small multilayer perceptrons written in numpy.

## Datasets

| Dataset | Source | Expected files |
|---|---|---|
| KAIST | Jung et al., Data in Brief 48 (2023) 109049 | vibration `.mat` files and current and temperature `.tdms` files, any folder layout |
| Paderborn | Lessmeier et al., PHM Europe 2016 | `.mat` files of bearings K001 to K005, KA04, KA15, KA16, KA22, KA30, KI04, KI14, KI16, KI18, KI21 |
| MCC5-THU | Chen et al., Data in Brief 54 (2024) 110453, Mendeley Data `p92gj2732w` | `.csv` files, or `.npy` files made by `--download-mcc5` |

MCC5-THU can be downloaded and converted with

```
python run_experiments.py --dataset mcc5 --data data/mcc5 --download-mcc5
```

## Running

```
python run_experiments.py --dataset kaist     --data data/kaist     --out results/kaist
python run_experiments.py --dataset paderborn --data data/paderborn --out results/paderborn
python run_experiments.py --dataset mcc5      --data data/mcc5      --out results/mcc5
python summarize.py results/kaist
```

`--seeds 0,1` runs a subset of the five seeds and `--only calibration,new_site` runs a subset of the experiments.
A quick check of the whole pipeline on a small synthetic machine takes a few minutes:

```
python run_experiments.py --dataset synthetic --seeds 0
python summarize.py results/synthetic
```

The feature table of each dataset is cached in the output folder (`table_*.pkl`) and reused by later runs.

## Output files

| File | Content |
|---|---|
| `c1.csv` | false alarm rate and detection power per seed, calibration size, method, client and sensor group |
| `fusion.csv` | fused macro F1 per seed, client, degradation scenario and weighting method |
| `partner_far.csv` | false alarm rate of healthy groups while one partner group is degraded |
| `c2.csv` | new site calibration at an unseen operating condition with contaminated windows |
| `c3.csv`, `c3_honest.csv` | manipulated sketches, and an honest client with a degraded sensor during calibration |
| `tail_attack.csv`, `tail_cal.csv` | manipulation of the upper tail only, and the cost of a tail aware distance |
| `scaling.csv` | results for different numbers of clients |
| `ablation.csv` | bandwidth constant, sketch grid and envelope features |
| `meta.json` | run times, sketch size and dataset summary |

## Main settings

All settings are in `fedscr/config.py`: target level 0.05, 5 seeds, 100 calibration windows per client
(20, 50 and 300 also run), 30 FedAvg rounds for the classifier heads and 15 for the subset predictors, two local
epochs, Adam with learning rate 0.001 and batch size 128, bandwidth constant 1.5, Dirichlet concentration 0.5 and
eight sensor degradation operators at severities 0.6 and 1.0.

## Code layout

| Module | Content |
|---|---|
| `fedscr/data.py` | file inventory, signal loaders, MCC5-THU download |
| `fedscr/features.py` | window features, degradation operators, window table |
| `fedscr/federated.py` | clients, numpy MLP, FedAvg, reliability scores |
| `fedscr/calibration.py` | quantile sketches, mixture weights, p values, adaptive partner selection, trimming, attacks |
| `fedscr/experiments.py` | the experiments |

## License

MIT
