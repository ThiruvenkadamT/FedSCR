"""Experiment settings. `configure` fills the module level names used by the other modules."""
import os

ALPHA = 0.05
SEEDS = [0, 1, 2, 3, 4]
N_CAL_GRID = [20, 50, 100, 300]
N_CAL_MAIN = 100
FL_ROUNDS, PRED_ROUNDS, LOCAL_EPOCHS, BATCH, LR = 30, 15, 2, 128, 1e-3
H_C = 1.5
DIR_ALPHA = 0.5
KINDS = ["noise", "gain", "bias", "stuck", "clip", "dropout", "bandwidth", "fail"]
SEVS = [0.6, 1.0]
EPS_GRID = [0.0, 0.05, 0.10, 0.20]
ATTACKS = [("none", 1.0), ("deflate", 0.5), ("deflate", 0.2), ("inflate", 3.0), ("inflate", 10.0)]
N_BYZ = 2
VARIANTS_FUSION = ["adaptive", "q2", "q1", "max", "min", "cross_only", "typ_only"]
RUN_ABLATIONS = True
RUN_SCALING = True
C2_MAX_FOLDS = 4
NB, NE = 48, 24              # spectral bands and envelope spectrum bands per channel
SPLIT = (0.6, 0.2, 0.2)      # contiguous split of each recording (KAIST)
CHUNK = 8                    # interleaved split (MCC5-THU): chunks of 8 windows, pattern train x3, cal, test

PU_BEARINGS = {"healthy": ["K001", "K002", "K003", "K004", "K005"],
               "outer": ["KA04", "KA15", "KA16", "KA22", "KA30"],
               "inner": ["KI04", "KI14", "KI16", "KI18", "KI21"]}
PU_CONDS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]

# filled by configure() and by data.inventory()
DATASET = DATA_ROOT = OUT_DIR = None
FS = WIN = HOP = None
EVID = PARTNER = GROUPS = GTYPE = None
N_SHARDS = SPLIT_MODE = SCAL_SHARDS = None
CONDS = CLASSES = None
INVENTORY = []


def configure(dataset, data_root=".", out_dir=None, seeds=None, split=None):
    global DATASET, DATA_ROOT, OUT_DIR, FS, WIN, HOP, EVID, PARTNER, GROUPS, GTYPE, N_SHARDS, SPLIT_MODE
    global SCAL_SHARDS, SEEDS, FL_ROUNDS, PRED_ROUNDS, VARIANTS_FUSION
    DATASET, DATA_ROOT = dataset, data_root
    if dataset == "kaist":
        FS, WIN, HOP = 25600, 8192, 4096
        EVID, PARTNER = ["vibA", "vibB", "current"], ["temp"]
        GTYPE = {"vibA": "dyn", "vibB": "dyn", "current": "dyn", "temp": "slow"}
        N_SHARDS, SPLIT_MODE, SCAL_SHARDS = 3, "time", [1, 2, 3, 4, 6]
    elif dataset == "paderborn":
        FS, WIN, HOP = 64000, 16384, 16384
        EVID, PARTNER = ["vib", "cur1", "cur2"], ["mech"]
        GTYPE = {"vib": "dyn", "cur1": "dyn", "cur2": "dyn", "mech": "slow"}
        N_SHARDS, SPLIT_MODE, SCAL_SHARDS = 3, "recording", [1, 2, 3, 4]
    elif dataset == "mcc5":
        FS, WIN, HOP = 12800, 4096, 4096
        EVID, PARTNER = ["motorVib", "gearVib"], ["mech"]
        GTYPE = {"motorVib": "dyn", "gearVib": "dyn", "mech": "slow"}
        N_SHARDS, SPLIT_MODE, SCAL_SHARDS = 1, "interleaved", []
        VARIANTS_FUSION = ["adaptive", "q1", "max", "min", "cross_only", "typ_only"]
    elif dataset == "synthetic":
        FS, WIN, HOP = 25600, 8192, 4096
        EVID, PARTNER = ["vibA", "vibB", "current"], ["temp"]
        GTYPE = {"vibA": "dyn", "vibB": "dyn", "current": "dyn", "temp": "slow"}
        N_SHARDS, SPLIT_MODE, SCAL_SHARDS = 3, "time", [1, 3, 6]
        SEEDS, FL_ROUNDS, PRED_ROUNDS = [0, 1], 12, 8
    else:
        raise ValueError(f"unknown dataset {dataset}")
    if split:
        SPLIT_MODE = split
    if seeds:
        SEEDS = list(seeds)
    GROUPS = EVID + PARTNER
    OUT_DIR = out_dir or os.path.join("results", dataset)
    os.makedirs(OUT_DIR, exist_ok=True)
