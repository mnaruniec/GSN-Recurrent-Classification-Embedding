import torch


DATA_PATH = "./data/"
TRAIN_X_PATH = DATA_PATH + "train_x.csv"
TRAIN_Y_PATH = DATA_PATH + "train_y.csv"
TEST_X_PATH = DATA_PATH + "test_x.csv"
TEST_Y_PATH = DATA_PATH + "test_y.csv"

NUM_CLASSES = 4

SNAPSHOT_PATH = "./snapshots/"
# BEST_SNAPSHOT_PATH = "./snapshots/Snap_a9674_i9383_14_05_2020_15_28"

DEFAULT_TRUNCATION_AMOUNT = 1

DEFAULT_MB_SIZE = 16
DEFAULT_STAT_PERIOD = 80
DEFAULT_STAT_MBS = 40

DEFAULT_NUM_EPOCHS = 50
DEFAULT_PATIENCE = 3
DEFAULT_EPOCH_TRAIN_EVAL = False
DEFAULT_LR = 0.0001
DEFAULT_WEIGHT_DECAY = 0.001

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if not torch.cuda.is_available():
    print('WARNING! CUDA is not available - running on CPU.')
