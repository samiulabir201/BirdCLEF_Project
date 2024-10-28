import os

class Config:
    # Directories
    KAGGLE = True
    if KAGGLE:
        DATA_DIR = "/kaggle/input/birdclef-2024/"
    else:
        DATA_DIR = "/mnt/d/kaggle/birdclef-2024/"

    WAVE_PATH = "original_waves/second_30/"

    # Model settings
    MODEL_NAME = 'eca_nfnet_l0'
    POOL_TYPE = 'avg'
    NUM_CLASSES = 183  # Adjust based on your label count

    # Training settings
    TRAIN_DURATION = 30
    SLICE_DURATION = 5
    TEST_DURATION = 5
    TRAIN_DROP_DURATION = 1

    # Spectrogram parameters
    SR = 32000
    FMIN = 20
    FMAX = 15000
    N_MELS = 128
    N_FFT = N_MELS * 8
    SIZE_X = 512
    HOP_LENGTH = int(SR * SLICE_DURATION / SIZE_X)
    TEST_HOP_LENGTH = int(SR * TEST_DURATION / SIZE_X)
    BINS_PER_OCTAVE = 12

    # Cross-validation
    N_FOLDS = 5
    INFERENCE_FOLDS = [4]

    # Optimization
    ENABLE_AMP = True
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 1
    LOSS_TYPE = "BCEFocalLoss"
    LR = 1.0e-03
    OPTIMIZER = 'adan'
    WEIGHT_DECAY = 1.0e-02
    ES_PATIENCE = 5
    DETERMINISTIC = True
    MAX_EPOCH = 9
    AUG_EPOCH = 6
    SEED = 42

    # Augmentation
    AUG_NOISE = 0.0
    AUG_GAIN = 0.0
    AUG_WAVE_PITCHSHIFT = 0.0
    AUG_WAVE_SHIFT = 0.0
    AUG_SPEC_XYMASKING = 0.0
    AUG_SPEC_COARSEDROP = 0.0
    AUG_SPEC_HFLIP = 0.0
    AUG_SPEC_MIXUP = 0.0
    AUG_SPEC_MIXUP_PROB = 0.5
    ALPHA = 0.95

    # Others
    USE_SECONDARY = True
    SECONDARY_LABEL_VALUE = 0.5
    OVERSAMPLE = False
    OVERSAMPLE_THRESHOLD = 60
    WANDB = True

cfg = Config()
