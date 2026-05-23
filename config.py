VIDEO_PATH = "data/QAvideo1.mp4"  # set to a video path, e.g. "data/video3.mp4"; None uses IMAGE_PATH
IMAGE_PATH = "data/dog.png"  # set to an RGB image path, e.g. "data/dog.png"; None uses VIDEO_PATH
CHECKPOINT_PATH = "frequency_jepa_sr.pth"
PHASE1_CHECKPOINT_PATH = "phase1_jepa.pth"
PHASE1_READY_FOR_PHASE2_PATH = "jepa_ready_for_phase2.pth"
INFERENCE_OUTPUT_PATH = "final_reconstruction.jpg"

# Used only when frequency_jepa_sr.pth does not exist.
PHASE1_CHECKPOINT_CANDIDATES = [
    PHASE1_READY_FOR_PHASE2_PATH,
    PHASE1_CHECKPOINT_PATH,
]

# Every Run All adds this many more phase-1 epochs before phase 2.
PHASE1_ADDITIONAL_EPOCHS = 1
PHASE1_LEARNING_RATE = 1e-4
PHASE1_WEIGHT_DECAY = 1e-4
PHASE1_EMA_DECAY = 0.996
PHASE1_MAX_STEPS_PER_EPOCH = None  # set to an int for quick tests, e.g. 20

# Every Run All adds this many more phase-2 epochs.
ADDITIONAL_EPOCHS = 0
BATCH_SIZE = 16
MAX_STEPS_PER_EPOCH = None  # set to an int for quick tests, e.g. 20
SAVE_EVERY_STEPS = 100
LEARNING_RATE = 1e-3
TARGET_HW = (256, 256)
RADIUS_RATIO = 0.25
LATENT_DIM = 256
PREDICTOR_HIDDEN_DIM = 512
USE_PRETRAINED_BACKBONE = False
ARCHITECTURE_VERSION = "mobilenet_2ch_phase2_v1"