# Define dataset paths
DATASET_ROOT_DIR = '//dataset/'

VIDEOS_ROOT_DIR = DATASET_ROOT_DIR + 'RawVideos/'      
RESULTS_FOLDER = '//results/'

# Define video sampling parameters
SKIPPED_FRAMES = 24  
DURATION = 30
CAPTURED_FPS = 25
DIM_RESIZE = (224, 224)   # Input size to the neural network

# Training hyperparameters
LEARNING_RATE = 1e-5
USING_SCHEDULER = True
PATIENCE = 3
FACTOR = 0.1
USING_EARLY_STOP = True
E_STOP_PATIENCE = 5

WEIGHT_DECAY = 0  
NUM_EPOCHS = 15
N_CLASSES = 2
BATCH_SIZE = 6 
N_WORKERS = 6 
# # # DEBUG PARAMS
# BATCH_SIZE = 1
# N_WORKERS = 0
AUGMENTATION = True #False

# BACKBONE = 'convnextv2'
# BACKBONE = 'resnet18'
BACKBONE = 'resnet50'
# BACKBONE = 'resnext50_32x4d'

# Transformer hyperparameters
DIM = 64
DEPTH = 2
HEADS = 8
MLP_DIM = 128
DIM_HEAD = 64
DROPOUT = 0
EMB_DROPOUT = 0
