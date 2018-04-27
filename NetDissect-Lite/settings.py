######### global settings  #########
GPU = True                                  # running on GPU is highly suggested
MODEL = 'resnet50'                          # model arch: resnet18, alexnet, resnet50, densenet161
DATASET = 'places365'                       # model trained on: places365 or imagenet
OUTPUT_FOLDER = "results/" + MODEL + '/'        # result will be stored in this folder

########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# FEATURE_NAMES: the array of layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own

if MODEL != 'alexnet':
    IMG_SIZE = 224
else:
    IMG_SIZE = 227

if DATASET == 'places365':
    NUM_CLASSES = 365
elif DATASET == 'imagenet':
    NUM_CLASSES = 1000

if MODEL == 'resnet18':
    LAYERS_NAMES = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
    if DATASET == 'places365':
        TRAINED_MODEL = 'resnet18_places365.pth.tar'
        MODEL_PARALLEL = True
elif MODEL == 'resnet50':
    LAYERS_NAMES = ['layer3', 'layer4']
    if DATASET == 'places365':
        TRAINED_MODEL = 'resnet50_places365.pth.tar'
        MODEL_PARALLEL = True
