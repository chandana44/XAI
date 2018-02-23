import torch
import torchvision
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.transforms as transforms
import collections
from map_units import *
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw, ImageColor
from scipy.misc import imread, imresize, imsave
import settings

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATA_FOLDER = 'data/'
MODEL_FOLDER = DATA_FOLDER + settings.MODEL + '/'
MODEL_FILE = MODEL_FOLDER + settings.TRAINED_MODEL

features_blobs = []


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


def load_model(hook_fn):  ##hook_fn copies the layer output into a numpy array
    if MODEL_FILE is None:
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
    else:
        checkpoint = torch.load(MODEL_FILE)
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
            if settings.MODEL_PARALLEL:
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            model = checkpoint

    # print model._modules
    for name in settings.LAYERS_NAMES:
        model._modules.get(name).register_forward_hook(hook_fn)  ##extracts features of an image
    if settings.GPU:
        model.cuda()
    model.eval()  # test
    return model


def get_units_names(feature_blobs, layer):
    maxfeatures = np.max(np.max(feature_blobs, 3), 2)
    activations = maxfeatures[0]
    units = sorted(range(len(activations)), key=lambda i: activations[i])[-3:]
    mapunits = MapUnits(MODEL_FOLDER, layer)
    units_map = mapunits.getunitsmap()
    names = [units_map[unit + 1]['label'] for unit in units]
    print units, names
    return units, names

def get_bounding_box(layer, units, unit_num, color, image_shape):
    unit = units[layer][unit_num]
    mask = imresize(features_blobs[layer][0][unit], image_shape, mode='F')
    thresholds = np.load(MODEL_FOLDER + settings.LAYERS_NAMES[layer] + '_quantile.npy')
    indexes = np.argwhere(mask > thresholds[unit])
    if indexes.size == 0:
        return patches.Rectangle((0, 0), 0, 0, linewidth=2, edgecolor=color, facecolor='none')
    else:
        min = indexes.min(0)
        x1 = min[0]
        y1 = min[1]
        max = indexes.max(0)
        x2 = max[0]
        y2 = max[1]
        rect = patches.Rectangle((y1, x1), y2 - y1, x2 - x1, linewidth=2, edgecolor=color, facecolor='none')
        # ellipse = patches.Ellipse(((y1+y2)/2, ((x1+x2)/2)), y2-y1+2, x2-x1+2, linewidth=1, edgecolor= color, facecolor='none')
        return rect

model = load_model(hook_feature)

# pass the image through the model
scaler = transforms.Scale((settings.IMG_SIZE, settings.IMG_SIZE))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

colors = ['red', 'blue', 'green', 'cyan']
IMAGES_DIR = 'val_images/'
if not os.path.exists(settings.OUTPUT_FOLDER):
    os.makedirs(settings.OUTPUT_FOLDER)

for imagename in os.listdir(IMAGES_DIR):
    imagefile = IMAGES_DIR + imagename
    image_grounded = settings.OUTPUT_FOLDER + imagename

    image = Image.open(imagefile)
    image_var = Variable(normalize(to_tensor(scaler(image))).unsqueeze(0)).cuda()
    logit = model.forward(image_var)
    print imagename

    units = []
    words = []
    for layer_num in range(0, 5):
        layer_units, layer_words = get_units_names(features_blobs[layer_num], settings.LAYERS_NAMES[layer_num])
        units.append(layer_units)
        words.append(layer_words)

    image_shape = imread(imagefile).shape
    image_array = np.array(image, dtype=np.uint8)
    # Create figure and axes
    fig, ax = plt.subplots(1)

    word_rect_map = collections.OrderedDict()
    color_id = 0
    answer = words[4][0]
    for layer in range(2, 4):
        for topunit in range(0, 2):
            word_rect_map[words[layer][topunit]] = get_bounding_box(layer, units, topunit, colors[color_id], image_shape)
            color_id += 1

    # Display the image
    ax.imshow(image_array)
    # Add the patch to the Axes
    for word in word_rect_map.keys():
        ax.add_patch(word_rect_map[word])
    plt.axis('off')
    plt.savefig(image_grounded)
    plt.close()

    img = Image.open(image_grounded)
    w,h = img.size
    img = img.crop((0, 0, w, h + h / 16))
    width, height = img.size
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, h, width, height), fill='white')
    font = ImageFont.truetype("fonts/SSbold.ttf", 18)
    text_w, text_h = draw.textsize("Chandana", font)

    start_pixel = 100
    draw.text((start_pixel, height - 4 * text_h), answer, ImageColor.getrgb('black'), font=font)

    for (word, color) in zip(word_rect_map.keys(), colors):
        draw.text((start_pixel, height - 2 * text_h), word, ImageColor.getrgb(color), font=font)
        start_pixel += draw.textsize(word + ' ', font = font)[0]
        print start_pixel
    img.save(image_grounded)

    features_blobs = []
