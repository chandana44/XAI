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
from bounding_box import BoundingBox
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
    #print units, names
    return units, names

def get_bounding_box(layer, units, unit_num, image_shape):
    unit = units[layer][unit_num]
    mask = imresize(features_blobs[layer][0][unit], image_shape, mode='F')
    thresholds = np.load(MODEL_FOLDER + settings.LAYERS_NAMES[layer] + '_quantile.npy')
    indexes = np.argwhere(mask > thresholds[unit])
    if indexes.size == 0:
        return 0, 0, 0, 0
    else:
        min = indexes.min(0)
        x1 = min[0]
        y1 = min[1]
        max = indexes.max(0)
        x2 = max[0]
        y2 = max[1]
        return x1, y1, x2, y2
        #rect = patches.Rectangle((y1, x1), y2 - y1, x2 - x1, linewidth=2, edgecolor=color, facecolor='none')
        # ellipse = patches.Ellipse(((y1+y2)/2, ((x1+x2)/2)), y2-y1+2, x2-x1+2, linewidth=1, edgecolor= color, facecolor='none')
        #return rect

model = load_model(hook_feature)

# pass the image through the model
scaler = transforms.Scale((settings.IMG_SIZE, settings.IMG_SIZE))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

colors = ['red', 'blue', 'green', 'cyan']
IMAGES_DIR = 'sample_test_images/'
if not os.path.exists(settings.OUTPUT_FOLDER):
    os.makedirs(settings.OUTPUT_FOLDER)

for imagename in os.listdir(IMAGES_DIR):
#for imagename in ['Places365_test_00000689.jpg']:
    print imagename
    imagefile = IMAGES_DIR + imagename
    image_grounded = settings.OUTPUT_FOLDER + imagename

    image = Image.open(imagefile)
    image_var = Variable(normalize(to_tensor(scaler(image))).unsqueeze(0)).cuda()
    logit = model.forward(image_var)

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
    color_id = -1
    answer = words[4][0]
    for layer in range(2, 4):
        for topunit in range(0, 2):
            #print words[layer][topunit]
            if not words[layer][topunit] in word_rect_map:
                word_rect_map[words[layer][topunit]] = []
                color_id +=1
                color = colors[color_id]
            else:
                color = word_rect_map[words[layer][topunit]][0].color
            x1, y1, x2, y2 = get_bounding_box(layer, units, topunit, image_shape)
            #print x1, y1, x2, y2
            r = BoundingBox(x1, y1, x2, y2, color)
            word_rect_map[words[layer][topunit]].append(r)

    # Display the image
    ax.imshow(image_array)
    # Add the patch to the Axes
    for value in word_rect_map.values():
        for r in value:
            ax.add_patch(patches.Rectangle((r.y1, r.x1), r.y2 - r.y1, r.x2 - r.x1, linewidth=2, edgecolor=r.color, facecolor = 'None'))
    #plt.axis('off')
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

    dist_from_boundary = 50

    def write_text(pixel_w, line, text, color):
        text_size = draw.textsize(text, font=font)[0]
        if pixel_w + text_size > width-dist_from_boundary:
            pixel_w = dist_from_boundary
            line -= 1
        draw.text((pixel_w, height - line * text_h), text, ImageColor.getrgb(color), font=font)
        pixel_w += draw.textsize(text, font=font)[0]
        return pixel_w, line


    line_num = 3
    start_pixel_w = dist_from_boundary
    text = 'This is ' + answer + ' because there is '
    start_pixel_w, line_num = write_text(start_pixel_w, line_num, text, 'black')

    for word_id in range(len(word_rect_map)):
        if word_id == len(word_rect_map)-1:
            text = 'and '
            start_pixel_w, line_num = write_text(start_pixel_w, line_num, text, 'black')
            text = word_rect_map.keys()[word_id]
        elif word_id == len(word_rect_map)-2:
            text = word_rect_map.keys()[word_id] + ' '
        else:
            text = word_rect_map.keys()[word_id] + ', '
        start_pixel_w, line_num = write_text(start_pixel_w, line_num, text, colors[word_id])
    img.save(image_grounded)

    features_blobs = []
