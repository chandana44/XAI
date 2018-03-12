from __future__ import division
import torch
import torchvision
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.transforms as transforms
import collections
from heatmap import get_heatmap
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
TEXTURE = 'texture'

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

    for name in settings.LAYERS_NAMES:
        model._modules.get(name).register_forward_hook(hook_fn)  ##extracts features of an image
    if settings.GPU:
        model.cuda()
    model.eval()
    return model


def get_names(layer, units):
    mapunits = MapUnits(MODEL_FOLDER, layer)
    units_map = mapunits.getunitsmap()
    names = [units_map[unit + 1]['label'] for unit in units]
    categories = [units_map[unit + 1]['category'] for unit in units]
    return names, categories


def get_units_names(feature_blobs, layer):
    maxfeatures = np.max(np.max(feature_blobs, 3), 2)
    activations = maxfeatures[0]
    units = sorted(range(len(activations)), key=lambda i: activations[i])[-6:]
    names, cats = get_names(layer, units)
    return units, names, cats


def get_unit_bounding_box(layer, unit, image_shape):
    mask = imresize(features_blobs[layer][0][unit], image_shape, mode='F')
    thresholds = np.load(MODEL_FOLDER + settings.LAYERS_NAMES[layer] + '_quantile.npy')
    indexes = np.argwhere(mask > thresholds[unit])
    if indexes.size == 0:
        indexes = np.argwhere(mask > (thresholds[unit] / 4))
        if indexes.size == 0:
            return 0, 0, 0, 0
    min = indexes.min(0)
    x1 = min[0] if min[0] != 0 else 1
    y1 = min[1] if min[1] != 0 else 1
    max = indexes.max(0)
    x2 = max[0] if max[0] != image_shape[1] else image_shape[1] - 1
    y2 = max[1] if max[1] != image_shape[0] else image_shape[0] - 1
    return indexes, x1, y1, x2, y2


def check_heat_box(heatmap, x1, x2, y1, y2):
    maxHeat = 0
    totalHeat = 0

    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            redValue = heatmap[x][y][2]
            totalHeat += redValue
            if redValue > maxHeat:
                maxHeat = redValue

    avgHeat = totalHeat / ((x2 - x1 + 1) * (y2 - y1 + 1))
    print maxHeat, avgHeat
    if maxHeat == 255 and avgHeat > 200:
        return True
    else:
        return False


def check_heat_indexes(heatmap, indexes):
    maxHeat = 0
    totalHeat = 0

    for index in indexes:
        redValue = heatmap[index[0]][index[1]][2]
        totalHeat += redValue
        if redValue > maxHeat:
            maxHeat = redValue

    avgHeat = totalHeat / len(indexes)
    # print maxHeat, avgHeat
    return maxHeat, avgHeat
    # if maxHeat == 255 and avgHeat > 200:
    #     return True
    # else:
    #     return False


if __name__ == '__main__':

    # pass the image through the model
    scaler = transforms.Scale((settings.IMG_SIZE, settings.IMG_SIZE))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    # load the class label
    file_name = 'categories_places365.txt'
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    if not os.path.exists(settings.OUTPUT_FOLDER):
        os.makedirs(settings.OUTPUT_FOLDER)

    colors = ['red', 'blue', 'green', 'cyan', 'teal', 'pink', 'orange', 'lightgreen', 'magenta', 'yellow', 'maroon',
              'brown', 'gray', 'gold', 'darkblue', 'lavender', 'khaki', 'chocolate']
    frequent_words = ['painting', 'hair', 'airport_terminal', 'eye']
    IMAGES_DIR = 'sample_val_images/'

    model = load_model(hook_feature)

    for imagename in os.listdir(IMAGES_DIR):
    #for imagename in ['Places365_val_00000200.jpg']:
        print 'image: ' + imagename
        imagefile = IMAGES_DIR + imagename
        heatmap = get_heatmap(imagefile, model)  # width*height*3
        image_grounded = settings.OUTPUT_FOLDER + imagename

        image = Image.open(imagefile)
        image_var = Variable(normalize(to_tensor(scaler(image))).unsqueeze(0)).cuda()
        logit = model.forward(image_var)
        logit_array = logit.data.cpu().numpy()
        class_predicted = np.argmax(logit_array)
        answer = classes[class_predicted]
        image_shape = imread(imagefile).shape

        word_rect_map = collections.OrderedDict()
        color_id = -1


        def layer_boxes(layer, units, names, categories, map, color_id):
            #print 'layer : ' + str(layer)
            #count = 0
            boxes = []
            for index in range(len(names)):
                word = names[index]
                if word.endswith('-s') or word.endswith('-c'):
                    word = word[:-2]
                category = categories[index]
                if category == TEXTURE or word.split('-')[0] in answer or answer.split('/')[
                    0] in word or word in frequent_words:
                    continue
                for character in ['_', '-', '/']:
                    if character in word:
                        word.replace(character, ' ')
                # print word
                unit = units[index]
                indexes, x1, y1, x2, y2 = get_unit_bounding_box(layer, unit, image_shape)
                if x1 + x2 + y1 + y2 == 0 or (
                                x1 == 1 and y1 == 1 and x2 == image_shape[1] - 1 and y2 == image_shape[0] - 1):
                    continue
                # print word
                # print x1, x2, y1, y2

                maxheat, avgheat = check_heat_indexes(heatmap, indexes)

                boxes.append(BoundingBox(x1, y1, x2, y2, len(indexes), word, maxheat, avgheat, 'black', layer, unit))

            boxes.sort(key=lambda x: x.maxheat + x.avgheat + (int(x.num_indexes/(image_shape[0]*image_shape[1])) * 200), reverse=True)
            for box in boxes:
                #print box.word
                #print int((box.num_indexes/(image_shape[0]*image_shape[1])) * 200)
                if not box.maxheat > 230 or (int((box.num_indexes/(image_shape[0]*image_shape[1])) * 200)< 5):
                    continue
                if box.avgheat < 200:
                    break
                if not box.word in map:
                    map[box.word] = []
                    #map[box.word + '-' + str(box.layer+1) + '-' + str(box.unit+1)] = []
                    color_id += 1
                    box.color = colors[color_id]
                else:
                    box.color = map[box.word][0].color
                map[box.word].append(box)
                #map[box.word + '-' + str(box.layer + 1) + '-' + str(box.unit + 1)].append(box)

            return color_id


        units = []
        words = []
        categories = []
        for layer_num in range(2, 5):
            layer_units, layer_words, layer_cats = get_units_names(features_blobs[layer_num],
                                                                   settings.LAYERS_NAMES[layer_num])
            color_id = layer_boxes(layer_num, layer_units, layer_words, layer_cats, word_rect_map, color_id)

        image_array = np.array(image, dtype=np.uint8)
        # Create figure and axes
        fig, ax = plt.subplots(1)
        # Display the image
        ax.imshow(image_array)
        # Add the patch to the Axes
        for value in word_rect_map.values():
            for r in value:
                ax.add_patch(patches.Rectangle((r.y1, r.x1), r.y2 - r.y1, r.x2 - r.x1, linewidth=2, edgecolor=r.color,
                                               facecolor='None'))
        plt.axis('off')
        plt.savefig(image_grounded)
        plt.close()

        img = Image.open(image_grounded)
        w, h = img.size
        img = img.crop((0, 0, w, h + h / 16))
        width, height = img.size
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, h, width, height), fill='white')
        ssbold_font = ImageFont.truetype("fonts/SSbold.ttf", 18)
        answer_font = ImageFont.truetype("fonts/answer.ttf", 18)
        text_w, text_h = draw.textsize("Chandana", ssbold_font)

        dist_from_boundary = 50


        def write_text(pixel_w, line, text, color, font=ssbold_font):
            text_size = draw.textsize(text, font=font)[0]
            if pixel_w + text_size > width - dist_from_boundary:
                pixel_w = dist_from_boundary
                line -= 1
            draw.text((pixel_w, height - line * text_h), text, ImageColor.getrgb(color), font=font)
            pixel_w += draw.textsize(text, font=font)[0]
            return pixel_w, line


        if len(word_rect_map) != 0:
            line_num = 3
            start_pixel_w = dist_from_boundary
            text = 'This is '
            start_pixel_w, line_num = write_text(start_pixel_w, line_num, text, 'black')
            text = answer
            start_pixel_w, line_num = write_text(start_pixel_w, line_num, text, 'black', answer_font)
            text = ' because there is '
            start_pixel_w, line_num = write_text(start_pixel_w, line_num, text, 'black')

            for word_id in range(len(word_rect_map)):
                if word_id == len(word_rect_map) - 1 and word_id != 0:
                    text = 'and '
                    start_pixel_w, line_num = write_text(start_pixel_w, line_num, text, 'black')
                    text = word_rect_map.keys()[word_id]
                elif word_id == len(word_rect_map) - 2 or ((word_id == len(word_rect_map) - 1) and word_id == 0):
                    text = word_rect_map.keys()[word_id] + ' '
                else:
                    text = word_rect_map.keys()[word_id] + ', '
                start_pixel_w, line_num = write_text(start_pixel_w, line_num, text, colors[word_id])
        img.save(image_grounded)

        features_blobs = []
