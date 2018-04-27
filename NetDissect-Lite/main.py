from __future__ import division
import argparse
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
from rcnn_model import get_hot_objects
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


def load_model(hook_fn):
    if MODEL_FILE is None:
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
    else:
        checkpoint = torch.load(MODEL_FILE)
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
            if settings.MODEL_PARALLEL:
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                    'state_dict'].items()}  #
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            model = checkpoint

    for name in settings.LAYERS_NAMES:
        model._modules.get(name).register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model


def get_names(layer, units):
    mapunits = MapUnits(MODEL_FOLDER, layer)
    units_map = mapunits.getunitsmap()
    names = [units_map[unit + 1]['label'] for unit in units]
    categories = [units_map[unit + 1]['category'] for unit in units]
    scores = [units_map[unit + 1]['score'] for unit in units]
    return names, categories, scores


def get_units_names_from_weights(weights, class_predicted, feature_blobs, layer):
    fc_activation = feature_blobs.copy()
    fc_activation.resize((2048,))
    w_x = np.multiply(weights[class_predicted], fc_activation)
    units = w_x.argsort()[-3:][::-1]
    # print 'top units as per weights: '
    # print units
    # return units
    names, cats, scores = get_names(layer, units)
    return units, names, cats, scores


def get_units_names(feature_blobs, layer):
    maxfeatures = np.sum(np.sum(feature_blobs, 3), 2)
    activations = maxfeatures[0]
    units = sorted(range(len(activations)), key=lambda i: activations[i])[-6:]
    names, cats, scores = get_names(layer, units)
    return units, names, cats, scores


def get_unit_bounding_box(layer, unit, image_shape):
    mask = imresize(features_blobs[layer][0][unit], image_shape, mode='F')
    thresholds = np.load(MODEL_FOLDER + settings.LAYERS_NAMES[layer] + '_quantile.npy')
    indexes = np.argwhere(mask > thresholds[unit])
    if indexes.size == 0:
        return indexes, 0, 0, 0, 0
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
    if maxHeat > 230 and avgHeat > 200:
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
    return maxHeat, avgHeat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_heatmap', default=True, help='Use heatmap to filter words')
    args = parser.parse_args()

    OUTPUT_FOLDER = "results/" + settings.MODEL + "/"

    # if args.use_heatmap:
    #     OUTPUT_FOLDER = "results/"+ settings.MODEL +"/dissect_gradcam_model/"
    # else:
    #     OUTPUT_FOLDER = "results/"+ settings.MODEL +"/dissect_model/"

    # pass the image through the model
    scaler = transforms.Scale((settings.IMG_SIZE, settings.IMG_SIZE))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    # load the class label
    file_name = 'data/categories_places365.txt'
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    colors = ['red', 'blue', 'green', 'cyan', 'teal', 'pink', 'orange', 'lightgreen', 'magenta', 'yellow', 'maroon',
              'brown', 'gray', 'gold', 'darkblue', 'lavender', 'khaki', 'chocolate']
    IMAGES_DIR = 'sample_val_images/'

    model = load_model(hook_feature)

    Iou_thresholds = [0.01, 0.02, 0.04, 0.04]
    nocaption_count = 0

    weights = model.fc.weight.data.cpu().numpy()

    for imagename in os.listdir(IMAGES_DIR):
    #for imagename in ['Places365_val_00024170.jpg']:
        print 'image: ' + imagename
        imagefile = IMAGES_DIR + imagename
        if args.use_heatmap:
            image_grounded = OUTPUT_FOLDER + imagename.split('.')[0] + '_' + settings.MODEL + '_nd_rc_gradcam.jpg'
        else:
            image_grounded = OUTPUT_FOLDER + imagename.split('.')[0] + '_' + settings.MODEL + '_dissect.jpg'

        heatmap = get_heatmap(imagefile, model, OUTPUT_FOLDER)  # width*height*3

        image = Image.open(imagefile)
        image_var = Variable(normalize(to_tensor(scaler(image))).unsqueeze(0)).cuda()
        logit = model.forward(image_var)
        logit_array = logit.data.cpu().numpy()
        class_predicted = np.argmax(logit_array)
        answer = classes[class_predicted]
        # print 'answer: ' + answer
        for character in ['_', '-', '/']:
            if character in answer:
                answer = answer.replace(character, ' ')
        answer_firstword = answer.split(' ')[0]

        image_shape = imread(imagefile).shape

        word_rect_map = collections.OrderedDict()
        color_id = -1


        def layer_boxes(layer, units, names, categories, scores, map, color_id):
            boxes = []
            count = 0
            for index in range(len(names)):
                word = names[index]
                if word.endswith('-s') or word.endswith('-c'):
                    word = word[:-2]
                category = categories[index]
                for character in ['_', '-', '/']:
                    if character in word:
                        word = word.replace(character, ' ')
                if category == TEXTURE or answer == word:
                    # or float(scores[index]) < Iou_thresholds[layer]:
                    continue

                unit = units[index]
                indexes, x1, y1, x2, y2 = get_unit_bounding_box(layer, unit, image_shape)
                if x1 + x2 + y1 + y2 == 0:
                    continue

                boxes.append(BoundingBox(x1, y1, x2, y2, 0, word, 0, 0, 'black', 0, 0, scores[index]))
                count += 1
                # if count == 3:
                #     break

            boxes.sort(key=lambda x: x.unit_IoU_score, reverse=True)
            for box in boxes[:9]:
                keyword = box.word
                if not keyword in map:
                    map[keyword] = []
                    color_id += 1
                    box.color = colors[color_id]
                else:
                    box.color = map[keyword][0].color
                map[keyword].append(box)

            return color_id


        def layer_boxes_with_heat(layer, units, names, categories, scores, map, color_id):
            boxes = []
            for index in range(len(names)):
                word = names[index]
                # print word
                if word.endswith('-s') or word.endswith('-c'):
                    word = word[:-2]
                category = categories[index]
                for character in ['_', '-', '/']:
                    if character in word:
                        word = word.replace(character, ' ')
                if category == TEXTURE or answer == word:
                    # or float(scores[index]) < Iou_thresholds[layer]:
                    continue

                unit = units[index]
                indexes, x1, y1, x2, y2 = get_unit_bounding_box(layer, unit, image_shape)
                if x1 + x2 + y1 + y2 == 0:
                    continue

                maxheat, avgheat = check_heat_indexes(heatmap, indexes)
                boxes.append(BoundingBox(x1, y1, x2, y2, len(indexes), word, maxheat, avgheat, 'black', layer, unit,
                                         scores[index]))

            boxes.sort(
                key=lambda x: x.maxheat + x.avgheat + (
                int(x.num_indexes * 200 / (image_shape[0] * image_shape[1]))) + int(float(x.unit_IoU_score) * 1000),
                reverse=True)

            # for box in boxes:
            #     print box.word

            for box in boxes:
                if box.maxheat < 230:  # or (int((box.num_indexes / (image_shape[0] * image_shape[1])) * 200) < 5):
                    continue
                if box.avgheat < 170:
                    continue
                if (int(box.num_indexes * 200 / (image_shape[0] * image_shape[1])) == 0):
                    continue
                keyword = box.word
                # print box.word + ' adding in map'
                if not keyword in map:
                    map[keyword] = []
                    color_id += 1
                    box.color = colors[color_id]
                else:
                    box.color = map[keyword][0].color
                map[keyword].append(box)

            return color_id


        units = []
        words = []
        categories = []
        for layer_num in range(1, 2):
            layer_units, layer_words, layer_cats, layer_scores = get_units_names(features_blobs[layer_num],
                                                                                 settings.LAYERS_NAMES[layer_num])
            #layer_units, layer_words, layer_cats, layer_scores = get_units_names_from_weights(weights, class_predicted,
                                                                                              # features_blobs[layer_num],
                                                                                              # settings.LAYERS_NAMES[
                                                                                              #     layer_num])
            # print 'layer: ' + str(layer_num)
            # print layer_units
            # print layer_words
            if args.use_heatmap:
                color_id = layer_boxes_with_heat(layer_num, layer_units, layer_words, layer_cats, layer_scores,
                                                 word_rect_map,
                                                 color_id)
            else:
                color_id = layer_boxes(layer_num, layer_units, layer_words, layer_cats, layer_scores, word_rect_map,
                                       color_id)

        word_rect_map = get_hot_objects(imagefile, heatmap, word_rect_map, color_id)

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
        plt.savefig(image_grounded, bbox_inches='tight')
        plt.close()

        img = Image.open(image_grounded)
        w, h = img.size
        img = img.crop((0, 0, w, h + h / 16))
        width, height = img.size
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, h, width, height), fill='white')
        ssbold_font = ImageFont.truetype("fonts/SSbold.ttf", 12)
        answer_font = ImageFont.truetype("fonts/answer.ttf", 12)
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
        else:
            nocaption_count += 1
        img.save(image_grounded)

        features_blobs = []

    print 'no caption images: ' + str(nocaption_count)
