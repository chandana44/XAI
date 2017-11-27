import json
from utils import *
import random
from scipy.misc import imread, imresize


def calculateHeatParams(img_path, box):
    x_start = int(box[0])
    y_start = int(box[1])
    width = int(box[2])
    height = int(box[3])

    maxHeat = 0
    totalHeat = 0

    img = imread(img_path)

    for x in range(x_start, x_start + width):
        for y in range(y_start, y_start + height):
            if (x < 0 or x > 223 or y < 0 or y > 223):
                continue
            redValue = img[x, y]
            totalHeat += redValue
            if redValue > maxHeat:
                maxHeat = redValue

    avgHeat = totalHeat / (width * height)

    return maxHeat, avgHeat


def getExplanationWords(jsonFile, heatmap, image, question, imagename, answer):
    with open(jsonFile) as data_file:
        data = json.load(data_file)

    scores = data["results"][0]["scores"]
    captions = data["results"][0]["captions"]
    boxes = data["results"][0]["boxes"]
    lambda_1 = 0.6  # co-efficient for max heat
    lambda_2 = 0.4  # co-efficient for average heat
    sentencesBeam = Beam(5)

    for i in range(0, len(boxes)):
        maxHeat, avgHeat = calculateHeatParams(heatmap, boxes[i])
        heatMetric = lambda_1 * maxHeat + lambda_2 * avgHeat
        sentencesBeam.add(captionBox(captions[i], boxes[i]), heatMetric)

    # plotBoxes(boxes)
    print repr(sentencesBeam)
    finalboxes = [e.box for e in sentencesBeam.get_elts()]
    final_captions = [e.caption for e in sentencesBeam.get_elts()]
    question_words = question.split(' ')
    question_tag = question_words[0] + '_' + question_words[1] + '_'
    plotBoxes(heatmap, finalboxes, final_captions, 'results/' + question_tag + imagename+ '_heatmap.png', question, answer)
    plotBoxes(image, finalboxes, final_captions, 'results/' + question_tag + imagename+ '_output.png', question, answer)


def plotBoxes(heatmap, boxes, captions, filename, question, answer):
    import matplotlib as mpl
    mpl.use('Agg')

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image, ImageFont, ImageDraw, ImageColor
    import numpy as np

    colors = ['red', 'blue', 'green', 'cyan', 'magenta']

    im = np.array(imresize(Image.open(heatmap), (224, 224)), dtype=np.uint8)
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    for (box, color) in zip(boxes, colors):
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor=color, facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.savefig(filename)
    plt.close()

    img = Image.open(filename)
    w, h = img.size
    img = img.crop((0, 0, w, h + h/4))

    width, height = img.size
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, h, width, height), fill='white')
    font = ImageFont.truetype("fonts/SSbold.ttf", 18)
    text_w, text_h = draw.textsize("Chandana", font)

    i = 8
    #(width - text_w) // 2
    draw.text((200, height - i* text_h), question + '  ' + answer, ImageColor.getrgb('black'), font=font)
    i -= 1
    for (box, caption, color) in zip(boxes, captions, colors):
        #draw.text((box[1], box[0]), caption, ImageColor.getrgb(color), font=font)
        draw.text((200, height - i* text_h), caption, ImageColor.getrgb(color), font=font)
        i -= 1
    img.save(filename)


if __name__ == '__main__':
    getExplanationWords('results.json', 'image.jpg')
