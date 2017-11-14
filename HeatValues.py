import json
from utils import *
import random
from scipy.misc import imread

def calculateHeatParams(img_path, box):
    x_start = int(box[0])
    y_start = int(box[1])
    width = int(box[2])
    height = int(box[3])

    maxHeat = 0
    totalHeat = 0
    redValue = 0.0

    img = imread(img_path)

    #print 'x_start ' + str(x_start) + ' y_start ' + str(y_start) + ' width ' + str(width) + ' height ' + str(height)

    for x in range(x_start, x_start+width):
        for y in range(y_start, y_start+height):
            if(x<0 or x>223 or y<0 or y>223):
                continue
            redValue = img[x,y]
            totalHeat += redValue
            if redValue > maxHeat:
                maxHeat = redValue

    avgHeat = totalHeat/(width*height)

    return maxHeat, avgHeat

def getExplanationWords(jsonFile, heatmap):
    with open(jsonFile) as data_file:
        data = json.load(data_file)

    scores = data["results"][0]["scores"]
    captions = data["results"][0]["captions"]
    boxes = data["results"][0]["boxes"]
    lambda_1 = 0.6 #co-efficient for max heat
    lambda_2 = 0.4 #co-efficient for average heat
    sentencesBeam = Beam(5)

    for i in range(0,len(boxes)):
        maxHeat, avgHeat = calculateHeatParams(heatmap, boxes[i])
        heatMetric = lambda_1 * maxHeat + lambda_2 * avgHeat
        sentencesBeam.add(captionBox(captions[i], boxes[i]), heatMetric)

    #plotBoxes(boxes)
    print repr(sentencesBeam)
    finalboxes = [e.box for e in sentencesBeam.get_elts()]
    plotBoxes(heatmap, finalboxes)

def plotBoxes(heatmap, boxes):
    import matplotlib as mpl
    mpl.use('Agg')

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import numpy as np

    im = np.array(Image.open(heatmap), dtype=np.uint8)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    for box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.savefig('output.png')


if __name__ == '__main__':
   getExplanationWords('results.json', 'image.jpg')
