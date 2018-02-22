import os, sys
import argparse
import subprocess
import re
from HeatValues import *

# python captionSentences.py --image COCO_train2014_000000000009.jpg --question "how many cookies are there?"
# python captionSentences.py --image COCO_train2014_000000000025.jpg --question "how many giraffes are there?"
# python captionSentences.py --image COCO_train2014_000000000036.jpg --question "what color is the umbrella"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, dest="image", help='path to the image')
    parser.add_argument('--question', required=True, dest="question", help='question on the image')
    args = parser.parse_args()

    print 'image: ' + args.image
    print 'question: ' + args.question

    image_path_prefix = '../data/Sample_Images/'
    gradcam_path = '../grad-cam/'
    vqaexp_path = '../XAI/'
    densecap_path = '../densecap-master/'
    heatmap_path_prefix = 'output/vqa_gcam_'
    captions_json_path = 'vis/data/results.json'
    image = image_path_prefix + args.image

    # grad cam heat map
    # th visual_question_answering.lua -input_image_path <path> -question <question> -gpuid 0 -outpath <path>
    os.chdir(gradcam_path)
    result = subprocess.check_output(
        ['luajit', 'visual_question_answering.lua', '-input_image_path', image, '-question', args.question, '-gpuid',
         '0'])
    m = re.search("Grad-CAM answer:(.*)", result)
    answer = m.group(1).strip()

    heatmap_image = gradcam_path + heatmap_path_prefix + answer + '.png'
    print 'heat map: ' + heatmap_image

    os.chdir(densecap_path)
    # dense captioning
    # th run_model.lua -input_image imgs/elephant.jpg
    print subprocess.check_output(
        ['luajit', 'run_model.lua', '-input_image', image_path_prefix + args.image])

    os.chdir(vqaexp_path)
    captions_json = densecap_path + captions_json_path
    getExplanationWords(captions_json, heatmap_image, image, args.question, args.image, answer)
