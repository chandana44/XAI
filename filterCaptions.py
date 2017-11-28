import os
import argparse
import subprocess
import re
from HeatValues import *
import json

if __name__ == '__main__':
    image_path_prefix = '/work/04445/camanchi/maverick/VQA/Images/Test/test2015'
    gradcam_path = '/work/04445/camanchi/maverick/XAI/grad-cam'
    vqaexp_path = '/work/04445/camanchi/maverick/XAI/XAI'
    densecap_path = '/work/04445/camanchi/maverick/XAI/densecap-master/'
    heatmap_path_prefix = 'output/vqa_gcam_'
    captions_json_path = 'vis/data/results.json'

    with open('sample_test_questions.json') as data_file:
        data = json.load(data_file)

    questions_size = len(data["questions"])
    print 'questions count: ' + str(questions_size)
    questions_all = [data["questions"][id]["question"] for id in range(0, questions_size)]
    image_ids = [data["questions"][id]["image_id"] for id in range(0, questions_size)]

    for (image, question) in zip(image_ids, questions_all):
        image_filename = str(image)
        image_filename = 'COCO_test2015_' + '0' * (12 - len(image_filename)) + image_filename
        full_image_path = image_path_prefix + image_filename + '.jpg '

        # grad cam heat map
        # th visual_question_answering.lua -input_image_path <path> -question <question> -gpuid 0 -outpath <path>
        os.chdir(gradcam_path)
        result = subprocess.check_output(
            ['luajit', 'visual_question_answering.lua', '-input_image_path', full_image_path, '-question', question, '-gpuid',
             '0'])
        m = re.search("Grad-CAM answer:(.*)", result)
        answer = m.group(1).strip()

        heatmap_image = gradcam_path + heatmap_path_prefix + answer + '.png'
        print 'heat map: ' + heatmap_image

        # dense captioning
        # th run_model.lua -input_image imgs/elephant.jpg
        os.chdir(densecap_path)
        print subprocess.check_output(
            ['luajit', 'run_model.lua', '-input_image', full_image_path])

        os.chdir(vqaexp_path)
        captions_json = densecap_path + captions_json_path
        getExplanationWords(captions_json, heatmap_image, image, question, image_filename, answer)
