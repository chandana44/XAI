import json
from pprint import pprint
from utils import *

with open('v2_OpenEnded_mscoco_train2014_questions.json') as data_file:
    data = json.load(data_file)

questions_size = len(data["questions"])
print 'questions count: ' + str(questions_size)
questions_all = [data["questions"][id]["question"] for id in range(0,questions_size)]
counter = Counter()

for question in questions_all:
    words = question.split(' ')
    counter.increment_count(words[0]+' '+words[1], 1)
print repr(counter)
print len(counter.keys())




