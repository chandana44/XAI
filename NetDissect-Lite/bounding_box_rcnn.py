class BoundingBox:
    def __init__(self, x1, y1, x2, y2, word, maxheat=0, avgheat=0, color = 'black', det_score=0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.word = word
        self.maxheat = maxheat
        self.avgheat = avgheat
        self.color = color
        self.det_score = det_score