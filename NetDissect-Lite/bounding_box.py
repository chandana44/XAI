class BoundingBox:
    def __init__(self, x1, y1, x2, y2, num_indexes, word, maxheat=0, avgheat=0, color = 'black', layer=0, unit=0, unit_IoU_score=0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.num_indexes = num_indexes
        self.word = word
        self.maxheat = maxheat
        self.avgheat = avgheat
        self.color = color
        self.layer = layer
        self.unit = unit
        self.unit_IoU_score = unit_IoU_score