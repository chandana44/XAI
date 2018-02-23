import csv

class MapUnits:
    def __init__(self, data_dir, layer):
        self.data_dir = data_dir
        self.layer = layer
        self.units_map = {}

    def getunitsmap(self):
        file_map = csv.DictReader(open(self.data_dir + self.layer + '_tally.csv'))
        for row in file_map:
            self.units_map[int(row['unit'])] = row
        return self.units_map