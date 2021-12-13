import time
import pickle
import os


class DataSaver():
    data = []
    savingPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/data_{}.pickle'.format(int(time.time())))

    def addData(self, img, palm, hand, label):
        self.data.append((img, palm, hand, int(label)))

        with open(self.savingPath, 'wb') as f:
            pickle.dump(self.data, f)