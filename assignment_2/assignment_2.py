# Kim Lillé and Lukas Einler Larsson
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn

class spamDetection:
    def __init__(self):
        df = pd.DataFrame()
        trainingSet = pd.DataFrame()
        return
    def readData(self, data = "spambase/spambase.data"):
        self.df = pd.read_csv(data, header=None, dtype= np.float64)
        return
    def setHeader(self, headerSrc = "spambase/spambase.header"):
        header = ""
        with open(headerSrc) as f:
            header = f.readlines()
        for i, _ in enumerate(header):
            header[i] = header[i].strip()
            header[i] = header[i].strip(',')
        self.df.columns = header
        return
    def clean(self):
        self.df.dropna()
        self.df = self.df.drop_duplicates()

    def datasetCreation(self):
        spamDict = {}
        spamDict["frame"] = self.df
        data = []
        target = []
        for i in range(len(self.df)):
            instance = self.df.iloc[i].to_list()
            data.append(instance[:-1])
            target.append(instance[-1])
        spamDict["data"] = data
        spamDict["target"] = target
        spamDict["target_names"] = ["spam", "ham"]
        spamDict["feature_names"] = self.df.columns.to_list()[:-1]
        # self.spamDict = spamDict
        return spamDict


def main():
    spamDe = spamDetection()
    spamDe.readData()
    spamDe.setHeader()
    spamDe.clean()
    dict = spamDe.datasetCreation()
    print(dict["feature_names"])

if __name__ == '__main__':
    main()
