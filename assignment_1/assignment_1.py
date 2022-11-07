#https://github.com/sampepose/SpamClassifier
#https://rstudio-pubs-static.s3.amazonaws.com/89589_4d2a66ce5276434c9f56a54df5739e85.html
#https://www2.stat.duke.edu/courses/Spring06/sta293.3/topic9/spam_name.txt
#https://en.wikipedia.org/wiki/Receiver_operating_characteristic
#https://en.wikipedia.org/wiki/Confusion_matrix

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer


class spamDetection:
    def __init__(self):
        df = pd.DataFrame()
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

    def plotData(self):
        """
            plots data
        """
        plt.plot(self.df)
        plt.show()

    def transformData(self):
        """
            Transforms the data
        """
        # trans = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
        # self.df.iloc[:,:-4] = trans.fit_transform(self.df.iloc[:,:-4])
        # intervalDict = {"a": (0, 0.2), "b": (0.2, 0.4), "c": (0.4, 0.6), "d": (0.6, 0.8), "e": (0.8, 1)}
        for i in range(1679):#len(self.df)):
            for j in range(len(self.df.iloc[i]-3)):
                if self.df.iloc[i][j] > 0 and self.df.iloc[i][j] < 0.2:
                    self.df.iloc[i][j] = 100.0 # first group
                if self.df.iloc[i][j] > 0.2 and self.df.iloc[i][j] < 0.4:
                    self.df.iloc[i][j] = 200.0 # second group
                if self.df.iloc[i][j] > 0.4 and self.df.iloc[i][j] < 0.6:
                    self.df.iloc[i][j] = 300.0 # third group
                if self.df.iloc[i][j] > 0.6 and self.df.iloc[i][j] < 0.8:
                    self.df.iloc[i][j] = 400.0 # fourth group
                if self.df.iloc[i][j] > 0.8 and self.df.iloc[i][j] < 1:
                    self.df.iloc[i][j] = 500.0 # fifth group

    def LGG_Set(self, D):
        """ s. 108 in the course book
            Input: Data D,
            Output Logical expression H
        """
        x = D.iloc[0].to_list()
        H = x
        for i in range(1, 1679): #len(D)):
            x = D.iloc[i].to_list()
            H = self.LGG_Conj(H, x)
        return H

    def LGG_Conj(self, H, x):
        """ s. 110 in the course book
            input: conjunctions H, x
            output: conjunction z
        """
        z = x
        for i in range(len(x)):
            if H[i] == "?": # feature already considered general
                continue
            if x[i] != H[i]: # no conjunction
                z[i] = "?" # feature is general
        return z

def main():
    spamDe = spamDetection()
    spamDe.readData()
    spamDe.setHeader()
    spamDe.clean()
    spamDe.transformData()
    # spamDe.plotData()
    print(spamDe.LGG_Set(spamDe.df))

if __name__ == '__main__':
    main()
