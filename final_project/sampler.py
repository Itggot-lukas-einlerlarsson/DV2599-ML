# Kim Lillé and Lukas Einler Larsson
import pandas as pd
import numpy as np

class spamDetection:
    def __init__(self):
        df = pd.DataFrame()
        return
    def readData(self, data = "dataset/creditcard.csv"):
        self.df = pd.read_csv(data, header=None, dtype= np.float64)
        return
    def setHeader(self, headerSrc = "dataset/creditcard.header"):
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
    def writeFraudCSV(self):
        print("hello")
        outFolderName = "dataset"
        with open("dataset/frauds.csv", mode='w', encoding='utf-8') as dateFile:
            fraudDF = self.df
            fraudDF = self.df.loc[self.df["Class"] == 1] # is fraud
            print(fraudDF)
            for index, instance in fraudDF.iterrows():
                print(fraudDF.iloc[index])
                # dateFile.write(instance)


def main():
    spamDe = spamDetection()
    spamDe.readData()
    spamDe.setHeader()
    # spamDe.clean()
    print("hello1")
    spamDe.writeFraudCSV()


main()
