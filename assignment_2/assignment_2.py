# Kim Lillé and Lukas Einler Larsson
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# TODO:
# 1. välja 3 algoritmer - KNN, SVM, Naive bayes
# 2. ten fold test
# 3. Skapa ett table som figur 12.4, SIDA 350
# 4. Kör ett friedman test, figur 12.8, SIDA 355
# 5. om det är signifikant differens, köra Nemeyi test SIDA 356

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
        spamDict["data"] = np.array(data)
        spamDict["target"] = np.array(target)
        spamDict["target_names"] = np.array(["spam", "ham"])
        spamDict["feature_names"] = np.array(self.df.columns.to_list()[:-1])
        self.spamDict = spamDict

    def run_kNNclassifier(self, n_neighbors = 5):
        X, y = self.spamDict["data"], self.spamDict["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print(X_train.shape, X_test.shape)
        knn = KNeighborsClassifier(n_neighbors)
        knn.fit(X_train, y_train)
        print("Accuracy:", knn.score(X_test, y_test))
        y_pred = knn.predict(X_test)
        print("{0} / {1} correct".format(np.sum(y_test == y_pred), len(y_test)))
        print(confusion_matrix(y_test, y_pred))

def main():
    spamDe = spamDetection()
    spamDe.readData()
    spamDe.setHeader()
    spamDe.clean()
    spamDe.datasetCreation()
    spamDe.run_kNNclassifier()


if __name__ == '__main__':
    main()
