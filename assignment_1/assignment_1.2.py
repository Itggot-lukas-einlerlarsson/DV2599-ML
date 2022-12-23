# Kim Lillé and Lukas Einler Larsson
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random

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

    def plotData(self):
        """
            plots data
        """
        fig=plt.figure()
        ax=fig.add_axes([0,0,1,1])
        avg_spam_df = self.df.loc[self.df["spam"] == 1].mean() # is spam
        avg_ham_df = self.df.loc[self.df["spam"] == 0].mean() # is ham
        ax.scatter(self.df.columns[:-4], avg_spam_df.to_list()[:-4], color='r')
        ax.scatter(self.df.columns[:-4], avg_ham_df.to_list()[:-4], color='b')
        plt.xlabel('column')
        plt.ylabel('value')
        plt.show()

    def transformData(self, data):
        """
            Transforms the data
        """
        for column in data.columns:
            if column == "spam":
                continue
            data = data.sort_values(column)
            data[column] = pd.qcut(data[column], q= 4,  duplicates = "drop", precision = 2)
        return data

    def LGG_Set(self, D):
        """ s. 108 in the course book
            Input: Data D,
            Output Logical expression H
        """
        x = D.iloc[0].to_list()
        H = x
        for i in range(1, len(D)):
            x = D.iloc[i].to_list()
            H = self.LGG_Conj(H, x)
        return H

    def LGG_Conj(self, H, x):
        """ s. 110 in the course book
            input: conjunctions H, x
            output: conjunction H
        """
        for i, e in enumerate(H):
            if i == 57:
                H[i] = "?"
            if e != "?": # feature already considered general
                if x[i] != e: # no conjunction
                    H[i] = "?" # feature is general
        return H

    def test(self):
        """
            creates, trains and tests concept learning model
        """
        H = self.LGG_Set(self.trainingSet) # get hypothesis
        indices = []
        for i,value in enumerate(H):
            if value != "?": # feature is general and not interesting
                indices.append(i)
        print("Hypothesis:", H)
        print("-"*20)
        TN = 0
        FN = 0
        TP = 0
        FP = 0
        for i in range(len(self.testSet)):
            lst = self.testSet.iloc[i].to_list()
            for j in indices:
                if not (H[j].left < lst[j] and H[j].right >= lst[j]): # if instance value is not in interval
                    if lst[57] == 0: #if ham
                        TN += 1
                    else:
                        FN += 1
                    break
                if j == indices[-1]:
                    if lst[57] == 1: #if spam
                        TP += 1
                    else:
                        FP += 1
        print("Positive indicates Spam, negative indicates Ham")
        print("-"*20)
        print("True Positive:", TP)
        print("False Positive:", FP)
        print("True Negative:", TN)
        print("False Negative:", FN)
        print("Total Amount:", TP+FP+TN+FN)
        print("-"*20)
        print("Precision:", TP / (TP + FP))
        print("Recall:", TP / (TP + FN))
        print("Accuracy:", (TP + TN) / (TP + FP + FN + TN))
        self.printConfusionMatrix(TP, FP, TN, FN)

    def createTrainingSet(self, col = "spam", val = 1, parts = 5):
        trainingSet = self.df.loc[self.df[col] == val] # 1679
        self.testSet = self.df.loc[self.df[col] == 0] # 2531
        interval = [(0, len(trainingSet)//parts)]
        for i in range(1, parts):
            interval.append((interval[i-1][1] + 1, (i + 1) * len(trainingSet)//parts))
        testIndex = random.randint(0, parts-1)
        self.testSet = pd.concat([self.testSet, trainingSet[interval[testIndex][0]:interval[testIndex][1] ]], axis=0)# + 335 = 2866
        self.trainingSet = trainingSet[interval[0][0]:interval[testIndex][0]]
        self.trainingSet = pd.concat([self.trainingSet, trainingSet[interval[testIndex][1]+1:]], axis=0)
        return

    def computeHypothesisSpace(self, quartiles = 4):
        """ page 106 in the course book
        """
        sizeOfPossibleInstances = 1
        numberOfPossibleExtensions = pow(2, sizeOfPossibleInstances)
        sizeOfHypothesisSpace = 1
        for column in self.trainingSet:
            if column == "spam" or self.trainingSet[column].nunique() == 0:
                continue
            sizeOfPossibleInstances *= self.trainingSet[column].nunique()
            sizeOfHypothesisSpace *= (self.trainingSet[column].nunique()+1)
        print("-"*20)
        print(f"Size of possible instances: {sizeOfPossibleInstances}")
        print(f"Number of possible extensions:  2 ^ ({sizeOfPossibleInstances})")
        print(f"Size of hypothesis space: {sizeOfHypothesisSpace}")
        print("-"*20)

    def printConfusionMatrix(self, TP, FP, TN, FN):
        """ page 54 in the course book
        """
        print("-"*60)
        print("\t\tPredicted +\tPredicted -")
        print("-"*60)
        print(f"Actual +\t\t{TP}\t\t{FN}\t\t{TP+FN}")
        print(f"Actual -\t\t{FP}\t\t{TN}\t\t{FP+TN}")
        print(f"\t\t\t{TP+FP}\t\t{FN+TN}\t\t{TP+TN+FN+FP}")
        print("-"*60)


def main():
    spamDe = spamDetection()
    spamDe.readData()
    spamDe.setHeader()
    spamDe.clean()
    spamDe.createTrainingSet()
    spamDe.trainingSet = spamDe.transformData(spamDe.trainingSet)
    spamDe.computeHypothesisSpace()
    spamDe.test()

if __name__ == '__main__':
    main()
