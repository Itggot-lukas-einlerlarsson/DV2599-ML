# Kim Lillé and Lukas Einler Larsson
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# TODO:
# 1. välja 3 algoritmer - KNN, SVM, Naive bayes
# 2. ten fold test
# 3. Skapa ett table som figur 12.4, SIDA 350
# 4. Kör ett friedman test, figur 12.8, SIDA 355
# 5. om det är signifikant differens, köra Nemeyi test SIDA 356

# Evaluation measures: perform a comparison between the selected algorithms based on
# 1) computational performance in terms of training time,
# 2) predictive performance based on accuracy, and
# 3) predictive performance based on F-measure.


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

    def run_kNNclassifier(self, n_neighbors, X_train, X_test, y_train, y_test):
        kNNmodel = KNeighborsClassifier(n_neighbors)
        kNNmodel.fit(X_train, y_train)
        accuracy = kNNmodel.score(X_test, y_test)
        y_pred = kNNmodel.predict(X_test)
        return accuracy

    def run_SVMclassifier(self, X_train, X_test, y_train, y_test):
        SVMmodel = SVC()
        SVMmodel.fit(X_train, y_train)
        accuracy = SVMmodel.score(X_test, y_test)
        return accuracy


    def run_NaiveBayesClassifier(self, X_train, X_test, y_train, y_test):
        NBmodel = ComplementNB()
        NBmodel.fit(X_train, y_train)
        accuracy = NBmodel.score(X_test, y_test)
        return accuracy


    def run_stratifiedKfoldTest(self, n = 10):
        """ page 349, 350
            runs stratified k fold test
        """
        X, y = self.spamDict["data"], self.spamDict["target"]
        skf = StratifiedKFold(n_splits=n, random_state=None)
        accuracy_kNN = []
        accuracy_SVM = []
        accuracy_NaiveBayes = []
        for train_index, test_index in skf.split(X,y): # X is the feature set and y is the target
            print("Train:", train_index, "Validation:", test_index[0:2]) #val_index  = test_index?
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            accuracy_kNN.append(self.run_kNNclassifier(10, X_train, X_test, y_train, y_test))
            accuracy_SVM.append(self.run_SVMclassifier(X_train, X_test, y_train, y_test))
            accuracy_NaiveBayes.append(self.run_NaiveBayesClassifier(X_train, X_test, y_train, y_test))
        ranks = self.run_friedmanTest(accuracy_kNN, accuracy_SVM, accuracy_NaiveBayes)
        self.printTable(np.array(accuracy_kNN), np.array(accuracy_SVM), np.array(accuracy_NaiveBayes), ranks)

    def run_friedmanTest(self, accuracy_kNN, accuracy_SVM, accuracy_NaiveBayes, k = 3):
        """ Page 355, 356 in the course book
            idea: rank the performance of all k algorithms per data set
            from best performance (rank 1) to worst (rank k)
            R_i_j = rank of the j:th algorithm on the i:th dataset
            R_j = average rank of the j:th algorithm
            Null hypothesis H_0: all algorithms perform the same, all avg. ranks are the same
        """
        ranks_kNN = []
        ranks_SVM = []
        ranks_NaiveBayes = []
        for i in range(len(accuracy_kNN)):
            accuracies = []
            accuracies.append((accuracy_kNN[i], "kNN"))
            accuracies.append((accuracy_SVM[i], "SVM"))
            accuracies.append((accuracy_NaiveBayes[i], "NaiveBayes"))
            accuracies = sorted(accuracies, reverse=True)
            rank = 1
            for value in accuracies:
                if value[1] == "kNN":
                    ranks_kNN.append(rank)
                if value[1] == "SVM":
                    ranks_SVM.append(rank)
                if value[1] == "NaiveBayes":
                    ranks_NaiveBayes.append(rank)
                rank += 1
        # ranks = [np.array(ranks_kNN), np.array(ranks_SVM), np.array(ranks_NaiveBayes)]
        ranks = {}
        ranks["kNN"] = np.array(ranks_kNN)
        ranks["SVM"] = np.array(ranks_SVM)
        ranks["NaiveBayes"] = np.array(ranks_NaiveBayes)
        print(ranks)
        return ranks

    def run_nemenyiTest(self, k = 3, n = 10, sigLevel = 0.05):
        criticalDifference = sigLevel * math.sqrt( (k*(k+1))/(6*n) )
        return criticalDifference



    def printTable(self, accuracy_kNN, accuracy_SVM, accuracy_NaiveBayes, ranks = None):
        """ Page 350.
            printing a table similar to the figure 12.4 in the course book
        """
        print("-" * 60)
        print("Fold\t\tkNN\t\tSVM\t\tNaive Bayes")
        print("-" * 60)
        np.set_printoptions(precision=4) # Doesnt work unless print whole array it seems
        for i in range(len(accuracy_kNN)):
            # print(accuracy_kNN)
            # print(f"{i+1}\t\t{accuracy_kNN[i]}{ranks["kNN"][i]}\t\t{accuracy_SVM[i]}{ranks["SVM"][i]}\t\t{accuracy_NaiveBayes[i]}{ranks["NaiveBayes"][i]}")
            print(i+1, end = "\t\t")
            print(accuracy_kNN[i], ranks["kNN"][i], end = "\t\t")
            print(accuracy_SVM[i], ranks["SVM"][i],end = "\t\t")
            print(accuracy_NaiveBayes[i], ranks["NaiveBayes"][i])
        print("-" * 60)
        print(f"avg\t\t{accuracy_kNN.mean()}\t\t{accuracy_SVM.mean()}\t\t{accuracy_NaiveBayes.mean()}")
        print(f"stdev\t\t{accuracy_kNN.std()}\t\t{accuracy_SVM.std()}\t\t{accuracy_NaiveBayes.std()}")

        # Friedman: Average rank
        mean_kNNrank = ranks["kNN"].mean()
        mean_SVMrank = ranks["SVM"].mean()
        mean_NaiveBayesrank = ranks["NaiveBayes"].mean()
        print(f"avgRank\t\t{mean_kNNrank}\t\t{mean_SVMrank}\t\t{mean_NaiveBayesrank}")

        # sum of squared differences
        mean_TOTrank = (mean_kNNrank + mean_SVMrank + mean_NaiveBayesrank)/3
        sum_squaredDiff = (mean_TOTrank-mean_kNNrank)**2
        sum_squaredDiff = (mean_TOTrank-mean_SVMrank)**2
        sum_squaredDiff = (mean_TOTrank-mean_NaiveBayesrank)**2
        assert len(ranks["kNN"]) == 10
        sum_squaredDiff *= len(ranks["kNN"])

        sum_squaredDiff_nk = 0
        for i in range(len(ranks["kNN"])):
            sum_squaredDiff_nk += (ranks["kNN"][i] - mean_TOTrank)**2
            sum_squaredDiff_nk += (ranks["SVM"][i] - mean_TOTrank)**2
            sum_squaredDiff_nk += (ranks["NaiveBayes"][i] - mean_TOTrank)**2
        sum_squaredDiff_nk *= 1/(len(ranks["kNN"])*(len(ranks)-1))

        print("avg total rank:", mean_TOTrank)
        print("the sum of squared differences:", sum_squaredDiff)
        print("the sum of squared differences nk:", sum_squaredDiff_nk)
        # print("avgRank", end = "\t\t")
        # print(ranks["kNN"].mean(), end = "\t\t")
        # print(ranks["SVM"].mean(),end = "\t\t")
        # print(ranks["NaiveBayes"].mean())

        # print(f"avgRank\t\t{ranks["kNN"].mean()}\t\t{ranks["SVM"].mean()}\t\t{ranks["NaiveBayes"].mean()}")


def main():
    spamDe = spamDetection()
    spamDe.readData()
    spamDe.setHeader()
    spamDe.clean()
    spamDe.datasetCreation()
    spamDe.run_stratifiedKfoldTest()


if __name__ == '__main__':
    main()
