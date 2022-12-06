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

        ranks = self.get_friedmanRanks(accuracy_kNN, accuracy_SVM, accuracy_NaiveBayes)
        self.printTable(np.array(accuracy_kNN), np.array(accuracy_SVM), np.array(accuracy_NaiveBayes), ranks)
        mean_TOTrank, sum_squaredDiff, sum_squaredDiff_nk, mean_ranks = self.run_FriedmanTest(ranks)
        CV = self.run_calcCriticalValue(k = 3, n = 10, siglevel = 0.05)
        
        if sum_squaredDiff > CV:
            print(f"Friedman Test: There is a significant difference between the algorithms, {sum_squaredDiff} > {CV}")
            print("Running running Nemeyi test:", end = " ")
            self.run_nemenyiTest(mean_ranks)
        else:
            print("Friedman Test: There is no significant difference between the algorithms")


    def get_friedmanRanks(self, accuracy_kNN, accuracy_SVM, accuracy_NaiveBayes, k = 3):
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
        ranks = {}
        ranks["kNN"] = np.array(ranks_kNN)
        ranks["SVM"] = np.array(ranks_SVM)
        ranks["NaiveBayes"] = np.array(ranks_NaiveBayes)
        return ranks

    def run_FriedmanTest(self, ranks, k = 3):
        print("-" * 60)
        print("\t\tkNN\t\tSVM\t\tNaive Bayes")
        print("-" * 60)
        # Friedman: Average rank
        mean_kNNrank = ranks["kNN"].mean() # 2.1
        mean_SVMrank = ranks["SVM"].mean() # 1.6
        mean_NaiveBayesrank = ranks["NaiveBayes"].mean() # 2.3
        mean_ranks = [mean_kNNrank, mean_SVMrank, mean_NaiveBayesrank]
        print(f"avgRank\t\t{mean_kNNrank}\t\t{mean_SVMrank}\t\t{mean_NaiveBayesrank}")

        # sum of squared differences
        # mean_TOTrank = (mean_kNNrank + mean_SVMrank + mean_NaiveBayesrank)/3
        mean_TOTrank = (k+1)/2
        sum_squaredDiff = (mean_kNNrank - mean_TOTrank)**2
        sum_squaredDiff += (mean_SVMrank - mean_TOTrank)**2
        sum_squaredDiff += (mean_NaiveBayesrank - mean_TOTrank)**2
        sum_squaredDiff *= len(ranks["kNN"])

        sum_squaredDiff_nk = 0
        for i in range(len(ranks["kNN"])):
            sum_squaredDiff_nk += (ranks["kNN"][i] - mean_TOTrank)**2
            sum_squaredDiff_nk += (ranks["SVM"][i] - mean_TOTrank)**2
            sum_squaredDiff_nk += (ranks["NaiveBayes"][i] - mean_TOTrank)**2
        sum_squaredDiff_nk *= 1/(len(ranks["kNN"])*(len(ranks)-1))

        print("-" * 60)
        print("avg total rank:", mean_TOTrank)
        print("the sum of squared differences(spread of rank centriods):", sum_squaredDiff)
        print("the sum of squared differences nk(spread over all ranks):", sum_squaredDiff_nk)
        return (mean_TOTrank, sum_squaredDiff, sum_squaredDiff_nk, mean_ranks)

    def run_calcCriticalValue(self, k, n, siglevel):
        """ page 356
            calcs critical value via statistical table :)
            https://home.ubalt.edu/ntsbarsh/business-stat/StatistialTables.pdf
            (from TABLE A.4, accessed 2022-12)
        """
        return 7.815

    def run_nemenyiTest(self, mean_ranks, k = 3, n = 10, sigLevel = 0.05):
        """ page 356
            p_sigLevel is taken from the book
        """
        p_sigLevel = 2.343
        criticalDifference = p_sigLevel * math.sqrt( (k*(k+1))/(6*n) )
        if max(mean_ranks) - min(mean_ranks) > criticalDifference:
            print(f"There is a significant difference between the algorithms with average ranks: {max(mean_ranks)} and {min(mean_ranks)}")
        else:
            print(f"There is a significant difference between the algorithms")



    def printTable(self, accuracy_kNN, accuracy_SVM, accuracy_NaiveBayes, ranks = None):
        """ Page 350.
            printing a table similar to the figure 12.4(8) in the course book
        """
        print("-" * 60)
        print("Fold\t\tkNN\t\tSVM\t\tNaive Bayes")
        print("-" * 60)
        np.set_printoptions(precision=4) # Doesnt work unless print whole array it seems
        for i in range(len(accuracy_kNN)):
            print(i+1, end = "\t\t")
            print(accuracy_kNN[i], ranks["kNN"][i], end = "\t\t")
            print(accuracy_SVM[i], ranks["SVM"][i],end = "\t\t")
            print(accuracy_NaiveBayes[i], ranks["NaiveBayes"][i])
        print("-" * 60)
        print(f"avg\t\t{accuracy_kNN.mean()}\t\t{accuracy_SVM.mean()}\t\t{accuracy_NaiveBayes.mean()}")
        print(f"stdev\t\t{accuracy_kNN.std()}\t\t{accuracy_SVM.std()}\t\t{accuracy_NaiveBayes.std()}")


def main():
    spamDe = spamDetection()
    spamDe.readData()
    spamDe.setHeader()
    spamDe.clean()
    spamDe.datasetCreation()
    spamDe.run_stratifiedKfoldTest()


if __name__ == '__main__':
    main()
