# Kim Lillé and Lukas Einler Larsson
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import time
from datetime import timedelta

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

class spamDetection:
    def __init__(self):
        df = pd.DataFrame()
        trainingSet = pd.DataFrame()
        self.kNNtime = []
        self.SVMtime = []
        self.NaiveBayestime = []
        self.kNNFscore = []
        self.SVMFscore = []
        self.NaiveBayesFscore = []
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
        """
            Create a dataset similar to sklearn's datasets used on the lab.
        """
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

    def run_kNNclassifier(self, n_neighbors, X_train, X_test, y_train, y_test, foldNr):
        kNNmodel = KNeighborsClassifier(n_neighbors)
        startTime = time.time()
        kNNmodel.fit(X_train, y_train)
        endTime = time.time()
        accuracy = kNNmodel.score(X_test, y_test)
        y_pred = kNNmodel.predict(X_test)
        self.kNNFscore.append(f1_score(y_test, y_pred))
        self.kNNtime.append(timedelta(seconds=endTime - startTime))
        return accuracy

    def run_SVMclassifier(self, X_train, X_test, y_train, y_test, foldNr):
        SVMmodel = SVC()
        startTime = time.time()
        SVMmodel.fit(X_train, y_train)
        endTime = time.time()
        accuracy = SVMmodel.score(X_test, y_test)
        y_pred = SVMmodel.predict(X_test)
        self.SVMFscore.append(f1_score(y_test, y_pred))
        self.SVMtime.append(timedelta(seconds=endTime - startTime))
        return accuracy


    def run_NaiveBayesClassifier(self, X_train, X_test, y_train, y_test, foldNr):
        NBmodel = ComplementNB()
        startTime = time.time()
        NBmodel.fit(X_train, y_train)
        endTime = time.time()
        accuracy = NBmodel.score(X_test, y_test)
        y_pred = NBmodel.predict(X_test)
        self.NaiveBayesFscore.append(f1_score(y_test, y_pred))
        self.NaiveBayestime.append(timedelta(seconds=endTime - startTime))
        return accuracy


    def run_stratifiedKfoldTest(self, n = 10):
        """ page 349, 350
            runs stratified k fold test
            During each learning iteration, for each algorithm: accuracy, fscore and time is measured
        """
        X, y = self.spamDict["data"], self.spamDict["target"]
        skf = StratifiedKFold(n_splits=n, random_state=None)
        accuracy_kNN = []
        accuracy_SVM = []
        accuracy_NaiveBayes = []
        foldnr = 1
        for train_index, test_index in skf.split(X,y): # X is the feature set and y is the target
            print("training... foldnr:", foldnr)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            accuracy_kNN.append(self.run_kNNclassifier(20, X_train, X_test, y_train, y_test, foldnr))
            accuracy_SVM.append(self.run_SVMclassifier(X_train, X_test, y_train, y_test, foldnr))
            accuracy_NaiveBayes.append(self.run_NaiveBayesClassifier(X_train, X_test, y_train, y_test, foldnr))
            foldnr += 1

        # run test to see if there is difference between the algorithms measurements
        # Accuracy
        print("\n\n\n",30 * "-", " Accuracy ", 37 * "-", end = "")
        self.run_friedmanTest(accuracy_kNN, accuracy_SVM, accuracy_NaiveBayes, "Accuracy")

        #Time
        print("\n\n\n",30 * "-", " Time(ms) ", 37 * "-", end = "")
        for i in range(len(self.kNNtime)):
            self.kNNtime[i] = millisecondsFromTimedelta(self.kNNtime[i], 5)
            self.SVMtime[i] = millisecondsFromTimedelta(self.SVMtime[i], 5)
            self.NaiveBayestime[i] = millisecondsFromTimedelta(self.NaiveBayestime[i], 5)
        self.run_friedmanTest(self.kNNtime, self.SVMtime, self.NaiveBayestime, "Time(ms)")

        #F1-score
        print("\n\n\n",30 * "-", " F1-score ", 37 * "-", end = "")
        self.run_friedmanTest(self.kNNFscore, self.SVMFscore, self.NaiveBayesFscore, "F1-score")


    def run_friedmanTest(self, kNN_data, SVM_data, naiveBayes_data, measurement):
        ranks = self.get_friedmanRanks(kNN_data, SVM_data, naiveBayes_data, measurement)
        self.printMeasurementTable(np.array(kNN_data), np.array(SVM_data), np.array(naiveBayes_data), ranks, measurement)
        self.printFriedmanTable(np.array(kNN_data), np.array(SVM_data), np.array(naiveBayes_data), ranks, measurement)
        mean_TOTrank, sum_squaredDiff, sum_squaredDiff_nk, mean_ranks = self.get_FriedmanTestMeasurements(ranks)
        CV = self.run_calcCriticalValue(k = 3, n = 10, siglevel = 0.05)

        #see if significant difference
        if sum_squaredDiff/sum_squaredDiff_nk > CV:
            print(f"Friedman Test: There is a significant difference between the algorithms, {sum_squaredDiff}/{sum_squaredDiff_nk} > {CV}")
            print("Running running Nemeyi test:", end = " ")
            self.run_nemenyiTest(mean_ranks)
        else:
            print("Friedman Test: There is no significant difference between the algorithms when it comes to: " + measurement)


    def get_friedmanRanks(self, data_kNN, data_SVM, data_NaiveBayes, measurementStr):
        """ Page 355, 356 in the course book
            idea: rank the performance of all k algorithms per data set
            from best performance (rank 1) to worst (rank k)

            course books symbols:
            R_i_j = rank of the j:th algorithm on the i:th dataset
            R_j = average rank of the j:th algorithm
            Null hypothesis, H_0: all algorithms perform the same
        """
        ranks_kNN = []
        ranks_SVM = []
        ranks_NaiveBayes = []
        for i in range(len(data_kNN)):
            measurement = []
            measurement.append((data_kNN[i], "kNN"))
            measurement.append((data_SVM[i], "SVM"))
            measurement.append((data_NaiveBayes[i], "NaiveBayes"))
            if "Time" in measurementStr :
                measurement = sorted(measurement, reverse=False)
            else:
                measurement = sorted(measurement, reverse=True)
            rank = 1
            for value in measurement:
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

    def get_FriedmanTestMeasurements(self, ranks, k = 3):
        """ page 355
        This Function calculates the three points from the course book:
        1) The average rank
        2) The sum of square differences, measuring the spread of rank centroids
        3) The sum of square differences, measuring the spread of all rank
        """
        mean_kNNrank = ranks["kNN"].mean() # 2.1
        mean_SVMrank = ranks["SVM"].mean() # 1.6
        mean_NaiveBayesrank = ranks["NaiveBayes"].mean() # 2.3
        mean_ranks = [mean_kNNrank, mean_SVMrank, mean_NaiveBayesrank]
        mean_TOTrank = (k+1)/2 # 1)

        # sum of squared differences # 2)
        sum_squaredDiff = (mean_kNNrank - mean_TOTrank)**2
        sum_squaredDiff += (mean_SVMrank - mean_TOTrank)**2
        sum_squaredDiff += (mean_NaiveBayesrank - mean_TOTrank)**2
        sum_squaredDiff *= len(ranks["kNN"])

        sum_squaredDiff_nk = 0 # 3)
        for i in range(len(ranks["kNN"])):
            sum_squaredDiff_nk += (ranks["kNN"][i] - mean_TOTrank)**2
            sum_squaredDiff_nk += (ranks["SVM"][i] - mean_TOTrank)**2
            sum_squaredDiff_nk += (ranks["NaiveBayes"][i] - mean_TOTrank)**2
        sum_squaredDiff_nk *= 1/(len(ranks["kNN"])*(len(ranks)-1))
        print("\nFriedman test:")
        print("avg total rank:", mean_TOTrank)
        print("the sum of squared differences(spread of rank centroids):", sum_squaredDiff)
        print("the sum of squared differences nk(spread over all ranks):", sum_squaredDiff_nk)
        return (mean_TOTrank, sum_squaredDiff, sum_squaredDiff_nk, mean_ranks)

    def run_calcCriticalValue(self, k, n, siglevel):
        """ page 356
            ref: P. Flach, “12 Machine Learning experiments” in Machine Learning: The Art and Science of Algorithms that Make Sense of Data,
            1st ed, Cambridge, United Kingdom, Cambridge university press, 2012
            returns the critical value for k = 3, n = 10
        """
        return 7.8

    def run_nemenyiTest(self, mean_ranks, k = 3, n = 10, sigLevel = 0.05):
        """ page 356
            p_sigLevel = 3.314/math.sqrt(2)
            value 3.314 taken from https://www.real-statistics.com/statistics-tables/studentized-range-q-table/
            alpha level 0.05
        """
        p_sigLevel = 2.343
        CD = p_sigLevel * math.sqrt( (k*(k+1))/(6*n) )
        mean_ranks = sorted(mean_ranks)
        criticalDifferenceAlgorithms = [] # holds all algorithm's that have a critical difference
        if abs(max(mean_ranks) - min(mean_ranks)) > CD:
            criticalDifferenceAlgorithms.append(max(mean_ranks))
            criticalDifferenceAlgorithms.append(min(mean_ranks))
            if abs(mean_ranks[1] - mean_ranks[0]) > CD:
                criticalDifferenceAlgorithms.append(mean_ranks[1])
            if mean_ranks[1] not in criticalDifferenceAlgorithms:
                if abs(mean_ranks[2] - mean_ranks[1]) > CD:
                    criticalDifferenceAlgorithms.append(mean_ranks[1])
            print(f"There is a significant difference between the algorithms with average ranks: ",criticalDifferenceAlgorithms)
        else:
            print(f"There is no significant difference between the algorithms")

    def printMeasurementTable(self, data_kNN, data_SVM, data_NaiveBayes, ranks, measurement):
        """ Page 350.
            printing a table similar to the figure 12.4 in the course book
        """
        print("\nFigure 12.4:")
        print("-" * 80)
        print("Fold\t\tkNN\t\t\tSVM\t\t\tNaive Bayes")
        print("-" * 80)
        for i in range(len(data_kNN)):
            print(i+1, end = " " + measurement + "\t")
            print(np.round(data_kNN[i], 5), end = "\t\t\t")
            print(np.round(data_SVM[i], 5),end = "\t\t\t")
            print(np.round(data_NaiveBayes[i], 5))
        print("-" * 80)
        print("avg " + measurement + f"\t{np.round(data_kNN.mean(), 5)}\t\t\t{np.round(data_SVM.mean(), 5)}\t\t\t{np.round(data_NaiveBayes.mean(), 5)}")
        print(f"stdev\t\t{np.round(data_kNN.std(), 5)}\t\t\t{np.round(data_SVM.std(), 5)}\t\t\t{np.round(data_NaiveBayes.std(), 5)}")
        print("-" * 80)


    def printFriedmanTable(self, data_kNN, data_SVM, data_NaiveBayes, ranks, measurement):
        """ Page 356.
            printing a table similar to the figure 12.8 in the course book
        """
        print("\n\nFigure 12.8:")
        print("-" * 80)
        print("Fold\t\tkNN\t\t\tSVM\t\t\tNaive Bayes")
        print("-" * 80)
        for i in range(len(data_kNN)):
            print(i+1, end = " " + measurement + "\t")
            print(np.round(data_kNN[i], 5),ranks["kNN"][i] ,end = "\t\t")
            print(np.round(data_SVM[i], 5),ranks["SVM"][i] ,end = "\t\t")
            print(np.round(data_NaiveBayes[i], 5),ranks["NaiveBayes"][i])
        mean_kNNrank = ranks["kNN"].mean()
        mean_SVMrank = ranks["SVM"].mean()
        mean_NaiveBayesrank = ranks["NaiveBayes"].mean()
        print("-" * 80)
        print(f"avg rank\t{mean_kNNrank}\t\t\t{mean_SVMrank}\t\t\t{mean_NaiveBayesrank}")
        print("-" * 80)

def millisecondsFromTimedelta(timedelta, digits = 6):
    """Compute the milliseconds in a timedelta"""
    return round(timedelta.total_seconds() * 1000, ndigits = digits)


def main():
    spamDe = spamDetection()
    spamDe.readData()
    spamDe.setHeader()
    spamDe.clean()
    spamDe.datasetCreation()
    spamDe.run_stratifiedKfoldTest()


if __name__ == '__main__':
    main()
