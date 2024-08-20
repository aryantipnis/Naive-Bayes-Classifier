from utils import *
import pprint
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

def naive_bayes():
	percentage_positive_instances_train = 0.2
	percentage_negative_instances_train = 0.2

	percentage_positive_instances_test  = 0.2
	percentage_negative_instances_test  = 0.2
	
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

	print("Number of positive training instances:", len(pos_train))
	print("Number of negative training instances:", len(neg_train))
	print("Number of positive test instances:", len(pos_test))
	print("Number of negative test instances:", len(neg_test))

	with open('vocab.txt','w') as f:
		for word in vocab:
			f.write("%s\n" % word)
	print("Vocabulary (training set):", len(vocab))

	# Q1(pos_train, neg_train, pos_test, neg_test)
	# Q2(pos_train, neg_train, pos_test, neg_test, len(vocab))
	Q3to6(pos_train, neg_train, pos_test, neg_test, len(vocab))

def Q1(pos_train, neg_train, pos_test, neg_test):
	pos_wordCount, pos_total = wordCount(pos_train)
	neg_wordCount, neg_total = wordCount(neg_train)

	#Predict using Posterior Probabilites
	actualLabels = np.concatenate((np.ones(len(pos_test)), np.zeros(len(neg_test))))
	predicted_post = postProbability(pos_test, pos_wordCount, neg_wordCount, pos_total, neg_total, len(pos_train), len(neg_train))
	predicted_post += postProbability(neg_test, pos_wordCount, neg_wordCount, pos_total, neg_total, len(pos_train), len(neg_train))

	# Predict using Log Probabilites
	predicted_log = logProbability(pos_test, pos_wordCount, neg_wordCount, pos_total, neg_total, len(pos_train), len(neg_train))
	predicted_log += logProbability(neg_test, pos_wordCount, neg_wordCount, pos_total, neg_total, len(pos_train), len(neg_train))

	print('\nMetrics for Q1 Posterior Probabilities')
	evaluateMetrics(actualLabels, predicted_post)
	print('\nMetrics for Q1 Log Probabilities')
	evaluateMetrics(actualLabels, predicted_log)

def Q2(pos_train, neg_train, pos_test, neg_test, vocabLen):
	pos_wordCount, pos_total = wordCount(pos_train)
	neg_wordCount, neg_total = wordCount(neg_train)

	#Predict using Laplace Smoothing for apha = 1
	alpha = 1
	actualLabels = np.concatenate((np.ones(len(pos_test)), np.zeros(len(neg_test))))

	predictedLabels = LaplaceProbability(pos_test, pos_wordCount, neg_wordCount, pos_total, neg_total, len(pos_train), len(neg_train), alpha, vocabLen)
	predictedLabels += LaplaceProbability(neg_test, pos_wordCount, neg_wordCount, pos_total, neg_total, len(pos_train), len(neg_train), alpha, vocabLen)

	print('\nMetrics for Q2 Laplace Smoothing when alpha = 1')
	evaluateMetrics(actualLabels, predictedLabels)

	#Predict using Laplace Smoothing for varying apha 
	k = 0.0001
	accuracies = []
	alphaValues = []
	while(k <= 1000):
		predictedLabels = LaplaceProbability(pos_test, pos_wordCount, neg_wordCount, pos_total, neg_total, len(pos_train), len(neg_train), k, vocabLen)
		predictedLabels += LaplaceProbability(neg_test, pos_wordCount, neg_wordCount, pos_total, neg_total, len(pos_train), len(neg_train), k, vocabLen)

		print(k)
		accuracy, _, _, _ = evaluateMetrics(actualLabels, predictedLabels)
		alphaValues.append(k)
		accuracies.append(accuracy)
		k*=10
	
	plt.figure(figsize=(10, 6))
	plt.plot(alphaValues, accuracies, marker='o', linestyle='-')
	plt.xscale('log')
	plt.xlabel('Alpha')
	plt.ylabel('Accuracy')
	plt.title('Accuracy vs. Alpha')
	plt.grid(True)
	plt.show()

def Q3to6(pos_train, neg_train, pos_test, neg_test, vocabLen):
	pos_wordCount, pos_total = wordCount(pos_train)
	neg_wordCount, neg_total = wordCount(neg_train)

	#Predict using Laplace Smoothing for apha = 10
	alpha = 10
	actualLabels = np.concatenate((np.ones(len(pos_test)), np.zeros(len(neg_test))))

	predictedLabels = LaplaceProbability(pos_test, pos_wordCount, neg_wordCount, pos_total, neg_total, len(pos_train), len(neg_train), alpha, vocabLen)
	predictedLabels += LaplaceProbability(neg_test, pos_wordCount, neg_wordCount, pos_total, neg_total, len(pos_train), len(neg_train), alpha, vocabLen)

	print('\nMetrics for Laplace Smoothing when alpha = 10')
	evaluateMetrics(actualLabels, predictedLabels)

def wordCount(dataset):
	wordCount = defaultdict(int)
	total = 0
	for doc in dataset:
		words, counts = np.unique(doc, return_counts=True)
		for word, count in zip(words, counts):
			wordCount[word] += count
		total += np.sum(counts)
	return wordCount, total

def postProbability(testData, pos_wordCount, neg_wordCount, pos_total, neg_total, pos_trainLen, neg_trainLen):
	predictedLabels = []
	N = pos_trainLen + neg_trainLen

	for doc in testData:
		pos_Pr = pos_trainLen/N
		neg_Pr = neg_trainLen/N
		for word in doc:
			freq_pos = pos_wordCount[word]
			freq_neg = neg_wordCount[word]

			pos_Pr *= freq_pos/pos_total if pos_total != 0 else 0
			neg_Pr *= freq_neg/neg_total if neg_total != 0 else 0

		if pos_Pr == neg_Pr:
			predictedLabels.append(random.randint(0, 1))
		else: 
			predictedLabels.append(1 if pos_Pr > neg_Pr else 0)

	return predictedLabels

def logProbability(testData, pos_wordCount, neg_wordCount, pos_total, neg_total, pos_trainLen, neg_trainLen):
	predictedLabels = []
	N = pos_trainLen + neg_trainLen
	
	for doc in testData:
		pos_Log = np.log(pos_trainLen/N)
		neg_Log = np.log(neg_trainLen/N)

		for word in doc:
			freq_pos = pos_wordCount[word]
			freq_neg = neg_wordCount[word]
			
			pos_Log += np.log(1e-8) if freq_pos == 0 else np.log(freq_pos/pos_total)
			neg_Log += np.log(1e-8) if freq_neg == 0 else np.log(freq_neg/neg_total)

		if pos_Log == neg_Log:
			predictedLabels.append(random.randint(0, 1))
		else: 
			predictedLabels.append(1 if pos_Log > neg_Log else 0)
	
	return predictedLabels

def LaplaceProbability(testData, pos_wordCount, neg_wordCount, pos_total, neg_total, pos_trainLen, neg_trainLen, alpha, vocabLen):
	predictedLabels = []
	N = pos_trainLen + neg_trainLen

	for doc in testData:
		pos_Log = np.log(pos_trainLen/N)
		neg_Log = np.log(neg_trainLen/N)

		for word in doc:
			freq_pos = pos_wordCount[word]
			freq_neg = neg_wordCount[word]
			
			pos_Log += np.log((freq_pos + alpha) / (pos_total + alpha * vocabLen))
			neg_Log += np.log((freq_neg + alpha) / (neg_total + alpha * vocabLen))

		if pos_Log == neg_Log:
			predictedLabels.append(random.randint(0, 1))
		else: 
			predictedLabels.append(1 if pos_Log > neg_Log else 0)
	
	return predictedLabels


def evaluateMetrics(actualLabels, predictedLabels):
    TP = FP = FN = TN = 0

    for actual, predicted in zip(actualLabels, predictedLabels):
        if actual == 1 and predicted == 1:
            TP += 1
        elif actual == 0 and predicted == 1:
            FP += 1
        elif actual == 1 and predicted == 0:
            FN += 1
        elif actual == 0 and predicted == 0:
            TN += 1

    confusionMatrix = [[TP, FN], [FP, TN]]
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0

    print('Confusion Matrix:', confusionMatrix)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)

    return accuracy, precision, recall, confusionMatrix

if __name__=="__main__":
	naive_bayes()
