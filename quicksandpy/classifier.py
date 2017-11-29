import numpy as np 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class HierClassifier(object):
	def __init__(self):

		self.cl1 = LogisticRegression()
		self.cl2 = LogisticRegression()
		self.cl3 = LogisticRegression()

	def fit(self, train, labels):

		labels1 = labels.copy()
		nobj_ind = np.where(labels!="obj")[0]
		labels1[nobj_ind] = "nobj"

		train2 = train[nobj_ind]
		labels2 = labels.copy()

		ncom_ind = np.intersect1d(np.where(labels!="com")[0], np.where(labels!="obj")[0])
		labels2[ncom_ind] = "ncom"
		labels2 = labels2[nobj_ind]

		train3 = train[ncom_ind]
		labels3 = labels[ncom_ind]

		self.cl1.fit(train, labels1)
		self.cl2.fit(train2, labels2)
		self.cl3.fit(train3, labels3)


	def predict(self, test):

		pred1 = self.cl1.predict(test)
		pred2 = self.cl2.predict(test)
		pred3 = self.cl3.predict(test)

		obj_ind = np.where(pred1=="obj")[0]
		com_ind = np.where(pred2=="com")[0]

		hier_pred = pred3
		hier_pred[com_ind] = "com"
		hier_pred[obj_ind] = "obj"

		return hier_pred



class nn_model(nn.Module):
	def __init__(self, feature_dim, num_classes):
		super(nn_model, self).__init__()

		self.l1 = nn.Linear(feature_dim, 200)
		self.l2 = nn.Linear(200, 200)
		self.l3 = nn.Linear(200, num_classes)
	
	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = F.softmax(self.l3(x))
		return x 


class nn_classifier(object):
	def __init__(self, feature_dim, num_classes):
		self.nn = nn_model(feature_dim, num_classes).cuda()
		self.optimizer = torch.optim.Adam(self.nn.parameters())
		self.criterion = nn.CrossEntropyLoss()


	def fit(self, train, labels, iterations=100, batch_size=100):

		for it in range(iterations):
			train_batch, labels_batch = get_batch(train, labels, batch_size)

			train_var = Variable(torch.from_numpy(train_batch).float()).cuda()
			labels_var = Variable(torch.from_numpy(labels_batch)).long().cuda()

			pred = self.nn(train_var)
			loss = self.criterion(pred, labels_var)

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()


	def predict(self, test):
		test = Variable(torch.from_numpy(test).float()).cuda()
		pred = self.nn(test).cpu().data.numpy()
		return np.argmax(pred, axis=1)		


def get_batch(x, y, batch_size=100):
	ind = np.random.randint(0, x.shape[0], size=batch_size)
	return x[ind], y[ind]


def test_classifiers(train_tweets, test_tweets, train_labels, test_labels, num_classes):

	clfs = []

	# clf = nn_classifier(feature_dim=train_tweets.shape[1], num_classes=num_classes)
	# clfs.append(clf)

	clf = HierClassifier()
	clfs.append(clf)

	clf = LogisticRegression()
	clfs.append(clf)

	clf = KNeighborsClassifier(n_neighbors=2)
	clfs.append(clf)

	clf = LinearSVC()
	clfs.append(clf)

	# clf = GaussianNB()
	# clfs.append(clf)

	clf = DecisionTreeClassifier()
	clfs.append(clf)

	clf = RandomForestClassifier()
	clfs.append(clf)

	scores = []
	for clf in clfs:
		clf.fit(train_tweets, train_labels)
		pred_labels = clf.predict(test_tweets)
		score = classification_report(test_labels, pred_labels, digits=3)
		scores.append(scores)
		print(clf)
		print(score)
		print("------------------------------------")

	return scores 


if __name__ == "__main__":
	train_tweets = np.array([[1,1,1,1], [1,1,1,0], [1,0,1,0], [0,1,0,1]])
	test_tweets = np.array([[0,0,0,0], [0,0,0,1], [1,0,1,0], [0,1,0,1]])
	train_labels = np.array(["obj","com", "pos", "neg"])
	test_labels = np.array(["com", "obj", "pos", "neg"])

	test_classifiers(train_tweets, test_tweets, train_labels, test_labels, 2)