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
from quicksand.quicksandpy import tweet, load_data, util


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

		self.l1 = nn.Linear(feature_dim, 1024)
		self.l2 = nn.Linear(1024, 1024)
		self.l3 = nn.Linear(1024, num_classes)
	
	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.l3(x)
		return x 


class nn_classifier(object):
	def __init__(self, feature_dim, num_classes=4, loss_func="OURS", lr=1e-3, weight_decay=0):
		self.nn = nn_model(feature_dim, num_classes).cuda()
		self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=lr, weight_decay=weight_decay)
		self.criterion = nn.MSELoss()
		self.loss_func = loss_func

	def fit(self, train, labels, iterations=1000, batch_size=64):

		# labels = one_hot(label2digit(labels))
		# train = train.todense().astype(np.float)

		for it in range(iterations):
			train_batch, labels_batch = get_batch(train, labels, batch_size)

			train_var = Variable(torch.from_numpy(train_batch).float()).cuda()
			labels_var = Variable(torch.from_numpy(labels_batch).float()).cuda()

			pred = self.nn(train_var)

			if self.loss_func=="MSE":
				loss = self.criterion(pred, labels_var)
			else:
				loss = self.new_loss(F.softmax(pred), labels_var)

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()


	def predict(self, test):
		test = Variable(torch.from_numpy(test.astype(np.float)).float()).cuda()
		pred = self.nn(test).cpu().data.numpy()
		return digit2label(np.argmax(pred, axis=1))


	def new_loss(self, pred, target):
		loss = target * torch.log(pred)
		return -loss.mean()


def get_batch(x, y, batch_size=100):
	ind = np.random.randint(0, x.shape[0], size=batch_size)
	return x[ind], y[ind]


def label2digit(labels):
	new_labels = np.zeros(labels.shape)
	new_labels[np.where(labels=="pos")[0]] = 0
	new_labels[np.where(labels=="neg")[0]] = 1
	new_labels[np.where(labels=="com")[0]] = 2
	new_labels[np.where(labels=="obj")[0]] = 3

	return new_labels


def digit2label(labels):
	new_labels = np.empty((labels.shape[0]), dtype="<U10")
	new_labels[np.where(labels==0)[0]] = "pos"
	new_labels[np.where(labels==1)[0]] = "neg"
	new_labels[np.where(labels==2)[0]] = "com"
	new_labels[np.where(labels==3)[0]] = "obj"

	return new_labels


def one_hot(labels, num_classes=4):
	ind = np.array(labels, dtype=np.int16)
	onehot = np.zeros((labels.shape[0], num_classes))
	onehot[np.arange(labels.shape[0]), ind] = 1
	return onehot



def test_classifiers(train_tweets, test_tweets, train_labels, test_labels, nn_labels, num_classes=4, nn_only=False):

	clfs = []

	clf = nn_classifier(feature_dim=train_tweets.shape[1], num_classes=num_classes)

	print (clf)
	clf.fit(train_tweets, nn_labels)
	pred_labels = clf.predict(test_tweets)
	score = classification_report(test_labels, pred_labels, digits=3)
	print(score)
	print("------------------------------------")
	score = accuracy_score(test_labels, pred_labels)
	print("accuracy", score)
	print("------------------------------------")

	if not nn_only: 
		clf = HierClassifier()
		clfs.append(clf)

		clf = LogisticRegression()
		clfs.append(clf)

		clf = KNeighborsClassifier(n_neighbors=3)
		clfs.append(clf)

		clf = LinearSVC()
		clfs.append(clf)

		clf = GaussianNB()
		clfs.append(clf)

		clf = DecisionTreeClassifier()
		clfs.append(clf)

		clf = RandomForestClassifier()
		clfs.append(clf)

		for clf in clfs:
			print (clf)
			clf.fit(train_tweets, train_labels)
			pred_labels = clf.predict(test_tweets)
			score = classification_report(test_labels, pred_labels, digits=3)
			print(score)
			print("------------------------------------")
			score = accuracy_score(test_labels, pred_labels)
			print("accuracy", score)
			print("------------------------------------")


if __name__ == "__main__":
	
	train_tw = np.load("quicksand/labelled_data/train_tweets_with_labels.npy")
	test_tw = np.load("quicksand/labelled_data/test_tweets_with_labels.npy")

	label_type = util.MAJORITY_RULE

	train_tweets = []
	train_labels = []
	nn_labels = []
	for tw in train_tw:
		train_tweets.append(tw.get_feature_vector())
		train_labels.append(tw.get_labelling(label_type))
		nn_labels.append(tw.get_labelling(util.SOFTMAX))

	test_tweets = []
	test_labels = []
	for tw in test_tw:
		test_tweets.append(tw.get_feature_vector())
		test_labels.append(tw.get_labelling(label_type))

	# train_tweets, train_labels, test_tweets, test_labels = load_data.get_all_data()
	test_classifiers(np.array(train_tweets), np.array(test_tweets), np.array(train_labels), np.array(test_labels), np.array(nn_labels))
