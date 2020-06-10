import timeit
from organize_data import get_train_and_test_data
import numpy
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


start = timeit.default_timer()
train_data, test_data = get_train_and_test_data()
# train_data and test_data are dicts of key = screen_name and val = [BOW vector, bioname, econ_pc, social_pc]

midpoint = timeit.default_timer()
print("Finished creating features. Runtime: ", midpoint - start)

# Split train and test data into X and Y for econ and social
X_train_social = []
Y_train_social = []
X_train_econ = []
Y_train_econ = []

for key, value in train_data.items():
	BOW_vector = value[0]
	X_train_social.append(BOW_vector)
	X_train_econ.append(BOW_vector)
	econ_pc = value[2]
	social_pc = value[3]
	Y_train_social.append(social_pc)
	Y_train_econ.append(econ_pc)

X_test_social = []
Y_test_social = []
X_test_econ = []
Y_test_econ = []

for key, value in test_data.items():
	BOW_vector = value[0]
	X_test_social.append(BOW_vector)
	X_test_econ.append(BOW_vector)
	econ_pc = value[2]
	social_pc = value[3]
	Y_test_social.append(social_pc)
	Y_test_econ.append(econ_pc)

# Run Gaussian Naive-Bayes
gnb = GaussianNB()
maxScore = float(len(X_test_social))
predictScore = 0.0

# Econ
Y_pred_econ = gnb.fit(X_train_econ, Y_train_econ).predict(X_test_econ)
econ_score = float((Y_test_econ == Y_pred_econ).sum())
predictScore += (econ_score * 0.5)

# Social
Y_pred_social = gnb.fit(X_train_social, Y_train_social).predict(X_test_social)
social_score = float((Y_test_social == Y_pred_social).sum())
predictScore += (social_score * 0.5)

print("Total accuracy: ", predictScore / maxScore)
print("Economic accuracy: ", econ_score / maxScore)
print("Social accuracy: ", social_score / maxScore)


