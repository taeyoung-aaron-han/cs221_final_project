import timeit
from organize_data import get_train_and_test_data
import numpy
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


classifier_type = input("Naive bayes or regression?").lower()

if classifier_type != "naive bayes" and classifier_type != "regression":
    print("Unacceptable classifier type. Exiting...")
    quit() 

start = timeit.default_timer()
train_data, test_data = get_train_and_test_data()
# train_data and test_data are dicts of key = screen_name and val = [BOW vector, bioname, econ_pc, social_pc]

midpoint = timeit.default_timer()
print("Finished creating features. Runtime: ", midpoint - start)

X_train = []
Y_train = []

for key, value in train_data.items():
    BOW_vector = value[0]
    X_train.append(BOW_vector)
    pc = value[4]
    Y_train.append(pc)

X_test = []
Y_test = []

for key, value in test_data.items():
    BOW_vector = value[0]
    X_test.append(BOW_vector)
    pc = value[4]
    Y_test.append(pc)

# Choose classifier based on input
if classifier_type == "regression":
    classifier = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
else:
    classifier = GaussianNB()

maxScore = float(len(X_test))
predictScore = 0.0

Y_pred = classifier.fit(X_train, Y_train).predict(X_test)
predictScore = float((Y_test == Y_pred).sum())

print("Total accuracy: ", predictScore / maxScore)
