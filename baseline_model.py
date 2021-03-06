# All imports
import timeit
from organize_data import get_train_and_test_data
#run linear regression separately for social and economic leanings

train_data, test_data = get_train_and_test_data()
start = timeit.default_timer()
#borrowing functions from sentiment assignment, adjusted for this model
def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for i in range(len(d1)):
        d1[i] += d2[i] * scale

eta = 0.01
numIters = 10
weights_social = [0] * len(train_data['BillCassidy'][0])
weights_economic = [0] * len(train_data['BillCassidy'][0])

#train
for i in range(numIters):
    print("Iteration: " , i)
    for key in train_data.keys():
        feature = train_data[key][0]
        economic_y = train_data[key][2]
        social_y = train_data[key][3]
        if economic_y * numpy.dot(feature, weights_economic) < 1:
            increment(weights_economic, eta * economic_y, feature)
        if social_y * numpy.dot(feature, weights_social) < 1:
            increment(weights_social, eta * social_y, feature)

#prediction function
def predict(phi, weight):
    if numpy.dot(phi, weight) < 0:
        return -1
    else:
        return 1

#predict

maxScore = float(len(test_data.keys()))
predictScore = 0.0
economic_score = 0.0
social_score = 0.0

for key in test_data.keys():
    if predict(test_data[key][0], weights_economic) == test_data[key][2]:
        predictScore += 0.5
        economic_score += 1
    if predict(test_data[key][0], weights_social) == test_data[key][3]:
        predictScore += 0.5
        social_score += 1

print("Total accuracy: ", predictScore / maxScore)
print("Economic accuracy: ", economic_score / maxScore)
print("Social accuracy: ", social_score / maxScore)
stop = timeit.default_timer()
print("Runtime: ", stop - start)
