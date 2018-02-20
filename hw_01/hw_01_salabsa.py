import numpy as np
import pandas as pd
from collections import defaultdict

def find_majority(labels):
    """
    Find the majority label. We pass in a list of "labels" for which images are the closests.
    We then create a dictionary with these labels as the key and we use a counter as the value.
    From there we iterate over the dictionary and find the highest value and then we return the label.

    This doesn't account for labels that happen to be the same. So if we have 7 labels coming in and we have 3
    which have the label 2, 3 more have the label of 3, and 1 has a label of 0 then the first one inserted should
    be returned. So a weighted label might be better based of distance. But I'm lazy.
    """
    c = defaultdict(int)
    for l in labels:
        c[l] += 1

    majority = max(c.values())
    for k, v in c.items():
        if v == majority:
            return k


def knn(test_data, training_data, k):
    """
    We take in the test data, training data, and a "k". From there we iterate over each row which represents an 
    "img". Then we iterate over the training data and find the distance between the images. Then we pull the "K" 
    shortest distances and find which label is closest. Once we've gone over the entire test data set we check 
    the accuracy.
    """
    prediction = []
    for idx, img in test_data.iterrows():
        distance = []
        d = 0.0
        for i, t in training_data.iterrows():
            d = np.math.sqrt(sum((img.values.astype(float)[1:] - t.values.astype(float)[1:])**2))
            distance.append((t[0], d))
        label, distance = zip(*(sorted(distance, key=lambda tup: tup[1])[:k]))
        guess = find_majority(label)
        prediction.append(img[0] == guess)

    return prediction.count(True) / float(len(prediction)) * 100

training_data = pd.read_csv('MNIST_training.csv', skiprows=[0],header=None)
test_data = pd.read_csv('MNIST_test.csv', skiprows=[0],header=None)

for k in range(1,41,2):
    p = knn(test_data, training_data, k)
    print "Accuracy for k %i is %f" % (k, p)
