import util
import numpy as np
import sys
import random

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

def small_classify(y):
    classifier, data = y
    return classifier.classify(data)

class AdaBoostClassifier:
    """
    AdaBoost classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, boosting_iterations):
        self.legalLabels = legalLabels
        self.boosting_iterations = boosting_iterations
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.boosting_iterations)]
        self.alphas = [0]*self.boosting_iterations

    def train( self, trainingData, trainingLabels):
        """
        The training loop trains weak learners with weights sequentially. 
        The self.classifiers are updated in each iteration and also the self.alphas 
        """
        
        self.features = trainingData[0].keys()
        "*** YOUR CODE HERE ***"
        trainingData = np.array(trainingData)
        trainingLabels = np.array(trainingLabels)
        size = trainingLabels.size
        ratio = 0.75
        weights = np.ones(size) / size

        for i, classifier in enumerate(self.classifiers):
            sample_indices = np.random.choice(np.arange(size), int(ratio * size), p=weights)
            sample_data = trainingData[sample_indices]
            sample_labels = trainingLabels[sample_indices]
            
            classifier.train(sample_data, sample_labels)
            outputs = classifier.classify(trainingData)

            error = 0

            for index, output in enumerate(outputs):
                if output != trainingLabels[index]:
                    error += weights[index]
            
            if error > 1 or error == 0:
                print "Classifier Output: ", outputs
                print "Truth: ", sample_labels
                print "Data: ", sample_data
                print "Weights: ", weights
                print "Error: ", error
                print "Model ", i
                raise Exception("Abnormal error")

            for index, output in enumerate(outputs):
                if output == trainingLabels[index]:
                    weights[index] *= error / (1 - error)
            
            weights = weights / np.sum(weights)
            self.alphas[i] = np.log((1-error)/error)

    def classify( self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.

        The function should return a list of labels where each label should be one of legaLabels.
        """

        "*** YOUR CODE HERE ***"
        weak_classifier_outputs = [0] * len(self.classifiers)
        for i in range(len(self.classifiers)):
            weak_classifier_outputs[i] = self.alphas[i] * np.array(self.classifiers[i].classify(data))

        guesses = reduce(lambda x, y: np.add(x,y), weak_classifier_outputs)
        guesses = map(lambda x: int(np.sign(x)), guesses)
        
        return guesses
