# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 


# Mira implementation
import util
PRINT = True


class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            cGrid = [0.001, 0.002, 0.004, 0.008]
        else:
            cGrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, cGrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, cGrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """

        # bestAccuracyCount = -1  # best accuracy so far on validation set
        # cGrid.sort(reverse=True)
        # bestParams = cGrid[0]
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        best_weights = {}
        best_accuracy_count = None
        best_params = cGrid[0]
        for c in cGrid:
            weights = self.weights.copy()
            for n in range(self.max_iterations):
                for i, datum in enumerate(trainingData):
                    optimum_score = None
                    optimum_label = None
                    for label in self.legalLabels:
                        score = datum * weights[label]
                        if optimum_score is None or score > optimum_score:
                            optimum_score = score
                            optimum_label = label
                    actual_label = trainingLabels[i]
                    if optimum_label != actual_label:
                        # update weights
                        f = datum.copy()
                        tau = min(c, ((weights[optimum_label] - weights[actual_label]) * f + 1.0) / (2.0 * (f*f)))
                        f.divideAll(1.0/tau)
                        weights[actual_label] = weights[actual_label] + f
                        weights[optimum_label] = weights[optimum_label] - f
                # Evaluate the accuracy
                correct = 0
                guesses = self.classify(validationData)
                for i, guess in enumerate(guesses):
                    if validationLabels[i] == guess:
                        correct += 1.0
                accuracy = correct / len(guesses)
                if best_accuracy_count is None or accuracy > best_accuracy_count:
                    best_accuracy_count = accuracy
                    best_weights = weights
                    best_params = c
        self.weights = best_weights

        print("finished training. Best cGrid param = ", best_params)

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        for datum in data:
            vectors = util.Counter()
            for label in self.legalLabels:
                vectors[label] = self.weights[label] * datum
            guesses.append(vectors.argMax())
        return guesses

        return guesses


    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        featureWeights = self.weights[label].sortedKeys()[:100]
        return featuresWeights
