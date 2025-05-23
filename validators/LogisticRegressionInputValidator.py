from .BaseAlgorithmValidator import BaseAlgorithmValidator
import numpy as np

class LogisticRegressionInputValidator(BaseAlgorithmValidator):

    def __init__(self, input, output, threshold):
        super().validateInputShape(input,output)
        self.__checkLabelsProvided(output)
        self.__checkThreshold(threshold)
    
    def __checkLabelsProvided(self, labels):
        uniques = np.unique(labels)
        if len(uniques) > 2:
            print(uniques)
            raise ValueError("More than 2 classes found , logistic regression needs only 2 classes at most")
    
    def __checkThreshold(self, threshold):
        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold indicates cut-off probability and it can't be below 0 or above 1")