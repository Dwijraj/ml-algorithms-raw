from .BaseAlgorithmValidator import BaseAlgorithmValidator
import numpy as np

class SoftMaxModelInputValidator(BaseAlgorithmValidator):

    def __init__(self, input, output):
        super().validateInputShape(input,output)
        self.__checkLabelsProvided(output)
    
    def __checkLabelsProvided(self, labels):

        if not np.all(np.equal(np.mod(labels, 1), 0)):
            raise ValueError("Input contains non-integer values")
