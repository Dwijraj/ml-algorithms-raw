from .BaseAlgorithmValidator import BaseAlgorithmValidator

class LinearRegressionInputValidator(BaseAlgorithmValidator):

    def __init__(self, input, output):
        super().validateInputShape(input,output)
    
    def 