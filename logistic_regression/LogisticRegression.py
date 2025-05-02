from validators.LogisticRegressionInputValidator import LogisticRegressionInputValidator
from utils.ModelWeightInitializer import ModelWeightInitializer
import numpy as np

class LogisticRegression:

    def __init__(self, feature_data, label, learning_rate=0.01, threshold=0.5):
        if feature_data.ndim == 1:
            feature_data = feature_data.reshape(-1, 1)
        if label.ndim == 1:
            label = label.reshape(-1, 1)
        self.input_validator= LogisticRegressionInputValidator(feature_data, label, threshold)
        self.feature_data=feature_data
        self.label=label
        self.learning_rate = learning_rate
        self.threshold= threshold
        data_shape = feature_data.shape
        self.sample_count = data_shape[0]
        self.feature_count = 1 if len(data_shape) == 1 else data_shape[1]
        self._weight= ModelWeightInitializer.generateRandomWeightsVector(self.feature_count)
        self._bias=0
        self._loss=[]

    def train(self, iterations, logTrainingParams= False):
        self._loss=[]
        for i in range(iterations):
            predicted_probability = self.__sigmoidFunction(np.dot(self.feature_data, self.weight) + self.bias)

            # Parital differentiation of cost function MSE J = sum(-ylog(y_pred) -(1-y)log(1-y_pred))/sample_count y in {0,1} y_pred 0 <= y <=1
            # def calculateBinaryCrossEntropyLoss(actual, predicted)
            # dJ/dW = X.T*(prediction - actual)/sample_count
            # dJ/dB = (prediction - actual)/sample_count

            error = predicted_probability-self.label
            dW = ((np.dot(self.feature_data , error))/self.sample_count)
            dJ = ((np.sum(error))/self.sample_count)

            self._weight = self._weight- self.learning_rate*dW
            self._bias = self._bias - self.learning_rate*dJ
            loss = self.calculateBinaryCrossEntropyLoss(self.label, predicted_probability)
            self._loss.append(loss)

            if logTrainingParams and i%1000 == 0:
                print("Iteration ", i , " weights ", self.weight, " bias ", self.bias, "loss" , loss)

    def predict(self, attributes):
        probs = self.__sigmoidFunction(np.dot(attributes, self.weight) + self.bias)
        return (probs >= self.threshold).astype(int)
    
    def __sigmoidFunction(self, z):
            z = np.asarray(z, dtype=np.float64)
            z = np.clip(z, -500, 500)                   
            return 1.0 / (1.0 + np.exp(-z))
    
    def calculateBinaryCrossEntropyLoss(self, actual, predicted):
        target = np.asarray(actual, dtype=np.float64)
        predicted = np.asanyarray(predicted, dtype=np.float64)

        if target.shape != predicted.shape:
            raise ValueError("Predicted vector and target vector lists don't match.")
        
        if not np.isin(target, [0, 1]).all():
            raise ValueError("Target contains values other than 0 or 1")
    
        if not ((predicted >= 0) & (predicted <= 1)).all():
            raise ValueError("Predicted contains values which are not in range 0 to 1")
        
        shape = target.shape
        total_loss = 0

        # for i in range(shape[0]):

        #     predicted_probability=predicted[i]
        #     actual_label = actual[i]
        #     total_loss = total_loss + self.__calculateLoss(actual_label, predicted_probability)

        # return total_loss
        
        epsilon = 1e-15  # to avoid log(0)
        predicted = np.clip(predicted, epsilon, 1 - epsilon)

        loss = -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
        return loss


    def __calculateLoss(self, actual : bool, predictedProbability: float):

        if actual:
            return -np.log(predictedProbability)
        else:
            return -np.log(1-predictedProbability)

    @property
    def bias(self):
        return self._bias       

    @property
    def weight(self):
        return self._weight
    
    @property
    def trainingLoss(self):
        return self._loss