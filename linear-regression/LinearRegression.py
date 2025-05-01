import numpy as np

class LinearRegression:
    
    def __init__(self, feature_data, label):
        self.__check(feature_data, label)
        self.feature_data=np.arrafeature_data
        self.label=label
    
    def __train():
        pass

    def __check(self, feature_data, label):
        if feature_data == None or label == None:
            raise ValueError("Invalid input!")
        
        if isinstance(feature_data, np.ndarray):
            raise ValueError("Not valid training data of numpy array input!")

        if isinstance(label, np.ndarray):
            raise ValueError("Not valid label of array input!")
        
        shape_features = feature_data.shape
        label_shape = label.shape

        if shape_features[1]== label_shape[0]:
            raise ValueError("Training data shape and label shape don't match")


lin = LinearRegression(np.array([1,2]), np.array([2,3]))