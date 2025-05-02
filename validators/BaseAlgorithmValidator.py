from abc import ABC, abstractmethod
import numpy as np

class BaseAlgorithmValidator(ABC):

    def validateInputShape(self, feature_data, label):
        if feature_data is None or label is None:
            raise ValueError("Invalid input!")
        
        if not isinstance(feature_data, np.ndarray):
            raise ValueError("Not valid training data of numpy array input!")

        if not isinstance(label, np.ndarray):
            raise ValueError("Not valid label of array input!")

        if feature_data.ndim == 1:
            feature_data = feature_data.reshape(-1, 1)

        shape_features = feature_data.shape
        label_shape = label.shape

        if label.ndim == 1:
            label = label.reshape(-1, 1)

        if shape_features[0] != label_shape[0]:
            print("Feature shape:", shape_features, "Label shape:", label_shape)
            raise ValueError("Training data shape and label shape don't match")