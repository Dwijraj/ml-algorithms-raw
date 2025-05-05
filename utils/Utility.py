from abc import ABC
import numpy as np

class Utility(ABC):

    @staticmethod
    def createOneHotEncoding(labels):
        
        labels = np.asarray(labels, dtype=int).ravel()
        unique_labels, inverse = np.unique(labels, return_inverse=True)
        num_classes = unique_labels.size
        one_hot = np.zeros((labels.size, num_classes), dtype=int)
        one_hot[np.arange(labels.size), inverse] = 1

        return num_classes, one_hot, unique_labels       

    @staticmethod
    def createOneHotEncodingTop1(scores, dtype=int):
        scores = np.asarray(scores)
        if scores.ndim == 1:
            # single vector → return 1-D one-hot
            k = scores.argmax()
            one_hot = np.zeros_like(scores, dtype=dtype)
            one_hot[k] = 1
            return one_hot

        elif scores.ndim == 2:
            # batch of NxK → return N×K one-hot
            N, K = scores.shape
            indices = scores.argmax(axis=1)         # shape (N,)
            one_hot = np.zeros((N, K), dtype=dtype)
            one_hot[np.arange(N), indices] = 1
            return one_hot
   