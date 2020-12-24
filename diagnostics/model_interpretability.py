import matplotlib.pyplot as plt
import numpy as np


def plotFeatureImportance(features, importances, additionalTitle= '', figsize = (15,10)):
    indices = np.argsort(importances)
    plt.figure(figsize = figsize)
    plt.title('Feature Importance ' + additionalTitle)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.show()