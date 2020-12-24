import numpy as np

def tree_optimization_single_variable(variable_thresholds, target, variable):
    '''
    Computes variance at various levels of thresholds for a tree split. Mimicks manually optimization process of a tree
    :param variable_thresholds: (List-like)  threshold values to split variable parameter
    :param target: (array or pandas series) target value for regression/classification
    :param variable: (array or pandas series) variable to split on
    :return: variances (list in order of thresholds provided)
    use case example:
    variable_thresholds = range(0,4,2)
    target = [10, 50]
    variable = [3, 8]
    variances = tree_optimization_single_variable(variable_thresholds, target, variable)
    plt.plot(threshold, variances)

    '''
    variances = []
    for threshold in variable_thresholds:
        grouping = variable > threshold
        varianceGroup1 = np.var(target[grouping])
        varianceGroup2 = np.var(target[~grouping])
        variance = varianceGroup1 + varianceGroup2
        variances.append(variance)
    return variances


