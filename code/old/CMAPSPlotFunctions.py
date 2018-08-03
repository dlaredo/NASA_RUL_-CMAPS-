import matplotlib.pyplot as plt


def plotRUL(cycles, rulArray, pred, engineUnit, methodName):
	"""plot the trend of the predictions made by the predictor"""
    
    plt.clf()
    plt.plot(cycles, rulArray, 'bo-', label='RUL')
    plt.plot(cycles, nnPred, 'go-', label=methodName)
    plt.legend()
    plt.xlabel("Time (Cycle)")
    plt.ylabel("RUL")
    plt.title("Test Engine Unit #{}".format(engineUnit))
    plt.show()