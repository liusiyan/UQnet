''' Functions for plotting '''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
# print(plt.get_backend())


class CL_plotter:
    def __init__(self):
        pass
    def plotTrainValidationLoss(self, train_loss, valid_loss, test_loss=None,
                                trainPlotLabel=None, validPlotLabel=None, testPlotLabel=None,
                                xlabel=None, ylabel=None,
                                title=None,
                                gridOn=False, legendOn=True,
                                xlim=None, ylim=None,
                                saveFigPath=None):

        pass


        # iter_arr = np.arange(len(train_loss))
        # plt.plot(iter_arr, validation_loss, label=validPlotLabel)
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        # plt.title(title)
        # if gridOn is True:
        #     plt.grid()
        # if legendOn is True:
        #     plt.legend()
        # if xlim is not None:
        #     plt.xlim(xlim)
        # if ylim is not None:
        #     plt.ylim(ylim)
        # if saveFigPath is not None:
        #     plt.savefig(saveFigPath)
        #     plt.clf()
        # if saveFigPath is None:
        #     plt.show()
