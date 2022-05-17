import os
import sys
import traceback
import numpy as np
from termcolor import colored
try:
    from tensorflow.keras.callbacks import *
except:
    class Callback():
        def __init__(self):
            return
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore")
    import matplotlib.pyplot as plt
    import matplotlib
    # matplotlib.use("wxAgg")
    matplotlib.rcParams["toolbar"] = "toolmanager"

from matplotlib.backend_tools import ToolBase

class HistoryPlotCallback(Callback):
    def __init__(self,
                 modelName="",
                 folder=None,
                 verbose=1,
                 showDialog=True,
                 printcolor="yellow",
                 backColor = (0.2, 0.2, 0.2),
                 foreColor = "white",
                 trainColor="orange",
                 goodValColor="yellowgreen",
                 badValColor="salmon",
                 epochsBetweenUpdates=10,
                 firstEpochsToSkip=0,
                 description=None,
                 plotAverage=False,
                 earlyStopPatience=100,
                 saveModelCallback=None,
                 earlyStopMinEpochsBetweenCallback=10,
                 invokeSaveModelCallbackOnTrain=False,
                 model=None):

        if not hasattr(self, "model"):
            self.model = model
        self.modelName = modelName
        self.folder = folder
        self.figOK = False
        self.fig = None
        self.description = description
        self.axes = None
        self.showDialog = showDialog
        self.backColor = backColor
        self.foreColor = foreColor
        self.trainColor = trainColor
        self.goodValColor = goodValColor
        self.badValColor = badValColor
        self.printcolor = printcolor
        self.subplotsCount = 0
        self.verbose = verbose
        self.epochsBetweenUpdates = epochsBetweenUpdates
        # self.totalEpochsCount = -1
        self.firstEpochsToSkip = firstEpochsToSkip
        self.hshMetrics = {}
        self.hshBestMetricValues = {}
        self.hshSubPlots = {}
        self.epochs = []
        self.plotAverage = plotAverage
        self.newValBest = False
        self.earlyStopPatience = earlyStopPatience
        self.saveModelCallback = saveModelCallback
        self.invokeSaveModelCallbackOnTrain = invokeSaveModelCallbackOnTrain
        self.earlyStopEpochWithoutImprovement = 0
        self.earlyStopMinEpochsBetweenCallback = earlyStopMinEpochsBetweenCallback
        self.earlyStopEpochsSinceLastCallback = 0
        self.earlyStopBestWeights = self.__getModelWeights()
        plt.rcParams.update({'font.size': 8})

    def createMetrics(self, logs):
        if len(self.hshMetrics) > 0:
            return

        for metric in logs:
            if "val_" + metric in logs:
                self.hshSubPlots[metric] = self.subplotsCount
                self.hshMetrics[metric] = []
                self.hshMetrics["val_" + metric] = []
                self.hshMetrics["val_g_" + metric] = []
                self.hshMetrics["val_b_" + metric] = []

                self.hshBestMetricValues[metric] = None
                self.hshBestMetricValues["val_" + metric] = None

                # adds additional plots
                i = 0
                while metric + str(i) in logs:
                    self.hshMetrics[metric + str(i)] = []
                    i += 1

                self.subplotsCount += 1

    def addMetrics(self, logs):
        accuracyExists = False
        for metric in logs:
            if "accuracy" in metric:
                accuracyExists = True

        for metric in logs:
            if "val_" + metric in logs:
                metricValue = logs[metric]
                valMetricValue = logs["val_" + metric]
                self.hshMetrics[metric].append(metricValue)
                self.hshMetrics["val_" + metric].append(valMetricValue)

                # adds additional plots
                i = 0
                while metric + str(i) in logs:
                    self.hshMetrics[metric + str(i)].append(logs[metric + str(i)])
                    i += 1

                # we have to identify when up or down are good for each metric
                # this is temporary until we have a better way
                f = 1
                currentIsAccuracy = False
                if "accuracy" in metric:
                    currentIsAccuracy = True
                    f = -1

                # store best losses/accuracies for train
                if self.hshBestMetricValues[metric] is None or \
                    f * metricValue < f * self.hshBestMetricValues[metric]:
                    self.hshBestMetricValues[metric] = metricValue
                    if currentIsAccuracy or not accuracyExists:
                        if self.invokeSaveModelCallbackOnTrain:
                            self.newValBest = True
                            # print("self.newValBest=True1 " + metric)

                # store best losses/accuracies for val
                if self.hshBestMetricValues["val_" + metric] is None or \
                   f * valMetricValue < f * self.hshBestMetricValues["val_" + metric]:
                    # print("self.newValBest=True2 " + metric + " " + str(valMetricValue) + " " + str(self.hshBestMetricValues["val_" + metric]))
                    self.hshBestMetricValues["val_" + metric] = valMetricValue
                    if currentIsAccuracy or not accuracyExists:
                        self.newValBest = True

                # handle coloring the val plot in red/green according to its relation to train plot
                if f * valMetricValue <= f * metricValue:
                    if len(self.hshMetrics["val_g_" + metric]) > 0 and self.hshMetrics["val_g_" + metric][-1] is np.nan:
                        self.hshMetrics["val_g_" + metric][-1] = self.hshMetrics["val_b_" + metric][-1]
                    self.hshMetrics["val_g_" + metric].append(valMetricValue)
                    self.hshMetrics["val_b_" + metric].append(np.nan)
                else:
                    if len(self.hshMetrics["val_b_" + metric]) > 0 and self.hshMetrics["val_b_" + metric][-1] is np.nan:
                        self.hshMetrics["val_b_" + metric][-1] = self.hshMetrics["val_g_" + metric][-1]
                    self.hshMetrics["val_g_" + metric].append(np.nan)
                    self.hshMetrics["val_b_" + metric].append(valMetricValue)

    def callSaveCallback(self, force=False):
        if self.newValBest and \
           (force or self.earlyStopEpochsSinceLastCallback >= self.earlyStopMinEpochsBetweenCallback):
            if self.saveModelCallback is not None:
                currentWeights = self.__getModelWeights()
                self.__setModelWeights(self.earlyStopBestWeights)
                self.saveModelCallback(self.epochs[-1])
                # self.figOK = False
                self.__setModelWeights(currentWeights)
            self.earlyStopEpochsSinceLastCallback = 0
        self.newValBest = False
        # print("self.newValBest=False")

    def __getModelWeights(self):
        if self.model is None:
            return None
        return self.model.get_weights()

    def _StopButton__setModelWeights(self, weights):
        self.__setModelWeights(weights)

    def __setModelWeights(self, weights):
        if self.model is None:
            return
        self.model.set_weights(weights)

    def getMetricAsString(self, metric, logs):
        return "    " + metric + ": " + str(round(logs[metric], 3)) + " (best: " + str(round(self.hshBestMetricValues[metric], 3)) + ")"

    def getAllMetricsAsString(self, logs):
        msg1 = ""
        msg2 = ""
        for metric in logs:
            if "val_" + metric in logs:
                msg1 += self.getMetricAsString(metric, logs)
                msg2 += self.getMetricAsString("val_" + metric, logs)
        return msg1 + "  " + msg2

    def on_batch_end(self, batch, logs=None):
        if batch % 10 == 0:
            # plt.pause(0.0001)
            mypause(0.0001)

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.firstEpochsToSkip:
            return

        # self.totalEpochsCount = self.params["epochs"]
        self.epochs.append(epoch)

        self.createMetrics(logs)
        self.addMetrics(logs)

        if self.newValBest:
            self.earlyStopBestWeights = self.__getModelWeights()
            self.earlyStopEpochWithoutImprovement = 0
        else:
            # if no new best losses/accuracies for var
            self.earlyStopEpochWithoutImprovement += 1

        self.earlyStopEpochsSinceLastCallback += 1
        # only if we have a new val best AND more then
        # earlyStopMinEpochsBetweenCallback epochs have been executed since last callback
        self.callSaveCallback()

        if self.earlyStopEpochWithoutImprovement >= self.earlyStopPatience:
            # if we have new best - call the callback before stopping
            self.callSaveCallback(True)
            if self.earlyStopBestWeights is not None:
                self.__setModelWeights(self.earlyStopBestWeights)
            self.model.stop_training = True

        if self.verbose >= 1:
            msgEpochs = "Epochs " + str(epoch+1)  # + "/" + str(self.totalEpochsCount)

        if self.verbose >= 2:
            msg = msgEpochs + self.getAllMetricsAsString(logs)
            sys.stdout.write("\r")
            print(colored(msg, self.printcolor), end=" ", flush=True)

        if self.verbose >= 1:
            try:
                if (epoch + 1) % self.epochsBetweenUpdates == 0:
                    if not self.figOK:
                        if False:
                            height_ratios = [1] * (self.subplotsCount + 1)
                            height_ratios[-1] = 0.25
                            self.fig, self.axes = plt.subplots(self.subplotsCount + 1, gridspec_kw={'height_ratios': height_ratios})
                        else:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                self.fig = plt.figure("HistoryWindow")
                                self.axes = self.fig.subplots(self.subplotsCount)

                        self.fig.set_facecolor(self.backColor)
                        # this is just a trick so that referencing self.axes[0] will always be valid
                        if self.subplotsCount == 1:
                            self.axes = [self.axes]
                        # self.fig.suptitle("\n" + self.description + "\n")

                        # import warnings
                        # with warnings.catch_warnings():
                        #     warnings.simplefilter("ignore")
                        self.fig.canvas.manager.toolmanager.add_tool('stop', StopButton, self)
                        self.fig.canvas.manager.toolbar.add_tool(self.fig.canvas.manager.toolmanager.get_tool("stop"), "toolgroup")

                    self.fig.canvas.set_window_title(self.modelName + " " + msgEpochs)
                    # win = plt.gcf().canvas.manager.window
                    # win.overrideredirect(1)  # draws a completely frameless window

                    for metric in logs:
                        if "val_" + metric in logs:
                            val_metric = "val_" + metric
                            lineWidth = 0.8
                            markerSize = 1.6
                            ax = self.axes[self.hshSubPlots[metric]]
                            ax.clear()
                            ax.plot(self.epochs, self.hshMetrics[metric], label=metric, color=self.trainColor, marker=".", markersize=markerSize, linewidth=lineWidth)

                            # plots additional plots
                            i = 0
                            import random
                            while metric + str(i) in self.hshMetrics:
                                random.seed(i)
                                color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
                                ax.plot(self.epochs, self.hshMetrics[metric + str(i)], label=metric + str(i), color=color, marker=".", markersize=markerSize, linewidth=lineWidth)
                                i += 1

                            ax.plot(self.epochs, self.hshMetrics["val_g_" + metric], label="good " + val_metric, color=self.goodValColor, marker=".", markersize=markerSize, linewidth=lineWidth)
                            ax.plot(self.epochs, self.hshMetrics["val_b_" + metric], label="bad " + val_metric, color=self.badValColor, marker=".", markersize=markerSize, linewidth=lineWidth)

                            title = "epochs: " + str(len(self.epochs)) + "   " + self.getMetricAsString(metric, logs) + self.getMetricAsString(val_metric, logs)
                            if self.plotAverage:
                                avg = np.average([self.hshMetrics["val_" + metric], self.hshMetrics[metric]], axis=0)
                                ax.plot(self.epochs, avg, label="avg", color="skyblue", marker=".", markersize=markerSize, linewidth=lineWidth)
                                title += "  avg: " + str(round(avg[-1], 3))
                            ax.set_title(title, color=self.foreColor, fontsize=7)

                            # auto scale y axis so that MOST points are visible (omitting anomalies)
                            plots = [self.hshMetrics[metric], self.hshMetrics["val_g_" + metric], self.hshMetrics["val_b_" + metric]]
                            avg = 0
                            std = 0
                            plotCount = 0
                            for plot in plots:
                                if len(plot) > 0:
                                    arrPlot = np.array(plot)
                                    arrPlot = arrPlot[~np.isnan(arrPlot)]
                                    if len(arrPlot) > 0:
                                        import math
                                        arrPlot = np.log2(arrPlot)
                                        # avg += (np.median(arrPlot))  # + math.log2(abs(np.average(arrPlot))))/2
                                        std += np.median(np.abs(arrPlot-np.median(arrPlot)))
                                        avg += np.median(arrPlot)  # + math.log2(abs(np.average(arrPlot))))/2
                                        # std = max(std, np.std(arrPlot))
                                        plotCount += 1
                            avg /= plotCount
                            std /= plotCount
                            avg = 2**avg
                            std = 2**std
                            try:
                                ax.set_ylim([avg-2.5*std, avg+2.5*std])
                            except:
                                dummy = 0

                    if self.description is not None:
                        self.axes[-1].text(-0.1, -0.4, str(self.description), bbox=dict(facecolor='white', alpha=1), wrap=True, fontsize=6, transform=self.axes[-1].transAxes)

                    if not self.figOK:
                        for metric in logs:
                            if "val_" + metric in logs:
                                ax = self.axes[self.hshSubPlots[metric]]
                                lgnd = ax.legend(facecolor=self.backColor)
                                for text in lgnd.get_texts():
                                    text.set_color(self.foreColor)
                                ax.set_facecolor(self.backColor)
                                ax.tick_params(color=self.foreColor, labelcolor=self.foreColor, labelsize=7)
                                if self.hshSubPlots[metric] < self.subplotsCount-1:
                                    ax.get_xaxis().set_visible(False)
                                for spine in ax.spines.values():
                                    spine.set_edgecolor(self.foreColor)
                                ax.grid(True, linestyle='--', alpha=0.4)
                        # self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                        plt.subplots_adjust(top=0.95)
                        if self.showDialog:
                            plt.show(block=False)
                        self.figOK = True

                    # plt.draw()
                    mypause(0.001)

                    if self.folder is not None:
                        self.fig.savefig(self.folder + "history_" + self.modelName, facecolor=self.fig.get_facecolor(), transparent=True, edgecolor='none')
            except:
                print(colored(traceback.format_exc(), "red"))
                plt.close(self.fig)
                self.figOK = False

def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw_idle()
            canvas.start_event_loop(interval)
            return


class StopButton(ToolBase):
    def __init__(self, toolmanager, name, callback: HistoryPlotCallback):
        super(ToolBase, self).__init__()
        self.callback = callback
        self._name = name

    def trigger(self, *args, **kwargs):
        self.callback.callSaveCallback(True)
        self.callback.__setModelWeights(self.callback.earlyStopBestWeights)
        self.callback.model.stop_training = True


# example (keras):
# model.fit(..., callbacks=[HistoryPlotCallback()])