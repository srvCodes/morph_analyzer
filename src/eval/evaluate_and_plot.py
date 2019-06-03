from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals.funcsigs import signature
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize

feature_cnt = 0
feature_map = {0:'POS', 1:'gender', 2:'number', 3:'person', 4:'case', 5:'TAM'}

class EvaluatePerformance():
    def __init__(self, words, root_outputs, orig_features, pred_features,  classes):
        self.orig_roots, self.pred_roots = root_outputs
        self.orig_features = orig_features
        self.pred_features = pred_features
        self.classes = classes
        self.words = words


    def p_r_curve_plotter(self, lang='hindi'):
        for orig, pred, list_of_classes in zip(self.orig_features, self.pred_features, self.classes):
            binarized_orig_features = self.binarize(orig, list_of_classes)
            _ = self.plot_curve(binarized_orig_features, pred, list_of_classes, lang=lang)
            # input()

    @staticmethod
    def binarize(Y, c):
        res = label_binarize(Y, classes=c)
        return res

    @staticmethod
    def plot_curve(Y, f, c, lang='hindi'):
        precision = dict()
        recall = dict()
        average_precision = dict()
        n_classes = len(c)
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y[:, i], f[:, i])
            average_precision[i] = average_precision_score(Y[:, i], f[:, i])
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y.ravel(),
                                                                        f.ravel())
        average_precision["micro"] = average_precision_score(Y, f,
                                                             average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'
              .format(average_precision["micro"]))
        # Plot average precision_recall_curve
        plt.figure()
        # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall['micro'], precision['micro'], color='r', alpha=0.2,
                 where='post')
        plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                         **step_kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Average precision score, micro-averaged over all Gender tags: AP={0:0.2f}'
                  .format(average_precision["micro"]))
        # plot precision-recall curve for each class and iso-f1 curves
        # setup plot details
        colors = cycle(['navY', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
        plt.figure(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                      ''.format(average_precision["micro"]))
        for i, color in zip(range(n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(i, average_precision[i]))
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
        global feature_cnt
        plt.savefig('graph_outputs/'+lang+'/' + feature_map[feature_cnt] + '_curve')
        feature_cnt += 1
        plt.show()




