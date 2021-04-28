from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np
import tensorflow as tf
from numpy import array
from pandas import DataFrame

import GNN.GNN_metrics as mt
from GNN.graph_class import GraphObject


class BaseGNN(ABC):
    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self,
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss_function: tf.keras.losses.Loss,
                 loss_arguments: Optional[dict],
                 addressed_problem: str,
                 extra_metrics: Optional[dict] = None,
                 extra_metrics_arguments: Optional[dict[str, dict]] = None,
                 path_writer: str = 'writer/',
                 namespace='GNN') -> None:
        """ CONSTRUCTOR - Other attributes must be defined in inheriting class

        :param optimizer: (tf.keras.optimizers) for gradient application, initialized externally.
        :param loss_function: (tf.keras.losses) for the loss computation.
        :param loss_arguments: (dict) with some {'argument':values} one could pass to loss when computed.
        :param addressed_problem: (str) in ['r','c'], 'r':regression, 'c':classification for the addressed problem.
        :param extra_metrics: None or dict {'name':function} for metrics to be watched during training/validation/test procedures.
        :param extra_metrics_arguments: None or dict {'name':{'argument':value}} for arguments passed to extra_metrics['name'].
        :param path_writer: (str) path for saving TensorBoard objects in training procedure. If folder is not empty, all files are removed.
        :param namespace: (str) namespace for tensorboard visualization.
        """
        # check types and values
        if addressed_problem not in ['c', 'r']: raise ValueError('param <addressed_problem> not in [\'c\',\'r\']')
        if not isinstance(extra_metrics, (dict, type(None))): raise TypeError('type of param <extra_metrics> must be None or dict')

        # set attributes
        self.loss_function = loss_function
        self.loss_args = dict() if loss_arguments is None else loss_arguments
        self.optimizer = optimizer

        # Problem type: c: Classification | r: Regression
        self.addressed_problem = addressed_problem

        # Metrics to be evaluated during training process
        self.extra_metrics = dict() if extra_metrics is None else extra_metrics
        self.mt_args = dict() if extra_metrics_arguments is None else extra_metrics_arguments

        # Writer and Namespace for Tensorboard - Nets histograms and Distributions
        if path_writer[-1] != '/': path_writer += '/'
        if type(namespace) != list: namespace = [namespace]
        if os.path.exists(path_writer): shutil.rmtree(path_writer)
        self.path_writer = path_writer
        self.namespace = namespace

        # history object (dict) - to summarize the training process, initialized as empty dict
        self.history = dict()

    ## ABSTRACT METHODS ###############################################################################################
    @abstractmethod
    def copy(self, *, path_writer: str = '', namespace: str = '', copy_weights: bool = True):
        """ COPY METHOD - defined in inheriting class

        :param path_writer: None or (str), to save copied gnn tensorboard writer.
        :param namespace: (str) for tensorboard visualization in model training procedure.
        :param copy_weights: (bool) True: state and output weights are copied in new model, otherwise they are re-initialized.
        :return: a Deep Copy of the GNN instance.
        """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def save(self, path: str) -> None:
        """ Save model to folder <path> """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def load(path: str, path_writer: str, namespace: str):
        """ Load model from folder

        :param path: (str) folder path containing all useful files to load the model
        :param path_writer: (str) path for writer folder. !!! Constructor method makes delete a non-empty folder and makes a new empty one.
        :param namespace: (str) namespace for tensorboard visualization of the model in training procedure.
        :return: the model
        """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def trainable_variables(self) -> tuple[list[list[tf.Tensor]], list[list[tf.Tensor]]]:
        """ Get tensor weights for net_state and net_output for each gnn layer """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def get_weights(self) -> tuple[list[list[array]], list[list[array]]]:
        """ Get array weights for net_state and net_output for each gnn layer """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def set_weights(self, weights_state: Union[list[array], list[list[array]]],
                    weights_output: Union[list[array], list[list[array]]]) -> None:
        """ Set weights for net_state and net_output """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def Loop(self, g: GraphObject, *, training: bool = False) -> tuple[int, tf.Tensor, tf.Tensor]:
        """ Process a single GraphObject element g, returning iteration, states and output """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def __call__(self, g: GraphObject):
        """ Return the mdoel output in test mode (training == False) for graph g of type GraphObject """
        pass

    ## HISTORY METHOD #################################################################################################
    def printHistory(self) -> None:
        """ Print self.history as a pd.Dataframe. Pandas automatically detects terminal width """
        print('\n', DataFrame(self.history), end='\n\n')

    # -----------------------------------------------------------------------------------------------------------------
    def saveHistory_csv(self, path) -> None:
        """ Save history attribute to .csv file. Extension can be omitted from path string """
        if path[-3:] != '.csv': path += '.csv'
        df = DataFrame(self.history)
        df.to_csv(path, index=False)

    # -----------------------------------------------------------------------------------------------------------------
    def saveHistory_txt(self, path) -> None:
        """ Save history attribute to .txt file. Extension can be omitted from path string """
        if path[-3:] != '.txt': path += '.txt'
        df = DataFrame(self.history)
        with open(path, 'w') as txt:
            txt.write(df.to_string(index=False))

    ## EVALUATE METHODs ###############################################################################################
    def evaluate_single_graph(self, g: GraphObject, class_weights: Union[int, float, list[float]], training: bool) -> tuple:
        """ Evaluate method for evaluating one GraphObjetc element g. Returns iteration, loss, targets and outputs """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    def evaluate(self, g: Union[GraphObject, list[GraphObject]], class_weights: Union[int, float, list[float]] = 1) -> tuple:
        """ Return ALL the metrics in self.extra_metrics + Iter & Loss for a GraphObject or a list of GraphObjects in test mode

        :param g: element/list of GraphObject to be evaluated.
        :param class_weights: (list) [w0, w1,...,wc] for classification task, specify the weight for weighted loss.
        :return: metrics, float(loss) target_labels, prediction_labels, targets_raw and prediction_raw.
        """
        # chech if inputs are GraphObject OR list(s) of GraphObject(s)
        if not (type(g) == GraphObject or (type(g) == list and all(isinstance(x, GraphObject) for x in g))):
            raise TypeError('type of param <g> must be GraphObject or list of GraphObjects')
        if type(g) == GraphObject: g = [g]

        # process input data
        iters, losses, targets, outs = zip(*[self.evaluate_single_graph(i, class_weights, training=False) for i in g])

        # concatenate all the values from every graph and take clas labels or values
        loss = tf.concat(losses, axis=0)
        targets = tf.concat(targets, axis=0)
        y_score = tf.concat(outs, axis=0)
        y_true = tf.argmax(targets, axis=1) if self.addressed_problem == 'c' else targets
        y_pred = tf.argmax(y_score, axis=1) if self.addressed_problem == 'c' else y_score

        # evaluate metrics
        metr = {k: float(tf.reduce_mean(self.extra_metrics[k](y_true, y_pred, **self.mt_args.get(k, dict())))) for k in self.extra_metrics}
        metr['It'] = int(tf.reduce_mean(iters))
        metr['Loss'] = float(tf.reduce_mean(loss))
        return metr, metr['Loss'], y_true, y_pred, targets, y_score

    ## TRAINING METHOD ################################################################################################
    def train(self, gTr: Union[GraphObject, list[GraphObject]], epochs: int, gVa: Union[GraphObject, list[GraphObject], None] = None,
              update_freq: int = 10, max_fails: int = 10, class_weights: Union[int, list[float]] = 1,
              *, mean: bool = True, verbose: int = 3) -> None:
        """ TRAINING PROCEDURE

        :param gTr: GraphObject or list of GraphObjects used for the learning procedure.
        :param epochs: (int) the max number of epochs for the learning procedure.
        :param gVa: element/list of GraphsObjects for early stopping. Default None, no early stopping performed.
        :param update_freq: (int) how many epochs must be completed before evaluating gVa and gTr and/or print learning progress. Default 10.
        :param max_fails: (int) specifies the max number of failures in gVa improvement loss evaluation before early sopping. Default 10.
        :param class_weights: (list) [w0, w1,...,wc] in classification task when targets are 1-hot, specify the weight for weighted loss. Default 1.
                                     > removed in future version
        :param mean: (bool) if False the applied gradients are computed as the sum of every iteration, otherwise as the mean. Default True.
        :param verbose: (int) 0: silent mode; 1: print history; 2: print epochs/batches, 3: history + epochs/batches. Default 3.
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        def update_history(name: str, val: dict[str, float]) -> None:
            """ update self.history with a dict s.t. val.keys()==self.history.keys()^{'Epoch','Best Loss Va'} """
            # name must be 'Tr' or 'Va', to update correctly training or validation history
            if name not in ['Tr', 'Va']: raise TypeError('param <name> must be \'Tr\' or \'Va\'')
            for key in val: self.history[f'{key} {name}'].append(val[key])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        def checktype(elem: Optional[Union[GraphObject, list[GraphObject]]]) -> list[GraphObject]:
            """ check if type(elem) is correct. If so, return None or a list og GraphObjects """
            if elem is None:
                pass
            elif type(elem) == GraphObject:
                elem = [elem]
            elif isinstance(elem, (list, tuple)) and all(isinstance(x, GraphObject) for x in elem):
                elem = list(elem)
            else:
                raise TypeError('Error - <gTr> and/or <gVa> are not GraphObject or LIST/TUPLE of GraphObjects')
            return elem

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        def reset_validation(valid_loss: float) -> tuple[float, int, list[list[array]], list[list[array]]]:
            """ reset the validation check parameters and to save the 'best weights until now' """
            wst, wout = self.get_weights()
            return valid_loss, 0, wst, wout

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        def training_step(gTr: GraphObject, mean: bool) -> None:
            """ compute the gradients and apply them """
            with tf.GradientTape() as tape:
                iter, loss, *_ = self.evaluate_single_graph(gTr, class_weights, training=True)
            wS, wO = self.trainable_variables()
            dwbS, dwbO = tape.gradient(loss, [wS, wO])
            # average net_state dw and db w.r.t. the number of iteration.
            if mean: dwbS = [[j / it for j in i] for it, i in zip([iter] if type(iter) != list else iter, dwbS)]
            # apply gradients
            zipped = zip([i for j in dwbS + dwbO for i in j], [i for j in wS + wO for i in j])
            self.optimizer.apply_gradients(zipped)

        ### TRAINING FUNCTION -----------------------------------------------------------------------------------------
        if verbose not in range(4): raise ValueError('param <verbose> not in [0,1,2,3]')

        # Checking type for gTr and gVa + Initialization of Validation parameters
        gTr, gVa = checktype(gTr), checktype(gVa)

        # initialize history attribute and writer directory
        if not self.history:
            keys = ['Epoch'] + [i + j for i in ['It', 'Loss'] + list(self.extra_metrics) for j in ([' Tr', ' Va'] if gVa else [' Tr'])]
            if gVa: keys += ['Fail', 'Best Loss Va']
            self.history.update({i: list() for i in keys})
            os.makedirs(self.path_writer)

        # Writers: Training, Validation (scalars) + Net_state, Net_output (histogram for weights/biases)
        netS_writer = tf.summary.create_file_writer(self.path_writer + 'Net - State')
        netO_writer = tf.summary.create_file_writer(self.path_writer + 'Net - Output')
        trainining_writer = tf.summary.create_file_writer(self.path_writer + 'Training')
        if gVa:
            lossVa = self.history['Best Loss Va'][-1] if self.history['Best Loss Va'] else float(1e30)
            vbest_loss, vfails, ws, wo = reset_validation(lossVa)
            validation_writer = tf.summary.create_file_writer(self.path_writer + 'Validation')

        # pre-Training procedure: check if it's the first learning time to correctly update tensorboard
        initial_epoch = self.history['Epoch'][-1] + 1 if self.history['Epoch'] else 0
        epochs += initial_epoch

        ### TRAINING PROCEDURE
        for e in range(initial_epoch, epochs):

            # TRAINING STEP
            for i, elem in enumerate(gTr):
                training_step(elem, mean=mean)
                if verbose > 2: print(f' > Epoch {e:4d}/{epochs} \t\t> Batch {i + 1:4d}/{len(gTr)}', end='\r')

            # TRAINING EVALUATION STEP
            if e % update_freq == 0:
                metricsTr, *_ = self.evaluate(gTr, class_weights)
                # History Update
                self.history['Epoch'].append(e)
                update_history('Tr', metricsTr)
                # TensorBoard Update Tr: Losses, Interation@Convergence, Accuracies + histograms of weights
                self.write_scalars(trainining_writer, metricsTr, e)
                for i, j, namespace in zip(*self.get_weights(), self.namespace):
                    self.write_net_weights(netS_writer, namespace, 'N1', i, e)
                    self.write_net_weights(netO_writer, namespace, 'N2', j, e)

            # VALIDATION STEP
            if (e % update_freq == 0) and gVa:
                metricsVa, lossVa, *_ = self.evaluate(gVa, class_weights)
                # Validation check
                if lossVa < vbest_loss:
                    vbest_loss, vfails, ws, wo = reset_validation(lossVa)
                else:
                    vfails += 1
                # History Update
                self.history['Best Loss Va'].append(vbest_loss)
                self.history['Fail'].append(vfails)
                update_history('Va', metricsVa)
                # TensorBoard Update Va: Losses, Interation@Convergence, Accuracies + histograms of weights
                self.write_scalars(validation_writer, metricsVa, e)
                # Early Stoping - reached max_fails for validation set
                if vfails >= max_fails:
                    self.set_weights(ws, wo)
                    if verbose in [1, 3]: self.printHistory()
                    print('\r Validation Stop')
                    break

            # PRINT HISTORY
            if (e % update_freq == 0) and verbose in [1, 3]: self.printHistory()

        else:
            print('\r End of Epochs Stop')

        # Tensorboard Update FINAL: write BEST WEIGHTS + BIASES
        for i, j, namespace in zip(*self.get_weights(), self.namespace):
            self.write_net_weights(netS_writer, namespace, 'N1', i, e)
            self.write_net_weights(netO_writer, namespace, 'N2', j, e)

    ## TEST METHOD ####################################################################################################
    def test(self, gTe: Union[GraphObject, list[GraphObject]], *, class_weights=1, acc_classes: bool = False, rocdir: str = '',
             micro_and_macro: bool = False, prisofsdir: str = '', pos_label=0) -> dict[str, list[float]]:
        """ TEST PROCEDURE

        :param gTe: element/list of GraphObjects for testing procedure.
        :param class_weights: (list) [w0, w1,...,wc] in classification task when targets are 1-hot, specify the weight for weighted loss. Default 1.
                                     > removed in future version.
        :param acc_classes: (bool) if True print accuracy for each class, in classification problems.
        :param rocdir: (str) path for saving ROC images file.
        :param micro_and_macro: (bool) for computing micro and macro average quantities in roc curve.
        :param prisofsdir: (str) path for saving Precision-Recall curve with ISO F-Score images file.
        :param pos_label: (int) for classification problems, identify the positive class.
        :return: metrics for gTe.
        """
        if type(gTe) != GraphObject and not (type(gTe) == list and all(isinstance(x, GraphObject) for x in gTe)):
            raise TypeError('type of param <gTe> must be GraphObject or list of GraphObjects')
        if not all(isinstance(x, str) for x in [rocdir, prisofsdir]):
            raise TypeError('type of params <roc> and <prisofs> must be str')

        # Evaluate all the metrics in gnn.extra_metrics + Iter and Loss
        metricsTe, lossTe, y_true, y_pred, targets, y_score = self.evaluate(gTe, class_weights=class_weights)

        # Accuracy per Class: shape = (1,number_classes)
        if acc_classes and self.addressed_problem == 'c':
            accuracy_classes = mt.accuracy_per_class(y_true, y_pred)
            metricsTe['Acc Classes'] = accuracy_classes.tolist()

        # ROC e PR curves
        if rocdir: mt.ROC(targets, y_score, rocdir, micro_and_macro, pos_label=pos_label)
        if prisofsdir: mt.PRISOFS(targets, y_score, prisofsdir, pos_label=pos_label)
        return metricsTe

    ## K-FOLD CROSS VALIDATION METHOD #################################################################################
    def LKO(self, dataset: Union[GraphObject, list[GraphObject], list[list[GraphObject]]],
            number_of_batches: int = 10, useVa: bool = False, seed: Optional[float] = None, normalize_method: str = 'gTr',
            node_aggregation: str = 'average', acc_classes: bool = False, epochs: int = 500, training_mode='parallel',
            update_freq: int = 10, max_fails: int = 10, class_weights: Union[int, float, list[Union[float, int]]] = 1,
            mean: bool = True, verbose: int = 3, pos_label=0) -> dict[str, list[float]]:
        """ LEAVE K OUT CROSS VALIDATION PROCEDURE

        :param dataset: a single GraphObjetc OR a list of GraphObject OR list of lists of GraphObject on which <gnn> has to be valuated
                        > NOTE: for graph-based problem, if type(dataset) == list of GraphObject,
                        s.t. len(dataset) == number of graphs in the dataset, then i-th class will may be have different frequencies among batches
                        [so the i-th class may me more present in a batch and absent in another batch].
                        Otherwise, if type(dataset) == list of lists, s.t. len(dataset) == number of classes AND len(dataset[i]) == number of graphs
                        belonging to i-th class, then i-th class will have the same frequency among all the batches
                        [so the i-th class will be as frequent in a single batch as in the entire dataset].
        :param number_of_batches: (int) define how many batches will be considered in LKO procedure.
        :param useVa: (bool) if True, Early Stopping is considered during learning procedure; None otherwise.
        :param seed: (int or None) for fixed-shuffle options.
        :param normalize_method: (str) in ['','gTr,'all'], see normalize_graphs for details. If equal to '', no normalization is performed.
        :param node_aggregation: (str) for node aggregation method during dataset creation. See GraphObject for details.
        :param acc_classes: (bool) return or not the accuracy for each class in metrics.
        :param epochs: (int) number of epochs for training <gnn>, the gnn will be trained for all the epochs.
        :param training_mode: (str) for lgnn cross validation procedure. LGNN.train() for details.
        :param update_freq: (int) specifies how many epochs must be completed before evaluating gVa and gTr.
        :param max_fails: (int) specifies the max number of failures before early sopping.
        :param class_weights: (list) [w0, w1,...,wc] for classification task, specify the weight for weighted loss.
        :param mean: (bool) if False the applied gradients are computed as the sum of every iteration, else as the mean.
        :param verbose: (int) 0: silent mode; 1: print history; 2: print epochs/batches, 3: history + epochs/batches. Default 3.
        :param pos_label: (int) for classification problems, identify the positive class.
        :return: a dict containing all the considered metrics in <gnn>.history.
        """
        from GNN.GNN_utils import normalize_graphs, getbatches
        from numpy import random, arange, array_split
        from GNN.GNN import GNNnodeBased, GNNedgeBased, GNNgraphBased
        from GNN.LGNN.LGNN import LGNN

        # Shuffling procedure: set or not seed parameter, then shuffle classes and/or elements in each class/dataset
        if seed: random.seed(seed)

        # define useful lambda function to be used in any case
        flatten = lambda l: [item for sublist in l for item in sublist]

        # define lists for LKO
        gTRs, gTEs, gVAs = list(), list(), list()

        # SINGLE GRAPH CASE: batches are obtaind by setting set_masks for training, test and validation (if any)
        if isinstance(dataset, GraphObject):
            dataset.set_mask = np.zeros(len(dataset.set_mask), dtype=bool)

            mask_indicess = arange(len(dataset.set_mask))
            random.shuffle(mask_indicess)
            masks = array_split(mask_indicess, number_of_batches)

            for i in range(len(masks)):
                M = masks.copy()

                # test batch
                mTe = M.pop(i)
                gTe = dataset.copy()
                gTe.set_mask[mTe] = True

                # validation batch
                gVa = None
                if useVa:
                    mVa = M.pop(-1)
                    gVa = dataset.copy()
                    gVa.set_mask[mVa] = True

                # training batch - all the others
                mTr = flatten(M)
                gTr = dataset.copy()
                gTr.set_mask[mTr] = True

                # append batch graphs
                gTRs.append(gTr)
                gTEs.append(gTe)
                gVAs.append(gVa)

        # MULTI GRAPH CASE: dataset is a list of graphs or a list of lists of graphs. :param dataset_ for details
        elif isinstance(dataset, list):
            # check type if dataset is a list
            if all(isinstance(i, GraphObject) for i in dataset): dataset = [dataset]
            assert all(len(i)>number_of_batches for i in dataset)
            assert all(isinstance(i, list) for i in dataset) and all(isinstance(j, GraphObject) for i in dataset for j in i)

            # shuffle entire dataset or classes sub-dataset
            for i in dataset: random.shuffle(i)

            # get problem_based param
            problems = {GNNnodeBased: 'n', GNNedgeBased: 'a', GNNgraphBased: 'g'}
            if type(self) == LGNN:  problem_based = problems[self.GNNS_TYPE]
            else: problem_based = problems[type(self)]

            # get dataset batches and flatten lists to obtain a list of lists, then shuffle again to mix classes inside batches
            dataset_batches = [getbatches(elem, problem_based, node_aggregation, -1, number_of_batches, False) for i, elem in enumerate(dataset)]
            flattened = [flatten([i[j] for i in dataset_batches]) for j in range(number_of_batches)]
            for i in flattened: random.shuffle(i)

            # Final dataset for LKO procedure: merge graphs belonging to classes/dataset to obtain 1 GraphObject per batch
            dataset = [GraphObject.merge(i, problem_based=problem_based, node_aggregation=node_aggregation) for i in flattened]

            # split dataset in training/validation/test set
            for i in range(len(dataset)):
                gTr = dataset.copy()
                gTe = gTr.pop(i)
                gVa = gTr.pop(-1) if useVa else None

                # append batch graphs
                gTRs.append(gTr)
                gTEs.append(gTe)
                gVAs.append(gVa)

        else:
            raise TypeError('param <dataset> must be a GraphObject, a list of GraphObjects or a list of lists of Graphobjects')

        # initialize results
        metrics = {i: list() for i in list(self.extra_metrics) + ['It', 'Loss']}
        if acc_classes: metrics['Acc Classes'] = list()

        # If model is LGNN, integrate training_mode in training procedure as kwargs dict
        kwargs = dict()
        if type(self) == LGNN: kwargs['training_mode'] = training_mode

        # LKO PROCEDURE
        for i, gTr, gTe, gVa in zip(range(number_of_batches), gTRs, gTEs, gVAs):

            # normalization procedure
            if normalize_method: normalize_graphs(gTr, gVa, gTe, based_on=normalize_method)

            # model creation, learning and test
            print(f'\nBATCH K-OUT {i + 1}/{number_of_batches}')
            temp = self.copy(copy_weights=False, path_writer=self.path_writer + str(i), namespace=f'Batch {i + 1}-{number_of_batches}')
            temp.train(gTr, epochs, gVa, update_freq, max_fails, class_weights, mean=mean, verbose=verbose, **kwargs)
            res = temp.test(gTe, acc_classes=acc_classes, pos_label=pos_label)

            # evaluate metrics
            for m in res: metrics[m].append(res[m])

            # verbose option
            if verbose > 1: print(f'\nRESULTS BATCH {i + 1}/{number_of_batches}\n', DataFrame(res, index=['res']).transpose())
        return metrics

    ## STATIC METHODs #################################################################################################
    @staticmethod
    def get_graph_target(g):
        """ Get targets for node-based or edge-based problems: nodes states are filtered by set_mask and output_mask """
        targs = tf.constant(g.getTargets(), dtype=tf.float32)
        mask = tf.boolean_mask(g.set_mask, g.output_mask)
        return tf.boolean_mask(targs, mask)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def ArcNode2SparseTensor(ArcNode) -> tf.Tensor:
        """ Get the transposed sparse tensor of the ArcNode matrix """
        # ArcNode Tensor, then reordered to be correctly computable. NOTE: reorder() recommended by TF2.0+
        indices = [[ArcNode.row[i], ArcNode.col[i]] for i in range(ArcNode.shape[0])]
        arcnode = tf.SparseTensor(indices, values=ArcNode.data, dense_shape=ArcNode.shape)
        arcnode = tf.sparse.transpose(arcnode)
        arcnode = tf.sparse.reorder(arcnode)
        arcnode = tf.cast(arcnode, dtype=tf.float32)
        return arcnode

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def write_scalars(writer: tf.summary.SummaryWriter, metrics: dict[str, float], epoch: int) -> None:
        """ TENSORBOARD METHOD: writes scalars values of the metrics """
        if type(metrics) != dict: raise TypeError('type of param <metrics> must be dict')
        names = {'Acc': 'Accuracy', 'Bacc': 'Balanced Accuracy', 'Ck': 'Cohen\'s Kappa', 'Js': 'Jaccard Score',
                 'Fs': 'F1-Score', 'Prec': 'Precision Score', 'Rec': 'Recall Score', 'Tpr': 'TPR', 'Tnr': 'TNR',
                 'Fpr': 'FPR', 'Fnr': 'FNR', 'Loss': 'Loss', 'It': 'Iteration @ Convergence'}

        namescopes = {**{i: 'Accuracy & Loss' for i in ['Acc', 'Bacc', 'It', 'Loss']},
                      **{i: 'F-Score, Precision and Recall' for i in ['Fs', 'Prec', 'Rec']},
                      **{i: 'Positive and Negative Rates' for i in ['Tpr', 'Tnr', 'Fpr', 'Fnr']},
                      **{i: 'Other Scores' for i in ['Ck', 'Js']}}

        with writer.as_default():
            for i in metrics:
                name = names.get(i, i)
                with tf.name_scope(namescopes.get(i, 'Other Scores')):
                    tf.summary.scalar(name, metrics[i], step=epoch, description=name)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def write_net_weights(writer: tf.summary.SummaryWriter, namespace: str, net_name: str, val_list: list[array], epoch: int) -> None:
        """ TENSORBOARD METHOD: writes histograms of the nets weights """
        W, B, names_layers = val_list[0::2], val_list[1::2], [f'{net_name} L{i}' for i in range(len(val_list) // 2)]
        assert len(names_layers) == len(W) == len(B)

        with writer.as_default():
            for n, w, b in zip(names_layers, W, B):
                with tf.name_scope(f'{namespace}: Weights'):
                    tf.summary.histogram(n, w, step=epoch)
                with tf.name_scope(f'{namespace}: Biases'):
                    tf.summary.histogram(n, b, step=epoch)