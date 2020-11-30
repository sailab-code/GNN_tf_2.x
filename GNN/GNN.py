# coding=utf-8
import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import GNN.GNN_metrics as mt
import GNN.GNN_utils as utils
from GNN.graph_class import GraphObject
pd.options.display.max_rows = 15


#######################################################################################################################
### CLASS GNN - NODE BASED ############################################################################################
#######################################################################################################################
class GNN:
    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self, net_state, net_output, optimizer, loss_function,
                 loss_arguments: dict = None,
                 output_activation=None,
                 max_iteration: int = 30,
                 threshold: float = 0.1,
                 path_writer: str = 'writer/',
                 addressed_problem: str = 'c',
                 extra_metrics=None,
                 metrics_arguments=None,
                 state_vect_dim: int = 0):
        """ CONSTRUCTOR
        :param net_state: (tf.keras.model.Sequential) MLP for the state network, initialized externally
        :param net_output: (tf.keras.model.Sequential) MLP for the output network, initialized externally
        :param optimizer: (tf.keras.optimizers) for gradient application, initialized externally
        :param loss_function: (tf.keras.losses) or (tf.function) for the loss computation
        :param loss_arguments: (dict) with some {'argument':values} one could pass to loss when computed
        :param output_activation: (tf.keras.activation) function in case net_output.layers[-1] is 'linear'
        :param max_iteration: (int) max number of iteration for the unfolding procedure (to reach convergence)
        :param threshold: threshold for specifying if convergence is reached or not
        :param path_writer: (str) path for saving TensorBoard objects
        :param addressed_problem: (str) in ['r','c'], 'r'_regression, 'c':classification for the addressed problem
        :param extra_metrics: None or dict {'name':function} for metrics to be watched during training/validaion/test
        :param state_vect_dim: None or (int)>=0, vector dim for a GNN which does not initialize states with node labels
        """
        # check types and values
        if addressed_problem not in ['c', 'r']: raise ValueError('param <addressed_problem> not in [\'c\',\'r\']')
        if type(state_vect_dim) != int or state_vect_dim < 0: raise TypeError('param <state_vect_dim> must be int>=0')
        if not isinstance(extra_metrics, (dict, type(None))):
            raise TypeError('type of param <extra_metrics> must be None or dict')
        # parameters and hyperparameters
        self.net_state = net_state
        self.net_output = net_output
        self.loss_function = loss_function
        self.loss_args = dict() if loss_arguments is None else loss_arguments
        self.optimizer = optimizer
        self.max_iteration = max_iteration
        self.state_threshold = threshold
        self.state_vect_dim = state_vect_dim
        # check last activation function: in case loss works with logits, it is set by <output_activation> parameter
        if output_activation: self.output_activation = output_activation
        else: self.output_activation = tf.keras.activations.linear
        # Writer for Tensorboard - Nets histograms and Distributions
        self.path_writer = path_writer if path_writer[-1] == '/' else path_writer + '/'
        if os.path.exists(self.path_writer): shutil.rmtree(self.path_writer)
        os.makedirs(self.path_writer)
        # Problem type: c: Classification | r: Regression
        self.addressed_problem = addressed_problem
        # Metrics to be evaluated during training process
        self.extra_metrics = dict() if extra_metrics is None else extra_metrics
        self.mt_args = metrics_arguments if metrics_arguments else dict()
        # history object (dict) - to summarize the training process
        keys = ['Epoch', 'It Tr', 'It Va', 'Fail', 'Best Loss Va', 'Loss Tr', 'Loss Va']
        keys += [i + j for i in self.extra_metrics for j in [' Tr', ' Va']]
        self.history = {k: list() for k in keys}

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self, *, path_writer: str = '', copy_weights: bool = True):
        """ COPY METHOD
        :param path_writer: None or (str), to save copied gnn writer. Default is in the same folder + '_copied'
        :param copy_weights: (bool) True: copied_gnn.nets==self.nets; False: state and output are re-initialized
        :return: a Deep Copy of the GNN instance.
        """
        netS = tf.keras.models.clone_model(self.net_state)
        netO = tf.keras.models.clone_model(self.net_output)
        if copy_weights:
            netS.set_weights(self.net_state.get_weights())
            netO.set_weights(self.net_output.get_weights())
        return self.__class__(net_state=netS,
                              net_output=netO,
                              optimizer=self.optimizer.__class__(**self.optimizer.get_config()),
                              loss_function=self.loss_function,
                              loss_arguments=self.loss_args,
                              output_activation=self.output_activation,
                              max_iteration=self.max_iteration,
                              threshold=self.state_threshold,
                              path_writer=path_writer if path_writer else self.path_writer + '_copied/',
                              addressed_problem=self.addressed_problem,
                              extra_metrics=self.extra_metrics,
                              metrics_arguments=self.mt_args,
                              state_vect_dim=self.state_vect_dim)

    ## HISTORY METHODS ################################################################################################
    def update_history(self, name, val):
        """ update self.history with a dict s.t. val.keys()==self.history.keys()^{'Epoch','Best Loss Va'} """
        # name must be 'Tr' or 'Va', to update correctly training or validation history
        if name not in ['Tr', 'Va']: raise TypeError('param <name> must be \'Tr\' or \'Va\'')
        for key in val: self.history[key + ' ' + name].append(val[key])

    # -----------------------------------------------------------------------------------------------------------------
    def printHistory(self):
        """ print self.history """
        # CLEAR CONSOLE - only if in terminal, not in a pycharm-like software
        # os.system('cls' if os.name == 'nt' else 'clear')
        # pandas automatically detect terminal width: print dataframe, not dataframe.to_string()
        history = {i: self.history[i] for i in self.history if self.history[i]}
        p = pd.DataFrame(history)
        print(p, end='\n\n')

    ## LOOP METHODS ###################################################################################################
    ## EDGE-BASED: @tf.function may raise error in tape.gradients(loss,...) because of the tf.gather in apply_filters()
    ## In TF2.3+ this will be hopefully fixed. LOOK https://github.com/tensorflow/tensorflow/issues/36236
    #@tf.function
    def convergence(self, k, state, state_old, nodes, nodes_index, arcs_label, arcnode, training):
        # compute the incoming message for each node: shape == (len(source_nodes_index, Num state components)
        source_state = tf.gather(state, nodes_index[:, 0])
        # concatenate the gathered source node states with the corresponding arc labels
        arc_message = tf.concat([source_state, arcs_label], axis=1)
        if self.state_vect_dim:
            source_label = tf.gather(nodes, nodes_index[:, 0])
            arc_message = tf.concat([source_label, arc_message], axis=1)
        # multiply by ArcNode matrix to get the incoming average/total/normalized messages on each node
        message = tf.sparse.sparse_dense_matmul(arcnode, arc_message)
        # concatenate the destination node 'old' states to the incoming messages
        inp_state = tf.concat((nodes, state, message) if self.state_vect_dim else (state, message), axis=1)
        # compute new state and update step iteration counter
        state_new = self.net_state(inp_state, training=training)
        return k + 1, state_new, state, nodes, nodes_index, arcs_label, arcnode, training

    # -----------------------------------------------------------------------------------------------------------------
    #@tf.function
    def condition(self, k, state, state_old, *args) -> tf.bool:
        """ Boolean function condition for tf.while_loop correct processing graphs """
        # distance_vector is the Euclidean Distance: √ Σ(xi-yi)² between current state xi and past state yi
        outDistance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(state, state_old)), axis=1))
        # state_norm is the norm of state_old, defined by ||state_old|| = √ Σxi²
        state_norm = tf.sqrt(tf.reduce_sum(tf.square(state_old), axis=1))
        # boolean vector that stores the "convergence reached" flag for each node
        scaled_state_norm = tf.math.scalar_mul(self.state_threshold, state_norm)
        # check whether global convergence and/or the maximum number of iterations have been reached
        checkDistanceVec = tf.greater(outDistance, scaled_state_norm)
        # compute boolean
        c1 = tf.reduce_any(checkDistanceVec)
        c2 = tf.less(k, self.max_iteration)
        return tf.logical_and(c1, c2)

    # -----------------------------------------------------------------------------------------------------------------
    #@tf.function
    def apply_filters(self, state_converged, nodes, nodes_index, arcs_label, mask):
        """ takes only nodes states for those with output_mask==1 AND belonging to set (in case Dataset == 1 Graph) """
        if self.state_vect_dim: state_converged = tf.concat((nodes, state_converged), axis=1)
        return tf.boolean_mask(state_converged, mask)

    # -----------------------------------------------------------------------------------------------------------------
    def Loop(self, g, *, training=False):
        """ process a single graph, returning iteration, states and output """
        # retrieve quantities from graph f
        nodes = tf.constant(g.getNodes(), dtype=tf.float32)
        nodes_index = tf.constant(g.getArcs()[:, :2], dtype=tf.int32)
        arcs_label = tf.constant(g.getArcs()[:, 2:], dtype=tf.float32)
        arcnode = self.ArcNode2SparseTensor(g.getArcNode())
        mask = tf.logical_and(g.getSetMask(), g.getOutputMask())
        # initialize all the useful variables for convergence loop
        k = tf.constant(0, dtype=tf.float32)
        state = tf.constant(g.initState(self.state_vect_dim), dtype=tf.float32)
        state_old = tf.ones_like(state, dtype=tf.float32)
        training = tf.constant(training)
        # loop until convergence is reached
        k, state, state_old, *_ = tf.while_loop(self.condition, self.convergence, [k, state, state_old, nodes, nodes_index, arcs_label, arcnode, training])
        # out_st is the converged state for the filtered nodes, depending on g.set_mask
        out_st = self.apply_filters(state, nodes, nodes_index, arcs_label, mask)
        # compute the output of the gnn network
        out_gnn = self.net_output(out_st, training=training)
        return k, out_st, out_gnn

    ## CALL/PREDICT METHOD ############################################################################################
    def __call__(self, g):
        """ return ONLY the GNN output for graph g of type GraphObject """
        out = self.Loop(g, training=False)[-1]
        return self.output_activation(out)

    ## EVALUATE METHODs ###############################################################################################
    def evaluate_single_graph(self, g, class_weights):
        """ evaluate method for evaluating one graph single graph. Returns iteration, loss, target and output """
        # get targets
        targs = tf.constant(g.getTargets(), dtype=tf.float32)
        if g.problem_based != 'g': targs = targs[tf.logical_and(g.getSetMask(), g.getOutputMask())]
        # graph processing
        iter, outSt, out = self.Loop(g, training=False)
        # weighted loss if class_metrics != 1, else it does not modify loss values
        loss_weight = tf.reduce_sum(class_weights * targs, axis=1)
        loss = self.loss_function(targs, out, **self.loss_args)
        loss *= loss_weight
        return iter, loss, targs, self.output_activation(out)

    def evaluate(self, g, class_weights):
        """ return ALL the metrics in self.extra_metrics + Iter & Loss for a GraphObject or a list of GraphObjects
        :param g: element/list of GraphObject to be evaluated
        :param class_weights: (list) [w0, w1,...,wc] for classification task, specify the weight for weighted loss
        :return: metrics, float(loss) target_labels, prediction_labels, targets_raw and prediction_raw,
        """
        # chech if inputs are GraphObject OR list(s) of GraphObject(s)
        if not (type(g) == GraphObject or (type(g) == list and all(isinstance(x, GraphObject) for x in g))):
            raise TypeError('type of param <g> must be GraphObject or list of GraphObjects')
        if type(g) == GraphObject: g = [g]
        iters, losses, targets, outs = zip(*[self.evaluate_single_graph(i, class_weights) for i in g])
        # concatenate all the values from every graph and take clas labels or values
        loss = tf.concat(losses, axis=0)
        targets = tf.concat(targets, axis=0)
        y_score = tf.concat(outs, axis=0)
        y_true = tf.argmax(targets, axis=1) if self.addressed_problem == 'c' else targets
        y_pred = tf.argmax(y_score, axis=1) if self.addressed_problem == 'c' else y_score
        # evaluate metrics
        metr={k:float(self.extra_metrics[k](y_true,y_pred,**self.mt_args.get(k, dict()))) for k in self.extra_metrics}
        metr['It'] = int(tf.reduce_mean(iters))
        metr['Loss'] = float(tf.reduce_mean(loss))
        return metr, metr['Loss'], y_true, y_pred, targets, y_score

    ## TRAINING METHOD ################################################################################################
    def train(self, gTr, epochs: int = 10, gVa=None, validation_freq: int = 10, max_fails: int = 10, class_weights=1,
              *, mean: bool = False, verbose: int = 3):
        """ TRAIN PROCEDURE
        :param gTr: element/list of GraphObjects used for the learning procedure
        :param epochs: (int) the max number of epochs for the learning procedure
        :param gVa: element/list of GraphsObjects for early stopping
        :param validation_freq: (int) specifies how many epochs must be completed before evaluating gVa and gTr
        :param max_fails: (int) specifies the max number of failures before early sopping
        :param class_weights: (list) [w0, w1,...,wc] for classification task, specify the weight for weighted loss
        :param mean: (bool) if False the applied gradients are computed as the sum of every iteration, else as the mean
        :param verbose: (int) 0: silent mode; 1:print epochs/batches; 2: print history; 3: history + epochs/batches
        """
        def checktype(elem):
            """ check if type(elem) is correct. If so, return None or a list og GraphObjects """
            if elem is None: return None
            if type(elem) == GraphObject: return [elem]
            elif isinstance(elem, (list, tuple)) and all(isinstance(x, GraphObject) for x in elem): return list(elem)
            else: raise TypeError('Error - <gTr> and/or <gVa> are not GraphObject or LIST/TUPLE of GraphObjects')
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        def reset_validation(valid_loss):
            """ inner-method used to reset the validation check parameters and to save the 'best weights until now' """
            return valid_loss, 0, self.net_state.get_weights(), self.net_output.get_weights()
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        def training_step(gTr, mean):
            """ compute the gradients and apply them """
            with tf.GradientTape() as tape:
                iter, loss, *_ = self.evaluate_single_graph(gTr, class_weights)
            dwbS, dwbO = tape.gradient(loss, [self.net_state.trainable_variables, self.net_output.trainable_variables])
            # average net_state dw and db w.r.t. the number of iteration.
            if mean: dwbS = [i / iter for i in dwbS]
            # apply gradients
            zipped = zip(dwbS + dwbO, self.net_state.trainable_variables + self.net_output.trainable_variables)
            self.optimizer.apply_gradients(zipped)
        ### TRAINING FUNCTION -----------------------------------------------------------------------------------------
        if verbose not in range(4): raise ValueError('param <verbose> not in [0,1,2,3]')
        # Checking type for gTr and gVa + Initialization of Validation parameters
        gTr, gVa = checktype(gTr), checktype(gVa)
        # Writers: Training, Validation (scalars) + Net_state, Net_output (histogram for weights/biases)
        netS_writer = tf.summary.create_file_writer(self.path_writer + 'Net - State')
        netO_writer = tf.summary.create_file_writer(self.path_writer + 'Net - Output')
        trainining_writer = tf.summary.create_file_writer(self.path_writer + 'Training')
        if gVa:
            lossVa = self.history['Best Loss Va'][-1] if self.history['Best Loss Va'] else float(1e30)
            vbest_loss, vfails, ws, wo = reset_validation(lossVa)
            validation_writer = tf.summary.create_file_writer(self.path_writer + 'Validation')
        # pre-Training procedure: check if it's the first learning time to correctly update tensorboard
        # os.system('cls' if os.name == 'nt' else 'clear')
        initial_epoch = self.history['Epoch'][-1] + 1 if self.history['Epoch'] else 0
        epochs += initial_epoch
        for e in range(initial_epoch, epochs):
            # TRAINING STEP
            for i, elem in enumerate(gTr):
                training_step(elem, mean=mean)
                if verbose > 1: print(' > Epoch {:4d}/{} \t\t> Batch {:4d}/{}'.format(e,epochs, i+1,len(gTr)),end='\r')
            # TRAINING EVALUATION STEP
            if e % validation_freq == 0:
                metricsTr, *_ = self.evaluate(gTr, class_weights)
                # History Update
                self.history['Epoch'].append(e)
                self.update_history('Tr', metricsTr)
                # TensorBoard Update Tr: Losses, Interation@Convergence, Accuracies + histograms of weights
                self.write_vals(trainining_writer, metricsTr, e)
                self.write_net_weights(netS_writer, self.net_state.get_weights(), e, net_name='N1')
                self.write_net_weights(netO_writer, self.net_output.get_weights(), e, net_name='N2')
            # VALIDATION STEP
            if (e % validation_freq == 0) and gVa:
                metricsVa, lossVa, *_ = self.evaluate(gVa, class_weights)
                # Validation check
                if lossVa < vbest_loss: vbest_loss, vfails, ws, wo = reset_validation(lossVa)
                else: vfails += 1
                # History Update
                self.history['Best Loss Va'].append(vbest_loss)
                self.history['Fail'].append(vfails)
                self.update_history('Va', metricsVa)
                # TensorBoard Update Va: Losses, Interation@Convergence, Accuracies + histograms of weights
                self.write_vals(validation_writer, metricsVa, e)
                # Early Stoping - reached max_fails for validation set
                if vfails >= max_fails:
                    self.net_state.set_weights(ws)
                    self.net_output.set_weights(wo)
                    print('\r Validation Stop')
                    break
            # PRINT HISTORY
            if (e % validation_freq == 0) and verbose in [1, 3]: self.printHistory()
        else: print('\r End of Epochs Stop')
        # Tensorboard Update FINAL: write BEST WEIGHTS + BIASES
        self.write_net_weights(netS_writer, self.net_state.get_weights(), e, net_name='N1')
        self.write_net_weights(netO_writer, self.net_output.get_weights(), e, net_name='N2')

    ## TEST METHOD ####################################################################################################
    def test(self, gTe, *, acc_classes=True, rocdir='', micro_and_macro=False, prisofsdir=''):
        """ TEST PROCEDURE
        :param gTe: element/list of GraphObjects for testing procedure
        :param accuracy_class: (bool) if True print accuracy for classes
        :param rocdir: (str) path for saving ROC images file
        :param micro_and_macro: (bool) for computing micro and macro average quantities in roc curve
        :param prisofsdir: (str) path for saving Precision-Recall curve with ISO F-Score images file
        :return: metrics for gTe
        """
        if type(gTe) != GraphObject and not (type(gTe) == list and all(isinstance(x, GraphObject) for x in gTe)):
            raise TypeError('type of param <gTe> must be GraphObject or list of GraphObjects')
        if not all(isinstance(x, str) for x in [rocdir, prisofsdir]):
            raise TypeError('type of params <roc> and <prisofs> must be str')
        # Evaluate all the metrics in gnn.extra_metrics + Iter and Loss
        metricsTe, lossTe, y_true, y_pred, targets, y_score = self.evaluate(gTe, class_weights=1)
        # Accuracy per Class: shape = (1,number_classes)
        if acc_classes and self.addressed_problem == 'c':
            accuracy_classes = mt.accuracy_per_class(y_true, y_pred)
            metricsTe['Acc Classes'] = accuracy_classes
        # ROC e PR curves
        if rocdir: mt.ROC(targets, y_score, rocdir, micro_and_macro)
        if prisofsdir: mt.PRISOFS(targets, y_score, prisofsdir)
        return metricsTe

    ## K-FOLD CROSS VALIDATION METHOD #################################################################################
    def LKO(self, dataset, node_aggregation, number_of_batches=10, seed=None, normalize_method='gTr', verbose=3,
            acc_classes=False, epochs=500, Va=False, validation_freq=10, max_fails=10, class_weights=1, mean=False):
        """ LEAVE K OUT PROCEDURE
        :param dataset: (list) of GraphObject OR (list) of lists of GraphObject on which <gnn> has to be valuated
                            > NOTE: for graph-based problem, if type(dataset) == list of GraphObject,
                            s.t. len(dataset) == number of graphs in the dataset,
                            then i-th class will may be have different frequencies among batches
                            [so the i-th class may me more present in a batch and absent in another batch].
                            Otherwise, if type(dataset) == list of lists, s.t. len(dataset) == number of classes AND
                            len(dataset[i]) == number of graphs belonging to i-th class,
                            then i-th class will have the same frequency among all the batches
                            [so the i-th class will be as frequent in a single batch as in the entire dataset].
        :param node_aggregation: (str) for node aggregation method during dataset creation. See GraphObject for details
        :param number_of_batches: (int) define how many batches will be considered in LKO procedure
        :param seed: (int or None) for fixed-shuffle options
        :param normalize_method: (str) in ['gTr,'all'], see normalize_graphs for details
        :param verbose: (int) 0: silent mode; 1:print epochs/batches; 2: print history; 3: history + epochs/batches
        :param acc_classes: (bool) return or not the accuracy for each class in metrics
        :param epochs: (int) number of epochs for training <gnn>, the gnn will be trained for all the epochs
        :param Va: (bool) if True, Early Stopping is considered during learning procedure; None otherwise
        :param validation_freq: (int) specifies how many epochs must be completed before evaluating gVa and gTr
        :param max_fails: (int) specifies the max number of failures before early sopping
        :param class_weights: (list) [w0, w1,...,wc] for classification task, specify the weight for weighted loss
        :param mean: (bool) if False the applied gradients are computed as the sum of every iteration, else as the mean
        :return: a dict containing all the considered metrics in <gnn>.history
        """
        # classification vs regression LKO problem: see :param dataset: for details
        if all(isinstance(i, GraphObject) for i in dataset): dataset = [dataset]
        # Shuffling procedure: fix/not fix seed parameter, then shuffle classes and/or elements in each class/dataset
        if seed: np.random.seed(seed)
        np.random.shuffle(dataset)
        for i in dataset: np.random.shuffle(i)
        # Dataset creation, based on param <dataset>
        if Va: number_of_batches += 1
        dataset_batches = [utils.getbatches(elem, node_aggregation=node_aggregation,
                                            number_of_batches=number_of_batches, one_graph_per_batch=False)
                           for i, elem in enumerate(dataset)]
        flatten = lambda l: [item for sublist in l for item in sublist]
        flattened = [flatten([i[j] for i in dataset_batches]) for j in range(number_of_batches)]
        # shuffle again to mix classes inside batches, so that i-th class does not appears there at the same position
        for i in flattened: np.random.shuffle(i)
        # Final dataset for LKO procedure: merge graphs belonging to classes/dataset to obtain 1 GraphObject per batch
        dataset = [GraphObject.merge(i, node_aggregation=node_aggregation) for i in flattened]
        # results
        metrics = {i: list() for i in list(self.extra_metrics) + ['It', 'Loss']}
        if acc_classes: metrics['Acc Classes'] = list()
        # LKO PROCEDURE
        len_dataset = len(dataset) - (1 if Va else 0)
        for i in range(len_dataset):
            # split dataset in training/validation/test set
            gTr = dataset.copy()
            gTe = gTr.pop(i)
            gVa = gTr.pop(-1) if Va else None
            # normalization procedure
            utils.normalize_graphs(gTr, gVa, gTe, based_on=normalize_method)
            # gnn creation, learning and test
            print('GNN {0}/{1}'.format(i + 1, len_dataset))
            gnn_temp = self.copy(copy_weights=False, path_writer=self.path_writer + str(i))
            gnn_temp.train(gTr, epochs, gVa, validation_freq, max_fails, class_weights, mean=mean, verbose=verbose)
            M = gnn_temp.test(gTe, acc_classes=acc_classes)
            # evaluate metrics
            for m in M: metrics[m].append(M[m])
        return metrics

    ## STATIC METHODs #################################################################################################
    @staticmethod
    def ArcNode2SparseTensor(ArcNode):
        # ArcNode Tensor, then reordered to be correctly computable. NOTE: reorder() recommended by TF2.0+
        indices = [[ArcNode.row[i], ArcNode.col[i]] for i in range(ArcNode.shape[0])]
        arcnode = tf.SparseTensor(indices, values=ArcNode.data, dense_shape=ArcNode.shape)
        arcnode = tf.sparse.transpose(arcnode)
        arcnode = tf.sparse.reorder(arcnode)
        arcnode = tf.cast(arcnode, dtype=tf.float32)
        return arcnode

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def write_net_weights(writer, val_list, epoch, net_name):
        if net_name not in ['N1', 'N2']: raise ValueError('param net_name must be in [N1,N2]')
        weights, biases = val_list[0::2], val_list[1::2]
        length = len(weights)
        names = [net_name + ' L' + str(i) for i in range(length)]
        with writer.as_default():
            with tf.name_scope('Nets: Weights'):
                for i in range(length): tf.summary.histogram(names[i], weights[i], step=epoch)
            with tf.name_scope('Net: Biases'):
                for i in range(length): tf.summary.histogram(names[i], biases[i], step=epoch)

    # -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def write_vals(writer, metrics, epoch):
        if type(metrics) != dict: raise TypeError('type of param <metrics> must be dict')
        names = {'Acc': 'Accuracy', 'Bacc': 'Balanced Accuracy', 'Ck': 'Cohen\'s Kappa', 'Js': 'Jaccard Score',
                 'Fs': 'F1-Score', 'Prec': 'Precision Score', 'Rec': 'Recall Score', 'Tpr': 'TPR', 'Tnr': 'TNR',
                 'Fpr': 'FPR', 'Fnr': 'FNR', 'Loss': 'Loss', 'It': 'Iteration @ Convergence'}
        namescopes = {**{i: 'Accuracy & Loss' for i in ['Acc', 'Bacc', 'It', 'Loss']},
                      **{i: 'F-Score, Precision and Recall' for i in ['Fs', 'Prec', 'Rec']},
                      **{i: 'Positive and Negative Rates' for i in ['Tpr', 'Tnr', 'Fpr', 'Fnr']},
                      **{i: 'Scores' for i in ['Ck', 'Js']}}
        with writer.as_default():
            for i in metrics:
                with tf.name_scope(namescopes[i]):
                    tf.summary.scalar(names[i], metrics[i], step=epoch, description=names[i])


#######################################################################################################################
### CLASS GNN - GRAPH BASED ###########################################################################################
#######################################################################################################################
class GNNgraphBased(GNN):
    def Loop(self, g, *, training=False):
        iter, state_nodes, out_nodes = GNN.Loop(self, g, training=training)
        # obtain a single output for each graph, by averaging the output of all of its nodes
        nodegraph = tf.constant(g.getNodeGraph(), dtype=tf.float32)
        out_gnn = tf.matmul(nodegraph, out_nodes, transpose_a=True)
        return iter, state_nodes, out_gnn


#######################################################################################################################
### CLASS GNN - EDGE BASED ############################################################################################
#######################################################################################################################
class GNNedgeBased(GNN):
    #@tf.function
    def apply_filters(self, state_converged, nodes, nodes_index, arcs_label, mask):
        """ takes only arcs info of those with output_mask==1 AND belonging to set (in case Dataset == 1 Graph) """
        if self.state_vect_dim: state_converged = tf.concat((nodes, state_converged), axis=1)
        # gather source nodes state
        source_state = tf.gather(state_converged, nodes_index[:, 0])
        source_state = tf.cast(source_state, tf.float32)
        # gather destination nodes state
        destination_state = tf.gather(state_converged, nodes_index[:, 1])
        destination_state = tf.cast(destination_state, tf.float32)
        # concatenate source and destination states to arc labels
        arc_state = tf.concat([source_state, destination_state, arcs_label], axis=1)
        # takes only arcs states for those with output_mask==1 AND belonging to the set (in case Dataset == 1 Graph)
        return tf.boolean_mask(arc_state, mask)


#######################################################################################################################
### CLASS GNN - NODE BASED ## First MLP, then sum-up for states #######################################################
#######################################################################################################################
## GNN v1 by A.Rossi and M.Tiezzi
class GNN2(GNN):
    @tf.function
    def convergence(self, k, state, state_old, nodes, nodes_index, arcs_label, arcnode, training):
        # gather source nodes label
        source_label = tf.gather(nodes, nodes_index[:, 0])
        source_label = tf.cast(source_label, tf.float32)
        # gather destination nodes label
        destination_label = tf.gather(nodes, nodes_index[:, 1])
        destination_label = tf.cast(destination_label, tf.float32)
        # gather destination nodes state
        destination_state = tf.gather(state, nodes_index[:, 1])
        destination_state = tf.cast(destination_state, tf.float32)
        # concatenate: source_node_label, destination_node_label, edge_attributes and destinatino_node_state
        arc_message = tf.concat([source_label, destination_label, arcs_label, destination_state], axis=1)
        arc_message = tf.cast(arc_message, tf.float32)  # re-cast to be computable, CAN BE OMITTED
        # compute the incoming message for each node with MLP: shape == (len(source_nodes_index, Num state components)
        messages = self.net_state(arc_message, training=training)
        # compute new state and update step iteration counter
        # multiply by ArcNode matrix to get the incoming average/total/normalized messages on each node
        state_new = tf.sparse.sparse_dense_matmul(arcnode, messages)
        state_new = tf.cast(state_new, tf.float32)
        k = k + 1
        return k, state_new, state, nodes, nodes_index, arcs_label, arcnode, training
