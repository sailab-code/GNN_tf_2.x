from __future__ import annotations

from typing import Union, Optional, Any

import tensorflow as tf
from numpy import array, zeros, concatenate, logical_and

from GNN.GNN import GNNnodeBased, GNNgraphBased, GNNedgeBased, GNN2
from GNN.GNN_BaseClass import BaseGNN
from GNN.graph_class import GraphObject


class LGNN(BaseGNN):
    ## CONSTRUCTORS METHODS ###########################################################################################
    def __init__(self,
                 gnns: list[GNNnodeBased, GNNedgeBased, GNNgraphBased, GNN2],
                 get_state: bool,
                 get_output: bool,
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss_function: tf.keras.losses.Loss,
                 loss_arguments: Optional[dict],
                 addressed_problem: str,
                 extra_metrics: Optional[dict] = None,
                 extra_metrics_arguments: Optional[dict[str, dict]] = None,
                 path_writer: str = 'writer/',
                 namespace: str = 'LGNN') -> None:
        """ CONSTRUCTOR

        :param gnns: (list) of instances of type GNN representing LGNN layers, initialized externally
        :param get_state: (bool) if True node_state are propagated through LGNN layers
        :param get_output: (bool) if True gnn_outputs are propagated through LGNN layers
        :param optimizer: (tf.keras.optimizers) for gradient application, initialized externally
        :param loss_function: (tf.keras.losses) or (tf.function) for the loss computation
        :param loss_arguments: (dict) with some {'argument':values} one could pass to loss when computed
        :param addressed_problem: (str) in ['r','c'], 'r':regression, 'c':classification for the addressed problem
        :param extra_metrics: None or dict {'name':function} for metrics to be watched during training/validaion/test
        :param extra_metrics_arguments: None or dict {'name':{'argument':value}} for arguments to be passed to extra_metrics
        :param path_writer: (str) path for saving TensorBoard objects
        :param namespace: (str) namespace for tensorboard visualization
        """
        # BaseGNN constructor
        super().__init__(optimizer, loss_function, loss_arguments, addressed_problem, extra_metrics, extra_metrics_arguments, path_writer,
                         namespace)

        ### LGNNs parameter
        self.get_state = get_state
        self.get_output = get_output
        self.gnns = gnns
        self.layers = len(gnns)
        self.namespace = [f'{namespace} - GNN{i}' for i in range(self.layers)]
        self.training_mode = None

        # Change namespace for self.gnns
        for gnn, name in zip(self.gnns, self.namespace):
            gnn.namespace = [name]
            gnn.path_writer = f'{self.path_writer}{name}/'

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self, *, path_writer: str = '', namespace: str = '', copy_weights: bool = True) -> 'LGNN':
        """ COPY METHOD

        :param path_writer: None or (str), to save copied lgnn writer. Default is in the same folder + '_copied'
        :param copy_weights: (bool) True: state and output weights are copied; False: state and output weights are re-initialized
        :return: a Deep Copy of the lgnn instance.
        """
        return self.__class__(gnns=[i.copy(copy_weights=copy_weights) for i in self.gnns], get_state=self.get_state,
                              get_output=self.get_output,
                              optimizer=self.optimizer.__class__(**self.optimizer.get_config()), loss_function=self.loss_function,
                              loss_arguments=self.loss_args, addressed_problem=self.addressed_problem, extra_metrics=self.extra_metrics,
                              extra_metrics_arguments=self.mt_args,
                              path_writer=path_writer if path_writer else self.path_writer + '_copied/',
                              namespace=namespace if namespace else 'LGNN')

    ## SAVE AND LOAD METHODs ##########################################################################################
    def save(self, path: str):
        """ save model to folder <path>"""
        from json import dump

        # check paths
        if path[-1] != '/': path += '/'

        # save GNNs
        for i, gnn in enumerate(self.gnns): gnn.save(f'{path}GNN{i}/')

        # save configuration file in json format
        config = {'get_state': self.get_state, 'get_output': self.get_output,
                  'loss_function': tf.keras.losses.serialize(self.loss_function), 'loss_arguments': self.loss_args,
                  'optimizer': tf.keras.optimizers.serialize(self.optimizer),
                  'extra_metrics': list(self.extra_metrics), 'extra_metrics_arguments': self.mt_args,
                  'addressed_problem': self.addressed_problem}

        with open(f'{path}config.json', 'w') as json_file:
            dump(config, json_file)

    # -----------------------------------------------------------------------------------------------------------------
    @classmethod
    def load(self, path: str, path_writer: str, namespace: str = 'LGNN'):
        """ load model from folder

        :param path: (str) folder path containing all useful files to load the model
        :param path_writer: (str) path for writer folder. !!! Constructor method makes delete a non-empty folder and makes a new empty one
        :param namespace: (str) namespace for tensorboard visualization of the model in training procedure
        :return: the model
        """
        from json import loads
        from os import listdir
        from os.path import isdir
        from GNN.GNN_metrics import Metrics

        # check paths
        if path[-1] != '/': path += '/'
        if path_writer[-1] != '/': path_writer += '/'

        # load GNNs
        gnns = [GNNnodeBased.load(f'{path}{i}', path_writer=f'{path_writer}{namespace} - GNN{i}/', namespace='GNN')
                for i in listdir(path) if isdir(f'{path}{i}')]

        # load configuration file
        with open(f'{path}config.json', 'r') as read_file:
            config = loads(read_file.read())

        # get optimizer, loss function
        optz = tf.keras.optimizers.deserialize(config.pop('optimizer'))
        loss = tf.keras.losses.deserialize(config.pop('loss_function'))

        # get extra metrics
        extra_metrics = {i: Metrics[i] for i in config.pop('extra_metrics')}
        return self(gnns=gnns, optimizer=optz, loss_function=loss, extra_metrics=extra_metrics,
                    path_writer=path_writer, namespace=namespace, **config)

    ## GETTERS AND SETTERS METHODs ####################################################################################
    def trainable_variables(self) -> tuple[list[list[tf.Tensor]], list[list[tf.Tensor]]]:
        """ get tensor weights for net_state and net_output for each gnn layer """
        return [i.net_state.trainable_variables for i in self.gnns], [i.net_output.trainable_variables for i in self.gnns]

    # -----------------------------------------------------------------------------------------------------------------
    def get_weights(self) -> tuple[list[list[array]], list[list[array]]]:
        """ get array weights for net_state and net_output for each gnn layer """
        return [i.net_state.get_weights() for i in self.gnns], [i.net_output.get_weights() for i in self.gnns]

    # -----------------------------------------------------------------------------------------------------------------
    def set_weights(self, weights_state: list[list[array]], weights_output: list[list[array]]) -> None:
        """ set weights for net_state and net_output for each gnn layer """
        assert len(weights_state) == len(weights_output) == self.layers
        for gnn, wst, wout in zip(self.gnns, weights_state, weights_output):
            gnn.net_state.set_weights(wst)
            gnn.net_output.set_weights(wout)

    ## CALL/PREDICT METHOD ############################################################################################
    def __call__(self, g: GraphObject) -> tf.Tensor:
        """ return ONLY the LGNN output for graph g of type GraphObject """
        out = self.Loop(g, training=False)[-1]
        return out[-1]

    # -----------------------------------------------------------------------------------------------------------------
    def predict(self, g: GraphObject, idx: Union[int, list[int], range] = -1) -> Union[tf.Tensor, list[tf.Tensor]]:
        """ get LGNN output(s)

        :param g: (GraphObject) single element GraphObject
        :param idx: set the layer whose output is wanted to be returned.
                    More than one layer output can be returned, setting idx as ordered list/range
        :return: a list of output(s) of the model processing graph g
        """
        if type(idx) == int:
            assert idx in range(-self.layers, self.layers)
        elif isinstance(idx, (list, range)):
            assert all(i in range(-self.layers, self.layers) for i in idx) and list(idx) == sorted(idx)
        else:
            raise ValueError('param <idx> must be int or list of int in range(-self.layers, self.layers)')

        # get only outputs, without iteration and states
        out = self.Loop(g, training=False)[-1]
        return out[idx] if type(idx) == int else [out[i] for i in idx]

    ## EVALUATE METHODS ###############################################################################################
    def evaluate_single_graph(self, g: GraphObject, class_weights: Union[int, float, list[int, float], array[int, float]], training: bool) -> tuple:
        """
        :param g: (GraphObject) single element GraphObject
        :param class_weights: in classification task, it can be an int, flot, list of ints or floats, compatible with 1hot target matrix (under review)
        :param training: (bool) set internal models behavior, s.t. they work in training or testing mode
        :return: (tuple) convergence iteration, loss value, target and output of the model
        """
        # get targets
        targs = tf.constant(g.getTargets(), dtype=tf.float32)
        if g.problem_based != 'g': targs = tf.boolean_mask(targs, g.set_mask[g.output_mask])

        # graph processing
        it, _, out = self.Loop(g, training=training)

        # if class_metrics != 1, else it does not modify loss values
        loss_weight = tf.reduce_sum(class_weights * targs, axis=1)
        if training and self.training_mode == 'residual':
            loss = self.loss_function(targs, tf.reduce_sum(out, axis=0)) * loss_weight
        else:
            loss = tf.reduce_sum([self.loss_function(targs, o, **self.loss_args) * loss_weight for o in out], axis=0)

        return it, loss, targs, out[-1]

    ## LOOP METHODS ###################################################################################################
    def update_graph(self, g: GraphObject, state: Union[tf.Tensor, array], output: Union[tf.Tensor, array]) -> GraphObject:
        """
        :param g: (GraphObject) single element GraphObject the update process is based on
        :param state: (tensor) output of the net_state model of a single gnn layer
        :param output: (tensor) output of the net_output model of a single gnn layer
        :return: (GraphObject) a new GraphObject where state and/or output are integrated in nodes/arcs label
        """
        # copy graph to preserve original graph data
        g = g.copy()

        # define tensors with shape[1]==0 so that it can be concatenate with tf.concat()
        arcplus, nodeplus = tf.zeros((g.arcs.shape[0], 0), dtype=tf.float32), tf.zeros((g.nodes.shape[0], 0), dtype=tf.float32)

        # check state
        if self.get_state: nodeplus = tf.concat([nodeplus, state], axis=1)

        # check output
        if self.get_output:
            # process output to make it shape compatible.
            # Note that what is concatenated is not nodeplus/arcplus, but out, as it has the same length of nodes/arcs
            mask = logical_and(g.set_mask, g.output_mask)
            row = g.nodes.shape[0] if g.problem_based != 'a' else g.arcs.shape[0]

            # define a zero matrix
            out = zeros((row, g.DIM_TARGET), dtype='float32')
            out[mask] = output

            if g.problem_based != 'a': nodeplus = tf.concat([nodeplus, out], axis=1)
            else: arcplus = tf.concat([arcplus, out], axis=1)

        # nodeplus, arcplus = self.update_labels(g, state, output) ## cancellare riga
        g.nodes = concatenate([g.nodes, nodeplus.numpy()], axis=1)
        g.arcs = concatenate([g.arcs, arcplus.numpy()], axis=1)
        return g

    '''def update_labels(self, g: GraphObject, state: Union[tf.Tensor, array], output: Union[tf.Tensor, array]) \
            -> tuple[Optional[tf.Tensor], Optional[tf.Tensor]]:

        # define tensors with shape[1]==0 so that it can be concatenate with tf.concat()
        arcplus, nodeplus = tf.zeros((g.arcs.shape[0], 0), dtype=tf.float32), tf.zeros((g.nodes.shape[0], 0), dtype=tf.float32)

        # check state
        if self.get_state: nodeplus = tf.concat([nodeplus, state], axis=1)

        # check output
        if self.get_output:
            # process output to make it shape compatible.
            # Note that what is concatenated is not nodeplus/arcplus, but out, as it has the same length of nodes/arcs
            mask = logical_and(g.set_mask, g.output_mask)
            row = g.nodes.shape[0] if g.problem_based != 'a' else g.arcs.shape[0]

            # define a zero matrix
            out = zeros((row, g.DIM_TARGET), dtype='float32')
            out[mask] = output

            if g.problem_based != 'a':
                nodeplus = tf.concat([nodeplus, out], axis=1)
            else:
                arcplus = tf.concat([arcplus, out], axis=1)

        return nodeplus, arcplus'''

    # -----------------------------------------------------------------------------------------------------------------
    def Loop(self, g: GraphObject, *, training: bool = False) -> tuple[list[Union[int, Any]], tf.Tensor, list[Union[tf.Tensor, Any]]]:
        """ process a single graph, returning iterations, states and outputs """
        K, outs = list(), list()
        gtmp = g.copy()

        # forward pass
        for idx, gnn in enumerate(self.gnns[:-1]):

            if type(gnn) == GNNgraphBased:
                k, state, out = super(GNNgraphBased, gnn).Loop(gtmp, training=training)
                nodegraph = tf.constant(gtmp.getNodeGraph(), dtype=tf.float32)
                outs.append(tf.matmul(nodegraph, out, transpose_a=True))

            else:
                k, state, out = gnn.Loop(gtmp, training=training)
                outs.append(out)

            K.append(k)
            gtmp = self.update_graph(g, state, out)

        k, state, out = self.gnns[-1].Loop(gtmp, training=training)
        return K + [k], state, outs + [out]


    ## TRAINING METHOD ################################################################################################
    def train(self, gTr: Union[GraphObject, list[GraphObject]], epochs: int, gVa: Union[GraphObject, list[GraphObject], None] = None,
              update_freq: int = 10, max_fails: int = 10, class_weights: Union[int, list[float]] = 1,
              *, mean: bool = True, training_mode: str = 'residual', verbose: int = 3) -> None:
        """ LEARNING PROCEDURE

        :param gTr: GraphObject or list of GraphObjects used for the learning procedure
        :param epochs: (int) the max number of epochs for the learning procedure
        :param gVa: element/list of GraphsObjects for early stopping. Default None, no early stopping performed
        :param update_freq: (int) how many epochs must be completed before evaluating gVa and gTr and/or print learning progress. Default 10.
        :param max_fails: (int) specifies the max number of failures before early sopping. Default 10.
        :param class_weights: (list) [w0, w1,...,wc] in classification task when targets are 1-hot, specify the weight for weighted loss. Default 1.
        :param mean: (bool) if False the applied gradients are computed as the sum of every iteration, otherwise as the mean. Default True.
        :param training_mode: (str) in ['serial','parallel','residual']
            > 'serial' - GNNs are trained separately, from layer 0 to layer N
            > 'parallel' - GNNs are trained all together, from loss = sum( Loss_Function( t, Oi) ) where Oi is the output of GNNi
            > 'residual' - GNNs are trained all together, from loss = Loss_Function(i, phi(Oi))
        :param verbose: (int) 0: silent mode; 1: print history; 2: print epochs/batches, 3: history + epochs/batches. Default 3.
        :return: None
        """

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

        ### TRAINING FUNCTION -----------------------------------------------------------------------------------------
        assert training_mode in ['parallel', 'serial', 'residual']
        self.training_mode = training_mode

        if training_mode == 'serial':
            gTr, gVa = checktype(gTr), checktype(gVa)
            gTr1, gVa1 = [i.copy() for i in gTr], [i.copy() for i in gVa] if gVa else None

            for idx, gnn in enumerate(self.gnns):
                if verbose in [1, 3]: print('\n\n------------------- GNN{} -------------------\n'.format(idx))

                # train the idx-th gnn
                gnn.train(gTr1, epochs, gVa1, update_freq, max_fails, class_weights, mean=mean, verbose=verbose)

                # extrapolate state and output to update labels
                _, sTr, oTr = zip(*[gnn.Loop(i) if i.problem_based != 'g' else super(GNNgraphBased, gnn).Loop(i) for i in gTr1])
                gTr1 = [self.update_graph(i, s, o) for i, s, o in zip(gTr, sTr, oTr)]
                if gVa:
                    _, sVa, oVa = zip(*[gnn.Loop(i) if i.problem_based != 'g' else super(GNNgraphBased, gnn).Loop(i) for i in gVa1])
                    gVa1 = [self.update_graph(i, s, o) for i, s, o in zip(gVa, sVa, oVa)]

        else:
            super().train(gTr, epochs, gVa, update_freq, max_fails, class_weights, mean=mean, verbose=verbose)
