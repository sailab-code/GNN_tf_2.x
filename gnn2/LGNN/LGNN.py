from __future__ import annotations

from typing import Union, Optional

import tensorflow as tf
from numpy import array, zeros

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

        # BaseGNN constructor
        super().__init__(optimizer, loss_function, loss_arguments, addressed_problem, extra_metrics, extra_metrics_arguments, path_writer, namespace)

        ### LGNNs parameter
        self.get_state = get_state
        self.get_output = get_output
        self.gnns = gnns
        self.layers = len(gnns)
        self.namespace = ['{} - GNN{}'.format(namespace, i) for i in range(self.layers)]

        # Change namespace for self.gnns
        for gnn, name in zip(self.gnns, self.namespace):
            gnn.namespace = [name]
            gnn.path_writer = self.path_writer + name + '/'

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self, *, path_writer: str = '', namespace: str = '', copy_weights: bool = True) -> 'LGNN':
        return self.__class__(gnns=[i.copy(copy_weights=copy_weights) for i in self.gnns], get_state=self.get_state, get_output=self.get_output,
                              optimizer=self.optimizer.__class__(**self.optimizer.get_config()), loss_function=self.loss_function,
                              loss_arguments=self.loss_args, addressed_problem=self.addressed_problem, extra_metrics=self.extra_metrics,
                              extra_metrics_arguments=self.mt_args, path_writer=path_writer if path_writer else self.path_writer + '_copied/',
                              namespace=namespace if namespace else 'LGNN')


    ## GETTERS AND SETTERS METHODs ####################################################################################
    def trainable_variables(self) -> tuple[list[list[tf.Tensor]], list[list[tf.Tensor]]]:
        return [i.net_state.trainable_variables for i in self.gnns], [i.net_output.trainable_variables for i in self.gnns]

    # -----------------------------------------------------------------------------------------------------------------
    def get_weights(self) -> tuple[list[list[array]], list[list[array]]]:
        return [i.net_state.get_weights() for i in self.gnns], [i.net_output.get_weights() for i in self.gnns]

    # -----------------------------------------------------------------------------------------------------------------
    def set_weights(self, weights_state: list[list[array]], weights_output: list[list[array]]) -> None:
        assert len(weights_state) == len(weights_output) == self.layers
        for gnn, wst, wout in zip(self.gnns, weights_state, weights_output):
            gnn.net_state.set_weights(wst)
            gnn.net_output.set_weights(wout)


    ## CALL/PREDICT METHOD ############################################################################################
    def __call__(self, g: GraphObject) -> tf.Tensor:
        """ return ONLY the LGNN output for graph g of type GraphObject """
        out = tf.reduce_sum(self.Loop(g, training=False)[-1], axis=0)
        return self.gnns[-1].output_activation(out)

    # -----------------------------------------------------------------------------------------------------------------
    def predict(self, g: GraphObject, idx:Union[int, list[int], range]=-1) -> Union[tf.Tensor, list[tf.Tensor]]:
        if type(idx)==int: assert idx in range(-self.layers, self.layers)
        elif isinstance(idx, (list, range)): assert all(i in range(-self.layers, self.layers) for i in idx) and list(idx)==sorted(idx)
        else: raise ValueError('param <idx> must be int or list of int in range(-self.layers, self.layers)')

        # get only outputs, without iteration and states
        out = [gnn.output_activation(o) for gnn,o in zip(self.gnns, self.Loop(g, training=False)[-1])]
        return out[idx] if type(idx)==int else [out[i] for i in idx]


    ## EVALUATE METHODS ###############################################################################################
    def evaluate_single_graph(self, g: GraphObject, class_weights: Union[int, float, list[float]], training: bool) -> tuple:
        it, loss_weight, targs, out = super().evaluate_single_graph(g, class_weights, training)
        loss = tf.reduce_sum([self.loss_function(targs, i, **self.loss_args) * loss_weight for i in out], axis=0)
        # Add a residual connection to the lgnn output, s.t. outputs are merged and then are processed by last output activation
        out = tf.reduce_sum(out, axis=0)
        return it, loss, targs, self.gnns[-1].output_activation(out)


    ## LOOP METHODS ###################################################################################################
    def update_labels(self, g: GraphObject, state: Union[tf.Tensor, array], output: Union[tf.Tensor, array]) \
            -> tuple[Optional[tf.Tensor], Optional[tf.Tensor]]:
        arcplus, nodeplus = None, None
        # check state
        if self.get_state: nodeplus = state

        # check output
        if self.get_output:
            # process output to make it shape compatible
            mask = tf.logical_and(g.getSetMask(), g.getOutputMask())
            row = g.nodes.shape[0] if g.problem_based != 'a' else g.arcs.shape[0]
            out = zeros((row, g.targets.shape[1]))
            out[mask] = output
            if g.problem_based != 'a' and nodeplus is None: nodeplus = out
            elif g.problem_based != 'a': nodeplus = tf.concat([nodeplus, out], axis=1)
            else: arcplus = out

        return nodeplus, arcplus

    # -----------------------------------------------------------------------------------------------------------------
    def Loop(self, g: GraphObject, *, nodeplus=None, arcplus=None, training: bool = False) -> tuple[list[int], tf.Tensor, tf.Tensor]:
        # graph processing
        K, outs, nodeplus, arcplus = list(), list(), None, None
        for idx, gnn in enumerate(self.gnns[:-1]):

            if type(gnn) == GNNgraphBased:
                k, state, out = super(GNNgraphBased, gnn).Loop(g, nodeplus=nodeplus, arcplus=arcplus, training=training)
                nodegraph = tf.constant(g.getNodeGraph(), dtype=tf.float32)
                outs.append(tf.matmul(nodegraph, out, transpose_a=True))

            else:
                k, state, out = gnn.Loop(g, nodeplus=nodeplus, arcplus=arcplus, training=training)
                outs.append(out)

            K.append(k)
            nodeplus, arcplus = self.update_labels(g, state, gnn.output_activation(out))

        k, state, out = self.gnns[-1].Loop(g, nodeplus=nodeplus, arcplus=arcplus, training=training)
        return K + [k], state, outs + [out]


    ## TRAINING METHOD ################################################################################################
    def train(self, gTr: Union[GraphObject, list[GraphObject]], epochs: int, gVa: Union[GraphObject, list[GraphObject], None] = None,
              update_freq: int = 10, max_fails: int = 10, class_weights: Union[int, list[float]] = 1,
              *, mean: bool = True, serial_training=False, verbose: int = 3) -> None:
        """ TRAIN PROCEDURE
        :param gTr: GraphObject or list of GraphObjects used for the learning procedure
        :param epochs: (int) the max number of epochs for the learning procedure
        :param gVa: element/list of GraphsObjects for early stopping. Default None, no early stopping performed
        :param update_freq: (int) how many epochs must be completed before evaluating gVa and gTr and/or print learning progress. Default 10.
        :param max_fails: (int) specifies the max number of failures before early sopping. Default 10.
        :param class_weights: (list) [w0, w1,...,wc] in classification task when targets are 1-hot, specify the weight for weighted loss. Default 1.
        :param mean: (bool) if False the applied gradients are computed as the sum of every iteration, otherwise as the mean. Default True.
        :param serial_training: (bool) True: GNNs trained separately, otherwise trained all together (loss from LGNN's output). Default False.
        :param verbose: (int) 0: silent mode; 1: print history; 2: print epochs/batches, 3: history + epochs/batches. Default 3.
        :return: None
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        def checktype(elem: Optional[Union[GraphObject, list[GraphObject]]]) -> list[GraphObject]:
            """ check if type(elem) is correct. If so, return None or a list og GraphObjects """
            if elem is None: pass
            elif type(elem) == GraphObject: elem = [elem]
            elif isinstance(elem, (list, tuple)) and all(isinstance(x, GraphObject) for x in elem): elem = list(elem)
            else: raise TypeError('Error - <gTr> and/or <gVa> are not GraphObject or LIST/TUPLE of GraphObjects')
            return elem

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        def update_graph(g: GraphObject, state: Union[tf.Tensor, array], output: Union[tf.Tensor, array]) -> GraphObject:
            g = g.copy()
            nodeplus, arcplus = self.update_labels(g, state, output)
            if nodeplus is not None: g.nodes = concatenate([g.nodes, nodeplus.numpy()], axis=1)
            if arcplus is not None: g.arcs = concatenate([g.arcs, arcplus.numpy()], axis=1)
            return g

        ### TRAINING FUNCTION -----------------------------------------------------------------------------------------
        if serial_training:
            from numpy import concatenate
            gTr, gVa = checktype(gTr), checktype(gVa)
            gTr1, gVa1 = [i.copy() for i in gTr], [i.copy() for i in gVa] if gVa else None

            for idx, gnn in enumerate(self.gnns):
                if verbose in [1, 3]: print('\n\n------------------- GNN{} -------------------\n'.format(idx))

                # train the idx-th gnn
                gnn.train(gTr1, epochs, gVa1, update_freq, max_fails, class_weights, mean=mean, verbose=verbose)

                # extrapolate state and output to update labels
                _, sTr, oTr = zip(*[gnn.Loop(i) if i.problem_based != 'g' else super(GNNgraphBased, gnn).Loop(i) for i in gTr1])
                gTr1 = [update_graph(i, s, gnn.output_activation(o)) for i, s, o in zip(gTr, sTr, oTr)]
                if gVa:
                    _, sVa, oVa = zip(*[gnn.Loop(i) if i.problem_based != 'g' else super(GNNgraphBased, gnn).Loop(i) for i in gVa1])
                    gVa1 = [update_graph(i, s, gnn.output_activation(o)) for i, s, o in zip(gVa, sVa, oVa)]

        else:
            super().train(gTr, epochs, gVa, update_freq, max_fails, class_weights, mean=mean, verbose=verbose)