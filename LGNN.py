from __future__ import annotations

from typing import Union, Optional

import tensorflow as tf
from numpy import array

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
        ### CHECK LENGTHs
        # if type(net_state_params) == dict: net_state_params = [net_state_params.copy() for _ in range(layers)]
        # if type(net_output_params) == dict: net_output_params = [net_output_params.copy() for _ in range(layers)]
        # BaseGNN constructor
        super().__init__(optimizer, loss_function, loss_arguments, addressed_problem, extra_metrics, extra_metrics_arguments, path_writer, namespace)
        ### LGNNs parameter
        self.get_state = get_state
        self.get_output = get_output
        self.gnns = gnns
        self.layers = len(gnns)
        self.namespace = ['{} - GNN{}'.format(namespace, i) for i in range(self.layers)]
        # Change namespace for GNNs inside LGNN
        for gnn, name in zip(self.gnns, self.namespace):
            gnn.namespace = [name]
            gnn.path_writer = self.path_writer + name + '/'

    # -----------------------------------------------------------------------------------------------------------------
    def copy(self, *, path_writer: str = '', namespace: str = '', copy_weights: bool = True) -> 'LGNN':
        return self.__class__(gnns=[i.copy(copy_weights=copy_weights) for i in self.gnns],
                              get_state=self.get_state,
                              get_output=self.get_output,
                              optimizer=self.optimizer.__class__(**self.optimizer.get_config()),
                              loss_function=self.loss_function,
                              loss_arguments=self.loss_args,
                              addressed_problem=self.addressed_problem,
                              extra_metrics=self.extra_metrics,
                              extra_metrics_arguments=self.mt_args,
                              path_writer=path_writer if path_writer else self.path_writer + '_copied/',
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
        return self.Loop(g, training=False)[-1]


    ## EVALUATE METHODs ###############################################################################################
    def update_labels(self, g: GraphObject, state: Union[tf.Tensor, array], output: Union[tf.Tensor, array]):
        if self.get_state and self.get_output and g.problem_based != 'a': nodeplus, arcplus = tf.concat([state, output], axis=1), None
        elif self.get_state and self.get_output and g.problem_based == 'a': nodeplus, arcplus = state, output
        elif self.get_state: nodeplus, arcplus = state, None
        elif self.get_output and g.problem_based != 'a': nodeplus, arcplus = output, None
        else: nodeplus, arcplus = None, output
        return nodeplus, arcplus


    ## LOOP METHODS ###################################################################################################
    def Loop(self, g: GraphObject, *, nodeplus=None, arcplus=None, training: bool = False) -> tuple[list[int], tf.Tensor, tf.Tensor]:
        # graph processing
        K, nodeplus, arcplus = list(), None, None
        for idx, gnn in enumerate(self.gnns[:-1]):
            if g.problem_based != 'g': k, state, out = gnn.Loop(g, nodeplus=nodeplus, arcplus=arcplus, training=training)
            else: k, state, out = super(GNNgraphBased, gnn).Loop(g, nodeplus=nodeplus, arcplus=arcplus, training=training)
            nodeplus, arcplus = self.update_labels(g, state, out)
            K.append(k)
        k, state, out = self.gnns[-1].Loop(g, nodeplus=nodeplus, arcplus=arcplus, training=training)
        return K + [k], state, out


    ## TRAINING METHOD ################################################################################################
    def train(self, gTr: Union[GraphObject, list[GraphObject]], epochs: int = 10, gVa: Union[GraphObject, list[GraphObject], None] = None,
              update_freq: int = 10, max_fails: int = 10, class_weights: Union[int, list[float]] = 1,
              *, mean: bool = True, serial_training=False, verbose: int = 3) -> None:
        """ TRAIN PROCEDURE
        :param gTr: element/list of GraphObjects used for the learning procedure
        :param epochs: (int) the max number of epochs for the learning procedure
        :param gVa: element/list of GraphsObjects for early stopping
        :param update_freq: (int) specifies how many epochs must be completed before evaluating gVa and gTr
        :param max_fails: (int) specifies the max number of failures before early sopping
        :param class_weights: (list) [w0, w1,...,wc] for classification task, specify the weight for weighted loss
        :param mean: (bool) if False the applied gradients are computed as the sum of every iteration, else as the mean
        :param one_gnn_at_a_time: (bool). True: each GNN is trained separately. False: GNNs are trained all together (loss from LGNN's output)
        :param verbose: (int) 0: silent mode; 1: print history; 2:print epochs/batches, 3: history + epochs/batches
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
            from numpy import concatenate
            g, state, output = g.copy(), state.numpy(), output.numpy()
            if self.get_state: g.nodes = concatenate([g.nodes, state], axis=1)
            if self.get_output and g.problem_based != 'a': g.nodes = concatenate([g.nodes, output], axis=1)
            if self.get_output and g.problem_based == 'a': g.arcs = concatenate([g.arcs, output], axis=1)
            return g

        ### TRAINING FUNCTION -----------------------------------------------------------------------------------------
        if serial_training:
            gTr, gVa = checktype(gTr), checktype(gVa)
            gTr1, gVa1 = [i.copy() for i in gTr], [i.copy() for i in gVa] if gVa else None
            for idx, gnn in enumerate(self.gnns):
                if verbose in [1,3]: print('\n\n------------------- GNN{} -------------------\n'.format(idx))
                # train the idx-th gnn
                gnn.train(gTr1, epochs, gVa1, update_freq, max_fails, class_weights, mean=mean, verbose=verbose)
                # extrapolate state and output to update labels
                _, sTr, oTr = zip(*[gnn.Loop(i) if i.problem_based != 'g' else super(GNNgraphBased, gnn).Loop(i) for i in gTr1])
                gTr1 = [update_graph(i, s, o) for i, s, o in zip(gTr, sTr, oTr)]
                if gVa:
                    _, sVa, oVa = zip(*[gnn.Loop(i) if i.problem_based != 'g' else super(GNNgraphBased, gnn).Loop(i) for i in gVa1])
                    gVa1 = [update_graph(i, s, o) for i, s, o in zip(gVa, sVa, oVa)]
        else: super().train(gTr, epochs, gVa, update_freq, max_fails, class_weights, mean=mean, verbose=verbose)
