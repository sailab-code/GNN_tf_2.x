from __future__ import annotations

from typing import Optional, Union

import tensorflow as tf
from numpy import random

from GNN import GNN_metrics as mt, GNN_utils as utils
from GNN.GNN import GNNnodeBased, GNNedgeBased, GNNgraphBased
from GNN.LGNN.LGNN import LGNN
from GNN.MLP import MLP, get_inout_dims
from GNN.graph_class import GraphObject

#######################################################################################################################
# SCRIPT OPTIONS - modify the parameters to adapt the execution to the problem under consideration ####################
#######################################################################################################################

# MUTAG option - if True, gnn/lgnn is trained on a real-world dataset MUTAG
# problem is set automatically to graph classification -> addressed_problem='c', problem_based='g'
use_MUTAG: bool = False

# GENERIC GRAPH PARAMETERS. See utils.randomGraph for details
# Node and edge labels are initialized randomly. Target clusters are given by sklearn.
# Each graph has at least <min_nodes_number> nodes and at most <max_nodes_number> nodes
# Possible <aggregation_mode> for matrix ArcNoe belonging to graphs in ['average', 'normalized', 'sum']
# problem_based in ['n', 'a','g'] -> ['c' classification, 'r' regression]
# addressed_problem in ['c', 'r'] -> ['g' graph-based; 'n' node-based; 'a' arc-based;]
problem_based       : str = 'g'
addressed_problem   : str = 'c'
graphs_number       : int = 100
min_nodes_number    : int = 15
max_nodes_number    : int = 40
dim_node_label      : int = 3
dim_arc_label       : int = 1
dim_target          : int = 2
density             : float = 0.7
aggregation_mode    : str = 'average'

# LEARNING SETS PARAMETERS
perc_Train          : float = 0.7
perc_Valid          : float = 0.2
batch_size          : int = 32
normalize           : bool = True
seed                : Optional[int] = None
norm_nodes_range    : Optional[tuple[Union[int, float], Union[int, float]]] = None  # (-1,1) # other possible value
norm_arcs_range     : Optional[tuple[Union[int, float], Union[int, float]]] = None  # (0,1) # other possible value

# NET STATE PARAMETERS
activations_net_state   : str = 'selu'
kernel_init_net_state   : str = 'lecun_normal'
bias_init_net_state     : str = 'lecun_normal'
kernel_reg_net_state    : str = 'l1'
bias_reg_net_state      : str = 'l1'
dropout_rate_st         : float = 0.1
dropout_pos_st          : Union[list[int], int] = 0
hidden_units_net_state  : Optional[Union[list[int], int]] = [150, 150]

### NET OUTPUT PARAMETERS
activations_net_output  : str = 'softmax'
kernel_init_net_output  : str = 'glorot_normal'
bias_init_net_output    : str = 'glorot_normal'
kernel_reg_net_output   : str = 'l1'
bias_reg_net_output     : str = 'l1'
dropout_rate_out        : float = 0.1
dropout_pos_out         : Union[list[int], int] = 0
hidden_units_net_output : Optional[Union[list[int], int]] = [150]

# GNN PARAMETERS
dim_state       : int = 0
max_iter        : int = 5
state_threshold : float = 0.01

# LGNN PARAMETERS
layers          : int = 5
get_state       : bool = True
get_output      : bool = True
path_writer     : str = 'writer/'
optimizer       : tf.keras.optimizers = tf.optimizers.Adam(learning_rate=0.001)
lossF           : tf.function = tf.keras.losses.categorical_crossentropy
lossArguments   : Optional[dict[str, callable]] = {'from_logits': False}
extra_metrics   : Optional[dict[str, callable]] = {i: mt.Metrics[i] for i in
                                                   ['Acc', 'Bacc', 'Tpr', 'Tnr', 'Fpr', 'Fnr', 'Ck', 'Js', 'Prec', 'Rec', 'Fs']}
metrics_args    : Optional[dict[str, dict[str, any]]] = {i: {'average': 'weighted', 'zero_division': 0} for i in ['Fs', 'Prec', 'Rec', 'Js']}

#######################################################################################################################
# SCRIPT ##############################################################################################################
#######################################################################################################################

### LOAD DATASET
if use_MUTAG:
    # from MUTAG
    addressed_problem = 'c'
    problem_based = 'g'
    from load_MUTAG import graphs
else:
    # random graphs
    graphs = [utils.randomGraph(nodes_number=int(random.choice(range(min_nodes_number, max_nodes_number))),
                                dim_node_label=dim_node_label,
                                dim_arc_label=dim_arc_label,
                                dim_target=dim_target,
                                density=density,
                                normalize_features=False,
                                aggregation_mode=aggregation_mode,
                                problem_based=problem_based)
              for i in range(graphs_number)]

### PREPROCESSING
# SPLITTING DATASET in Train, Validation and Test set
iTr, iTe, iVa = utils.getindices(len(graphs), perc_Train, perc_Valid, seed=seed)
gTr = [graphs[i] for i in iTr]
gTe = [graphs[i] for i in iTe]
gVa = [graphs[i] for i in iVa]

# BATCHES - gTr is list of GraphObject; gVa and gTe are GraphObjects + use gTr[0] for taking useful dimensions
gTr = utils.getbatches(gTr, batch_size=batch_size, problem_based=problem_based, aggregation_mode=aggregation_mode)
gVa = GraphObject.merge(gVa, problem_based=problem_based, aggregation_mode=aggregation_mode)
gTe = GraphObject.merge(gTe, problem_based=problem_based, aggregation_mode=aggregation_mode)
gGen = gTr[0].copy()

# GRAPHS NORMALIZATION, based on training graphs
if normalize:
    utils.normalize_graphs(gTr, gVa, gTe,
                           based_on='gTr',
                           norm_rangeN=norm_nodes_range,
                           norm_rangeA=norm_arcs_range)

### MODELS
# NETS - STATE
input_net_st, layers_net_st = zip(*[get_inout_dims(net_name='state', dim_node_label=gGen.DIM_NODE_LABEL,
                                                   dim_arc_label=gGen.DIM_ARC_LABEL, dim_target=gGen.DIM_TARGET,
                                                   problem_based=problem_based, dim_state=dim_state,
                                                   hidden_units=hidden_units_net_state,
                                                   layer=i, get_state=get_state, get_output=get_output) for i in range(layers)])
nets_St = [MLP(input_dim=i, layers=j,
               activations=activations_net_state,
               kernel_initializer=kernel_init_net_state,
               bias_initializer=bias_init_net_state,
               kernel_regularizer=kernel_reg_net_state,
               bias_regularizer=bias_reg_net_state,
               dropout_rate=dropout_rate_st,
               dropout_pos=dropout_pos_st) for i, j in zip(input_net_st, layers_net_st)]

# NETS - OUTPUT
input_net_out, layers_net_out = zip(*[get_inout_dims(net_name='output', dim_node_label=gGen.DIM_NODE_LABEL,
                                                     dim_arc_label=gGen.DIM_ARC_LABEL, dim_target=gGen.DIM_TARGET,
                                                     problem_based=problem_based, dim_state=dim_state,
                                                     hidden_units=hidden_units_net_output,
                                                     layer=i, get_state=get_state, get_output=get_output) for i in range(layers)])
nets_Out = [MLP(input_dim=i, layers=j,
                activations=activations_net_output,
                kernel_initializer=kernel_init_net_output,
                bias_initializer=bias_init_net_output,
                kernel_regularizer=kernel_reg_net_output,
                bias_regularizer=bias_reg_net_output,
                dropout_rate=dropout_rate_out,
                dropout_pos=dropout_pos_out) for i, j in zip(input_net_out, layers_net_out)]

# GNNs
gnntype = {'n': GNNnodeBased, 'a': GNNedgeBased, 'g': GNNgraphBased}[problem_based]
# noinspection PyTypeChecker
gnns = [gnntype(net_state=st,
                net_output=out,
                optimizer=optimizer.__class__(**optimizer.get_config()),
                loss_function=lossF,
                loss_arguments=lossArguments,
                state_vect_dim=dim_state,
                max_iteration=max_iter,
                threshold=state_threshold,
                addressed_problem=addressed_problem,
                extra_metrics=extra_metrics,
                extra_metrics_arguments=metrics_args,
                path_writer=f'{path_writer}/GNN{idx}') for idx, st, out in zip(range(layers), nets_St, nets_Out)]

# SINGLE GNN
gnn = gnns[0].copy(path_writer=f'{path_writer}GNN_single', copy_weights=True)

# LGNN
lgnn = LGNN(gnns=gnns,
            get_state=get_state,
            get_output=get_output,
            optimizer=optimizer,
            loss_function=lossF,
            loss_arguments=lossArguments,
            addressed_problem=addressed_problem,
            extra_metrics=extra_metrics,
            extra_metrics_arguments=metrics_args,
            path_writer=f'{path_writer}LGNN',
            namespace='LGNN')
