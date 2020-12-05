import numpy as np
import tensorflow as tf
import GNN.GNN as GNN
import GNN.GNN_metrics as mt
import GNN.GNN_utils as utils
from GNN.graph_class import GraphObject


#######################################################################################################################
# Possible values for extra_metrics ###################################################################################
#######################################################################################################################
Metrics = {'Acc': mt.accuracy_score, 'Bacc': mt.balanced_accuracy_score, 'Js': mt.jaccard_score,
           'Ck': mt.cohen_kappa_score, 'Prec': mt.precision_score, 'Rec': mt.recall_score,
           'Fs': mt.fbscore, 'Tpr': mt.TPR, 'Tnr': mt.TNR, 'Fpr': mt.FPR, 'Fnr': mt.FNR}


#######################################################################################################################
# SCRIPT OPTIONS - modify the parameters to adapt the execution to the problem under consideration ####################
#######################################################################################################################
### USE MUTAG DATASET - Graph-Based Problem (problem_type=='cg1', see below for details)
use_MUTAG   : bool = False


### GENERIC GRAPH PARAMETERS. See utils.randomGraph for details
# Node and edge labels are initialized randomly. Target clusters are given by sklearn. 
# Possible node_aggregation for matrix ArcNoe belonging to graphs in ['average', 'normalized', 'sum']
# Possible problem_type (str) s.t. len(problem_tye) in[2,3]: 'outputModel + problemAddressed + typeGNNTobeUsed'.
# > ['c' classification, 'r' regression] + ['g' graph-based; 'n' node-based; 'a' arc-based;] +[('','1') GNN1, '2' GNN2]
# > Example: 'cn' or 'cn1': node-based classification with GNN; 'ra2' arc-based regression with GNN2 (Rossi-Tiezzi)
problem_type        : str   = 'cn'
graphs_number       : int   = 100
max_nodes_number    : int   = 40  # each graph has at most 30 nodes
min_nodes_number    : int   = 15  # each graph has at least 30 nodes
dim_node_label      : int   = 5
dim_arc_label       : int   = 2
dim_target          : int   = 2
density             : float = 0.7
node_aggregation    : str   = 'sum'

### LEARNING SETS PARAMETERS
perc_Train          : float = 0.8
perc_Valid          : float = 0.1
batch_size          : int   = 32
normalize           : bool  = True
seed                        = None
norm_nodes_range            = None  # (-1,1) # other possible value
norm_arcs_range             = None  # (0,1) # other possible value

### NET STATE PARAMETERS
hidden_units_net_state  : list  = [150, 150]
activations_net_state   : str   = 'selu'
kernel_init_net_state   : str   = 'lecun_uniform'
bias_init_net_state     : str   = 'lecun_uniform'
dropout_rate_st         : float = 0.1
dropout_pos_st          : list  = [0]

### NET OUTPUT PARAMETERS
hidden_units_net_output : list  = [150]
activations_net_output  : str   = 'linear'
kernel_init_net_output  : str   = 'glorot_uniform'
bias_init_net_output    : str   = 'glorot_uniform'
dropout_rate_out        : float = 0.1
dropout_pos_out         : list  = [0]

### GNN PARAMETERS
learning_rate   = 0.001
optgnn          = tf.optimizers.Adam(learning_rate=learning_rate)
lossF           = tf.nn.softmax_cross_entropy_with_logits
lossArguments   = None
extra_metrics   = {i: Metrics[i] for i in ['Acc', 'Bacc', 'Tpr', 'Tnr', 'Fpr', 'Fnr', 'Ck', 'Js', 'Prec', 'Rec', 'Fs']}
metrics_args    = {i: {'avg': 'weighted', 'pos_label': None} for i in ['Fs', 'Prec', 'Rec', 'Js']}
output_f        = tf.keras.activations.softmax
epochs          = 100
max_iter        = 5
state_threshold = 0.01
path_writer     = 'writer'
dim_state       = 0

### LEARNING / TEST OPTIONS
training    : True  = False
testing     : True  = False
rocdir              = 'roc'


#######################################################################################################################
# SCRIPT ##############################################################################################################
#######################################################################################################################

if use_MUTAG:
    print('> Loading MUTAG')
    ### CREATE DATASET FROM MUTAG
    problem_type = 'cg1'
    from load_MUTAG import graphs
else:
    print('> Creating Dataset')
    ### CREATE RANDOM DATASET
    if len(problem_type) == 2: problem_type += '1'
    graphs = [utils.randomGraph(nodes_number=int(np.random.choice(range(min_nodes_number,max_nodes_number))),
                                dim_node_label=dim_node_label,
                                dim_arc_label=dim_arc_label,
                                dim_target=dim_target,
                                density=density,
                                normalize_features=False,
                                aggregation=node_aggregation,
                                problem_based=problem_type[1])
              for i in range(graphs_number)]

### PRINT SETUP
print('Problem Addressed:\t{} \nProblem Based:\t\t{} \nGNN:\t\t\t\t{}\n'.format(*problem_type.upper()))

### SPLITTING DATASET in Train, Validation and Test set
problem_based = problem_type[1]
print('Splitting Graphs in Sets')
iTr, iVa, iTe = utils.getindices(len(graphs), perc_Train, perc_Valid, seed=seed)
gTr = [graphs[i] for i in iTr]
gVa = [graphs[i] for i in iVa]
gTe = [graphs[i] for i in iTe]

### BATCHES - gTr is list of GraphObject; gVa and gTe are GraphObjects + use gTr[0] for taking useful dimensions
print('Creating Batches and Merging Graphs')
gTr = utils.getbatches(gTr, batch_size=batch_size, node_aggregation=node_aggregation)
gVa = GraphObject.merge(gVa, node_aggregation=node_aggregation)
gTe = GraphObject.merge(gTe, node_aggregation=node_aggregation)
gGen = gTr[0].copy()

### GRAPHS NORMALIZATION, based on training graphs
if normalize:
    utils.normalize_graphs(gTr, gVa, gTe,
                           based_on='gTr',
                           norm_rangeN=norm_nodes_range,
                           norm_rangeA=norm_arcs_range)

### NETS - STATE
input_net_st, layers_net_st = utils.get_inout_dims(net_name='state', dim_node_label=gGen.DIM_NODE_LABEL,
                                                   dim_arc_label=gGen.DIM_ARC_LABEL, dim_target=gGen.DIM_TARGET,
                                                   problem=problem_type[1:], dim_state=dim_state,
                                                   hidden_units=hidden_units_net_state)
netSt = utils.MLP(input_dim=input_net_st,
                  layers=layers_net_st,
                  activations=activations_net_state,
                  kernel_initializer=kernel_init_net_state,
                  bias_initializer=bias_init_net_state,
                  dropout_percs=dropout_rate_st,
                  dropout_pos=dropout_pos_st)

### NETS - OUTPUT
input_net_out, layers_net_out = utils.get_inout_dims(net_name='output', dim_node_label=gGen.DIM_NODE_LABEL,
                                                     dim_arc_label=gGen.DIM_ARC_LABEL, dim_target=gGen.DIM_TARGET,
                                                     problem=problem_type[1:], dim_state=dim_state,
                                                     hidden_units=hidden_units_net_output)
netOut = utils.MLP(input_dim=input_net_out,
                   layers=layers_net_out,
                   activations=activations_net_output,
                   kernel_initializer=kernel_init_net_output,
                   bias_initializer=bias_init_net_output,
                   dropout_percs=dropout_rate_out,
                   dropout_pos=dropout_pos_out)

### GNN
gnntype = {'n1': GNN.GNN, 'n2': GNN.GNN2, 'a1': GNN.GNNedgeBased, 'g1': GNN.GNNgraphBased}
gnn = gnntype[problem_type[1:]](net_state=netSt,
                                net_output=netOut,
                                optimizer=optgnn,
                                loss_function=lossF,
                                loss_arguments=lossArguments,
                                output_activation=output_f,
                                max_iteration=max_iter,
                                threshold=state_threshold,
                                path_writer=path_writer,
                                addressed_problem=problem_type[0],
                                extra_metrics=extra_metrics,
                                metrics_arguments=metrics_args,
                                state_vect_dim=dim_state)

if training: gnn.train(gTr,epochs=epochs, gVa=gVa)
if testing:
    metrics = gnn.test(gTe, acc_classes=True, rocdir=rocdir)
    print('\nTest Res')
    for i in metrics: print(i, metrics[i])



