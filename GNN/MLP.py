from __future__ import annotations

from typing import Union, Optional

from numpy import array, arange
from tensorflow.keras.layers import Dense, Dropout, AlphaDropout
from tensorflow.keras.models import Sequential


# ---------------------------------------------------------------------------------------------------------------------
def MLP(input_dim: int, layers: list[int], activations, kernel_initializer, bias_initializer,
        dropout_rate: Union[list[float], float, None] = None, dropout_pos: Optional[Union[list[int], int]] = None,
        alphadropout: bool = False):
    """ Quick building function for MLP model. All lists must have the same length

    :param input_dim: (int) specify the input dimension for the model
    :param layers: (int or list of int) specify the number of units in every layers
    :param activations: (functions or list of functions)
    :param kernel_initializer: (initializers or list of initializers) for weights initialization (NOT biases)
    :param bias_initializer: (initializers or list of initializers) for biases initialization (NOT weights)
    :param dropout_rate: (float) s.t. 0 <= dropout_percs <= 1 for dropout rate
    :param dropout_pos: int or list of int describing dropout layers position
    :param alphadropout: (bool) for dropout type, if any
    :return: Sequential (MLP) model
    """

    # check type
    if type(activations) != list: activations = [activations for _ in layers]
    if type(kernel_initializer) != list: kernel_initializer = [kernel_initializer for _ in layers]
    if type(bias_initializer) != list: bias_initializer = [bias_initializer for _ in layers]
    if type(dropout_pos) == int:  dropout_pos = [dropout_pos]
    if type(dropout_rate) == float: dropout_rate = [dropout_rate for _ in dropout_pos]
    if dropout_rate == None or dropout_pos == None: dropout_rate, droout_pos = list(), list()

    # check lengths
    if not (len(activations) == len(kernel_initializer) == len(bias_initializer) == len(layers)):
        raise ValueError('Dense parameters must have the same length to be correctly processed')
    if len(dropout_rate) != len(dropout_pos):
        raise ValueError('Dropout parameters must have the same length to be correctly processed')

    # Dense layers
    keys = ['units', 'activation', 'kernel_initializer', 'bias_initializer']
    vals = [[layers[i], activations[i], kernel_initializer[i], bias_initializer[i]] for i in range(len(layers))]
    params = [dict(zip(keys, i)) for i in vals]

    # Dropout layers
    if dropout_rate and dropout_pos:
        dropout_pos = list(array(dropout_pos) + arange(len(dropout_pos)))
        for i, elem in enumerate(dropout_rate): params.insert(dropout_pos[i], {'rate': elem})

    # set input shape for first layer
    params[0]['input_shape'] = (input_dim,)

    # return MLP model
    dropout = AlphaDropout if alphadropout else Dropout
    mlp_layers = [Dense(**i) if 'units' in i else dropout(**i) for i in params]
    return Sequential(mlp_layers)


# ---------------------------------------------------------------------------------------------------------------------
def get_inout_dims(net_name: str, dim_node_label: int, dim_arc_label: int, dim_target: int, problem: str, dim_state: int,
                   hidden_units: Union[None, int, list[int]],
                   *, layer: int = 0, get_state: bool = False, get_output: bool = False) -> tuple[int, list[int]]:
    """ Calculate input and output dimension for the MLP of state and output

    :param net_name: (str) in ['state','output']
    :param dim_node_label: (int) dimension of node label
    :param dim_arc_label: (int) dimension of arc label
    :param dim_target: (int) dimension of target
    :param problem: (str) s.t. len(problem) in [1,2] -> [{'a','n','g'} | {'1','2'}]
    :param dim_state: (int)>=0 for state dimension paramenter of the gnn
    :param hidden_units: (int or list of int) for specifying units on hidden layers
    :param layer: (int) LGNN USE: get the dims at gnn of the layer <layer>, from graph dims on layer 0. Default is 0, since GNN==LGNN in this case
    :param get_state: (bool) LGNN USE: set accordingly to LGNN behaviour, if gnns get state, output or both from previous layer
    :param get_output: (bool) LGNN USE: set accordingly to LGNN behaviour, if gnns get state, output or both from previous layer
    """
    assert net_name in ['state', 'output']
    assert len(problem) in [1,2]
    assert layer >= 0

    if len(problem) == 1: problem += '1'
    DS = dim_state
    NL, AL, T = dim_node_label, dim_arc_label, dim_target

    # if LGNN, get MLPs layers for gnn in layer 2+
    if layer > 0:
        GS, GO, P = get_state, get_output, problem[0]
        if DS != 0:
            NL = NL + DS * GS + T * (P != 'a') * GO
            AL = AL + T * (P == 'a') * GO
        else:
            NL = NL + layer * NL * GS + ((layer - 1) * GS + 1) * T * (P != 'a') * GO
            AL = AL + T * (P == 'a') * GO

    # MLP state
    if net_name == 'state':
        input_shape = AL + 2 * NL + DS * (1 + (problem[1] == '1'))
        input_shape += NL * (DS == 0) * (problem[1] == '2')
        output_shape = DS if DS else NL
        
    # MLP output
    elif net_name == 'output':
        input_shape = (problem[0] == 'a') * (NL + AL + DS) + NL + dim_state
        output_shape = T
    
    else:
        raise ValueError()
    # hidden part
    if hidden_units is None or type(hidden_units) == int and hidden_units <= 0: hidden_units = []
    if type(hidden_units) == list:
        layers = hidden_units + [output_shape]
    else:
        layers = [hidden_units, output_shape]
    return input_shape, layers
