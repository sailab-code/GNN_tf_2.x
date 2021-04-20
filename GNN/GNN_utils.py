from __future__ import annotations

from typing import Union, Optional

import numpy as np

from GNN.graph_class import GraphObject


#######################################################################################################################
# FUNCTIONS ###########################################################################################################
#######################################################################################################################

# ---------------------------------------------------------------------------------------------------------------------
def randomGraph(nodes_number: int, dim_node_label: int, dim_arc_label: int, dim_target: int, density: float,
                *, normalize_features: bool = False, aggregation: str = 'average',
                problem_based: str = 'n') -> GraphObject:
    """ Create randoms nodes and arcs matrices, such that label of arc (i,j) == (j,i)

    :param nodes_number: number of nodes belonging to the graph
    :param dim_node_label: number of components for a generic node's label
    :param dim_arc_label: number of components for a generic arc's label
    :param dim_target: number of components for a generic target 1-hot
    :param density: define the "max" density for the graph
    :param normalize_features: (bool) if True normalize the column of the labels, otherwise raw data will be considered
    :param aggregation: (str) in ['average','normalized','sum']. Default 'average'. Go to GraphObject.buildArcNode() for details
    :param problem_based: (str) in ['n','a','g']: 'n'-nodeBased; 'a'-arcBased; 'g'-graphBased
    :return: GraphObject
    """
    # NODES
    nodes_ids = range(nodes_number)
    nodes = 2 * np.random.random((nodes_number, dim_node_label)) - 1

    # ARCS
    arcs_number = round(density * nodes_number * (nodes_number - 1) / 2)
    sources = np.random.choice(nodes_ids[:-1], arcs_number // 2)

    # max_diff is the maximum id for picking the destination node when random.random() is called
    max_diff = np.ones_like(sources) * nodes_number - sources - 1

    # destination obtained by adding a random id from 'source' to nodes_number
    destination = sources + np.ceil(max_diff * np.random.random(len(sources)))

    # arcs id node1 - id node2
    arcs_ascend = np.zeros((arcs_number // 2, 2))
    arcs_ascend[:, 0] = sources
    arcs_ascend[:, 1] = destination
    arcs_ascend = np.unique(arcs_ascend, axis=0)
    arcs_descend = np.flip(arcs_ascend, axis=1)

    # arc labels
    arcs_ids = np.concatenate((arcs_ascend, arcs_descend))
    arcs_label = 2 * np.random.random((arcs_ascend.shape[0], dim_arc_label)) - 1
    arcs_label = np.concatenate((arcs_label, arcs_label))
    arcs = np.concatenate((arcs_ids, arcs_label), axis=1)
    arcs = np.unique(arcs, axis=0)

    # TARGETS - 1-HOT
    tn = {'n': nodes.shape[0], 'a': arcs.shape[0], 'g': 1}
    assert problem_based in tn.keys()
    target_number = tn[problem_based]
    targs = np.zeros((target_number, dim_target))

    # clusters
    if problem_based in ['a', 'n']:
        from sklearn.cluster import AgglomerativeClustering
        j = AgglomerativeClustering(n_clusters=dim_target).fit(arcs[:, 2:] if problem_based == 'a' else nodes)
        i = range(target_number)
        targs[i, j.labels_] = 1
    else:
        targs[0, np.random.choice(range(targs.shape[1]))] = 1

    # OUTPUT MASK - all nodes/arcs have known target
    output_mask = np.ones(arcs.shape[0] if problem_based == 'a' else nodes.shape[0], dtype=bool)

    # NORMALIZE FEATURES
    if normalize_features:
        nodes = nodes / np.max(nodes, axis=0)
        arcs[:, 2:] = arcs[:, 2:] / np.max(arcs[:, 2:], axis=0)

    # RETURN GRAPH
    return GraphObject(arcs=arcs, nodes=nodes, targets=targs, problem_based=problem_based,
                       output_mask=output_mask, node_aggregation=aggregation)


# ---------------------------------------------------------------------------------------------------------------------
def simple_graph(problem_based: str, aggregation: str = 'average') -> GraphObject:
    """ return a single simple GraphObject for debugging purpose """
    nodes = np.array([[11, 21], [12, 22], [13, 23], [14, 24]])
    arcs = np.array([[0, 1, 10], [0, 2, 40], [1, 0, 10], [1, 2, 20], [2, 0, 40], [2, 1, 20], [2, 3, 30], [3, 2, 30]])

    tn = {'n': nodes.shape[0], 'a': arcs.shape[0], 'g': 1}
    target_number = tn[problem_based]
    targs = np.zeros((target_number, 2))

    if problem_based in ['a', 'n']:
        from sklearn.cluster import AgglomerativeClustering
        j = AgglomerativeClustering(n_clusters=2).fit(arcs[:, 2:] if problem_based == 'a' else nodes)
        i = range(target_number)
        targs[i, j.labels_] = 1
    else:
        targs[0, 1] = 1

    return GraphObject(arcs=arcs, nodes=nodes, targets=targs, problem_based=problem_based, node_aggregation=aggregation)


# ---------------------------------------------------------------------------------------------------------------------
def progressbar(percent: float, width: int = 30) -> None:
    """ Print a progressbar, given a percentage in [0,100] and a fixed length """
    left = round(width * percent / 100)
    right = int(width - left)
    print('\r[', '#' * left, ' ' * right, ']', f' {percent:.1f}%', sep='', end='', flush=True)


# ---------------------------------------------------------------------------------------------------------------------
def getindices(len_dataset: int, perc_Train: float = 0.7, perc_Valid: float = 0.1, seed=None) -> Union[
    tuple[list[int], list[int]], tuple[list[int], list[int], list[int]]]:
    """ Divide the dataset into Train/Test or Train/Validation/Test

    :param len_dataset: length of the dataset
    :param perc_Train: (float) in [0,1]
    :param perc_Valid: (float) in [0,1]
    :param seed: (float/None/False) Fixed shuffle mode / random shuffle mode / no shuffle performed
    :return: 2 or 3 list of indices
    """
    if perc_Train < 0 or perc_Valid < 0 or perc_Train + perc_Valid > 1:
        raise ValueError('Error - percentage must stay in [0-1] and their sum must be <= 1')

    # shuffle elements
    idx = list(range(len_dataset))
    if seed: np.random.seed(seed)
    if seed is not False: np.random.shuffle(idx)

    # samples
    perc_Test = 1 - perc_Train - perc_Valid
    sampleTest = round(len_dataset * perc_Test)
    sampleValid = round(len_dataset * perc_Valid)

    # test indices
    test_idx = idx[:sampleTest]

    # validation indices
    valid_idx = idx[sampleTest:sampleTest + sampleValid]

    # train indices (usually the longest set)
    train_idx = idx[sampleTest + sampleValid:]

    return (train_idx, test_idx, valid_idx)


# ---------------------------------------------------------------------------------------------------------------------
def getSet(glist: list[GraphObject], set_indices: list[int], problem_based: str, node_aggregation: str,
           verbose: bool = False) -> list[GraphObject]:
    """ get the Set from a dataset given its set of indices

    :param glist: (list of GraphObject or str) dataset from which the set is picked
    :param set_indices: (list of int) indices of the elements belonging to the Set
    :param problem_based: (str) in ['n','a','g'] defining the problem to be faced: [node, arcs, graph]-based
    :param verbose: (bool) if True print the progressbar, else silent mode
    :return: list of GraphObjects, composing the Set
    """
    # check type
    if not (type(glist) == list and all(isinstance(x, str) for x in glist)):
        raise TypeError('type of param <glist> must be list of str \'path-like\' or GraphObjects')

    # get set
    length, setlist = len(set_indices), list()
    for i, elem in enumerate(set_indices):
        setlist.append(glist[elem])
        if verbose: progressbar((i + 1) * 100 / length)

    return [GraphObject.load(i, problem_based=problem_based, node_aggregation=node_aggregation) for i in setlist]


# ---------------------------------------------------------------------------------------------------------------------
def getbatches(glist: list[GraphObject], problem_based: str, node_aggregation: str, batch_size: int = 32, number_of_batches=None,
               one_graph_per_batch=True) -> Union[list[GraphObject], list[list[GraphObject]]]:
    """ Divide the Set into batches, in which every batch is a GraphObject or a list of GraphObject

    :param glist: (list of GraphObject) to be split into batches
    :param batch_size: (int) specify the size of a normal batch. Note: len(batches[-1])<=batch_size
    :param number of batches: (int) specify in how many batches glist will be partitioned.
                                > Default value is None; if given, param <batch_size> will be ignored.
    :param one_graph_per_batch: (bool) if True, all graphs belonging to a batch are merged to form a single GraphObject
    :return: a list of batches
    """
    if number_of_batches is None:
        batches = [glist[i:i + batch_size] for i in range(0, len(glist), batch_size)]
    else:
        batches = [list(i) for i in np.array_split(glist, number_of_batches)]
    if one_graph_per_batch: batches = [GraphObject.merge(i, problem_based=problem_based, node_aggregation=node_aggregation) for i in batches]
    return batches


# ---------------------------------------------------------------------------------------------------------------------
def normalize_graphs(gTr: Union[GraphObject, list[GraphObject]], gVa: Optional[Union[GraphObject, list[GraphObject]]],
                     gTe: Optional[Union[GraphObject, list[GraphObject]]], based_on: str = 'gTr',
                     norm_rangeN: Optional[Union[tuple[float, float]]] = None,
                     norm_rangeA: Optional[Union[tuple[float, float], None]] = None) -> None:
    """ Normalize graph by using gTr or gTr+gVa+gTe information

    :param gTr: (GraphObject or list of GraphObjects) for Training Set
    :param gVa: (GraphObject or list of GraphObjects or None) for Validation Set
    :param gTe: (GraphObject or list of GraphObjects or None) for Test Set
    :param based_on: (str) in ['gTr','all']. If 'gTr', only gTr data is used; if 'all', entire dataset data is used
    """

    def checktype(g: Union[list[GraphObject], None], name: str):
        """ check g: it must be a GraphObect or a list of GraphObjects """
        if g is None: return list()
        if not (type(g) == GraphObject or (type(g) == list and all(isinstance(x, GraphObject) for x in g))):
            raise TypeError(f'type of param <{name}> must be GraphObject or list of Graphobjects')
        return g if type(g) == list else [g]

    # check if inputs are GraphObject OR list(s) of GraphObject(s) and the normalization method
    gTr, gVa, gTe = map(checktype, [gTr, gVa, gTe], ['gTr', 'gVa', 'gTe'])
    if based_on not in ['gTr', 'all']: raise ValueError('param <based_on> must be \'gTr\' or \'all\'')

    # merge all the graphs into a single one
    G = GraphObject.merge(gTr, problem_based='n', node_aggregation='sum')
    if based_on == 'all': G = GraphObject.merge(G + gTe + gVa, problem_based='n', node_aggregation='sum')

    from sklearn.preprocessing import MinMaxScaler
    node_scaler = MinMaxScaler(feature_range=(0, 1) if norm_rangeN is None else norm_rangeN)
    arcs_scaler = MinMaxScaler(feature_range=(0, 1) if norm_rangeA is None else norm_rangeA)

    node_scaler.fit(G.nodes)
    arcs_scaler.fit(G.arcs)

    for i in gTr + gVa + gTe:
        i.nodes = node_scaler.transform(i.nodes)
        i.arcs = arcs_scaler.transform(i.arcs)