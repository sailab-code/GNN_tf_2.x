# coding=utf-8

import os
import numpy as np

#######################################################################################################################
## GRAPH OBJECT CLASS #################################################################################################
#######################################################################################################################
class GraphObject:
	## CONSTRUCTORS METHODS ###########################################################################################
	def __init__(self, arcs, nodes, targets,
				 problem_based:str = 'n',
				 set_mask = None,
				 output_mask = None,
				 NodeGraph = None,
				 ArcNode = None,
				 node_aggregation:str = "average"):
		""" CONSTRUCTOR METHOD
		:param arcs: Ordered Arcs Matrix where arcs[i] = [ID Node From | ID NodeTo | Arc Label]
		:param nodes: Ordered Nodes Matrix where nodes[i] = [ID Node | Node Label]
		:param targets: Targets Array with shape (Num of targeted example [nodes or arcs], dim_target example)
		:param problem_based: (str) define the type of problem: 'a' arcs-based, 'g' graph-based, 'n' node-based
		:param set_mask: Array of {0,1} to define arcs/nodes belonging to a set, when dataset == single GraphObject
		:param output_mask: Array of {0,1} to define the sub-set of arcs/nodes whose output is needed
		:param NodesIndex: Matrix (arcs.shape[0],2) s.t. NodesIndex[i,j] = nodes[:,0].index(arcs[i,j])
		:param NodeGraph: Matrix (nodes.shape[0],{Num graphs|1}) s.t. node-based problem -> graph-based one
		:param ArcNode: Matrix of shape (num_of_arcs, num_of_nodes) s.t. A[i,j]=value if arc[i,2]==node[j]
		:param node_aggregation:Define how graph-based problem are faced. It can be 'average', 'sum' or 'normalized'
		"""
		# store arcs, nodes, targets
		self.arcs = arcs.astype(np.float32)
		self.nodes = nodes.astype(np.float32)
		self.targets = targets.astype(np.float32)
		# store dimensions
		self.DIM_NODE_LABEL = nodes.shape[1]  	# first column contains node indices (output_mask is not here!)
		self.DIM_ARC_LABEL = (arcs.shape[1] - 2)  	# first two columns contain nodes indices
		self.DIM_TARGET = targets.shape[1]
		# setting the problem type: node, arcs or graph based + check existence of passed parameters in keys
		lenMask = {'n':nodes.shape[0], 'a':arcs.shape[0], 'g':nodes.shape[0]}
		self.problem_based = problem_based
		# build set_mask, for a dataset composed of only a single graph: its nodes have to be divided in Tr, Va and Te
		self.set_mask = np.ones(lenMask[self.problem_based]) if set_mask is None else set_mask.astype(bool)
		# build output_mask
		self.output_mask = np.ones(len(self.set_mask)) if output_mask is None else output_mask.astype(bool)
		# check lengths: output_mask must be as long as set_mask
		if len(self.set_mask)!=len(self.output_mask): raise ValueError('Error - len(<set_mask>) != len(<output_mask>)')
		# build Adjancency Matrix
		self.Adjacency = self.buildAdiacencyMatrix()
		# build ArcNode tensor or acquire it from input
		self.ArcNode = self.buildArcNode(node_aggregation=node_aggregation) if ArcNode is None else ArcNode.astype(np.float32)
		# build node_graph conversion matrix
		self.NodeGraph = self.buildNodeGraph() if NodeGraph is None else NodeGraph.astype(np.float32)

	# -----------------------------------------------------------------------------------------------------------------
	def copy(self):
		""" COPY METHOD
		:return: a Deep Copy of the Graph Object instance.
		"""
		return GraphObject(arcs=self.getArcs(), nodes=self.getNodes(), targets=self.getTargets(),
						   problem_based=self.getProblemBased(), set_mask = self.getSetMask(),
						   output_mask=self.getOutputMask(), NodeGraph=self.getNodeGraph(), ArcNode=self.getArcNode())

	# -----------------------------------------------------------------------------------------------------------------
	def buildAdiacencyMatrix(self):
		from scipy.sparse import coo_matrix
		values = np.ones(self.arcs.shape[0], dtype=np.float32)
		indices = self.arcs[:,:2].astype(int)
		return coo_matrix((values, (indices[:, 0], indices[:, 1])), shape=(len(self.nodes), len(self.nodes)), dtype=np.float32)

	# -----------------------------------------------------------------------------------------------------------------
	def buildArcNode(self, node_aggregation:str):
		""" Build ArcNode Matrix A of shape (number_of_arcs, number_of_nodes) where A[i,j]=value if arc[i,2]==node[j]
		Compute the matmul(m:=message,A) to get the incoming message on each node
		:param node_aggregation: (str) It defines the aggregation mode for the incoming message of a node:
			> 'average': elem(A)={0-1} -> matmul(m,A) gives the average of incoming messages, s.t. sum(A[:,i])=1
			> 'normalized': elem(A)={0-1} -> matmul(m,A) gives the normalized message wrt the total number of g.nodes
			> 'sum': elem(A)={0,1} -> matmul(m,A) gives the total sum of incoming messages
		:return: sparse ArcNode Matrix, for memory efficiency
		:raise: Error if <node_aggregation> is not in ['average','sum','normalized']
		"""
		col = self.arcs[:, 1]  # column indices of A are located in the second column of the arcs tensor
		row = np.arange(0, len(col))  # arc id (from 0 to #arcs)
		values_vector = np.ones(len(col))
		val, col_index, destination_node_counts = np.unique(col, return_inverse=True, return_counts=True)
		if node_aggregation == "average": values_vector = values_vector / destination_node_counts[col_index]
		elif node_aggregation == "normalized": values_vector = values_vector * float(1 / float(len(col)))
		elif node_aggregation == "sum": pass
		else: raise ValueError("ERROR: Unknown aggregation mode")
		# isolated nodes correction: if nodes[i] is isolated, then ArcNode[:,i]=0, to maintain nodes ordering
		from scipy.sparse import coo_matrix
		return coo_matrix((values_vector, (row, self.arcs[:,1])), shape=(len(self.arcs), len(self.nodes)), dtype=np.float32)

	# -----------------------------------------------------------------------------------------------------------------
	def setArcNode(self, node_aggregation:str):
		self.ArcNode = self.buildArcNode(node_aggregation=node_aggregation)

	# -----------------------------------------------------------------------------------------------------------------
	def buildNodeGraph(self):
		""" Build Node-Graph Aggregation Matrix, to transform a node-based problem in a graph-based one.
		It has dimensions (nodes.shape[0], 1) for a single graph, or (nodes.shape[0], Num graphs) for a graph containing
		2+ graphs, built by merging the single graphs into a bigger one, such that after the node-graph aggregation
		process gnn can compute (Num graphs, targets.shape[1]) as output.
		It's normalized wrt the number of nodes whose output is computed, i.e. the number of ones in output_mask
		:return: node-graph matrix
		"""
		# nodes_output_coefficient = np.count_nonzero(self.output_mask)
		nodes_output_coefficient = self.nodes.shape[0]
		return np.ones((nodes_output_coefficient, 1), dtype=np.float32) * 1 / nodes_output_coefficient

	## GETTERS ########################################################################################################
	def getArcs(self): 			return self.arcs.copy()
	def getNodes(self): 		return self.nodes.copy()
	def getTargets(self): 		return self.targets.copy()
	def getProblemBased(self): 	return self.problem_based[:]
	def getSetMask(self): 		return self.set_mask.copy()
	def getOutputMask(self):	return self.output_mask.copy()
	def getAdjacency(self): 	return self.Adjacency.copy()
	def getArcNode(self): 		return self.ArcNode.copy()
	def getNodeGraph(self): 	return self.NodeGraph.copy()
	def initState(self,v:int=0):return np.zeros((self.nodes.shape[0],v)) if v>0 else self.nodes.copy()
	
	## STATIC METHODS #################################################################################################
	@staticmethod
	def load(graph_folder_path:str, problem_based:str, *, node_aggregation:str):
		""" Load a graph from a directory which contains at least 3 txt files referring to nodes, arcs and targets
		:param graph_folder_path: directory containing at least 3 files: 'nodes.txt', 'arcs.txt' and 'targets.txt'
			> other possible files: 'NodeGraph.txt','output_mask.txt' and 'set_mask.txt'. No other files required!
		:param node_aggregation: node aggregation mode: 'average','sum','normalized'. Go to BuildArcNode for details
		:param problem_based: (str) : 'n'-nodeBased; 'a'-arcBased; 'g'-graphBased
			> NOTE For graph_based problems, file 'NodeGraph.txt' has to be present in folder
		:return: GraphObject described by the files contained inside <graph_folder_path> folder
		"""
		# load all the files inside <graph_folder_path> folder
		if graph_folder_path[-1] != '/': graph_folder_path += '/'
		files = sorted(os.listdir(graph_folder_path))
		keys = [i.rsplit('.')[0] for i in files]+['problem_based', 'node_aggregation']
		vals = [np.loadtxt(graph_folder_path+i,delimiter=',',ndmin=2) for i in files]+[problem_based, node_aggregation]
		# create a dictionary with parameters and values to be passed to constructor and return GraphObject
		params = dict(zip(keys,vals))
		return GraphObject(**params)
	
	# -----------------------------------------------------------------------------------------------------------------
	@staticmethod
	def save(graph_folder_path:str, g, *, save_set_mask:bool=False, format:str='%.10g'):
		""" Save a graph to a directory, creating txt files referring to nodes, arcs, targets and possibly output_mask
		:param graph_folder_path: new directory for saving the graph
		:param g: graph of type GraphObject to be saved
		:param save_set_mask: (bool) if True, save also g.set_mask, by default a graph is loaded with set_mask==ones
		:param format: param to be passed to np.savetxt
		"""
		import shutil
		if graph_folder_path[-1] != '/': graph_folder_path+='/'
		if os.path.exists(graph_folder_path): shutil.rmtree(graph_folder_path)
		os.makedirs(graph_folder_path)
		np.savetxt(graph_folder_path + "arcs.txt", g.arcs, fmt=format, delimiter=',')
		np.savetxt(graph_folder_path + "nodes.txt", g.nodes, fmt=format, delimiter=',')
		np.savetxt(graph_folder_path + "targets.txt", g.targets, fmt=format, delimiter=',')
		if save_set_mask:
			np.savetxt(graph_folder_path + 'set_mask.txt', g.set_mask, fmt=format, delimiter=',')
		if not np.array_equal(g.output_mask, np.ones(len(g.output_mask))):
			np.savetxt(graph_folder_path + "out_mask.txt", g.output_mask, fmt=format, delimiter=',')
		if g.problem_based == 'g' and g.targets.shape[0]>1:
			np.savetxt(graph_folder_path + 'NodeGraph.txt', g.NodeGraph, fmt=format, delimiter=',')
			
	# -----------------------------------------------------------------------------------------------------------------
	@staticmethod
	def merge(glist, node_aggregation:str):
		""" Method to merge graphs: it takes in input a list of graphs and returns them as a single graph
		:param glist: list of GraphObjects
			> NOTE If glist[:].problem_based=='g', NodeGraph will have dimension (Num nodes, Num graphs), else (Num nodes,1)
		:param node_aggregation: str, node aggregation mode for new GraphObject
		:return: a new GraphObject containing all the information (nodes, arcs, targets) in glist
		"""
		# check parameters
		if not (type(glist) == list and all(isinstance(x, (GraphObject, str)) for x in glist)):
			raise TypeError('type of param <glist> must be list of str \'path-like\' or GraphObjects')
		# check problem_based parameter for all the graphs
		problem_based_set = list(set([i.problem_based for i in glist]))
		if len(problem_based_set)!=1 or problem_based_set[0] not in ['n','a','g']:
			raise ValueError('All graphs in <glist> must have the same <g.problem_based> parameter')
		# retrieve matrices from graph list
		problem_based = problem_based_set.pop()
		nodes, nodes_lens, arcs, targets, set_mask, output_mask, nodegraph = zip(*[(i.getNodes(), i.nodes.shape[0], i.getArcs(), i.getTargets(),
																					i.getSetMask(), i.getOutputMask(), i.getNodeGraph())
																				   for i in glist])
		# get single matrices for new graph
		for i, elem in enumerate(arcs): elem[:, :2] += sum(nodes_lens[:i])
		arcs = np.concatenate(arcs, axis=0)
		nodes = np.concatenate(nodes, axis=0)
		targets = np.concatenate(targets, axis=0)
		set_mask = np.concatenate(set_mask, axis=0)
		output_mask = np.concatenate(output_mask, axis=0)
		from scipy.linalg import block_diag
		nodegraph = None if problem_based != 'g' else block_diag(*nodegraph)
		# resulting GraphObject
		return GraphObject(arcs=arcs, nodes=nodes, targets=targets, set_mask=set_mask, output_mask=output_mask,
						   problem_based=problem_based, NodeGraph=nodegraph, node_aggregation=node_aggregation)