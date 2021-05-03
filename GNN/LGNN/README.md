# LGNN - Layered Graph Neural Network Model
This repo contains a Tensorflow 2.x implementations of the Layered Graph Neural Network (LGNN) Model.

**Author:** [Niccolò Pancino](http://sailab.diism.unisi.it/people/niccolo-pancino/)


## Simple usage example
In the following scripts, lgnn is by default a 5-layered GNN trained to solve a binary node-focused classification task on graphs with random nodes/edges/targets

Open the script `starter` and set parameters in section *SCRIPT OPTIONS* to change dataset and/or GNN/LGNN models architectures and learning behaviour.

In particular, set `use_MUTAG=True` to get the real-world dataset MUTAG for solving a graph-based problem ([details](https://github.com/NickDrake117/GNN_tf_2.x/blob/main/MUTAG_raw/Mutagenicity_label_readme.txt))

Note that a single layered LGNN behaves exaclty like a GNN, as it is composed of a single GNN.


### Single model training and testing
LGNN can be trained in parallel, serial or residual mode. 

- *Serial Mode*: GNNs layers are trained separately, one by one; Each GNN layer is trained as a standalone GNN model, therefore becoming an *expert* which solves the considered problem using the original data and the experience obtained from the previous GNN layer, so as to "correct" the errors made by the previous network, rather than solving the whole problem.

- *Parallel Mode*: GNN layers are trained simultaneously, by processing loss on the output of each GNNs layers, e.g. `loss = sum( loss_function(targets, oi) for oi in gnns_layers_output)` and backpropagating the error throughout the GNN layers;

- *Residual Mode*: GNN layers are trained simultaneously, by processing loss on the sum of the outputs of all GNNs layers, e.g. `loss = loss_function(targets, sum(oi for oi in gnns_layers_output))` and backpropagating the error throughout the GNN layers.

Training mode can be set as argument when calling `LGNN.train()`. Default value is `parallel`.


To perform both lgnn training and testing, run:

    from starter import lgnn, gTr, gTe, gVa
    
    epochs = 200
    
    # As mentioned, arg training_mode affects LGNN training process. 
    # lgnn.train(gTr, epochs, gVa, training_mode='serial')
    # lgnn.train(gTr, epochs, gVa, training_mode='residual')
    lgnn.train(gTr, epochs, gVa)
    
    # test the lgnn
    res = lgnn.test(gTe)

    # print test result
    for i in res:  
        print('{}: \t{:.4g}'.format(i, res[i]))

### K-fold Cross Validation
To perform a 10-fold cross validation on lgnn, run:

    from starter import lgnn, graphs
    from numpy import mean
    
    epochs = 200
    
    # LKO: as mentioned, arg training_mode affects LGNN training process
    # lko_res = lgnn.LKO(graphs, 10, epochs=epochs, training_mode='serial')
    # lko_res = lgnn.LKO(graphs, 10, epochs=epochs, training_mode='residual')
    lko_res = lgnn.LKO(graphs, 10, epochs=epochs)
    
    # print test result
    for i in lko_res: 
        for i in m: print('{}: \t{:.4f} \t{}'.format(i, mean(lko_res[i]), lko_res[i]))


### TensorBoard
To visualize learning progress, use TensorBoard --logdir command providing the log directory. Default value is `writer`.

    ...\projectfolder> tensorboard --logdir writer


## Citing
### Implementation
To cite the LGNN implementation please use the following publication:

    Pancino, N., Rossi, A., Ciano, G., Giacomini, G., Bonechi, S., Andreini, P., Scarselli, F., Bianchini, M., Bongini, P. (2020),
    "Graph Neural Networks for the Prediction of Protein–Protein Interfaces",
    In ESANN 2020 proceedings (pp.127-132).
    
Bibtex:

    @inproceedings{Pancino2020PPI,
      title={Graph Neural Networks for the Prediction of Protein–Protein Interfaces},
      author={Niccolò Pancino, Alberto Rossi, Giorgio Ciano, Giorgia Giacomini, Simone Bonechi, Paolo Andreini, Franco Scarselli, Monica Bianchini, Pietro Bongini},
      booktitle={28th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (online event)},
      pages={127-132},
      year={2020}
    }


---------
### Original Paper
To cite LGNN please use the following publication:

    N. Bandinelli, M. Bianchini and F. Scarselli, 
    "Learning long-term dependencies using layered graph neural networks", 
    The 2010 International Joint Conference on Neural Networks (IJCNN), 
    Barcelona, 2010, pp. 1-8, doi: 10.1109/IJCNN.2010.5596634.
    
Bibtex:

    @inproceedings{Scarselli2010LGNN,
      title={Learning long-term dependencies using layered graph neural networks}, 
      author={Niccolò Bandinelli, Monica Bianchini, Franco Scarselli},
      booktitle={The 2010 International Joint Conference on Neural Networks (IJCNN)}, 
      year={2010},
      volume={},
      pages={1-8},
      doi={10.1109/IJCNN.2010.5596634}
    }
