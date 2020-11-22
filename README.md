# Graph Neural Network Model - TF 2.x
This repo contains a Tensorflow 2.x implementation of the Graph Neural Network Model.

- **Authors:** [Niccolò Pancino](http://sailab.diism.unisi.it/people/niccolo-pancino/), [Pietro Bongini](http://sailab.diism.unisi.it/people/pietro-bongini/)

## Install
### Requirements
The GNN framework requires the packages **tensorflow**, **numpy**, **pandas**, **sklearn**, **matplotlib**.

To install the requirements you can use the following command

    pip install -U -r requirements.txt


## Simple usage example
Simply run starter.py to train a GNN on a dataset composed of graphs with random nodes/edges/targets. 

In starter.py set parameters in Section *script options* to change dataset and/or GNN behavihor, then run the script. 

Set 

    use_MUTAG=True 
   
to train a GNN on the real-world dataset MUTAG, (graph-based problem)



## Citing
To cite the GNN implementation please use the following publication:

    Pancino, N., Rossi, A., Ciano, G., Giacomini, G., Bonechi, S., Andreini, P., Scarselli, F., Bianchini, M., Bongini, P. (2020),
    "Graph Neural Networks for the Prediction of Protein–Protein Interfaces",
    In ESANN 2020 proceedings (pp.127-132).
    
Bibtex:

    @proceedings{Pancino2020PPI,
      title={Graph Neural Networks for the Prediction of Protein–Protein Interfaces},
      author={Niccolò Pancino, Alberto Rossi, Giorgio Ciano, Giorgia Giacomini, Simone Bonechi, Paolo Andreini, Franco Scarselli, Monica Bianchini, Pietro Bongini},
      booktitle={28th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (online event)},
      pages={127-132},
      year={2020}
    }

---------

To cite GNN please use the following publication:

    F. Scarselli, M. Gori,  A. C. Tsoi, M. Hagenbuchner, G. Monfardini, 
    "The Graph Neural Network Model", IEEE Transactions on Neural Networks,
    vol. 20(1); p. 61-80, 2009.
    
Bibtex:

    @article{Scarselli2009TheGN,
      title={The Graph Neural Network Model},
      author={Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, Gabriele Monfardini},
      journal={IEEE Transactions on Neural Networks},
      year={2009},
      volume={20},
      pages={61-80}
    }

## Contributions
Part of the code was inspired by [M. Tiezzi](http://sailab.diism.unisi.it/people/matteo-tiezzi/) and [A. Rossi](http://sailab.diism.unisi.it/people/alberto-rossi/) GNN implementation [repo](https://github.com/sailab-code/gnn) .

## License
Released under the 3-Clause BSD license (see `LICENSE.txt`):

    Copyright (C) 2004-2020 Niccolò Pancino
    Niccolò Pancino <niccolo.pancino@unifi.it>
    Pietro Bongini <pietro.bongini@gmail.com >
    Matteo Tiezzi <mtiezzi@diism.unisi.it>
