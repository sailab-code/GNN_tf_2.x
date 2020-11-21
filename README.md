# Graph Neural Network Model - tf2.x
This repo contains a Tensorflow 2.x implementation of the Graph Neural Network Model.

- **Authors:** [Niccolò Pancino](http://sailab.diism.unisi.it/people/niccolo-pancino/), [Pietro Bongini](http://sailab.diism.unisi.it/people/pietro-bongini/)

## Install
### Requirements
The GNN framework requires the packages **tensorflow**, **numpy**, **pandas**, **sklearn**.

To install the requirements you can use the following command

    pip install -U -r requirements.txt


## Simple usage example
You can simply run starter.py to train a GNN on a dataset composed of graphs with random nodes/edges/targets. 

Open starter.py to modify parameters in Section *script options*. 
Examples on real-world datasets are coming... *Stay Tuned!*


## Citing
Part of the code was inspired by M. Tiezzi and A. Rossi [gnn](https://github.com/sailab-code/gnn) implementation.

To cite GNN please use the following publication:

    F. Scarselli, M. Gori,  A. C. Tsoi, M. Hagenbuchner, G. Monfardini, 
    "The Graph Neural Network Model", IEEE Transactions on Neural Networks,
    vol. 20(1); p. 61-80, 2009.
    
Bibtex:

    @article{Scarselli2009TheGN,
      title={The Graph Neural Network Model},
      author={Franco Scarselli and Marco Gori and Ah Chung Tsoi and Markus Hagenbuchner and Gabriele Monfardini},
      journal={IEEE Transactions on Neural Networks},
      year={2009},
      volume={20},
      pages={61-80}
    }

## License
Released under the 3-Clause BSD license (see `LICENSE.txt`):

    Copyright (C) 2004-2020 Niccolò Pancino
    Niccolò Pancino <niccolo.pancino@unifi.it>
    Pietro Bongini <pietro.bongini@gmail.com >
    Matteo Tiezzi <mtiezzi@diism.unisi.it>
