# Quadratic Assignment with Graph Neural Networks

Code accompanying the paper: [A Note on Learning Algorithms for Quadratic Assignment with Graph Neural Networks](https://arxiv.org/pdf/1706.07450.pdf)

# Reproduce Experiments
## Prerequisites

- Computer with Linux or OSX
- [PyTorch](http://pytorch.org)
- For training, an NVIDIA GPU is needed. CPU not supported.

## Graph Matching
```
python src/qap/main.py \
--path_dataset '' \ # path to load/save dataset
--path_logger '' \ # path to log experiment
--num_examples_train 20000 \ # number of training examples
--num_examples_test 1000 \ # number of testing examples
--edge_density 0.2 \ # edge density of the generated graph
--random_noise \ # set noise to unif(0, 0.05)
--noise 0.03 \ # fix noise level
--noise_model 2 # model 1/2 of https://pdfs.semanticscholar.org/1070/2de697c21155782346001189c25fe036dedd.pdf (eqs 3.8 and 3.9)
--generative_model 'ErdosRenyi' \ # generative model of graphs
--iterations 60000 \ # total number of iterations
--batch_size 1 \ # size of training batches
--mode 'train' \ # mode train/test of SiameseGNN
--clip_grad_norm 40.0 \ # clip norm level
--num_features 20 \ # number of features of GNN
--num_layers 20 \ # number of layers of GNN
--J 4 \ # number of dyadic powers of the adjacency matrix appearing in the operators set of the GNN
```
## TSP
```
python src/tsp/main.py \
--path_dataset '' \ # path to load/save dataset
--path_logger '' \ # path to log experiment
--path_tsp '' \ # path to the LKH folder (inside tsp/)
--num_examples_train 20000 \ # number of training examples
--num_examples_test 1000 \ # number of testing examples
--iterations 60000 \ # total number of iterations
--batch_size 32 \ # size of training batches
--beam_size 40 \ # size of the beam to generate the path
--mode 'train' \ # mode train/test of SiameseGNN
--clip_grad_norm 40.0 \ # clip norm level
--num_features 20 \ # number of features of GNN
--num_layers 20 \ # number of layers of GNN
--J 4 \ # number of dyadic powers of the adjacency matrix appearing in the operators set of the GNN
--N 20 \ # number of cities of geometric TSP
```
