from Node import *
import random
import numpy as np


class Network:

    all_layer = []
    shapes = []


    def __init__(self, nodes, next_layer_nodes, input_layer = False, output_layer=False, batchnorm_alg=False):


        self.layer = len(Network.all_layer)
        self.is_output_layer = False
        self.input_or_pre_activation_matrix = np.array([])


        if output_layer:
            self.is_output_layer = True
            self.softmax = None
        else:
            self.next_layer = None

        if not input_layer:
            self.number_of_nodes = next_layer_nodes
            self.weights_matrix = None
            self.bias = None
            self.setRandomWeights(nodes, next_layer_nodes, batchnorm_alg)
            self.dbias = None
            self.dweight = None
            self.data_mean = None
            self.data_std = None
            self.matrix_computed_weight_act_bias = []
            self.layer_activation_matrix = np.array([])
            self.prev_layer = None
            self.input_layer = False
            self.layer_dropout = None
            self.layer_dropout_prob = None
        else:
            self.input_layer = True
            self.number_of_nodes = nodes


    def setRandomWeights(self, nodes, nextLayer, batchnorm):

        if batchnorm:
            self.weights_matrix = np.random.randn(nextLayer, nodes)


        else:
            self.weights_matrix = np.random.standard_normal(size=(nextLayer, nodes)) * np.sqrt(2./nodes)
        self.bias = np.zeros((nextLayer, 1))
        self.shapes.append([self.weights_matrix.shape, self.bias.shape ])

