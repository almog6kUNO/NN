from Network import *
import numpy as np
import datetime


class NN:

    def __init__(self, layers, nodes_per_layer, first_input_data, batchnorm=False):

        self.layers = layers
        self.nodes_per_layer = nodes_per_layer
        self.max_pool_reshape_vect = []
        self.max_pool_reshape = []
        self.loss = None
        self.dweight = []
        self.dbias = []
        set_batchnorm_values = False


        #Create FCN layers
        for num in range(layers):
            if num == 0:
                #Create first input layer
                new_layer = Network(nodes_per_layer[num], 0, input_layer= True, batchnorm_alg=batchnorm)

            elif 0 < num < layers - 1:
                #Create layers in between
                new_layer = Network(nodes_per_layer[num-1], nodes_per_layer[num], batchnorm_alg=batchnorm)
                new_layer.prev_layer = Network.all_layer[-1]
                Network.all_layer[-1].next_layer = new_layer

            else:
                # Create output layer
                new_layer = Network(nodes_per_layer[num-1], nodes_per_layer[num], batchnorm_alg=batchnorm, output_layer=True)
                new_layer.prev_layer = Network.all_layer[-1]
                Network.all_layer[-1].next_layer = new_layer
                set_batchnorm_values = True
            if batchnorm and not set_batchnorm_values:
                pass
                # new_layer.data_mean = mean_std_batchnorm[0]
                # new_layer.data_std = mean_std_batchnorm[1]
            Network.all_layer.append(new_layer)

    #Softmax operation
    def softmax(self,array):


        pred = np.exp(array)
        return pred / np.sum(pred, axis=0)

    #Cross entorpy (loss value)
    def cross_entropy(self, probs, label):

        return  -np.sum(label * np.log(probs),axis=0)

    def cross_entropy_test(self, probs, label):

        return -np.sum(label * np.log(probs),axis=0)

    def get_input_layers_nodes_number(self):
        return self.nodes_per_layer[0]



    #Relu forward
    def relu(self, array):

        array[array <= 0] = 0

        return array

    #Relu back
    def relu_bck(self, array, layer):

        array[layer <= 0] = 0

        return array

    #Sigmoid. Not in use.
    def sgm(self, value, der=False):

        if not der:
            return 1/(1+np.exp(-value))
        else:
            value_mat = np.asarray(value)
            return value_mat * (1 - value_mat)

    def batchnorm_forward(self, x, gamma, beta, eps=1e-8):

        N, D = x.shape
        # step1: calculate mean
        mu = 1. / N * np.sum(x, axis=0)

        print (mu)
        exit(0)

        # step2: subtract mean vector of every trainings example
        xmu = x - mu

        # step3: following the lower branch - calculation denominator
        sq = xmu ** 2

        # step4: calculate variance
        var = 1. / N * np.sum(sq, axis=0)

        # step5: add eps for numerical stability, then sqrt
        sqrtvar = np.sqrt(var + eps)

        # step6: invert sqrtwar
        ivar = 1. / sqrtvar

        # step7: execute normalization
        xhat = xmu * ivar

        # step8: Nor the two transformation steps
        gammax = gamma * xhat

        # step9
        out = gammax + beta

        # store intermediate
        cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

        return out, cache

    #Forward action
    def forward_propagation(self, expected, train = True, dropout= False, dropout_prob=0.7, batchnorm=False):
        #Get all layers
        output_layer_index = len(Network.all_layer)-1
        dropout_layer = []

        for index, layer in enumerate(Network.all_layer):
            if index < output_layer_index:

                if index is 0:
                #Input layer only
                    #Multiply weights by the input values
                    results =layer.weights_matrix.copy().dot(layer.input_or_pre_activation_matrix.copy())



                else:
                    #Hidden layers. Multiply weights by the activation matrix (RELU output)
                    results = layer.weights_matrix.copy().dot(layer.layer_activation_matrix.copy())



                #layer.next_layer.matrix_computed_weight_act_bias = results
                #Add bias and save in the preactivation matrix in the next layer
                layer.next_layer.input_or_pre_activation_matrix = results.copy() + layer.bias.copy()

                if batchnorm:
                    self.batchnorm_forward(layer.next_layer.input_or_pre_activation_matrix, layer.data_mean, layer.data_std)
                    exit (441)
                if dropout:
                    # Create ranndom vector (0,1) with activation shape
                    prob_vector_random = np.random.random(layer.next_layer.input_or_pre_activation_matrix.shape)
                    # If prob below dropout_prob set to 1, else 0.

                    prob_vector_probable = prob_vector_random < dropout_prob
                    #Save probable matrix and probable value
                    layer.next_layer.layer_dropout = prob_vector_probable
                    layer.next_layer.layer_dropout_prob = dropout_prob



                if layer.next_layer.is_output_layer:
                    #If the next layer is output layer, save the results in the activation matrix. Else, apply RELU

                    if dropout:
                        # Pairwise multipication to randomly drop nodes
                        layer.next_layer.input_or_pre_activation_matrix = np.multiply(
                            layer.next_layer.input_or_pre_activation_matrix.copy(), prob_vector_probable)
                        # Rescale the activation array by the number of the prob
                        layer.next_layer.input_or_pre_activation_matrix /= dropout_prob

                    layer.next_layer.layer_activation_matrix = layer.next_layer.input_or_pre_activation_matrix.copy()

                else:
                    #Apply RELU for the preactivation matrix
                    layer.next_layer.layer_activation_matrix = self.relu(
                        layer.next_layer.input_or_pre_activation_matrix.copy())

                    if dropout:
                        layer.next_layer.layer_activation_matrix = np.multiply(
                            layer.next_layer.layer_activation_matrix.copy(), prob_vector_probable)

                        layer.next_layer.layer_activation_matrix /= dropout_prob


            else:
                #Perform softmax for the activation matrix
                layer.softmax = self.softmax(layer.layer_activation_matrix.copy())




                #Get the loss value
                self.loss = self.cross_entropy(layer.softmax.copy(), expected)

                if train:
                    #During training phase, return the loss, else (test) return the predection

                    return self.loss

                else:
                    return np.argmax(layer.softmax), np.max(layer.softmax)


    def forward_propagation_test(self, expected, train = True, dropout= False, dropout_prob=0.7, batchnorm=False):
        #Get all layers
        output_layer_index = len(Network.all_layer)-1
        dropout_layer = []

        for index, layer in enumerate(Network.all_layer):
            if index < output_layer_index:

                if index is 0:
                #Input layer only
                    #Multiply weights by the input values


                    results =layer.weights_matrix.dot(layer.input_or_pre_activation_matrix)



                else:
                    #Hidden layers. Multiply weights by the activation matrix (RELU output)
                    results = layer.weights_matrix.copy().dot(layer.layer_activation_matrix.copy())



                #layer.next_layer.matrix_computed_weight_act_bias = results
                #Add bias and save in the preactivation matrix in the next layer
                layer.next_layer.input_or_pre_activation_matrix = results.copy() + layer.bias.copy()

                if batchnorm:
                    self.batchnorm_forward(layer.next_layer.input_or_pre_activation_matrix, layer.data_mean, layer.data_std)

                if dropout:
                    # Create ranndom vector (0,1) with activation shape
                    prob_vector_random = np.random.random(layer.next_layer.input_or_pre_activation_matrix.shape)
                    # If prob below dropout_prob set to 1, else 0.

                    prob_vector_probable = prob_vector_random < dropout_prob
                    #Save probable matrix and probable value
                    layer.next_layer.layer_dropout = prob_vector_probable
                    layer.next_layer.layer_dropout_prob = dropout_prob



                if layer.next_layer.is_output_layer:
                    #If the next layer is output layer, save the results in the activation matrix. Else, apply RELU

                    if dropout:
                        # Pairwise multipication to randomly drop nodes
                        layer.next_layer.input_or_pre_activation_matrix = np.multiply(
                            layer.next_layer.input_or_pre_activation_matrix.copy(), prob_vector_probable)
                        # Rescale the activation array by the number of the prob
                        layer.next_layer.input_or_pre_activation_matrix /= dropout_prob

                    layer.next_layer.layer_activation_matrix = layer.next_layer.input_or_pre_activation_matrix.copy()

                else:
                    #Apply RELU for the preactivation matrix
                    layer.next_layer.layer_activation_matrix = self.relu(
                        layer.next_layer.input_or_pre_activation_matrix.copy())

                    if dropout:
                        layer.next_layer.layer_activation_matrix = np.multiply(
                            layer.next_layer.layer_activation_matrix.copy(), prob_vector_probable)

                        layer.next_layer.layer_activation_matrix /= dropout_prob


            else:
                #Perform softmax for the activation matrix


                layer.softmax = self.softmax(layer.layer_activation_matrix.copy())



                #Get the loss value



                if train:
                    self.loss = self.cross_entropy_test(layer.softmax.copy(), expected)
                    #During training phase, return the loss, else (test) return the predection

                    return self.loss

                else:

                    print (layer.softmax.shape)

                    return np.argmax(layer.softmax, axis=0), np.max(layer.softmax,axis=0)




    def forward_propagation_test_A(self, expected, train = True, dropout= False, dropout_prob=0.7, batchnorm=False):
        #Get all layers
        output_layer_index = len(Network.all_layer)-1



        for index, layer in enumerate(Network.all_layer[1:],start=1):

            if index < output_layer_index:

                if layer.prev_layer.input_layer:

                    #Input layer only
                    #Multiply weights by the input values

                    results = layer.weights_matrix.dot(layer.prev_layer.input_or_pre_activation_matrix)

                else:
                    #Hidden layers. Multiply weights by the activation matrix (RELU output)
                    results = layer.weights_matrix.copy().dot(layer.prev_layer.layer_activation_matrix.copy())

                #layer.next_layer.matrix_computed_weight_act_bias = results
                #Add bias and save in the preactivation matrix in the next layer
                layer.input_or_pre_activation_matrix = results.copy() + layer.bias.copy()

                if batchnorm:
                    self.batchnorm_forward(layer.input_or_pre_activation_matrix, layer.data_mean, layer.data_std)

                if dropout:
                    # Create ranndom vector (0,1) with activation shape
                    prob_vector_random = np.random.random(layer.input_or_pre_activation_matrix.shape)
                    # If prob below dropout_prob set to 1, else 0.

                    prob_vector_probable = prob_vector_random < dropout_prob
                    #Save probable matrix and probable value
                    layer.layer_dropout = prob_vector_probable
                    layer.layer_dropout_prob = dropout_prob


                    # Pairwise multipication to randomly drop nodes
                    layer.input_or_pre_activation_matrix = np.multiply(
                        layer.input_or_pre_activation_matrix.copy(), prob_vector_probable)
                    # Rescale the activation array by the number of the prob
                    layer.input_or_pre_activation_matrix /= dropout_prob

                layer.layer_activation_matrix = layer.input_or_pre_activation_matrix.copy()

                #Apply RELU for the preactivation matrix
                layer.layer_activation_matrix = self.relu(layer.input_or_pre_activation_matrix.copy())



                if dropout:
                    layer.layer_activation_matrix = np.multiply(
                        layer.layer_activation_matrix.copy(), prob_vector_probable)

                    layer.layer_activation_matrix /= dropout_prob


            else:

                results = layer.weights_matrix.copy().dot(layer.prev_layer.layer_activation_matrix.copy())
                layer.input_or_pre_activation_matrix = results.copy() + layer.bias.copy()
                layer.layer_activation_matrix = layer.input_or_pre_activation_matrix.copy()



                #Perform softmax for the activation matrix


                layer.softmax = self.softmax(layer.layer_activation_matrix.copy())


                #Get the loss value

                if train:
                    self.loss = self.cross_entropy_test(layer.softmax.copy(), expected)
                    #During training phase, return the loss, else (test) return the predection
                    return self.loss

                else:
                    return np.argmax(layer.softmax, axis=0), np.max(layer.softmax, axis=0)



    def back_propagation(self, expected, reshape, dropout=False):
        #Reverse loop.
        for index, layer in (enumerate(reversed(Network.all_layer))):

            if layer.is_output_layer:
                # Output layer. Calculate the derivative of the loss value
                loss = layer.softmax.copy() - expected  # derivative of loss w.r.t. final dense layer output
                #Save in the layer ouput value. For loss propagation
                layer.output = loss
                if dropout:
                    layer.output = np.multiply(layer.output.copy(), layer.layer_dropout)
                    layer.output/= layer.layer_dropout_prob

            else:

                if not layer.input_layer:
                    #Hidden Layers...

                    #Get the partial derivative for the weights by multiply the next layer output with the activation matrix
                    layer.dweight = layer.next_layer.output.copy().dot(layer.layer_activation_matrix.T.copy())




                    #Sum the next layer output and save it as the bias values
                    layer.dbias = np.sum(layer.next_layer.output.copy(), axis=1).reshape(layer.bias.shape)

                    # Get the partial derivative for the output by multiply the layer weights with the next layer output (Sum loss)

                    d_output = layer.weights_matrix.copy().T.dot(layer.next_layer.output.copy())

                    #Perform relu for the loss and save in the layer output value
                    layer.output = self.relu_bck(d_output.copy(), layer.input_or_pre_activation_matrix.copy())

                    if dropout:
                        layer.output = np.multiply(layer.output.copy(), layer.layer_dropout)
                        layer.output/= layer.layer_dropout_prob

                else:
                    #print(layer.prev_layer.layer_dropout)
                    #Input layer
                    #Get the partial derivative for the weights by multiply the next layer output with the activation matrix

                    layer.dweight = layer.next_layer.output.copy().dot(layer.input_or_pre_activation_matrix.copy().T)
                    #Sum the next layer output and save it as the bias values


                    layer.dbias = np.sum(layer.next_layer.output.copy(), axis=1).reshape(layer.bias.shape)

                    # Get the partial derivative for the output by multiply the layer weights with the next layer output (Sum loss)
                    doutput = layer.weights_matrix.T.copy().dot(layer.next_layer.output.copy())

                    #Return the reshape of output (loss) for the max pool backward filter



                    return doutput.reshape(reshape)



    def back_propagation_test(self, expected, reshape, dropout=False):
        # Reverse loop.

        for index, layer in (enumerate(reversed(Network.all_layer))):

            if layer.is_output_layer:
                # Output layer. Calculate the derivative of the loss value

                loss = layer.softmax.copy() - expected  # derivative of loss w.r.t. final dense layer output
                # Save in the layer ouput value. For loss propagation
                layer.output = loss

                if dropout:
                    layer.output = np.multiply(layer.output.copy(), layer.layer_dropout)
                    layer.output /= layer.layer_dropout_prob

            else:
                if not layer.input_layer:
                    # Hidden Layers...
                    # Get the partial derivative for the weights by multiply the next layer output with the activation matrix
                    layer.dweight = layer.next_layer.output.copy().dot(layer.layer_activation_matrix.copy().T)

                    # Sum the next layer output and save it as the bias values
                    layer.dbias = np.sum(layer.next_layer.output.copy(), axis=1).reshape(layer.bias.shape)

                    # Get the partial derivative for the output by multiply the layer weights with the next layer output (Sum loss)

                    d_output = layer.weights_matrix.copy().T.dot(layer.next_layer.output.copy())


                    # Perform relu for the loss and save in the layer output value
                    layer.output = self.relu_bck(d_output.copy(), layer.input_or_pre_activation_matrix.copy())

                    if dropout:
                        layer.output = np.multiply(layer.output.copy(), layer.layer_dropout)
                        layer.output /= layer.layer_dropout_prob

                else:

                    # Get the partial derivative for the weights by multiply the next layer output with the activation matrix

                    layer.dweight = layer.next_layer.output.copy().dot(layer.input_or_pre_activation_matrix.copy().T)

                    # Sum the next layer output and save it as the bias values

                    layer.dbias = np.sum(layer.next_layer.output.copy(), axis=1).reshape(layer.bias.shape)
                    # Get the partial derivative for the output by multiply the layer weights with the next layer output (Sum loss)
                    doutput = layer.weights_matrix.T.copy().dot(layer.next_layer.output.copy()).T


                    return doutput.reshape(reshape)


    def back_propagation_test_A(self, expected, reshape, dropout=False):
        # Reverse loop.

        for index, layer in (enumerate(reversed(Network.all_layer))):

            if layer.is_output_layer:
                # Output layer. Calculate the derivative of the loss value

                loss = layer.softmax.copy() - expected  # derivative of loss w.r.t. final dense layer output
                # Save in the layer ouput value. For loss propagation


                if dropout:
                    loss = np.multiply(loss, layer.layer_dropout)
                    loss /= layer.layer_dropout_prob

                layer.dweight = loss.dot(layer.prev_layer.layer_activation_matrix.copy().T)


                layer.dbias = np.sum(loss, axis=1).reshape(layer.bias.shape)

                layer.output = layer.weights_matrix.copy().T.dot(loss)


            else:
                if not layer.input_layer:
                    # Hidden Layers...
                    # Get the partial derivative for the weights by multiply the next layer output with the activation matrix


                    output = self.relu_bck(layer.next_layer.output.copy(), layer.layer_activation_matrix.copy())

                    if dropout:
                        output = np.multiply(output, layer.layer_dropout)
                        output /= layer.layer_dropout_prob


                    if layer.prev_layer.input_layer:

                        layer.dweight = output.dot(
                            layer.prev_layer.input_or_pre_activation_matrix.copy().T)

                    else:
                        layer.dweight = output.dot(layer.prev_layer.layer_activation_matrix.copy().T)


                    # Sum the next layer output and save it as the bias values
                    layer.dbias = np.sum(output, axis=1).reshape(layer.bias.shape)

                    # Get the partial derivative for the output by multiply the layer weights with the next layer output (Sum loss)

                    layer.output = layer.weights_matrix.copy().T.dot(output)

                else:

                    return layer.next_layer.output.T.reshape(reshape)

    def setNetwork_input_data(self, input_data):
        #Set an input for the first layer (input layer) in the FCN
        Network.all_layer[0].input_or_pre_activation_matrix = input_data


    def saveWeightsBias(self):
        #Save all of the weights. Not used. Debug only
        for x, layer in enumerate(Network.all_layer[:-1]):
            weight = 'layer_' + str(x) + str(datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
            bias = 'bias_' + str(x) + str (datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
            error = 'error'+ str (datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
            np.save("SavedWeightsandBias/"+weight, layer.weights_matrix)
            np.save("SavedWeightsandBias/"+bias, layer.bias)
            np.save("SavedWeightsandBias/" + error, self.error)

    def returnWeightBiasShapes(self):
        #Return the reshape value for the Back propagation
        return Network.shapes

    #Get layer object
    def getlayer(self, layer):
        return Network.all_layer[layer]
    #Get Partial derivative weight from a layer object
    def getdweight(self, layer):

        return Network.all_layer[layer].dweight.copy()

    # Get weight from a layer object
    def getweight(self, layer):

        return Network.all_layer[layer].weights_matrix.copy()

    #Set Partial derivative weight in a layer object
    def set_der_as_weight(self, layer, weight):

        Network.all_layer[layer].weights_matrix = weight

    # Get Partial derivative bias from a layer object
    def getdbias(self, layer):

        return Network.all_layer[layer].dbias.copy()

    # Get bias from a layer object
    def getbias(self, layer):

        return Network.all_layer[layer].bias.copy()

    # Set Partial derivative bias in a layer object
    def set_der_as_bias(self, layer, bias):

        Network.all_layer[layer].bias = bias

    #DEBUG
    def debug(self):
        print (Network.all_layer)
        for layer in Network.all_layer[1:]:
            print('-------')
            print (layer.layer_activation_matrix)

            print('-------')

    # DEBUG
    def print_network(self):

        for layer in Network.all_layer:
            if layer is not Network.all_layer[-1]:
                print('----------------------------')
                print("Layer Weights: {}\n Layer Pre Activations {}\n Layer Activations ".
                      format(layer.weights_matrix, layer.input_or_pre_activation_matrix))
                print('----------------------------')
            for node in layer.nodes_objects:
                if layer is Network.all_layer[-1]:
                    print("layer - Node Location  {}-{}\n  Input {}\n Act {}\n Loss {}".format(
                        node.layer, node.location_in_layer, node.input_before_activation,
                        node.input_activation, node.loss))
                    print('----------------------------')

                elif layer is Network.all_layer[0]:
                    print("layer - Node Location  {}-{}\n W {}\n Input {}\n Bias {}".format(
                        node.layer, node.location_in_layer, node.weights, node.input_value, node.bias))
                    print('----------------------------')

                else:
                    print("layer - Node Location  {}-{}\n W {}\n Input {}\n Act {}\n Bias {}".format(
                        node.layer, node.location_in_layer, node.weights, node.input_before_activation,
                        node.input_activation, node.bias))
                    print('----------------------------')
