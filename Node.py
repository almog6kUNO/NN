#Currenty not used. DEBUG
class Node:

    def __init__(self, weights, bias, layer, location_in_layer, input_before_activation=None, output_layer=False):

        if output_layer:
            self.loss = None
            self.input_before_activation = input_before_activation
            self.input_activation = None

        elif input_before_activation is not None:
            self.weights = weights
            self.bias = bias
            self.input_value = input_before_activation
        else:
            self.weights = weights
            self.input_activation = None
            self.bias = bias
            self.input_before_activation = input_before_activation

        self.layer = layer
        self.location_in_layer = location_in_layer
        self.output_layer = output_layer
