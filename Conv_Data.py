import numpy as np

class Conv_Data:
    #Object values for each conv_data
    def __init__(self, size_filter):
        self.filter = None
        self.bias = None
        self.output_conv = None
        self.relu_forward = None
        self.max_pooling = None
        self.max_pooling_index = None
        self.max_pool_backprop = None
        self.relu_backprop = None
        self.d_out_matrix = None
        self.d_out_filter =None
        self.d_out_bias = None
        self.initializeFilter(size_filter)

    #Create a random (according to a normal value) values for the filter and set an empty vector for the bias values
    def initializeFilter(self, size, scale=1.0):

        stddev = scale / np.sqrt(np.prod(size))
        self.filter = np.random.normal(loc=0, scale=stddev, size=size)
        self.bias = np.zeros((size[0], 1))

    #Set a new filter
    def setfilter(self, filt):
        self.filter = filt

    #Get a filter
    def getfilter(self):
        return self.filter.copy()

    # Set a new bias
    def setbias(self, bias):
        self.bias = bias

    # Set a new bias
    def getbias(self):
        return self.bias.copy()
