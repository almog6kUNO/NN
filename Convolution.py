import numpy as np

class Convolution:


    #Save max pooling backwards matrix
    def __init__(self):
        pass

    #Filter rotation
    def weights_forward(self, image_array, conv_data, padding =0, stride=1, sec = False):


        #Get dim values for output matrix and for loop
        dimone, dimtwo, size, _ = conv_data.filter.shape


        #Get size of output matrix
        array_size = self.empty_array_size(image_array.shape[1], size, padding, stride)
        #Create an output array with the needed size
        output_array = np.zeros((int(dimone), array_size, array_size))
        x, y = 0, 0
        # if image_array.ndim < 3:
        #     image_array = np.reshape(image_array, (1, image_array.shape[0], image_array.shape[0]))

        #Move throught all filter dimensions
        for dimension in range(0, dimone):
            y=0
            #Move through the image on the y axis
            for y_movement in range(0, array_size, stride):
                #Move through the image on the x axis
                for x_movement in range(0, array_size, stride):
                    #Get subset of the image according the the filter size and stride
                    subset_image_array = image_array[:, y_movement:y_movement +
                                                    size, x_movement:x_movement + size]




                    #Apply filter multipication on the subset of the input image and save the sum in the output array
                    output_array[dimension,y,x] = np.sum(conv_data.filter[dimension] * subset_image_array) + conv_data.bias[dimension]


                    x += 1
                x = 0
                y += 1
        #Save output in the object output_conv value

        conv_data.output_conv = output_array
        return output_array

    def weights_forward_test(self, image_array, conv_data, padding=0, stride=1, sec = False):

        # Get dim values for output matrix and for loop
        dimone, dimtwo, size, _ = conv_data.filter.shape
        # Get size of output matrix
        array_size = self.empty_array_size(image_array.shape[3], size, padding, stride)

        # Create an output array with the needed size
        output_array = np.zeros((image_array.shape[0],int(dimone), array_size, array_size))
        x, y = 0, 0


        # Move throught all filter dimensions

        for y_movement in range(0, array_size, stride):
            # Move through the image on the x axis
            for x_movement in range(0, array_size, stride):
                # Get subset of the image according the the filter size and stride
                subset_image_array = image_array[:, :, y_movement:y_movement +
                                                               size, x_movement:x_movement + size]

                filter_shape = conv_data.filter.shape
                sub_shape = subset_image_array.shape




                filter_reshaped = np.reshape(conv_data.filter.copy(), (1,filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]))

                subset_reshaped = np.reshape(subset_image_array.copy() ,(sub_shape[0],1,sub_shape[1], sub_shape[2],sub_shape[3]))

                all_dim =  np.sum((filter_reshaped * subset_reshaped), axis=(3,4)) + conv_data.bias

                if all_dim.shape[2] > 1:

                    all_dim= (np.sum(all_dim.copy(), axis=2))

                all_dim = np.reshape(all_dim.copy(),(all_dim.shape[0],all_dim.shape[1]))
                output_array[:, :, y, x] = all_dim

                x += 1
            x = 0
            y+=1
        # Save output in the object output_conv value
        conv_data.output_conv = output_array

        return output_array


    #Filter rotation (Backwards)
    def weights_backwards(self, loss, input_data, conv_data, stride=1, two_itr=False):

        #Create empty matrices for the partial derivative for output, filter and bias
        d_out_matrix = np.zeros(input_data.shape)
        d_out_filter = np.zeros(conv_data.filter.shape)
        d_out_bias = np.zeros((conv_data.bias.shape[0], 1))
        #Copy filter values
        conv_filter = conv_data.filter.copy()
        #Get filter size (for loop movement)
        filter_size = conv_filter.shape[2]


        x, y = 0, 0
        #Run through all image dimensions
        for dim in range(conv_filter.shape[0]):
            y = 0
            # Move filter throught filter and inputdata on the y axis
            for y_movement in range(0, loss.shape[2], stride):
                # Move filter throught filter and inputdata on the x axis
                for x_movement in range(0, loss.shape[2], stride):
                    #Calculate the partial derivative of the image from the loss( previous total error) and the filter
                    d_out_matrix[:, y_movement:y_movement + filter_size, x_movement:x_movement + filter_size] += loss[dim, y, x] * conv_filter[dim]

                    # Calculate the partial derivative of the filter from the loss( previous total error) and the original input image
                    d_out_filter[dim] += loss[dim, y, x] * input_data[:, y_movement:y_movement + filter_size, x_movement:x_movement + filter_size]

                    x += 1



                y += 1
                x = 0
            #Sum all of the loss error to update the bias value
            d_out_bias[dim] = np.sum(loss[dim])

        #Save all matrices  on the object value
        conv_data.d_out_matrix = d_out_matrix
        conv_data.d_out_filter = d_out_filter
        conv_data.d_out_bias = d_out_bias
        return d_out_matrix, d_out_filter , d_out_bias


    def weights_backwards_test(self, loss, input_data, conv_data, stride=1, two_itr=False):

        #Create empty matrices for the partial derivative for output, filter and bias
        d_out_matrix = np.zeros(input_data.shape)

        filter_size = conv_data.filter.shape


        d_out_filter = np.zeros(conv_data.filter.shape)

        d_out_bias = np.zeros((conv_data.bias.shape[0], 1))
        #Copy filter values

        #Get filter size (for loop movement)

        x, y = 0, 0
        #Run through all image dimensions

            # Move filter throught filter and inputdata on the y axis
        for y_movement in range(0, loss.shape[2], stride):
            # Move filter throught filter and inputdata on the x axis
            for x_movement in range(0, loss.shape[2], stride):
                #Calculate the partial derivative of the image from the loss( previous total error) and the filter


                loss_all_dim = loss[:,:,y, x]
                loss_all_dim_reshape = np.reshape(loss_all_dim, (loss_all_dim.shape[0],loss_all_dim.shape[1],1,1,1))
                conv_filter_reshape = np.reshape(conv_data.filter, (1,filter_size[0],filter_size[1],filter_size[2],filter_size[3]))
                d_out_matrix[:,:, y_movement:y_movement + filter_size[2], x_movement:x_movement + filter_size[2]]+= np.sum(loss_all_dim_reshape*conv_filter_reshape, axis=1)


                input_data_sub = input_data[:,:, y_movement:y_movement + filter_size[2], x_movement:x_movement + filter_size[2]]


                input_data_reshape = np.reshape(input_data_sub, (input_data_sub.shape[0], 1, input_data_sub.shape[1], input_data_sub.shape[2], input_data_sub.shape[3]))


                d_alldim_filter_out = loss_all_dim_reshape *input_data_reshape




                d_out_filter +=np.sum(d_alldim_filter_out, axis=0)


                x += 1

            y += 1
            x = 0

            #Sum all of the loss error to update the bias value

            d_out_bias = np.reshape(np.sum(loss, axis=(0, 2, 3)), (conv_data.bias.shape[0], 1))


        #Save all matrices  on the object value
        conv_data.d_out_matrix = d_out_matrix
        conv_data.d_out_filter = d_out_filter
        conv_data.d_out_bias = d_out_bias

        return d_out_matrix, d_out_filter , d_out_bias


    #Get max values from a matrix
    def max_pooling_forward(self, image_array, conv_data, max_pooling_size=2):

        # if (image_array.shape[2]%2):
        #
        #     add_zero_one = np.zeros((image_array.shape[0],image_array.shape[1], 1))
        #     add_zero_two = np.zeros((image_array.shape[0], 1, image_array.shape[1]+1))
        #     image_array = np.concatenate((image_array.copy(), add_zero_one), axis=2)
        #     image_array = np.concatenate((image_array.copy(), add_zero_two), axis=1)
        #Get size for output image
        array_size = self.empty_array_size(image_array.shape[2], max_pooling_size, 0, max_pooling_size)

        #Create an empty matrix for the output
        max_pool_array = np.zeros((image_array.shape[0], int(array_size), int(array_size)))



        #Create an empty matrix for a map that shows where the highest value was found (set as 1). To be used in Max pool backwards
        max_pool_index = np.zeros(image_array.shape)

        x, y = 0, 0
        #Run through all dimensions
        for dim in range(image_array.shape[0]):
            y = 0
            # Run window on the y axis
            for y_movement in range(0, image_array.shape[2], max_pooling_size):
                # Run window on the x axis
                for x_movement in range(0, image_array.shape[2], max_pooling_size):
                    #Get a matrix subset
                    subset_image_array = (image_array[dim, y_movement:y_movement +
                                                                      max_pooling_size, x_movement:x_movement
                                                                                                   + max_pooling_size])
                    #Get highest value on the vertical axis
                    max_v = (np.amax(subset_image_array, axis=1))
                    # Get highest value on the horizontal axis
                    max_h = (np.amax(max_v, axis=0))

                    #Find the index of where the highest number is located in the subset
                    ind = np.unravel_index(np.nanargmax(subset_image_array, axis=None), subset_image_array.shape)
                    #Save the highest value in the output matrix
                    max_pool_array[dim, y, x] = max_h
                    #Change 0 to 1 in the location where the highest value was found
                    max_pool_index[dim, y_movement + ind[0], x_movement+ind[1]] = 1

                    x += 1
                x = 0
                y += 1
        #Save output and index_map in the object values
        conv_data.max_pooling_index = max_pool_index
        conv_data.max_pooling = max_pool_array
        return max_pool_index, max_pool_array, max_pool_array.shape




    def max_pooling_forward_test(self, image_array, conv_data, max_pooling_size=2):



        array_size = self.empty_array_size(image_array.shape[2], max_pooling_size, 0, max_pooling_size)

        #Create an empty matrix for the output
        max_pool_array = np.zeros((image_array.shape[0],image_array.shape[1], int(array_size), int(array_size)))
        #Create an empty matrix for a map that shows where the highest value was found (set as 1). To be used in Max pool backwards
        max_pool_index = np.zeros(image_array.shape)



        index = 0
        x, y = 0, 0
        #Run through all dimensions

            # Run window on the y axis
        for y_movement in range(0, image_array.shape[2], max_pooling_size):
            # Run window on the x axis
            for x_movement in range(0, image_array.shape[2], max_pooling_size):

                #Get a matrix subset
                subset_image_array = (image_array[:, :, y_movement:y_movement + max_pooling_size,
                                      x_movement:x_movement + max_pooling_size])

                subset_image_reshape = np.reshape(subset_image_array, (subset_image_array.shape[0], subset_image_array.shape[1], subset_image_array.shape[2]**2))
                max_value = np.amax(subset_image_reshape, axis=2)


                max_pool_array[:, :, y, x] = max_value



                index_loc = np.argmax(subset_image_reshape, axis=2)



                for image in range(subset_image_array.shape[0]):
                    for dim in range(subset_image_array.shape[1]):
                        index_location =  np.unravel_index(subset_image_array[image,dim].argmax(), subset_image_array[image,dim].shape)
                        max_pool_index[image, dim, y_movement + index_location[0], x_movement + index_location[1]] = 1



                x += 1
            x = 0
            y += 1
        #Save output and index_map in the object values
        conv_data.max_pooling_index = max_pool_index
        conv_data.max_pooling = max_pool_array
        return max_pool_index, max_pool_array, max_pool_array.shape



    #Back propagation for the max pooling filter
    def max_pooling_backward(self, fc_back, conv_data, stride=2):
        #Set an emtpy matrix with the shape from the input size of the max pool forward operation
        max_pool_array = np.zeros(conv_data.max_pooling_index.shape)


        x, y = 0, 0
        # Run through all dimensions
        for dim in range(0, max_pool_array.shape[0]):
            y = 0
            #Move window in the y axis. Number increment by stide
            for y_movement in range(0, max_pool_array.shape[1], stride):
                # Move window in the x axis. Number increment by stide
                for x_movement in range(0, max_pool_array.shape[1], stride):

                    #Run through the index matrix
                    subset_image_array = (conv_data.max_pooling_index[dim, y_movement:y_movement + stride,
                                          x_movement:x_movement + stride])
                    #Find location of 1 in the array and return the y and x values
                    y_loc, x_loc = np.where(subset_image_array == 1)
                    #Store the value of the highest value from the previous array
                    max_pool_array[dim, y_movement + y_loc, x_movement+x_loc] = fc_back[dim, y, x]

                    x += 1
                y += 1
                x = 0

        conv_data.max_pool_backprop = max_pool_array
        return max_pool_array



    def max_pooling_backward_test(self, fc_back, conv_data, stride=2):
        #Set an emtpy matrix with the shape from the input size of the max pool forward operation

        max_pool_array = np.zeros(conv_data.max_pooling_index.shape)

        x, y = 0, 0
        # # Run through all dimensions

        #Move window in the y axis. Number increment by stide
        for y_movement in range(0, max_pool_array.shape[2], stride):
            # Move window in the x axis. Number increment by stide
            for x_movement in range(0, max_pool_array.shape[2], stride):

                #Run through the index matrix
                subset_image_array = (conv_data.max_pooling_index[:, :, y_movement:y_movement + stride,
                                      x_movement:x_movement + stride])


                duplicate_matrix=  (np.repeat(fc_back[:, :, y, x], stride**2))

                duplicate_matrix_reshape = np.reshape(duplicate_matrix, (subset_image_array.shape))

                backward =  (np.multiply(duplicate_matrix_reshape, subset_image_array))

                max_pool_array[:,:,y_movement:y_movement+stride,x_movement:x_movement+stride]= backward


                x += 1
            y += 1
            x = 0


        conv_data.max_pool_backprop = max_pool_array

        return max_pool_array













    ''' Different application of filter back propagation. More testing is required
    def weights_input_backpropagation(self, input_next_layer, input_image, stride):

        newfilter_array = np.zeros(input_next_layer[1].shape)
        newinput_array = np.zeros(input_image.shape)

        x, y = 0, 0
        for y_movement in range(0, input_next_layer[1].shape[0], stride):
            for x_movement in range(0, input_next_layer[1].shape[0], stride):

                subset_image_array = (input_image[x_movement:x_movement+input_next_layer[0].shape[0],
                                      y_movement:y_movement + input_next_layer[0].shape[0]].T)

                filter_result = (input_next_layer[0].T * subset_image_array)
                newfilter_array[x][y] = (np.sum(np.sum(filter_result, axis=1), axis=1))
                x += 1
            x = 0
            y += 1

        x, y = 0, 0
        transpose_filter = (input_next_layer[1].T)

        for y_movement in reversed(range(0, newinput_array.shape[1], stride)):

            for x_movement in reversed(range(0, newinput_array.shape[1], stride)):

                sum_inputA = transpose_filter[abs(x - (transpose_filter.shape[1]-1)) if (x -
                (transpose_filter.shape[1]-1)) <0 else 0:abs(x - newinput_array.shape[1])
                if abs(x - newinput_array.shape[1]) < transpose_filter.shape[1] else transpose_filter.shape[1],
                abs(y - (transpose_filter.shape[1]-1)) if (y - (transpose_filter.shape[1]-1)) < 0
                else 0 :abs(y - newinput_array.shape[1]) if abs(y - newinput_array.shape[1])
                < transpose_filter.shape[1] else transpose_filter.shape[1]].T

                sum_inputB = input_next_layer[0][abs(x_movement - (transpose_filter.shape[1]-1)) if (x_movement -
                (transpose_filter.shape[1]-1)) <0 else 0:abs(x_movement - newinput_array.shape[1])
                if abs(x_movement - newinput_array.shape[1]) < transpose_filter.shape[1] else transpose_filter.shape[1],
                abs(y_movement - (transpose_filter.shape[1]-1)) if (y_movement - (transpose_filter.shape[1]-1)) < 0
                else 0 :abs(y_movement - newinput_array.shape[1]) if abs(y_movement - newinput_array.shape[1])
                < transpose_filter.shape[1] else transpose_filter.shape[1]].T


                newinput_array[y][x] = np.sum(sum_inputA*sum_inputB)

                x += 1
            x = 0
            y += 1

    '''
    #Return size for the output array.
    def empty_array_size(self, output_size, size_of_filter, padding, stride):

        return int((output_size-size_of_filter+2*padding)/stride)+1

    #Perform Relu operation and store in the relu_forward value
    def relu(self, array, conv_data):
        array[array <= 0] = 0
        conv_data.relu_forward = array

    def relu_test(self, array, conv_data):


        array[array <= 0] = 0
        conv_data.relu_forward = array
        return array



    # Perform Relu operation and store in the relu_backward value
    def relu_backward(self, array, conv_data):

        array[conv_data.output_conv <= 0] = 0

        conv_data.relu_backprop = array

        return conv_data.relu_backprop

    def relu_backward_test(self, array, conv_data):



        array[conv_data.output_conv <= 0] = 0
        conv_data.relu_backprop = array
        return conv_data.relu_backprop






