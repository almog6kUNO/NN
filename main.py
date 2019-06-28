from Conv_Data import *
from NN import *
from Data import *
from Convolution import *
from tqdm import tqdm
import argparse
import time

#Train function. 64 Images per iterations. Epochs 2 (2 repeat iterations)
def train_CNN(savedname, num_epochs = 2+1, batch_size = 64):

    #50,000 training images
    num_images = 50000
    image_size = 28
    #Create a data object
    data_array = Data()
    #Load training and labels into object.
    data_array.readByte('train-images-idx3-ubyte.gz', image_size, num_images)
    data_array.readLables('train-labels-idx1-ubyte.gz', num_images, dataaugment=True)

    #return a normalized, shuffled training dataset
    training_data, mean_and_std = data_array.combine_and_shuffle(dataaugment=True)


    num_images_aug = len(training_data)
    filter_one_size = (8, 1, 3, 3)
    filter_two_size = (8, 8, 3, 3)
    fcn_layers = [1152, 128, 10]

    #Create first conv layer with a set size of 8 filters with a 5x5 size
    filter_one = Conv_Data(filter_one_size)
    # Create second conv layer with a set size of 8 dimensions with 8 filters with a 5x5 size
    filter_two = Conv_Data(filter_two_size)

    #Set max_pool_reshape size and generate filter and bias
    #pool_reshape_back = (batch_size, 8, 10, 10)

    #Create a convolution object and generate filter and bias
    layer = Convolution()
    #Create a fully connected network. First layer 800 nodes, second layer 128, and third 10
    nodes_per_layers = fcn_layers

    #Initialize fully connected
    new_NN = NN(len(nodes_per_layers), nodes_per_layers, training_data[0][0], batchnorm=False)
    #Set return maxpool shape in fcn
    layers_weight_bias_shapes = new_NN.returnWeightBiasShapes()

    learning_rate = 0.02
    learning_decay = 1

    # Epoch run
    for epoch in range(1, num_epochs):
        #Reshuffle data
        np.random.shuffle(training_data)
        #Create batches of data. Each 32 items
        batches = [training_data[k:k + batch_size] for k in range(0, num_images_aug, batch_size)]
        learning_rate_decay = (1/(1+learning_decay*epoch))*learning_rate

        #Set progress bar
        train_batch = tqdm(batches)
        train_batch.set_description('Training Phase - Epoch: %i' % epoch)


        for batch in train_batch:


            #Running training phase. Return cost value
            batch_cost = batch_run(batch, layer, filter_one, filter_two, new_NN, layers_weight_bias_shapes,learning_rate =learning_rate_decay)
            #Update progress-bar


            train_batch.set_postfix(batch_cost=batch_cost)



    all_data =[filter_one.getfilter(), filter_two.getfilter(), new_NN.getweight(1), new_NN.getweight(2),
               filter_one.getbias(), filter_two.getbias(), new_NN.getbias(1), new_NN.getbias(2),filter_one_size, filter_two_size, fcn_layers]
    #Save all
    np.save("SavedWeightsandBias/"+savedname, all_data)

    # Test data after training. Create a new data object
    test_data_array = Data()
    #Set number of testing images
    testing_images = 10000
    #Load testing data
    test_data_array.readByte('t10k-images-idx3-ubyte.gz', image_size, testing_images)
    test_data_array.readLables('t10k-labels-idx1-ubyte.gz', testing_images, dataaugment=False)
    #Return a testing dataset
    testing_data = test_data_array.combine_and_shuffle(train=False)
    labels = test_data_array.labels



    test_stack_images = np.reshape(testing_data[0], (1, 1, testing_data[0].shape[0], testing_data[0].shape[0]))



    for image_batch in tqdm(range(1, len(testing_data)),desc='Stacking Data'):
        image = np.reshape(testing_data[image_batch], (1, 1, testing_data[image_batch].shape[0], testing_data[image_batch].shape[0]))
        test_stack_images = (np.vstack((test_stack_images.copy(), image)))



    #Count correct and wrong.
    correct = 0
    wrong = 0
    review = []
    pred, prob = predict_test(test_stack_images, layer, filter_one, filter_two, new_NN)

    prediction = tqdm (pred)

    for index, run in enumerate(prediction):
        if run == labels[index]:
            correct += 1
        else:
            wrong += 1
            wrong_pred = (test_stack_images[index],labels[index])
            review.append(wrong_pred)


        # Update Report
        prediction.set_postfix(correct=correct, wrong=wrong)
    # Print Accuracy
    print("Accuracy {}".format(correct / testing_images))
    np.save("Review_train_test", review)





    #
    #
    # #Testing phase
    # test_batch = tqdm(testing_data)
    # for trial, image in enumerate(test_batch):
    #     #Receive a highest prediction and check label.
    #     pred, prob = predict_test(image, layer, filter_one, filter_two, new_NN, labels_array[trial])
    #     #Increase variables according to successful or failure identification
    #     if pred == labels[trial]:
    #         correct += 1
    #     else:
    #         wrong += 1
    #     #Update Report
    #     test_batch.set_postfix(correct=correct, wrong=wrong)
    # #Print Accuracy
    # print ("Accuracy {}".format(correct/testing_images))





def predict_test(image, layer, filter_one, filter_two, new_NN):



    #Input image into first convolution layer
    layer.weights_forward_test(image, filter_one)
    #Relu output from filter one
    layer.relu_test(filter_one.output_conv, filter_one)
    #Insert Relu output in second convolution layer
    layer.weights_forward_test(filter_one.relu_forward, filter_two)
    # Relu output from filter two
    layer.relu_test(filter_two.output_conv, filter_two)
    #Insert Relu from second layer output to the max pooling layer.
    layer.max_pooling_forward_test(filter_two.relu_forward, filter_two)
    #Reshape max pooling output to a flatten vector for the first layer in the FCN
    fully_connected_image_input = np.reshape(filter_two.max_pooling.flatten(),
                                             (image.shape[0],new_NN.get_input_layers_nodes_number())).T



    #Insert vector into the FCN
    new_NN.setNetwork_input_data(fully_connected_image_input)
    #Run FCN. Return prediction
    argmax, max = new_NN.forward_propagation_test_A(None, train=False, dropout=False, batchnorm=False)

    return argmax, max



#Run network. Forward only
def predict(image, layer, filter_one, filter_two, new_NN, labels):
    #Reshape Image to support dimensions.
    image_input = np.reshape(image, (1, image.shape[0], image.shape[0]))

    #Input image into first convolution layer
    layer.weights_forward(image_input, filter_one)
    #Relu output from filter one
    layer.relu(filter_one.output_conv, filter_one)
    #Insert Relu output in second convolution layer
    layer.weights_forward(filter_one.relu_forward, filter_two)
    # Relu output from filter two
    layer.relu(filter_two.output_conv, filter_two)
    #Insert Relu from second layer output to the max pooling layer.
    layer.max_pooling_forward(filter_two.relu_forward, filter_two)
    #Reshape max pooling output to a flatten vector for the first layer in the FCN
    fully_connected_image_input = np.reshape(filter_two.max_pooling.flatten(),
                                             (filter_two.max_pooling.flatten().shape[0], 1))

    #Insert vector into the FCN
    new_NN.setNetwork_input_data(fully_connected_image_input)
    #Run FCN. Return prediction
    argmax, max = new_NN.forward_propagation(labels, train=False, dropout=False, batchnorm=False)

    return argmax, max


#Run first batch.
def batch_run(batch, layer, filter_one, filter_two, new_NN, layers_weight_bias_shapes, learning_rate =0.01, betaA = 0.95, betaB=0.99):

    # #initialize cost to 0.
    # loss_value = 0
    #
    # # Create empty (zero) matrices for partial derivative for filters, weights and biases. df= filter, dw= weights, db=bias
    # df1 = np.zeros(filter_one.filter.shape)
    # db1 = np.zeros(filter_one.bias.shape)
    #
    # df2 = np.zeros(filter_two.filter.shape)
    # db2 = np.zeros(filter_two.bias.shape)
    #
    # dw3 = np.zeros(layers_weight_bias_shapes[0][0])
    # db3 = np.zeros(layers_weight_bias_shapes[0][1])
    #
    # dw4 = np.zeros(layers_weight_bias_shapes[1][0])
    # db4 = np.zeros(layers_weight_bias_shapes[1][1])

    # Create zero matrices for RMSProp and Momentum update
    filter_momentum1 = np.zeros(filter_one.filter.shape)
    filter_momentum2 = np.zeros(filter_two.filter.shape)
    weight_momentum3 = np.zeros(layers_weight_bias_shapes[0][0])
    weight_momentum4 = np.zeros(layers_weight_bias_shapes[1][0])
    bias_momentum1 = np.zeros(filter_one.bias.shape)
    bias_momentum2 = np.zeros(filter_two.bias.shape)
    bias_momentum3 = np.zeros(layers_weight_bias_shapes[0][1])
    bias_momentum4 = np.zeros(layers_weight_bias_shapes[1][1])

    RMSProp_filter1 = np.zeros(filter_one.filter.shape)
    RMSProp_filter2 = np.zeros(filter_two.filter.shape)
    RMSProp_weight3 = np.zeros(layers_weight_bias_shapes[0][0])
    RMSProp_weight4 = np.zeros(layers_weight_bias_shapes[1][0])
    bias_RMSProp1 = np.zeros(filter_one.bias.shape)
    bias_RMSProp2 = np.zeros(filter_two.bias.shape)
    bias_RMSProp3 = np.zeros(layers_weight_bias_shapes[0][1])
    bias_RMSProp4 = np.zeros(layers_weight_bias_shapes[1][1])




    #size of batch
    batch_itr = len(batch)

    batch_stack_images = np.reshape(batch[0][0], (1, 1, batch[0][0].shape[0], batch[0][0].shape[0]))
    batch_stack_label = batch[0][1]


    for image_batch in range(1, batch_itr):
        image = np.reshape(batch[image_batch][0], (1, 1, batch[image_batch][0].shape[0], batch[image_batch][0].shape[0]))
        label = batch[image_batch][1]
        batch_stack_images = (np.vstack((batch_stack_images.copy(), image)))
        batch_stack_label = (np.hstack((batch_stack_label.copy(), label)))


    loss, df1, db1, df2, db2, dw3, db3, dw4, db4 = batch_one_run(batch_stack_images, batch_stack_label, layer,
                                                                        filter_one, filter_two, new_NN,x=True)

    # a,b,c,d= batch_one_run(batch_stack_images, batch_stack_label, layer,
    #                                                         filter_one, filter_two, new_NN,x=True)
    # # end = time.time()
    # print("Vectorized running time {}".format(end - start))

    # starta = time.time()

    #Iterate over batch

    # for image in range(batch_itr):
    #     #Reshape image to support dimension
    #     image_input = np.reshape(batch[image][0],(1, batch[image][0].shape[0],batch[image][0].shape[0]))
    #
    #     #Run image with forward and back propagation. Return derivatives from filters, weights and biases with a loss value
    #     #loss, d_f1, d_b1, d_f2, d_b2, d_w3, d_b3, d_w4, d_b4 = batch_one_run(image_input, batch[image][1], layer, filter_one, filter_two, new_NN,x=False)
    #     one, two,three,four = batch_one_run(image_input, batch[image][1], layer, filter_one, filter_two, new_NN, x=False)
    #
    #
    #
    #
    #     # print ("false") if not (np.array_equal(np.round(d[image], 11), np.round(four, 11))) else None
    #     # print (a.shape)
    #     # print (one.shape)
    #     #
    #     # print (a[0,0,0:4,0:4])
    #     #
    #     # print ()
    #     # print(one[0,0:4, 0:4])
    #     # exit(45)
    #
    #     #
    #     testb+= two
    #     testc += three
    #     #
    #
    #
    #     #testa += loss
    #     #testb += loss1
    #     ######################
    #     ###############
    #     #######
    #     # print()
    #     # print (a[0,0,0].shape)
    #     # print (loss.shape)
    #
    #     # print (a[0,0,0:6,14:20])
    #     # print(loss[0, 0:6, 14:20])
    #
    #     #
    #     # print (np.array_equal(np.round(a[0,0,0],10) ,np.round(loss, 10)))
    #     # exit(3234)
    #     # #
    #     # # #
    #     # # print (loss[0])
    #     # # print (a[0])
    #     # # print (np.sum(a)/32)
    #     # # print(np.sum(a))
    #     #
    #     # print (testa)
    #
    #
    #     #increment results for all the derivatives metrics images in the batch.
    #     # df1+=d_f1
    #     # db1+=d_b1
    #     # df2+=d_f2
    #     # db2+=d_b2
    #     # dw3+=d_w3
    #     # db3+=d_b3
    #     # dw4+=d_w4
    #     # db4+=d_b4
    #     #
    #     # loss_value += loss



    # Calculate Momentum update
    filter_momentum1 = betaA * filter_momentum1 + (1 - betaA) * df1 / batch_itr
    # Calculate RMSProp update
    RMSProp_filter1 = betaB * RMSProp_filter1 + (1 - betaB) * (df1 / batch_itr) ** 2
    #Get filter matrix
    f1 = filter_one.getfilter()
    # Calculate filter change with RMS and momentum
    f1 -= learning_rate * filter_momentum1 / np.sqrt(RMSProp_filter1+1e-7)
    #Save matrix
    filter_one.setfilter(f1)

    bias_momentum1 = betaA * bias_momentum1 + (1 - betaA) * db1 / batch_itr
    bias_RMSProp1 = betaB * bias_RMSProp1 + (1 - betaB) * (db1 / batch_itr) ** 2
    b1 = filter_one.getbias()
    b1 -=  learning_rate * bias_momentum1 / np.sqrt(bias_RMSProp1 + 1e-7)

    filter_one.setbias(b1)

    filter_momentum2 = betaA * filter_momentum2 + (1 - betaA) * df2 / batch_itr
    RMSProp_filter2 = betaB * RMSProp_filter2 + (1 - betaB) * (df2 / batch_itr) ** 2
    f2 = filter_two.getfilter()
    f2 -=  learning_rate * filter_momentum2 / np.sqrt(RMSProp_filter2 + 1e-7)
    filter_two.setfilter(f2)

    bias_momentum2 = betaA * bias_momentum2 + (1 - betaA) * db2 / batch_itr
    bias_RMSProp2 = betaB * bias_RMSProp2 + (1 - betaB) * (db2 / batch_itr) ** 2
    b2 = filter_two.getbias()
    b2 -= learning_rate * bias_momentum2 / np.sqrt(bias_RMSProp2 + 1e-7)
    filter_two.setbias(b2)

    weight_momentum3 = betaA * weight_momentum3 + (1 - betaA) * dw3 / batch_itr
    RMSProp_weight3 = betaB * RMSProp_weight3 + (1 - betaB) * (dw3 / batch_itr) ** 2
    w3 = new_NN.getweight(1)
    w3 -=  learning_rate * weight_momentum3 / np.sqrt(RMSProp_weight3 + 1e-7)
    new_NN.set_der_as_weight(1, w3)

    bias_momentum3 = betaA * bias_momentum3 + (1 - betaA) * db3 / batch_itr
    bias_RMSProp3 = betaB * bias_RMSProp3 + (1 - betaB) * (db3 / batch_itr) ** 2
    b3 = new_NN.getbias(1)
    b3 -= learning_rate * bias_momentum3 / np.sqrt(bias_RMSProp3 + 1e-7)
    new_NN.set_der_as_bias(1, b3)

    weight_momentum4 = betaA * weight_momentum4 + (1 - betaA) * dw4 / batch_itr
    RMSProp_weight4 = betaB * RMSProp_weight4 + (1 - betaB) * (dw4 / batch_itr) ** 2
    w4 = new_NN.getweight(2)
    w4 -=learning_rate * weight_momentum4 / np.sqrt(RMSProp_weight4 + 1e-7)
    new_NN.set_der_as_weight(2, w4)

    bias_momentum4 = betaA * bias_momentum4 + (1 - betaA) * db4 / batch_itr
    bias_RMSProp4 = betaB * bias_RMSProp4 + (1 - betaB) * (db4 / batch_itr) ** 2
    b4 = new_NN.getbias(2)
    b4 -= learning_rate * bias_momentum4 / np.sqrt(bias_RMSProp4 + 1e-7)
    new_NN.set_der_as_bias(2, b4)

    #Return loss value divided by the number of batch
    return np.average(loss)

#Back and forward propogation
def batch_one_run(image, label, layer, filter_one, filter_two, new_NN, x):

    #Load input image in the first convolutional  layer

    if x:
        v =layer.weights_forward_test(image, filter_one)
    else:

        v = layer.weights_forward(image, filter_one)



    #Relu output matrix from the first convolutional  layer

    abc = layer.relu_test(v.copy(), filter_one)

    #layer.relu(filter_one.output_conv.copy(), filter_one)





    # Load relu output in the second convolutional layer
    if x:

        bla = layer.weights_forward_test(abc.copy(), filter_two, sec=True)
    else:
        bla = layer.weights_forward(abc.copy(), filter_two, sec =True)



    if x:
        bl = layer.relu_test(bla, filter_two)
    else:
        bl = layer.relu_test(bla, filter_two)

    # Relu output matrix from the second convolutional layer
    #layer.relu(filter_two.output_conv.copy(), filter_two)






    # Max pool output from the relu output from the second relu
    if x:
        a,b,shape = layer.max_pooling_forward_test(bl, filter_two)

        #return a,b


    else:

        a,b,shape = layer.max_pooling_forward(bl, filter_two)
        #return a, filter_two.max_pooling





    if x:
        #print (filter_two.max_pooling.shape)

        fcn = np.reshape(filter_two.max_pooling.copy(),
                                             (image.shape[0], new_NN.get_input_layers_nodes_number())).T



        #print ("fcn {}".format(fcn.shape))
        #return (fcn)
    else:


        fully_connected_image_input = np.reshape(filter_two.max_pooling.copy(),
                                             (image.shape[0],new_NN.get_input_layers_nodes_number())).T

        #print("fully_connected_image_input {}".format(fully_connected_image_input.shape))
        #return (fully_connected_image_input)




    if x:
        new_NN.setNetwork_input_data(fcn)
    else:
        new_NN.setNetwork_input_data(fully_connected_image_input)





    #Run forward FCN. Return loss
    if x:
        loss = new_NN.forward_propagation_test_A(label)

    else:

        loss = new_NN.forward_propagation(label)




    #Run backpropagation. Return derivative (sum loss)
    if x:
        fc_back = new_NN.back_propagation_test_A(label,shape)


    else:
        fc_back = new_NN.back_propagation(label,shape)




    #Max pooling backward layer
    if x:
        t = layer.max_pooling_backward_test(fc_back, filter_two)
    else:
        t = layer.max_pooling_backward(fc_back, filter_two)


    #Apply Relu layer on Max pool back
    if x:

        sf =layer.relu_backward_test(t, filter_two)
    else:
        sf = layer.relu_backward(t, filter_two)


    # Apply weight back backpropagation layer on Relu d_output
    if x:

        a,b,c=layer.weights_backwards_test(sf, filter_one.relu_forward.copy(), filter_two)
    else:
        a, b, c=layer.weights_backwards(sf, filter_one.relu_forward.copy(), filter_two)




    if x:
        d =layer.relu_backward(a, filter_one)

    else:
        d = layer.relu_backward(a, filter_one)




    # Apply Relu layer on d_output from second layer


    # Apply weight back backpropagation layer on Relu d_output

    if x:
        v,x,f= layer.weights_backwards_test(d, image, filter_one,two_itr=True)

    else:
        v,x,f = layer.weights_backwards(d, image, filter_one, two_itr=True)



    # Return d_outputs
    return loss, filter_one.d_out_filter.copy(), filter_one.d_out_bias.copy(), filter_two.d_out_filter.copy(), filter_two.d_out_bias.copy(), \
           new_NN.getdweight(1).copy(), new_NN.getdbias(1).copy(), new_NN.getdweight(2).copy(), new_NN.getdbias(2).copy()



def test_CNN(path):

    image_size =28
    #Load parameters
    [f1, f2, w3, w4, b1, b2, b3, b4, filter_one_size, filter_two_size, fcn_layers]  = np.load(path)

    #Create a dataobject
    test_data_array = Data()
    testing_images = 10000

    #Load testing data input and images into object

    test_data_array.readByte('t10k-images-idx3-ubyte.gz', image_size, testing_images)
    test_data_array.readLables('t10k-labels-idx1-ubyte.gz', testing_images, dataaugment=False)
    #Get testing data
    testing_data = test_data_array.combine_and_shuffle(train=False)
    # Get label number
    labels = test_data_array.labels
    # Get label vector
    labels_array = test_data_array.labels_array



    #create filter objects
    filter_one = Conv_Data(filter_one_size)
    filter_two = Conv_Data(filter_two_size)


    # create convolution objects
    layer = Convolution()
    nodes_per_layers = fcn_layers
    new_NN = NN(len(nodes_per_layers), nodes_per_layers)

    #Load data from file in filters, weights and bias
    filter_one.setfilter(f1)
    filter_two.setfilter(f2)
    filter_one.setbias(b1)
    filter_two.setbias(b2)
    new_NN.set_der_as_weight(1, w3)
    new_NN.set_der_as_weight(2, w4)
    new_NN.set_der_as_bias(1, b3)
    new_NN.set_der_as_bias(2, b4)


    test_data_array = Data()
    #Set number of testing images
    testing_images = 10000
    #Load testing data
    test_data_array.readByte('t10k-images-idx3-ubyte.gz', image_size, testing_images)
    test_data_array.readLables('t10k-labels-idx1-ubyte.gz', testing_images, dataaugment=False)
    #Return a testing dataset
    testing_data = test_data_array.combine_and_shuffle(train=False)
    labels = test_data_array.labels

    test_stack_images = np.reshape(testing_data[0], (1, 1, testing_data[0].shape[0], testing_data[0].shape[0]))
    print (len(testing_data))

    for image_batch in tqdm(range(1,len(testing_data)),desc="Stacking Data"):

        image = np.reshape(testing_data[image_batch], (1, 1, testing_data[image_batch].shape[0], testing_data[image_batch].shape[0]))
        test_stack_images = (np.vstack((test_stack_images.copy(), image)))


    #Count correct and wrong.
    correct = 0
    wrong = 0
    pred, prob = predict_test(test_stack_images, layer, filter_one, filter_two, new_NN)

    prediction = tqdm(pred)
    review = []

    for index, run in enumerate(prediction):
        if run == labels[index]:
            correct += 1
        else:
            wrong += 1
            review_add = (test_stack_images[index],labels[index])
            review.append(review_add)

        # Update Report
        prediction.set_postfix(correct=correct, wrong=wrong)
    # Print Accuracy
    print("Accuracy {}".format(correct / testing_images))
    np.save("Review_test_set", review)



def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()

    group.add_argument("-t", "--train", help="Training Phase",
                        action="store_true")

    group.add_argument("-v", "--validate", help="Training Phase",
                        action="store_true")

    parser.add_argument("file", help="Data file")

    args = parser.parse_args()


    if args.train:
        print ('Training Phase')
        train_CNN(args.file)
    elif args.validate:
        print('Validation Phase')
        test_CNN(args.file)




main()
