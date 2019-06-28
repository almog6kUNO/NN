import os
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import random
import gzip
import numpy as np


class Data:
    #Data object. Used for MNIST and Cats vs Dogs.
    def __init__(self):

        self.data = []
        self.width = []
        self.height = []
        self.cropped_data = []
        self.labels = None
        self.labels_array = []
        self.data_aug_data = []
        self.data_aug_label = []


    def data_augment(self,index, label, mirror=False, flip180=False, flip180mirror =False, convert=False, convert_num=99):
        if mirror:
            self.data_aug_data.append(np.flip(self.data[index], 1))
            self.data_aug_label.append(label)



        if flip180:
            self.data_aug_data.append(np.rot90(self.data[index], 2))
            self.data_aug_label.append(label)

        if flip180mirror:
            self.data_aug_data.append(np.flip(np.rot90(self.data[index], 2), 1))
            self.data_aug_label.append(label)

        if convert:
            if convert_num == 6:
                self.data_aug_data.append(np.rot90(self.data[index].copy(), 2))
                label[6] =0
                label[9] =1
                self.data_aug_label.append(label)


            elif convert_num == 9:

                self.data_aug_data.append(np.rot90(self.data[index].copy(), 2))
                label[6] =1
                label[9] =0
                self.data_aug_label.append(label)


    #Shuffle the dataset
    def shuffle_data(self):
        random.shuffle(self.data)

    #Read the bytes from the zipped MNIST file and store it in the data value
    def readByte(self, file, matrix_size, num_images):

        with gzip.open(file) as bytestream:
            bytestream.read(16)

            buf = bytestream.read(matrix_size * matrix_size * num_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(num_images, matrix_size, matrix_size)
            self.data = data


    def gen_image(self, arr):
        two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
        img = Image.fromarray(two_d, 'L')
        img.show()

    #Read the labels from the zipped MNIST file and store it in the labels value

    def readLables(self, file, num_images, dataaugment=True):

        with gzip.open(file) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

        #Generate an empty (zero) vector and set 1 in the label
        self.labels = labels

        for index, label in enumerate(labels):

            temp = np.zeros((10, 1))
            temp[label][0] = 1
            self.labels_array.append(temp)
            if dataaugment:
                if label == 0 or label == 1 or label == 8:

                    self.data_augment(index,temp, flip180=True, flip180mirror=True, mirror=True)

                elif label == 6 or label == 9:
                    pass

                    #self.data_augment(index, temp, convert=True, convert_num=label)

                elif label ==3:

                    self.data_augment(index, temp, flip180mirror=True)

    #Generate the data
    def combine_and_shuffle(self, train=True, dataaugment=True, batchnorm=False):
        if train and dataaugment:
            self.data = np.vstack((self.data.copy(), self.data_aug_data))
            self.labels_array = np.vstack((self.labels_array.copy(),  self.data_aug_label))

            #self.labels_array = self.labels_array.copy() + self.data_aug_label
        #Subtract the mean from the data value

        #Data mean Value
        mean_value = np.mean(self.data)
        #Data STD
        standard_value = np.std(self.data)

        if not batchnorm:

            normalize_training_mean = self.data - mean_value
            #Divide the previous dataset with the standard deviation
            normalize_training_std = normalize_training_mean / standard_value
        else:
            normalize_training_std = self.data


        #If train is true
        if train:

            #Combine the train and label (vector) into one array
            train_and_label = zip(normalize_training_std, self.labels_array)

            #Set the results as a list
            list_train_and_label = list(train_and_label)
            #Shuffle data
            np.random.shuffle(list_train_and_label)
            # num = 981
            # print (list_train_and_label[num][1])
            # self.gen_image(list_train_and_label[num][0])


            return list_train_and_label, (mean_value, standard_value)
        else:
            #If test, return the normalized dataset.
            return normalize_training_std



    #Cats VS Dogs
    def generate_data_from_path(self, mypath):
        #Get all the files from the path
        files = os.listdir(mypath)

        for index, filename in enumerate(files):
            #Open image file
            im = Image.open(mypath+'/'+filename)
            self.data.append([])
            #Store full path to image
            self.data[index].append(mypath+'/'+filename)
            width, height = im.size
            #Store the width and height for each image
            self.width.append(width)
            self.height.append(height)

            #Create a vector with the right identification according to cat or dog
            if filename[0] == 'c':
                self.data[index].append(filename)
                self.data[index].append([0])

            elif filename[0] == 'd':
                self.data[index].append(filename)
                self.data[index].append([1])



    #Plot the height and width values for picture analysis
    def plot_data(self, ignore):

        main_height = Counter(self.height)
        main_width = Counter(self.width)

        for k in list(main_height):
            if main_height[k] < ignore:
                del main_height[k]

        plt.bar(range(len(main_height)), main_height.values())
        plt.xticks(range(len(main_height)), list(main_height.keys()), rotation='vertical')
        plt.title('height')

        plt.show()

        for k in list(main_width):
            if main_width[k] < ignore:
                del main_width[k]

        plt.bar(range(len(main_width)), main_width.values())
        plt.xticks(range(len(main_width)), list(main_width.keys()), rotation='vertical')
        plt.title('width')
        plt.show()

    #Crop all images to a uniform size. If picture less than the size, enlrage it, if small, crop
    def crop_images(self, crop_size, path):

        crop = True
        im = Image.open(path)
        #Find center
        width, height = im.size
        left = (width - crop_size) / 2
        top = (height - crop_size) / 2
        right = (width + crop_size) / 2
        bottom = (height + crop_size) / 2

        if left < 0 or right < 0 or top < 0 or bottom < 0:
            crop = False

        #print ("l {} t {} r{} b {} ".format(left, top, right, bottom))
        #Crop in the center of the image
        if crop:
            image = im.crop((left, top, right, bottom))
        #If image small resize.
        else:
            image = im.resize((crop_size, crop_size), PIL.Image.ANTIALIAS)

        return image





