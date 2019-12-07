from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import csv
import datetime

dic19 = {'2': 0,
         '3': 1,
         '4': 2,
         '5': 3,
         '7': 4,
         '9': 5,
         'A': 6,
         'C': 7,
         'F': 8,
         'H': 9,
         'K': 10,
         'M': 11,
         'N': 12,
         'P': 13,
         'Q': 14,
         'R': 15,
         'T': 16,
         'Y': 17,
         'Z': 18}


def to_onelist(text):
    label_list = []
    for c in text:
        onehot = [0 for _ in range(19)]
        onehot[dic19[c]] = 1
        label_list.append(onehot)
    return label_list


def readData(img_path, begin, end, label_path, print_out = ""):
    print("Reading {} data...".format(print_out))
    data = np.stack([np.array(Image.open(img_path + str(index) + ".jpg")) / 255.0 for index in range(begin, end + 1)])
    with open(label_path, 'r', encoding='utf8') as label_csv:
        read_label = [to_onelist(row[0]) for row in csv.reader(label_csv)]
        label = [[] for _ in range(4)]
        for arr in read_label:
            for index in range(4):
                label[index].append(arr[index])
        label = [arr for arr in np.asarray(label)]
    print("Shape of {} data = {}".format(print_out, data.shape))
    return data, label


def plot(x_axis, curves, tags, title):
    fig = plt.figure()
    ax = fig.add_axes([0.12, 0.12, 0.55, 0.8])
    for i in range(len(tags)):
        ax.plot(x_axis, curves[tags[i]])
    ax.set_title(title)
    ax.legend(tags, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.set_xlabel("epochs")

num_train, num_valid, total_num = 3600, 600, 5000   # number of images for _

if __name__ == "__main__":
    print('Creating CNN model...')     # create model
    tensor_in = Input((48, 140, 3))

    output = tensor_in
    output = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(output)
    output = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(output)
    output = MaxPooling2D(pool_size=(2, 2))(output)
    output = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(output)
    output = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(output)
    output = MaxPooling2D(pool_size=(2, 2))(output)
    output = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(output)
    output = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(output)
    output = BatchNormalization(axis=1)(output)
    output = MaxPooling2D(pool_size=(2, 2))(output)
    output = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(output)
    output = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(output)
    output = MaxPooling2D(pool_size=(2, 2))(output)
    output = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(output)
    output = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(output)  # Addition
    output = BatchNormalization(axis=1)(output)
    output = MaxPooling2D(pool_size=(2, 2))(output)

    output = Flatten()(output)
    output = Dropout(0.5)(output)

    # The output of CNN layers branch to 4 DNN layers
    # Hidden layer
    output = [Dense(4096, activation='relu')(output),
              Dense(4096, activation='relu')(output),
              Dense(4096, activation='relu')(output),
              Dense(4096, activation='relu')(output)]

    output = [Dense(19, name='char1', activation='softmax')(output[0]),
              Dense(19, name='char2', activation='softmax')(output[1]),
              Dense(19, name='char3', activation='softmax')(output[2]),
              Dense(19, name='char4', activation='softmax')(output[3])]

    model = Model(inputs=tensor_in, outputs=output)    # model ends

    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])  # Adamax
    model.summary()   # Print model structure

    # input data
    train_img, train_label = readData("preprocessed_img/", 1, num_train, 'label/train.csv', "training")
    valid_img, vali_label = readData("preprocessed_img/", num_train + 1, num_train + num_valid, 'label/valid.csv', "validation")

    # training
    epoch = 20
    result = model.fit(train_img, train_label, batch_size=50, epochs=epoch, verbose=2, validation_data=(valid_img, vali_label))
    print('{}---training finished------------------------------'.format(datetime.datetime.now()))
    model.save('model/cnn_model.hdf5')   # Saving model

    # testing
    test_img, test_label = readData("preprocessed_img/", num_train + num_valid + 1, total_num, 'label/testing.csv', "testing")
    test_result = model.evaluate(test_img, test_label)   # Testing

    test_result_tags = ['total_loss', 'char1_loss', 'char2_loss', 'char3_loss', 'char4_loss', 'test_char1_accuracy', 'test_char2_accuracy', 'test_char3_accuracy', 'test_char4_accuracy']
    for i, tag in enumerate(test_result_tags):
        print("{} = {:2.4f}".format(tag, test_result[i]))

    # plot history
    acc_tags = ['char1_accuracy', 'char2_accuracy', 'char3_accuracy', 'char4_accuracy', 'val_char1_accuracy', 'val_char2_accuracy', 'val_char3_accuracy', 'val_char4_accuracy']
    loss_tags = ['loss', 'char1_loss', 'char2_loss', 'char3_loss', 'char4_loss', 'val_char1_loss', 'val_char2_loss', 'val_char3_loss', 'val_char4_loss']
    plot(list(range(1, epoch + 1)), result.history, acc_tags, 'Training and validation accuracy')
    plot(list(range(1, epoch + 1)), result.history, loss_tags, 'Training and validation loss')
    plt.show()