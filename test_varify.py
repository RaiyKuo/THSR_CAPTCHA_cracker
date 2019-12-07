from keras.models import load_model
from Train import *

def predict(model, img_path, begin, end):
    img = []
    for i in range(begin, end+1):            # Load the target data
        img.append(np.array(Image.open(img_path + str(i) + ".jpg"))/255.0)
    img = np.stack(img)
    num = img.shape[0]

    prediction = model.predict(img)          # Use the model to predict

    text_predict = ["" for _ in range(num)]
    for p in prediction:
        for index in range(num):
            text_predict[index] += list(dic19.keys())[np.argmax(p[index])]  # Turn from one-hot encoding back to texts

    return text_predict



if __name__ == "__main__":

    result_list = predict(load_model('model/cnn_model.hdf5'), "preprocessed_img/", num_train + num_valid + 1, total_num)
    with open('label/testing.csv', 'r', encoding='utf8') as f:                   # Match the label to find the true accuracy
        read = list(csv.reader(f))
        if len(read) != len(result_list):
            print('Error: Label size dose not match to the prediction array')
        else:
            total = success = 0
            for row in read:
                answer = row[0]
                if answer == result_list[total]:
                    success += 1
                else:
                    pass
                    print("Not match. Predict = {}; Answer = {}".format(result_list[total], answer))
                total += 1
            print("Success rate = {}".format(success/(total+1)))