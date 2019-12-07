from keras.models import load_model
from test_varify import predict

for result in predict(load_model('model/cnn_model.hdf5'), 'demo/preprocessed_img/', 1, 5):
    print(result)      # Show the result
