from keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, Dense, MaxPooling1D, Convolution1D, Dropout, Activation, GlobalMaxPool1D
from keras.models import Model, Sequential
from tensorflow.keras import regularizers

def lr_scheduler(epoch, lr):      #For tuning the learning rate
    if epoch > 0:
        lr = 0.0001
        return lr
    return lr
model = Sequential()  #Sequential layers
model.add(Embedding(max_vocab, 150, input_length = max_sent_length)) #Embedding layer
model.add(Bidirectional(LSTM(60, return_sequences = True, dropout = 0.2))) #BiLSTM
model.add(Convolution1D(32, 3, padding = 'valid', activation = 'relu'))  # 1D Conv
model.add(GlobalMaxPool1D())
model.add(Dropout(0.6))     #High droput to reduce overfitting
model.add(Dense(40, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation =  'sigmoid'))
optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])  #Adam gave better results than SGD
print(model.summary())
batch_size = 64
epochs = 10
callbacks = [keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)]
hist = model.fit(X_train_tokenized, np.array(Y_train), batch_size = batch_size, epochs = epochs, verbose = 1,  validation_data = (X_test_tokenized, np.array(Y_test)),  callbacks = callbacks)
