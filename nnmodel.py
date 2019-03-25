import keras
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation,Input
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D
from keras.layers import GRU
from keras.models import model_from_json

import os


def simpleMLP(vocab, num_labels):
    model = Sequential()
    model.add(Dense(512, input_shape=(len(vocab) + 1,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def TextCNN(pad_length, embedding_matrix, vocab, num_labels):
    main_input = Input(shape=(pad_length,), dtype='float64')
    embedder = Embedding(len(vocab) + 1, 300, input_length=pad_length, weights=[embedding_matrix], trainable=True)
    embed = embedder(main_input)
    cnn1 = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPool1D(pool_size=4)(cnn1)
    cnn2 = Convolution1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPool1D(pool_size=4)(cnn2)
    cnn3 = Convolution1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPool1D(pool_size=4)(cnn3)
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(num_labels, activation='softmax')(drop)
    model = Model(inputs=main_input, outputs=main_output)

    return model

def CNNRNN(vec_size):
    model = Sequential()
    # model.add(Embedding(len(vocab) + 1, 300, input_length=20))
    model.add(Convolution1D(256, 3, padding='same', strides=1,input_shape=(vec_size, 1)))
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(2, activation='softmax'))

    return model

def CNNRNN_words(embeddings_matrix, pad_length, num_labels, vec_size):
    model = Sequential()
    model.add(Embedding(len(embeddings_matrix),
                        vec_size,
                        weights=[embeddings_matrix],
                        input_length=pad_length,
                        trainable=True))
    model.add(Convolution1D(256, 3, padding='same', strides=1))
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1))
    model.add(Dense(num_labels, activation='softmax'))

    return model

def fit_eva(model, x_train, y_train, x_test, y_test, loaddir=None,savedir=None, loadname='model',savename='model', early_stop=False):
    if loaddir:
        loadjson = os.path.join(loaddir,loadname+'.json' )
        loadh5 = os.path.join(loaddir,loadname+'.h5' )
        json_file = open(loadjson, 'r')

        loaded_model_json = json_file.read()
        json_file.close()
        usemodel = model_from_json(loaded_model_json)
        usemodel.load_weights(loadh5)
        print("Loaded model from disk")
    else:
        usemodel = model

    usemodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if early_stop:
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        usemodel.fit(x_train, y_train,
                  batch_size=200,
                  epochs=20,
                  shuffle=True,
                  validation_data=(x_test, y_test),
                     callbacks=[early_stop])
    else:
        usemodel.fit(x_train, y_train,
                     batch_size=200,
                     epochs=20,
                     shuffle=True,
                     validation_data=(x_test, y_test))

    if savedir:
        savejson = os.path.join(savedir, savename + '.json')
        saveh5 = os.path.join(savedir, savename + '.h5')
        model_json = model.to_json()
        with open(savejson, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(saveh5)
        print("Saved model to disk")

    score = usemodel.evaluate(x_test, y_test, verbose=0)
    print('Test loss: {}'.format(score[0]))
    print('Test accuracy: {}'.format(score[1]))




