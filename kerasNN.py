import time

def kerasNN(hidden_layers = 2, neurons = 20, activation = 'elu', optimizer = 'adam', learning_rate = 0.001,
            l2_penalty = 0.0001, l1_penalty = 0, loss= 'mse', weight_initializer = 'glorot_uniform', input_shape = None,
            early_stopping = False, loss_to_monitor = 'val', patience = 10, min_delta = 0.0001, dropout= False,
            dropout_rate= 0.2, batch_normalization = False, metrics = None, validation_split = 0.0):
    from tensorflow.keras.layer import Dense, Dropout, BatchNormalization
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import optimizers, regularizers, losses
    from tensorflow.callbacks import EarlyStopping
    import tensorflow.keras.backend as Kb
    import tensorflow as tf
    K.clear_session()
    gc.collect()
    Kb.set_session(tf.compat.v1.Session())

    if hasattr(neurons, "__len__"):
        if len(neurons) > hidden_layers:
            neurons = neurons[:hidden_layers]
        else:
            neurons = [neurons] * hidden_layers

    adam = optimizers.Adam(learning_rate = learning_rate, beta_1 = 0.9, beta_2=0.999, amsgrad = False)
    rmsprop = optimizers.RMSprop(lr=learning_rate)
    if optimizer == 'adam':
        optim = adam
    elif optimizer == 'rmsprop':
        optim = rmsprop

    if early_stopping == True:
        es = EarlyStopping(monitor = loss_to_monitor, mode='min', verbose=1, patience=patience, min_delta =min_delta)
        callbacks = [es]
    else:
        callbacks = None

    regularizer = regularizers.l1_l2(l1=l1_penalty, l2=l2_penalty)

    #add input layer and first hidden layer
    model_init = Sequential()
    if dropout == True:
        model_init.add(Dropout(dropout_rate), input_shape)
        model_init.add(Dense(neurons[0], activation = activation, kernel_regularizer = regularizer,
                             kernel_initializer = weight_initializer, input_shape = input_shape))
    if batch_normalization == True:
        model_init.add(BatchNormalization())

    #add more hidden layers
    for i in range(1, int(hidden_layers)):
        if dropout == True:
            model_init.add(Dropout(dropout_rate))
        model_init.add(Dense(units = neurons[i], activation = activation, kernel_regularizer = regularizer,
                             kernel_initializer = weight_initializer))
        if batch_normalization == True:
            model_init.add(BatchNormalization())

    #output layer
    if dropout == True:
        model_init.add(Dropout(dropout_rate))
    model_init.add(Dense(1, activation = 'linear', kernel_regularizer = regularizer,
                             kernel_initializer = weight_initializer))
    if batch_normalization == True:
        model_init.add(BatchNormalization())
    #tensorboard= TensorBoard(log_dir = "logs/{}".format(time()))

    model_init.compile(loss= loss, optimizer = optim, metrics = metrics, callbacks = callbacks)
    print(model_init.summary())
    return model_init

#usage: model = KerasRegressor(build_fn = kerasNN, epochs = 100, batch_size=32, verbose=0)
# model.set_params(optimizer = 'adam', learning_rate = 0.01, l2_penalty= 0.01 etc)
