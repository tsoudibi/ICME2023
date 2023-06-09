def fit_model(model, train_generator, valid_generator, initial_epoch=0,
              epochs=100, model_save_dir='models/main', log_dir='logs/main',
              train_steps=None, valid_steps=None):
    """
    fit model with generator and save checkpoint
    :param model: keras model
    :param train_generator: generator with __getitem__, __len__
    :param valid_generator: generator with __getitem__, __len__
    :param epochs: epochs
    :param model_save_dir: model directory to save
    :param log_dir: log directory to save
    :param train_steps: None is len(train_generator)
    :param valid_steps: None is len(valid_generator)
    :return: trained model
    """

    import os
    import keras

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = model_save_dir + '/model.{epoch:02d}-{val_acc:.3f}.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_acc', save_best_only=False, mode='auto',
                                                 save_weights_only=False)
    tb = keras.callbacks.TensorBoard(log_dir=log_dir)

    train_steps = len(train_generator) if train_steps is None else train_steps
    valid_steps = len(valid_generator) if valid_steps is None else valid_steps

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        callbacks=[checkpoint, tb],
        validation_data=valid_generator,
        validation_steps=valid_steps,
        initial_epoch=initial_epoch,
        use_multiprocessing=True,
    )
    model.save(model_save_dir + '/model.{epoch:02d}-{val_acc:.3f}.h5',save_format='h5')
    return model

def load_latest_model(model_load_dir='models/main'):
    """
    Load latest saved model. Model must be saved like "model.01-0.98.h5".
    :param model_load_dir: Saved model's directory
    :return: model, epoch
    """

    import tensorflow as tf
    import keras
    from keras.backend import set_session
    import os
    import glob

    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # set_session(tf.Session(config=config))

    # model.99-0.98.h5
    files = glob.glob(model_load_dir + '/model.*.h5')

    if len(files) == 0:
        raise Exception('Trained model not found from ' + model_load_dir + '/model.*.h5')

    last_file = max(files, key=os.path.getctime)

    file_name = last_file.replace('\\', '/').split('/')[-1].replace('model.', '').replace('.h5', '')
    epoch = int(file_name.split('-')[0])
    acc = float(file_name.split('-')[1])
    last_file = '/home/tsou/Desktop/ICME2023/models/main/model.01.h5'
    print('loaded:',last_file)
    with keras.utils.CustomObjectScope({'relu6': tf.nn.relu6, 'DepthwiseConv2D': keras.layers.DepthwiseConv2D, 'tf': tf}):
        model = keras.models.load_model(last_file)

    # model.summary()

    print('Loaded last model - {}, epoch: {}, acc: {}'.format(last_file, epoch, acc))

    return model, epoch

def ask_load(build = None):
    """
    Ask user "Load last trained model? (y/n)"
    :return: (bool) load
    """

    while True:
        print('Load last trained model? (y/n)')
        if build == True:
            load = False
            print('Building new model...')
            break
        elif build == False:
            load = True
            print('Loading model...')
            break
        else:
            answer = input()
            if (answer == 'y') :
                load = True
                print('Loading model...')
                break
            elif (answer == 'n'):
                load = False
                print('Building new model...')
                break
            else:
                print("Please enter 'y' or 'n'")
                continue

    return load
