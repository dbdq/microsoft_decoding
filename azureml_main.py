# The script MUST contain a function named azureml_main
# which is the entry point for this module.
#
# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame


def azureml_main(dataframe1=None, dataframe2=None):
    # If a zip file is connected to the third input port is connected,
    # it is unzipped under ".\Script Bundle". This directory is added
    # to sys.path. Therefore, if your zip file contains a Python file
    # mymodule.py you can import it using:
    # import mymodule
    # print('Input pandas.DataFrame #1:\r\n\r\n{0}'.format(dataframe1))

    # Import dependent modules. Run the tester module (tester.py) from your local machine to train and cross-validate your models.
    import sys
    import sklearn
    import numpy
    import pandas
    import q_common as qc
    import tester

    # System envrionment check
    print(sys.version)
    print('\nPlatform: %s' % tester.PLATFORM)
    print('sklearn: %s' % sklearn.__version__)
    print('pandas: %s' % pandas.__version__)
    print('numpy: %s' % numpy.__version__)
    print('MY_PATH: %s\n\n' % tester.MY_PATH)

    # Create a timer object to measure the runnning time
    tm = qc.Timer()

    # Load trained classifiers saved in a Python pickle format
    model_file = '%s/classifiers.pkl' % tester.MY_PATH
    model = qc.load_obj(model_file)
    assert model is not None

    # Load preprocessing and feature computation parameters
    cfg = model['cfg']
    psd_params = model['psd_params']
    epochs = model['epochs']

    # Compute features from raw data
    features = tester.get_features(dataframe1, cfg, psd_params, epochs)

    # Test classifiers on computed features
    answers_pd = tester.predictor(features, model)

    # Print out predictions and running time
    print('Done. Took %.1f seconds.' % tm.sec())
    print('\n*** Predicted labels start ***\n')
    print(answers_pd)
    print('\n*** Predicted labels end ***\n')

    # Return predictions
    return answers_pd
