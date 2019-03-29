from __future__ import print_function
from __future__ import division

"""
Prediction of labels based on ECoG data

Labels
1 = House
2 = Face

Running this code will compute features, cross validate, and train classifiers.
Computed features can be saved and loaded later. There are two classifiers trained per subject.
One classifer is trained from frequency-domain features and the other from time-domain features.

Before running, modify the following two variables to the correct path:
TRAIN_FILE: The path where the training file (csv format) is located.
LOCAL_PATH: The path where the classifier file should be saved. Make sure to upload it
            as a bundle when you test on AzureML platform later.

-- List of functions --
preprocess:
  Preprocess signals. Preprocessing is applied epoch-wise, not on the whole time series.

get_features:
  Run get_features_subject() in parallel.

get_features_subject:
  Compute frequency-domain and time-domain features of a subject.

cross_validate:
  Perform cross validation. It calls fit_predict() in parallel to compute the score for each fold.

fit_predict:
  Train and test classifiers on a given fold.

get_final_label:
  Compute the final class label given the class probabilities from classifiers.

trainer:
  Train classifiers for all subjects and save into a file.

predictor:
  Predict class labels and output in Pandas DataFrame format.

Many codes used here are part of my online BCI decoding package in development:
git clone https://anonymous@git.epfl.ch/repo/pycnbi.git

Kyuhwa Lee
Swiss Federal Institute of Technology in Lausanne (EPFL)
2016

"""

# Environment variables
TRAIN_FILE = './raw/ecog_train_with_labels.csv'
LOCAL_PATH = './upload/'
AZURE_PATH = './Script Bundle/'

import os
import sys
import mne
import sklearn
import numpy as np
import pandas as pd
import q_common as qc
import sklearn.metrics as skmetrics
from sklearn.model_selection import StratifiedShuffleSplit, KFold, LeaveOneOut
from sklearn.ensemble import GradientBoostingClassifier
from scipy.signal import hilbert
mne.set_log_level('ERROR')

if os.path.exists('Script Bundle'):
    PLATFORM = 'AzureML'
    MY_PATH = AZURE_PATH
    N_JOBS = 1
else:
    PLATFORM = 'Local'
    MY_PATH = LOCAL_PATH
    import multiprocessing as mp
    import traceback
    N_JOBS = mp.cpu_count()


def preprocess(raw, sfreq=None, spatial=None, spatial_ch=None, spectral=None, spectral_ch=None,
               notch=None, notch_ch=None, multiplier=1, ref_ch=None):
    """
    Apply spatial, spectral, notch filters and convert unit.
    raw is modified in-place.

    Input
    ------
    numpy.array (n_channels x n_samples)

    sfreq: source sampling frequency

    spatial: None | 'car' | 'laplacian'
    	Spatial filter type.

    spatial_ch: None | list (for CAR) | dict (for LAPLACIAN)
    	Reference channels for spatial filtering.
    	'car': channel indices used for CAR filtering. If None, use all channels except
    		   the trigger channel (index 0).
    	'laplacian': {channel:[neighbor1, neighbor2, ...], ...}
    	*** Note ***
    	Since PyCNBI puts trigger channel as index 0, data channel starts from index 1.

    spectral: None | [l_freq, h_freq]
    	Spectral filter. See mne.io.Raw.filter manual for how to set l_freq and h_freq.
    	if l_freq is None: lowpass filter is applied.
    	if h_freq is None: highpass filter is applied.
    	otherwise, bandpass filter is applied.

    spectral_ch: None | list
    	Channel picks for spectra filtering.

    notch: None | float | list of frequency in floats
    	Notch filter.

    notch_ch: None | list
    	Channel picks for notch filtering.

    multiplier: float
    	If not 1, multiply data values excluding trigger values.

    ref_ch: None | int | str
    	Re-reference to this channel. (simple substraction)

    ref_ch_new: None | int | str
    	************** TODO **************
    	If True, ref_ch becomes


    Output
    ------
    True if no error.

    """

    # Check datatype
    # Numpy array: assume we don't have event channel
    data = raw
    assert sfreq is not None and sfreq > 0, 'Wrong sfreq value.'
    n_channels = data.shape[0]
    eeg_channels = list(range(n_channels))

    # Do unit conversion
    if multiplier != 1:
        data[eeg_channels] *= multiplier

    # Re-reference
    if ref_ch is not None:
        data[eeg_channels] -= data[ref]

    # Apply spatial filter
    if spatial == 'car':
        if spatial_ch is None:
            data[eeg_channels] = data[eeg_channels] - np.mean(data[eeg_channels], axis=0)
        else:
            data[spatial_ch] = data[spatial_ch] - np.mean(data[spatial_ch], axis=0)
    elif spatial == 'laplacian':
        if type(spatial_ch) is not dict:
            raise RuntimeError('For Lapcacian, spatial_ch must be of form {CHANNEL:[NEIGHBORS], ...}')
        rawcopy = data.copy()
        for src in spatial_ch:
            nei = spatial_ch[src]
            data[src] = rawcopy[src] - np.mean(rawcopy[nei], axis=0)
    elif spatial is None:
        pass
    else:
        raise RuntimeError('Unknown spatial filter %s' % spatial)

    # Apply spectral filter
    if spectral is not None:
        if spectral_ch is None:
            spectral_ch = eeg_channels
        if spectral[0] is None:
            mne.filter.filter_data(data, sfreq, l_freq=None, h_freq=spectral[1], picks=spectral_ch, method='fft', copy=False, verbose='ERROR')
        elif spectral[1] is None:
            mne.filter.filter_data(data, sfreq, l_freq=spectral[0], h_freq=None, picks=spectral_ch, method='fft', copy=False, verbose='ERROR')
        else:
            mne.filter.filter_data(data, sfreq, l_freq=spectral[0], h_freq=spectral[1], picks=spectral_ch, method='fft', copy=False, verbose='ERROR')

    # Apply notch filter
    if notch is not None:
        if notch_ch is None:
            notch_ch = eeg_channels
        # parallel processing not working in AzureML
        mne.filter.notch_filter(data, Fs=sfreq, freqs=notch, notch_widths=None, picks=notch_ch, method='fft', copy=False, verbose='ERROR')

    return True


def get_features(dataframe, cfg, psd_params, epochs):
    """
    Compute feature vectors (Wrapper)

    Input
    -----
    dataframe: Pandas 2-D dataframe
    cfg: dict(sfreq, spatial, spatial_ch, spectral, spectral_ch, noth, noth_ch, ref_ch)
    psd_params: { subject:{fmin, fmax}, ... }
    epochs: [ [start,end], ... ]

    """

    subjects = np.sort(np.unique(dataframe['PatientID']))

    psd_subjects = {}
    if 'fmin' in psd_params and 'fmax' in psd_params:
        for subject in subjects:
            psd_subjects[subject] = psd_params
    else:
        for subject in subjects:
            psd_subjects[subject] = psd_params[subject]

    epochs_subjects = {}
    if type(epochs) == list:
        for subject in subjects:
            epochs_subjects[subject] = epochs
    else:
        for subject in subjects:
            epochs_subjects[subject] = epochs[subject]

    data = {}
    if N_JOBS > 1:
        pool = mp.Pool(N_JOBS)
        results = {}
        for subject in subjects:
            results[subject] = pool.apply_async(get_features_subject, [dataframe, subject, cfg, psd_subjects[subject], epochs_subjects[subject]])
        for s in results:
            data[s] = results[s].get()
        pool.close()
        pool.join()
    else:
        for subject in subjects:
            data[subject] = get_features_subject(dataframe, subject, cfg, psd_subjects[subject], epochs_subjects[subject])

    return data


def get_features_subject(dataframe, subject, cfg, psd_params, epochs, ica=None):
    """
    Compute feature vectors

    Input
    -----
    dataframe: Pandas 2-D dataframe
    cfg: dict(sfreq, spatial, spatial_ch, spectral, spectral_ch, noth, noth_ch, ref_ch)
    psd_params: dict(fmin, fmax) ]
    epochs: [ [start,end],,, ]

    Output
    ------
    X1: Frequency-domain features (2D array)
    X2: Time-domain features (2D array)
    Y: Class labels (1D array of 1-2)
    SID: Stimuls Types (1D array of 1-100)
    """

    assert type(epochs[0]) is list

    # PSD estimator
    psde = mne.decoding.PSDEstimator(sfreq=cfg['sfreq'], fmin=psd_params['fmin'], fmax=psd_params['fmax'],
                                     bandwidth=None, adaptive=False, low_bias=True,
                                     n_jobs=1, normalization='length', verbose='ERROR')

    raw = dataframe[dataframe['PatientID'] == subject]
    raw = raw[raw['Stimulus_Type'] >= 1]
    raw = raw[raw['Stimulus_ID'] >= 1]
    stims_all = raw['Stimulus_ID'].as_matrix()
    stims = np.sort(np.unique(stims_all))
    print('Computing features for %s, stimuli %d-%d' % (subject, stims[0], stims[-1]))

    # Select signals
    labels_np = raw['Stimulus_Type'].as_matrix()
    sigall = raw.ix[:, 'Electrode_1':'Electrode_64'].as_matrix().T
    sigall = sigall[np.where(sigall[:, 0] != -999999)[0]]
    n_channels = sigall.shape[0]

    X1 = None  # frequency-domain features
    X2 = None  # time-domain features
    Y = []
    SID = []
    onset = 400  # relative to the beginning of an epoch
    last_label = -1
    for i, r in enumerate(labels_np):
        if r != last_label and last_label == 101 and 1 <= r <= 100:
            sig = sigall[:, i - 400:i + 400]  # single epoch. onset=400

            # Frequency-domain features
            sig_p = sig.copy()
            preprocess(sig_p, sfreq=cfg['sfreq'], spatial=cfg['spatial'], spatial_ch=cfg['spatial_ch'],
                                   spectral=cfg['spectral'], spectral_ch=cfg['spectral_ch'], notch=cfg['notch'], notch_ch=cfg['notch_ch'],
                            multiplier=1, ref_ch=cfg['ref_ch'])

            feature1 = None
            for ep in epochs:
                s = onset + int(round(cfg['sfreq'] * ep[0]))
                e = onset + int(round(cfg['sfreq'] * ep[1])) + 1  # inclusive
                f = psde.transform(sig_p[:, s:e].reshape(1, n_channels, -1)).reshape(1, -1)
                fd = f
                if feature1 is None:
                    feature1 = fd
                else:
                    feature1 = np.concatenate((feature1, fd), axis=1)

            # Time-domain features
            sig_l = sig.copy()
            preprocess(sig_l, sfreq=cfg['sfreq'], spatial=cfg['spatial'], spatial_ch=cfg['spatial_ch'],
                       spectral=None, spectral_ch=None, notch=cfg['notch'], notch_ch=cfg['notch_ch'],
                       multiplier=1, ref_ch=cfg['ref_ch'])
            iir_params = mne.filter.construct_iir_filter({'order': 2, 'ftype': 'butter'}, 10.0, None, cfg['sfreq'], 'lowpass', return_copy=False)
            mne.filter.filter_data(sig_l, cfg['sfreq'], l_freq=None, h_freq=10.0, method='iir', iir_params=iir_params, copy=False, verbose='ERROR')
            feature_l = sig_l[:, 600::5].reshape(1, -1)

            sig_m = sig.copy()
            preprocess(sig_m, sfreq=cfg['sfreq'], spatial=cfg['spatial'], spatial_ch=cfg['spatial_ch'],
                       spectral=None, spectral_ch=None, notch=cfg['notch'], notch_ch=cfg['notch_ch'],
                       multiplier=1, ref_ch=cfg['ref_ch'])
            iir_params = mne.filter.construct_iir_filter({'order': 4, 'ftype': 'butter'}, [10, 70], None, cfg['sfreq'], 'bandpass', return_copy=False)
            mne.filter.filter_data(sig_m, cfg['sfreq'], l_freq=10, h_freq=70, method='iir', iir_params=iir_params, copy=False, verbose='ERROR')
            iir_params = mne.filter.construct_iir_filter({'order': 2, 'ftype': 'butter'}, 10.0, None, cfg['sfreq'], 'lowpass', return_copy=False)
            mne.filter.filter_data(sig_m, cfg['sfreq'], l_freq=None, h_freq=10.0, method='iir', iir_params=iir_params, copy=False, verbose='ERROR')
            feature_m = abs(hilbert(sig_m))[:, 600::5].reshape(1, -1)

            # Merge features
            feature2 = np.concatenate((feature_l, feature_m), axis=1)

            # X
            if X1 is None:
                X1 = feature1
                X2 = feature2
            else:
                X1 = np.concatenate((X1, feature1), axis=0)
                X2 = np.concatenate((X2, feature2), axis=0)

            # Y
            if 1 <= r <= 50:
                Y.append(1)
            elif 51 <= r <= 100:
                Y.append(2)
            else:
                raise RuntimeError('Unexpected label %d' % label)

            # SID
            SID.append(stims_all[i])

        last_label = r

    data = {'X1': X1, 'X2': X2, 'Y': np.array(Y), 'SID': np.array(SID)}
    return data


def cross_validate(data, cls_params):
    """
    Do cross-validation

    CV_PERFORM= ['LeaveOneOut' | 'StratifiedShuffleSplit' | 'KFold']
    """

    CV_PERFORM = 'KFold'
    CV_FOLDS = 6
    # parameters for StratifiedShuffleSplit only
    CV_TEST_RATIO = 0.2
    CV_SEED = 0

    acc_subject = {}
    scores_all = []
    for subject in data:
        gbp = cls_params[subject]
        cls1 = GradientBoostingClassifier(n_estimators=gbp['trees'], learning_rate=gbp['learning_rate'], max_depth=gbp['max_depth'], max_features=gbp['max_features'], subsample=gbp['subsample'], random_state=gbp['random_state'])
        cls2 = GradientBoostingClassifier(n_estimators=gbp['trees'], learning_rate=gbp['learning_rate'], max_depth=gbp['max_depth'], max_features=gbp['max_features'], subsample=gbp['subsample'], random_state=gbp['random_state'])
        qc.print_c('Parameters\nGB %s' % (gbp), 'W')

        X1 = data[subject]['X1']
        X2 = data[subject]['X2']
        Y = data[subject]['Y']
        if CV_PERFORM == 'LeaveOneOut':
            print('\n>> %s: %d-fold leave-one-out cross-validation' % (subject, ntrials))
            cv = LeaveOneOut()
        elif CV_PERFORM == 'StratifiedShuffleSplit':
            print('\n>> %s: %d-fold stratified cross-validation with test set ratio %.2f' %
                              (subject, CV_FOLDS, CV_TEST_RATIO))
            cv = StratifiedShuffleSplit(CV_FOLDS, test_size=CV_TEST_RATIO, random_state=CV_SEED)
        elif CV_PERFORM == 'KFold':
            cv = KFold(CV_FOLDS)

        label_set = list(np.unique(Y))
        scores = []
        num_labels = len(label_set)
        cms = np.zeros((num_labels, num_labels))
        cnum = 1

        if N_JOBS > 1:
            results = []
            pool = mp.Pool(mp.cpu_count())
            for train, test in cv.split(Y):
                p = pool.apply_async(fit_predict, [cls1, cls2, X1[train], X2[train], Y[train], X1[test], X2[test], Y[test], cnum, label_set])
                results.append(p)
                cnum += 1
            pool.close()
            pool.join()

            for r in results:
                score, cm = r.get()
                scores.append(score)
                cms += cm
        else:
            for train, test in cv.split(Y):
                score, cm = fit_predict(cls1, cls2, X1[train], X2[train], Y[train], X1[test], X2[test], Y[test], cnum, label_set)
                scores.append(score)
                cms += cm
                cnum += 1

        # Show confusion matrix
        cm_rate = cms.astype('float') / cms.sum(axis=1)[:, np.newaxis]
        cm_txt = '\nY: ground-truth, X: predicted\n'
        for l in label_set:
            cm_txt += '%4d\t' % l
        cm_txt += '\n'
        for r in cm_rate:
            for c in r:
                cm_txt += '%-4.3f\t' % c
            cm_txt += '\n'
        acc = np.mean(scores)
        print('Average accuracy: %.3f' % acc)
        print(cm_txt)
        acc_subject[subject] = acc
        scores_all += scores

    # Assuming every subject has equal number of trials
    for subject in acc_subject:
        print('%s: %.3f' % (subject, acc_subject[subject]))

    acc_all = np.mean(scores_all)
    print('Average accuracy over all subjects: %.3f' % acc_all)

    return acc_all, acc_subject


def get_final_label(cls1, cls2, X1, X2, alphas=None):
    """ Compute the final class prediction based on two classifiers """

    # Get class probabilities
    Y_pred1 = cls1.predict_proba(X1)
    Y_pred2 = cls2.predict_proba(X2)

    # Add with optional weights
    if alphas is None:
        Yj = Y_pred1 + Y_pred2
    else:
        Yj = alphas[0] * Y_pred1 + alphas[1] * Y_pred2
    Ys = np.sum(Yj, axis=1)
    Yn = Yj / Ys[:, None]

    # Predict final labels [#samples x 1]
    Y_pred = np.zeros(X1.shape[0])
    Y_pred[np.where(Yn[:, 0] > Yn[:, 1])[0]] = cls1.classes_[0]
    Y_pred[np.where(Yn[:, 0] <= Yn[:, 1])[0]] = cls1.classes_[1]

    return Y_pred


def fit_predict(cls1, cls2, X1_train, X2_train, Y_train, X1_test, X2_test, Y_test, cnum, label_set):
    """	Train and test a single fold """

    tm = qc.Timer()
    cls1.fit(X1_train, Y_train)
    cls2.fit(X2_train, Y_train)
    Y_pred = get_final_label(cls1, cls2, X1_test, X2_test)
    score = skmetrics.accuracy_score(Y_test, Y_pred)
    cm = skmetrics.confusion_matrix(Y_test, Y_pred, label_set)
    print('Cross-validation %d (%.3f) - %.1f sec' % (cnum, score, tm.sec()))

    return score, cm


def trainer(features, cfg, psd_params, epochs, cls_params):
    """ Train classifiers using computed features """

    psd_subjects = {}
    if 'fmin' in psd_params and 'fmax' in psd_params:
        for subject in features:
            psd_subjects[subject] = psd_params
    else:
        for subject in features:
            psd_subjects[subject] = psd_params[subject]

    cls1 = {}
    cls2 = {}
    for subject in features:
        print('Training %s' % subject)
        X1 = features[subject]['X1']
        X2 = features[subject]['X2']
        Y = features[subject]['Y']

        gbp = cls_params[subject]
        cls1[subject] = GradientBoostingClassifier(n_estimators=gbp['trees'], learning_rate=gbp['learning_rate'], max_depth=gbp['max_depth'], max_features=gbp['max_features'], subsample=gbp['subsample'], random_state=gbp['random_state'])
        cls1[subject].fit(X1, Y)
        cls1[subject].n_jobs = 1  # set to 1 for testing
        cls2[subject] = GradientBoostingClassifier(n_estimators=gbp['trees'], learning_rate=gbp['learning_rate'], max_depth=gbp['max_depth'], max_features=gbp['max_features'], subsample=gbp['subsample'], random_state=gbp['random_state'])
        cls2[subject].fit(X2, Y)
        cls2[subject].n_jobs = 1  # set to 1 for testing

    clsfile = '%s/classifiers.pkl' % MY_PATH
    qc.save_obj(clsfile, dict(cls1=cls1, cls2=cls2, cfg=cfg, psd_params=psd_subjects, epochs=epochs, cls_params=cls_params))
    print('Classifiers exported to %s' % clsfile)


def predictor(features, model):
    """ Predict classes using computed feataures and classifiers """

    answers = []
    for subject in features:
        print('Predicting %s' % subject)
        cls1 = model['cls1'][subject]
        cls2 = model['cls2'][subject]
        cls1.n_jobs = N_JOBS
        cls2.n_jobs = N_JOBS
        Y_pred = get_final_label(cls1, cls2, features[subject]['X1'], features[subject]['X2'])

        for sid, label in zip(features[subject]['SID'], Y_pred):
            answers.append([str(subject), int(sid), int(label)])

    cols = ['PatientID', 'Stimulus_ID', 'Scored Labels']
    answers_pd = pd.DataFrame(answers, columns=cols)
    return answers_pd


if __name__ == '__main__':
    # Epoch ranges
    epochs = {
        'p2': [[0, 0.199], [0.1, 0.299], [0.2, 0.399]],
        'p3': [[0.1, 0.299], [0.2, 0.399]],
        'p1': [[0.1, 0.299], [0.2, 0.399]],
        'p4': [[0, 0.399]]
    }

    # Preprocessing parameters
    cfg = dict(sfreq=1000.0, spatial='car', spatial_ch=None, spectral=[0.6, 200],
               spectral_ch=None, notch=[60, 120, 180, 240], notch_ch=None, ref_ch=None)

    # PSD parameters
    psd_params = dict(p1=dict(fmin=1, fmax=150), p2=dict(fmin=1, fmax=150), p3=dict(fmin=1, fmax=150), p4=dict(fmin=1, fmax=150))

    # Classifer parameters
    gb_params = {
        'p2': dict(trees=1000, learning_rate=0.01, max_depth=2, max_features=90, subsample=1.0, random_state=666),
        'p3': dict(trees=1000, learning_rate=0.01, max_depth=2, max_features=40, subsample=1.0, random_state=666),
        'p1': dict(trees=1000, learning_rate=0.01, max_depth=2, max_features='sqrt', subsample=1.0, random_state=666),
        'p4': dict(trees=1000, learning_rate=0.01, max_depth=2, max_features=40, subsample=0.9, random_state=666)
    }

    # Compute features
    # features= qc.load_obj('features.pkl') # load precomputed features to save time
    features = get_features(pd.read_csv(TRAIN_FILE), cfg, psd_params, epochs)
    qc.save_obj('features.pkl', features)  # save features

    # Cross validation
    acc = cross_validate(features, gb_params)

    # Train classifiers
    trainer(features, cfg, psd_params, epochs, gb_params)
