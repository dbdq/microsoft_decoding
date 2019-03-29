## My approach to Microsoft Brain Decoding competition - First Prize award


**How to run**

Uncompress the data files in raw/ and run tester.py to automatically perform a cross validation and train models for each subject.


**Background**

Data acquisition method and protocol is published [here][1].


**Overview of the method**

Two classifiers were trained per subject and the final decision was made by averaging class probabilities. To start with, a grand average of time-frequency representation (TFR) plots were computed to see the general trend. For this purpose, [mne.tfr_multitaper][2] function of [Python MNE library][3] was used.


**Feature Computation**

Two types of features were used: 1) frequency-domain and 2) time-domain features. Once features are computed, feature importances were auotmatically ranked, i.e. no manual feature selection. Each classifier was trained separately from each domain of features. Throughout this document, “0 ms” denotes the time point when either face or house image was shown. All features were computed within each epoch. Since the classifer was able to efficiently deal with high dimensional features and feature importances were computed automatically during the training stage, no manual feature index selection was performed for both types of features. 

**1) Frequency-domain features**

Raw signals of each epoch were first common average-referenced (CAR) since the electrodes were implanted over a wide area of a brain (frontal, parietal, temporal, and occipital cortex). Notch filter in the multiples of 60 Hz was applied as ECoG signals are not immune to powerline noise, as also mentioned in [here][4] and [here][5]. Power spectral density (PSD) was computed in the range of [1, 150] Hz within each epoch using a [multitaper method][6], a robust non-parametric PSD estimation method. It was observed from TFR plots that some channels contained distinct “state transitions” along the time axis. So instead of computing PSD over the whole epoch range [0, 400]ms, PSD was computed from one or more (overlapping) sub-epochs separately, e.g. [100, 300]ms and [200, 400]ms, and concatenated to roughly capture the local dynamics. 

The dimension of each PSD vector is determined by the frequency resolution, which is determined by the sampling frequency and the epoch length. PSD vectors obtained from every sub-epoch are concatenated to form a single long vector, which represents a single trial of a channel. By concatenating these vectors over all channels, a frequency-domain feature vector is obtained. After observing that subjects had different temporal transition patterns, a simple grid search over different combinations of sub-epoch lengths and start positions were performed based on cross-validation accuracy. Epoch length was chosen among the following three values: 200, 300, and 400ms, while start position was chosen among the following three values: 0, 100 and 200ms. Any epoch that spans beyond 400ms was discarded.

[Random Forests (RF)][7] were used to cross-validate frequency-domain features during optimization. Being a non-linear classifier and not very sensitive to the selection of hyperparameters, it was also fast in training even with high-dimensional features.

**2) Time-domain features**

Based on TFR plots, roughly two groups of frequency ranges were defined: “Low“ (up to 10 Hz) and “High” (10-70 Hz). (70-150 Hz) range was also considered but it was eventually discarded as it didn’t contribute much to the cross-validation performance in my case. Although applying CAR is not ideal in this case, the same preprocessing code was re-used as in frequency-domain features for the sake of simplicity in the code.

For Low-range filtering, a 2nd order low-pass Butterworth filter was used. For High-range filtering, a 4th order band-pass Butterworth filter was used, followed by a Hilbert transform and taking their absolute values. After applying filters, samples of only 200-400ms range were kept after filtering and they were downsampled to 200 Hz (1 out of every 5 samples) to reduce the computational burden. The samples from both Low and High ranges were concatenated to form a long feature vector. These vectors obtained from all channels were concatenated to finally form a time-domain feature vector, which represents a single trial.


**Classification**

Once the features were computed, [Gradient Boosting Machines (GBM)][8] were trained on these pre-computed features. Two GBMs were trained for each subject, one with frequency-domain features and another with time-domain features. The class probabilities computed from GBMs were averaged to make the final decision.

To optimize GBMs, strategies explained in [here][9] and [here][10] were consulted while the number of decision trees was fixed to 1000. Although AzureML had lower scikit-learn version (0.15.1) than the latest (0.17.1) at the time of writing, the cross-validation results were identical. A K-Fold instead of randomized cross-validation was used to preserve the temporal distance between trials since the signals are more likely to be similar if their temporal proximity is lower, which may lead to an undesired boosted cross-validation accuracy. With K=6, the average cross-validation accuracy was 91.1%. My method achieved 92.5% in the final private test set.


**Running the code**

Simply running the Python code “tester.py” included in bundle.zip will read the training csv file, perform cross-validation and train classifiers. After the training is done, classifiers are saved into a single file, which needs to be uploaded as a bundle file to AzureML for testing. Please see the comments in the code (tester.py). For testing in AzureML, run the Python code defined in the “Execute Python Script” module of the experiment. This will output the result to the Web service output module.

Many codes used here are part of my online Python BCI decoding package currently in development. 
It can be downloaded from [here][11]. 


  [1]: http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004660
  [2]: http://martinos.org/mne/dev/generated/mne.time_frequency.tfr_multitaper.html
  [3]: http://martinos.org/mne/dev/python_reference.html
  [4]: http://arxiv.org/abs/1402.6862
  [5]: http://www.ncbi.nlm.nih.gov/pubmed/26157639
  [6]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1456701
  [7]: http://link.springer.com/article/10.1023/A:1010933404324
  [8]: http://www.sciencedirect.com/science/article/pii/S0167947301000652
  [9]: http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3885826/
  [10]: http://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python
  [11]: https://c4science.ch/diffusion/1299/
