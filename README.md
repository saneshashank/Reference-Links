# Reference-Links
This repository contains a curated list of articles for ML/AI/NLP.

## General trivia about basic DS libraries
* matplotlib plotting in 2D and 3D: http://nbviewer.jupyter.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-4-Matplotlib.ipynb
* Difference b/w size and count with groupby in pandas: https://stackoverflow.com/questions/33346591/what-is-the-difference-between-size-and-count-in-pandas
* pandas regex to create columns: https://chrisalbon.com/python/data_wrangling/pandas_regex_to_create_columns/
* regex in python and pandas: https://www.dataquest.io/blog/regular-expressions-data-scientists/
* plotting boxplots in plotly in python: https://plot.ly/python/box-plots/
* extract high correlation values: https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
* converting group by object to data frame (also how to avoid converting columns to indices when doing group by):https://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-object-to-dataframe
* Progress Bars in jupyter notebook with tqdm : https://towardsdatascience.com/progress-bars-in-python-4b44e8a4c482
* list comprehensions: https://www.machinelearningplus.com/python/list-comprehensions-in-python/
* Numpy 1 - basic: https://www.machinelearningplus.com/python/numpy-tutorial-part1-array-python-examples/
* Numpy 2 - advanced: https://www.machinelearningplus.com/python/numpy-tutorial-python-part2/
* Numpy 101 practice: https://www.machinelearningplus.com/python/101-numpy-exercises-python/
* Pandas 101 practice: https://www.machinelearningplus.com/python/101-pandas-exercises-python/

--------------------------------------------------------------------------------------------------------------

## Inferential Statistics 

### General
* p value: https://www.statsdirect.com/help/basics/p_values.htm
* normality tests in python: https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
* parametric significance tests in python: https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/
* non-parametric significance tests in python: https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
* multicollinearity in regression analysis: http://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/
* Effect of multicollinearity on Ordinary Least Squares solution for regression: https://en.wikipedia.org/wiki/Multicollinearity
* What is the difference between Liklihood and probability: https://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-discriminative-algorithm

### Frequentist AB testing:
* http://ethen8181.github.io/machine-learning/ab_tests/frequentist_ab_test.html

### ANOVA tests:
* https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/
* https://pythonfordatascience.org/anova-python/
* https://riffyn.com/riffyn-blog/2017/10/29/family-wise-error-rate

--------------------------------------------------------------------------------------------------

## Machine Learning

### General
* How is Bayesian classifer different from MLE classifier?: https://stats.stackexchange.com/questions/74082/what-is-the-difference-in-bayesian-estimate-and-maximum-likelihood-estimate
* Cross Validation - need for test set: https://stats.stackexchange.com/questions/223408/how-does-k-fold-cross-validation-fit-in-the-context-of-training-validation-testi AND
https://stackoverflow.com/questions/43663365/cross-validation-use-testing-set-or-validation-set-to-predict
* why LASSO shrinkag works: https://stats.stackexchange.com/questions/179864/why-does-shrinkage-work
* Why is gradient descent or optimization methods required at all if cost function minima can be found directly say by using linear algebra or differentiation ? : https://stats.stackexchange.com/questions/212619/why-is-gradient-descent-required
https://stats.stackexchange.com/questions/278755/why-use-gradient-descent-for-linear-regression-when-a-closed-form-math-solution
* Introduction to **Matrix Decomposition**: https://machinelearningmastery.com/introduction-to-matrix-decompositions-for-machine-learning/
* How to Compare ML models: https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
* difference between generative and discriminative algorithm: https://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-discriminative-algorithm
* ML crash course by google: https://developers.google.com/machine-learning/crash-course/prereqs-and-prework
* Book on Feature Engineering (Max Kuhn): http://www.feat.engineering/
* Understanding Cost Functions (video series): https://www.youtube.com/watch?v=euhATa4wgzo&index=1&list=PLNlkREaquqc6WUPMRicPbEvLyZe-7b-GT
* Build better predictive models using segmentation: https://www.analyticsvidhya.com/blog/2016/02/guide-build-predictive-models-segmentation/
* using AWS for Deep Learning: https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/
* **Extra Tree Classifier**- difference b/w extra tree classifier and random forest:https://www.thekerneltrip.com/statistics/random-forest-vs-extra-tree/

### Feature Scaling
* Standardisation vs Normalization:  https://stackoverflow.com/questions/32108179/linear-regression-normalization-vs-standardization
* Importance of Feature Scaling: http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
* feature Scaling with scikit-learn (good): http://benalexkeen.com/feature-scaling-with-scikit-learn/
* about feature scaling (bit mathematical): http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

### Feature reduction/Feature Selection:
* feature reduction using varrank: https://cran.r-project.org/web/packages/varrank/vignettes/varrank.html
* LDA based feat reduction: https://towardsdatascience.com/dimensionality-reduction-with-latent-dirichlet-allocation-8d73c586738c
* LDA usage: http://jmlr.org/papers/volume3/blei03a/blei03a.pdf
* using LDA for text classification: https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28
* http://blog.echen.me/2012/03/20/infinite-mixture-models-with-nonparametric-bayes-and-the-dirichlet-process/
* A Comprehensive Survey on various Feature Selection Methods to Categorize Text Documents: https://pdfs.semanticscholar.org/d773/c6e96ed04f6d63531ff703303404e959f82f.pdf
* A Review of Feature Selection on Text Classification: http://umpir.ump.edu.my/id/eprint/23030/7/A%20Review%20of%20Feature%20Selection%20on%20Text2.pdf
* A Robust Hybrid Approach for Textual Document Classification: https://arxiv.org/pdf/1909.05478.pdf

##### dummy vars
* dummy vars transfromation applied on prediction data: https://stackoverflow.com/questions/43578799/how-to-save-mapping-of-data-frame-to-model-matrix-and-apply-to-new-observations

### Pipelines in sklearn
* Pipelines and composite estimators: https://scikit-learn.org/stable/modules/compose.html#
* Deep dive into sklearn pipelines: https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines
* Feature Union: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion
* Column Transformer with Heterogeneous Data Sources: https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer.html#sphx-glr-auto-examples-compose-plot-column-transformer-py

### Boosting:
* A gentle introduction to boosting algos: https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/
* CatBoost Algorithm resources: https://tech.yandex.com/catboost/
* Light GBM: http://lightgbm.readthedocs.io/en/latest/index.html
* Light GBM github: https://github.com/Microsoft/LightGBM
* XGBoost Conceptual Understanding of Algo: http://xgboost.readthedocs.io/en/latest/model.html
* XGBoost Site: http://xgboost.readthedocs.io/en/latest/
* Difference b/w lightgbm and xgboost: https://mlexplained.com/2018/01/05/lightgbm-and-xgboost-explained/

### Metrics for ML model evaluation
* metrics for model evaluation: http://scikit-learn.org/stable/modules/model_evaluation.html
* f1 score macro vs micro: https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult
* when to use f1 macro vs f1 micro:https://datascience.stackexchange.com/questions/36862/macro-or-micro-average-for-imbalanced-class-problems
* Top 15: https://www.machinelearningplus.com/machine-learning/evaluation-metrics-classification-models-r/
* ROC : https://stats.stackexchange.com/questions/105501/understanding-roc-curve
* ROC detailed analysis: http://mlwiki.org/index.php/ROC_Analysis

### Dimensionality Reduction

#### PCA
* variance in PCA explained: https://ro-che.info/articles/2017-12-11-pca-explained-variance
* PCA on large matrices: https://amedee.me/post/pca-large-matrices/
* radomizedSVD: http://alimanfoo.github.io/2015/09/28/fast-pca.html

#### t-SNE
* Laurens van der Maaten's (creator of t-SNE) website: https://lvdmaaten.github.io/tsne/
* Visualising data using t-SNE: Journal of Machine Learning Research: http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
* How to use t-SNE effectively: https://distill.pub/2016/misread-tsne/

#### ICA
* Stanford notes on ICA: http://cs229.stanford.edu/notes/cs229-notes11.pdf

### Model Stacking
* stacking: https://dkopczyk.quantee.co.uk/stacking/

### Distance Metrics
* Mahalnobis distance: https://www.machinelearningplus.com/statistics/mahalanobis-distance/
* Cosine similarity: https://www.machinelearningplus.com/nlp/cosine-similarity/
* 3 basic Distance Measurement in Text Mining: https://towardsdatascience.com/3-basic-distance-measurement-in-text-mining-5852becff1d7
* Word Mover’s Distance (WMD): https://towardsdatascience.com/word-distance-between-word-embeddings-cc3e9cf1d632
* WMD Tutorial: https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html
* Word Mover’s Distance in Python: https://vene.ro/blog/word-movers-distance-in-python.html
* probability distance metrics:https://markroxor.github.io/gensim/static/notebooks/distance_metrics.html

### Clustering
* assessing clustering tendancy:https://www.datanovia.com/en/lessons/assessing-clustering-tendency/
* Hopkins test for cluster tendency: https://matevzkunaver.wordpress.com/2017/06/20/hopkins-test-for-cluster-tendency/
* Clustering validation tests: http://www.sthda.com/english/wiki/print.php?id=241
* silhoutte method for cluster quality: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
* K Modes Clustering: https://shapeofdata.wordpress.com/2014/03/04/k-modes/
* Hirarchial Clustering: http://www.saedsayad.com/clustering_hierarchical.htm
* Linkage methods of hierarchical agglomerative cluster analysis (HAC): https://stats.stackexchange.com/questions/195446/choosing-the-right-linkage-method-for-hierarchical-clustering

### Handling imbalanced data set:
* how to handle imbalanced data with code: https://elitedatascience.com/imbalanced-classes
* good read: https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/
* concept read: https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
* imbalanced-learn library: https://github.com/scikit-learn-contrib/imbalanced-learn
* anomaly detection in python: https://www.datascience.com/blog/python-anomaly-detection
* scikit learn novelty and outlier detection: https://www.datascience.com/blog/python-anomaly-detection
* Imbalanced data handling tutorial in Python: https://blog.dominodatalab.com/imbalanced-datasets/
* imbalanced data sets (Good Read): https://www.svds.com/learning-imbalanced-classes/
* cost sensitive learning: https://machinelearningmastery.com/cost-sensitive-learning-for-imbalanced-classification/
* develop cost sensitive neural network: https://machinelearningmastery.com/cost-sensitive-neural-network-for-imbalanced-classification/

### Handling Skewed data:
* Top 3 methods for handling skewed data: https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45

### Multi-label Classification
* https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff

### Probability Callibration:
* https://scikit-learn.org/stable/modules/calibration.html

#### Sparse Matrix
* https://dziganto.github.io/Sparse-Matrices-For-Efficient-Machine-Learning/

### Design of Experiment
* https://machinelearningmastery.com/controlled-experiments-in-machine-learning/
* https://www.analyticsvidhya.com/blog/2015/10/guide-design-of-experiments-case-study/
* https://towardsdatascience.com/design-your-engineering-experiment-plan-with-a-simple-python-command-35a6ba52fa35
* https://pythonhosted.org/pyDOE/

### Model Explainibility/Interpretable ML
* ELI5 - TextExplainer: debugging black-box text classifiers: https://eli5.readthedocs.io/en/latest/tutorials/black-box-text-classifiers.html
* interpretable ML book: https://christophm.github.io/interpretable-ml-book/

### Anomaly Detection
* Note on anomaly detection: https://towardsdatascience.com/a-note-about-finding-anomalies-f9cedee38f0b
* Four Techniques for Anomaly detection: https://dzone.com/articles/four-techniques-for-outlier-detection-knime
* Novelty and Outlier Detection: https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-detection
* One Class SVM Anomaly detection: https://www.kaggle.com/amarnayak/once-class-svm-to-detect-anomaly
* PyOD for anomaly detection: https://github.com/yzhao062/Pyod#quick-start-for-outlier-detection
* text anomaly detection: https://arxiv.org/pdf/1701.01325.pdf
* Outlier Detection for Text Data: https://epubs.siam.org/doi/pdf/10.1137/1.9781611974973.55
* Text Anomaly Detection using Doc2Vec and cosine sim: https://medium.com/datadriveninvestor/unsupervised-outlier-detection-in-text-corpus-using-deep-learning-41d4284a04c8
*  https://github.com/avisheknag17/public_ml_models/blob/master/outlier_detection_in_movie_plots_ann/notebook/movie_plots_outlier_detector.ipynb

### XGBoost Installation:
* check you python version - by opening CMD and typing python -> ENTER
* Go to this link and search on XGBoost: https://www.lfd.uci.edu/~gohlke/pythonlibs/
* download the installable based on python version + Windows 32 or 64 bit, for example download xgboost-0.71-cp36-cp36m-win_amd64.whl for python version 3.6 and 64 bit machine.
* open cmd in downloaded location and run the following command: pip install xgboost-0.71-cp36-cp36m-win_amd64.whl

--------------------------------------------------------------------------------------------------------------------------------------

## Deep Learning:

### General
* The Perceptron Learning Algorithm and its Convergence: https://www.cse.iitb.ac.in/~shivaram/teaching/old/cs344+386-s2017/resources/classnote-1.pdf
* Deep Dive into Math Behind Deep Networks: https://towardsdatascience.com/https-medium-com-piotr-skalski92-deep-dive-into-deep-networks-math-17660bc376ba
* Recent Advances for a Better Understanding of Deep Learning − Part I: https://towardsdatascience.com/recent-advances-for-a-better-understanding-of-deep-learning-part-i-5ce34d1cc914
* using neural nets to recognize handwritten digits: http://neuralnetworksanddeeplearning.com/chap1.html
* Tinker with Neural Networks in browser: https://playground.tensorflow.org
* Dimensions and manifolds: https://datascience.stackexchange.com/questions/5694/dimensionality-and-manifold
* Play with Generative Adverserial Networks: https://poloclub.github.io/ganlab/
* Overfitting and how to prevent it: https://hackernoon.com/memorizing-is-not-learning-6-tricks-to-prevent-overfitting-in-machine-learning-820b091dc42
* 37 reasons for neural n/w not working properly: https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
* list of cost functions to be used with Gradient Descent: https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
* What is validation data used for in a Keras Sequential model? : https://stackoverflow.com/questions/46308374/what-is-validation-data-used-for-in-a-keras-sequential-model

### CNN and Image Processing:
* Why do we need to normalize the images before we put them into CNN? : https://stats.stackexchange.com/questions/185853/why-do-we-need-to-normalize-the-images-before-we-put-them-into-cnn
* Neural Network data type conversion - float from int? : https://datascience.stackexchange.com/questions/13636/neural-network-data-type-conversion-float-from-int
* Image Pre-processing (Keras): https://keras.io/preprocessing/image/
* Trick to prevent Overfitting: https://hackernoon.com/memorizing-is-not-learning-6-tricks-to-prevent-overfitting-in-machine-learning-820b091dc42
* keras callbacks: https://keras.io/callbacks/
* How to Check-Point Deep Learning Models in Keras: https://machinelearningmastery.com/check-point-deep-learning-models-keras/
* In Depth understanding of Convolutions: http://timdettmers.com/2015/03/26/convolution-deep-learning/
* friendly introduction to Cross Entropy: https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/
* Understanding Cross Entropy Loss - Visual Information Theory: http://timdettmers.com/2015/03/26/convolution-deep-learning/
* Papers on imp CNN architectures: https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html
* CNN using numpy: https://becominghuman.ai/only-numpy-implementing-convolutional-neural-network-using-numpy-deriving-forward-feed-and-back-458a5250d6e4
* Image Transformations Using OpenCV: https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
* List of Open Source **Medical Image Analysis** Softwares:http://www0.cs.ucl.ac.uk/opensource_mia_ws_2012/links.html
* Natural Images: https://stats.stackexchange.com/questions/25737/definition-of-natural-images-in-the-context-of-machine-learning
* ResNet - understanding the bottleneck unit: https://stats.stackexchange.com/questions/347280/regarding-the-understanding-of-bottleneck-unit-of-resnet
* Visual Question Answering: https://github.com/anujshah1003/VQA-Demo-GUI
* https://iamaaditya.github.io/2016/04/visual_question_answering_demo_notebook
* CNN+LSTM: https://machinelearningmastery.com/cnn-long-short-term-memory-networks/

#### 1D - CNNs
* Introduction to 1D CNNs: https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
* Why does each filter learn a different feature in CNN: https://www.quora.com/Why-does-each-filter-learn-different-features-in-a-convolutional-neural-network

#### Keras Embedding Layer
* Using Embedding Layer in Keras: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
* how does keras embedding layer work: https://stats.stackexchange.com/questions/270546/how-does-keras-embedding-layer-work

#### Keras generators
* A detailed example of how to use data generators with Keras: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

#### Saving Keras Models
* https://jovianlin.io/saving-loading-keras-models/

#### Clustering using DL
* unsupervised clustering in keras: https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/
* overview of DL based clustering methods: https://divamgupta.com/unsupervised-learning/2019/03/08/an-overview-of-deep-learning-based-clustering-techniques.html

#### Large Model Support usage in keras
* https://towardsdatascience.com/deep-learning-analysis-using-large-model-support-3a67a919255
* https://github.com/pierpaolo28/Artificial-Intelligence-Projects/blob/master/IBM%20Large%20Model%20Support/LargeModelSupport.ipynb

------------------------------------------------------------------------------------------------------------------

## Natural Language Processing (NLP) and Natural Language Understanding (NLU)

### General
* text tutorial: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
* text classification ref: https://scikit-learn.org/0.19/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
* Multiclass vs Multilabel vs Multioutput classification: https://scikit-learn.org/stable/modules/multiclass.html
* Out of core classification of text documents: https://scikit-learn.org/0.15/auto_examples/applications/plot_out_of_core_classification.html#example-applications-plot-out-of-core-classification-py
* Library for handling multilabel classification: http://scikit.ml/index.html
* NLTK Book: http://www.nltk.org/book/ 
* NLTK tutorial: https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/
* Extracting Text Meta Features: https://www.kaggle.com/shivamb/spacy-text-meta-features-knowledge-graphs
* jaccard distance using NLP: https://python.gotrained.com/nltk-edit-distance-jaccard-distance/#Jaccard_Distance
* Text Encoding Unicode: https://docs.python.org/3/howto/unicode.html
* Roudup of Python NLP libraries: https://nlpforhackers.io/libraries/
* Spacy Tutorial (AV): https://www.analyticsvidhya.com/blog/2017/04/natural-language-processing-made-easy-using-spacy-%E2%80%8Bin-python/
* SpaCy Tutorial: https://nlpforhackers.io/complete-guide-to-spacy/
* Generate text using word level neural language model: https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/
* Generate text using LSTM: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
* SIF embeddings implementation: https://www.kaggle.com/procode/sif-embeddings-got-69-accuracy
* **Ontology based text classification**: https://sci2lab.github.io/mehdi/icsc2014.pdf
* **fast text analysis using Vowpal Wabbit :** https://www.kaggle.com/kashnitsky/vowpal-wabbit-tutorial-blazingly-fast-learning

### Text Classification using Deep Learning:
* what kagglers are using for text classification: https://mlwhiz.com/blog/2018/12/17/text_classification/
* text CNN: https://www.kaggle.com/mlwhiz/learning-text-classification-textcnn/comments
* text pre-processing methods for DL: https://mlwhiz.com/blog/2019/01/17/deeplearning_nlp_preprocess/
* How to pre-process when using embeddings: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings
* A Layman guide to moving from Keras to Pytorch: https://mlwhiz.com/blog/2019/01/06/pytorch_keras_conversion/
* Toxic comments classification: https://www.kaggle.com/larryfreeman/toxic-comments-code-for-alexander-s-9872-model
* Text Blob: Simplified Text Processing: https://textblob.readthedocs.io/en/dev/

### Spacy resources
* text classification using spacy: https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/
* ml for text classification using spacy: https://towardsdatascience.com/machine-learning-for-text-classification-using-spacy-in-python-b276b4051a49
* tricks for using spacy at scale: https://towardsdatascience.com/a-couple-tricks-for-using-spacy-at-scale-54affd8326cf
*	Modified skip gram based on spacy dependency parser: https://medium.com/reputation-com-datascience-blog/keywords-extraction-with-ngram-and-modified-skip-gram-based-on-spacy-14e5625fce23
* SpaCy Tutorial: https://course.spacy.io/
* Spacy NLP faster using Cython: https://medium.com/huggingface/100-times-faster-natural-language-processing-in-python-ee32033bdced
* enable previously disabled pipes: https://stackoverflow.com/questions/53052687/spacy-enable-previous-disabled-pipes
* Spacy rule based matching - https://github.com/explosion/spaCy/blob/develop/website/docs/usage/rule-based-matching.md#combining-models-and-rules-models-rules
* Spacy Information extraction examples: https://spacy.io/usage/examples
* Training custom ner model in spacy: https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/
* BRAT: open source annotation tool: http://brat.nlplab.org/examples.html

### Topic Modelling:
* Topic modeling in gensim:  https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
* Topic modeling in sklearn (with NMF): https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
* sklearn: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
* What is topic coherence: https://rare-technologies.com/what-is-topic-coherence/
* Evaluation of Topic Modeling: Topic Coherence: https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/
* Exploring the Space of Topic Coherence Measures (paper): http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf
* Good Read - choosing topics using coherence measures in LDA,LSI,HDP etc. https://markroxor.github.io/gensim/static/notebooks/gensim_news_classification.html
* Dynamic Topic Models tutorial: https://markroxor.github.io/gensim/static/notebooks/ldaseqmodel.html
* Dynamic topic model google talk: https://www.youtube.com/watch?v=7BMsuyBPx90
* LDA using TF-IDF: https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

### Top2Vec
* https://github.com/ddangelov/Top2Vec
* paper: https://arxiv.org/pdf/2008.09470.pdf

### Text Summarization:
* https://becominghuman.ai/text-summarization-in-5-steps-using-nltk-65b21e352b65

### keyword-phrase extraction
* Intro to Automatic Keyphrase Extraction: https://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/
* Beyond bag of words: Using PyTextRank to find Phrases and Summarize text: https://medium.com/@aneesha/beyond-bag-of-words-using-pytextrank-to-find-phrases-and-summarize-text-f736fa3773c5
* NLP keyword extraction tutorial with RAKE and Maui: https://www.airpair.com/nlp/keyword-extraction-tutorial

### Gensim
* introduction to gensim: https://www.machinelearningplus.com/nlp/gensim-tutorial/
* usine soft cosine similarity in gensim: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb

### Natural Language Understanding
* https://web.stanford.edu/class/cs224u/

### NLG
* NLG using markovify: https://github.com/jsvine/markovify
* training bot to comment on current affairs: https://www.kaggle.com/aashita/training-a-bot-to-comment-on-current-affairs

### EVT
* Reducing Uncertainty in Document Classification with Extreme Value Theory: https://medium.com/cognigo/reducing-uncertainty-in-document-classification-with-extreme-value-theory-97508ebd76f

### Doc2Vec
* Doc2Vec : https://radimrehurek.com/gensim/models/doc2vec.html
* https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb
* Doc2Vec : https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
* How to use Doc2Vec as input to Keras model: https://stackoverflow.com/questions/50564928/how-to-use-sentence-vectors-from-doc2vec-in-keras-sequntial-model-for-sentence-s


#### Transfer learning in NLP:
* BERT: https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html
* BERT Research Paper: https://arxiv.org/abs/1810.04805 
* blog: http://jalammar.github.io/
* Blog for understanding ELMO and BERT: http://jalammar.github.io/illustrated-bert/
* ULMFIT tutorial: https://www.analyticsvidhya.com/blog/2018/11/tutorial-text-classification-ulmfit-fastai-library/
* DSSM: https://www.microsoft.com/en-us/research/project/dssm/


### Latest Language Models usage & applications
* When Not to Choose the Best NLP Model (Comparison of Elmo, USE, BERT & XLNET): https://blog.floydhub.com/when-the-best-nlp-model-is-not-the-best-choice/
* Using NLP to Automate Customer Support, Part Two (using Universal Sentence Encoding - USE): https://blog.floydhub.com/automate-customer-support-part-two/
* Paper Dissected: “XLNet: Generalized Autoregressive Pretraining for Language Understanding” Explained: https://mlexplained.com/2019/06/30/paper-dissected-xlnet-generalized-autoregressive-pretraining-for-language-understanding-explained/
* using USE + Keras: https://www.dlology.com/blog/keras-meets-universal-sentence-encoder-transfer-learning-for-text-data/
* NLP as service: Project Insight: https://github.com/abhimishra91/insight

-------------------------------------------------------------------------------------------

## Advanced Topics

### Knowledge Graphs
* Knowledge Graphs: https://web.stanford.edu/class/cs520/
* Domain specific knowledge graphs: https://www.springer.com/gp/book/9783030123741
* Ampligraph Open source Python library that predicts links between concepts in a knowledge graph - https://docs.ampligraph.org/en/1.3.1/index.html
* IBM technique: https://github.com/IBM/build-knowledge-base-with-domain-specific-documents/blob/master/README.md
* KG Intro: https://github.com/kramankishore/Knowledge-Graph-Intro
* Neo4j - Graph Data Science - GDS: https://neo4j.com/blog/announcing-neo4j-for-graph-data-science/
* Mining Knowledge Graph from text: https://kgtutorial.github.io/
* KG pipeline: https://towardsdatascience.com/conceptualizing-the-knowledge-graph-construction-pipeline-33edb25ab831
* Graph Data Base: Neo4j - https://neo4j.com/

* (Good Summary of resources 2020) https://dzone.com/articles/knowledge-graphs-power-scientific-research-and-bus

#### Deep Learning and Graphs:
* Graph Neural Networks an Overview: https://towardsdatascience.com/graph-neural-networks-an-overview-dfd363b6ef87
* Deep Graph Library: https://www.dgl.ai/
* Tensorflow: Neural Structured Learning: https://www.tensorflow.org/neural_structured_learning

#### Geometric Deep Learning & Graph learning.
* (Part 1) What is Geometric Deep Learning & its relation to Graph Learning: https://medium.com/@flawnsontong1/what-is-geometric-deep-learning-b2adb662d91d
* (Part 2) Everything you need to know about Graph Theory for Deep Learning: https://towardsdatascience.com/graph-theory-and-deep-learning-know-hows-6556b0e9891b
* (Part 3) Graph Embedding for Deep Learning: https://towardsdatascience.com/overview-of-deep-learning-on-graph-embeddings-4305c10ad4a4
* (Part 4) Graph Convolutional Networks for Geometric Deep Learning: https://towardsdatascience.com/graph-convolutional-networks-for-geometric-deep-learning-1faf17dee008
* Graph Embeddings Summary: https://towardsdatascience.com/graph-embeddings-the-summary-cc6075aba007

#### Text GCN:
* https://towardsdatascience.com/text-based-graph-convolutional-network-for-semi-supervised-bible-book-classification-c71f6f61ff0f
* https://arxiv.org/pdf/1809.05679.pdf
* https://github.com/plkmo/Bible_Text_GCN/blob/master/generate_train_test_datasets.py
* https://towardsdatascience.com/kegra-deep-learning-on-knowledge-graphs-with-keras-98e340488b93
* https://github.com/tkipf/relational-gcn
* https://arxiv.org/pdf/1703.06103.pdf
* https://twitter.com/fchollet/status/971121430531252224?lang=en
* https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/

#### RBF Neural Networks
* https://pythonmachinelearning.pro/using-neural-networks-for-regression-radial-basis-function-networks/
* http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
* rbf - custom keras layer: https://www.kaggle.com/residentmario/radial-basis-networks-and-custom-keras-layers
* research net discussion for unknown class: https://www.researchgate.net/post/How_to_determine_unknown_class_using_neural_network
* Titanic survivors using RBF: https://medium.com/datadriveninvestor/building-radial-basis-function-network-with-keras-estimating-survivors-of-titanic-a06c2359c5d9
* Custom RBF Keras Layer: https://github.com/PetraVidnerova/rbf_keras

### Probabilistic programming in tensorflow
* paper: https://arxiv.org/pdf/1711.10604.pdf
* https://medium.com/tensorflow/introducing-tensorflow-probability-dca4c304e245
* https://www.youtube.com/watch?v=BjUkL8DFH5Q&t=4s
* https://www.youtube.com/watch?v=ngFU7Rwl76g
* https://medium.com/tensorflow/industrial-ai-bhges-physics-based-probabilistic-deep-learning-using-tensorflow-probability-5f215c791863

## Docker
* https://www.analyticsvidhya.com/blog/2017/11/reproducible-data-science-docker-for-data-science/
* Docker for ML: https://pratos.github.io/2017-04-24/docker-for-data-science-part-1/
* conceptual - introduction to VM's and Docker: https://medium.freecodecamp.org/a-beginner-friendly-introduction-to-containers-vms-and-docker-79a9e3e119b
* lighter docker images: https://medium.com/swlh/build-fast-deploy-faster-creating-lighter-docker-images-11540ce0db14



* sentiment analysis using VADER: https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f

#### Active Learning
* How Active Learning can help you train your models with less Data: https://towardsdatascience.com/how-active-learning-can-help-you-train-your-models-with-less-data-389da8a5f7ea
* Active Learning Tutorial: https://towardsdatascience.com/active-learning-tutorial-57c3398e34d

#### Bayesian Optimization
* RoBo - Bayesian Optimization Framework: https://automl.github.io/RoBO/tutorials.html
* Implementing bayesian optimization from scratch: https://machinelearningmastery.com/what-is-bayesian-optimization/

#### Variational Autoencoder
* VAE an intutive explanation: https://hsaghir.github.io/data_science/denoising-vs-variational-autoencoder/
* text generation using VAE: https://nicgian.github.io/text-generation-vae/
* text VAE in keras: http://alexadam.ca/ml/2017/05/05/keras-vae.html
* tutorial on VAE: https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/
* From Autoencoders to Beta-VAE (Disentangled VAE): https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html
* Autoencoder - Image Compression: https://ai.googleblog.com/2016/09/image-compression-with-neural-networks.html
* video: https://www.youtube.com/watch?v=9zKuYvjFFS8




### Model Fracking and Concept Drift:
* https://towardsdatascience.com/fracking-features-in-machine-learning-b8247626e582
* github: https://github.com/saneshashank/Fracking-Features-in-Machine-Learning
* https://towardsdatascience.com/concept-drift-and-model-decay-in-machine-learning-a98a809ea8d4
* github: https://github.com/saneshashank/Concept-Drift-and-Model-Decay
 

### Spark

* Why should one use spark for ML: https://www.infoworld.com/article/3031690/analytics/why-you-should-use-spark-for-machine-learning.html

* Multi-Class Text Classification with PySpark: https://towardsdatascience.com/multi-class-text-classification-with-pyspark-7d78d022ed35

### Python
* Python OOP tutorial: https://www.youtube.com/watch?v=ZDa-Z5JzLYM
* OOPS illustrated using ML example: https://dziganto.github.io/classes/data%20science/linear%20regression/machine%20learning/object-oriented%20programming/python/Understanding-Object-Oriented-Programming-Through-Machine-Learning/
* vectorized string operations in Python(using pandas): https://jakevdp.github.io/PythonDataScienceHandbook/03.10-working-with-strings.html

* Use YouTube as a Free Screencast Recorder: https://www.youtube.com/watch?v=0i9C8GpRedc



* Parallel processing in Python: https://www.machinelearningplus.com/python/parallel-processing-python/

* https://thispointer.com/5-different-ways-to-read-a-file-line-by-line-in-python/
* https://www.learnpython.org/en/Map,_Filter,_Reduce
* cytoolz: https://pypi.org/project/cytoolz/
* https://cmdlinetips.com/2019/03/how-to-get-top-n-rows-with-in-each-group-in-pandas/

* intermediate, tips for python: https://book.pythontips.com/en/latest/index.html

##### writing better code for DS:
* https://towardsdatascience.com/how-a-simple-mix-of-object-oriented-programming-can-sharpen-your-deep-learning-prototype-19893bd969bd
* https://towardsdatascience.com/notes-on-software-construction-from-code-complete-8d2a8a959c69

#### Generators
* Jeff Knup's blog: 'Yield' and Generator Functions: https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/
* Corey Schafer (YouTube video): Generator functions: https://www.youtube.com/watch?v=bD05uGo_sVI
* Data streaming in Python: generators, iterators, iterables: https://rare-technologies.com/data-streaming-in-python-generators-iterators-iterables/

* Python inheritance, multiple inheritance & operator overloading: https://www.programiz.com/python-programming/inheritance
* Python Closure, Decorators & Python property: https://www.programiz.com/python-programming/closure
* Inheritance and Composition: A Python OOP Guide: https://realpython.com/inheritance-composition-python/
* Python’s super() considered super!: https://rhettinger.wordpress.com/2011/05/26/super-considered-super/
* Dunder or magic methods in Python: https://rszalski.github.io/magicmethods/
* intermediate python - https://realpython.com/intermediate-python/


### Reinforcement Learning
* Dynamic Programming: https://web.stanford.edu/class/cs97si/04-dynamic-programming.pdf
* When are Monte Carlo methods preferred over Temporal Difference methods: https://stats.stackexchange.com/questions/336974/when-are-monte-carlo-methods-preferred-over-temporal-difference-ones
* https://simoninithomas.github.io/Deep_reinforcement_learning_Course/#
* Off-Policy Monte Carlo Control: https://cs.wmich.edu/~trenary/files/cs5300/RLBook/node56.html
* https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

### PGM/ Causal Inference in ML
* Using Deep Neural Network Approximate Bayesian Network: https://arxiv.org/pdf/1801.00282.pdf
* A Comprehensive guide to Bayesian Convolutional Neural Network with Variational Inference https://arxiv.org/pdf/1901.02731.pdf
* Bayesian Methods for Hackers: https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
* Causal Inference survey of current areas of research: https://stats.stackexchange.com/questions/328602/what-are-some-current-research-areas-of-interest-in-machine-learning-and-causal
* Causal Inference in everyday ML: https://www.youtube.com/watch?v=HOgx_SBBzn0
* Causal Inference in everyday ML notebook: https://colab.research.google.com/drive/1rjjjA7teiZVHJCMTVD8KlZNu3EjS7Dmu#scrollTo=qsuGNCvtVbsr

### Open Datasets
* https://skymind.ai/wiki/open-datasets
* Chest X-ray data: https://www.kaggle.com/nih-chest-xrays

### Image Captioning
* https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8
* https://www.analyticsvidhya.com/blog/2018/04/solving-an-image-captioning-task-using-deep-learning/
* https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
* A Comprehensive Survey of Deep Learning for Image Captioning: https://arxiv.org/pdf/1810.04020.pdf

### Image Segmentation:
* Image Segmentation Keras : Implementation of Segnet, FCN, UNet, PSPNet and other models in Keras: https://github.com/divamgupta/image-segmentation-keras

### Customer Analytics
https://towardsdatascience.com/predictive-customer-analytics-4064d881b649 (part 1)

### Time Series
* time series analysis in python: https://www.machinelearningplus.com/time-series/time-series-analysis-python/

### ML Engineering & Architecture considerations
* https://towardsdatascience.com/putting-ml-in-production-i-using-apache-kafka-in-python-ce06b3a395c8
* https://towardsdatascience.com/putting-ml-in-production-ii-logging-and-monitoring-algorithms-91f174044e4e
* https://towardsdatascience.com/getting-started-with-mlflow-52eff8c09c61

* https://towardsdatascience.com/creating-a-solid-data-science-development-environment-60df14ce3a34
* DVC version control for ML projects: https://dvc.org/
* papermill :https://papermill.readthedocs.io/en/latest/
* Minimum Valuable Data Products: From 0 to data science pipeline: https://www.youtube.com/watch?v=UZg45yRTzwo

* FastAPI: https://github.com/tiangolo/fastapi

* Fullstack DS 1: https://medium.com/applied-data-science/the-full-stack-data-scientist-part-1-productionise-your-models-with-django-apis-7799b893ce7c
* Fullstack DS 2: https://medium.com/applied-data-science/the-full-stack-data-scientist-part-2-a-practical-introduction-to-docker-1ea932c89b57
* Fullstack DS 3: https://medium.com/applied-data-science/a-case-for-interpretable-data-science-using-lime-to-reduce-bias-e44f48a95f75
* Fullstack DS 4: https://medium.com/applied-data-science/the-full-stack-data-scientist-part-4-building-front-ends-in-streamlit-1c2903d4b1fe

* Fullstack Deep Learning: https://course.fullstackdeeplearning.com/

### Information Theory of Deep Learning
* https://lilianweng.github.io/lil-log/2017/09/28/anatomize-deep-learning-with-information-theory.html
* https://adityashrm21.github.io/Information-Theory-In-Deep-Learning/

###
* https://medium.com/bcggamma/an-ensemble-approach-to-large-scale-fuzzy-name-matching-b3e3fa124e3c

### Latent Aspect Ratio Analysis (LARA) for CSAT
* https://github.com/redris96/LARA
* CSAT Key topics extraction and contextual sentiment of users’ reviews: https://tech.goibibo.com/key-topics-extraction-and-contextual-sentiment-of-users-reviews-20e63c0fd7ca

### Data products
* Designing great data products:The Drivetrain Approach - https://www.oreilly.com/radar/drivetrain-approach-data-products/
* What do machine learning practitioners actually do? - https://www.fast.ai/2018/07/12/auto-ml-1/
* From Predictive Modelling to Optimization: The Next Frontier: https://www.youtube.com/watch?v=vYrWTDxoeGg

### Streamlit
* Repository of Awesome streamlit apps: https://awesome-streamlit.org/
* Summarizer and Named Entity Checker App with Streamlit and SpaCy: https://blog.jcharistech.com/2019/11/28/summarizer-and-named-entity-checker-app-with-streamlit-and-spacy/

### Bayesian Deep Learning
* https://emtiyaz.github.io/
* https://slideslive.com/38921489/deep-learning-with-bayesian-principles
* (paper) https://emtiyaz.github.io/papers/learning_from_bayes.pdf

### Kalman filters
* http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/

### Geometric deep learning
* http://geometricdeeplearning.com/

### Neuraxle 
* https://github.com/Neuraxio/Neuraxle
* https://www.neuraxle.org/stable/index.html
* https://github.com/Neuraxio/New-Empty-Python-Project-Base

### Math & Deep learning
* Aerin Kim is a senior research engineer at Microsoft and writes about topics related to
applied Math and deep learning: https://towardsdatascience.com/@aerinykim
* Matrices as tensor n/w diagrams: https://www.math3ma.com/blog/matrices-as-tensor-network-diagrams

### programming environments
* nbdev: https://www.fast.ai/2019/12/02/nbdev/
* https://github.com/fastai/nbdev/

### Dashboarding in jupyter notebook
* Dashboarding with Jupyter Notebooks, Voila and Widgets: https://www.youtube.com/watch?v=VtchVpoSdoQ
* Voila: https://voila.readthedocs.io/en/stable/index.html

### this missing CS semester
* https://missing.csail.mit.edu/

### writing your own blog:
* Blog: https://www.fast.ai/2020/01/16/fast_template/

### Quantum Physics & Quantum computing
* https://thenextweb.com/insights/2020/03/03/is-time-broken-physicists-filmed-a-quantum-measurement-but-the-moment-was-blurry/
* Tracking the Dynamics of an Ideal Quantum Measurement: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.124.080401
* Quantum NLP: https://medium.com/cambridge-quantum-computing/quantum-natural-language-processing-748d6f27b31d

### Information Retrieval, text search:
* 101 ways to solve search: https://www.youtube.com/watch?v=VHm6_uC4vxM
* BM25 - https://www.quora.com/How-does-BM25-work
* Python library for BM25 - https://pypi.org/project/rank-bm25/
* Building NLP text search system: https://towardsdatascience.com/building-a-sentence-embedding-index-with-fasttext-and-bm25-f07e7148d240
* https://www.kaggle.com/davidmezzetti/cord-19-analysis-with-sentence-embeddings
* Faiss is a library for efficient similarity search and clustering of dense vectors - https://github.com/facebookresearch/faiss
* MILVUS: Open source vector similarity search engine: https://milvus.io/
* Text similarity search with vector fields: https://www.elastic.co/blog/text-similarity-search-with-vectors-in-elasticsearch

* Softcosine similarity paper: http://www.scielo.org.mx/pdf/cys/v18n3/v18n3a7.pdf
* Text similarity using Softcosine similarity: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb
* Document Similarity Queries: https://radimrehurek.com/gensim/auto_examples/core/run_similarity_queries.html#sphx-glr-auto-examples-core-run-similarity-queries-py
* Document Distance metrics: https://radimrehurek.com/gensim/auto_examples/tutorials/run_distance_metrics.html#sphx-glr-auto-examples-tutorials-run-distance-metrics-py
* Similarity Queries with Annoy and Word2Vec: https://radimrehurek.com/gensim/auto_examples/tutorials/run_annoy.html#sphx-glr-auto-examples-tutorials-run-annoy-py
* similarities.docsim – Document similarity queries: https://radimrehurek.com/gensim/similarities/docsim.html

#### Longform Question Answering
* https://yjernite.github.io/lfqa.html

#### NLP spell correction:
* https://norvig.com/spell-correct.html
* https://nbviewer.jupyter.org/url/norvig.com/ipython/How%20to%20Do%20Things%20with%20Words.ipynb

### Multitask Learning (MTL)
* An Overview of Multi-Task Learning in Deep Neural Networks: https://ruder.io/multi-task/

### Weak Supervison & Semi-Supervised Learning
* Hazy Research Group (leading research group on Weak Supervision): http://hazyresearch.stanford.edu/
* Software 2.0 and Data Programming: Lessons Learned, and What’s Next: http://hazyresearch.stanford.edu/software2
* Weak Supervision: A New Programming Paradigm for Machine Learning: http://ai.stanford.edu/blog/weak-supervision/
* Introducing Snorkel: https://www.snorkel.org/blog/hello-world-v-0-9
* Snorkel Tutorial: https://www.snorkel.org/use-cases/
* Fonduer: Knowledge Base Construction from Richly Formatted Data (Intro): https://github.com/HazyResearch/fonduer-tutorials/tree/master/intro
* Snorkel (Paper): https://arxiv.org/pdf/1711.10160.pdf
* Training Classifiers with Natural Language Explanations (paper): https://arxiv.org/pdf/1805.03818.pdf
* Babble Labble: https://github.com/HazyResearch/babble
* Sippy Cup (tool for Semantic Parsing): https://github.com/wcmac/sippycup
* Data Programming:Creating Large Training Sets, Quickly (paper): https://arxiv.org/pdf/1605.07723.pdf
* Snuba: Automating Weak Supervision to Label Training Data (paper): http://www.vldb.org/pvldb/vol12/p223-varma.pdf
* Confident Learning: Estimating Uncertainty in Dataset Labels (paper): https://arxiv.org/abs/1911.00068
* CleanLab package: https://l7.curtisnorthcutt.com/cleanlab-python-package
* CleanLab (github): https://github.com/cgnorthcutt/cleanlab
* An Introduction to Confident Learning: https://l7.curtisnorthcutt.com/confident-learning
* Simplified Confident Learning tutorial: https://github.com/cgnorthcutt/cleanlab/blob/master/examples/simplifying_confident_learning_tutorial.ipynb
* HoloClean:Weakly Supervised Data Repairing: https://holoclean.github.io/gh-pages/blog/holoclean.html
* Introduction to Semantic Parsing (SippyCup): https://nbviewer.jupyter.org/github/wcmac/sippycup/blob/master/sippycup-unit-0.ipynb
* Using Snorkel for Multilabel classification: https://towardsdatascience.com/using-snorkel-for-multi-label-annotation-cc2aa217986a
* MixMatch: A Holistic Approach to Semi-Supervised Learning: https://arxiv.org/abs/1905.02249

### Future research topics
* self supervised learning - https://thenextweb.com/neural/2020/04/05/self-supervised-learning-is-the-future-of-ai-syndication/
* Software 2.0 and Data Programming: Lessons Learned, and What’s Next: http://hazyresearch.stanford.edu/software2
* NLP Transfer learning Thesis (Sebastian Ruder): https://ruder.io/thesis/neural_transfer_learning_for_nlp.pdf

#### Visual C++ build tools:
* https://visualstudio.microsoft.com/downloads/ --> choose --> Build Tools for Visual Studio 2019 --> in the installer choose build tools --> choose win 10 SDK only

### Chatbots
* Using voice to control a website with Amazon Alexa: https://blog.prototypr.io/using-voice-commands-to-control-a-website-with-amazon-echo-alexa-part-1-6-a35edbfef405
* How to build a State-of-the-Art Conversational AI with Transfer Learning: https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313
* How to build a State-of-the-Art Conversational AI with Transfer Learning (github link): https://github.com/huggingface/transfer-learning-conv-ai

## References (business):
* What is an ad impression: https://www.mediapost.com/publications/article/219695/the-definition-of-an-ad-impression.html
* ML in fraud detection: https://www.marutitech.com/machine-learning-fraud-detection/
* Customer Segmentation: http://analyticstraining.com/2011/cluster-analysis-for-business/
* Telecom churn customer model: https://parcusgroup.com/Telecom-Customer-Churn-Prediction-Models
* customer churn in mobile markets: https://arxiv.org/ftp/arxiv/papers/1607/1607.07792.pdf
* Survey text analytics: https://www.linkedin.com/pulse/how-choose-survey-text-analysis-software-discussion-draft-fitzgerald?trk=prof-post
* What’s Your Customer Effort Score?: https://www.gartner.com/smarterwithgartner/unveiling-the-new-and-improved-customer-effort-score/
* A Guide to Customer Satisfaction Metrics - NPS vs CSAT and CES: https://www.retently.com/blog/customer-satisfaction-metrics/
* Building an NLP solution to provide in-depth analysis of what your customers are thinking is a serious undertaking, and this guide helps you scope out the entire project: https://www.kdnuggets.com/2020/03/build-feedback-analysis-solution.html

### MWL - Made with ML
* Your one-stop platform to explore, learn and build all things
machine learning: https://madewithml.com/

