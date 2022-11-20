
# Speech Emotion Recognition

Speech Emotion Recognition, abbreviated as SER, is the act of attempting to recognize human emotion and affective states from speech. This is capitalizing on the fact that voice often reflects underlying emotion through tone and pitch. This is also the phenomenon that animals like dogs and horses employ to be able to understand human emotion.



## DataSet

In this we used the RAVDESS dataset; this is the Ryerson Audio-Visual Database of Emotional Speech and Song dataset. This dataset has 7356 files rated by 247 individuals 10 times on emotional validity, intensity, and genuineness. The entire dataset is 24.8GB from 24 actors, but we’ve lowered the sample rate on all the files.

you can download the dataset from : https://drive.google.com/file/d/1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7/view
## Library used -LIBROSA
Librosa is a Python package for music and audio analysis. Librosa is basically used when we work with audio data.

It provides the building blocks necessary to create the music information retrieval systems. Librosa helps to visualize the audio signals and also do the feature extractions in it using different signal processing techniques.

By using this library we extrcted three main information from audio to build the model
* mfcc: Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound
* chroma: Pertains to the 12 different pitch classes
* mel: Mel Spectrogram Frequency

## Requirements
* python
* sklearn
* librosa (https://librosa.org/doc/latest/index.html)
* soundfile (https://pysoundfile.readthedocs.io/en/latest/)
* glob
* pickle
## Approach
MLPClassifier is used, this is a Multi-layer Perceptron Classifier; it optimizes the log-loss function using LBFGS or stochastic gradient descent. Unlike SVM or Naive Bayes, the MLPClassifier has an internal neural network for the purpose of classification.
The model delivered an accuracy of 81.25%
## Deployment

Deployment is done using Streamlit
( refer : https://docs.streamlit.io/ )

To deploy this project run

streamlit run speech_recog_deploy.py 

