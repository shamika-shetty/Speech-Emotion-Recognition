import librosa
import os  
import soundfile
import glob, pickle
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import streamlit as st

st.title('SPEECH EMOTION RECOGNITION')

st.header('UPLOAD THE AUDIO CLIP TO DETECT EMOTIONS ')

	
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result	
		

emotions={
'01':'neutral',
'02':'calm',
'03':'happy',
'04':'sad',
'05':'angry',
'06':'fearful',
'07':'disgust',
'08':'surprised'	
}


def user_input_features():
	x= []
	audio = st.file_uploader("Upload file ",type = 'wav')
	if audio:
		feature=extract_feature(audio,mfcc=True,chroma=True,mel=True)
		x.append(feature)
		return np.array(x)
	else:
		return -1

df = user_input_features()

	
	
def load_data(test_size=0.25):
	x,y=[],[]
	for file in glob.glob("C:\\Users\\Shamikarani\\Downloads\\Speech\\Actor_*\\*.wav"):
		file_name=os.path.basename(file)
		emotion=emotions[file_name.split("-")[2]]
		if emotion not in ['calm', 'happy','fearful','angry']:
			continue
		feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
		x.append(feature)
		y.append(emotion)
	return [x,y]

X,Y  = load_data(test_size=0.25)

model=MLPClassifier(alpha=0.03, batch_size=350, epsilon=1e-09, hidden_layer_sizes=(330,), learning_rate='adaptive', max_iter=550,activation='relu')

model.fit(X,Y)

if df.size == 0:
	st.write("Please upload the audio")
else :
	dff = np.reshape(df,(1,-1))
	pred=model.predict(dff)
	st.write(" PREDICTED EMOTION IS : ", pred[0])

