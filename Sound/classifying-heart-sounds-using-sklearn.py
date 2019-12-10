import os
import librosa
import numpy as np
from sklearn import svm


def decodeFolder(category):
	print("Starting decoding folder "+category+" ...")
	listOfFiles = os.listdir(category)
	arrays_sound = np.empty((0,193))
	for file in listOfFiles:
		filename = os.path.join(category,file)
		features_sound = extract_feature(filename)
		print(len(features_sound))
		arrays_sound = np.vstack((arrays_sound,features_sound))
	return arrays_sound

def extract_feature(file_name):
	print("Extracting "+file_name+" ...")
	X, sample_rate = librosa.load(file_name)
	stft = np.abs(librosa.stft(X))
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
	return np.hstack((mfccs,chroma,mel,contrast,tonnetz))

#train data
normal_sounds = decodeFolder("normal")
normal_labels = [0 for items in normal_sounds]
murmur_sounds = decodeFolder("murmur")
murmur_labels = [1 for items in murmur_sounds]
train_sounds = np.concatenate((normal_sounds, murmur_sounds))
train_labels = np.concatenate((normal_labels, murmur_labels))

#test_data
test_sound = decodeFolder("test")

clf =svm.SVC()
clf.fit(train_sounds,train_labels)
print("training done")
print(clf.predict(test_sound))