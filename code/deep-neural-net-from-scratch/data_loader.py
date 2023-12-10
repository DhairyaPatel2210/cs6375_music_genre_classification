import librosa
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder

# generating onehotencoding for the categories
dictonary = {}
def generate_onehotencoding(data):
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_values = encoder.fit_transform(np.array(data).reshape(-1, 1))
    
    for i in range(len(encoded_values)):
        dictonary[data[i]] = encoded_values[i]

#to get the mfcc features from the audio files
def process_all_files(parent_folder):
    inputs =  []
    outputs = []

    for genre_folder in os.listdir(parent_folder):
        genre_path = os.path.join(parent_folder, genre_folder)

        # Check if it's a directory
        if os.path.isdir(genre_path):
            print(f"Processing files in {genre_folder} folder:")

            for file_name in os.listdir(genre_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(genre_path, file_name)
                    # print(f"Processing file: {file_name}")

                    try:
                        # Extract MFCC features for each file
                        audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
                        mfcc_features = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc = 40).T, axis=0)

                        inputs.append(mfcc_features)
                        outputs.append(dictonary[genre_folder])

                    except:
                        continue

    return inputs, outputs

categories = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
generate_onehotencoding(categories)

inputs, outputs = process_all_files("D:\\UTD\\2.Machine Learning(6375)\\cs6375_music_genre_classification\\Data\\Data\\genres_original")
train_x,train_y = np.array(inputs[:800]).T,np.array(outputs[:800]).T
test_x, test_y = np.array(inputs[800:998]).T, np.array(outputs[800:998]).T

