import librosa
import numpy as np
import os
import math
from sklearn.preprocessing import OneHotEncoder

# figuring out to get the mfcc features from the audio files
def process_all_files(parent_folder):
    inputs =   []
    outputs = []

    for genre_folder in os.listdir(parent_folder):
        genre_path = os.path.join(parent_folder, genre_folder)

        # Check if it's a directory
        if os.path.isdir(genre_path):
            print(f"Processing files in {genre_folder} folder:")

            for file_name in os.listdir(genre_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(genre_path, file_name)
                    print(f"Processing file: {file_name}")

                    try:
                        # Extract MFCC features for each file
                        audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
                        mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc = 40)
                        mfcc_scaled_features = np.mean(mfcc_features.T, axis=0)

                        inputs.append(mfcc_scaled_features)
                        outputs.append(genre_folder)

                    except:
                        print(file_path + "is currepted")

    print(len(inputs), len(inputs[1]))

def preprocess(dataset_path, num_mfcc=40, n_fft=2048, hop_length=512, num_segment=10):
  
  data = {"labels": [], "mfcc": []}
  sample_rate = 22050
  samples_per_segment = int(sample_rate*30/num_segment)
  for label_idx, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
      print("hi")
      if dirpath == dataset_path:
          continue
      for f in sorted(filenames):
          if not f.endswith('.wav'):
              continue
          file_path = str(str(dirpath).split('\\')[-1]) + "/" + str(f)
          file_path = dataset_path + "//" + file_path
          try:
              y, sr = librosa.load(file_path, sr = sample_rate)
          except:
              print("Track Name", file_path)
              print(1)
              continue
          for n in range(num_segment):
            start_sample = samples_per_segment * n
            finish_sample = start_sample + samples_per_segment
            mfcc = librosa.feature.mfcc(y=y[start_sample:finish_sample], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            #   mfcc = librosa.feature.mfcc(y[samples_per_segment*n: samples_per_segment* (n+1)], sample_rate, n_mfcc = num_mfcc, n_fft = n_fft, hop_length = hop_length)
            mfcc = mfcc.T
            if len(mfcc) == math.ceil(samples_per_segment/hop_length):
                data["mfcc"].append(mfcc.tolist())
                data["labels"].append(label_idx-1)
  return data

# process_all_files("D:\\UTD\\2.Machine Learning(6375)\\cs6375_music_genre_classification\\Data\\Data\\genres_original")
mfcc = preprocess("D:\\UTD\\2.Machine Learning(6375)\\cs6375_music_genre_classification\\Data\\Data\\genres_original")
print(len(mfcc["labels"]), len(mfcc["mfcc"]))