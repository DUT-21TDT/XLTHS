import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import glob
from sklearn import neighbors

config = {
    "train_path":"signals/NguyenAmHuanLuyen-16k",
    "valid_path":"signals/NguyenAmKiemThu-16k",
    "nffts": [512, 1024, 2048],
    "am": ["a", "e", "i", "o", "u"],
    "mfcc": 13,
    "n_neighbors": 10,
}

def euclidean_distance(a, b):
    return abs(a - b)

def load_audio(filename):
    try:
        return librosa.load(filename, sr=None)
    except Exception as e:
        print(f"Cannot load '{filename}': {e}")
        return None

def extract_mfcc(y, sr=22050, n_mfcc=10, n_fft=1024):
    try:
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    except Exception as e:
        print(f"Cannot extract MFCC: {e}")
        return None

def getFeaturesVector(file_path, n_mfcc = 13, n_fft=1024):
  # ==== get label =====================
  filename = os.path.basename(file_path)
  # ====================================

  # === get human say ==================
  human = file_path.split("/")[-2]
  # ====================================

  # Load audio file
  y, fs = librosa.load(file_path, sr=None)

  # Calculate Short-Time Energy (STE)
  window_size = 0.03  # Size of the window (30 ms)
  overlap = 0.5  # Overlap between windows (50%)
  ste_threshold = 0.01  # STE threshold for classifying sound and silence

  # Calculate window parameters
  window_length = int(window_size * fs)
  overlap_length = int(overlap * window_length)

  # Calculate STE for each window
  ste = np.zeros_like(y)
  for i in range(0, len(y) - window_length, window_length - overlap_length):
      window = y[i:i + window_length]
      ste[i:i + window_length] = np.sum(window ** 2)

  # Classify sound and silence
  sound_segments = y[ste > ste_threshold]
  silence_segments = y[ste <= ste_threshold]

  # Create time vectors
  time_sound = np.arange(len(sound_segments)) / fs
  time_silence = np.arange(len(silence_segments)) / fs

  # ==========================
  offset = sound_segments.shape[0]
  # x1 = sound_segments[int(offset * 1/3): int(offset * 2/3)]
  x1 = sound_segments
  # t = time_sound[int(offset * 1/3): int(offset * 2/3)]
  t = time_sound
  time_step = 0.03 # = 30ms
  m = len(np.arange(t[0], t[-1], time_step))

  features = np.zeros((n_mfcc, n_mfcc))
  for (i) in range(m):
    frame = x1[int(x1.shape[0] * i/m ): int(x1.shape[0] * (i + 1)/m)]
    features += extract_mfcc(frame, fs, n_mfcc, n_fft)
  return features / m

for n_fft in config["nffts"]:
  data = {
    "a":[],
    "e":[],
    "i":[],
    "o":[],
    "u":[],
  }

  X, Y = [], []
  for path in glob.glob(config["train_path"] + "/**/*.wav"):
    label = os.path.basename(path).split(".")[0]
    features = getFeaturesVector(path, n_mfcc=config["mfcc"], n_fft = n_fft)
    X.append(features)
    Y.append(label)

  X = np.array(X)
  nsamples, nx, ny = X.shape
  X = X.reshape((nsamples,nx*ny))

  # ================
  clf = neighbors.KNeighborsClassifier(n_neighbors = config["n_neighbors"])
  clf.fit(X, Y)


  X_valid = []
  Y_valid = []
  for path in glob.glob(config["valid_path"]+"/**/*.wav"):
      label = os.path.basename(path).split(".")[0]
      features = getFeaturesVector(path, n_mfcc=config["mfcc"], n_fft = n_fft)
      X_valid.append(features)
      Y_valid.append(label)

  nsamples, nx, ny = np.array(X_valid).shape
  X_valid = np.array(X_valid).reshape((nsamples,nx*ny))

  print("n_fft  = ", n_fft)
  from sklearn.metrics import classification_report, accuracy_score
  print(classification_report(Y_valid, clf.predict(X_valid)))
  print(accuracy_score(Y_valid, clf.predict(X_valid)))

  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  cm = confusion_matrix(Y_valid, clf.predict(X_valid), labels=config["am"])
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config["am"])
  # disp.plot()
  # plt.show()
  fig, ax = plt.subplots(figsize=(10,10))
  disp.plot(ax=ax)
  fig.set_figwidth(3)
  fig.set_figheight(3)
  fig.show()
  plt.show()