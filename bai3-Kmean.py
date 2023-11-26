import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import glob
from sklearn.cluster import KMeans

config = {
    "train_path":"signals/NguyenAmHuanLuyen-16k",
    "valid_path":"signals/NguyenAmKiemThu-16k",
    "nffts": [512, 1024, 2048],
    "am": ["a", "e", "i", "o", "u"],
    "mfcc": 13,
    "K":[2, 3, 4, 5]
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


def modelPredictFeature2(features, am_key, K, n_mfcc, data, m):
  cluster_dict = {i: np.where(m[am_key].labels_ == i)[0] for i in range(m[am_key].n_clusters)}
  # cluster_id = m[am_key].predict([features])[0]

  features_means = []
  for (id) in range(K):
    features_mean = np.zeros(n_mfcc * n_mfcc)
    for human_index in cluster_dict[id]:
      features_mean += data[am_key][human_index]

    features_mean /= cluster_dict[id].shape[0]
    features_means.append(features_mean)
  return features_means


for K in config["K"]:
  for n_fft in config["nffts"]:
    data = {
      "a":[],
      "e":[],
      "i":[],
      "o":[],
      "u":[],
    }

    for path in glob.glob(config["train_path"] + "/**/*.wav"):
      label = os.path.basename(path).split(".")[0]
      features = getFeaturesVector(path, n_mfcc=config["mfcc"], n_fft=n_fft)
      data[label].append(features)

    for a in config["am"]:
      nsamples, nx, ny = np.array(data[a]).shape
      data[a] = np.array(data[a]).reshape((nsamples,nx*ny))
    # ================
    m = {
      "a":KMeans(n_clusters=K),
      "e":KMeans(n_clusters=K),
      "i":KMeans(n_clusters=K),
      "o":KMeans(n_clusters=K),
      "u":KMeans(n_clusters=K),
    }

    for a in config["am"]:
      m[a].fit(data[a])

    plt.rcParams["figure.figsize"] = (20, 5)
    fig, axs = plt.subplots(1, 5)
    fig.figsize = (20,30)
    fig.suptitle("Vector đặc trưng mfcc (m={}) với FFT (N = {}) - Phân cụm K = {}".format(config["mfcc"], n_fft, K), fontsize=16)
    i_am_key = 0
    for am_key in config["am"]:
      fmean = modelPredictFeature2(features, am_key, K, config["mfcc"], data, m)
      for id in range(K):
        axs[i_am_key].set_title(f"/{am_key}/")
        axs[i_am_key].plot(fmean[id], label = f"/{am_key}/ - cụm {id}")
      i_am_key = i_am_key + 1

    fig.tight_layout()
    fig.show()

    y_pred = []
    y_true = []

    for path in glob.glob(config["valid_path"] + "/**/*.wav"):
      label = os.path.basename(path).split(".")[0]
      features = getFeaturesVector(path, n_mfcc=config["mfcc"], n_fft=n_fft)
      features = np.array(features).reshape((config["mfcc"]*config["mfcc"]))

      pred = []
      for am_key in config["am"]:
        fmean = modelPredictFeature2(features, am_key, K, config["mfcc"], data, m)
        smean = []
        for id in range(K):
          s = 0
          for (i) in range(config["mfcc"] * config["mfcc"]):
            s += euclidean_distance(fmean[id][i], features[i])
          smean.append(s)
        pred.append(np.min(smean))

      y_pred.append(config["am"][np.argmin(pred)])
      y_true.append(label)

    print("n_fft  = ", n_fft, " K = ", K)
    from sklearn.metrics import classification_report, accuracy_score
    print(classification_report(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred, labels=config["am"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config["am"])
    # disp.plot()
    # plt.show()

    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(ax=ax)
    fig.set_figwidth(3)
    fig.set_figheight(3)
    fig.show()
    plt.show()