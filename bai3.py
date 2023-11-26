import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import glob

config = {
  "train_path":"signals/NguyenAmHuanLuyen-16k",
  "valid_path":"signals/NguyenAmKiemThu-16k",
  "nffts": [512, 1024, 2048],
  "am": ["a", "e", "i", "o", "u"],
  "mfcc": 13
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
  m = {
    "a":np.zeros((config["mfcc"], config["mfcc"])),
    "e":np.zeros((config["mfcc"], config["mfcc"])),
    "i":np.zeros((config["mfcc"], config["mfcc"])),
    "o":np.zeros((config["mfcc"], config["mfcc"])),
    "u":np.zeros((config["mfcc"], config["mfcc"])),
  }

  for path in glob.glob(config["train_path"] + "/**/*.wav"):
    label = os.path.basename(path).split(".")[0]
    fmc = getFeaturesVector(path, n_mfcc=config["mfcc"], n_fft=n_fft)
    m[label] += fmc

  for a in config["am"]:
    m[a] /= len(os.listdir(config["train_path"]))

  y_true = []
  y_pred = []

  for fname in glob.glob(config["valid_path"] + "/**/*.wav"):
    label = os.path.basename(fname).split(".")[0]
    yhat = getFeaturesVector(fname, n_mfcc=config["mfcc"], n_fft=n_fft)
    pred = []
    for a in config["am"]:
      s = 0
      for (i) in range(config["mfcc"]):
        for(j) in range(config["mfcc"]):
          s += euclidean_distance(m[a][i][j], yhat[i][j])
      pred.append(s)

    y_pred.append(config["am"][np.argmin(pred)])
    y_true.append(label)

  plt.rcParams["figure.figsize"] = (20, 5)
  from matplotlib import cm
  fig, axs = plt.subplots(1, 5)
  fig.figsize = (20,30)
  fig.suptitle("Ảnh mfcc(m = {}) đặc trưng FFT (N = {})".format(config["mfcc"],n_fft), fontsize=16)
  axs[0].set_title("/a/")
  axs[0].imshow(m["a"])

  axs[1].set_title("/e/")
  axs[1].imshow(m["e"])

  axs[2].set_title("/i/")
  axs[2].imshow(m["i"])

  axs[3].set_title("/o/")
  axs[3].imshow(m["o"])

  axs[4].set_title("/u/")
  axs[4].imshow(m["u"])
  fig.tight_layout()
  fig.show()

  #==========================
  fig, axs = plt.subplots(1, 5)
  fig.figsize = (20,30)
  fig.suptitle("Vector đặc trưng mfcc (m={}) với FFT (N = {})".format(config["mfcc"], n_fft), fontsize=16)
  axs[0].set_title("/a/")
  axs[0].plot(m["a"][:, 0])

  axs[1].set_title("/e/")
  axs[1].plot(m["e"][:, 0])

  axs[2].set_title("/i/")
  axs[2].plot(m["i"][:, 0])

  axs[3].set_title("/o/")
  axs[3].plot(m["o"][:, 0])

  axs[4].set_title("/u/")
  axs[4].plot(m["u"][:, 0])

  fig.tight_layout()
  fig.show()

  # ==========================
  fig, axs = plt.subplots(1)
  fig.suptitle("Vector đặc trưng mfcc (m={}) với FFT (N = {})".format(config["mfcc"], n_fft), fontsize=16)
  axs.plot(m["a"][:, 0], label = "/a/")
  axs.plot(m["e"][:, 0], label = "/e/")
  axs.plot(m["i"][:, 0], label = "/i/")
  axs.plot(m["o"][:, 0], label = "/o/")
  axs.plot(m["u"][:, 0], label = "/u/")
  axs.legend()
  fig.show()

  from sklearn.metrics import classification_report, accuracy_score
  print(classification_report(y_true, y_pred))
  print(accuracy_score(y_true, y_pred))

  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  cm = confusion_matrix(y_true, y_pred, labels=config["am"])

  plt.rcParams["figure.figsize"] = (10, 8)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config["am"])
  fig, ax = plt.subplots(figsize=(10,10))
  disp.plot(ax=ax)
  fig.set_figwidth(3)
  fig.set_figheight(3)
  fig.show()
  plt.show()