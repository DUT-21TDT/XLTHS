import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import glob

config = {
  "train_path":"signals/NguyenAmHuanLuyen-16k",
  "valid_path":"signals/NguyenAmKiemThu-16k",
  "nffts": [512, 1024, 2048],
  "am": ["a", "e", "i", "o", "u"]
}

def SingleFrameFFT(frame, nFFT):
   return np.fft.fft(frame,nFFT)


def euclidean_distance(a, b):
    return abs(a - b)

def getFeaturesVector(file_path, nFFT):
  # ==== get label =====================
  filename = os.path.basename(file_path)
  # ====================================

  # === get human say ==================
  human = file_path.split("/")[-2]
  # ====================================

  # Load audio file
  y, fs = librosa.load(file_path, sr = None)

  # Calculate Short-Time Energy (STE)
  window_size = 0.03  # Size of the window (30 ms)
  overlap = 0.0  # Overlap between windows (50%)
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
  x1 = sound_segments[int(offset * 1/3): int(offset * 2/3)]
  t = time_sound[int(offset * 1/3): int(offset * 2/3)]
  time_step = 0.03 # = 30ms
  # m = len(np.arange(t[0], t[-1], time_step))
  m = 2

  features = np.zeros(nFFT, dtype=np.complex128)
  for (i) in range(m):
    frame = x1[int(x1.shape[0] * i/m ): int(x1.shape[0] * (i + 1)/m)]
    features += SingleFrameFFT(frame, nFFT = nFFT)

  return features / m


for NFFT in config["nffts"]:
  m = {
    "a":np.zeros(NFFT, dtype=np.complex128),
    "e":np.zeros(NFFT, dtype=np.complex128),
    "i":np.zeros(NFFT, dtype=np.complex128),
    "o":np.zeros(NFFT, dtype=np.complex128),
    "u":np.zeros(NFFT, dtype=np.complex128),
  }

  for path in glob.glob(config["train_path"] + "/**/*.wav"):
    label = os.path.basename(path).split(".")[0]
    features = getFeaturesVector(path, nFFT = NFFT)
    m[label] += features

  for a in config["am"]:
    m[a] /= len(os.listdir(config["train_path"]))

  y_true = []
  y_pred = []

  for fname in glob.glob(config["valid_path"] + "/**/*.wav"):
    label = os.path.basename(fname).split(".")[0]
    yhat = getFeaturesVector(fname, nFFT = NFFT)
    pred = []
    for a in config["am"]:
      s = 0
      for (i) in range(NFFT):
        s += euclidean_distance(abs(m[a][i]), abs(yhat[i]))
      pred.append(s)

    y_pred.append(config["am"][np.argmin(pred)])
    y_true.append(label)
  
  plt.rcParams["figure.figsize"] = (20, 5)
  from matplotlib import cm
  fig, axs = plt.subplots(1, 5)
  fig.figsize = (20,30)
  fig.suptitle("Vector đặc trưng với FFT (N = {})".format(NFFT), fontsize=16)
  for i in range(len(config["am"])):
    axs[i].set_title("/"+config["am"][i]+"/")
    axs[i].plot(20*np.log10(abs(m[config["am"][i]][:len(m[config["am"][i]])//2])))
    axs[i].set_ylim([-100, 20])
    axs[i].set_ylabel("dB")

  fig.tight_layout()
  fig.show()

  fig, axs = plt.subplots(1)
  for a in config["am"]:
    axs.plot(20*np.log10(abs(m[a][:len(m[a])//2])), label = "/"+a+"/")

  axs.set_ylabel("dB")
  axs.legend()
  fig.show()

  from sklearn.metrics import classification_report, accuracy_score
  print(classification_report(y_true, y_pred))
  print(accuracy_score(y_true, y_pred))

  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  cm = confusion_matrix(y_true, y_pred, labels=config["am"])
  
  plt.rcParams["figure.figsize"] = (10, 8)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config["am"])
  disp.plot()
  plt.show()
