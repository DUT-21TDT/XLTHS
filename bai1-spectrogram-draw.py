import os
import glob
import scipy
import numpy as np

config = {
  "train_path":"signals/NguyenAmHuanLuyen-16k",
  "valid_path":"signals/NguyenAmKiemThu-16k",
}

def getWavFileContents(pathFile):
    sample_rate, data = scipy.io.wavfile.read(pathFile)
    data = np.array(data, dtype=float)
    length = data.shape[0] / sample_rate
    return sample_rate, data, length

import os
import glob
import scipy
import numpy as np
import librosa

class SignalDataset():
  def __init__(self, data_directory):
    self.audio_paths = self.load(data_directory)

  def __getitem__(self, idx):
    audio_path = self.audio_paths[idx]

    # ==== get label =====================
    filename = os.path.basename(audio_path)
    label = filename.split(".")[0]
    # ====================================

    # ==== read data =====================
    y, fs = librosa.load(audio_path, sr = None)
    # ====================================

    # === get human say ==================
    human = self.audio_paths[idx].split("/")[-2]
    # ====================================

    return (fs, y, y.shape[0] / fs), human , label

  def __len__(self):
    return len(self.audio_paths)

  def load(self, data_directory):
    audio_paths = []
    for audio_path in glob.glob(data_directory+"/**/*.wav"):
        audio_paths.append(audio_path)
    return audio_paths
  

print(os.listdir(config["train_path"]))

train_data = SignalDataset(config["train_path"])
valid_data = SignalDataset(config["valid_path"])

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)

human_nums = 4
for i in range (human_nums):
  fig, axs = plt.subplots(5)
  fig.figsize = (20,30)

  for (j) in range(5):
    (sample_rate, data, length), human,label = train_data[j + i * 5]
    time = np.linspace(0., length, data.shape[0])
    axs[j].set_title("{} - {}".format(human, label))
    axs[j].specgram(data, Fs = sample_rate)
    axs[j].set(xlabel='Time', ylabel='Frequency')

  fig.tight_layout()
  fig.show()

  plt.show()