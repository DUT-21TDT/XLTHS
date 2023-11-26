import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import lfilter
import glob


config = {
  "train_path":"signals/NguyenAmHuanLuyen-16k",
  "valid_path":"signals/NguyenAmKiemThu-16k",
}

def getSilenceAndSound(file_path):

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

  return fs, sound_segments, time_sound, silence_segments, time_silence

# This function is copied directly from https://github.com/cournape/talkbox/blob/master/scikits/talkbox/linpred/py_lpc.py
# Copyright (c) 2008 Cournapeau David
# (MIT licensed)
def levinson_1d(r, order):
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.

    Parameters
    ---------
    r : array-like
        input array to invert (since the matrix is symmetric Toeplitz, the
        corresponding pxp matrix is defined by p items only). Generally the
        autocorrelation of the signal for linear prediction coefficients
        estimation. The first item must be a non zero real.

    Notes
    ----
    This implementation is in python, hence unsuitable for any serious
    computation. Use it as educational and reference purpose only.

    Levinson is a well-known algorithm to solve the Hermitian toeplitz
    equation:

                       _          _
        -R[1] = R[0]   R[1]   ... R[p-1]    a[1]
         :      :      :          :      *  :
         :      :      :          _      *  :
        -R[p] = R[p-1] R[p-2] ... R[0]      a[p]
                       _
    with respect to a (  is the complex conjugate). Using the special symmetry
    in the matrix, the inversion can be done in O(p^2) instead of O(p^3).
    """
    r = np.atleast_1d(r)
    if r.ndim > 1:
        raise ValueError("Only rank 1 are supported for now.")

    n = r.size
    if n < 1:
        raise ValueError("Cannot operate on empty array !")
    elif order > n - 1:
        raise ValueError("Order should be <= size-1")

    if not np.isreal(r[0]):
        raise ValueError("First item of input must be real.")
    elif not np.isfinite(1/r[0]):
        raise ValueError("First item should be != 0")

    # Estimated coefficients
    a = np.empty(order+1, r.dtype)
    # temporary array
    t = np.empty(order+1, r.dtype)
    # Reflection coefficients
    k = np.empty(order, r.dtype)

    a[0] = 1.
    e = r[0]

    for i in range(1, order+1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i-j]
        k[i-1] = -acc / e
        a[i] = k[i-1]

        for j in range(order):
            t[j] = a[j]

        for j in range(1, i):
            a[j] += k[i-1] * np.conj(t[i-j])

        e *= 1 - k[i-1] * np.conj(k[i-1])

    return a, e, k

def lpc(wave, order):
    """Compute LPC of the waveform.
    a: the LPC coefficients
    e: the total error
    k: the reflection coefficients

    Typically only a is required.
    """
    # only use right half of autocorrelation, normalised by total length
    autocorr = scipy.signal.correlate(wave, wave)[len(wave)-1:]/len(wave)
    a, e, k  = levinson_1d(autocorr, order)
    return a,e,k

def formant_estimation(audio_path):
  fs, y, _, _, _ = getSilenceAndSound(audio_path)
  dt = 1/fs;
  I0 = round(0.0/dt);
  Iend = round(0.75/dt);
  x = y[I0:Iend];
  x1 = x * np.hamming(x.shape[0]);

  preemph = [1, 0.63];
  x1 = lfilter(preemph, 1, x1);

  A = lpc(x1, order=12)[0]
  rts = np.roots(A)
  rts = rts[np.imag(rts) >= 0]
  angz = np.arctan2(np.imag(rts), np.real(rts))
  frqs = angz * fs / (2 *  np.pi)

  bw = -1/2*(fs/(2*np.pi))*np.log(abs(rts))
  formants = []
  for i in range(len(bw)):
    if (frqs[i] > 90) and (bw[i] < 400):
      formants.append(frqs[i])

  formants.sort()

  return formants

datalist = {
    "a":{"f1": [], "f2": [], "f3": []},
    "e":{"f1": [], "f2": [], "f3": []},
    "i":{"f1": [], "f2": [], "f3": []},
    "o":{"f1": [], "f2": [], "f3": []},
    "u":{"f1": [], "f2": [], "f3": []},
}
f1_list = []
f2_list = []
f3_list = []


# humans = ["24FTL", "25MLM", "27MCM", "28MVN"]
humans = os.listdir(config["train_path"])

for h in humans:
  fnames = glob.glob(config["train_path"] + "/" +h+"/*.wav")
  fnames.sort()
  for audio_path in fnames:
      label = os.path.basename(audio_path).split(".")[0]
      human = audio_path.split("/")[-2]
      formants = formant_estimation(audio_path)

      print(h, "/" + label + "/", np.int32(formants)[:3])

      f1= formants[0]
      f2= formants[1]
      f3= formants[2]
      datalist[label]["f1"].append(f1)
      datalist[label]["f2"].append(f2)
      datalist[label]["f3"].append(f3)

def getF1F2F3(am):
  f1 = datalist[am]["f1"]
  f2 = datalist[am]["f2"]
  f3 = datalist[am]["f3"]
  return np.mean(f1), np.mean(f2), np.mean(f3)

print("/a/", getF1F2F3("a"))
print("/e/", getF1F2F3("e"))
print("/i/", getF1F2F3("i"))
print("/o/", getF1F2F3("o"))
print("/u/", getF1F2F3("u"))