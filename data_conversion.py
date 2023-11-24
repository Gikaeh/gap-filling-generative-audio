import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from glob import glob
import librosa as lb
from librosa import display
from tqdm import tqdm

class DataConversion:
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.data = glob(file_dir)
        self.y = []
        self.sr = []

    def load_data(self):
        print('Loading Data:')

        for x in tqdm(range(len(self.data))):
            yt, srt = lb.load(self.data[x])
            self.y.append(yt)
            self.sr.append(srt)

    def print_sr(self):
        for x in range(len(self.data)):
            print(self.sr[x])

    def data_to_mel(self):
        S_db_mel = []

        print('Converting to Mel-Spectrogram:')
        for x in tqdm(range(len(self.data))):
            S = lb.feature.melspectrogram(y=self.y[x], sr=self.sr[x], n_mels=128)
            S_db_mel.append(lb.amplitude_to_db(S, ref=np.max))

        return S_db_mel
    
    def display_mel(self, mel_list, num):
        fig, ax = plt.subplots(figsize=(10,5))
        img = display.specshow(mel_list[num], x_axis='time', y_axis='log', ax=ax)
        ax.set_title(f'File {self.data[num]} Mel-Spectrogram', fontsize=20)
        plt.show()


test1 = DataConversion('./dataset/*.mp3')
test1.load_data()
spect_list = test1.data_to_mel()
test1.display_mel(spect_list, 20)


# pd.Series(y).plot(figsize=(10,5), lw=1, title='Raw Audio Example')
# plt.show()

# # D = lb.stft(y)
# # S_db = lb.amplitude_to_db(np.abs(D), ref=np.max)

# # fig, ax = plt.subplots(figsize=(10,5))
# # img = display.specshow(S_db, x_axis='time', y_axis='log', ax=ax)
# # plt.show()

# S = lb.feature.melspectrogram(y=y, sr=sr, n_mels=128)
# S_db_mel = lb.amplitude_to_db(S, ref=np.max)

# fig, ax = plt.subplots(figsize=(10,5))
# img = display.specshow(S_db_mel, x_axis='time', y_axis='log', ax=ax)
# plt.show()