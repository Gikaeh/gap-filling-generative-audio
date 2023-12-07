import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from glob import glob
import librosa as lb
from librosa import display
from tqdm import tqdm
import random
import torch

hop = 128
either_side = 10
fill_in = 0.5
move_between = 9.75
global_sr = 22050
n_mels = 128

class DataConversion:
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.data = glob(file_dir)
        self.y = []
        self.sr = []

    def load_data(self):
        print('Loading Data:')

        for x in tqdm(range(len(self.data))):
            yt, srt = lb.load(self.data[x], sr = global_sr)
            # TODO: Change this to upsample/downsample other sample rates
            assert srt == global_sr
            self.y.append(yt)

    def display_data(self):
        for x in range(len(self.data)):
            print(self.data[x])

    #def display_sr(self):
    #    for x in range(len(self.data)):
    #        print(self.sr[x])

    def display_y(self):
        for x in range(len(self.data)):
            print(self.y[x])

    def display_raw_audio(self, num):
        pd.Series(self.y[num]).plot(figsize=(10,5), lw=1, title=f'Raw Audio for {self.data[num]}')
        plt.show()

    # def data_to_stft(self):
    #     S_db = []

    #     print('Converting to Spectrogram:')
    #     for x in tqdm(range(len(self.data))):
    #         D = lb.stft(self.y[x])
    #         S_db.append(lb.amplitude_to_db(np.abs(D), ref=np.max))

    #     return S_db
    
    # def display_stft(self, stft_list, num):
    #     fig, ax = plt.subplots(figsize=(10,5))
    #     img = display.specshow(stft_list[num], x_axis='time', y_axis='log', ax=ax)
    #     ax.set_title(f'File {self.data[num]} Spectrogram', fontsize=20)
    #     plt.show()

    def data_to_mel(self):
        S_db_mel = []

        print('Converting to Mel-Spectrogram:')
        for x in tqdm(range(len(self.data))):
            S = lb.feature.melspectrogram(y=self.y[x], sr=global_sr, n_mels=n_mels, hop_length = hop)
            S_db_mel.append(lb.amplitude_to_db(S, ref=np.max))
        for x in range(5):
            print(self.y[x].shape)
            print(S_db_mel[x].shape)
        return S_db_mel

    def process_and_save_data(self):
        size = len(self.y)
        for start, end, name in [(0, 0.2, "trainbatch1"),
                                 (0.2, 0.4, "trainbatch2"),
                                 (0.4, 0.6, "trainbatch3"),
                                 (0.6, 0.8, "trainbatch4"),
                                 (0.8, 0.9, "validation"),
                                 (0.9, 1.0, "test")]:
            data = []
            for x in tqdm(range(int(start * size), int(end * size))):
                for i in range((len(self.y[x]) - (either_side * 2 - fill_in) * global_sr)//(global_sr * move_between)):
                    first = self.y[x][i * move_between * global_sr : (i * move_between + either_side) * global_sr]
                    middle = self.y[x][(i * move_between + either_side) * global_sr : (i * move_between + either_side + fill_in) * global_sr]
                    end = self.y[x][(i * move_between + either_side + fill_in) * global_sr : (i * move_between + either_side * 2 + fill_in) * global_sr]
                    first_mel = lb.amplitude_to_db(lb.feature.melspectrogram(y=first, sr=global_sr, n_mels=n_mels, hop_length = hop), ref = np.max)
                    middle_mel = lb.amplitude_to_db(lb.feature.melspectrogram(y=middle, sr=global_sr, n_mels=n_mels, hop_length = hop), ref = np.max)
                    end_mel = lb.amplitude_to_db(lb.feature.melspectrogram(y=end, sr=global_sr, n_mels=n_mels, hop_length = hop), ref = np.max)
                    data.append([first, middle, end, first_mel, middle_mel, end_mel])
            print(len(data))
            torch.save(data[:int(0.1 * len(data))], name + ".pt")
    
    def display_mel(self, mel_list, num):
        fig, ax = plt.subplots(figsize=(10,5))
        img = display.specshow(mel_list[num], x_axis='time', y_axis='log', ax=ax)
        ax.set_title(f'File {self.data[num]} Mel-Spectrogram', fontsize=20)
        plt.show()
        
if __name__ == "__main__":
    test1 = DataConversion('./dataset/*.mp3')
    test1.load_data()
    test1.display_raw_audio(20)
    #spect_list = test1.data_to_stft()
    #test1.display_stft(spect_list, 20)
    #spect_list_mel = test1.data_to_mel()
    #test1.display_mel(spect_list_mel, 20)
    test1.process_and_save_data()
