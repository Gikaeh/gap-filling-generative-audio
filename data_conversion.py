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

hop = 450
either_side = 10
fill_in = 2
move_between = 2
global_sr = 22050

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
            S = lb.feature.melspectrogram(y=self.y[x], sr=global_sr, n_mels=128, hop_length = hop)
            S_db_mel.append(lb.amplitude_to_db(S, ref=np.max))
        for x in range(5):
            print(self.y[x].shape)
            print(S_db_mel[x].shape)
        return S_db_mel

    def process_and_save_data(self, S_db_mel):
        data = []
        random.shuffle(S_db_mel)
        print("Splitting files to generate test cases:")
        for x in tqdm(range(len(S_db_mel))):
            fps = int(global_sr / hop)
            mel = S_db_mel[x]
            melt = np.transpose(mel)
            for i in range(int(melt.shape[0]/(move_between * fps))):
                first_mel = melt[i * move_between * fps : (i * move_between + either_side) * fps]
                middle_mel = melt[(i * move_between + either_side) * fps : (i * move_between + either_side + fill_in) * fps]
                end_mel = melt[(i * move_between + either_side + fill_in) * fps : (i * move_between + either_side * 2 + fill_in) * fps]
                first = self.y[x][i * move_between * global_sr : (i * move_between + either_side) * global_sr]
                middle = self.y[x][(i * move_between + either_side) * global_sr : (i * move_between + either_side + fill_in) * global_sr]
                end = self.y[x][(i * move_between + either_side + fill_in) * global_sr : (i * move_between + either_side * 2 + fill_in) * global_sr]
                data.append([first_mel, middle_mel, end_mel, first, middle, end])
        random.shuffle(data)
        torch.save({"training_data":data[:int(0.7 * len(data))],
                    "validation_data":data[int(0.7 * len(data)):int(0.8 * len(data))],
                    "test_data":data[int(0.8 * len(data)):]}, "data.pt")
    
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
    spect_list_mel = test1.data_to_mel()
    test1.display_mel(spect_list_mel, 20)
    test1.process_and_save_data(spect_list_mel)
