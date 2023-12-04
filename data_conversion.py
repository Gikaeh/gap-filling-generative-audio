import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from glob import glob
import librosa as lb
from librosa import display
from tqdm import tqdm
import random

hop = 128
either_side = 10
fill_in = 2
move_between = 10
global_sr = 22050
n_mels = 128

class DataConversion:
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.data = glob(file_dir)
        self.y = []
        self.sr = []
        self.mel_full = []
        self.mel_cut = []

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

    def data_to_mel(self, device):
        S_db_mel = []

        print('Converting to Mel-Spectrogram:')

        for x in tqdm(range(len(self.data))):
            if hop > 0: S = lb.feature.melspectrogram(y=self.y[x], sr=self.sr[x], n_mels=128, hop_length = hop)
            else: S = self.y[x]
            S_db_mel.append(lb.amplitude_to_db(S, ref=np.max))
        for x in range(5):
            print(self.y[x].shape)
            print(self.sr[x])
            print(S_db_mel[x].shape)
        return S_db_mel

    def make_inputs_outputs(self, S_db_mel):
        inputs_before = []
        inputs_after = []
        outputs = []
        random.shuffle(S_db_mel)
        print("Splitting files to generate test cases:")
        for x in tqdm(range(len(S_db_mel))):
            fps = int(self.sr[x] / hop) if hop > 0 else self.sr[x]
            mel = S_db_mel[x]
            melt = np.transpose(mel)
            for i in range(int(melt.shape[0]/(move_between * fps))):
                first = melt[i * move_between * fps : (i * move_between + either_side) * fps]
                middle = melt[(i * move_between + either_side) * fps : (i * move_between + either_side + fill_in) * fps]
                end = melt[(i * move_between + either_side + fill_in) * fps : (i * move_between + either_side * 2 + fill_in) * fps]
                inputs_before.append(first)
                inputs_after.append(end)
                outputs.append(middle)
        for i in range(5):
            print(inputs_before[i].shape)
            print(inputs_after[i].shape)
            print(outputs[i].shape)
        return (inputs_before, inputs_after, outputs)
    
    def display_mel(self, mel_list, num):
        if mel_list == 'full':
            mel_list = self.mel_full
        if mel_list == 'cut':
            mel_list = self.mel_cut
        fig, ax = plt.subplots(figsize=(10,5))
        img = display.specshow(mel_list[num], x_axis='time', y_axis='log', ax=ax)
        ax.set_title(f'File {self.data[num]} Mel-Spectrogram', fontsize=20)
        plt.show()
        
if __name__ == "__main__":
    test1 = DataConversion('./dataset/*.mp3')
    test1.load_data()
    test1.display_raw_audio(20)
    spect_list = test1.data_to_stft()
    test1.display_stft(spect_list, 20)
    spect_list_mel = test1.data_to_mel()
    test1.display_mel(spect_list_mel, 20)
