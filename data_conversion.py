import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from glob import glob
import librosa as lb
from librosa import display
from tqdm import tqdm
import random
import copy
from pydub import AudioSegment

global_sr = 22050

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
            # print(yt)
            assert srt == global_sr
            normalized_audio = (yt - np.min(yt)) / (np.max(yt) - np.min(yt))
            # print(normalized_audio)

            self.y.append(normalized_audio)

    # def convert_mp3_to_wav(self):
    #     print('Converting from mp3 to wav')
    #     for x in tqdm(range(len(self.data))):
    #         file_name, file_extension = os.path.splitext(self.data[x])
    #         wav_file = file_name + ".wav"
    #         audio = AudioSegment.from_mp3(self.data[x])
    #         audio.export(wav_file, format="wav")

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
        print('Converting to Mel-Spectrogram:')

        for x in tqdm(range(len(self.data))):
            for y in range(0,3):
                # Extract a 20-second segment
                start_time = np.random.uniform(0, max(0, len(self.y[x]) - 20 * global_sr))
                segment = self.y[x][int(start_time):int(start_time) + 20 * global_sr]

                # Generate mel-spectrogram for the complete 20 seconds
                mel_spect = lb.feature.melspectrogram(y=segment, sr=global_sr)
                self.mel_full.append(mel_spect)

                # Save a randim cut from the middle
                cut_start = int((mel_spect.shape[1] / 2) - 43.1 * (.25/2))
                cut_end = int(cut_start + 43.1 * .25)
                
                # Set the random cut in the original mel-spectrogram to zero to create the input
                mel_cut = copy.deepcopy(mel_spect)
                mel_cut[:, cut_start:cut_end] = 0
                self.mel_cut.append(mel_cut)

        return self.mel_cut, self.mel_full

    def display_mel(self, mel_list, num):
        if mel_list == 'full':
            mel_list = self.mel_full
        if mel_list == 'cut':
            mel_list = self.mel_cut

        plt.figure(figsize=(10, 4))
        lb.display.specshow(lb.power_to_db(mel_list[num], ref=np.max), y_axis='mel', x_axis='time')
        plt.title(f'File {self.data[num]} Mel-Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.show()
        
if __name__ == "__main__":
    test1 = DataConversion('./dataset/1036800.low.mp3')
    # test1.convert_mp3_to_wav()
    test1.load_data()
    test1.data_to_mel()
    # test1.display_raw_audio(20)
    # spect_list = test1.data_to_stft()
    # test1.display_stft(spect_list, 20)
    # spect_list_cut, spect_list_full = test1.data_to_mel()
    test1.display_mel('full', 0)
    test1.display_mel('cut', 0)
