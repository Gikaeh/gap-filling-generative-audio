from model import SimpleCNN
from data_conversion import DataConversion
import data_conversion
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile
from sklearn.preprocessing import MinMaxScaler

test1 = DataConversion('./dataset/1036800.low.mp3')
test1.load_data()
mel_spect_train, mel_spect_test = test1.data_to_mel( )

# Load the saved model for generation
loaded_model = SimpleCNN()
loaded_model.load_state_dict(torch.load('output/CNN2.pth'))
loaded_model
loaded_model.eval()

orig_incomplete_spec = mel_spect_train[0]
orig_incomplete_spec = torch.tensor(orig_incomplete_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

# Generate missing chunk using the loaded model
with torch.no_grad():
    new_complete_spec = loaded_model(orig_incomplete_spec)

# Convert the generated chunk back to a NumPy array
new_complete_spec_np = new_complete_spec.squeeze().cpu().numpy()

# Assuming you have an incomplete mel spectrogram named 'incomplete_spec'
# Display the original incomplete spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(orig_incomplete_spec.squeeze(), ref=np.max), y_axis='mel', x_axis='time')
plt.title('Original Incomplete Mel Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.show()

# Display the generated mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(new_complete_spec_np, ref=np.max), y_axis='mel', x_axis='time')
plt.title('Generated Mel Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.show()

# Convert the file back and save it
audio = librosa.feature.inverse.mel_to_audio(new_complete_spec_np, sr=data_conversion.global_sr)
soundfile.write("output/genaudio2-1.wav",audio, samplerate=data_conversion.global_sr)
