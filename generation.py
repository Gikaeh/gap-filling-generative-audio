from model import SimpleCNN
from data_conversion import DataConversion
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import librosa
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test1 = DataConversion('./dataset/*.mp3')
test1.load_data()
mel_spect_train, mel_spect_test = test1.data_to_mel( )

# Load the saved model for generation
loaded_model = SimpleCNN()
loaded_model.load_state_dict(torch.load('CNN.pth'))
loaded_model.to(device)
loaded_model.eval()

# Assuming you have a new incomplete mel spectrogram for generation
# Replace this with your actual incomplete mel spectrogram
new_incomplete_spec = mel_spect_test[20]

# Preprocess the new incomplete spectrogram
scaler = MinMaxScaler()
new_incomplete_spec = scaler.fit_transform(new_incomplete_spec)
new_incomplete_spec = torch.tensor(new_incomplete_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)

# Generate missing chunk using the loaded model
with torch.no_grad():
    generated_chunk = loaded_model(new_incomplete_spec)

# Convert the generated chunk back to a NumPy array
generated_chunk_np = generated_chunk.squeeze().cpu().numpy()

# Assuming you have an incomplete mel spectrogram named 'incomplete_spec'
# Display the original incomplete spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(new_incomplete_spec.squeeze(), ref=np.max), y_axis='mel', x_axis='time')
plt.title('Original Incomplete Mel Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.show()

# Display the generated mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(generated_chunk_np, ref=np.max), y_axis='mel', x_axis='time')
plt.title('Generated Mel Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.show()
