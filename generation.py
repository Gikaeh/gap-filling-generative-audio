from model import SimpleCNN
from data_conversion import DataConversion
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test1 = DataConversion('./dataset/1036800.low.mp3')
test1.load_data()
mel_spect_train, mel_spect_test = test1.data_to_mel( )

# Load the saved model for generation
loaded_model = SimpleCNN()
loaded_model.load_state_dict(torch.load('output/CNN.pth'))
loaded_model.to(device)
loaded_model.eval()

# Assuming you have a new incomplete mel spectrogram for generation
# Replace this with your actual incomplete mel spectrogram
orig_incomplete_spec = mel_spect_train[0]

# Preprocess the new incomplete spectrogram
# Assuming you standardized to common values during training
scaler = MinMaxScaler(feature_range=(-1,1))
fitted_incomplete_spec = scaler.fit_transform(orig_incomplete_spec.squeeze())
fitted_incomplete_spec = torch.tensor(fitted_incomplete_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)

# Generate missing chunk using the loaded model
with torch.no_grad():
    new_complete_spec = loaded_model(fitted_incomplete_spec)

# Convert the generated chunk back to a NumPy array
new_complete_spec_np = new_complete_spec.squeeze().cpu().numpy()
new_complete_spec_np = scaler.inverse_transform(new_complete_spec_np)

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
