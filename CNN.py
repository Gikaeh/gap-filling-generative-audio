from data_conversion import DataConversion
from model import SimpleCNN
import vessl
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    vessl.init()

    train_losses = []
    val_losses = []
    test_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    test1 = DataConversion('./dataset/*.mp3')
    test1.load_data()
    mel_spect_train, mel_spect_test = test1.data_to_mel()

    X_train, X_test, y_train, y_test = train_test_split(mel_spect_train, mel_spect_test, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    model = SimpleCNN().to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert the data to PyTorch DataLoader
    X_train = np.array(X_train, dtype=np.float32)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)

    X_val = np.array(X_val, dtype=np.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)

    X_test = np.array(X_test, dtype=np.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)


    y_train = np.array(y_train, dtype=np.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    y_val = np.array(y_val, dtype=np.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    y_test = np.array(y_test, dtype=np.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)


    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for partial_spectrogram, full_spectrogram in tqdm(train_loader, desc=f'Epoch {epoch + 1}/Training Encoder'):
            # Forward pass
            outputs = model(partial_spectrogram.to(device))

            # Compute the loss
            loss = criterion(outputs, full_spectrogram.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            

        avg_train_loss = train_loss / len(train_loader)
        vessl.log(step=epoch + 1, payload={"loss": train_loss})

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_partial_spectrogram, val_full_spectrogram in tqdm(val_loader, desc=f'Epoch {epoch + 1}/Validation Encoder'):

                val_outputs = model(val_partial_spectrogram.to(device))
                loss = criterion(val_outputs, val_full_spectrogram.to(device))
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_partial_spectrogram)
            vessl.log(step=epoch + 1, payload={"validation_loss": val_loss})

        # Test loop
        with torch.no_grad():
            test_loss = 0.0
            for test_partial_spectrogram, test_full_spectrogram in tqdm(test_loader, desc=f'Epoch {epoch + 1}/Testing Encoder'):

                test_outputs = model(test_partial_spectrogram.to(device))
                loss = criterion(test_outputs, test_full_spectrogram.to(device))
                test_loss += loss.item()

            avg_test_loss = test_loss / len(X_test)
            vessl.log(step=epoch + 1, payload={"test_loss": test_loss})

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        test_losses.append(avg_test_loss)

        print(f'Epoch {epoch + 1}:')
        print(f'Training Encoder- Training Loss: {avg_train_loss}')
        print(f'Validation Encoder- Validation Loss: {avg_val_loss}')
        print(f'Testing Encoder - Testing Loss: {avg_test_loss}')


    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), 'output/CNN.pth')