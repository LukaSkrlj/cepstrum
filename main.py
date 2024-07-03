import numpy as np
import scipy.io.wavfile as wav
import pandas as pd
from scipy.fftpack import fft, dct
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Directory paths
wav_dir = os.path.abspath(os.path.join(__file__, "../..", 'wav_sm04'))
lab_dir = os.path.abspath(os.path.join(__file__, "../..", 'lab_sm04'))

# Number of files to process
n_files = 100  # Change this to the number of files you want to process

true_labels = []
predicted_labels = []

# Function to process a single file
def process_file(file_name):
    try:
        # Load audio signal
        rate, signal = wav.read(os.path.join(wav_dir, file_name + '.wav'))
        # Load labels
        df_lab = pd.read_csv(os.path.join(lab_dir, file_name + '.lab'), sep=' ', header=None, names=['start', 'end', 'label'])

        # Preprocessing labels
        df_lab['label'] = df_lab['label'].replace({'a:': 'a', 'e:': 'e', 'i:': 'i', 'o:': 'o', 'u:': 'u', 'r:' : 'r'})

        # Normalize signal
        signal = signal / np.max(np.abs(signal))

        # Parameters
        frame_length = int(rate / 50)  # 20 ms window length
        hop_length = int(frame_length / 2)  # 50% overlap
        n_fft = frame_length
        n_mels = 40  # Number of Mel filters

        # Apply Hamming window
        hamming_window = np.hamming(frame_length)
        frames = [signal[i:i + frame_length] * hamming_window for i in range(0, len(signal) - frame_length, hop_length)]

        # Function to convert frequency to Mel scale
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        # Function to convert Mel scale to frequency
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)

        # Function to get Mel filter banks
        def get_mel_filterbanks(n_mels, n_fft, sample_rate):
            low_freq_mel = 0
            high_freq_mel = hz_to_mel(sample_rate / 2)
            mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
            hz_points = mel_to_hz(mel_points)
            bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
            filterbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
            for i in range(1, n_mels + 1):
                left = bin_points[i - 1]
                center = bin_points[i]
                right = bin_points[i + 1]
                for j in range(left, center):
                    filterbank[i - 1, j] = (j - bin_points[i - 1]) / (bin_points[i] - bin_points[i - 1])
                for j in range(center, right):
                    filterbank[i - 1, j] = (bin_points[i + 1] - j) / (bin_points[i + 1] - bin_points[i])
            return filterbank

        # Get Mel filter banks
        mel_filterbank = get_mel_filterbanks(n_mels, n_fft, rate)

        # FFT and apply Mel filter banks
        mel_spectrogram = []
        for frame in frames:
            spectrum = fft(frame, n=n_fft)
            power_spectrum = np.abs(spectrum[:n_fft // 2 + 1]) ** 2
            mel_spectrum = np.dot(mel_filterbank, power_spectrum)
            mel_spectrogram.append(np.log(mel_spectrum + 1e-10))  # Adding a small value to avoid log(0)

        # Apply DCT to compute MFCCs (extract first 13 coefficients)
        mfcc = np.array([dct(mel_spectrum)[:13] for mel_spectrum in mel_spectrogram])

        # Remove the first and last label (if present)
        df_lab = df_lab.drop([0, len(df_lab) - 1])

        # Group by phonemes
        phoneme_models = {}
        for phoneme, group in df_lab.groupby('label'):
            segments = []
            for _, row in group.iterrows():
                start_sample = int(row['start'] * rate / 10**7)
                end_sample = int(row['end'] * rate / 10**7)
                start_index = start_sample // hop_length
                end_index = end_sample // hop_length
                if start_index < len(mfcc) and end_index <= len(mfcc):
                    segments.extend(mfcc[start_index:end_index])
            if len(segments) == 0:
                continue
            combined_segments = np.vstack(segments)
            mean_vector = np.mean(combined_segments, axis=0)
            cov_matrix = np.cov(combined_segments, rowvar=False)
            phoneme_models[phoneme] = {'mean': mean_vector, 'cov': cov_matrix}

        # Classify segments
        for i in range(len(frames)):
            cepstral_features = mfcc[i]
            min_distance = float('inf')
            classified_phoneme = None
            for phoneme, model in phoneme_models.items():
                mean_vector = model['mean']
                cov_matrix = model['cov']
                try:
                    # Handle NaN and infinite values in mean_vector and cov_matrix
                    if np.isnan(mean_vector).any() or np.isnan(cov_matrix).any() or np.isinf(mean_vector).any() or np.isinf(cov_matrix).any():
                        continue

                    # Regularize covariance matrix if singular
                    epsilon = 1e-6
                    cov_matrix_regularized = cov_matrix + epsilon * np.eye(len(mean_vector))

                    distance_value = mahalanobis(cepstral_features, mean_vector, np.linalg.inv(cov_matrix_regularized))
                except np.linalg.LinAlgError:
                    distance_value = float('inf')
                if distance_value < min_distance:
                    min_distance = distance_value
                    classified_phoneme = phoneme

            true_labels.append(df_lab.iloc[i % len(df_lab)]['label'])  # Assuming the label file matches the segments
            predicted_labels.append(classified_phoneme)

    except FileNotFoundError as e:
        print(f"Ignoring missing file: {file_name}.lab")

# Process first n .wav files in the directory
processed_count = 0
for file in os.listdir(wav_dir):
    if file.endswith('.wav'):
        file_name = os.path.splitext(file)[0]
        print(f"Processing file: {file_name}")
        process_file(file_name)
        processed_count += 1
        if processed_count >= n_files:
            break

# Preprocess labels (merge 'e:', 'i:', 'o:', 'u:' with 'e', 'i', 'o', 'u' respectively)
true_labels_processed = [label if label not in ['e:', 'i:', 'o:', 'u:', 'r:'] else label[:-1] for label in true_labels]
predicted_labels_processed = [label if label not in ['e:', 'i:', 'o:', 'u:', 'r:'] else label[:-1] for label in predicted_labels]

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels_processed, predicted_labels_processed)

# Get unique labels for axis tick labels
unique_labels = sorted(set(true_labels_processed + predicted_labels_processed))

# Print classification report
print(classification_report(true_labels_processed, predicted_labels_processed, target_names=unique_labels))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Counts)')
plt.show()
