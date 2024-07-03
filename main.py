import numpy as np
import scipy.io.wavfile as wav
import pandas as pd
from scipy.fftpack import fft, dct
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import os

def calculate_matching_percentage(dictionary):
    result = {}

    for key, value in dictionary.items():
        result[key] = value.count(key) / len(value)

    return result

def calculate_matching_counts(dictionary):
    result = {}

    for key, value in dictionary.items():
        counts = {}
        for v in value:
            if v in counts:
                counts[v] += 1
            else:
                counts[v] = 1
        result[key] = counts

    return result

wav_directory = '../wav_sm04'
lab_directory = '../lab_sm04'
n_files = 100

mfcc_all = []

file_list = os.listdir(wav_directory)[:n_files]
errors = {}
for file_name in file_list:
    if file_name.endswith('.wav'):
        file_prefix = os.path.splitext(file_name)[0]

        wav_file = os.path.join(wav_directory, file_name)
        rate, signal = wav.read(wav_file)

        label_file = os.path.join(lab_directory, file_prefix + '.lab')
        if os.path.exists(label_file):
            df_lab = pd.read_csv(label_file, sep=' ', header=None, names=['start', 'end', 'label'])
        else:
            continue

        signal = signal / np.max(np.abs(signal))

        frame_length = int(rate / 50)
        hop_length = int(frame_length / 2)
        hamming_window = np.hamming(frame_length)
        frames = [signal[i:i + frame_length] * hamming_window for i in range(0, len(signal) - frame_length, hop_length)]

        log_spectrum = []
        for frame in frames:
            spectrum = fft(frame, n=frame_length)
            log_spectrum.append(np.log(np.abs(spectrum)))

        mfcc = np.array([dct(log_spectrum_frame)[:13] for log_spectrum_frame in log_spectrum])

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

        classified_segments = []

        for i in range(len(frames)):
            cepstral_features = mfcc[i]

            min_distance = float('inf')
            classified_phoneme = None

            for phoneme, model in phoneme_models.items():
                mean_vector = model['mean']
                cov_matrix = model['cov']

                try:
                    distance_value = mahalanobis(cepstral_features, mean_vector, np.linalg.inv(cov_matrix))
                except np.linalg.LinAlgError:
                    distance_value = float('inf')

                if distance_value < min_distance:
                    min_distance = distance_value
                    classified_phoneme = phoneme

            classified_segments.append((i, frames[i], classified_phoneme, min_distance))

        for i, frame, phoneme, distance in classified_segments:
            time = (i + 1) * 10 / 10e3
            f = df_lab[(df_lab['start'] / 10e7 <= time) & (df_lab['end'] / 10e7 >= time)]

            if not f.empty:
                p = f['label'].values[0]
                if p in errors:
                    errors[p].append(phoneme)
                else:
                    errors[p] = []
                    errors[p].append(phoneme)

matching_counts = calculate_matching_counts(errors)

phonemes = sorted(list(matching_counts.keys()))
matrix_data = np.zeros((len(phonemes), len(phonemes)), dtype=int)

for i, true_phoneme in enumerate(phonemes):
    if true_phoneme in matching_counts:
        for predicted_phoneme, count in matching_counts[true_phoneme].items():
            if predicted_phoneme in phonemes:
                j = phonemes.index(predicted_phoneme)
                matrix_data[i, j] = count



plt.figure(figsize=(18, 14)) 

plt.imshow(matrix_data, cmap='Blues', interpolation='nearest', aspect='auto')

plt.title('Misclassification Matrix of Phonemes', fontsize=16)
plt.colorbar()
plt.xticks(np.arange(len(phonemes)), phonemes, rotation=45, ha='right', fontsize=12) 
plt.yticks(np.arange(len(phonemes)), phonemes, fontsize=12)
plt.xlabel('Predicted Phonemes', fontsize=14)  
plt.ylabel('True Phonemes', fontsize=14)  


for i in range(len(phonemes)):
    for j in range(len(phonemes)):
        plt.text(j, i, str(matrix_data[i, j]), ha='center', va='center', color='black', fontsize=10) 

plt.tight_layout()
plt.show()


matching_percentage = calculate_matching_percentage(errors)

plt.figure(figsize=(10, 6))
plt.bar(matching_percentage.keys(), matching_percentage.values(), color='skyblue')
plt.xlabel('Phonemes')
plt.ylabel('Accuracy Percentage')
plt.title('Percentage of Correctly Identified Phonemes')
plt.ylim(0, 1)
plt.grid(True)
plt.show()


overall_accuracy = round(sum([matrix_data[i, i] for i in range(len(phonemes))]) / np.sum(matrix_data) * 100, 2)
print(f'Overall Accuracy: {overall_accuracy:.2f}%')
