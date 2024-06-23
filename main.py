import numpy as np
import scipy.io.wavfile as wav
import pandas as pd
from scipy.fftpack import fft, dct
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

# Učitavanje zvučnog signala
rate, signal = wav.read('test.wav')

# Normalizacija signala
signal = signal / np.max(np.abs(signal))

# Parametri
frame_length = 2000  # Dužina prozora 20 ms
hop_length = int(frame_length / 2)  # Korak (preklapanje) prozora
n_fft = frame_length
n_mels = 40  # Broj Mel filtara

# Primjena Hammingovog prozora
hamming_window = np.hamming(frame_length)
frames = [signal[i:i + frame_length] * hamming_window for i in range(0, len(signal) - frame_length, hop_length)]

# Funkcija za izračunavanje frekvencija na Mel skali
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def get_mel_filterbanks(n_mels, n_fft, sample_rate):
    # Mel skala
    low_freq_mel = 0
    high_freq_mel = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Binovi frekvencija
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    
    # Filter-banka
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

# Dobivanje Mel filterbanke
mel_filterbank = get_mel_filterbanks(n_mels, n_fft, rate)

# FFT i primjena Mel filterbanke
mel_spectrogram = []
for frame in frames:
    spectrum = fft(frame, n=n_fft)
    power_spectrum = np.abs(spectrum[:n_fft // 2 + 1]) ** 2
    mel_spectrum = np.dot(mel_filterbank, power_spectrum)
    mel_spectrogram.append(np.log(mel_spectrum + 1e-10))  # Dodavanje male vrijednosti kako bi se izbjegli log(0) problemi

# Primjena DCT-a za izračun MFCC-a (ekstrakcija prvih 13 koeficijenata kepstra)
mfcc = np.array([dct(mel_spectrum)[:13] for mel_spectrum in mel_spectrogram])

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(mfcc[0])
plt.title('MFCC')
plt.xlabel('Koeficijenti')
plt.ylabel('Vrijednost')

# Učitavanje oznaka iz test.lab
df_lab = pd.read_csv('test.lab', sep=' ', header=None, names=['start', 'end', 'label'])

# Uklanjanje početne i završne oznake (ako su prisutne)
df_lab = df_lab.drop([0, len(df_lab) - 1])

# Grupiranje po fonemima
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

    # Kombiniranje svih segmenata u jedan skup
    combined_segments = np.vstack(segments)

    # Izračun srednjeg vektora i kovarijacijske matrice
    mean_vector = np.mean(combined_segments, axis=0)
    cov_matrix = np.cov(combined_segments, rowvar=False)

    # Spremanje modela fonema
    phoneme_models[phoneme] = {'mean': mean_vector, 'cov': cov_matrix}

# Klasifikacija segmenata
classified_segments = []

# Iteriramo kroz sve segmente i klasificiramo ih
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

# Ispis rezultata klasifikacije
for i, frame, phoneme, distance in classified_segments:
    print(f"Segment {i}: Fonem '{phoneme}' (Udaljenost: {distance:.2f})")

# Prikaz prvog prozora DCT i FFT

# FFT graf
plt.subplot(3, 1, 2)
plt.plot(np.abs(fft(frames[0], n=frame_length)))
plt.title('FFT prvog prozora')
plt.xlabel('Frekvencija (Hz)')
plt.ylabel('Amplituda')

# DCT graf
plt.subplot(3, 1, 3)
plt.plot(dct(mel_spectrogram[0])[1:])
plt.title('DCT prvog prozora')
plt.xlabel('Koeficijent')
plt.ylabel('Vrijednost')

plt.tight_layout()
plt.show()
