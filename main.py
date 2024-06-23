import numpy as np
import scipy.io.wavfile as wav
import pandas as pd
from scipy.fftpack import fft, ifft, dct
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

# Učitavanje zvučnog signala
rate, signal = wav.read('test.wav')

# Normalizacija signala
signal = signal / np.max(np.abs(signal))

# Primjena Hammingovog prozora
frame_length = 2048
hop_length = 512  # Dodana varijabla za korak (preklapanje) prozora
hamming_window = np.hamming(frame_length)
frames = [signal[i:i + frame_length] * hamming_window for i in range(0, len(signal) - frame_length, hop_length)]

# FFT i logaritamska amplitudna spektralna gustoća
log_spectrum = []
for frame in frames:
    spectrum = fft(frame, n=frame_length)
    log_spectrum.append(np.log(np.abs(spectrum)))

# Inverzna Fourierova transformacija (kepstrum)
cepstrum = [ifft(log_spectrum_frame).real for log_spectrum_frame in log_spectrum]

# Primjena DCT-a za izračun MFCC-a (ekstrakcija prvih 13 koeficijenata kepstra)
mfcc = np.array([dct(cepstrum_frame)[:13] for cepstrum_frame in cepstrum])

# Učitavanje transkripcije iz test.txt
with open('test.txt', 'r', encoding='utf-8') as f:
    transcript = f.read().strip().split("\n")

transcript = transcript[0].replace(' ', '')

# Učitavanje oznaka iz test.lab
df_lab = pd.read_csv('test.lab', sep=' ', header=None, names=['start', 'end', 'label'])

# Uklanjanje početne i završne oznake (ako su prisutne)
df_lab = df_lab.drop([0, len(df_lab) - 1])

# Provjera duljine transkriptnih oznaka
if len(transcript) != len(df_lab):
    raise ValueError("Duljina transkripcije ne odgovara duljini df_lab!")

# Povezivanje oznaka s transkripcijom
df_lab['transcript'] = [transcript[i] for i in range(len(df_lab))]

# Grupiranje po fonemima
phoneme_models = {}
for phoneme, group in df_lab.groupby('label'):
    # Prvo izračunajmo MFCC za svaki segment
    segments = []
    for _, row in group.iterrows():
        # Konverzija iz vremenskih jedinica (0.1 mikro sekunda) u sample-ove
        start_sample = int(row['start'] * rate / 10**7)
        end_sample = int(row['end'] * rate / 10**7)
        start_index = start_sample // hop_length
        end_index = end_sample // hop_length
    
        if start_index < len(mfcc) and end_index <= len(mfcc):
            segments.extend(mfcc[start_index:end_index])

    if len(segments) == 0:
        continue

    # Kombiniranje svih segmenata u jedan skup za treniranje
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
    # Izračunaj MFCC za trenutni segment
    cepstral_features = mfcc[i]
    
    # Inicijalizacija varijabli za klasifikaciju
    min_distance = float('inf')
    classified_phoneme = None
    
    # Iteracija kroz sve modele fonema i određivanje Mahalanobisove udaljenosti
    for phoneme, model in phoneme_models.items():
        mean_vector = model['mean']
        cov_matrix = model['cov']
        
        # Izračun Mahalanobisove udaljenosti
        try:
            distance_value = mahalanobis(cepstral_features, mean_vector, np.linalg.inv(cov_matrix))
        except np.linalg.LinAlgError:
            distance_value = float('inf')
        
        # Provjera je li nova udaljenost manja od prethodne minimalne
        if distance_value < min_distance:
            min_distance = distance_value
            classified_phoneme = phoneme
    
    # Dodavanje klasificiranog fonema u rezultate
    classified_segments.append((i, frames[i], classified_phoneme, min_distance))

# Ispis rezultata klasifikacije
for i, frame, phoneme, distance in classified_segments:
    print(f"Segment {i}: Phoneme '{phoneme}' (Distance: {distance:.2f})")

# Prikaz prvog prozora DCT i FFT

# FFT graf
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(np.abs(fft(frames[0], n=frame_length)))
plt.title('FFT prvog prozora')
plt.xlabel('Frekvencija (Hz)')
plt.ylabel('Amplituda')

# DCT graf
plt.subplot(2, 1, 2)
plt.plot(dct(cepstrum[0]))
plt.title('DCT prvog prozora')
plt.xlabel('Koeficijent')
plt.ylabel('Vrijednost')

plt.tight_layout()
plt.show()
