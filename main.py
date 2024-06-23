import numpy as np
import scipy.io.wavfile as wav
import pandas as pd
from scipy.fftpack import fft, ifft, dct
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import os

file_name = input('Unesite ime datoteke: ')

# Učitavanje zvučnog signala
rate, signal = wav.read(os.path.abspath(os.path.join(__file__ ,"../..",'wav_sm04',file_name+ '.wav')))
# Učitavanje oznaka iz test.lab
df_lab = pd.read_csv(os.path.abspath(os.path.join(__file__ ,"../..", 'lab_sm04', file_name + '.lab')), sep=' ', header=None, names=['start', 'end', 'label'])

# Normalizacija signala
signal = signal / np.max(np.abs(signal))

# Primjena Hammingovog prozora
frame_length = int(rate / 50) # Dužina prozora 20 ms
hop_length = int(frame_length / 2)  # Dodana varijabla za korak (preklapanje) prozora
hamming_window = np.hamming(frame_length)
frames = [signal[i:i + frame_length] * hamming_window for i in range(0, len(signal) - frame_length, hop_length)]

# FFT i logaritamska amplitudna spektralna gustoća
log_spectrum = []
for frame in frames:
    spectrum = fft(frame, n=frame_length)
    log_spectrum.append(np.log(np.abs(spectrum)))

# Primjena DCT-a za izračun MFCC-a (ekstrakcija prvih 13 koeficijenata kepstra)
mfcc = np.array([dct(log_spectrum_frame)[:13] for log_spectrum_frame in log_spectrum])

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.hist(mfcc[0])
plt.title('MFCC')
plt.xlabel('Koeficijenti')
plt.ylabel('Vrijednost')


# Uklanjanje početne i završne oznake (ako su prisutne)
df_lab = df_lab.drop([0, len(df_lab) - 1])

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
plt.plot(dct(log_spectrum[0])[1:])
plt.title('DCT prvog prozora')
plt.xlabel('Ksoeficijent')
plt.ylabel('Vrijednost')

plt.tight_layout()
plt.show()
