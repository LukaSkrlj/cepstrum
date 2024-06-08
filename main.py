import numpy as np
import scipy.fftpack as fftpack
from scipy.spatial.distance import mahalanobis
from scipy.io import wavfile


def compute_cepstral_coefficients(s, num_ceps=13):
    # Compute the FFT of the signal
    spectrum = np.fft.fft(s)

    # Compute the log magnitude of the spectrum
    log_spectrum = np.log(np.abs(spectrum))

    # Compute the DCT of the log magnitude spectrum
    cepstral_coefficients = fftpack.dct(log_spectrum, type=2, norm='ortho')

    # Return the first 'num_ceps' coefficients
    return cepstral_coefficients[:num_ceps]


def build_statistical_models(d):
    m = {}
    for phoneme, signals in d.items():
        ceps = [compute_cepstral_coefficients(s) for s in signals]
        print(ceps)
        ceps = np.array(ceps)
        mean = np.mean(ceps, axis=0)
        cov = np.cov(ceps, rowvar=False)
        m[phoneme] = (mean, cov)
    return m


def classify_segment(segment, m):
    ceps = compute_cepstral_coefficients(segment)
    min_distance = float('inf')
    best_class = None
    for phoneme, (mean, cov) in m.items():
        inv_cov = np.linalg.inv(cov)
        distance = mahalanobis(ceps, mean, inv_cov)
        if distance < min_distance:
            min_distance = distance
            best_class = phoneme
    return best_class

fs, signal = wavfile.read('test.wav')

database = {}
with open('test.lab', 'r') as file:
    # Read each line in the file
    for line in file:
        # Split the line into words based on spaces
        words_in_line = line.strip().split()

        number_of_samples = fs / 10e6
        data = signal[int(number_of_samples * int(words_in_line[0])):int(number_of_samples * int(words_in_line[1]))]
        if words_in_line[2] in database:
            database[words_in_line[2]] = [*database[words_in_line[2]], data]
        else:
            database[words_in_line[2]] = [data]

print(database)
# Assuming database and sample signal are loaded
models = build_statistical_models(database)
print(models)
phoneme_class = classify_segment(signal, models)
print(f'The segment is classified as: {phoneme_class}')
