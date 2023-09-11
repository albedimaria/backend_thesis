import numpy as np
import essentia.standard as esstd

def extract_harmonicity_and_pitch(audio):
    # Create a Hann window
    window = esstd.Windowing(type='hann')

    # Compute the next power of two of the signal length
    nfft = 2 ** int(np.ceil(np.log2(len(audio))))

    # Pad the signal with zeros to the next power of two
    padded_audio = np.pad(audio, (0, nfft - len(audio)), mode='constant')

    # Apply the windowing function to the entire audio signal
    windowed_audio = window(padded_audio)

    # Remove the DC component from the windowed signal
    windowed_audio = windowed_audio - np.mean(windowed_audio)

    # Create a Spectrum algorithm instance
    spectrum = esstd.Spectrum()

    # Create a SpectralPeaksFunction algorithm instance
    spectral_peaks = esstd.SpectralPeaks(
        magnitudeThreshold=0.01,
        minFrequency=30,
        sampleRate=44100
    )

    # Create a HarmonicPeaks algorithm instance
    harmonic_peaks = esstd.HarmonicPeaks()

    # Compute the magnitude spectrum of the audio signal
    spectrum_audio = spectrum(windowed_audio)

    # Create a PitchYinFFT algorithm instance
    pitch_extractor = esstd.PitchYinFFT()

    # Compute the pitch of the audio signal
    pitch, pitch_confidence = pitch_extractor(spectrum_audio)

    # Convert pitch confidence to a percentage
    pitch_confidence_pct = f'{100 * pitch_confidence:.1f}%'

    # Extract the spectral peaks and harmonic peaks
    frequencies, magnitudes = spectral_peaks(spectrum_audio)
    harm_freq, harm_mag = harmonic_peaks(frequencies, magnitudes, pitch)

    # 0 if harmonic, 1 non-harmonic
    inharmonicity = esstd.Inharmonicity()(harm_freq, harm_mag)

    harmonicity = 1 - inharmonicity
    harmonicity_pct = f'{100 * harmonicity:.1f}%'

    # Check the pitch confidence and set pitch to None if it's below 0.5
    if pitch_confidence < 0.5:
        pitch = None
    else:
        pitch = round(pitch, 3)

    return {
        "pitch": pitch,
        "pitch_confidence_pct": pitch_confidence_pct,
        "harmonicity": harmonicity,
        "harmonicity_pct": harmonicity_pct
    }
