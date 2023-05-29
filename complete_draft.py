import os
import glob
import time
import json
import numpy as np
import essentia.standard as esstd
import essentia.streaming as esstr


app = Flask(__name__)

@app. route("/members")
def members():
 return {"members": ["member 1", "member 2", "member 3"]}


if __name__ == "__main__":
    app.run(debug = true)
# Define the directory path where the audio files are stored
dir_path = '/home/albertodimaria/thesis/test_analysis'

# Define the path to the ml_models directory
model_path = '/home/albertodimaria/thesis/ml_models/'

# load pre-trained Tempo Classification model
bpm_model = esstd.TempoCNN(graphFilename=os.path.join(model_path, 'deeptemp-k4-3.pb'))
bpm_model_square_16 = esstd.TempoCNN(graphFilename=os.path.join(model_path, 'deepsquare-k16-3.pb'))
bpm_model_temp_16 = esstd.TempoCNN(graphFilename=os.path.join(model_path, 'deeptemp-k16-3.pb'))

# valence-arousal pairs
va_embedding_model = esstd.TensorflowPredictMusiCNN(graphFilename=os.path.join(model_path, 'msd-musicnn-1.pb'), output="model/dense/BiasAdd")
va_model = esstd.TensorflowPredict2D(graphFilename=os.path.join(model_path, 'deam-musicnn-msd-2.pb'), output="model/Identity")

# mood 
mood_and_timbre_embedding_model = esstd.TensorflowPredictEffnetDiscogs(graphFilename=os.path.join(model_path, 'discogs_label_embeddings-effnet-bs64-1.pb'), output="PartitionedCall:1")
mood_model = esstd.TensorflowPredict2D(graphFilename=os.path.join(model_path, 'mtg_jamendo_moodtheme-effnet-discogs_label_embeddings-1.pb'))

with open(os.path.join(model_path, 'discogs_label_embeddings-effnet-bs64-1.json'), 'r') as json_file:
                metadata = json.load(json_file)
mood_labels = metadata['classes']

# Danceability
with open(os.path.join(model_path, 'danceability-vggish-audioset-1.json'), 'r') as json_file:
                metadata = json.load(json_file)
dance_index = metadata['classes'].index('danceable')
danceability_activations = esstd.TensorflowPredictVGGish(graphFilename=os.path.join(model_path, 'danceability-vggish-audioset-1.pb'))

# timbre
timbre_model = esstd.TensorflowPredict2D(graphFilename=os.path.join(model_path, 'timbre-effnet-discogs-1.pb'), output="model/Softmax")

with open(os.path.join(model_path, 'timbre-effnet-discogs-1.json'), 'r') as json_file:
                            metadata = json.load(json_file)
timbre_labels = metadata['classes']

# instrument 
instrument_embedding_model = esstd.TensorflowPredictEffnetDiscogs(graphFilename=os.path.join(model_path, 'discogs_artist_embeddings-effnet-bs64-1.pb'), output="PartitionedCall:1")
instrument_model = esstd.TensorflowPredict2D(graphFilename=os.path.join(model_path, 'mtg_jamendo_instrument-effnet-discogs_artist_embeddings-1.pb'))

with open(os.path.join(model_path, 'mtg_jamendo_instrument-effnet-discogs_artist_embeddings-1.json'), 'r') as json_file:
    metadata = json.load(json_file)
instrument_labels = metadata['classes']


# given the input and list of labels, predicts the labels for the audio
def predict_label(audio, embedding_model, classification_model, labels):
    """
    Predict the primary label(s) present in the audio file.
    Returns the predicted label(s).
    """
    # Extract embeddings using the embedding model
    embeddings = embedding_model(audio)
    
    # Predict labels using the classification model
    predictions = classification_model(embeddings)
    
    # Extract the tags from the predictions
    tags = predictions[0][:len(labels)]
    
    # Determine the most likely label(s) based on the tag probabilities
    max_index = int(np.argmax(tags))
    predicted_label = labels[max_index]
    
    return predicted_label


# Define the number of sectors
num_sectors = 16

# Define the sector angles
sector_angles = 2 * np.pi / num_sectors
sector_theta = [(i + 0.5) * sector_angles for i in range(num_sectors)]

# Define the colors and labels for each sector
sector_colors = ['orangered', 'orange', 'gold', 'yellow',
                 'yellowgreen', 'limegreen', 'green', 'seagreen',
                 'aquamarine', 'lightblue', 'steelblue', 'blue',
                 'blueviolet', 'violet', 'hotpink', 'red']

sector_labels = ['alert', 'excited', 'elated', 'happy',
                  'content', 'serene', 'relaxed', 'calm',
                  'fatigued', 'lethargic', 'depressed', 'sad',
                  'upset', 'stressed', 'nervous', 'tens']

# Define a threshold for the distance between the point and the origin
threshold = 0.2

# Define a function to get the sector color and label for a given valence and arousal pair
def get_sector_color_label(valence, arousal):
    # Calculate the polar coordinates of the point
    r = np.sqrt(valence**2 + arousal**2)
    theta = np.arctan2(valence, arousal)

    # Shift the angle to fall within the range [0, 2*pi)
    if theta < 0:
        theta += 2 * np.pi

    # If the distance is below the threshold, return 'neutral' color and label
    if abs(valence) < threshold and abs(arousal) < threshold:
        return 'white', 'neutral'

    # Find the sector that the point falls into
    sector_index = int(np.floor(theta / sector_angles))

    # Get the color and label of the sector
    sector_color = sector_colors[sector_index]
    sector_label = sector_labels[sector_index]

    return sector_color, sector_label


# Define the tempo range thresholds and labels
tempo_ranges = {
    'very slow': (30, 60),
    'slow': (60, 90),
    'moderate-slow': (90, 105),
    'moderate': (105, 120),
    'moderate-fast': (120, 135),
    'fast': (135, 150),
    'very fast': (150, 256)
}

# Function to assign a tempo label based on the BPM range
def assign_tempo_label(bpm):
    for label, (min_bpm, max_bpm) in tempo_ranges.items():
        if min_bpm <= bpm < max_bpm:
            return label
    return 'unknown'

# Function to calculate the valence/arousal values, normalized in the range [-1,1]
def calculate_valence_arousal(audio, va_embedding_model, va_model):
    va_embeddings = va_embedding_model(audio)
    va_predictions = va_model(va_embeddings)

    # Normalize valence and arousal values to range [-1, 1]
    valence_min, valence_max = 1, 9
    arousal_min, arousal_max = 1, 9

    valence_norm = (2 * (va_predictions[:, 0] - valence_min) / (valence_max - valence_min)) - 1
    arousal_norm = (2 * (va_predictions[:, 1] - arousal_min) / (arousal_max - arousal_min)) - 1

    # Calculate mean normalized valence and arousal values
    valence = round(float(np.mean(valence_norm)), 4)
    arousal = round(float(np.mean(arousal_norm)), 4)

    return valence, arousal


for root, dirs, files in os.walk(dir_path):
    for file_name in files:
        # Check if file is MP3 or WAV
        if file_name.lower().endswith('.mp3') or file_name.lower().endswith('.wav'):
            file_path = os.path.join(root, file_name)
            print("Loading file:", file_path)
            
            audio_og = esstd.MonoLoader(filename=file_path, sampleRate=44100)()
            # Resample to 16 kHz

            audio = esstd.Resample(inputSampleRate=44100, outputSampleRate=16000)(audio_og)
            cnn_audio = esstd.Resample(inputSampleRate=44100, outputSampleRate=11025)(audio)

            # BPM
            global_bpm, local_bpm, local_probs = bpm_model(cnn_audio)
            global_bpm_square_16, _, _ = bpm_model_square_16(cnn_audio)
            global_bpm_temp_16, _, _ = bpm_model_temp_16(cnn_audio)
            global_bpm_percival = esstd.PercivalBpmEstimator()(audio_og)
            # global_bpm_degara, _, _ = esstd.BeatTrackerDegara()(resampled_audio)
            global_bpm_ext, _, _, _, _ = esstd.RhythmExtractor2013()(audio_og)
            # Assign a tempo label based on the global BPM range
            tempo_label = assign_tempo_label(global_bpm)

            """# VALENCE AROUSAL
            va_embeddings = va_embedding_model(audio)
            va_predictions = va_model(va_embeddings)

            # Normalize valence and arousal values to range [0, 1]
            valence_min, valence_max = 1, 9
            arousal_min, arousal_max = 1, 9

            valence_norm = (2 * (va_predictions[:, 0] - valence_min) / (valence_max - valence_min)) - 1
            arousal_norm = (2 * (va_predictions[:, 1] - arousal_min) / (arousal_max - arousal_min)) - 1

            # Calculate mean normalized valence and arousal values
            valence = round(float(np.mean(valence_norm)), 4)
            arousal = round(float(np.mean(arousal_norm)), 4)"""
            
            valence, arousal = calculate_valence_arousal(audio, va_embedding_model, va_model)
           
            # COLOR
            color, emotion = get_sector_color_label(valence, arousal)

            # KEY
            tonal_extractor = esstd.TonalExtractor()
            tonal_features = tonal_extractor(audio_og)
            key = {
                    "estimated_key": tonal_features[9],
                    "key_scale": tonal_features[10],
                    "key_strength": round(tonal_features[11], 3)
                }
            

            # MOOD

            # Get the mood label with the corresponding index
            predicted_mood = predict_label(audio, mood_and_timbre_embedding_model, mood_model, mood_labels)


            # TIMBRE
            timbre_predicted = predict_label(audio, mood_and_timbre_embedding_model, timbre_model, timbre_labels)


            # INSTRUMENT
            instrument_predicted = predict_label(audio, instrument_embedding_model, instrument_model, instrument_labels)


            # DANCEABILITY

            activations = danceability_activations(audio)
            # Get the danceability probability from the activations array
            danceability_prob_decimal = activations.mean(axis=0)[dance_index]
            danceability_prob_percent = f'{100 * danceability_prob_decimal:.1f}%'


            # DYNAMIC COMPLEXITY

            # Predict the dynamic complexity using a machine learning model
            dynamic_complexity, global_loudness = esstd.DynamicComplexity()(audio_og)
            # Normalize the dynamic complexity coefficient by the global loudness level estimate
            normalized_dynamic_complexity = dynamic_complexity / abs(global_loudness)        


            # HARMONICITY AND PITCH

            # Create a Hann window
            window = esstd.Windowing(type='hann')

            # Compute the next power of two of the signal length
            nfft = 2 ** int(np.ceil(np.log2(len(audio_og))))

            # Pad the signal with zeros to the next power of two
            padded_audio = np.pad(audio_og, (0, nfft - len(audio_og)), mode='constant')

            # Apply the windowing function to the entire audio signal
            windowed_audio = window(padded_audio)

            # Remove the DC component from the windowed signal
            windowed_audio = windowed_audio - np.mean(windowed_audio)

            # Create a Spectrum algorithm instance
            spectrum = esstd.Spectrum()

            # Create a SpectralPeaksFunction algorithm instance
            spectral_peaks = esstd.SpectralPeaks(
                magnitudeThreshold=0.01,
                minFrequency = 30,
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

            

            # JSON

            # Define the path to the directory where JSON files will be saved
            json_dir = "/home/albertodimaria/thesis/json_results"

            # Create the JSON object with results and comments
            results = {
                "file_name": file_name,
                "summary": {
                   
                    "BPM": {
                        "value": global_bpm,
                        "description": "Beats per minute of the whole track",
                        "range": "from 20 to 200" 
                    }, 
                    "BPM_2": {
                        "value": global_bpm_square_16,
                        "description": "Beats per minute of the whole track",
                        "range": "from 20 to 200" 
                    },
                    "BPM_3": {
                        "value": global_bpm_temp_16,
                        "description": "Beats per minute of the whole track",
                        "more in detail": "most accurate with DNN",
                        "range": "from 20 to 200" 
                    },
                    "BPM_4": {
                        "value": global_bpm_percival,
                        "description": "Beats per minute of the whole track",
                        "more in detail": "not based on trained model",
                        "range": "from 20 to 200" 
                    },
                     "BPM_5": {
                        "value": global_bpm_ext,
                        "description": "Beats per minute of the whole track",
                        "more in detail": "not based on trained model",
                        "range": "from 20 to 200" 
                    },

                 
                    "key": {
                        "value": key,
                        "description": "Musical key (e.g. C, D, E), key scale (e.g. major/minor), strength of the estimated key in the range [0,1]",
                    
                    },

                     "mood": {
                        "value": predicted_mood,
                        "description": "Predicted mood of the music",
                        "available classes": mood_labels                        
                    },
                },
                "Tempo and Rhythm": {
                    "Tempo": {
                        "value": tempo_label,
                        "description": "Tempo label (e.g. fast, slow, moderate)",
                        "labels and range": tempo_ranges
                    },
                    "danceability": {
                        "value": danceability_prob_percent,
                        "description": "Probability of the music being danceable",
                        "range": "from 0 to 100%"
                    },
                    "dynamic_complexity_norm": {
                        "value": round(normalized_dynamic_complexity, 3),
                        "description": "Normalized dynamic complexity: higher values correspond to greater variations in the volume of the track",
                        "more in detail": "Calculated as the max variation wrt the global loudness (in dB), normalized by the global loudness",
                        "range": "from 0 to 1"
                    },
                    "global_loudness_dB": {
                        "value": round(global_loudness, 1),
                        "comment": "Global loudness in decibels"
                    }
                },
                "Emotion": {
                    "valence": {
                        "value": valence,
                        "comment": "Valence score (positive or negative mood)",
                        "range": "the values are normalized in [-1, 1]"
                    },
                    "arousal": {
                        "value": arousal,
                        "comment": "Arousal score (level of excitement or energy)",
                        "range": "the values are normalized in [-1, 1]"
                    },

                    
                    "color": {
                        "value": color,
                        "comment": "Color associated with valence/arousal values",
                        "list of colors": ['orangered', 'orange', 'gold', 'yellow',
                          'yellowgreen', 'limegreen', 'green', 'seagreen', 
                          'aquamarine', 'lightblue', 'steelblue',
                          'blue', 'blueviolet', 'violet', 'hotpink', 'red'],
                        "more in detail": "the 16 colors follow a wheel color map starting from arousal = 1 & valnce = 0. Each color will be also linked to an emotion (see below)",
                        "color - emotion": [("orangered", "alert"),
                                            ("orange", "excited"),
                                            ("gold", "elated"),
                                            ("yellow", "happy"),
                                            ("yellowgreen", "content"),
                                            ("limegreen", "serene"),
                                            ("green", "relaxed"),
                                            ("seagreen", "calm"),
                                            ("aquamarine", "fatigued"),
                                            ("lightblue", "lethargic"),
                                            ("steelblue", "depressed"),
                                            ("blue", "sad"),
                                            ("blueviolet", "upset"),
                                            ("violet", "stressed"),
                                            ("hotpink", "nervous"),
                                            ("red", "tens")]
                    },

                     "emotion": {
                        "value": emotion,
                        "comment": "emotion associated with valence/arousal values",
                        "list of emotions": ['alert', 'excited', 'elated', 'happy',
                         'content', 'serene', 'relaxed', 'calm',
                         'fatigued', 'lethargic', 'depressed', 'sad',
                        'upset', 'stressed', 'nervous', 'tens'],
                        "more in detail": "The 16 emotions follow a wheel map starting from arousal = 1 & valnce = 0. Each emotions is also linked to a color (see below)",
                        "emotion - color": [("alert", "orangered"),
                                            ("excited", "orange"),
                                            ("elated", "gold"),
                                            ("happy", "yellow"),
                                            ("content", "yellowgreen"),
                                            ("serene", "limegreen"),
                                            ("relaxed", "green"),
                                            ("calm", "seagreen"),
                                            ("fatigued", "aquamarine"),
                                            ("lethargic", "lightblue"),
                                            ("depressed", "steelblue"),
                                            ("sad", "blue"),
                                            ("upset", "blueviolet"),
                                            ("stressed", "violet"),
                                            ("nervous", "hotpink"),
                                            ("tens", "red")] 
                    } 
                },
                
                "Sound Characteristics": {
                    "timbre": {
                        "value": timbre_predicted,
                        "description": "Predicted timbre of the music",
                        "labels": timbre_labels
                    },
                    "instrument": {
                        "value": instrument_predicted,
                        "description": "Predicted instrument playing in the music",
                        "labels": instrument_labels
                    },
                    "pitch":{
                        "pitch": pitch,
                        "pitch confidence": pitch_confidence_pct,
                        "description": "estimated pitch and pitch confidence in percentage"
                    },
                    "harmonicity":{
                        "harmonicity": harmonicity_pct,
                        "description": "harmonicity given the spectral peaks of the signal, where the first peak is the fundamental frequency",
                        "more in detail": "The inharmonicity is computed as an energy weighted divergence of the spectral components from their closest multiple of the fundamental frequency. Then is reverse to find the harmonicity",
                        "range": "100 is a purely harmonic signal, 0 an inharmonic signal"
                    }
                }
            }

            # Construct the path to the JSON file
            json_file_path = os.path.join(json_dir, f"{os.path.splitext(file_name)[0]}.json")

            # Save the results to the JSON file with indentation
            try:
                with open(json_file_path, "w") as json_file:
                    json.dump(results, json_file, indent=4)
            except Exception as e:
                print(f"Error saving JSON file: {str(e)}")
            else:
                # Read the JSON file and print its contents with indentation
                try:
                    with open(json_file_path, "r") as json_file:
                        json_data = json.load(json_file)
                        print("Results for", file_name)
                        print(json.dumps(json_data, indent=4))
                except Exception as e:
                    print(f"Error reading JSON file: {str(e)}")