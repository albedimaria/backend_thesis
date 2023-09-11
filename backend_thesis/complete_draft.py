import os
import glob
import time
import json
import numpy as np
import essentia.standard as esstd
import essentia.streaming as esstr

from utils_features.featureFunctions import predict_label, get_sector_color_label, assign_tempo_label, calculate_valence_arousal

from utils_features.featureLists import tempo_ranges

# bpm, va, danceability
from utils_features.featureModelsNumeric import bpm_model, bpm_model_square_16, bpm_model_temp_16, va_embedding_model, va_model, dance_index, danceability_activations

# mood, timbre, instrumental
from utils_features.featureModelsClasses import mood_and_timbre_embedding_model, mood_model, mood_labels, timbre_model, timbre_labels, instrument_model, instrument_embedding_model, instrument_labels

from pathFolders import dir_path, json_dir

from utils_features.pitchAndHarm import extract_harmonicity_and_pitch

from featureJson import create_results_json


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

            # VA CALC
            valence, arousal = calculate_valence_arousal(audio, va_embedding_model, va_model)
           
            # COLOR
            color, emotion = get_sector_color_label(valence, arousal, threshold=0.2)

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

            pitch, pitch_confidence_pct, harmonicity, harmonicity_pct = extract_harmonicity_and_pitch(audio_og)

           # Call the function to create the JSON object
            results = create_results_json(file_name, global_bpm, global_bpm_square_16, global_bpm_temp_16,
                                        global_bpm_percival, global_bpm_ext, key, predicted_mood, tempo_label,
                                        danceability_prob_percent, normalized_dynamic_complexity, global_loudness,
                                        valence, arousal, color, emotion, timbre_predicted, instrument_predicted,
                                        pitch, pitch_confidence_pct, harmonicity_pct)


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