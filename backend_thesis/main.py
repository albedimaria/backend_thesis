import os, json
from utils_features.featureProcessing import process_audio_file
from featureJson import create_results_json
from paths.pathsToFolders import dir_path, json_dir


# Create an empty list to store the results for each audio file
all_results = []

def main():
    all_results = []

    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            if file_name.lower().endswith('.mp3') or file_name.lower().endswith('.wav'):
                file_path = os.path.join(root, file_name)
                print("Loading file:", file_path)
                
                # Process the audio file and append the results to the list
                results = process_audio_file(file_path, file_name)
                all_results.append(results)

    json_results_path = os.path.join(json_dir, 'all_results.json')

    # Save all the results to a JSON file
    with open(json_results_path, 'w') as json_file:
        json.dump(all_results, json_file, indent=4)
        
    # Create a separate JSON with the explanation of the data
    explanation = create_results_json() 
    
    json_explanation_path = os.path.join(json_dir, 'explanation.json')
    
    # Save the explanation JSON to a separate file
    with open(json_explanation_path, 'w') as json_file:
        json.dump(explanation, json_file, indent=4)

if __name__ == "__main__":
    main()

    
    
    

# audio_og, audio, cnn_audio = load_audio(file_path)

                # # BPM
                # global_bpm, local_bpm, local_probs = bpm_model(cnn_audio)
                # global_bpm_square_16, _, _ = bpm_model_square_16(cnn_audio)
                # global_bpm_temp_16, _, _ = bpm_model_temp_16(cnn_audio)
                # global_bpm_percival = esstd.PercivalBpmEstimator()(audio_og)
                # # global_bpm_degara, _, _ = esstd.BeatTrackerDegara()(resampled_audio)
                # global_bpm_ext, _, _, _, _ = esstd.RhythmExtractor2013()(audio_og)
                
                # # Assign a tempo label based on the global BPM range
                # tempo_label = assign_tempo_label(global_bpm)

                # # VA CALC
                # valence, arousal = calculate_valence_arousal(audio, va_embedding_model, va_model)
            
                # # COLOR
                # color, emotion = get_sector_color_label(valence, arousal, threshold=THRESHOLD)

                # # KEY
                # tonal_extractor = esstd.TonalExtractor()
                # tonal_features = tonal_extractor(audio_og)
                # key = {
                #         "estimated_key": tonal_features[9],
                #         "key_scale": tonal_features[10],
                #         "key_strength": round(tonal_features[11], 3)
                #     }
                

                # # MOOD

                # # Get the mood label with the corresponding index
                # predicted_mood = predict_label(audio, mood_and_timbre_embedding_model, mood_model, mood_labels)


                # # TIMBRE
                # timbre_predicted = predict_label(audio, mood_and_timbre_embedding_model, timbre_model, timbre_labels)


                # # INSTRUMENT
                # instrument_predicted = predict_label(audio, instrument_embedding_model, instrument_model, instrument_labels)


                # # DANCEABILITY

                # activations = danceability_activations(audio)
                # # Get the danceability probability from the activations array
                # danceability_prob_decimal = activations.mean(axis=0)[dance_index]
                # danceability_prob_percent = f'{100 * danceability_prob_decimal:.1f}%'


                # # DYNAMIC COMPLEXITY

                # # Predict the dynamic complexity using a machine learning model
                # dynamic_complexity, global_loudness = esstd.DynamicComplexity()(audio_og)
                # # Normalize the dynamic complexity coefficient by the global loudness level estimate
                # normalized_dynamic_complexity = dynamic_complexity / abs(global_loudness)        


                # # HARMONICITY AND PITCH

                # pitch, pitch_confidence_pct, harmonicity, harmonicity_pct = extract_harmonicity_and_pitch(audio_og)
                
                

                # Call the function to create the JSON object
                # results = create_results_json(file_name, global_bpm, global_bpm_square_16, global_bpm_temp_16,
                #                             global_bpm_percival, global_bpm_ext, key, predicted_mood, tempo_label,
                #                             danceability_prob_percent, normalized_dynamic_complexity, global_loudness,
                #                             valence, arousal, color, emotion, timbre_predicted, instrument_predicted,
                #                             pitch, pitch_confidence_pct, harmonicity_pct)


                # # Construct the path to the JSON file
                # json_file_path = os.path.join(json_dir, f"{os.path.splitext(file_name)[0]}.json")

                # # Save the results to the JSON file with indentation
                # try:
                #     with open(json_file_path, "w") as json_file:
                #         json.dump(results, json_file, indent=4)
                # except Exception as e:
                #     print(f"Error saving JSON file: {str(e)}")
                # else:
                #     # Read the JSON file and print its contents with indentation
                #     try:
                #         with open(json_file_path, "r") as json_file:
                #             json_data = json.load(json_file)
                #             print("Results for", file_name)
                #             print(json.dumps(json_data, indent=4))
                #     except Exception as e:
                #         print(f"Error reading JSON file: {str(e)}")