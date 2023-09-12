import essentia.standard as esstd
from .featureFunctions import (
    load_audio,
    predict_label,
    get_sector_color_label,
    assign_tempo_label,
    calculate_valence_arousal,
)

from .featureModelsNumeric import (
    bpm_model,
    bpm_model_square_16,
    bpm_model_temp_16,
    dance_index,
    danceability_activations,
    va_model,
    va_embedding_model
)
from .featureModelsClasses import (
    mood_and_timbre_embedding_model,
    instrument_embedding_model,
    mood_model,
    timbre_model,
    instrument_model,
    mood_labels,
    timbre_labels,
    instrument_labels
)
from paths.pathsToFolders import dir_path, json_dir
from .pitchAndHarm import extract_harmonicity_and_pitch
from featureJson import create_results_json


# Constants
# white/noMood threshold in [0,1]
THRESHOLD = 0.2

def process_bpm(audio, cnn_audio):
    global_bpm, local_bpm, local_probs = bpm_model(cnn_audio)
    global_bpm_square_16, _, _ = bpm_model_square_16(cnn_audio)
    global_bpm_temp_16, _, _ = bpm_model_temp_16(cnn_audio)
    global_bpm_percival = esstd.PercivalBpmEstimator()(audio)
    global_bpm_ext, _, _, _, _ = esstd.RhythmExtractor2013()(audio)
    
    tempo_label = assign_tempo_label(global_bpm)
    
    return global_bpm, global_bpm_square_16, global_bpm_temp_16, global_bpm_percival, global_bpm_ext, tempo_label

def process_valence_arousal(audio):
    valence, arousal = calculate_valence_arousal(audio, va_embedding_model, va_model)
    color, emotion = get_sector_color_label(valence, arousal, threshold=THRESHOLD)
    return valence, arousal, color, emotion

def process_key(audio_og):
    tonal_extractor = esstd.TonalExtractor()
    tonal_features = tonal_extractor(audio_og)
    key = {
        "estimated_key": tonal_features[9],
        "key_scale": tonal_features[10],
        "key_strength": round(tonal_features[11], 3)
    }
    return key

def process_mood_and_timbre(audio):
    predicted_mood = predict_label(audio, mood_and_timbre_embedding_model, mood_model, mood_labels)
    timbre_predicted = predict_label(audio, mood_and_timbre_embedding_model, timbre_model, timbre_labels)
    return predicted_mood, timbre_predicted

def process_instrument(audio):
    instrument_predicted = predict_label(audio, instrument_embedding_model, instrument_model, instrument_labels)
    return instrument_predicted

def process_danceability(audio):
    activations = danceability_activations(audio)
    danceability_prob_decimal = activations.mean(axis=0)[dance_index]
    danceability_prob_percent = f'{100 * danceability_prob_decimal:.1f}%'
    return danceability_prob_percent

def process_dynamic_complexity(audio_og):
    dynamic_complexity, global_loudness = esstd.DynamicComplexity()(audio_og)
    normalized_dynamic_complexity = dynamic_complexity / abs(global_loudness)
    return normalized_dynamic_complexity, global_loudness

def process_harmonicity_and_pitch(audio_og):
    pitch, pitch_confidence_pct, harmonicity, harmonicity_pct = extract_harmonicity_and_pitch(audio_og)
    return pitch, pitch_confidence_pct, harmonicity_pct


def process_audio_file(file_path, file_name):
    audio_og, audio, cnn_audio = load_audio(file_path)

    global_bpm, global_bpm_square_16, global_bpm_temp_16, global_bpm_percival, global_bpm_ext, tempo_label = process_bpm(audio, cnn_audio)

    valence, arousal, color, emotion = process_valence_arousal(audio)

    key = process_key(audio_og)

    predicted_mood, timbre_predicted = process_mood_and_timbre(audio)

    instrument_predicted = process_instrument(audio)

    danceability_prob_percent = process_danceability(audio)

    normalized_dynamic_complexity, global_loudness = process_dynamic_complexity(audio_og)

    pitch, pitch_confidence_pct, harmonicity_pct = process_harmonicity_and_pitch(audio_og)

    results = {
            "file_name": file_name,
            
            "BPM": global_bpm,
            "BPM_2": global_bpm_square_16,
            "BPM_3": global_bpm_temp_16,
            "BPM_4": global_bpm_percival,
            "BPM_5": global_bpm_ext,
            "key": key,
            "mood": predicted_mood,
            "Tempo": tempo_label,
            "danceability": danceability_prob_percent,
            "dynamic_complexity_norm": round(normalized_dynamic_complexity, 3),
            "global_loudness_dB": round(global_loudness, 1),
            "valence": valence,
            "arousal": arousal,
            "color": color,
            "emotion": emotion,
            "timbre": timbre_predicted,
            "instrument": instrument_predicted,
            "pitch": pitch,
            "pitch confidence": pitch_confidence_pct,
            "harmonicity %": harmonicity_pct 
        }
    
    return results

    # # Create a JSON template with explanations
    # json_data = create_results_json(file_name, results)
    
    # return json_data