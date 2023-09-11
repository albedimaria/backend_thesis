from featureLists import tempo_ranges
from featureModelsClasses import timbre_labels, mood_labels, instrument_labels

def create_results_json(file_name, global_bpm, global_bpm_square_16, global_bpm_temp_16,
                        global_bpm_percival, global_bpm_ext, key, predicted_mood, tempo_label,
                        danceability_prob_percent, normalized_dynamic_complexity, global_loudness,
                        valence, arousal, color, emotion, timbre_predicted, instrument_predicted,
                        pitch, pitch_confidence_pct, harmonicity_pct):
    
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