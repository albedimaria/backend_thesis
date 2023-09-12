import essentia.standard as esstd
import os
import json

from paths.pathsToFolders import model_path


# mood 
mood_and_timbre_embedding_model = esstd.TensorflowPredictEffnetDiscogs(graphFilename=os.path.join(model_path, 'discogs_label_embeddings-effnet-bs64-1.pb'), output="PartitionedCall:1")
mood_model = esstd.TensorflowPredict2D(graphFilename=os.path.join(model_path, 'mtg_jamendo_moodtheme-effnet-discogs_label_embeddings-1.pb'))

with open(os.path.join(model_path, 'discogs_label_embeddings-effnet-bs64-1.json'), 'r') as json_file:
                metadata = json.load(json_file)
mood_labels = metadata['classes']


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