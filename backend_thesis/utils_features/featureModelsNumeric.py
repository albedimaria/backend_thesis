import essentia.standard as esstd
import os, json

from paths.pathsToFolders import model_path


# BPM - load pre-trained Tempo Classification model
bpm_model = esstd.TempoCNN(graphFilename=os.path.join(model_path, 'deeptemp-k4-3.pb'))
bpm_model_square_16 = esstd.TempoCNN(graphFilename=os.path.join(model_path, 'deepsquare-k16-3.pb'))
bpm_model_temp_16 = esstd.TempoCNN(graphFilename=os.path.join(model_path, 'deeptemp-k16-3.pb'))


# VALENCE/AROUSAL pairs
va_embedding_model = esstd.TensorflowPredictMusiCNN(graphFilename=os.path.join(model_path, 'msd-musicnn-1.pb'), output="model/dense/BiasAdd")
va_model = esstd.TensorflowPredict2D(graphFilename=os.path.join(model_path, 'deam-musicnn-msd-2.pb'), output="model/Identity")

# Danceability
with open(os.path.join(model_path, 'danceability-vggish-audioset-1.json'), 'r') as json_file:
                metadata = json.load(json_file)
dance_index = metadata['classes'].index('danceable')
danceability_activations = esstd.TensorflowPredictVGGish(graphFilename=os.path.join(model_path, 'danceability-vggish-audioset-1.pb'))
