import os
import glob
import time
import json
import numpy as np
import essentia.standard as esstd
import essentia.streaming as esstr
import scipy.spatial.distance as dist


# Define the directory path where the audio files are stored
dir_path = '/home/albertodimaria/thesis/test_analysis'

# Define the path to the ml_models directory
model_path = '/home/albertodimaria/thesis/ml_models/'

frame_size = 2048
hop_size = 1024
sample_rate = 44100
mel_bands = 40
mel_min = 0
mel_max = sample_rate/2


def mel_spec_calc(audio, frame_size, hop_size, sample_rate, mel_bands, mel_min, mel_max):
    
    # Compute Mel spectrogram
    window = esstd.Windowing(type="hann")
    spectrum = esstd.Spectrum(size=frame_size)
    mel_bands = esstd.MelBands(
        numberBands=mel_bands,
        sampleRate=sample_rate,
        lowFrequencyBound=mel_min,
        highFrequencyBound=mel_max
    )
    mel_spec = []
    for frame in esstd.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
        mel_bands_values = mel_bands(spectrum(window(frame)))
        mel_spec.append(mel_bands_values)
    
    (mel_bands_values).T
    
    # Convert to numpy array and take the logarithm
    mel_spec = np.array(mel_spec)
    log_mel_spec = np.log(mel_spec + 1e-9)
    return log_mel_spec



def distance_matrix(x, y):
    N = x.shape[0]
    M = y.shape[0]
    dist_mat = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            dist_mat[i,j] = dist.cosine(x[i] - y[j])
    return dist_mat

def dp(dist_math):
    N, M = dist_math.shape
    
    #initialize the cost matrix
    cost_mat = np.zeros((N +1 , M +1))
    for i in range(1, N+1):
        cost_mat[i,0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf
        
    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i,j],      # match(0)
                cost_mat[i,j+1],    # insertion(1)
                cost_mat[i+1,j]]    # deletion(2)
            i_penalty = np.argmin(penalty)
            cost_mat[i +1 , j +1] = dist_math[i,j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty
            
    # traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i,j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i,j]
        if tb_type == 0:
            # match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # insertion 
            i = i - 1
        elif tb_type == 2:
            # deletion
            j = j - 1
        path.append((i,j))
        
    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    
    return (path[::-1], cost_mat)


prev_audio = None
prev_mel_spec = None
prev_file_name = None

for file_name in os.listdir(dir_path):
    # Check if item is a file
    if os.path.isfile(os.path.join(dir_path, file_name)):
        # Check if file is MP3 or WAV
        if file_name.lower().endswith('.mp3') or file_name.lower().endswith('.wav'):
            file_path = os.path.join(dir_path, file_name)
            print("Loading file:", file_path)

            # load the current audio
            audio_og = esstd.MonoLoader(filename=file_path, sampleRate=16000)()
            
            # Resample the audio signal to 44.1kHz
            resampler_1 = esstd.Resample(inputSampleRate=16000, outputSampleRate=44100)
            audio = resampler_1(audio_og)

            # compute the mel spectrogram
            mel_spec = mel_spec_calc(audio, frame_size=frame_size, hop_size=hop_size, sample_rate=sample_rate, mel_bands=mel_bands, mel_min=0, mel_max=mel_max)
            
            if prev_audio is not None:
                # compute DTW between the current and previous audio
                dist_mat = dist.cdist(prev_mel_spec, mel_spec, "cosine")
                path, cost_mat = dp(dist_mat)
                alignment_cost_norm = cost_mat[-1, -1] / (len(prev_mel_spec) + len(mel_spec))

                print("Alignment cost between {} and {}: {:.4f}".format(prev_file_name, file_name, alignment_cost_norm))

                
            prev_mel_spec = mel_spec
            prev_audio = audio
            prev_file_name = file_name
           
            

         


       
            

"""
            
            json_dir = "/home/albertodimaria/thesis/json_results"

            # Create a new directory for the JSON file with the same name as the original file
            file_dir = os.path.splitext(file_path)[0] + '_JSON'
            os.makedirs(file_dir, exist_ok=True)

            # Construct the path to the JSON file
            json_file_path = os.path.join(file_dir, f"{os.path.splitext(file_name)[0]}.json")

            results = {
                "file name": file_name,
               
            }

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
                    print(f"Error reading JSON file: {str(e)}")"""