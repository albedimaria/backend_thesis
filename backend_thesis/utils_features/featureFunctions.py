import numpy as np
from featureLists import sector_colors, sector_labels, tempo_ranges

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



# Define a function to get the sector color and label for a given valence and arousal pair
def get_sector_color_label(valence, arousal, threshold):
    # Calculate the polar coordinates of the point
    theta = np.arctan2(valence, arousal)

    # Shift the angle to fall within the range [0, 2*pi)
    if theta < 0:
        theta += 2 * np.pi

    # If the distance is below the threshold, return 'neutral' color and label
    if abs(valence) < threshold and abs(arousal) < threshold:
        return 'white', 'neutral'

    # Define the number of sectors
    num_sectors = 16

    # Define the sector angles
    sector_angles = 2 * np.pi / num_sectors
    
    # Find the sector that the point falls into
    sector_index = int(np.floor(theta / sector_angles))

    # Get the color and label of the sector
    sector_color = sector_colors[sector_index]
    sector_label = sector_labels[sector_index]

    return sector_color, sector_label


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