import essentia.streaming as ess

# Define your feature extractor
my_feature_extractor = ess.MyFeatureExtractor()

# Define your audio stream (load from a WAV file)
audio_stream = ess.MonoLoader(filename='test.wav', sampleRate=44100)()

# Define your feature stream
feature_stream = my_feature_extractor(audio_stream)

# Define a callback function to handle the extracted features
def handle_features(features):
    # Do something with the extracted features (e.g. send them over a network)
    print(features)

# Set the callback function for the feature stream
feature_stream.setCallback(handle_features)

# Run the processing pipeline
ess.run()


""" 
In this example, the `MonoLoader` block is used to load an audio file in WAV format and stream it to the processing pipeline. 
The processing pipeline then extracts features from the audio data in real-time, and the extracted features are handled by a callback function.
By keeping the audio data in WAV format throughout the processing pipeline, you can ensure that the quality of the audio is preserved and 
that the processing is optimized for real-time performance. 
"""