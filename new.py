import nemo.collections.asr as nemo_asr



# Path to your audio file
audio_file_path = "data/le.wav"

# Initialize the ClusteringDiarizer with the desired configuration
diarizer = nemo_asr.models.ClusteringDiarizer.from_pretrained(model_name="SpeakerNet_recognition")

# Configure the diarizer
# Note: You might need to adjust these parameters depending on your audio file and requirements
diarizer.manifest_filepath = audio_file_path
diarizer.out_dir = 'output_directory'  # Directory where the diarization outputs will be saved
diarizer.oracle_vad = True  # Set to True if you want to use oracle VAD

# Perform diarization
diarizer.diarize()