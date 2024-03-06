import json
import torch
import torchaudio
import nemo.collections.asr as nemo_asr
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
model.to(device)
model.eval()

audio_path = "/home/user/Desktop/projects/nemo_diarization/data/mono.wav"
audio_signal, sample_rate = torchaudio.load(audio_path)

if audio_signal.shape[0] == 2:
    audio_signal = torch.mean(audio_signal, dim=0, keepdim=True)

if sample_rate != model.preprocessor._sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=model.preprocessor._sample_rate)
    audio_signal = resampler(audio_signal)

if len(audio_signal.shape) == 1:
    audio_signal = audio_signal.unsqueeze(0)







def is_speech_segment(segment, threshold=0.01):
    """
    Determine if the segment contains speech based on RMS energy.
    segment: Tensor containing the audio signal.
    threshold: Energy threshold for determining speech presence.
    """
    rms_energy = torch.sqrt(torch.mean(segment ** 2))
    return rms_energy > threshold


segment_length = 1.0
overlap = 0.5
segment_length_samples = int(model.preprocessor._sample_rate * segment_length)
overlap_samples = int(model.preprocessor._sample_rate * overlap)

all_embeddings = []
start = 0
while start + segment_length_samples <= audio_signal.shape[1]:
    end = start + segment_length_samples
    segment = audio_signal[:, start:end].to(device)
    
    if segment.shape[1] < segment_length_samples:
        break
    
    signal_length = torch.tensor([segment.shape[1]]).to(device)
    with torch.no_grad():
        embeddings = model.forward(input_signal=segment, input_signal_length=signal_length)[1]  # Using the second tensor for embeddings
        all_embeddings.append(embeddings.detach().cpu().numpy())
    
    start += (segment_length_samples - overlap_samples)

# all_embeddings = []
# start = 0
# while start + segment_length_samples <= audio_signal.shape[1]:
#     end = start + segment_length_samples
#     segment = audio_signal[:, start:end]
    
#     # Apply VAD (skipping segment if it doesn't contain speech)
#     if not is_speech_segment(segment, threshold=0.01):  # Adjust the threshold based on your needs
#         start += (segment_length_samples - overlap_samples)
#         continue
    
#     # If VAD indicates speech, proceed with processing
#     segment = segment.to(device)
    
#     # Ensure segment is not too short
#     if segment.shape[1] < segment_length_samples:
#         break  # Break the loop if the remaining audio is shorter than a full segment
    
#     signal_length = torch.tensor([segment.shape[1]]).to(device)
#     with torch.no_grad():  # Inference only, no gradients
#         embeddings = model.forward(input_signal=segment, input_signal_length=signal_length)[1]
#         all_embeddings.append(embeddings.detach().cpu().numpy())
    
#     start += (segment_length_samples - overlap_samples)



# Combine embeddings from all segments into a single NumPy array
embeddings_np = np.vstack(all_embeddings)

n_clusters = min(len(all_embeddings), 3)

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings_np)
speaker_labels = kmeans.labels_

timestamps_and_speakers = [(i * (segment_length - overlap), speaker_labels[i]) for i in range(len(speaker_labels))]

for start_time, speaker in timestamps_and_speakers:
    print(f"Start Time: {start_time:.2f}s - Speaker: {speaker}")
