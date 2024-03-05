import json
import torch
import torchaudio
import nemo.collections.asr as nemo_asr

from sklearn.cluster import KMeans



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
model.to(device)

audio_path = "/home/user/Desktop/projects/nemo_diarization/data/new.wav"
audio_length = 30.306644

# audio_signal, audio_signal_len = model.preprocessor(input_signal=your_audio_data, length="30.306644")
# embeddings = model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
# manifest = json.dumps({"audio_filepath": audio_path, "duration": audio_path, "label": "Speaker"})

audio_signal, sample_rate = torchaudio.load(audio_path)

if audio_signal.shape[0] == 2:
    audio_signal = torch.mean(audio_signal, dim=0, keepdim=True)

if sample_rate != model.preprocessor._sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=model.preprocessor._sample_rate)
    audio_signal = resampler(audio_signal)

if len(audio_signal.shape) == 1:
    audio_signal = audio_signal.unsqueeze(0)

signal_length = torch.tensor([audio_signal.shape[1]])
audio_signal = audio_signal.to(device)
embeddings = model.forward(input_signal=audio_signal, input_signal_length=signal_length)
print("embeddings", embeddings)
