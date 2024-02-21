import nemo.collections.asr as nemo_asr


# Загрузка предварительно обученной модели диаризации говорящих
model = nemo_asr.models.SpeakerDiarizer.from_pretrained(model_name="titanet_large")

# Путь к вашему аудиофайлу
audio_filepath = "data/le.wav"

# Применение модели к аудиофайлу
# Это сгенерирует аннотации диаризации в формате RTTM
model.diarize(paths2audio_files=[audio_filepath])

# Manifest.json
# Указать спикеров -> num_speakers = None
