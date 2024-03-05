# import nemo.collections.asr as nemo_asr


# # Загрузка предварительно обученной модели диаризации говорящих
# SpeakerDiarizer
# model = nemo_asr.models.SpeakerDiarizer.from_pretrained(model_name="titanet_large")

# # Путь к вашему аудиофайлу
# audio_filepath = "data/le.wav"

# # Применение модели к аудиофайлу
# # Это сгенерирует аннотации диаризации в формате RTTM
# model.diarize(paths2audio_files=[audio_filepath])

# Manifest.json
# Указать спикеров -> num_speakers = None


import nemo.collections.asr as nemo_asr


audio_file_path = "data/le.wav"

model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")

predictions = model.transcribe(path2audio_files=[audio_file_path])

print(predictions)
