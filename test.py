import json
import os

from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf


INPUT_FILE = "data/le.wav"
MANIFEST_FILE = "MANIFEST_FILE.json"

meta = {
    "audio_filepath": INPUT_FILE,
    "offset": 0,
    "duration": None,
    "label": "infer",
    "text": '-',
    "num_speakers": None,
    "rttm_filepath": None,
    "uem_filepath": None
}

with open(MANIFEST_FILE, 'w') as fp:
    json.dump(meta, fp)
    fp.write('\n')


OUTPUT_DIR = os.getcwd()
MODEL_CONFIG = "config.yaml"

config = OmegaConf.load(MODEL_CONFIG)
config.diarizer.manifest_filepath = MANIFEST_FILE
config.diarizer.out_dir = OUTPUT_DIR
config.diarizer.oracle_vad = False
config.diarizer.clustering.parameters.oracle_num_speakers = False

sd_model = ClusteringDiarizer(cfg=config)

sd_model.diarize()
