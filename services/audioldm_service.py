import os
import sys
import yaml
import math
import logging
import librosa
import numpy as np
from flask import Flask, request, jsonify
from scipy.io.wavfile import write

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavcraft.utils import get_service_port
from audioldm import build_model, super_resolution_and_inpainting


CACHE_DIR = os.getenv(
    "AUDIOLDM_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache/audioldm"))

EPS = 1e-5

with open('wavcraft/configs.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Configure the logging format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create a FileHandler for the log file
os.makedirs('services_logs', exist_ok=True)
log_filename = 'services_logs/Wav-API.log'
file_handler = logging.FileHandler(log_filename, mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add the FileHandler to the root logger
logging.getLogger('').addHandler(file_handler)


audioldm = build_model(model_name=config["AudioLDM"]["model_size"])
logging.info('AudioLDM is loaded ...')


app = Flask(__name__)


@app.route('/audio_inpaint', methods=['POST'])
def audio_inpaint():
    # Receive the text from the POST request
    data = request.json
    wav_path = data['wav_path']
    text = data["text"]
    duration = data["duration"] + EPS # avoid zero division
    onset = data["onset"] / duration
    offset = data["offset"] / duration

    sample_rate = data.get('sample_rate', 32000)
    guidance_scale = data.get('guidance_scale', 2.5)
    ddim_steps = data.get('ddim_steps', 200)
    random_seed = data.get('seed', 42)
    output_wav = data.get('output_wav', 'out.wav')
    logging.info(f"Inpaint {wav_path} with the input '{text}'...")

    try:
        # target_duration = math.ceil(data["duration"] / 2.5) * 2.5
        target_duration = data["duration"]
        waveform = super_resolution_and_inpainting(
            audioldm,
            text, # The text prompt for inpainting generation
            wav_path, # This audio will be padded to 10.242 seconds and perform inpainting
            time_mask_ratio_start_and_end=(onset, offset), # This is a ratio for inpainting at a scale of 10.242 seconds
            seed=random_seed,
            duration=target_duration,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_candidate_gen_per_text=1,
            batchsize=1,
        )

        if sample_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=16000, target_sr=sample_rate)
        # Write audio to `output_wav` with `sample_rate`
        write(output_wav, sample_rate, np.round(waveform[:int(duration*sample_rate)] * 32767).astype(np.int16))
        
        # Return success message and the filename of the generated audio
        return jsonify({'message': f'Sucessful infill {data["onset"]}-{data["offset"]}s content in {wav_path}'})

    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'API error': str(e)}), 500


if __name__ == '__main__':
    service_port = get_service_port("AUDIOLDM_SERVICE_PORT")
    # We disable multithreading to force services to process one request at a time and avoid CUDA OOM
    app.run(debug=False, threaded=False, port=service_port)