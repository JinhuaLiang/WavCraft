import os
import sys
import yaml
import logging
import torch
import librosa
import numpy as np
from flask import Flask, request, jsonify
from scipy.io.wavfile import write

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavcraft.utils import get_service_port


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


"""
Initalize the AudioSep model here
"""
def inference(model, audio_file, text, output_file, device='cuda', use_chunk=False):
    print(f'Separate audio from [{audio_file}] with textual query [{text}]')
    mixture, fs = librosa.load(audio_file, sr=32000, mono=True)
    with torch.no_grad():
        text = [text]

        conditions = model.query_encoder.get_query_embed(
            modality='text',
            text=text,
            device=device
        )

        input_dict = {
            "mixture": torch.Tensor(mixture)[None, None, :].to(device),
            "condition": conditions,
        } 

        if use_chunk:
            foreground = model.ss_model.chunk_inference(input_dict)
            foreground = np.squeeze(foreground)
        else:
            foreground = model.ss_model(input_dict)["waveform"]
            foreground = foreground.squeeze(0).squeeze(0).data.cpu().numpy()

        background = mixture - foreground

        filedir, filename = os.path.split(output_file)
        fg_filepath = os.path.join(filedir, "fg_"+filename)
        bg_filepath = os.path.join(filedir, "bg_"+filename)

        write(fg_filepath, 32000, np.round(foreground * 32767).astype(np.int16))
        print(f'Write separated audio to [{fg_filepath}]')

        write(bg_filepath, 32000, np.round(background * 32767).astype(np.int16))
        print(f'Write the background audio to [{bg_filepath}]')

import sys
sys.path.append("ext/AudioSep")
from ss_pipeline import build_audiosep
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ss = build_audiosep(
      config_yaml='ext/AudioSep/config/audiosep_base.yaml', 
      checkpoint_path='ext/AudioSep/checkpoint/audiosep_base_4M_steps.ckpt', 
      device=device)

logging.info('AudioSep is loaded ...')


app = Flask(__name__)


@app.route('/source_separate', methods=['POST'])
def source_separate():
    # Receive the text from the POST request
    data = request.json
    wav_path = data['wav_path']
    text = data["text"]
    output_wav = data.get('output_wav', 'out.wav')

    logging.info(f"Separate '{text}' from {wav_path} ...")

    try:
        inference(ss, wav_path, text, output_wav, device)
        
        # Return success message and the filename of the generated audio
        return jsonify({'message': f'Sucessful separation from {wav_path}'})

    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'API error': str(e)}), 500


if __name__ == '__main__':
    service_port = get_service_port("AUDIOSEP_SERVICE_PORT")
    # We disable multithreading to force services to process one request at a time and avoid CUDA OOM
    app.run(debug=False, threaded=False, port=service_port)