import os
import sys
import yaml
import logging
import torch
import nltk
import torchaudio
import torchaudio.transforms as T
from torchaudio.transforms import SpeedPerturbation

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavcraft.apis import _WRITE_AUDIO, _LOUDNESS_NORM
from wavcraft.utils import fade, get_service_port
from flask import Flask, request, jsonify


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
Initialize the AudioCraft models here
"""
from audiocraft.models import AudioGen, MusicGen
tta_model_size = config['AudioCraft']['tta_model_size']
tta_model = AudioGen.get_pretrained(f'facebook/audiogen-{tta_model_size}')
logging.info(f'AudioGen ({tta_model_size}) is loaded ...')

ttm_model_size = config['AudioCraft']['ttm_model_size']
ttm_model = MusicGen.get_pretrained(f'facebook/musicgen-{ttm_model_size}')
logging.info(f'MusicGen ({ttm_model_size}) is loaded ...')


"""
Initialize the BarkModel here
"""
from transformers import BarkModel, AutoProcessor
import json

# Load voice map
with open("wavcraft/voice_preset/voice_map.json", 'r') as f:
    voice_map = json.load(f)

SPEED = float(config['Text-to-Speech']['speed'])
speed_perturb = SpeedPerturbation(32000, [SPEED])
tts_model = BarkModel.from_pretrained("suno/bark")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tts_model = tts_model.to(device)
tts_model = tts_model.to_bettertransformer()    # Flash attention
SAMPLE_RATE = tts_model.generation_config.sample_rate
SEMANTIC_TEMPERATURE = 0.9
COARSE_TEMPERATURE = 0.5
FINE_TEMPERATURE = 0.5
processor = AutoProcessor.from_pretrained("suno/bark")
logging.info('Bark model is loaded ...')


app = Flask(__name__)


@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    # Receive the text from the POST request
    data = request.json
    text = data['text']
    length = float(data.get('length', 5.0))
    volume = float(data.get('volume', -35))
    output_wav = data.get('output_wav', 'out.wav')

    logging.info(f'TTA (AudioGen): Prompt: {text}, length: {length} seconds, volume: {volume} dB')
    
    try:
        tta_model.set_generation_params(duration=length)
        wav = tta_model.generate([text])  
        wav = torchaudio.functional.resample(wav, orig_freq=16000, new_freq=32000)

        wav = wav.squeeze().cpu().detach().numpy()
        wav = fade(_LOUDNESS_NORM(wav, volume=volume))
        _WRITE_AUDIO(wav, name=output_wav)

        # Return success message and the filename of the generated audio
        return jsonify({'message': f'Text-to-Audio generated successfully | {text}', 'file': output_wav})

    except Exception as e:
        return jsonify({'API error': str(e)}), 500


@app.route('/generate_music', methods=['POST'])
def generate_music():
    # Receive the text from the POST request
    data = request.json
    text = data['text']
    melody_path = data.get('melody', None)
    length = float(data.get('length', 5.0))
    volume = float(data.get('volume', -35))
    sample_rate = int(data.get('sr', 32000))
    output_wav = data.get('output_wav', 'out.wav')

    logging.info(f'TTM (MusicGen): Prompt: {text}, length: {length} seconds, volume: {volume} dB')


    try:
        ttm_model.set_generation_params(duration=length)

        if melody_path is None:
            print("Use generate")
            wav = ttm_model.generate([text])

        else:
            print("Use generate_with_chroma")
            melody, sr = torchaudio.load(melody_path)
            # Resample the audio if sr does not match sample_rate
            if sr != sample_rate:
                resampler = T.Resample(sr, sample_rate, dtype=melody.dtype)
                melody = resampler(melody)
            # Generates using the melody from the given audio and the provided descriptions.
            wav = ttm_model.generate_with_chroma([text], melody[None].expand(1, -1, -1), sample_rate)
            
        wav = wav[0][0].cpu().detach().numpy()
        wav = fade(_LOUDNESS_NORM(wav, volume=volume))
        _WRITE_AUDIO(wav, name=output_wav)

        # Return success message and the filename of the generated audio
        return jsonify({'message': f'Text-to-Music generated successfully | {text}', 'file': output_wav})

    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'API error': str(e)}), 500


@app.route('/generate_speech', methods=['POST'])
def generate_speech():
    # Receive the text from the POST request
    data = request.json
    text = data['text']
    speaker_id = data['speaker_id']
    volume = float(data.get('volume', -35))
    output_wav = data.get('output_wav', 'out.wav')

    speaker_npz = voice_map[speaker_id]["npz_path"]
    
    logging.info(f'TTS (Bark): Speaker: {speaker_id}, Volume: {volume} dB, Prompt: {text}')

    try:   
        # Generate audio using the global pipe object
        text = text.replace('\n', ' ').strip()
        sentences = nltk.sent_tokenize(text)
        silence = torch.zeros(int(0.1 * SAMPLE_RATE), device=device).unsqueeze(0)  # 0.1 second of silence

        pieces = []
        for sentence in sentences:
            inputs = processor(sentence, voice_preset=speaker_npz).to(device)
            # NOTE: you must run the line below, otherwise you will see the runtime error
            # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
            inputs['history_prompt']['coarse_prompt'] = inputs['history_prompt']['coarse_prompt'].transpose(0, 1).contiguous().transpose(0, 1)

            with torch.inference_mode():
                # TODO: min_eos_p?
                output = tts_model.generate(
                    **inputs,
                    do_sample = True,
                    semantic_temperature = SEMANTIC_TEMPERATURE,
                    coarse_temperature = COARSE_TEMPERATURE,
                    fine_temperature = FINE_TEMPERATURE
                )

            pieces += [output, silence]

        result_audio = torch.cat(pieces, dim=1)
        wav_tensor = result_audio.to(dtype=torch.float32).cpu()
        wav = torchaudio.functional.resample(wav_tensor, orig_freq=SAMPLE_RATE, new_freq=32000)
        wav = speed_perturb(wav.float())[0].squeeze(0)
        wav = wav.numpy()
        wav = _LOUDNESS_NORM(wav, volume=volume)
        _WRITE_AUDIO(wav, name=output_wav)

        # Return success message and the filename of the generated audio
        return jsonify({'message': f'Text-to-Speech generated successfully | {speaker_id}: {text}', 'file': output_wav})

    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'API error': str(e)}), 500
    

if __name__ == '__main__':
    service_port = get_service_port("AUDIOCRAFT_SERVICE_PORT")
    # We disable multithreading to force services to process one request at a time and avoid CUDA OOM
    app.run(debug=False, threaded=False, port=service_port)