import os
import sys
import yaml
import random
import logging
import torch
import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify
from cog import BasePredictor, Input, Path
from audiosr import build_model
from audiosr import super_resolution as _super_resolution


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavcraft.apis import _WRITE_AUDIO
from wavcraft.utils import fade, get_service_port


os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")

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
Initialize the AudioSR models here
"""
class Predictor(BasePredictor):
    def setup(self, model_name="basic", device="auto"):
        self.model_name = model_name
        self.device = device
        self.sr = 48000
        self.audiosr = build_model(model_name=self.model_name, device=self.device)

    def predict(self,
        input_file: Path = Input(description="Audio to upsample"),
        output_path: Path = Input(description="Path to output audio"),
        ddim_steps: int = Input(description="Number of inference steps", default=50, ge=10, le=500),
        guidance_scale: float = Input(description="Scale for classifier free guidance", default=3.5, ge=1.0, le=20.0),
        seed: int = Input(description="Random seed. Leave blank to randomize the seed", default=None),
    ) -> np.ndarray:
        """Run a single prediction on the model"""
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Setting seed to: {seed}")

        waveform = _super_resolution(
            self.audiosr,
            input_file,
            seed=seed,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            latent_t_per_second=12.8
        )
        out_wav = (waveform[0] * 32767).astype(np.int16).T

        sf.write(output_path, data=out_wav, samplerate=48000)
        return Path(output_path)

sr_model = Predictor()
sr_model.setup()
logging.info('AudioSR model is loaded ...')


app = Flask(__name__)


@app.route('/super_resolution', methods=['POST'])
def super_resolution():
    # Receive the text from the POST request
    data = request.json
    wav_path = data['wav_path']
    ddim_steps = int(data.get('ddim_steps', 50))
    guidance_scale = float(data.get('guidance_scale', 3.5))
    seed = int(data.get('seed', 42))
    output_wav = data.get('output_wav', 'out.wav')

    logging.info(f"Super resolution: ddim_steps: {ddim_steps}, guidance_scale: {guidance_scale}.")

    try:
        sr_model.predict(
            wav_path,
            output_path=output_wav,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        # Return success message and the filename of the generated audio
        return jsonify({'message': 'Audio super resolution generated successfully', 'file': f"{output_wav}"})

    except Exception as e:
        return jsonify({'API error': str(e)}), 500


if __name__ == '__main__':
    service_port = get_service_port("AUDIOSR_SERVICE_PORT")
    # We disable multithreading to force services to process one request at a time and avoid CUDA OOM
    app.run(debug=False, threaded=False, port=service_port)