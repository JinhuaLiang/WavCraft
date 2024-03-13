import argparse
from wavcraft.apis import _DECODE_WATERMARK


parser = argparse.ArgumentParser()
parser.add_argument("--wav-path", type=str, help="Path to the audio file.")
args = parser.parse_args()

_DECODE_WATERMARK(args.wav_path, sample_rate=16000)