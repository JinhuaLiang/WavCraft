import time
import argparse

import wavcraft.utils as utils
import wavcraft.pipeline as pipeline

parser = argparse.ArgumentParser()
sub_parsers = parser.add_subparsers(dest="mode", help='Type of WavCraft to use')
# Basic mode
basic_parser = sub_parsers.add_parser("basic")
basic_parser.add_argument('-f', '--full', action='store_true', help='Go through the full process')
basic_parser.add_argument('--input-wav', nargs='+', default=[], help='a list of input wave paths')
basic_parser.add_argument('--input-text', type=str, help='input text or text file')
# gpt-4-0125-preview
basic_parser.add_argument('--model', type=str, default="gpt-4", help='ChatGPT model.')
basic_parser.add_argument('--session-id', type=str, default='', help='session id, if set to empty, system will allocate an id')
# Inspiration mode
inspire_parser = sub_parsers.add_parser("inspiration")
inspire_parser.add_argument('-f', '--full', action='store_true', help='Go through the full process')
inspire_parser.add_argument('--input-wav', nargs='+', default=[], help='a list of input wave paths')
inspire_parser.add_argument('--input-text', type=str, help='input text or text file')
inspire_parser.add_argument('--model', type=str, default="gpt-4", help='ChatGPT model.')
inspire_parser.add_argument('--session-id', type=str, default='', help='session id, if set to empty, system will allocate an id')

args = parser.parse_args()

if args.mode in ("basic", "inspiration"):
    if args.full:
        input_text = args.input_text
        input_wav = args.input_wav

        start_time = time.time()
        session_id = pipeline.init_session(args.session_id)
        api_key = utils.get_api_key()

        assert api_key != None, "Please set your openai_key in the environment variable."
        
        print(f"Session {session_id} is created.")

        pipeline.full_steps(session_id, input_wav, input_text, api_key, model=args.model, mode=args.mode)
        end_time = time.time()

        print(f"Audio editor took {end_time - start_time:.2f} seconds to complete.")
