import time
import argparse

import wavcraft.utils as utils
import wavcraft.pipeline as pipeline

parser = argparse.ArgumentParser()
sub_parsers = parser.add_subparsers(dest="mode")
# Basic mode
basic_parser = sub_parsers.add_parser("basic")
basic_parser.add_argument('-f', '--full', action='store_true', help='Go through the full proces')
basic_parser.add_argument('-c', '--chat', action='store_true', help='Chat with WavCraft.')
basic_parser.add_argument('--session-id', type=str, default='', help='session id, if set to empty, system will allocate an id')
# Inspiration mode
inspire_parser = sub_parsers.add_parser("inspiration")
inspire_parser.add_argument('-f', '--full', action='store_true', help='Go through the full process')
inspire_parser.add_argument('-c', '--chat', action='store_true', help='Chat with WavCraft')
inspire_parser.add_argument('--session-id', type=str, default='', help='session id, if set to empty, system will allocate an id')

args = parser.parse_args()

if args.mode in ("basic", "inspiration"):
    session_id = pipeline.init_session(args.session_id)
    print(f"Session {session_id} is created.")

    api_key = utils.get_api_key()
    assert api_key != None, "Please set your openai_key in the environment variable."
    
    input_wav = []

    while True:
        this_turn_wav = input("Add audio files(s) (each file starts with '+'): ")
        input_text = input("Enter your instruction (input `EXIT` to exit the process): ")

        if input_text == "EXIT":
            print("WavCraft is completed.")
            break
            
        if args.full:
            this_turn_wav = this_turn_wav.split('+')
            this_turn_wav = [wav.strip().strip("'").strip("\"") for wav in this_turn_wav if len(wav) > 0]
            input_wav.extend(this_turn_wav)

            pipeline.full_steps(session_id, input_wav, input_text, api_key, mode=args.mode)