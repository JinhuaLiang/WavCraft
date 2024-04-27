import datetime
import os
import openai
import re
import glob
import pickle
import time
import random
import string
from retrying import retry
from glob import glob

from wavcraft.mistral_api import ChatMistralMoE, ChatMistral
import wavcraft.utils as utils


# Enable this for debugging
USE_OPENAI_CACHE = False
openai_cache = []
if USE_OPENAI_CACHE:
    os.makedirs('cache', exist_ok=True)
    for cache_file in glob.glob('cache/*.pkl'):
        with open(cache_file, 'rb') as file:
            openai_cache.append(pickle.load(file))
USE_MISTRAL_CACHE = False
mistral_cache = []
if USE_MISTRAL_CACHE:
    os.makedirs('cache', exist_ok=True)
    for cache_file in glob.glob('cache/*.pkl'):
        with open(cache_file, 'rb') as file:
            mistral_cache.append(pickle.load(file))

chat_history = [{
                    "role": "system",
                    "content": "You are a helpful assistant.",
                }]
def chat_with_gpt(api_key, model="gpt-4"):
    #"gpt-4",  # "gpt-3.5-turbo"
    global chat_history

    if USE_OPENAI_CACHE:
        filtered_object = list(filter(lambda x: x['prompt'] == chat_history[-1]["content"], openai_cache))
        if len(filtered_object) > 0:
            response = filtered_object[0]['response']
            return response
    
    try:
        openai.api_key = api_key
        
        chat = openai.ChatCompletion.create(
            model=model,
            messages=chat_history,
            # messages=[
            #     {
            #         "role": "system",
            #         "content": "You are a helpful assistant."
            #     },
            #     {
            #         "role": "user",
            #         "content": prompt
            #     }]
        )
    finally:
        openai.api_key = ''

    if USE_OPENAI_CACHE:
        cache_obj = {
            'prompt': chat_history[-1]["content"],
            'response': chat['choices'][0]['message']['content']
        }
        with open(f'cache/{time.time()}.pkl', 'wb') as _openai_cache:
            pickle.dump(cache_obj, _openai_cache)
            openai_cache.append(cache_obj)

    chat_history.append({
                    "role": "system",
                    "content": chat['choices'][0]['message']['content'],
                    })

    return chat['choices'][0]['message']['content']



# Assuming the existence of USE_OPENAI_CACHE, chat_history, and openai_cache similar to GPT function
def chat_with_mistral():  
    global chat_history
    
    if USE_MISTRAL_CACHE:
        filtered_object = list(filter(lambda x: x['prompt'] == chat_history[-1]["content"], openai_cache))
        if len(filtered_object) > 0:
            return filtered_object[0]['response']

    llm = ChatMistral()  
    
    llm.messages = chat_history
    
    try:
        response = llm.get_response(chat_history[-1]["content"])
    finally:
        pass  
    
    if USE_MISTRAL_CACHE:
        cache_obj = {
            'prompt': chat_history[-1]["content"],
            'response': response
        }
        with open(f'cache/{time.time()}.pkl', 'wb') as _openai_cache:
            pickle.dump(cache_obj, _openai_cache)
            openai_cache.append(cache_obj)

    chat_history.append({
        "role": "assistant",
        "content": response,
    })

    return response




def get_file_content(filename):
    with open(filename, 'r') as file:
        return file.read().strip()


def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)


def extract_substring_with_quotes(input_string, quotes="'''"):
    pattern = f"{quotes}(.*?){quotes}"
    matches = re.findall(pattern, input_string, re.DOTALL)
    return matches


def maybe_remove_python_as_prefix(content):
    keyword = "python"
    
    content = content.strip()
    if content.startswith(keyword):
        # Remove the keyword and strip leading/trailing whitespaces
        return content[len(keyword):].strip()
    return content


def try_extract_content_from_quotes(content):
    if "'''" in content:
        return maybe_remove_python_as_prefix(extract_substring_with_quotes(content)[0])
    elif "```" in content:
        return maybe_remove_python_as_prefix(extract_substring_with_quotes(content, quotes="```")[0])
    else:
        return maybe_remove_python_as_prefix(content)

def maybe_get_content_from_file(content_or_filename):
    if os.path.exists(content_or_filename):
        with open(content_or_filename, 'r') as file:
            return file.read().strip()
    return content_or_filename


# Pipeline Interface Guidelines:
#
# Init calls:
# - Init calls must be called before running the actual steps
#   - init_session() is called every time a gradio webpage is loaded
#
# Single Step:
# - takes input (file or content) and output path as input
# - most of time just returns output content
# 
# Compositional Step:
# - takes session_id as input (you have session_id, you have all the paths)
# - run a series of steps

# This is called for every new gradio webpage

def init_session(session_id=''):
    def uid8():
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    if session_id == '':
        session_id = f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_{uid8()}'
    # create the paths
    os.makedirs(utils.get_session_audio_path(session_id))
    print(f'New session created, session_id={session_id}')
    return session_id


@retry(stop_max_attempt_number=3)
def _input_text_to_code_with_retry(log_path, api_key, model="gpt-4"):
    print("    trying ...")
    try:
        code_response = try_extract_content_from_quotes(chat_with_gpt(api_key, model))

    except Exception as err:
        global chat_history
        chat_log = f'\n{chat_history}\n\nINPUT ERROR: {err}'
        write_to_file(log_path, chat_log)
        raise err

    return code_response


wav_description = {}
# [Basic] Step 1: input to py code
def input_text_to_code(input_text, output_path, api_key, model="gpt-4"):
    # Claim input audio in the instruction
    input_description = "Input audio:\n"
    n_input_wavs = len(glob(os.path.join(output_path, 'audio', 'input_*.wav')))
    if n_input_wavs > 0:
        for i in range(n_input_wavs):
            in_wav = f"{output_path.absolute()}/audio/input_{i}.wav"

            if in_wav in wav_description:
                continue
            else:
                input_description += f"INPUT_WAV{i}\n"
                # Add placeholder to the log as the basic does not need to describe the audio
                wav_description[in_wav] = ""
    else:  # no input wav
        input_description = ""

    input_text = maybe_get_content_from_file(input_text)

    log_path = output_path / 'chat.log'
    if not os.path.exists(log_path):
        text_to_audio_script_prompt = get_file_content('wavcraft/prompts/text_to_code.prompt')
        prompt = f'{text_to_audio_script_prompt}\n\n{input_description}Instruction:\n{input_text}\nCode:\n'
    else:
        text_to_followup = get_file_content('wavcraft/prompts/text_to_followup.prompt')
        prompt = f'{input_description}{text_to_followup}\n{input_text}'

    global chat_history
    chat_history.append({
        "role": "user",
        "content": prompt,
        })
       
    write_to_file(log_path, f"{chat_history}")
    code_response = _input_text_to_code_with_retry(log_path, api_key, model)
    executable_code_filename = output_path / 'audio_executable.py'

    write_to_file(executable_code_filename, code_response)
    return code_response


# [Inspiration] Step 1: input to py code
def input_text_to_code_plus(input_text, output_path, api_key, model="gpt-4"):
    import sys

    sys.path.append(os.path.dirname(__file__))
    from wavcraft.apis import AU

    # Add description to the prompt
    input_description = "Input audio:\n"
    n_input_wavs = len(glob(os.path.join(output_path, 'audio', 'input_*.wav')))
    for i in range(n_input_wavs):
        in_wav = f"{output_path.absolute()}/audio/input_{i}.wav"

        if in_wav in wav_description:
            continue
        else:
            response = AU(in_wav, text="write an audio caption describing the sound")
            input_description += f"INPUT_WAV{i}: {response}\n"
            # Add to the log
            wav_description[in_wav] = response

    input_text = maybe_get_content_from_file(input_text)

    log_path = output_path / 'chat.log'
    if not os.path.exists(log_path):
        text_to_audio_script_prompt = get_file_content('wavcraft/prompts/text_to_code.prompt')
        prompt = f'{text_to_audio_script_prompt}\n\n{input_description}Instruction:\n{input_text}\nCode:\n'
    else:
        text_to_followup = get_file_content('wavcraft/prompts/text_to_followup.prompt')
        prompt = f'{input_description}{text_to_followup}\n{input_text}'

    global chat_history
    chat_history.append({
        "role": "user",
        "content": prompt,
        })
       
    write_to_file(log_path, f"{chat_history}")

    code_response = _input_text_to_code_with_retry(log_path, api_key, model)
    executable_code_filename = output_path / 'audio_executable.py'

    write_to_file(executable_code_filename, code_response)
    return code_response


# Step 2: py code to final wav
def audio_exe_to_result(code_response, output_path):
    executable_code_filename = output_path / 'audio_executable.py'

    # TODO: make this more easy to modify
    # Executable file header
    header = "from wavcraft.apis import LEN, OUTPUT, SPLIT, MIX, CAT, ADJUST_VOL, ADD_NOISE, LOW_PASS, HIGH_PASS, ADD_RIR, ROOM_SIMULATE, TTM, TTA, TTS, SR, TSS, INPAINT"
    
    input_claimer = ""
    n_input_wavs = len(glob(os.path.join(output_path, 'audio', 'input_*.wav')))
    for i in range(n_input_wavs):
        in_wav = f"\"{output_path.absolute()}/audio/input_{i}.wav\""
        input_claimer += f"INPUT_WAV{i} = {in_wav}\n"
    
    tail = "OUTPUT(OUTPUT_WAV)"
    code_response = maybe_get_content_from_file(code_response)
    command = f"{header}\n\n\n{input_claimer}{code_response}\n{tail}"
    write_to_file(executable_code_filename, command)
    
    os.system(f'PYTHONPATH=. python {executable_code_filename}')


# Function call used by Gradio: input_text to json
def generate_code(session_id, input_wav, input_text, api_key, model="gpt-4", mode="basic"):
    assert mode in ("basic", "inspiration")

    output_path = utils.get_session_path(session_id)
    os.makedirs(output_path, exist_ok=True)
    for i, in_wav in enumerate(input_wav):
        os.system(f"cp {in_wav} {os.path.join(output_path, 'audio', f'input_{i}.wav')}")

    # Step 1
    print(f'session_id={session_id}, Step 1: Writing executable code with LLM ...')
    if mode == "basic":
        return input_text_to_code(input_text, output_path, api_key, model)
    else:
        return input_text_to_code_plus(input_text, output_path, api_key, model)


# Function call used by Gradio: json to result wav
def generate_audio(session_id, code_response):
    output_path = utils.get_session_path(session_id)
    # Step 2
    print(f'session_id={session_id}, Step 2: Start running Python program...')
    audio_exe_to_result(code_response, output_path)


# Convenient function call used by wavjourney_cli
def full_steps(session_id, input_wav, input_text, api_key, mode, model="gpt-4"):
    code_script = generate_code(session_id, input_wav, input_text, api_key, model=model, mode=mode)
    return generate_audio(session_id, code_script)