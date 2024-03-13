import os
import torchaudio
import requests
import math
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy.io.wavfile import write
from retrying import retry
from gradio_client import Client
from audiomentations import AddGaussianSNR, LowPassFilter, HighPassFilter, ApplyImpulseResponse, RoomSimulator

from wavcraft.utils import get_service_port, get_service_url, get_path_from_target_dir, generate_random_series
 

os.environ['OPENBLAS_NUM_THREADS'] = '1'
SAMPLE_RATE = 16000  # 32000 is NOT supported by wavmark

localhost_addr = get_service_url()


def _LOUDNESS_NORM(wav, volume=-25, out_wav=generate_random_series()+'.wav', sr=SAMPLE_RATE):
    """
    Nomalize waveform and adjust the loadness as per BS.1770.
    """
    # peak normalize wav to -1 dB
    peak_normalized_wav = pyln.normalize.peak(wav, -10.0)
    # measure the loudness first 
    meter = pyln.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(peak_normalized_wav)
    # loudness normalize wav to -12 dB LUFS
    normalized_wav = pyln.normalize.loudness(peak_normalized_wav, loudness, volume)

    return normalized_wav


def _READ_AUDIO_NUMPY(wav, sr=SAMPLE_RATE):
    """
    Read audio numpy 
    Returns: 
        np.array [samples]
    """
    waveform, sample_rate = torchaudio.load(wav)

    if sample_rate != sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=sr)
    
    wav_numpy = waveform[0].numpy()

    return wav_numpy


def _WRITE_AUDIO(wav, name=None, sr=SAMPLE_RATE):
    """
    Write audio numpy to .wav file
    Params:
        wav: np.array [samples]
    """   
    if name is None:
        name = 'output.wav' 
    
    if len(wav.shape) > 1:
        wav = wav[0]

    # declipping
    max_value = np.max(np.abs(wav)) if wav.size > 0 else 0
    if max_value > 1:
        wav *= 0.9 / (max_value + 1e-5)
    
    # write audio
    write(name, sr, np.round(wav*32767).astype(np.int16))


def LEN(wav, sr=SAMPLE_RATE):
    """
    Returns the duration of audio in seconds.
    """
    wav= _READ_AUDIO_NUMPY(wav)

    return len(wav) / sr


# def OUTPUT(wav, out_wav="output.wav"):
#     output_wav = get_path_from_target_dir(out_wav, wav)
#     os.rename(wav, output_wav)
#     print(f'Done all processes, result: {output_wav}')
#     return output_wav


def OUTPUT(wav, out_wav="output.wav", sr=SAMPLE_RATE):
    # Add watermark to the generated audio
    _tmp_wav = _ENCODE_WATERMARK(wav, sample_rate=sr)
    
    output_wav = get_path_from_target_dir(out_wav, _tmp_wav)
    os.rename(_tmp_wav, output_wav)
    print(f'Done all processes, result: {output_wav}')
    return output_wav


"""     DSP modules     """
def SPLIT(wav_path, break_points=[], out_wav=generate_random_series()+'.wav', sr=SAMPLE_RATE):
    """
    Split audio into several pieces according to the breakpoints.
    Params:
        break_points: list[float]: a list of breakpoints (in seconds)
    Returns:
        Path to output wav file.
    """
    # Avoid `breakpoint` containing 0
    break_points = [p for p in break_points if p != 0]
    num_pieces = len(break_points) + 1

    prefix = out_wav.split(".")[0]

    wav = _READ_AUDIO_NUMPY(wav_path)

    results = []
    for i in range(num_pieces):
        onset = break_points[i - 1] * sr if i > 0 else 0
        offset = break_points[i] * sr if i < len(break_points) else len(wav)

        _o_wav = get_path_from_target_dir(prefix+f"_{i}.wav", wav_path)
        _WRITE_AUDIO(wav[int(onset):int(offset)], name=_o_wav)
        results.append(_o_wav)

    return results


def MIX(wavs=[['1.wav', 0.], ['2.wav', 0.]], out_wav=generate_random_series()+'.wav', sr=SAMPLE_RATE):
    """
    Mix multiple audio clips by considering their onset time.
    Returns:
        Path to output wav file.
    """
    max_length = max([int(wav[1]*sr + len(_READ_AUDIO_NUMPY(wav[0]))) for wav in wavs])
    template_wav = np.zeros(max_length)

    for wav in wavs:
        cur_name, cur_onset = wav
        cur_wav = _READ_AUDIO_NUMPY(cur_name)
        cur_len = len(cur_wav)
        cur_onset = int(cur_onset * sr)
        
        # mix
        template_wav[cur_onset:cur_onset+cur_len] += cur_wav
    
    out_wav = get_path_from_target_dir(out_wav, wavs[0][0])
    _WRITE_AUDIO(template_wav, name=out_wav)
    return out_wav


def CAT(wavs, out_wav=generate_random_series()+'.wav'):
    """
    Concat multiple audio clips together.
    Params:
        wavs: List of wav file ['1.wav', '2.wav', ...]
    """
    wav_num = len(wavs)

    segment0 = _READ_AUDIO_NUMPY(wavs[0])

    cat_wav = segment0

    if wav_num > 1:
        for i in range(1, wav_num):
            next_wav = _READ_AUDIO_NUMPY(wavs[i])
            cat_wav = np.concatenate((cat_wav, next_wav), axis=-1)

    out_wav = get_path_from_target_dir(out_wav, wavs[0])
    _WRITE_AUDIO(cat_wav, name=out_wav)
    return out_wav


def ADJUST_VOL(wav_path, volume, out_wav=generate_random_series()+'.wav', sr=SAMPLE_RATE):
    """
    Adjust the volume of waveform by `volume`.
    """
    wav, sample_rate = torchaudio.load(wav_path)

    if sample_rate != sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sample_rate, new_freq=sr)
    
    adj_vol_fn = torchaudio.transforms.Vol(gain=volume, gain_type="db")
    wav = adj_vol_fn(wav)

    # write audio
    wav = wav[0].numpy()  # convert to numpy
    out_wav = get_path_from_target_dir(out_wav, wav_path)
    write(out_wav, sr, np.round(wav*32767).astype(np.int16))
    return out_wav


# def INC_VOL(wav_path, volume, out_wav=generate_random_series()+'.wav', sr=SAMPLE_RATE):
#     """
#     Increase the volume of waveform by `volume`.
#     """
#     wav = _READ_AUDIO_NUMPY(wav_path)
#     # measure the loudness first 
#     meter = pyln.Meter(sr) # create BS.1770 meter
#     loudness = meter.integrated_loudness(wav)
#     # loudness normalize audio to the desired dB LUFS
#     volume += loudness
#     wav = pyln.normalize.loudness(wav, loudness, volume)

#     # write audio
#     out_wav = get_path_from_target_dir(out_wav, wav_path)
#     write(out_wav, sr, np.round(wav*32767).astype(np.int16))
#     return out_wav


# def DEC_VOL(wav_path, volume, out_wav=generate_random_series()+'.wav', sr=SAMPLE_RATE):
#     """
#     Decrease the volume of waveform by `volume`.
#     """
#     wav = _READ_AUDIO_NUMPY(wav_path)
#     # measure the loudness first 
#     meter = pyln.Meter(sr) # create BS.1770 meter
#     loudness = meter.integrated_loudness(wav)
#     # loudness normalize audio to the desired dB LUFS
#     volume -= loudness
#     wav = pyln.normalize.loudness(wav, loudness, volume)

#     # write audio
#     out_wav = get_path_from_target_dir(out_wav, wav_path)
#     write(out_wav, sr, np.round(wav*32767).astype(np.int16))
#     return out_wav


def ADD_NOISE(wav_path, min_snr_db=5.0, max_snr_db=40.0, out_wav=generate_random_series()+'.wav', sr=SAMPLE_RATE):
    wav = _READ_AUDIO_NUMPY(wav_path)
    transform = AddGaussianSNR(
        min_snr_db=min_snr_db,
        max_snr_db=max_snr_db,
        p=1.0
    )

    augmented_sound = transform(wav, sample_rate=sr)

    # write audio
    out_wav = get_path_from_target_dir(out_wav, wav_path)
    _WRITE_AUDIO(augmented_sound, name=out_wav)
    return out_wav


def LOW_PASS(wav_path, min_cutoff_freq=150.0, max_cutoff_freq=7500.0, min_rolloff=12, max_rolloff=24, zero_phase=False, out_wav=generate_random_series()+'.wav', sr=SAMPLE_RATE):
    wav = _READ_AUDIO_NUMPY(wav_path)
    transform = LowPassFilter(
        min_cutoff_freq=min_cutoff_freq,
        max_cutoff_freq=max_cutoff_freq,
        min_rolloff=min_rolloff,
        max_rolloff=max_rolloff,
        zero_phase=zero_phase,
        p=1.0
    )

    augmented_sound = transform(wav, sample_rate=sr)

    # write audio
    out_wav = get_path_from_target_dir(out_wav, wav_path)
    _WRITE_AUDIO(augmented_sound, name=out_wav)
    return out_wav


def HIGH_PASS(wav_path, min_cutoff_freq=20, max_cutoff_freq=2400, min_rolloff=12, max_rolloff=24, zero_phase=False, out_wav=generate_random_series()+'.wav', sr=SAMPLE_RATE):
    wav = _READ_AUDIO_NUMPY(wav_path)
    transform = HighPassFilter(
        min_cutoff_freq=min_cutoff_freq,
        max_cutoff_freq=max_cutoff_freq,
        min_rolloff=min_rolloff,
        max_rolloff=max_rolloff,
        zero_phase=zero_phase,
        p=1.0
    )

    augmented_sound = transform(wav, sample_rate=sr)

    # write audio
    out_wav = get_path_from_target_dir(out_wav, wav_path)
    _WRITE_AUDIO(augmented_sound, name=out_wav)
    return out_wav


def ADD_RIR(wav_path, ir_path=None, out_wav=generate_random_series()+'.wav', sr=SAMPLE_RATE):
    wav = _READ_AUDIO_NUMPY(wav_path)

    transform = ApplyImpulseResponse(ir_path=ir_path, p=1.0)    
    augmented_sound = transform(wav, sample_rate=sr)

    # write audio
    out_wav = get_path_from_target_dir(out_wav, wav_path)
    _WRITE_AUDIO(augmented_sound, name=out_wav)
    return out_wav


def ROOM_SIMULATE(wav_path, min_size_x=3.6, max_size_x=5.6, 
            min_size_y=3.6, max_size_y=3.9, 
            min_size_z=2.4, max_size_z=3.0, 
            min_absorption_value=0.075, max_absorption_value=0.4,
            min_source_x=0.1, max_source_x=3.5,
            min_source_y=0.1, max_source_y=2.7,
            min_source_z=1.0, max_source_z=2.1,
            min_mic_distance=0.15, max_mic_distance=0.35,
            min_mic_azimuth=-math.pi, max_mic_azimuth=math.pi,
            min_mic_elevation=-math.pi, max_mic_elevation=math.pi,
            out_wav=generate_random_series()+'.wav', sr=SAMPLE_RATE):
    wav = _READ_AUDIO_NUMPY(wav_path)

    transform = RoomSimulator(
        min_size_x=min_size_x, max_size_x=max_size_x,
        min_size_y=min_size_y, max_size_y=max_size_y,
        min_size_z=min_size_z, max_size_z=max_size_z,
        min_absorption_value=min_absorption_value, max_absorption_value=max_absorption_value,
        min_source_x=min_source_x, max_source_x=max_source_x,
        min_source_y=min_source_y, max_source_y=max_source_y,
        min_source_z=min_source_z, max_source_z=max_source_z,
        min_mic_distance=min_mic_distance, max_mic_distance=max_mic_distance,
        min_mic_azimuth=min_mic_azimuth, max_mic_azimuth=max_mic_azimuth,
        min_mic_elevation=min_mic_elevation, max_mic_elevation=max_mic_elevation,
        p=1.0)
    augmented_sound = transform(wav, sample_rate=sr)

    # write audio
    out_wav = get_path_from_target_dir(out_wav, wav_path)
    _WRITE_AUDIO(augmented_sound, name=out_wav)
    return out_wav
    

# def CLIP(wav_path, offset, onset=0, out_wav=generate_random_series()+'.wav', sr=SAMPLE_RATE):
#     """
#     Clip the audio using onset and offset time.
#     Params:
#         onset/offset: onset/offset time in seconds.
#     Returns:
#         Path to output wav file.
#     """
#     wav = _READ_AUDIO_NUMPY(wav_path)

#     # Get onset/offset with samples rates
#     onset *= SAMPLE_RATE
#     offset *= SAMPLE_RATE
#     assert 0 <= onset <= offset <= len(wav)

#     out_wav = get_path_from_target_dir(out_wav, wav_path)
#     _WRITE_AUDIO(wav[int(onset):int(offset)], name=out_wav)
#     return out_wav


"""     Deep-learning modules     """
@retry(stop_max_attempt_number=5, wait_fixed=2000)
def AU(wav_path, text="write an audio caption describing the sound"):
    HF_key = os.environ.get("HF_KEY")
    client = Client("https://yuangongfdu-ltu.hf.space/", hf_token=HF_key)
    response = client.predict(
        wav_path,
        "write an audio caption describing the sound",
        api_name="/predict",
    )
    return response
    

@retry(stop_max_attempt_number=5, wait_fixed=2000)
def TTM(text, melody=None, length=10, volume=-28, out_wav=generate_random_series()+'.wav', sr=SAMPLE_RATE):
    service_port = get_service_port("AUDIOCRAFT_SERVICE_PORT")
    url = f'http://{localhost_addr}:{service_port}/generate_music'

    # Change the name if file exist
    if os.path.exists(out_wav):
        out_wav = generate_random_series() + '.wav'
        
    data = {
        'text': f'{text}',
        'melody': melody,
        'length': f'{length}',
        'volume': f'{volume}',
        'sample_rate': f'{sr}',
        'output_wav': f'{out_wav}',
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        print('Success:', response.json()['message'])
        return out_wav
    else:
        print('Error:', response.json()['API error'])
        raise RuntimeError(response.json()['API error'])


@retry(stop_max_attempt_number=5, wait_fixed=2000)
def TTA(text, length=5, volume=-35, out_wav=generate_random_series()+'.wav'):
    service_port = get_service_port("AUDIOCRAFT_SERVICE_PORT")
    url = f'http://{localhost_addr}:{service_port}/generate_audio'

    # Change the name if file exist
    if os.path.exists(out_wav):
        out_wav = generate_random_series() + '.wav'

    data = {
        'text': f'{text}',
        'length': f'{length}',
        'volume': f'{volume}',
        'output_wav': f'{out_wav}',
    }

    response = requests.post(url, json=data)
    if response.status_code == 200:
        print('Success:', response.json()['message'])
        return out_wav
    else:
        print('Error:', response.json()['API error'])
        raise RuntimeError(response.json()['API error'])


@retry(stop_max_attempt_number=5, wait_fixed=2000)
def TTS(text, speaker="Male1_En", volume=-20, out_wav=generate_random_series()+'.wav'):
    service_port = get_service_port("AUDIOCRAFT_SERVICE_PORT")
    url = f'http://{localhost_addr}:{service_port}/generate_speech'

    # Change the name if file exist
    if os.path.exists(out_wav):
        out_wav = generate_random_series() + '.wav'

    data = {
    'text': f'{text}',
    'speaker_id': f'{speaker}',
    'volume': f'{volume}',
    'output_wav': f'{out_wav}',
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        print('Success:', response.json()['message'])
        return out_wav
    else:
        print('Error:', response.json()['API error'])
        raise RuntimeError(response.json()['API error'])


@retry(stop_max_attempt_number=5, wait_fixed=2000)
def SR(wav_path, out_wav=generate_random_series()+'.wav', ddim_steps=50, guidance_scale=3.5, seed=42):
    service_port = get_service_port("AUDIOSR_SERVICE_PORT")
    url = f'http://{localhost_addr}:{service_port}/super_resolution'
    out_wav = get_path_from_target_dir(out_wav, wav_path)
    data = {
        'wav_path': f'{wav_path}',
        'ddim_steps': f'{ddim_steps}',
        'guidance_scale': f'{guidance_scale}',
        'seed': f'{seed}',
        'output_wav':f'{out_wav}'
    }
    response = requests.post(url, json=data)

    if response.status_code == 200:
        print('Success:', response.json()['message'])
        return out_wav
    else:
        print('Error:', response.json()['API error'])
        raise RuntimeError(response.json()['API error'])


# @retry(stop_max_attempt_number=5, wait_fixed=2000)
# def VP(wav_path, out_dir):
#     url = f'http://{localhost_addr}:{service_port}/parse_voice'
#     data = {
#         'wav_path': f'{wav_path}', 
#         'out_dir':f'{out_dir}'
#     }

#     response = requests.post(url, json=data)

#     if response.status_code == 200:
#         print('Success:', response.json()['message'])
#     else:
#         print('Error:', response.json()['API error'])
#         raise RuntimeError(response.json()['API error'])
    

# @retry(stop_max_attempt_number=5, wait_fixed=2000)
# def EXTRACT(wav_path, text, out_wav=generate_random_series()+'.wav'):
#     service_port = get_service_port("AUDIOSEP_SERVICE_PORT")
#     url = f'http://{localhost_addr}:{service_port}/source_separate'
#     out_wav = get_path_from_target_dir(out_wav, wav_path)
#     data = {
#         'wav_path': f'{wav_path}', 
#         'text': f'{text}',
#         'output_wav':f'{out_wav}'
#     }

#     response = requests.post(url, json=data)

#     if response.status_code == 200:
#         filedir, filename = os.path.split(out_wav)
#         fg_filepath = os.path.join(filedir, "fg_"+filename)
#         bg_filepath = os.path.join(filedir, "bg_"+filename)
#         os.rename(fg_filepath, out_wav)
#         os.remove(bg_filepath)
#         print('Success:', response.json()['message'])
#         return out_wav
#     else:
#         print('Error:', response.json()['API error'])
#         raise RuntimeError(response.json()['API error'])
    

# @retry(stop_max_attempt_number=5, wait_fixed=2000)
# def DROP(wav_path, text, out_wav=generate_random_series()+'.wav'):
#     service_port = get_service_port("AUDIOSEP_SERVICE_PORT")
#     url = f'http://{localhost_addr}:{service_port}/source_separate'
#     out_wav = get_path_from_target_dir(out_wav, wav_path)
#     data = {
#         'wav_path': f'{wav_path}', 
#         'text': f'{text}',
#         'output_wav':f'{out_wav}'
#     }

#     response = requests.post(url, json=data)

#     if response.status_code == 200:
#         filedir, filename = os.path.split(out_wav)
#         fg_filepath = os.path.join(filedir, "fg_"+filename)
#         bg_filepath = os.path.join(filedir, "bg_"+filename)
#         os.rename(bg_filepath, out_wav)
#         os.remove(fg_filepath)
#         print('Success:', response.json()['message'])
#         return out_wav
#     else:
#         print('Error:', response.json()['API error'])
#         raise RuntimeError(response.json()['API error'])
    

@retry(stop_max_attempt_number=5, wait_fixed=2000)
def TSS(wav_path, text, out_wav=generate_random_series()+'.wav'):
    service_port = get_service_port("AUDIOSEP_SERVICE_PORT")
    url = f'http://{localhost_addr}:{service_port}/source_separate'
    out_wav = get_path_from_target_dir(out_wav, wav_path)
    data = {
        'wav_path': f'{wav_path}', 
        'text': f'{text}',
        'output_wav':f'{out_wav}'
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        filedir, filename = os.path.split(out_wav)
        fg_filepath = os.path.join(filedir, "fg_"+filename)
        bg_filepath = os.path.join(filedir, "bg_"+filename)
        print('Success:', response.json()['message'])
        return fg_filepath, bg_filepath
    else:
        print('Error:', response.json()['API error'])
        raise RuntimeError(response.json()['API error'])
    

@retry(stop_max_attempt_number=5, wait_fixed=2000)
def INPAINT(wav_path, text, onset, offset, duration, guidance_scale=2.5, ddim_steps=200, random_seed=42, sample_rate=SAMPLE_RATE, out_wav=generate_random_series()+'.wav',):
    service_port = get_service_port("AUDIOLDM_SERVICE_PORT")
    url = f'http://{localhost_addr}:{service_port}/audio_inpaint'
    out_wav = get_path_from_target_dir(out_wav, wav_path)
    data = {
        'wav_path': f'{wav_path}', 
        'text': f'{text}',
        'onset': onset,
        'offset': offset,
        'duration': duration,
        'output_wav':f'{out_wav}',
        # generation settings
        'sample_rate': sample_rate,
        'guidance_scale': guidance_scale,
        'ddim_steps': ddim_steps,
        'random_seed': random_seed,
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        print('Success:', response.json()['message'])
        return out_wav
    else:
        print('Error:', response.json()['API error'])
        raise RuntimeError(response.json()['API error'])


def _ENCODE_WATERMARK(wav_path, sample_rate=SAMPLE_RATE, out_wav=generate_random_series()+'.wav',):
    service_port = get_service_port("WAVMARK_SERVICE_PORT")
    url = f'http://{localhost_addr}:{service_port}/audio_watermark'
    out_wav = get_path_from_target_dir(out_wav, wav_path)
    data = {
        'wav_path': f'{wav_path}', 
        'action': "encode",
        'output_wav':f'{out_wav}',
        'sample_rate': sample_rate,
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        print('Success:', response.json()['message'])
        return out_wav
    else:
        print('Error:', response.json()['API error'])
        raise RuntimeError(response.json()['API error'])
    

def _DECODE_WATERMARK(wav_path, sample_rate=SAMPLE_RATE):
    service_port = get_service_port("WAVMARK_SERVICE_PORT")
    url = f'http://{localhost_addr}:{service_port}/audio_watermark'
    data = {
        'wav_path': f'{wav_path}', 
        'action': "decode",
        'sample_rate': sample_rate,
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        print('Success:', response.json()['message'])
        return wav_path
    else:
        print('Error:', response.json()['API error'])
        raise RuntimeError(response.json()['API error'])
    