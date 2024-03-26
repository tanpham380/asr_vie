from seamless_communication.inference import Translator
import numpy as np

import torch
from pydub import AudioSegment 


translator = Translator("seamlessM4T_v2_large", vocoder_name_or_card="vocoder_v2",device=torch.device("cpu") , dtype=torch.float32 )


def run_asr_facebook(input_audio: str, target_language: str) -> str:
    out_texts, _ = translator.predict(
        input=input_audio,
        task_str="ASR",
        src_lang=target_language,
        tgt_lang=target_language,
    )
    return str(out_texts[0])
def crop_audio(input_file, output_file, start_time_ms, end_time_ms):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Crop the audio
    cropped_audio = audio[start_time_ms:end_time_ms]

    # Export the cropped audio to a new file
    cropped_audio.export(output_file, format="mp3")  # Adjust the format as needed

if __name__ == '__main__':
    input = "converted_file.wav"
    output = "output.wav"
    start_time_ms = 1000
    end_time_ms = 10000
    crop_audio(input, output, start_time_ms, end_time_ms)
    
    a = run_asr_facebook(input_audio = output, target_language = "vie")
    print(a)
