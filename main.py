import os
from util import (
    ASR_TARGET_LANGUAGE_NAMES,
    LANGUAGE_NAME_TO_CODE,
)
from util import  crop_audio , preprocess_audio
import torch


# from seamless_communication.inference import Translator

from faster_whisper import WhisperModel


from transformers import pipeline




class AudioOfficial:
    def __init__(self, **kwargs):
        
        if torch.cuda.is_available():
            self.cuda = torch.device("cuda:0")
            self.compute_type=torch.float16
            self.devicewhisper = "cuda"
        else:
            self.cuda = torch.device("cpu")
            self.compute_type=torch.float32
            self.devicewhisper = "cpu"
        self.download_root = kwargs.get("down_nmodel_path", "./models/")
        computer_type = "float16" if self.compute_type == torch.float16 else "int8" #float32"
        # "vinai/PhoWhisper-large"
        # self.model = WhisperModel("./models/PhoWhisper-large-ct2" ,download_root=self.download_root, compute_type=computer_type, device=self.devicewhisper , num_workers= 4)

        self.model = WhisperModel("large-v2" ,download_root=self.download_root, compute_type=computer_type, device=self.devicewhisper , num_workers= 4)
        # self.translator = Translator("SeamlessM4T-Medium", vocoder_name_or_card="vocoder_v2",device=self.cuda , dtype=self.compute_type ,)
        #seamlessM4T_medium vocoder_36langs
        #seamlessM4T_v2_large vocoder_v2

        self.transcriber = pipeline("automatic-speech-recognition", model="models/PhoWhisper-large" , )
        # self.translator = Translator("seamlessM4T_medium", vocoder_name_or_card="vocoder_36langs",device=self.cuda , dtype=self.compute_type ,)

    def ExtractText(self, path_audio, **kwargs):
        DEFAULT_TARGET_LANGUAGE = "Vietnamese"
        segments_list = []
        Resuft = self.transcriber(path_audio , generate_kwargs={"language": "Vietnamese"})['text']
        input_audio, a = preprocess_audio(path_audio)
        print(a)
        DEFAULT_TARGET_LANGUAGE = kwargs.get("language", DEFAULT_TARGET_LANGUAGE)
        target_language_code = LANGUAGE_NAME_TO_CODE[DEFAULT_TARGET_LANGUAGE]
        text,info = self.model.transcribe(input_audio , beam_size = 5, vad_filter = True , vad_parameters=dict(min_silence_duration_ms=500),
                                           word_timestamps = False,
                                                 language="vi" , task= "transcribe" ) 
        for segment in text:
            end_sample = segment.end * 1000
            start_sample = segment.start * 1000
            if end_sample - start_sample < 500:
                continue
            # Initialize segment_dict here for each segment
            segment_dict = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
            if segment.text == " Cảm ơn các bạn đã theo dõi, hẹn gặp lại các bạn trong những video tiếp theo." or segment.text == ' Cảm ơn các bạn đã theo dõi và hẹn gặp lại các bạn trong những video tiếp theo.' or segment.text == ' Cảm ơn các bạn đã theo dõi và hẹn gặp lại các bạn trong những video tiếp theo.':
                segment_dict["text"] = " "
                # Assuming run_asr_facebook is a method of the current class, hence the use of self.
                # audio_crop = crop_audio(input_audio, start_sample, end_sample)
                # transcribed_text = self.run_asr_facebook(audio_crop, target_language_code)
                # segment_dict["text"] = transcribed_text  # Update the text in the dict
            segments_list.append(segment_dict)  # Append the updated dict to the list
        return segments_list , Resuft


        
        


    def run_asr_facebook(self,input_audio: str, target_language: str) -> str:
        out_texts, _ = self.translator.predict(
            input=input_audio,
            task_str="ASR",
            src_lang=target_language,
            tgt_lang=target_language,
        )
        return str(out_texts[0])
