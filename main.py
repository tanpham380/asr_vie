from util import (
    ASR_TARGET_LANGUAGE_NAMES,
    LANGUAGE_NAME_TO_CODE,
    decode_audio,
    format_text_segments,
)
import torch


# from seamless_communication.inference import Translator


from faster_whisper import WhisperModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


from vad import VadOptions, collect_chunks, get_speech_timestamps



class AudioOfficial:
    def loadmodel(self,model_id, pipeline_name):
        model = AutoModelForSpeechSeq2Seq.from_pretrained( model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True , cache_dir=self.download_root)
        model.to(self.cuda)
        processor = AutoProcessor.from_pretrained(model_id , cache_dir=self.download_root)
        pip = pipeline(
        pipeline_name,
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=self.compute_type,
        device=self.cuda,
        )
        return pip
        
        
        
        
    def __init__(self, **kwargs):
        
        if torch.cuda.is_available():
            self.cuda = torch.device("cuda:0")
            self.compute_type=torch.float32
            self.devicewhisper = "cuda"
        else:
            self.cuda = torch.device("cpu")
            self.compute_type=torch.float32
            self.devicewhisper = "cpu"
        self.download_root = kwargs.get("down_nmodel_path", "./models/")
        computer_type = "float32" if self.compute_type == torch.float32 else "float32" #float32"
#         # "vinai/PhoWhisper-large"
        
        self.transcriber1 = AudioOfficial.loadmodel(self,"openai/whisper-large-v3"  , "automatic-speech-recognition")

        
        # self.model_id = "vinai/PhoWhisper-large"
        self.transcriber2 = AudioOfficial.loadmodel(self,"./models/PhoWhisper-large"  , "automatic-speech-recognition")
        
        


        
        

        # self.model = WhisperModel("./models/PhoWhisper-large-ct2" ,download_root=self.download_root, compute_type=computer_type, device=self.devicewhisper , num_workers= 4)
        print("Loading model with device: ", self.devicewhisper , " and compute type: ", computer_type)
        # self.progesspipeline = FlaxWhisperPipline("openai/whisper-large-v3")

     #   self.transcriber2 = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-large" , device= self.cuda , return_timestamps=True #,chunk_length_s=30,batch_size=16,return_timestamps=False, max_new_tokens=128,)
    #    )
        # self.model = WhisperModel("large-v2" ,download_root=self.download_root, compute_type=computer_type, device=self.devicewhisper ,  num_workers= 4 , cpu_threads = 100)
        
        # self.translator = Translator("seamlessM4T_v2_large", vocoder_name_or_card="vocoder_v2",device=self.cuda , dtype=self.compute_type ,)
        #seamlessM4T_medium vocoder_36langs
        #seamlessM4T_v2_large vocoder_v2

        
        # self.translator = Translator("seamlessM4T_medium", vocoder_name_or_card="vocoder_36langs",device=self.cuda , dtype=self.compute_type ,)

    def ExtractText(self, path_audio, **kwargs):
        DEFAULT_TARGET_LANGUAGE = "Vietnamese"
        input_audio = decode_audio(path_audio)
        
        filtered_audio = self.filterVAD(input_audio)
        segments_list1 = self.transcriber1(filtered_audio , generate_kwargs={"language": "Vietnamese"})['chunks']
        segments_list2 = self.transcriber2(filtered_audio , generate_kwargs={"language": "Vietnamese"})['chunks']
        # Resuft = self.progesspipeline(input_audio)

        # DEFAULT_TARGET_LANGUAGE = kwargs.get("language", DEFAULT_TARGET_LANGUAGE)
        # target_language_code = LANGUAGE_NAME_TO_CODE[DEFAULT_TARGET_LANGUAGE]
        # text,info = self.model.transcribe(input_audio , beam_size = 1, vad_filter = True , vad_parameters=dict(min_silence_duration_ms=500),
        #                                    word_timestamps = False, 
        #                                     #   temperature= 0.0,
        #                                     #      language="vi" ,
        #                                          task= "transcribe" ) 
        # for segment in text:
        #     end_sample = segment.end * 1000
        #     start_sample = segment.start * 1000
        #     if end_sample - start_sample < 500:
        #         continue
        #     # Initialize segment_dict here for each segment
        #     segment_dict = {
        #         "start": segment.start,
        #         "end": segment.end,
        #         "text": segment.text,
        #     }
        #     if segment.text == " Cảm ơn các bạn đã theo dõi, hẹn gặp lại các bạn trong những video tiếp theo." or segment.text == ' Cảm ơn các bạn đã theo dõi và hẹn gặp lại các bạn trong những video tiếp theo.' or segment.text == ' Cảm ơn các bạn đã theo dõi và hẹn gặp lại các bạn trong những video tiếp theo.':
        #         segment_dict["text"] = " "
        #         # Assuming run_asr_facebook is a method of the current class, hence the use of self.
        #         # audio_crop = crop_audio(input_audio, start_sample, end_sample)
        #         # transcribed_text = self.run_asr_facebook(audio_crop, target_language_code)
        #         # segment_dict["text"] = transcribed_text  # Update the text in the dict
        #     segments_list.append(segment_dict)  # Append the updated dict to the list
        Result1 = format_text_segments(segments_list1)
        Result2 = format_text_segments(segments_list2)
        return   Result1, Result2


        
        


    def run_asr_facebook(self,input_audio: str, target_language: str) -> str:
        out_texts, _ = self.translator.predict(
            input=input_audio,
            task_str="ASR",
            src_lang=target_language,
            tgt_lang=target_language,
        )
        return str(out_texts[0])
    def filterVAD(self,audio, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500) , sampling_rate = 16000):
        # duration = audio.shape[0] / sampling_rate
        if vad_filter:
            if vad_parameters is None:
                vad_parameters = VadOptions()
            elif isinstance(vad_parameters, dict):
                vad_parameters = VadOptions(**vad_parameters)
            speech_chunks = get_speech_timestamps(audio, vad_parameters)
            audio = collect_chunks(audio, speech_chunks)
            return audio
        else:
            speech_chunks = None
            return None

