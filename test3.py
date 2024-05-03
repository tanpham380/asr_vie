
from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union
from diarize import DiarizationPipeline, assign_word_speakers
from util import (

    convert_segments,
    decode_audio,
    format_text_segments,
)
import torch
import gc

# from seamless_communication.inference import Translator


# from faster_whisper import WhisperModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, WhisperForConditionalGeneration, WhisperProcessor

from vad import VadOptions, collect_chunks, get_speech_timestamps, restore_speech_timestamps


class AudioOfficial:

    def loadmodel(self, model_id, pipeline_name):

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            # , use_flash_attention_2=True
            model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True, cache_dir=self.download_root

        )
        processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=self.download_root, )
        # model = model.to_bettertransformer()

        model.to(self.cuda)

        # processor = AutoProcessor.from_pretrained(model_id , cache_dir=self.download_root)
        pip = pipeline(
            pipeline_name,
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            # chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.compute_type,
            device=self.cuda,
        )
        return pip

    def __init__(self, **kwargs):

        if torch.cuda.is_available():
            self.cuda = torch.device("cuda:0")
            self.compute_type = torch.float32
            self.devicewhisper = "cuda"
        else:
            self.cuda = torch.device("cpu")
            self.compute_type = torch.float32
            self.devicewhisper = "cpu"
        self.download_root = kwargs.get("down_nmodel_path", "./models/")
        computer_type = "float32" if self.compute_type == torch.float32 else "float32"  # float32"
#         # "vinai/PhoWhisper-large"

        self.transcriber1 = AudioOfficial.loadmodel(
            self, "openai/whisper-large-v3", "automatic-speech-recognition")

        # self.model_id = "vinai/PhoWhisper-large"
        self.transcriber2 = AudioOfficial.loadmodel(
            self, "/home/gitlab/asr/models/PhoWhisper-large", "automatic-speech-recognition")
        self.diarize_model = DiarizationPipeline(
            use_auth_token="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", device=self.devicewhisper)
        # self.vadfilter = kwargs.get("vadfilter", None)
        # self.model = WhisperModel("./models/PhoWhisper-large-ct2" ,download_root=self.download_root, compute_type=computer_type, device=self.devicewhisper , num_workers= 4)
        print("Loading model with device: ", self.devicewhisper,
              " and compute type: ", computer_type)
        # self.vad_model = load_vad_model(
        #     device=self.cuda, use_auth_token=None, )

        # default_vad_options = {
        #     "vad_onset": 0.500,
        #     "vad_offset": 0.363
        # }
        # self.progesspipeline = FlaxWhisperPipline("openai/whisper-large-v3")

     #   self.transcriber2 = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-large" , device= self.cuda , return_timestamps=True #,chunk_length_s=30,batch_size=16,return_timestamps=False, max_new_tokens=128,)
    #    )
        # self.model = WhisperModel("large-v2" ,download_root=self.download_root, compute_type=computer_type, device=self.devicewhisper ,  num_workers= 4 , cpu_threads = 100)

        # self.translator = Translator("seamlessM4T_v2_large", vocoder_name_or_card="vocoder_v2",device=self.cuda , dtype=self.compute_type ,)
        # seamlessM4T_medium vocoder_36langs
        # seamlessM4T_v2_large vocoder_v2

        # self.translator = Translator("seamlessM4T_medium", vocoder_name_or_card="vocoder_36langs",device=self.cuda , dtype=self.compute_type ,)

    def ExtractText(self, path_audio, **kwargs):
        language = kwargs.get("language", "vi")
        # audio = whisperx.load_audio(path_audio)

        gc.collect()
        torch.cuda.empty_cache()
        input_audio = decode_audio(path_audio)
        # print(self.filterVAD(input_audio))

        filtered_audio, speech_chunks = self.filterVAD(input_audio)
        segments_list1 = self.transcriber1(filtered_audio, generate_kwargs={
                                           "language": "Vietnamese"})
        segments_list2 = self.transcriber2(filtered_audio, generate_kwargs={
                                           "language": "Vietnamese"})['text']
        segments = None
        if speech_chunks:
            segments = restore_speech_timestamps(
                segments, speech_chunks, 16000)

            # filtered_audio = self.filterVAD(input_audio)
        print("segments: {segments}")
        converted_json = convert_segments(segments_list1['chunks'])
        # print(json.dumps(converted_json, indent=2))
        print(converted_json)

        gc.collect()
        torch.cuda.empty_cache()
        # Resuft = self.progesspipeline(input_audio)

        diarize_segments = self.diarize_model(input_audio, max_speakers=2)
        result = assign_word_speakers(
            diarize_segments, converted_json)
        print(result)
        Result1 = format_text_segments(segments_list1['chunks'])
        Result2 = segments_list2
        return Result1, Result2, converted_json

    def run_asr_facebook(self, input_audio: str, target_language: str) -> str:
        out_texts, _ = self.translator.predict(
            input=input_audio,
            task_str="ASR",
            src_lang=target_language,
            tgt_lang=target_language,
        )
        return str(out_texts[0])

    def filterVAD(self, audio, vad_filter: bool = True, vad_parameters: Optional[Union[dict, VadOptions]] = None, sampling_rate=16000, clip_timestamps=0):
        duration = audio.shape[0] / sampling_rate
        duration_after_vad = duration
        print("Duration before VAD: ", duration)
        if vad_filter:
            if vad_parameters is None:
                vad_parameters = VadOptions()
            elif isinstance(vad_parameters, dict):
                vad_parameters = VadOptions(**vad_parameters)
            speech_chunks = get_speech_timestamps(audio, vad_parameters)
            audio = collect_chunks(audio, speech_chunks)
            duration_after_vad = audio.shape[0] / sampling_rate
            print("Duration after VAD: ", duration_after_vad)
            return audio, speech_chunks
        else:
            return audio, None
