

import torch
import gc

# from seamless_communication.inference import Translator


# from faster_whisper import WhisperModel

import whisperx

from contextlib import nullcontext

from util import format_timestamp


class AudioOfficial:

    def __init__(self, **kwargs):

        if torch.cuda.is_available():
            self.cuda = torch.device("cuda:0")
            self.compute_type = torch.float16
            self.devicewhisper = "cuda"
        else:
            self.cuda = torch.device("cpu")
            self.compute_type = torch.float32
            self.devicewhisper = "cpu"
        self.download_root = kwargs.get("down_nmodel_path", "./models/")
        self.computer_type = "float32" if self.compute_type == torch.float32 else "int8"  # float32"

        self.translate1 = whisperx.load_model("large-v2", device=self.devicewhisper, compute_type=self.computer_type, download_root=self.download_root, threads=100,
                                              asr_options={
                                                  "max_new_tokens": 120,


                                              },
                                              #   language="vi",
                                              vad_options={
                                                  "vad_onset": 0.500,
                                                  "vad_offset": 0.363
                                              })
        self.translate2 = whisperx.load_model("/home/gitlab/asr/models/PhoWhisper-large-ct", device=self.devicewhisper, compute_type=self.computer_type, download_root=self.download_root, threads=100,
                                              asr_options={
                                                  "max_new_tokens": 120,
                                              },
                                              language="vi",
                                              vad_options={
                                                  "vad_onset": 0.500,
                                                  "vad_offset": 0.363
                                              }
                                              )
        self.diarize_model = whisperx.DiarizationPipeline(
            use_auth_token="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", device=self.devicewhisper)

        print("Loading model with device: ", self.devicewhisper,
              " and compute type: ", self.computer_type)

    # def ExtractText(self, path_audio, **kwargs):
    #     language = kwargs.get("language", "vi")
    #     chunksize = kwargs.get("chunk", "16")
    #     batch_size = 32
    #     self.remove_cache()
    #     # audio = whisperx.load_audio(path_audio)
    #     input_audio = whisperx.load_audio(path_audio)

    #     segments_list1 = self.translate1.transcribe(
    #         input_audio, batch_size=batch_size, print_progress=False, chunk_size=chunksize)
    #     self.remove_cache()
    #     model_a, metadata = whisperx.load_align_model(
    #         language_code=segments_list1["language"], device=self.devicewhisper, compute_type=self.compute_type, download_root=self.download_root)
    #     result = whisperx.align(
    #         result["segments"], model_a, metadata, input_audio, self.devicewhisper, return_char_alignments=False)
    #     self.remove_cache()
    #     diarize_segments = self.diarize_model(
    #         input_audio, max_speakers=2, min_speakers=1)
    #     result = whisperx.assign_word_speakers(diarize_segments, result)
    #     self.remove_cache()

    #     segments_list1 = self.translate2.transcribe(
    #         input_audio, batch_size=batch_size, print_progress=False, chunk_size=chunksize)
    #     self.remove_cache()
    #     model_a, metadata = whisperx.load_align_model(
    #         language_code=segments_list1["language"], device=self.devicewhisper, compute_type=self.compute_type, download_root=self.download_root)
    #     result2 = whisperx.align(
    #         result["segments"], model_a, metadata, input_audio, self.devicewhisper, return_char_alignments=False)
    #     self.remove_cache()

    #     return result, result2
    def ExtractText(self, path_audio, **kwargs):
        language = kwargs.get("language", "Vietnamese")
        chunksize = kwargs.get("chunk", "16")
        batch_size = 32

        input_audio = whisperx.load_audio(path_audio)

        if self.remove_cache():
            result1 = self.transcribe_and_align(
                self.translate1,
                input_audio,
                self.devicewhisper,
                chunksize,
                batch_size,
                True,
                False
            )
        # print(result1)
        if self.remove_cache():
            diarize_segments = self.diarize_model(
                input_audio,
                max_speakers=2,
                min_speakers=1
            )
            result1 = whisperx.assign_word_speakers(diarize_segments, result1)

        ctx = self.remove_cache()
        if ctx:
            result2 = self.transcribe_and_align(
                self.translate2,
                input_audio,
                self.devicewhisper,
                99999,
                batch_size,
                False,
                True,

            )

        def format_segment_timestamps(segment):
            return f"{format_timestamp(segment['start'])} {format_timestamp(segment['end'])}"

        # # self.translate2.transcribe()
        # Text = [
        #     f"{format_segment_timestamps(segment)} {segment['speaker']} {segment['text']} \n"
        #     for segment in result1["segments"]
        #     # More descriptive name
        #     if segment["end"] - segment["start"] >= 0.6
        #     and "speaker" in segment
        #     and segment["text"] not in ["Hẹn gặp lại các bạn trong những video tiếp theo nhé!", "Cảm ơn các bạn đã theo dõi."]
        # ]
        Text = " ".join([
            f"{format_segment_timestamps(segment)} {segment['speaker']} {segment['text']}"
            for segment in result1["segments"]
            if segment["end"] - segment["start"] >= 0.6
            and "speaker" in segment
            and segment["text"] not in ["Hẹn gặp lại các bạn trong những video tiếp theo nhé!", "Cảm ơn các bạn đã theo dõi."]
        ])

        # for i in result1["segments"]:
        #     if i["end"] - i["start"] < 0.7:
        #         continue
        #     if "speaker" not in i:
        #         continue
        #     if i["text"] in [' Hẹn gặp lại các bạn trong những video tiếp theo nhé!', ' Cảm ơn các bạn đã theo dõi.']:

        #         continue

        #     Text += format_timestamp(i["start"]) + " " + format_timestamp(i["end"]) + " " + \
        # #         i["speaker"] + " " + i["text"] + "\n"
        # Text2 = [segment["text"] for segment in result2["segments"]
        #          if "text" in segment]  # Handle potential missing key
        Text2 = " ".join([segment["text"]
                         for segment in result2["segments"] if "text" in segment])
        Text = Text[1:-1].replace("'", "")
        Text2 = Text2[1:-1].replace("'", "")

        return Text, Text2

    def remove_cache(self):
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            return False
        return True

    def transcribe_and_align(self, translator, audio, device, chunksize, batch_size, have_align=True,  set_charalignments=False):
        segments = translator.transcribe(
            audio, print_progress=False, batch_size=batch_size, chunk_size=chunksize)

        if have_align:
            model_a, metadata = whisperx.load_align_model(
                language_code=segments["language"],
                device=device,
            )
            result = whisperx.align(
                segments["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=set_charalignments
            )
        else:
            result = segments
        return result
