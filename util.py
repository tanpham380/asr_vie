from gpustat import GPUStatCollection
import os
import psutil
import torchaudio
from pydub import AudioSegment
import tempfile
import gc
import io
import torch

from typing import BinaryIO, List, Optional, Union, NamedTuple
import itertools
import av
import numpy as np


class Word(NamedTuple):
    start: float
    end: float
    word: str
    probability: float


class Segment(NamedTuple):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]]


def convert_segments(data):
    """Converts a list of speech segments to a JSON object with 'segments' key.

    Args:
        data: A list of dictionaries, where each dictionary represents a speech segment
              with keys 'text', 'timestamp'. 'timestamp' is a tuple of (start, end) times.

    Returns:
        A dictionary with a key 'segments' containing a list of segment objects. Each segment
        object has keys 'text', 'start', and 'end'.
    """
    converted_data = {"segments": []}
    for segment in data:
        text = segment["text"]
        # Use 0.0 as default if timestamp is missing
        start = segment.get("timestamp", (0.0,))[0]
        # Use 0.0 as default if timestamp is missing
        end = segment.get("timestamp", (0.0,))[1]
        if end is not None:  # Only add segment if end is not None
            segment_obj = {
                "text": text,
                "start": round(start, 3),
                "end": round(end, 3)
            }
            converted_data["segments"].append(segment_obj)
    return converted_data

# def clearFolderContent(folder_path):
#     # Function to clear contents of a folder
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         try:
#             if os.path.isfile(file_path):
#                 os.unlink(file_path)
#         except Exception as e:
#             print(f"Failed to delete {file_path}. Reason: {e}")


def format_text_segments(text_segments):
    formatted_segments = []
    for segment in text_segments:
        start_time, end_time = segment['timestamp']

        if start_time is None or end_time is None:
            if start_time is None:
                start_time = 0.0
            if end_time is None:
                end_time = 0.0

        # if end_time - start_time < 0.5:
        #     continue
        start_time_str = format_timestamp(start_time)
        end_time_str = format_timestamp(end_time)
        formatted_segment = f"[{start_time_str}:{end_time_str}] : {segment['text']}"
        formatted_segments.append(formatted_segment)
    return formatted_segments


def decode_audio(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
):
    """Decodes the audio.

    Args:
      input_file: Path to the input file or a file-like object.
      sampling_rate: Resample the audio to this sample rate.
      split_stereo: Return separate left and right channels.

    Returns:
      A float32 Numpy array.

      If `split_stereo` is enabled, the function returns a 2-tuple with the
      separated left and right channels.
    """
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono" if not split_stereo else "stereo",
        rate=sampling_rate,
    )

    raw_buffer = io.BytesIO()
    dtype = None

    with av.open(input_file, mode="r", metadata_errors="ignore") as container:
        frames = container.decode(audio=0)
        frames = _ignore_invalid_frames(frames)
        frames = _group_frames(frames, 500000)
        frames = _resample_frames(frames, resampler)

        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)

    # It appears that some objects related to the resampler are not freed
    # unless the garbage collector is manually run.
    del resampler
    gc.collect()

    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)

    # Convert s16 back to f32.
    audio = audio.astype(np.float32) / 32768.0

    if split_stereo:
        left_channel = audio[0::2]
        right_channel = audio[1::2]
        return left_channel, right_channel

    return audio


def _group_frames(frames, num_samples=None):
    fifo = av.audio.fifo.AudioFifo()

    for frame in frames:
        frame.pts = None  # Ignore timestamp check.
        fifo.write(frame)

        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()

    if fifo.samples > 0:
        yield fifo.read()


def _resample_frames(frames, resampler):
    # Add None to flush the resampler.
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)


def _ignore_invalid_frames(frames):
    iterator = iter(frames)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue

# def preprocess_audio(input_audio: str, output_path: str = "./run/uploads/") -> str:
#     arr, org_sr = torchaudio.load(input_audio)
#     num_channels = arr.shape[0]
#     if num_channels == 2:
#         a = "Đây là stereo (dual channel) audio."
#     elif num_channels == 1:
#         a = "Đây là mono channel audio."
#     else:
#         a= "Không xác định được số kênh của tín hiệu âm thanh."
#     if arr.size(0) > 1:  # convert dual to mono channel
#         arr = arr.mean(dim=0, keepdim=True)
#     new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=16000)

#     if output_path is None or output_path == "":
#         # Save to a temporary file
#         with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
#             output_path = tmp_file.name
#             torchaudio.save(output_path, new_arr, sample_rate=16000)
#     else:
#         # Save to the provided output path
#         output_path = os.path.join(output_path, os.path.basename(input_audio))
#         torchaudio.save(output_path, new_arr, sample_rate=16000)

#     return output_path, a


def crop_audio(audio_tensor, sample_rate, start_time, end_time):
    """Crops a segment from the audio tensor.

    Args:
        audio_tensor: A torch.Tensor containing the audio waveform.
        sample_rate: The sample rate of the audio in Hz.
        start_time: The starting time of the segment in seconds.
        end_time: The ending time of the segment in seconds.

    Returns:
        A torch.Tensor containing the cropped audio segment.
    """

    # Convert start and end times to number of samples
    limit_range = 10000
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
#   if audio_tensor.shape[0] == 0 or audio_tensor.shape[1] == 0:
#       return None
    if (end_sample - start_sample) < limit_range:
        return None
#   end_sample = min(end_sample, audio_tensor.shape[1])

    # end_sample = min(end_sample, audio_tensor.shape[1])


#

    # Crop the audio tensor
    cropped_audio = audio_tensor[start_sample:end_sample]

    return cropped_audio


def get_assets_path():
    """Returns the path to the assets directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def format_timestamp(
    seconds: float,
    always_include_hours: bool = False,
    decimal_marker: str = ".",
) -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def update_gpu_status():
    if torch.cuda.is_available() == False:
        return "No Nvidia Device"
    try:
        gpu_stats = GPUStatCollection.new_query()
        for gpu in gpu_stats:
            # Assuming you want to monitor the first GPU, index 0
            gpu_id = gpu.index
            gpu_name = gpu.name
            gpu_utilization = gpu.utilization
            memory_used = gpu.memory_used
            memory_total = gpu.memory_total
            memory_utilization = (memory_used / memory_total) * 100
            gpu_status = (
                f"GPU {gpu_id}: {gpu_name}, Utilization: {gpu_utilization}%, Memory Used: {memory_used}MB, Memory Total: {memory_total}MB, Memory Utilization: {memory_utilization:.2f}%")
            return gpu_status

    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return torch_update_gpu_status()


def torch_update_gpu_status():
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.mem_get_info(0)
        total_memory = gpu_memory[1] / (1024 * 1024)
        free_memory = gpu_memory[0] / (1024 * 1024)
        used_memory = (gpu_memory[1] - gpu_memory[0]) / (1024 * 1024)

        gpu_status = f"GPU: {gpu_info} Free Memory:{free_memory}MB   Total Memory: {total_memory:.2f} MB  Used Memory: {used_memory:.2f} MB"
    else:
        gpu_status = "No GPU available"
    return gpu_status


def update_cpu_status():
    import datetime
    # Get the current time
    current_time = datetime.datetime.now().time()
    # Convert the time to a string
    time_str = current_time.strftime("%H:%M:%S")

    cpu_percent = psutil.cpu_percent()
    cpu_status = f"CPU Usage: {cpu_percent}% {time_str}"
    return cpu_status


def update_status():
    gpu_status = update_gpu_status()
    cpu_status = update_cpu_status()
    sys_status = gpu_status+"\n\n"+cpu_status
    return sys_status


def refresh_status():
    return update_status()


language_code_to_name = {
    "afr": "Afrikaans",
    "amh": "Amharic",
    "arb": "Modern Standard Arabic",
    "ary": "Moroccan Arabic",
    "arz": "Egyptian Arabic",
    "asm": "Assamese",
    "ast": "Asturian",
    "azj": "North Azerbaijani",
    "bel": "Belarusian",
    "ben": "Bengali",
    "bos": "Bosnian",
    "bul": "Bulgarian",
    "cat": "Catalan",
    "ceb": "Cebuano",
    "ces": "Czech",
    "ckb": "Central Kurdish",
    "cmn": "Mandarin Chinese",
    "cym": "Welsh",
    "dan": "Danish",
    "deu": "German",
    "ell": "Greek",
    "eng": "English",
    "est": "Estonian",
    "eus": "Basque",
    "fin": "Finnish",
    "fra": "French",
    "gaz": "West Central Oromo",
    "gle": "Irish",
    "glg": "Galician",
    "guj": "Gujarati",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hrv": "Croatian",
    "hun": "Hungarian",
    "hye": "Armenian",
    "ibo": "Igbo",
    "ind": "Indonesian",
    "isl": "Icelandic",
    "ita": "Italian",
    "jav": "Javanese",
    "jpn": "Japanese",
    "kam": "Kamba",
    "kan": "Kannada",
    "kat": "Georgian",
    "kaz": "Kazakh",
    "kea": "Kabuverdianu",
    "khk": "Halh Mongolian",
    "khm": "Khmer",
    "kir": "Kyrgyz",
    "kor": "Korean",
    "lao": "Lao",
    "lit": "Lithuanian",
    "ltz": "Luxembourgish",
    "lug": "Ganda",
    "luo": "Luo",
    "lvs": "Standard Latvian",
    "mai": "Maithili",
    "mal": "Malayalam",
    "mar": "Marathi",
    "mkd": "Macedonian",
    "mlt": "Maltese",
    "mni": "Meitei",
    "mya": "Burmese",
    "nld": "Dutch",
    "nno": "Norwegian Nynorsk",
    "nob": "Norwegian Bokm\u00e5l",
    "npi": "Nepali",
    "nya": "Nyanja",
    "oci": "Occitan",
    "ory": "Odia",
    "pan": "Punjabi",
    "pbt": "Southern Pashto",
    "pes": "Western Persian",
    "pol": "Polish",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "slk": "Slovak",
    "slv": "Slovenian",
    "sna": "Shona",
    "snd": "Sindhi",
    "som": "Somali",
    "spa": "Spanish",
    "srp": "Serbian",
    "swe": "Swedish",
    "swh": "Swahili",
    "tam": "Tamil",
    "tel": "Telugu",
    "tgk": "Tajik",
    "tgl": "Tagalog",
    "tha": "Thai",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzn": "Northern Uzbek",
    "vie": "Vietnamese",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "yue": "Cantonese",
    "zlm": "Colloquial Malay",
    "zsm": "Standard Malay",
    "zul": "Zulu",
}

LANGUAGE_NAME_TO_CODE = {v: k for k, v in language_code_to_name.items()}

# Source langs: S2ST / S2TT / ASR don't need source lang
# T2TT / T2ST use this
text_source_language_codes = [
    "afr",
    "amh",
    "arb",
    "ary",
    "arz",
    "asm",
    "azj",
    "bel",
    "ben",
    "bos",
    "bul",
    "cat",
    "ceb",
    "ces",
    "ckb",
    "cmn",
    "cym",
    "dan",
    "deu",
    "ell",
    "eng",
    "est",
    "eus",
    "fin",
    "fra",
    "gaz",
    "gle",
    "glg",
    "guj",
    "heb",
    "hin",
    "hrv",
    "hun",
    "hye",
    "ibo",
    "ind",
    "isl",
    "ita",
    "jav",
    "jpn",
    "kan",
    "kat",
    "kaz",
    "khk",
    "khm",
    "kir",
    "kor",
    "lao",
    "lit",
    "lug",
    "luo",
    "lvs",
    "mai",
    "mal",
    "mar",
    "mkd",
    "mlt",
    "mni",
    "mya",
    "nld",
    "nno",
    "nob",
    "npi",
    "nya",
    "ory",
    "pan",
    "pbt",
    "pes",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "slv",
    "sna",
    "snd",
    "som",
    "spa",
    "srp",
    "swe",
    "swh",
    "tam",
    "tel",
    "tgk",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "urd",
    "uzn",
    "vie",
    "yor",
    "yue",
    "zsm",
    "zul",
]
TEXT_SOURCE_LANGUAGE_NAMES = sorted(
    [language_code_to_name[code] for code in text_source_language_codes]
)

_LANGUAGE_CODES = (
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
    "yue",
)

# Language dict

ASR_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES


# ASR_TARGET_LANGUAGE_CODES = [LANGUAGE_NAME_TO_CODE.get(name, None) for name in ASR_TARGET_LANGUAGE_NAMES]
# ASR_TARGET_LANGUAGE_CODES = [code for code in ASR_TARGET_LANGUAGE_CODES if code is not None]

# _LANGUAGE_CODE_TO_NAME = {
#     code: language_code_to_name[code] for code in _LANGUAGE_CODES}
