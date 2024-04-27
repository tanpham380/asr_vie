import os
import torchaudio
from pydub import AudioSegment
import tempfile
import gc
import io

from typing import BinaryIO, Union
import itertools
import av
import numpy as np


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




# def crop_audio(audio_tensor, start_time, end_time):

#     audio = AudioSegment.from_file(audio_tensor)
#     segment = audio[start_time:end_time]
#     temp_path = "./run/uploads/temporary_audio.wav"
#     segment.export(temp_path, format="wav")
#     return temp_path

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
TEXT_SOURCE_LANGUAGE_NAMES = sorted([language_code_to_name[code] for code in text_source_language_codes])

# Target langs:
# S2ST / T2ST
s2st_target_language_codes = [
    "eng",
    "arb",
    "ben",
    "cat",
    "ces",
    "cmn",
    "cym",
    "dan",
    "deu",
    "est",
    "fin",
    "fra",
    "hin",
    "ind",
    "ita",
    "jpn",
    "kor",
    "mlt",
    "nld",
    "pes",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "spa",
    "swe",
    "swh",
    "tel",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "urd",
    "uzn",
    "vie",
]
# Language dict

S2ST_TARGET_LANGUAGE_NAMES = sorted([language_code_to_name[code] for code in s2st_target_language_codes])
T2ST_TARGET_LANGUAGE_NAMES = S2ST_TARGET_LANGUAGE_NAMES

# S2TT / T2TT / ASR
S2TT_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES
T2TT_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES
ASR_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES
