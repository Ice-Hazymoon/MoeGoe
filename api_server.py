import base64
import re
# import tempfile
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Union

# import librosa
# import numpy as np
import scipy
import torch
from pydantic import BaseModel
from torch import no_grad, LongTensor
import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence
import soundfile as sf


# , _clean_text

# from mel_processing import spectrogram_torch


# 类型
class MODEL_TYPE(Enum):
    TTS = "tts"
    W2V2 = "w2v2"
    HUBERT_SOFT = "hubert-soft"


class Utils(object):
    @staticmethod
    def get_text(text, hps, cleaned=False):
        if cleaned:
            text_norm = text_to_sequence(text, hps.symbols, [])
        else:
            text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm

    @staticmethod
    def get_speakers(speakers, escape=False):
        if len(speakers) > 100:
            return
        _speaker = []
        for ids, name in enumerate(speakers):
            if escape:
                name = name.encode('unicode_escape').decode()
            _speaker.append([ids, name])
        return _speaker

    @staticmethod
    def get_label_value(text, label, default, warning_name='value'):
        value = re.search(rf'\[{label}=(.+?)\]', text)
        if value:
            try:
                text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
                value = float(value.group(1))
            except Exception as e:
                print(f'Invalid {warning_name}!', e)
        else:
            value = default
        return value, text

    @staticmethod
    def get_label(text, label):
        if f'[{label}]' in text:
            return True, text.replace(f'[{label}]', '')
        else:
            return False, text

    @staticmethod
    def deal_hps_ms(hps_ms):
        # 角色
        _n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
        # 符号
        _n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
        _emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False
        #
        _speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
        _use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False
        return _n_speakers, _n_symbols, _emotion_embedding, _speakers, _use_f0

    @staticmethod
    def get_model_config_path(model_path):
        _model = model_path
        _config = f"{model_path}.json"
        return _model, _config


class TTS_Generate(object):
    def __init__(self, model_path: str, device: str = "cpu"):
        self._out_path = f"./tts/{0}.wav"
        self.model_path = model_path
        _model, _config = Utils.get_model_config_path(model_path=self.model_path)
        hps = utils.get_hparams_from_file(_config)
        self.hps_ms = hps
        model = SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model)
        utils.load_checkpoint(_model, model)
        devicer = torch.device(device)
        model.eval().to(devicer)
        self.net_g_ms = model
        self.n_speakers, self.n_symbols, self.emotion_embedding, self.speakers, self.use_f0 = Utils.deal_hps_ms(
            hps_ms=hps)
        if self.n_symbols != 0:
            if not self.emotion_embedding:
                self.model_type = MODEL_TYPE.TTS
            else:
                self.model_type = MODEL_TYPE.W2V2
        else:
            self.model_type = MODEL_TYPE.HUBERT_SOFT

    def check_model(self):
        _model, _config = Utils.get_model_config_path(model_path=self.model_path)
        if not Path(_model).exists() or not Path(_config).exists():
            return False, TTS_REQ_DATA(code=404, msg=f"CANT FIND MODEL", audio="").dict()
        return True, TTS_REQ_DATA(code=200, msg=f"YOU SHOULD NOT SEE THIS MSG", audio="").dict()

    @staticmethod
    def no_found(msg):
        return TTS_REQ_DATA(code=404, msg=f"{msg}:unknown or unsupported type TTS", audio="").dict()

    def get_speaker_list(self):
        # 二维数组 [id,name]
        id_list = []
        # if len(self.speakers) > 100:
        #     print("TOO MANY SPEAKERS")
        #     return id_list.append({"id": 0, "name": "default"})
        for ids, name in enumerate(self.speakers):
            id_list.append({"id": ids, "name": name})
        return id_list

    def ordinary(self,
                 c_text: str,
                 speaker_ids: int = 0,
                 audio_type: str = Union["ogg", "wav", "flac"],
                 length: float = 1,
                 noise: float = 0.667,
                 noise_w: float = 0.8):
        _msg = "ok"
        # 的链式调用
        _length_scale, c_text = Utils.get_label_value(c_text, 'LENGTH', length, 'length scale')
        _noise_scale, c_text = Utils.get_label_value(c_text, 'NOISE', noise, 'noise scale')
        _noise_scale_w, c_text = Utils.get_label_value(c_text, 'NOISEW', noise_w, 'deviation of noise')
        _cleaned, c_text = Utils.get_label(c_text, 'CLEANED')
        _stn_tst = Utils.get_text(c_text, self.hps_ms, cleaned=_cleaned)
        # 确定 ID
        find = False
        speaker_name = "none"
        speaker_list = self.get_speaker_list()
        for item in speaker_list:
            if speaker_ids == item["id"]:
                speaker_name = item['name']
                find = True
        if not find:
            speaker_ids = speaker_list[0]["id"]
            speaker_name = speaker_list[0]["name"]
            _msg = "Not Find Speaker,Use 0"
        # 构造对应 tensor
        with no_grad():
            _x_tst = _stn_tst.unsqueeze(0)
            _x_tst_lengths = LongTensor([_stn_tst.size(0)])
            _sid = LongTensor([speaker_ids])
            _audio = self.net_g_ms.infer(_x_tst, _x_tst_lengths, sid=_sid, noise_scale=_noise_scale, noise_scale_w=_noise_scale_w,
                                         length_scale=1.0 / _length_scale)[0][0, 0].data.cpu().float().numpy()
        # 写出返回
        file = BytesIO()
        # 使用 scipy 将 Numpy 数据写入字节流
        if audio_type == "ogg":
            sf.write(file, _audio, self.hps_ms.data.sampling_rate, format='ogg', subtype='vorbis')
        elif audio_type == "wav":
            # Write out audio as 24bit PCM WAV
            sf.write(file, _audio, self.hps_ms.data.sampling_rate, format='wav', subtype='PCM_24')
        elif audio_type == "flac":
            # Write out audio as 24bit Flac
            sf.write(file, _audio, self.hps_ms.data.sampling_rate, format='flac', subtype='PCM_24')
        else:
            scipy.io.wavfile.write(file, self.hps_ms.data.sampling_rate, _audio)
        # 获取 wav 数据
        voice_data = file.getvalue()
        audio_base64 = base64.b64encode(voice_data)
        return TTS_REQ_DATA(code=200,
                            msg=_msg,
                            audio=audio_base64,
                            speaker=speaker_name,
                            model_type="tts").dict()

    def convert(self, text, task_id: int = 1, speaker_ids: int = 0, audio_type: str = Union["ogg", "wav", "flac"]):
        """
        逻辑调度类
        """
        self._out_path = f"./tts/{task_id}.wav"
        _type = self.model_type
        if _type == MODEL_TYPE.TTS:
            _res = self.ordinary(c_text=text, speaker_ids=speaker_ids, audio_type=audio_type)
            return _res

        elif _type == MODEL_TYPE.HUBERT_SOFT:
            _res = self.ordinary(c_text=text, speaker_ids=speaker_ids, audio_type=audio_type)
            return _res

        elif _type == MODEL_TYPE.W2V2:
            _res = self.ordinary(c_text=text, speaker_ids=speaker_ids, audio_type=audio_type)
            return _res
        else:
            _res = self.no_found(msg=f"Unknown type {_type}")
            return _res


class TTS_REQ(BaseModel):
    model_name: str = ""
    task_id: int = 1
    text: str = "[ZH]你好[ZH]"
    speaker_id: int = 0
    audio_type: str = "ogg"


class TTS_REQ_DATA(BaseModel):
    code: int = 404
    msg: str = "unknown error"
    audio: str = ""
    speaker: str = ""
    model_type: str = ""
