# Modified for native Windows support with CLI arguments
import os
import sys
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
os.environ["HOME"] = os.path.expanduser("~")

import random
import typing as tp
from pathlib import Path
import numpy as np
import torch
import torchaudio
import omegaconf
from typing import Any
import argparse

# Fix for PyTorch weights_only error
from omegaconf import DictConfig, OmegaConf
torch.serialization.add_safe_globals([DictConfig, Any, OmegaConf])

# Configure BTC chord extraction
BTC_CONFIG_PATH = str(Path(__file__).parent / "audiocraft" / "modules" / "btc" / "run_config.yaml")

if not Path(BTC_CONFIG_PATH).exists():
    raise FileNotFoundError(f"BTC config file not found at {BTC_CONFIG_PATH}")

from audiocraft.modules.btc.utils import hparams as btc_hparams
from audiocraft.modules.btc.utils import chords as btc_chords

# Patch HParams config loading
original_hparams_load = btc_hparams.HParams.load

def patched_hparams_load(path, *args, **kwargs):
    if path == "/src/audiocraft/modules/btc/run_config.yaml":
        return original_hparams_load(BTC_CONFIG_PATH, *args, **kwargs)
    return original_hparams_load(path, *args, **kwargs)

btc_hparams.HParams.load = patched_hparams_load

# Improved chord parser patch
if hasattr(btc_chords, 'Chords'):
    original_chord_parser = btc_chords.Chords.chord
    
    def patched_chord_parser(self, chord_str):
        try:
            # First try the original parser
            return original_chord_parser(self, chord_str)
        except Exception:
            # Enhanced chord format handling
            try:
                chord_str = chord_str.strip()
                if chord_str.lower().endswith('m'):
                    return original_chord_parser(self, f"{chord_str[:-1].upper()}:min")
                elif ':' in chord_str:
                    return original_chord_parser(self, chord_str)
                else:
                    return original_chord_parser(self, f"{chord_str}:maj")
            except Exception:
                print(f"Warning: Failed to parse chord '{chord_str}' - using C major")
                return original_chord_parser(self, "C:maj")
    
    btc_chords.Chords.chord = patched_chord_parser
else:
    print("Warning: Could not patch chord parser")

# Model imports
from audiocraft.solvers.compression import CompressionSolver
from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.models.loaders import (
    load_compression_model,
    load_lm_model,
)
from audiocraft.data.audio import audio_write
from audiocraft.models.builders import get_lm_model
import subprocess
import tempfile

# Path configuration
MODEL_PATH = str(Path(__file__).parent / "musicgen_models")
os.makedirs(MODEL_PATH, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH
os.environ["TORCH_HOME"] = MODEL_PATH

def _delete_param(cfg, full_name: str):
    parts = full_name.split('.')
    for part in parts[:-1]:
        if part in cfg:
            cfg = cfg[part]
        else:
            return
    OmegaConf.set_struct(cfg, False)
    if parts[-1] in cfg:
        del cfg[parts[-1]]
    OmegaConf.set_struct(cfg, True)

def load_ckpt(path, device, url=False):
    if url:
        loaded = torch.hub.load_state_dict_from_url(str(path), map_location=device, weights_only=False)
    else:
        loaded = torch.load(str(path), map_location=device, weights_only=False)
        
    cfg = OmegaConf.create(loaded['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    _delete_param(cfg, 'conditioners.self_wav.chroma_chord.cache_path')
    _delete_param(cfg, 'conditioners.self_wav.chroma_stem.cache_path')
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')

    lm = get_lm_model(loaded['xp.cfg'])
    lm.load_state_dict(loaded['model']) 
    lm.eval()
    lm.cfg = cfg
    compression_model = CompressionSolver.model_from_checkpoint(cfg.compression_model_checkpoint, device=device)
    return MusicGen("musicgen-chord", compression_model, lm)

class MusicGenPredictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mbd = None
        try:
            self.mbd = MultiBandDiffusion.get_mbd_musicgen()
        except Exception as e:
            print(f"Warning: MultiBandDiffusion disabled - {str(e)}")
        self.model = None

    def load_model(self, model_version="stereo-chord-large", weights_path=None):
        """Load the model into memory"""
        if weights_path:
            print("Fine-tuned model weights loaded!")
            self.model = load_ckpt(weights_path, self.device, url=True)
        else:
            model_file = f'musicgen-{model_version}.th'
            model_path = os.path.join(MODEL_PATH, model_file)
            
            if not os.path.isfile(model_path):
                url = f"https://weights.replicate.delivery/default/musicgen-chord/musicgen-{model_version}.th"
                print(f"Downloading model weights from {url}")
                try:
                    import urllib.request
                    urllib.request.urlretrieve(url, model_path)
                except Exception as e:
                    print(f"Failed to download: {e}")
                    return False
            
            self.model = load_ckpt(model_path, self.device)
            self.model.lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True
        return True

    def generate_music(
        self,
        model_version="stereo-chord-large",
        prompt=None,
        text_chords=None,
        bpm=None,
        time_sig="4/4",
        audio_chords_path=None,
        audio_start=0,
        audio_end=None,
        duration=8,
        continuation=False,
        multi_band_diffusion=False,
        normalization_strategy="loudness",
        chroma_coefficient=1.0,
        top_k=250,
        top_p=0.0,
        temperature=1.0,
        classifier_free_guidance=3,
        output_format="wav",
        seed=None,
    ):
        """Main generation function"""
        if text_chords == '':
            text_chords = None
        
        if text_chords and audio_chords_path and not continuation:
            raise ValueError("Must provide either only one of `audio_chords` or `text_chords`.")
        if text_chords and not bpm:
            raise ValueError("There must be `bpm` value set when text based chord conditioning.")
        if text_chords and (not time_sig or time_sig==""):
            raise ValueError("There must be `time_sig` value set when text based chord conditioning.")
        if continuation and not audio_chords_path:
            raise ValueError("Must provide an audio input file via `audio_chords` if continuation is `True`.")
        if multi_band_diffusion and int(self.model.lm.cfg.transformer_lm.n_q) == 8:
            raise ValueError("Multi-band Diffusion only works with non-stereo models.")
        
        if prompt is None:
            prompt = ''

        if time_sig is not None and not time_sig == '':
            prompt = f"{prompt}, {time_sig}" if prompt else time_sig
        if bpm is not None:
            prompt = f"{prompt}, bpm : {bpm}" if prompt else str(bpm)

        if not self.model:
            if not self.load_model(model_version):
                raise Exception("Failed to load model")

        model = self.model

        def set_generation_params(duration):
            model.set_generation_params(
                duration=duration,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                cfg_coef=classifier_free_guidance,
            )

        model.lm.condition_provider.conditioners['self_wav'].chroma_coefficient = chroma_coefficient

        if not seed or seed == -1:
            seed = torch.seed() % 2 ** 32 - 1
            self.set_all_seeds(seed)
        self.set_all_seeds(seed)
        print(f"Using seed {seed}")

        if not audio_chords_path: 
            set_generation_params(duration)
            if text_chords is None or text_chords == '':
                wav, tokens = model.generate([prompt], progress=True, return_tokens=True)
            else:
                wav, tokens = model.generate_with_text_chroma(
                    descriptions=[prompt], 
                    chord_texts=[text_chords], 
                    bpm=[bpm], 
                    meter=[int(time_sig.split('/')[0])], 
                    progress=True, 
                    return_tokens=True
                )
        else:
            audio_chords, sr = torchaudio.load(audio_chords_path)
            audio_chords = audio_chords[None] if audio_chords.dim() == 2 else audio_chords

            audio_start = 0 if not audio_start else audio_start
            if audio_end is None or audio_end == -1:
                audio_end = audio_chords.shape[2] / sr

            if audio_start > audio_end:
                raise ValueError("`audio_start` must be less than or equal to `audio_end`")

            audio_chords_wavform = audio_chords[
                ..., int(sr * audio_start) : int(sr * audio_end)
            ]

            if continuation: 
                set_generation_params(duration)
                if text_chords is None or text_chords == '':
                    wav, tokens = model.generate_continuation(
                        prompt=audio_chords_wavform,
                        prompt_sample_rate=sr,
                        descriptions=[prompt],
                        progress=True,
                        return_tokens=True
                    )                        
                else:
                    wav, tokens = model.generate_continuation_with_text_chroma(
                        audio_chords_wavform, sr, [prompt], [text_chords], 
                        bpm=[bpm], meter=[int(time_sig.split('/')[0])], 
                        progress=True, return_tokens=True
                    )
            else:
                set_generation_params(duration)
                wav, tokens = model.generate_with_chroma(
                    [prompt], audio_chords_wavform, sr, progress=True, return_tokens=True
                )

        if multi_band_diffusion and self.mbd is not None:
            wav = self.mbd.tokens_to_wav(tokens)

        output_dir = tempfile.gettempdir()
        wav_path = os.path.join(output_dir, "out.wav")
        
        audio_write(
            os.path.join(output_dir, "out"),
            wav[0].cpu(),
            model.sample_rate,
            strategy=normalization_strategy,
        )

        if output_format == "mp3":
            mp3_path = os.path.join(output_dir, "out.mp3")
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
            
            ffmpeg_cmd = f'ffmpeg -i "{wav_path}" "{mp3_path}"'
            subprocess.call(ffmpeg_cmd, shell=True)
            
            os.remove(wav_path)
            return mp3_path
        else:
            return wav_path

    @staticmethod
    def set_all_seeds(seed):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate music with chord conditioning')
    parser.add_argument('--prompt', type=str, default="A happy jazz tune",
                      help='Description of the music style')
    parser.add_argument('--chords', type=str, default="C G Am F",
                      help='Chord progression (space separated)')
    parser.add_argument('--bpm', type=int, default=120,
                      help='Tempo in beats per minute')
    parser.add_argument('--duration', type=int, default=10,
                      help='Duration in seconds')
    parser.add_argument('--output', type=str, default="wav",
                      choices=["wav", "mp3"], help='Output format')
    parser.add_argument('--model', type=str, default="stereo-chord-large",
                      help='Model version to use')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')

    args = parser.parse_args()

    predictor = MusicGenPredictor()
    predictor.load_model(model_version=args.model)
    
    output_file = predictor.generate_music(
        prompt=args.prompt,
        text_chords=args.chords,
        bpm=args.bpm,
        duration=args.duration,
        output_format=args.output,
        seed=args.seed
    )
    
    print(f"Generated music saved to: {output_file}")
