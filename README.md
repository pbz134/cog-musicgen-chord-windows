# Cog Implementation of MusicGen-Chord
[![Replicate](https://replicate.com/sakemin/musicgen-chord/badge)](https://replicate.com/sakemin/musicgen-chord) 

MusicGen Chord is the modified version of Meta's [MusicGen](https://github.com/facebookresearch/audiocraft) Melody model, which can generate music based on audio-based chord conditions or text-based chord conditions.

I further modified Sakemin's original Linux-exclusive implementation to natively work on Windows.

Removed:
- Docker dependency
- Linux dependency
- Cog dependency

TO DO:
- Optimize model using a proper Triton installation
- Optimize model generation speed (seems rather slow for now)
- Create requirements.txt

Usage:
- python predict.py --prompt "Relaxing piano" --chords "C G Am F" --bpm 80
- python predict.py --prompt "Epic soundtrack" --chords "C Em F G" --bpm 140 --duration 20 --output mp3 --model stereo-chord-large --seed 42
