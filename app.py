import gradio as gr
import os
import time
from pathlib import Path
from predict import MusicGenPredictor
import argparse
import tempfile
import shutil

# Initialize the predictor
predictor = MusicGenPredictor()

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)

def generate_music(
    prompt,
    chords,
    bpm,
    duration,
    output_format,
    model_version,
    seed,
    progress=gr.Progress()
):
    # Create a unique filename based on prompt and timestamp
    timestamp = int(time.time())
    safe_prompt = "".join(c for c in prompt if c.isalnum() or c in " _-")[:50]
    filename = f"{safe_prompt}_{timestamp}"
    
    try:
        # Generate the music
        temp_file = predictor.generate_music(
            prompt=prompt,
            text_chords=chords,
            bpm=int(bpm),
            duration=int(duration),
            output_format=output_format,
            model_version=model_version,
            seed=int(seed) if seed else None,
        )
        
        # Determine the final output path
        output_filename = f"{filename}.{output_format}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Move the file from temp to output directory
        shutil.move(temp_file, output_path)
        
        # For WAV files, also create an MP3 version for better web playback
        if output_format == "wav":
            mp3_path = os.path.join(output_dir, f"{filename}.mp3")
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
            
            ffmpeg_cmd = f'ffmpeg -i "{output_path}" "{mp3_path}"'
            os.system(ffmpeg_cmd)
            
            return output_path, mp3_path
        else:
            return output_path, None
        
    except Exception as e:
        raise gr.Error(f"Error generating music: {str(e)}")

# Gradio UI
with gr.Blocks(title="MusicGen Chord - AI Music Generator", theme="soft") as demo:
    gr.Markdown("""
    # 🎵 MusicGen Chord - AI Music Generator
    Generate music with chord conditioning. Provide a description and chord progression to create custom music.
    """)
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Music Description",
                placeholder="e.g. Relaxing piano, Epic soundtrack, Happy jazz",
                value="Relaxing piano"
            )
            chords = gr.Textbox(
                label="Chord Progression (space separated)",
                placeholder="e.g. C G Am F",
                value="C G Am F"
            )
            bpm = gr.Slider(
                label="Tempo (BPM)",
                minimum=40,
                maximum=240,
                value=120,
                step=1
            )
            duration = gr.Slider(
                label="Duration (seconds)",
                minimum=1,
                maximum=60,
                value=10,
                step=1
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                model_version = gr.Dropdown(
                    label="Model Version",
                    choices=["stereo-chord-large", "melody-chord-large"],
                    value="stereo-chord-large"
                )
                output_format = gr.Radio(
                    label="Output Format",
                    choices=["wav", "mp3"],
                    value="wav"
                )
                seed = gr.Number(
                    label="Random Seed (-1 for random)",
                    value=-1,
                    precision=0
                )
            
            generate_btn = gr.Button("Generate Music", variant="primary")
        
        with gr.Column():
            audio_output_wav = gr.Audio(label="Generated Music (WAV)", visible=True)
            audio_output_mp3 = gr.Audio(label="Generated Music (MP3)", visible=False)
            download_wav = gr.File(label="Download WAV", visible=False)
            download_mp3 = gr.File(label="Download MP3", visible=False)
            
            # Hidden elements for file paths
            wav_path = gr.Textbox(visible=False)
            mp3_path = gr.Textbox(visible=False)
    
    # Examples
    gr.Examples(
        examples=[
            ["Relaxing piano", "C G Am F", 80, 10, "wav", "stereo-chord-large", -1],
            ["Epic soundtrack", "C Em F G", 140, 20, "mp3", "stereo-chord-large", 42],
            ["Happy jazz", "Dm7 G7 Cmaj7 Fmaj7", 120, 15, "wav", "stereo-chord-large", -1],
            ["Lo-fi beats", "Am F C G", 90, 30, "mp3", "melody-chord-large", -1]
        ],
        inputs=[prompt, chords, bpm, duration, output_format, model_version, seed],
        outputs=[audio_output_wav, audio_output_mp3],
        fn=generate_music,
        cache_examples=False
    )
    
    # Generation function
    generate_btn.click(
        fn=generate_music,
        inputs=[prompt, chords, bpm, duration, output_format, model_version, seed],
        outputs=[wav_path, mp3_path]
    )
    
    # Update UI based on outputs
    def update_audio_outputs(wav_path, mp3_path, output_format):
        outputs = []
        
        # For WAV format, show both WAV and MP3
        if output_format == "wav":
            outputs.extend([
                gr.Audio(value=wav_path, visible=True),
                gr.Audio(value=mp3_path, visible=mp3_path is not None),
                gr.File(value=wav_path, visible=True),
                gr.File(value=mp3_path, visible=mp3_path is not None)
            ])
        # For MP3 format, only show MP3
        else:
            outputs.extend([
                gr.Audio(visible=False),
                gr.Audio(value=wav_path, visible=True),
                gr.File(visible=False),
                gr.File(value=wav_path, visible=True)
            ])
        
        return outputs
    
    # Update the UI when generation is complete
    wav_path.change(
        fn=update_audio_outputs,
        inputs=[wav_path, mp3_path, output_format],
        outputs=[audio_output_wav, audio_output_mp3, download_wav, download_mp3]
    )
    
    # Also update when format changes
    output_format.change(
        fn=lambda fmt: [
            gr.Audio(visible=fmt == "wav"),
            gr.Audio(visible=fmt != "wav"),
            gr.File(visible=fmt == "wav"),
            gr.File(visible=fmt != "wav")
        ],
        inputs=output_format,
        outputs=[audio_output_wav, audio_output_mp3, download_wav, download_mp3]
    )

if __name__ == "__main__":
    # Load the model at startup
    predictor.load_model()
    
    # Parse command line arguments for Gradio settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Server address")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    args = parser.parse_args()
    
    # Launch the Gradio interface
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )
