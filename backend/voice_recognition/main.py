from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
import whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
import tempfile
import os
import torch
import asyncio
from pathlib import Path
import librosa
from transformers.models.whisper import tokenization_whisper
import time  # Add this import at the top
from pydub import AudioSegment
import soundfile as sf
import numpy as np
import noisereduce as nr
import subprocess
from starlette.background import BackgroundTask

# Add this for Malaysian model
tokenization_whisper.TASK_IDS = ["translate", "transcribe", "transcribeprecise"]

PROJECT_ROOT = Path(__file__).parent
MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "huggingface"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

torch.set_num_threads(1)

# Updated model configurations with clearer country labeling
COUNTRY_MODELS = {
    "Malaysia": {
        "name": "Malaysian Whisper Model",
        "model_id": "mesolitica/malaysian-whisper-small-v3",
        "language": "ms",
        "type": "malaysian"
    },
    "Singapore": {
        "name": "Singlish Whisper Model",
        "model_id": "jensenlwt/whisper-small-singlish-122k",
        "language": "en",
        "type": "pipeline"
    },
    "Thailand": {
        "name": "Thai Whisper Model",
        "model_id": "juierror/whisper-tiny-thai",
        "language": "Thai",
        "type": "thai"
    }
}

class ModelHandler:
    def __init__(self, model_config, cache_dir):
        self.config = model_config
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None
        self.pipeline = None
        self.device = "cpu"  # Always use CPU

    async def load(self):
        try:
            model_type = self.config["type"]
            model_id = self.config["model_id"]
            
            if model_type == "pipeline":
                print(f"Loading pipeline model: {model_id}")
                try:
                    # Create pipeline directly
                    self.pipeline = pipeline(
                        task="automatic-speech-recognition",
                        model=model_id,
                        chunk_length_s=30,
                        device=self.device
                    )
                    print(f"Pipeline model loaded successfully: {model_id}")
                    return True
                except Exception as e:
                    print(f"Error creating pipeline: {str(e)}")
                    raise
            
            elif model_type == "malaysian":
                print(f"Loading Malaysian model: {model_id}")
                # Load processor first
                self.processor = WhisperProcessor.from_pretrained(
                    model_id,
                    cache_dir=self.cache_dir
                )
                
                # Load model for CPU
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model_id,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float32  # Use float32 for CPU
                ).to(self.device)
                
                # Set to eval mode
                self.model.eval()
                print("Malaysian model loaded successfully")
                
            else:
                print(f"Loading custom model: {model_id}")
                self.processor = WhisperProcessor.from_pretrained(
                    model_id,
                    cache_dir=self.cache_dir,
                    language=self.config["language"],
                    task="transcribe"
                )
                
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model_id,
                    cache_dir=self.cache_dir
                ).to(self.device)
            
            return True
        except Exception as e:
            print(f"Error loading model {model_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    async def transcribe(self, file_path: str):
        try:
            if self.config["type"] == "pipeline":
                return await self._transcribe_pipeline(file_path)
            elif self.config["type"] == "malaysian":
                return await self._transcribe_malaysian(file_path)
            else:
                return await self._transcribe_thai(file_path)
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return None

    async def _transcribe_pipeline(self, file_path: str):
        try:
            print(f"Transcribing with pipeline model...")
            if self.pipeline is None:
                raise RuntimeError("Pipeline not initialized")
                
            # Use direct file path for pipeline
            result = await asyncio.to_thread(
                self.pipeline, 
                file_path,
                batch_size=8,
                return_timestamps=False
            )
            
            print(f"Pipeline transcription result: {result}")
            return result["text"] if isinstance(result, dict) else result
        except Exception as e:
            print(f"Error in pipeline transcription: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    async def _transcribe_malaysian(self, file_path: str):
        try:
            audio, sr = await asyncio.to_thread(
                librosa.load, 
                file_path, 
                sr=16000,
                mono=True
            )
            
            print(f"Loaded audio: {len(audio)} samples, {sr}Hz")

            # Process audio
            with torch.no_grad():
                inputs = self.processor(
                    audio, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                )
                
                # Move input features to CPU
                input_features = inputs["input_features"].to(self.device)
                
                # Generate transcription
                generated = await asyncio.to_thread(
                    self.model.generate,
                    input_features,
                    language="ms",
                    task="transcribe"
                )
                
                # Decode result
                transcription = self.processor.batch_decode(
                    generated, 
                    skip_special_tokens=True
                )[0]
                
                print(f"Malaysian model transcription: {transcription}")
                return transcription

        except Exception as e:
            print(f"Error in Malaysian transcription: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    async def _transcribe_thai(self, file_path: str):
        try:
            print("Starting Thai transcription...")
            audio, sr = await asyncio.to_thread(librosa.load, file_path, sr=16000)
            inputs = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features
            
            generated = await asyncio.to_thread(
                self.model.generate,
                input_features=inputs.to(self.device),
                max_new_tokens=255,
                language="th",  # Explicitly set Thai language
                task="transcribe"
            )
            
            # Get transcription and ensure proper encoding
            transcription = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
            print(f"Thai transcription (raw): {transcription.encode('utf-8')}")
            
            return transcription

        except Exception as e:
            print(f"Error in Thai transcription: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# Initialize model handlers
model_handlers = {}

async def initialize_models():
    """Initialize all models asynchronously"""
    print("Initializing models...")
    
    # Initialize base Whisper model first
    print("Loading base Whisper model...")
    global base_model
    try:
        base_model = whisper.load_model(
            "base", 
            device="cpu",  # Always use CPU
            download_root=str(PROJECT_ROOT / "models" / "whisper")
        )
        print("Base model loaded successfully")
    except Exception as e:
        print(f"Error loading base model: {str(e)}")
        raise RuntimeError("Failed to load base Whisper model")

    # Initialize fine-tuned models
    for country, config in COUNTRY_MODELS.items():
        model_specific_cache = MODEL_CACHE_DIR / config["model_id"].replace('/', '_')
        handler = ModelHandler(config, model_specific_cache)
        await handler.load()
        model_handlers[country] = handler

    print("Model initialization complete")

@app.on_event("startup")
async def startup_event():
    """Initialize models when the FastAPI app starts"""
    await initialize_models()

async def transcribe_with_base_model(file_path: str):
    """Transcribe audio using the base Whisper model"""
    try:
        print("Starting base model transcription...")
        result = base_model.transcribe(file_path)
        print(f"Base model transcription complete: {result['text']}")
        return result["text"]
    except Exception as e:
        print(f"Error in base model transcription: {str(e)}")
        return "Base model transcription failed"

@app.post("/transcribe/")
async def transcribe_audio(
    file: UploadFile = File(...),
    country: str = Form(None)
):
    try:
        print("\n=== Received Request ===")
        print(f"Country: '{country}'")
        start_time = time.time()

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
            print(f"File saved at: {temp_path}")

        # First denoise the audio
        print("Starting denoising process...")
        denoiser = AudioDenoiser(sample_rate=48000)
        denoised_result = await denoiser.process_audio(temp_path)
        denoised_path = denoised_result["output_path"]
        
        print(f"Audio denoised. Starting transcription...")
        print(f"Noise reduction metrics: {denoised_result['metrics']}")

        # Create tasks for both models using denoised audio
        print("Starting parallel transcription...")
        base_task = asyncio.create_task(transcribe_with_base_model(denoised_path))
        fine_tuned_task = asyncio.create_task(
            transcribe_with_fine_tuned_model(denoised_path, country)
        ) if country in COUNTRY_MODELS else None

        # Run both tasks in parallel
        if fine_tuned_task:
            base_result, fine_tuned_result = await asyncio.gather(base_task, fine_tuned_task)
        else:
            base_result = await base_task
            fine_tuned_result = None

        # Clean up temporary files
        for path in [temp_path, denoised_path]:
            if path and os.path.exists(path):
                os.unlink(path)
                print(f"Removed temporary file: {path}")

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        response_data = {
            "base_model": {
                "text": base_result,
                "model": "whisper-base"
            },
            "fine_tuned_model": {
                "text": fine_tuned_result,
                "model_name": COUNTRY_MODELS[country]["name"],
                "model_id": COUNTRY_MODELS[country]["model_id"],
                "language": COUNTRY_MODELS[country]["language"]
            } if fine_tuned_result and country in COUNTRY_MODELS else None,
            "country": country,
            "processing_time": f"{elapsed_time:.2f} seconds",
            "noise_reduction_metrics": denoised_result["metrics"]
        }
        
        print("\n=== Response Data ===")
        print(f"Base Model Result: {base_result}")
        print(f"Fine-tuned Model Result: {fine_tuned_result}")
        print(f"Country: {country}")
        print(f"Total processing time: {elapsed_time:.2f} seconds")
        
        return JSONResponse(
            content=jsonable_encoder(response_data),
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

    except Exception as e:
        print(f"\nError in transcribe_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up in case of error
        if 'temp_path' in locals() and temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        if 'denoised_path' in locals() and denoised_path and os.path.exists(denoised_path):
            os.unlink(denoised_path)
            
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

async def transcribe_with_fine_tuned_model(file_path: str, country: str):
    try:
        if country not in model_handlers:
            print(f"No model handler found for country: {country}")
            return None
        
        handler = model_handlers[country]
        if handler is None:
            print(f"Model handler is None for country: {country}")
            return None
            
        result = await handler.transcribe(file_path)
        if result is None:
            print(f"Transcription failed for {country}")
            return None
            
        return result
        
    except Exception as e:
        print(f"Error in fine-tuned transcription: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

class AudioDenoiser:
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        print(f"Noise reduction initialized with {sample_rate}Hz sample rate")

    async def process_audio(self, file_path: str, output_format: str = 'wav') -> dict:
        try:
            print("\n=== Starting Audio Processing ===")
            print(f"Input file: {file_path}")

            # Create new temporary WAV file path
            temp_dir = os.path.dirname(file_path)
            wav_path = os.path.join(temp_dir, f'temp_{os.path.basename(file_path)}_{int(time.time())}.wav')
            print("Converting to WAV...")
            
            result = subprocess.run([
                'ffmpeg',
                '-i', file_path,        # Input file
                '-acodec', 'pcm_s16le', # Convert to PCM 16-bit
                '-ar', str(self.sample_rate),  # Set sample rate
                '-ac', '1',             # Convert to mono
                wav_path               # Output to new temporary file
            ], capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"FFmpeg conversion failed: {result.stderr}")

            # Load and validate converted WAV
            audio, sample_rate = sf.read(wav_path, dtype='float32')
            
            # Check if audio has any content
            if len(audio) == 0:
                raise ValueError("Empty audio file")
                
            # Check if audio has any non-zero values
            if np.all(audio == 0):
                raise ValueError("Audio file contains only zeros")
                
            print(f"Raw audio shape: {audio.shape}")
            print(f"Raw audio range: {audio.min():.4f} to {audio.max():.4f}")
            
            # Ensure mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
                print("Converted stereo to mono")

            # Apply noise reduction
            print("Applying noise reduction...")
            denoised_audio = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                stationary=True,
                prop_decrease=0.75,
                freq_mask_smooth_hz=100,
                n_jobs=-1
            )
            
            # Calculate metrics
            original_rms = np.sqrt(np.mean(audio**2))
            denoised_rms = np.sqrt(np.mean(denoised_audio**2))
            noise_reduction = original_rms - denoised_rms
            
            # Save output
            output_path = wav_path.replace('.wav', '_denoised.wav')
            sf.write(output_path, denoised_audio, sample_rate)
            
            print("\n=== Processing Complete ===")
            print(f"Saved to: {output_path}")
            print(f"Original RMS: {original_rms:.4f}")
            print(f"Denoised RMS: {denoised_rms:.4f}")
            print(f"Noise Reduction: {noise_reduction:.4f}")

            # Clean up intermediate WAV file
            if os.path.exists(wav_path):
                os.unlink(wav_path)
            
            return {
                "output_path": output_path,
                "metrics": {
                    "original_rms": float(original_rms),
                    "denoised_rms": float(denoised_rms),
                    "noise_reduction": float(noise_reduction)
                }
            }

        except Exception as e:
            # Clean up any temporary files
            for path in [wav_path]:
                if 'wav_path' in locals() and os.path.exists(path):
                    os.unlink(path)
            
            print(f"Error in audio processing: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

@app.post("/denoise/")
async def denoise_audio(file: UploadFile = File(...)):
    """
    Endpoint to denoise audio using noisereduce.
    Accepts M4A files, converts to WAV, and returns denoised WAV.
    """
    temp_path = None
    wav_path = None
    denoised_path = None
    
    try:
        print("\n=== Received Audio File ===")
        print(f"Filename: {file.filename}")
        print(f"Content type: {file.content_type}")

        # Create temp directory to store files
        temp_dir = tempfile.mkdtemp()
        print(f"Created temp directory: {temp_dir}")

        # Save original M4A file
        temp_path = os.path.join(temp_dir, 'input.m4a')
        content = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(content)
        print(f"Saved original file: {temp_path}")

        # Convert M4A to WAV
        wav_path = os.path.join(temp_dir, 'converted.wav')
        print("Converting to WAV...")
        
        result = subprocess.run([
            'ffmpeg',
            '-i', temp_path,
            '-acodec', 'pcm_s16le',
            '-ar', '48000',
            '-ac', '1',
            '-y',
            wav_path
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"FFmpeg conversion failed: {result.stderr}")

        # Process WAV file
        denoiser = AudioDenoiser(sample_rate=48000)
        result = await denoiser.process_audio(wav_path)
        denoised_path = result["output_path"]
        
        if not os.path.exists(denoised_path):
            raise Exception("Denoising failed to create output file")

        # Create response
        response = FileResponse(
            denoised_path,
            media_type='audio/wav',
            headers={
                "X-Original-RMS": str(result["metrics"]["original_rms"]),
                "X-Denoised-RMS": str(result["metrics"]["denoised_rms"]),
                "X-Noise-Reduction": str(result["metrics"]["noise_reduction"])
            }
        )

        # Use BackgroundTask for cleanup
        def cleanup():
            try:
                for path in [temp_path, wav_path, denoised_path]:
                    if path and os.path.exists(path):
                        os.unlink(path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
                print("Cleanup completed")
            except Exception as e:
                print(f"Cleanup error: {e}")

        response.background = BackgroundTask(cleanup)
        return response

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        try:
            for path in [temp_path, wav_path, denoised_path]:
                if path and os.path.exists(path):
                    os.unlink(path)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as cleanup_error:
            print(f"Cleanup error: {cleanup_error}")
            
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
