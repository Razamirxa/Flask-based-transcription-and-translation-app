import os
import tempfile
from pathlib import Path
import ffmpeg
from pydub import AudioSegment
import sounddevice as sd
import numpy as np
import wave
import time
from datetime import datetime, timedelta
from langdetect import detect
import queue
import threading
import io
import json
import google.auth
import google.auth.transport.requests
import requests
from googleapiclient.discovery import build
# Import OpenAI instead of Google speech
import openai
from dotenv import load_dotenv
# Fix the import for Google Cloud Translation API
try:
    from google.cloud import translate_v2 as translate
except ImportError:
    # Fallback to direct import
    try:
        from google.cloud import translate
    except ImportError:
        translate = None
import yt_dlp
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI with API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session and flash messages
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max upload size (increased from 100MB)
app.config['DEMO_MODE'] = False  # Set to True to run in demo mode without OpenAI API key

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Check for OpenAI API key
if not openai.api_key:
    print("Warning: OpenAI API key not found. Set OPENAI_API_KEY in your .env file.")
    app.config['DEMO_MODE'] = True
    print("Running in DEMO MODE - transcription functionality will be simulated")
else:
    print("OpenAI API key found and configured")

# Supported languages
LANGUAGES = {
    'af': 'Afrikaans', 'ar': 'Arabic', 'bg': 'Bulgarian', 'bn': 'Bengali',
    'ca': 'Catalan', 'cs': 'Czech', 'da': 'Danish', 'de': 'German',
    'el': 'Greek', 'en': 'English', 'es': 'Spanish', 'et': 'Estonian',
    'fa': 'Persian', 'fi': 'Finnish', 'fr': 'French', 'gu': 'Gujarati',
    'hi': 'Hindi', 'hr': 'Croatian', 'hu': 'Hungarian', 'id': 'Indonesian',
    'is': 'Icelandic', 'it': 'Italian', 'iw': 'Hebrew', 'ja': 'Japanese',
    'kn': 'Kannada', 'ko': 'Korean', 'lt': 'Lithuanian', 'lv': 'Latvian',
    'ml': 'Malayalam', 'mr': 'Marathi', 'ms': 'Malay', 'nl': 'Dutch',
    'no': 'Norwegian', 'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian',
    'ru': 'Russian', 'sk': 'Slovak', 'sl': 'Slovenian', 'sr': 'Serbian',
    'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu',
    'th': 'Thai', 'tl': 'Filipino', 'tr': 'Turkish', 'uk': 'Ukrainian',
    'ur': 'Urdu', 'vi': 'Vietnamese', 'zh': 'Chinese'
}

# Language codes for Google Cloud Speech-to-Text API
SPEECH_TO_TEXT_LANGUAGE_CODES = {
    'af': 'af-ZA',
    'ar': 'ar-SA',
    'bg': 'bg-BG',
    'bn': 'bn-IN',
    'ca': 'ca-ES',
    'cs': 'cs-CZ',
    'da': 'da-DK',
    'de': 'de-DE',
    'el': 'el-GR',
    'en': 'en-US',
    'es': 'es-ES',
    'et': 'et-EE',
    'fa': 'fa-IR',
    'fi': 'fi-FI',
    'fr': 'fr-FR',
    'gu': 'gu-IN',
    'hi': 'hi-IN',
    'hr': 'hr-HR',
    'hu': 'hu-HU',
    'id': 'id-ID',
    'is': 'is-IS',
    'it': 'it-IT',
    'iw': 'iw-IL',
    'ja': 'ja-JP',
    'kn': 'kn-IN',
    'ko': 'ko-KR',
    'lt': 'lt-LT',
    'lv': 'lv-LV',
    'ml': 'ml-IN',
    'mr': 'mr-IN',
    'ms': 'ms-MY',
    'nl': 'nl-NL',
    'no': 'no-NO',
    'pl': 'pl-PL',
    'pt': 'pt-PT',
    'ro': 'ro-RO',
    'ru': 'ru-RU',
    'sk': 'sk-SK',
    'sl': 'sl-SI',
    'sr': 'sr-RS',
    'sv': 'sv-SE',
    'sw': 'sw-KE',
    'ta': 'ta-IN',
    'te': 'te-IN',
    'th': 'th-TH',
    'tl': 'tl-PH',
    'tr': 'tr-TR',
    'uk': 'uk-UA',
    'ur': 'ur-PK',
    'vi': 'vi-VN',
    'zh': 'zh-CN'
}

# Urdu language for priority selection
URDU_LANGUAGE = "ur - Urdu"

class AudioRecorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_data = []
        self.overflow_count = 0
        self.max_overflow = 5

    def callback(self, indata, frames, time, status):
        if status:
            if status.input_overflow:
                self.overflow_count += 1
                if self.overflow_count > self.max_overflow:
                    self.is_recording = False
                    return
            else:
                print(f"Status: {status}")
        
        if self.is_recording:  # Only add data if still recording
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        self.overflow_count = 0
        
        def record():
            try:
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=self.callback,
                    blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                    dtype=np.float32
                ):
                    while self.is_recording:
                        if not self.audio_queue.empty():
                            data = self.audio_queue.get()
                            # Convert float32 to int16
                            data = (data * 32767).astype(np.int16)
                            self.audio_data.append(data)
                        time.sleep(0.001)  # Small sleep to prevent CPU overuse
            except Exception as e:
                print(f"Recording error: {e}")
                self.is_recording = False
        
        self.recording_thread = threading.Thread(target=record)
        self.recording_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()
        
        if not self.audio_data:
            return np.array([], dtype=np.int16)
        
        return np.concatenate(self.audio_data)

def save_audio_to_wav(audio_data, sample_rate):
    """Save recorded audio to WAV file."""
    if len(audio_data) == 0:
        print("No audio data recorded!")
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(app.config['UPLOAD_FOLDER'], f"recording_{timestamp}.wav")
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    return filename

def convert_video_to_audio(video_path):
    """Extract audio from video file using ffmpeg."""
    audio_path = str(video_path).rsplit(".", 1)[0] + ".wav"
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16k')
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        return audio_path
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise

def convert_audio_to_wav(audio_path):
    """Convert audio file to WAV format."""
    audio = AudioSegment.from_file(audio_path)
    wav_path = str(audio_path).rsplit(".", 1)[0] + ".wav"
    audio.export(wav_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
    return wav_path

def translate_text(text, target_language="en"):
    """Translate text using OpenAI for better accuracy with robust token limit handling."""
    # If in demo mode, return dummy translation
    if app.config['DEMO_MODE']:
        return "This is a sample translation in demo mode. The actual translation would appear here when using OpenAI with valid credentials."
    
    try:
        # Estimate tokens: about 1 token per 4 characters for English (conservative estimate)
        estimated_tokens = len(text) // 3  # Even more conservative to be safe
        
        # Much smaller chunk size - targeting ~1500 tokens per chunk (max 8000 for context)
        # This is roughly 4500 characters per chunk
        MAX_CHUNK_CHARS = 1500
        
        # Always chunk the text regardless of size to ensure we don't hit limits
        print(f"Breaking text into smaller chunks. Text length: {len(text)} characters, estimated tokens: {estimated_tokens}")
        
        # Split into sentences for natural boundaries
        import re
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        
        chunks = []
        current_chunk = ""
        
        # Build chunks of smaller size
        for sentence in sentences:
            # If the sentence itself is too long, split it further by punctuation or spaces
            if len(sentence) > MAX_CHUNK_CHARS:
                # Try to split by commas or other mid-sentence punctuation
                sub_parts = re.split(r'(?<=,|;|:|\))\s', sentence)
                
                for part in sub_parts:
                    # If part is still too long, split by spaces every MAX_CHUNK_CHARS/2 characters
                    if len(part) > MAX_CHUNK_CHARS:
                        words = part.split()
                        sub_chunk = ""
                        for word in words:
                            if len(sub_chunk) + len(word) < MAX_CHUNK_CHARS - 100:  # Leave some buffer
                                sub_chunk += word + " "
                            else:
                                if sub_chunk:
                                    if current_chunk and len(current_chunk) + len(sub_chunk) < MAX_CHUNK_CHARS:
                                        current_chunk += sub_chunk
                                    else:
                                        if current_chunk:
                                            chunks.append(current_chunk.strip())
                                        current_chunk = sub_chunk
                                sub_chunk = word + " "
                        
                        if sub_chunk and len(current_chunk) + len(sub_chunk) < MAX_CHUNK_CHARS:
                            current_chunk += sub_chunk
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sub_chunk
                    else:
                        # Add the part if it fits in current chunk
                        if len(current_chunk) + len(part) < MAX_CHUNK_CHARS:
                            current_chunk += part + " "
                        else:
                            chunks.append(current_chunk.strip())
                            current_chunk = part + " "
            else:
                # Add the sentence if it fits in current chunk
                if len(current_chunk) + len(sentence) < MAX_CHUNK_CHARS:
                    current_chunk += sentence + " "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        print(f"Split text into {len(chunks)} chunks for translation (max {MAX_CHUNK_CHARS} chars per chunk).")
        
        # Translate each chunk
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            
            # Add retry logic for robustness
            max_retries = 3
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"You are a translator. Translate the following text to {target_language}. Provide only the translation with no additional text."},
                            {"role": "user", "content": chunk}
                        ],
                        temperature=0.3,
                        max_tokens=2048  # Reduced max tokens for safety
                    )
                    
                    translated_chunks.append(response.choices[0].message.content.strip())
                    success = True
                    
                except Exception as chunk_error:
                    retry_count += 1
                    print(f"Error translating chunk {i+1}, attempt {retry_count}: {str(chunk_error)}")
                    
                    # If we hit token limits, try splitting the chunk further
                    if "context_length" in str(chunk_error) and retry_count < max_retries:
                        # Split the chunk in half and try each half separately
                        half_point = len(chunk) // 2
                        # Try to find a sentence boundary near the middle
                        split_point = chunk.find('. ', half_point - 100, half_point + 100)
                        if split_point == -1:  # No sentence boundary found, try comma
                            split_point = chunk.find(', ', half_point - 100, half_point + 100)
                        if split_point == -1:  # No punctuation found, split at space
                            split_point = chunk.find(' ', half_point)
                        if split_point == -1:  # Last resort, split exactly in the middle
                            split_point = half_point
                            
                        chunk1 = chunk[:split_point].strip()
                        chunk2 = chunk[split_point:].strip()
                        
                        print(f"Splitting chunk {i+1} into two smaller chunks ({len(chunk1)} and {len(chunk2)} chars)")
                        
                        # Process first half
                        try:
                            response1 = openai.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": f"You are a translator. Translate the following text to {target_language}. Provide only the translation with no additional text."},
                                    {"role": "user", "content": chunk1}
                                ],
                                temperature=0.3,
                                max_tokens=2048
                            )
                            part1 = response1.choices[0].message.content.strip()
                            
                            # Process second half
                            response2 = openai.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": f"You are a translator. Translate the following text to {target_language}. Provide only the translation with no additional text."},
                                    {"role": "user", "content": chunk2}
                                ],
                                temperature=0.3,
                                max_tokens=2048
                            )
                            part2 = response2.choices[0].message.content.strip()
                            
                            # Combine the results
                            translated_chunks.append(part1 + " " + part2)
                            success = True
                            
                        except Exception as split_error:
                            print(f"Error in emergency splitting of chunk {i+1}: {str(split_error)}")
                    
                    # Sleep briefly before retrying
                    import time
                    time.sleep(2)
            
            if not success:
                print(f"Failed to translate chunk {i+1} after {max_retries} attempts. Using placeholder.")
                # Add placeholder for failed chunk
                translated_chunks.append(f"[Translation failed for this section: {chunk[:50]}...]")
        
        # Combine the translated chunks
        return " ".join(translated_chunks)
        
    except Exception as e:
        print(f"Error translating text: {str(e)}")
        return text  # Return original text if translation fails

def transcribe_audio(audio_path, language_code="en-US", translate_to_english=True, progress_callback=None):
    """
    Transcribe audio and optionally translate to English.
    
    Args:
        audio_path: Path to the audio file
        language_code: Language code for transcription
        translate_to_english: Whether to translate to English
        progress_callback: Function to call with progress updates
          This should accept (message, progress_value) parameters
    """
    # If in demo mode, return dummy data
    if app.config['DEMO_MODE']:
        # Simulate processing time with more gradual steps
        if progress_callback:
            progress_callback("Demo mode: Initializing transcription service...", 0.05)
            time.sleep(0.8)
            progress_callback("Demo mode: Loading audio file...", 0.1)
            time.sleep(0.8)
            progress_callback("Demo mode: Analyzing audio format...", 0.15)
            time.sleep(0.8)
            progress_callback("Demo mode: Preparing audio for transcription...", 0.2)
            time.sleep(0.8)
            progress_callback("Demo mode: Beginning language recognition...", 0.25)
            time.sleep(0.8)
            progress_callback("Demo mode: Processing first audio segment...", 0.35)
            time.sleep(0.8)
            progress_callback("Demo mode: Processing second audio segment...", 0.45)
            time.sleep(0.8)
            progress_callback("Demo mode: Processing third audio segment...", 0.55)
            time.sleep(0.8)
            progress_callback("Demo mode: Processing final audio segment...", 0.65)
            time.sleep(0.8)
            progress_callback("Demo mode: Combining transcription results...", 0.75)
            time.sleep(0.8)
            progress_callback("Demo mode: Finalizing transcription...", 0.85)
            time.sleep(0.8)
            
            # Only show translation message if needed
            if translate_to_english and not language_code.startswith("en"):
                progress_callback("Demo mode: Translating to English...", 0.9)
                time.sleep(0.8)
                progress_callback("Demo mode: Formatting final results...", 0.95)
                time.sleep(0.5)
            else:
                progress_callback("Demo mode: Formatting final results...", 0.95)
                time.sleep(0.5)
        
        # Return dummy transcription with timestamps (for SRT generation)
        dummy_transcripts = {
            "en-US": "This is a sample transcription in demo mode. The actual transcription would appear here when using OpenAI Whisper API with valid credentials.",
            "es-ES": "Esta es una transcripción de muestra en modo demo. La transcripción real aparecería aquí cuando se utiliza la API de OpenAI Whisper con credenciales válidas.",
            "fr-FR": "Ceci est un exemple de transcription en mode démo. La transcription réelle apparaîtrait ici lors de l'utilisation de l'API OpenAI Whisper avec des informations d'identification valides.",
            "de-DE": "Dies ist eine Beispieltranskription im Demo-Modus. Die tatsächliche Transkription würde hier erscheinen, wenn Sie die OpenAI Whisper-API mit gültigen Anmeldeinformationen verwenden.",
            "ur-PK": "یہ ڈیمو موڈ میں ایک نمونہ ٹرانسکرپشن ہے۔ اصل ٹرانسکرپشن یہاں ظاہر ہوگی جب درست اسناد کے ساتھ OpenAI Whisper API استعمال کی جائے گی۔"
        }
        
        # Get the language code prefix (e.g., "en" from "en-US")
        lang_prefix = language_code.split('-')[0].lower()
        
        # Get the appropriate dummy transcript or default to English
        original = dummy_transcripts.get(language_code, dummy_transcripts.get(f"{lang_prefix}-{lang_prefix.upper()}", dummy_transcripts["en-US"]))
        
        # Create dummy word timestamps for SRT generation
        dummy_words_with_time = []
        words = original.split()
        for i, word in enumerate(words):
            # Create a dummy timestamp (1 second per word)
            start_time = i * 1.0
            end_time = (i + 1) * 1.0
            dummy_words_with_time.append({
                "word": word,
                "start_time": start_time,
                "end_time": end_time
            })
        
        # If translating to English and not already in English
        if translate_to_english and not language_code.startswith("en"):
            translated = dummy_transcripts["en-US"]
            
            # Create dummy translated word timestamps
            dummy_translated_words_with_time = []
            translated_words = translated.split()
            for i, word in enumerate(translated_words):
                start_time = i * 1.0
                end_time = (i + 1) * 1.0
                dummy_translated_words_with_time.append({
                    "word": word,
                    "start_time": start_time,
                    "end_time": end_time
                })
            
            return {
                "original_transcript": original,
                "translated_transcript": translated,
                "words_with_time": dummy_words_with_time,
                "translated_words_with_time": dummy_translated_words_with_time
            }
        else:
            return {
                "original_transcript": original,
                "translated_transcript": original,
                "words_with_time": dummy_words_with_time,
                "translated_words_with_time": dummy_words_with_time
            }
    
    # Real transcription using OpenAI Whisper
    try:
        # Initial progress update to indicate we're starting
        if progress_callback:
            progress_callback("Initializing transcription service...", 0.05)
        
        # Make sure we have a valid language code for Speech-to-Text API
        # If we have a simple code like "ur", convert it to the full code like "ur-PK"
        if '-' not in language_code and language_code.lower() in SPEECH_TO_TEXT_LANGUAGE_CODES:
            language_code = SPEECH_TO_TEXT_LANGUAGE_CODES[language_code.lower()]
        elif language_code.lower() not in [code.lower() for code in SPEECH_TO_TEXT_LANGUAGE_CODES.values()]:
            # If language code is not in our predefined list, fall back to English
            print(f"Warning: Language code '{language_code}' not supported. Falling back to 'en-US'.")
            language_code = "en-US"
        
        # Update progress to indicate we're loading the audio
        if progress_callback:
            progress_callback("Loading and preparing audio file...", 0.1)
            time.sleep(0.5) # Add a small delay for better UX
            
        # Load the audio file
        audio_segment = AudioSegment.from_wav(audio_path)
        
        # Calculate total duration in seconds
        duration_seconds = len(audio_segment) / 1000
        
        # Update progress
        if progress_callback:
            progress_callback("Analyzing audio file characteristics...", 0.15)
            time.sleep(0.5) # Add a small delay for better UX
            
        # Split audio into chunks for processing (OpenAI can handle larger files)
        # Whisper supports up to 25MB files, which is roughly 2.5 hours of audio
        chunk_length_ms = 10 * 60 * 1000  # 10 minutes per chunk
        chunks = [audio_segment[i:i + chunk_length_ms] for i in range(0, len(audio_segment), chunk_length_ms)]
        
        if progress_callback:
            progress_callback(f"Audio split into {len(chunks)} chunks for processing.", 0.2)
            time.sleep(0.5) # Add a small delay for better UX
        
        # Process each chunk and combine transcriptions
        transcript = ""
        all_words_with_time = []
        chunk_offset_seconds = 0
        
        # Reserve 20% to 80% of the progress bar for chunk processing
        chunk_progress_start = 0.2
        chunk_progress_end = 0.8
        chunk_progress_range = chunk_progress_end - chunk_progress_start
        
        for i, chunk in enumerate(chunks):
            # Calculate progress for this chunk
            chunk_progress = chunk_progress_start + (chunk_progress_range * (i / len(chunks)))
            
            # Update progress before processing
            if progress_callback:
                progress_callback(f"Preparing chunk {i+1} of {len(chunks)} for transcription...", chunk_progress)
                time.sleep(0.3) # Add a small delay for better UX
            
            # Export chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                chunk_path = temp_file.name
                chunk.export(chunk_path, format="wav")
            
            try:
                # Update progress before API call
                if progress_callback:
                    progress_callback(f"Sending chunk {i+1} to OpenAI Whisper...", chunk_progress + (chunk_progress_range / (2 * len(chunks))))
                    time.sleep(0.3) # Add a small delay for better UX
                
                # Open the audio file
                with open(chunk_path, "rb") as audio_file:
                    # Process with OpenAI Whisper API
                    whisper_language = language_code.split('-')[0] if '-' in language_code else language_code
                    response = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=whisper_language,
                        response_format="verbose_json",
                        timestamp_granularities=["word"]
                    )
                    
                    # Handle the OpenAI Transcription object
                    try:
                        # Access the text and words directly from the response object
                        if hasattr(response, "text"):
                            # It's an object with attributes
                            chunk_transcript = response.text
                            words_data = getattr(response, "words", []) or []
                            print("DEBUG: Accessed properties as object attributes")
                        else:
                            # It's a dictionary
                            chunk_transcript = response.get('text', '')
                            words_data = response.get('words', [])
                            print("DEBUG: Accessed properties as dictionary keys")
                        
                        print(f"DEBUG: Transcript: {chunk_transcript[:30]}...")
                        print(f"DEBUG: Words data type: {type(words_data)}, length: {len(words_data)}")
                    except Exception as e:
                        print(f"DEBUG: Error accessing response properties: {str(e)}")
                        chunk_transcript = ""
                        words_data = []
                    
                    # Process word timestamps
                    try:
                        if words_data:
                            # Process each word
                            for word_info in words_data:
                                try:
                                    # Try to get start/end using both attribute and dictionary access
                                    if hasattr(word_info, "start") and hasattr(word_info, "end"):
                                        start_time = word_info.start + chunk_offset_seconds
                                        end_time = word_info.end + chunk_offset_seconds
                                        word = word_info.word
                                        print(f"DEBUG: Word data accessed as attributes: {word}")
                                    else:
                                        start_time = word_info.get('start', 0) + chunk_offset_seconds
                                        end_time = word_info.get('end', 0) + chunk_offset_seconds
                                        word = word_info.get('word', '')
                                        print(f"DEBUG: Word data accessed as dict: {word}")
                                    
                                    all_words_with_time.append({
                                        "word": word,
                                        "start_time": start_time,
                                        "end_time": end_time
                                    })
                                except Exception as word_e:
                                    print(f"DEBUG: Error processing word: {str(word_e)}")
                                    print(f"DEBUG: Word info type: {type(word_info)}")
                                    # Try printing the word info for debugging
                                    try:
                                        print(f"DEBUG: Word info content: {word_info}")
                                    except:
                                        pass
                        else:
                            # If word timestamps not available, estimate them
                            words = chunk_transcript.split()
                            avg_word_duration = chunk.duration_seconds / max(1, len(words))
                            for idx, word in enumerate(words):
                                start_time = chunk_offset_seconds + (idx * avg_word_duration)
                                end_time = start_time + avg_word_duration
                                all_words_with_time.append({
                                    "word": word,
                                    "start_time": start_time,
                                    "end_time": end_time
                                })
                    except Exception as timestamping_e:
                        print(f"DEBUG: Error in word timestamp processing: {str(timestamping_e)}")
                    
                    # Add to transcript
                    transcript += chunk_transcript + " "
                
                # Update progress after processing
                if progress_callback:
                    chunk_completed_progress = chunk_progress_start + (chunk_progress_range * ((i + 1) / len(chunks)))
                    progress_callback(f"Processed chunk {i+1} of {len(chunks)} ({int((i+1)*100/len(chunks))}% complete)", chunk_completed_progress)
                    time.sleep(0.3) # Add a small delay for better UX
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(chunk_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {chunk_path}: {e}")
            
            # Update the offset for the next chunk
            chunk_offset_seconds += chunk.duration_seconds
        
        transcript = transcript.strip()
        
        # Final processing steps
        if progress_callback:
            progress_callback("Finalizing transcription results...", 0.85)
            time.sleep(0.5) # Add a small delay for better UX
        
        # Check if we need to translate
        if translate_to_english and not language_code.startswith("en"):
            if progress_callback:
                progress_callback("Transcription complete. Starting translation to English...", 0.85)
                time.sleep(0.5) # Add a small delay for better UX
            
            # Check if transcript is long and will need chunking
            transcript_length = len(transcript)
            if transcript_length > 1000:  # Smaller threshold since we now always chunk
                if progress_callback:
                    progress_callback(f"Transcript is {transcript_length} characters. Using multi-part translation...", 0.86)
                
                # We're using smaller chunks now - estimate chunks for progress updates
                estimated_chunks = transcript_length // 1500 + 1
                if progress_callback:
                    progress_callback(f"Translation will use approximately {estimated_chunks} smaller chunks...", 0.87)
            
            # translate_text function will handle progress updates
            translated_transcript = translate_text(transcript)
            
            if progress_callback:
                progress_callback("Translation complete. Generating timestamps for translated text...", 0.9)
                time.sleep(0.5) # Add a small delay for better UX
            
            # For translated transcript, we need to estimate timestamps
            # Since we don't get word timestamps with translation,
            # we'll create estimated timestamps based on the original timing
            
            translated_words = translated_transcript.split()
            original_duration = all_words_with_time[-1]["end_time"] if all_words_with_time else 0
            
            # Create estimated timestamps for translated words
            translated_words_with_time = []
            for i, word in enumerate(translated_words):
                # Distribute words evenly across the same duration as the original
                ratio = i / len(translated_words)
                next_ratio = (i + 1) / len(translated_words)
                
                start_time = ratio * original_duration
                end_time = next_ratio * original_duration
                
                translated_words_with_time.append({
                    "word": word,
                    "start_time": start_time,
                    "end_time": end_time
                })
            
            # Final progress update
            if progress_callback:
                progress_callback("Translation complete. Preparing final results...", 0.95)
                time.sleep(0.5) # Add a small delay for better UX
                
            return {
                "original_transcript": transcript,
                "translated_transcript": translated_transcript,
                "words_with_time": all_words_with_time,
                "translated_words_with_time": translated_words_with_time
            }
        else:
            # Final progress update
            if progress_callback:
                progress_callback("Preparing final results...", 0.95)
                time.sleep(0.5) # Add a small delay for better UX
                
            return {
                "original_transcript": transcript, 
                "translated_transcript": transcript,
                "words_with_time": all_words_with_time,
                "translated_words_with_time": all_words_with_time
            }
        
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        raise e

def format_srt_timestamp(seconds):
    """
    Format seconds as SRT timestamp (HH:MM:SS,mmm)
    """
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def download_audio_from_youtube(youtube_url):
    """Download audio from a YouTube video"""
    # Create a subdirectory for YouTube downloads
    youtube_download_path = os.path.join(app.config['UPLOAD_FOLDER'], 'youtube_downloads')
    os.makedirs(youtube_download_path, exist_ok=True)
    
    # Generate a unique filename based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"youtube_{timestamp}.wav"
    output_path = os.path.join(youtube_download_path, filename)
    
    print(f"Downloading audio from YouTube URL: {youtube_url}")
    print(f"Target output path: {output_path}")
    
    # Important: Use prefix without .wav, let yt-dlp add the extension
    output_prefix = os.path.join(youtube_download_path, f"youtube_{timestamp}")
    
    # Simple yt-dlp options focused on reliability
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': output_prefix,  # No extension here
        'noplaylist': True,
        'quiet': False,
        'no_warnings': False,
        # Add advanced options to bypass YouTube restrictions
        'nocheckcertificate': True,
        'geo_bypass': True,
        'geo_bypass_country': 'US',
        'extractor_retries': 5,
        'skip_download_archive': True,  # Don't use the download archive
        'force_generic_extractor': True,  # Try generic extractor as fallback
        'allow_unplayable_formats': True,  # Try to download even if player says unplayable
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.youtube.com/',
        }
    }
    
    try:
        # First attempt with direct output path
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            print(f"Download completed. Video title: {info.get('title', 'Unknown')}")
        
        # Check if the file exists
        if os.path.exists(output_path):
            print(f"Found downloaded file at: {output_path}")
            
            # Check the file size
            file_size = os.path.getsize(output_path)
            print(f"File size: {file_size/1024/1024:.2f} MB")
            
            # If the file is too large for OpenAI API (25MB limit), optimize it
            max_openai_size = 25 * 1024 * 1024
            if file_size > max_openai_size:
                print(f"File too large for direct API processing ({file_size/1024/1024:.1f}MB). Optimizing...")
                
                # Load the audio using pydub
                audio = AudioSegment.from_wav(output_path)
                
                # Ensure mono and 16kHz sample rate to reduce size
                audio = audio.set_channels(1).set_frame_rate(16000)
                
                # Export the optimized version
                optimized_path = output_path.replace(".wav", "_optimized.wav")
                audio.export(optimized_path, format="wav")
                
                # If still too large, we'll keep the original and let the transcription function handle chunking
                if os.path.getsize(optimized_path) <= max_openai_size:
                    print(f"Successfully optimized to {os.path.getsize(optimized_path)/1024/1024:.1f}MB")
                    return optimized_path
                else:
                    print("File still large after optimization, but transcription function will handle chunking")
            
            return output_path
            
        # If file not found at exact path, look in the directory
        print(f"File not found at {output_path}. Looking in directory...")
        
        # List all files in the directory
        files = os.listdir(youtube_download_path)
        print(f"Files in directory: {files}")
        
        # Look for any WAV file created in the last few minutes
        recent_files = []
        current_time = time.time()
        # Look for files containing the timestamp or created recently
        for file in files:
            file_path = os.path.join(youtube_download_path, file)
            if timestamp in file and file.endswith('.wav'):
                print(f"Found matching file by timestamp: {file_path}")
                
                # Check size and optimize if needed
                file_size = os.path.getsize(file_path)
                max_openai_size = 25 * 1024 * 1024
                if file_size > max_openai_size:
                    print(f"File too large ({file_size/1024/1024:.1f}MB). Optimizing...")
                    
                    # Load the audio using pydub
                    audio = AudioSegment.from_wav(file_path)
                    
                    # Ensure mono and 16kHz sample rate to reduce size
                    audio = audio.set_channels(1).set_frame_rate(16000)
                    
                    # Export the optimized version
                    optimized_path = file_path.replace(".wav", "_optimized.wav")
                    audio.export(optimized_path, format="wav")
                    
                    if os.path.getsize(optimized_path) <= max_openai_size:
                        return optimized_path
                
                return file_path
            
            # Check creation time (files created in the last 5 minutes)
            file_time = os.path.getctime(file_path)
            if current_time - file_time < 300 and file.endswith(('.wav', '.mp3', '.m4a')):
                recent_files.append(file_path)
        
        if recent_files:
            # Use the most recently created file
            newest_file = max(recent_files, key=os.path.getctime)
            print(f"Using most recent file: {newest_file}")
            
            # If it's not a WAV file, convert it
            if not newest_file.endswith('.wav'):
                wav_path = newest_file.rsplit('.', 1)[0] + '.wav'
                audio = AudioSegment.from_file(newest_file)
                # Optimize while converting
                audio = audio.set_channels(1).set_frame_rate(16000)
                audio.export(wav_path, format="wav")
                print(f"Converted to WAV: {wav_path}")
                return wav_path
            
            # Check size and optimize if needed
            file_size = os.path.getsize(newest_file)
            max_openai_size = 25 * 1024 * 1024
            if file_size > max_openai_size:
                print(f"File too large ({file_size/1024/1024:.1f}MB). Optimizing...")
                
                # Load the audio using pydub
                audio = AudioSegment.from_wav(newest_file)
                
                # Ensure mono and 16kHz sample rate to reduce size
                audio = audio.set_channels(1).set_frame_rate(16000)
                
                # Export the optimized version
                optimized_path = newest_file.replace(".wav", "_optimized.wav")
                audio.export(optimized_path, format="wav")
                
                if os.path.getsize(optimized_path) <= max_openai_size:
                    return optimized_path
            
            return newest_file
            
        # If we still can't find the file, try a fallback approach with default naming
        print("File not found. Trying fallback download approach...")
        fallback_path = os.path.join(youtube_download_path, "%(title)s-%(id)s.%(ext)s")
        
        fallback_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': fallback_path,
            'noplaylist': True,
            # Add advanced options to bypass YouTube restrictions
            'nocheckcertificate': True,
            'geo_bypass': True,
            'geo_bypass_country': 'US',
            'extractor_retries': 5,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.youtube.com/',
            }
        }
        
        with yt_dlp.YoutubeDL(fallback_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_id = info.get('id', '')
            video_title = info.get('title', 'unknown')
            
            # Create a sanitized file name
            sanitized_title = ''.join(c for c in video_title if c.isalnum() or c in ' -_')
            sanitized_title = sanitized_title.replace(' ', '_')
            possible_filename = f"{sanitized_title}-{video_id}.wav"
            possible_path = os.path.join(youtube_download_path, possible_filename)
            
            print(f"Looking for fallback file at: {possible_path}")
            
            # Check if the file exists
            if os.path.exists(possible_path):
                print(f"Found fallback file at: {possible_path}")
                return possible_path
            
            # Final directory scan for any WAV file
            files = os.listdir(youtube_download_path)
            for file in files:
                if file.endswith('.wav') and (video_id in file or timestamp in file):
                    file_path = os.path.join(youtube_download_path, file)
                    print(f"Found matching file: {file_path}")
                    return file_path
        
        # Last resort: Try a direct command-line approach
        print("Attempting final fallback with direct command...")
        direct_output = os.path.join(youtube_download_path, f"direct_{timestamp}.wav")
        try:
            import subprocess
            cmd = [
                "yt-dlp", 
                "-x", 
                "--audio-format", "wav", 
                "-o", direct_output,
                "--no-check-certificate",
                "--geo-bypass",
                "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "--referer", "https://www.youtube.com/",
                "--extractor-retries", "5",
                "--force-generic-extractor",  # Try to use a generic extractor as a last resort
                youtube_url
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Check if the file was created
            if os.path.exists(direct_output):
                print(f"Direct command successful! File created at: {direct_output}")
                return direct_output
                
            # Check directory one last time
            files = os.listdir(youtube_download_path)
            new_files = [os.path.join(youtube_download_path, f) for f in files 
                        if 'direct' in f and timestamp in f and f.endswith(('.wav', '.mp3', '.m4a'))]
            
            if new_files:
                newest = max(new_files, key=os.path.getctime)
                print(f"Found file from direct command: {newest}")
                
                # Convert if not WAV
                if not newest.endswith('.wav'):
                    wav_path = newest.rsplit('.', 1)[0] + '.wav'
                    audio = AudioSegment.from_file(newest)
                    audio.export(wav_path, format="wav")
                    print(f"Converted to WAV: {wav_path}")
                    return wav_path
                return newest
        except Exception as cmd_err:
            print(f"Direct command failed: {cmd_err}")
        
        # Final attempt: Try using youtube-dl format instead
        try:
            print("Trying youtube-dl format as final attempt...")
            youtube_id = youtube_url.split("v=")[-1].split("&")[0] if "v=" in youtube_url else youtube_url.split("/")[-1]
            
            # Try using a mobile embed URL which might be less restricted
            mobile_url = f"https://m.youtube.com/watch?v={youtube_id}"
            embed_url = f"https://www.youtube.com/embed/{youtube_id}"
            
            final_output = os.path.join(youtube_download_path, f"final_{timestamp}.wav")
            
            # Try with mobile URL
            cmd_mobile = [
                "yt-dlp", 
                "-f", "bestaudio", 
                "--extract-audio",
                "--audio-format", "wav",
                "-o", final_output,
                "--referer", "https://m.youtube.com/",
                "--user-agent", "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.43 Mobile Safari/537.36",
                mobile_url
            ]
            
            print("Trying mobile URL approach...")
            subprocess.run(cmd_mobile, check=True, capture_output=True)
            
            if os.path.exists(final_output):
                print(f"Mobile URL approach successful! File created at: {final_output}")
                return final_output
                
            # Try with embed URL if mobile URL failed
            cmd_embed = [
                "yt-dlp", 
                "-f", "bestaudio", 
                "--extract-audio",
                "--audio-format", "wav",
                "-o", final_output,
                "--referer", "https://www.youtube.com/",
                embed_url
            ]
            
            print("Trying embed URL approach...")
            subprocess.run(cmd_embed, check=True, capture_output=True)
            
            if os.path.exists(final_output):
                print(f"Embed URL approach successful! File created at: {final_output}")
                return final_output
                
            # Final directory check
            files = os.listdir(youtube_download_path)
            final_files = [os.path.join(youtube_download_path, f) for f in files 
                          if ('final' in f or youtube_id in f) and f.endswith(('.wav', '.mp3', '.m4a'))]
            
            if final_files:
                newest = max(final_files, key=os.path.getctime)
                if not newest.endswith('.wav'):
                    wav_path = newest.rsplit('.', 1)[0] + '.wav'
                    audio = AudioSegment.from_file(newest)
                    audio.export(wav_path, format="wav")
                    return wav_path
                return newest
        except Exception as final_err:
            print(f"Final attempt failed: {final_err}")
            
        raise FileNotFoundError(f"Could not find downloaded audio file from YouTube")
        
    except Exception as e:
        print(f"Error downloading YouTube audio: {str(e)}")
        if "HTTP Error 403: Forbidden" in str(e):
            raise Exception("Access to this YouTube video is forbidden. It may be restricted content.")
        elif "Sign in" in str(e):
            raise Exception("This YouTube video requires sign-in (age-restricted content)")
        elif "unavailable" in str(e).lower():
            raise Exception("This YouTube video is unavailable or has been removed")
        else:
            raise

def generate_srt_file(words_with_time, segment_length=12):
    """
    Generate SRT file content from word timestamps
    
    Args:
        words_with_time: List of words with start and end times
        segment_length: Max number of words per subtitle segment
    
    Returns:
        String containing SRT file content
    """
    if not words_with_time:
        return ""
    
    srt_content = []
    segment_count = 0
    
    # Process words in segments
    for i in range(0, len(words_with_time), segment_length):
        segment_count += 1
        segment = words_with_time[i:i+segment_length]
        
        # Get start time from first word and end time from last word
        start_time = segment[0]["start_time"]
        end_time = segment[-1]["end_time"]
        
        # Format timestamps for SRT
        start_timestamp = format_srt_timestamp(start_time)
        end_timestamp = format_srt_timestamp(end_time)
        
        # Create segment text
        segment_text = " ".join(word["word"] for word in segment)
        
        # Add SRT entry
        srt_content.append(f"{segment_count}")
        srt_content.append(f"{start_timestamp} --> {end_timestamp}")
        srt_content.append(f"{segment_text}")
        srt_content.append("")  # Empty line between entries
    
    # Join all lines to create the SRT content
    return "\n".join(srt_content)

@app.route('/')
def index():
    """Render the main page"""
    # Pass demo mode status and current year to template
    current_year = datetime.now().year
    return render_template('index.html', languages=LANGUAGES, demo_mode=app.config['DEMO_MODE'], now={'year': current_year})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Process based on file type
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # Convert video to audio
            try:
                audio_path = convert_video_to_audio(filename)
                session['audio_path'] = audio_path
                return redirect(url_for('transcribe'))
            except Exception as e:
                flash(f'Error converting video: {str(e)}')
                return redirect(url_for('index'))
        
        elif file_ext in ['.mp3', '.ogg', '.flac', '.aac', '.m4a']:
            # Convert audio to WAV
            try:
                wav_path = convert_audio_to_wav(filename)
                session['audio_path'] = wav_path
                return redirect(url_for('transcribe'))
            except Exception as e:
                flash(f'Error converting audio: {str(e)}')
                return redirect(url_for('index'))
        
        elif file_ext == '.wav':
            # Already WAV format
            session['audio_path'] = filename
            return redirect(url_for('transcribe'))
        
        else:
            flash('Unsupported file format')
            return redirect(url_for('index'))

def is_valid_youtube_url(url):
    """Validate YouTube URL format"""
    # List of valid YouTube URL patterns
    youtube_patterns = [
        r'^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'^https?://(?:www\.)?youtube\.com/v/[\w-]+',
        r'^https?://youtu\.be/[\w-]+',
        r'^https?://(?:www\.)?youtube\.com/embed/[\w-]+',
        r'^https?://(?:www\.)?youtube\.com/shorts/[\w-]+'
    ]
    
    import re
    return any(re.match(pattern, url) for pattern in youtube_patterns)

@app.route('/youtube', methods=['POST'])
def process_youtube():
    """Process YouTube URL"""
    youtube_url = request.form.get('youtube_url', '').strip()
    if not youtube_url:
        flash('No YouTube URL provided')
        return redirect(url_for('index'))
    
    # Validate YouTube URL format
    if not is_valid_youtube_url(youtube_url):
        flash('Please enter a valid YouTube URL')
        return redirect(url_for('index'))
    
    try:
        audio_path = download_audio_from_youtube(youtube_url)
        session['audio_path'] = audio_path
        return redirect(url_for('transcribe'))
    except Exception as e:
        flash(f'Error downloading YouTube audio: {str(e)}')
        return redirect(url_for('index'))

@app.route('/record', methods=['POST'])
def record_audio():
    """Handle audio recording"""
    action = request.form.get('action')
    
    if action == 'start':
        # Set a flag in the session that recording has started
        session['is_recording'] = True
        # Generate a unique recording ID
        session['recording_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        return jsonify({'status': 'recording'})
    
    elif action == 'stop' or (action is None and 'audio' in request.files):
        # Check if recording was started
        if 'is_recording' in session and session['is_recording']:
            # Get the audio file from the request
            audio_file = request.files.get('audio')
            
            if audio_file:
                try:
                    # Generate temporary filenames
                    timestamp = session.get('recording_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
                    temp_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"recording_{timestamp}.webm")
                    wav_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"recording_{timestamp}.wav")
                    
                    # Save the original WebM file
                    audio_file.save(temp_filename)
                    
                    # Convert WebM to WAV using ffmpeg
                    try:
                        # Use ffmpeg-python to convert
                        stream = ffmpeg.input(temp_filename)
                        stream = ffmpeg.output(stream, wav_filename, acodec='pcm_s16le', ac=1, ar='16k')
                        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                        
                        # Delete the temporary WebM file
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)
                        
                        # Update the session with the WAV file path
                        session['audio_path'] = wav_filename
                        session['is_recording'] = False
                        
                        return jsonify({'status': 'success', 'redirect': url_for('transcribe')})
                    except ffmpeg.Error as e:
                        error_message = f"FFmpeg error: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}"
                        print(error_message)
                        return jsonify({'status': 'error', 'message': error_message})
                except Exception as e:
                    print(f"Error processing audio file: {str(e)}")
                    return jsonify({'status': 'error', 'message': f'Error processing audio file: {str(e)}'})
            else:
                return jsonify({'status': 'error', 'message': 'No audio file received'})
        else:
            return jsonify({'status': 'error', 'message': 'No recording in progress'})
    
    return jsonify({'status': 'error', 'message': 'Invalid action'})

@app.route('/transcribe')
def transcribe():
    """Render transcription page"""
    if 'audio_path' not in session:
        flash('No audio file to transcribe')
        return redirect(url_for('index'))
    
    current_year = datetime.now().year
    return render_template('transcribe.html', languages=LANGUAGES, now={'year': current_year})

@app.route('/download_srt', methods=['POST'])
def download_srt():
    """Generate and download SRT file from the transcription"""
    transcript_type = request.form.get('transcript_type', 'original')
    
    if 'transcription_result_id' not in session:
        return jsonify({'status': 'error', 'message': 'No transcription available'})
    
    result_id = session.get('transcription_result_id')
    result_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{result_id}.json")
    
    if not os.path.exists(result_file_path):
        return jsonify({'status': 'error', 'message': 'Transcription data not found'})
    
    try:
        # Load the result from the JSON file
        with open(result_file_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        if transcript_type == 'translated':
            words_with_time = result.get('translated_words_with_time', [])
        else:
            words_with_time = result.get('words_with_time', [])
        
        if not words_with_time:
            return jsonify({'status': 'error', 'message': 'No word timing information available'})
        
        # Generate SRT content
        srt_content = generate_srt_file(words_with_time)
        
        # Generate a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        language_suffix = "_original" if transcript_type == 'original' else "_translated"
        filename = f"transcription_{timestamp}{language_suffix}.srt"
        
        # Return the SRT file content
        return jsonify({
            'status': 'success',
            'srt_content': srt_content,
            'filename': filename
        })
    except Exception as e:
        print(f"Error in download_srt: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Error generating SRT: {str(e)}'})

@app.route('/process_transcription', methods=['POST'])
def process_transcription():
    """Process transcription request"""
    if 'audio_path' not in session and not app.config['DEMO_MODE']:
        return jsonify({'status': 'error', 'message': 'No audio file to transcribe'})
    
    # In demo mode, we don't need an actual audio file
    if app.config['DEMO_MODE']:
        audio_path = "demo_audio.wav"  # This file doesn't need to exist
    else:
        audio_path = session['audio_path']
    
    # Get language code from the form 
    language_code = request.form.get('language', 'en-US')
    
    # Handle language code format (e.g., if it's just 'ur' instead of 'ur-PK')
    if '-' not in language_code and language_code.lower() in SPEECH_TO_TEXT_LANGUAGE_CODES:
        language_code = SPEECH_TO_TEXT_LANGUAGE_CODES[language_code.lower()]
    
    translate_to_english = request.form.get('translate', 'false') == 'true'
    
    try:
        # Create a progress callback
        def progress_callback(message, progress_value):
            # We can't directly update the client, but we'll track progress in the session
            session['transcription_progress'] = {
                'message': message,
                'progress': progress_value
            }
            session.modified = True
        
        # Check file size for YouTube downloads - if too large, we'll need to handle differently
        file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        max_openai_size = 25 * 1024 * 1024  # 25MB (OpenAI's limit)
        
        if file_size > max_openai_size:
            # File is too large for a single API call - we need to handle it differently
            if progress_callback:
                progress_callback(f"Audio file is {file_size/1024/1024:.1f}MB - splitting into smaller chunks for processing", 0.1)
            
            # The transcribe_audio function already handles chunking, but we'll make sure
            # the file isn't WAY too big by pre-processing it first for YouTube downloads
            if 'youtube_downloads' in audio_path:
                if progress_callback:
                    progress_callback("Large YouTube download detected. Pre-processing audio...", 0.15)
                
                # Load the audio file
                try:
                    audio_segment = AudioSegment.from_wav(audio_path)
                    
                    # If it's extremely large, reduce quality to lower file size
                    if file_size > 100 * 1024 * 1024:  # If over 100MB
                        if progress_callback:
                            progress_callback("Very large audio file. Optimizing for processing...", 0.2)
                        
                        # Downmix to mono and lower quality if needed
                        audio_segment = audio_segment.set_channels(1)
                        
                        # If still too large, lower sample rate to 16kHz
                        if len(audio_segment) > 2 * 60 * 60 * 1000:  # If over 2 hours
                            audio_segment = audio_segment.set_frame_rate(16000)
                    
                    # The transcribe_audio function will handle the rest of the chunking
                except Exception as pre_err:
                    print(f"Error pre-processing large audio: {str(pre_err)}")
                    # Continue anyway, the transcribe_audio function will try its best
        
        # Call the transcribe_audio function with the progress callback
        result = transcribe_audio(audio_path, language_code, translate_to_english, progress_callback=progress_callback)
        
        # Handle translation progress separately if needed
        if translate_to_english and not language_code.startswith("en"):
            # Check if the translation was chunked (look for progress messages)
            transcript_length = len(result['original_transcript'])
            if transcript_length > 1000:  # Smaller threshold since we now always chunk
                # The translation was likely chunked, update progress to show this
                progress_callback(f"Translation complete. Combined {transcript_length // 1500 + 1} chunks.", 0.98)
        
        # Generate a unique result ID and save it in the session
        result_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(hash(str(result['original_transcript'])))[0:8]
        session['transcription_result_id'] = result_id
        
        # Save the complete result to a JSON file instead of in the session
        result_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{result_id}.json")
        with open(result_file_path, 'w', encoding='utf-8') as f:
            # Convert words with time info to serializable format
            serializable_result = {
                "original_transcript": result['original_transcript'],
                "translated_transcript": result['translated_transcript'],
                "words_with_time": [
                    {"word": w["word"], "start_time": w["start_time"], "end_time": w["end_time"]} 
                    for w in result.get('words_with_time', [])
                ],
                "translated_words_with_time": [
                    {"word": w["word"], "start_time": w["start_time"], "end_time": w["end_time"]} 
                    for w in result.get('translated_words_with_time', [])
                ]
            }
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'status': 'success',
            'original_transcript': result['original_transcript'],
            'translated_transcript': result['translated_transcript'],
            'has_word_timestamps': len(result.get('words_with_time', [])) > 0
        })
    except Exception as e:
        print(f"Error in process_transcription: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/transcription_progress', methods=['GET'])
def transcription_progress():
    """Get the current transcription progress"""
    progress_data = session.get('transcription_progress', {
        'message': 'Initializing...',
        'progress': 0
    })
    return jsonify(progress_data)

# Run the application
if __name__ == "__main__":
    # Run Flask app
    app.run(host='0.0.0.0', port=8080, debug=True) 