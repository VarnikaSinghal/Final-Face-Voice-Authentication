#---- Face related Modules Importing ----
import pickle
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity

#---- Audio related Modules Importing ----
import os
import io
import json
import torch
import base64
import librosa
import logging
import tempfile
import numpy as np
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
import speech_recognition as sr
from scipy.signal import resample
from vosk import Model, KaldiRecognizer
from logging.handlers import RotatingFileHandler
from sklearn.metrics.pairwise import cosine_similarity
from speechbrain.inference import SpeakerRecognition





# Setup logging
logger = logging.getLogger(__name__)

# Base directories
if os.name == 'nt':  # Check if the operating system is Windows
    base_dir = "D:\\Work_Data\\Face_Voice_Project\\"
else:  # Assume it's a Linux system if not Windows
    base_dir = "/var/www/Face_Voice_Authentication"

model_dir = os.path.join(base_dir, 'tmpdir')
face_data_dir = os.path.join(base_dir, "Face Data")
audio_data_dir = os.path.join(base_dir, "Audio Data")


# Ensure directories exist
os.makedirs(base_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(face_data_dir, exist_ok=True)
os.makedirs(audio_data_dir, exist_ok=True)

# Log file path
log_file_path = os.path.join(base_dir, "face_voice_project.log")

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# File handler with log rotation
file_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)  # 10 MB per log file, up to 5 files
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)





#----------------------- FACE DATA PROCESSING,REGISTRATION AND AUTHENTICATION -----------------------


# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
face_model = InceptionResnetV1(pretrained='vggface2').eval()

# Convert base64 to image
def base64_to_image(base64_str):
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return np.array(image)
    except Exception as e:
        logger.error(f"Error converting base64 to image: {e}", exc_info=True)
        return None

# Extract face embeddings using the deep learning model
def extract_face_embeddings(image_np, face_model):
    try:
        faces = mtcnn(Image.fromarray(image_np))
        if faces is None:
            return np.array([])

        embeddings = face_model(faces).detach().cpu().numpy()
        return embeddings
    except Exception as e:
        logger.error(f"Error extracting face model: {e}", exc_info=True)
        return np.array([])

# Save embeddings for a user face
def save_user_embeddings(name, embeddings):
    try:
        model_path = os.path.join(face_data_dir, f'{name}_face_model.pkl')
        with open(model_path, 'wb') as file:
            pickle.dump(embeddings, file)
        logger.info(f"Embeddings saved and face model trained for user '{name}' at {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving face model for user '{name}': {e}", exc_info=True)

# Load embeddings for a user
def load_user_embeddings(name):
    model_path = os.path.join(face_data_dir, f'{name}_face_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            embeddings = pickle.load(file)
        return embeddings
    else:
        logger.info(f"No model and scaler found for user '{name}'.")
        return None

# ----------------------------- Register a new user face -------------------------------

def register_user_face(name, image_base64_list):
    """Register a new user by saving face embeddings."""
    try:
        if len(image_base64_list) != 3:
            raise ValueError("Exactly 3 images are required for registration.")
        
        logger.info(f"Registering face of user {name}")
        face_embeddings = []
        for img_base64 in image_base64_list:
            frame = base64_to_image(img_base64)
            if frame is None:
                raise ValueError("Failed to convert one of the images from Base64.")

            embeddings = extract_face_embeddings(frame, face_model)
            if embeddings.size > 0:
                face_embeddings.extend(embeddings)
            else:
                logger.error(f"No face embeddings found in one of the provided images for user '{name}'.")
                msg = "Getting an unexpected error."
                return False, msg

        if len(face_embeddings) < 3:
            raise ValueError("Not all images contained detectable faces.")

        face_embeddings_np = np.array(face_embeddings)
        saved = save_user_embeddings(name, face_embeddings_np)
        

        if saved:
            msg = "User face data has been registered."
            return True, msg
        else:
            msg = "Error during face registration."
            return False, msg

    except Exception as e:
        logger.error(f"Error during user registration for '{name}': {e}", exc_info=True)
        msg = "Getting an unexpected error during face registration."
        return False, msg


# ----------------------------- Authenticate a user face -------------------------------

def authenticate_user_face(name, image_base64, threshold = 0.8): #threshold for face recognizing
    """Authenticate a user based on the provided image."""
    try:
        embeddings = load_user_embeddings(name)
        if embeddings is None:
            logger.info(f"No face model found for user '{name}'.")
            msg = "User is not registered."
            return False, msg

        rgb_frame = base64_to_image(image_base64)
        if rgb_frame is None:
            logger.info("The image could not be loaded or is empty.")
            msg = "Image format is not correct."
            return False, msg

        face_embeddings = extract_face_embeddings(rgb_frame, face_model)
        if face_embeddings.size == 0:
            logger.info("No face embeddings extracted from the login image.")
            msg = "No Face found in the image."
            return False, msg

        # Compare the provided face embeddings with stored embeddings
        for embedding in face_embeddings:
            distances = cosine_similarity([embedding], embeddings)
            logger.info(f"Distances calculated - {distances}")
            logger.info(f"Using threshold - {threshold}")
            if np.max(distances) > threshold:  # Adjust threshold based on your needs(0.7 was earlier)
                logger.info(f"Authentication successful for user '{name}'. distance > threshold")
                msg = "Authentication successful!"
                return True, msg

        logger.info(f"Authentication failed for user '{name}'. distance < threshold")
        msg = "Authentication Failed, Either user is not authorized or image quality is not clear enough."
        return False, msg

    except Exception as e:
        logger.error(f"Error during user authentication for '{name}': {e}", exc_info=True)
        msg = "Getting an unexpected error during authentication."
        return False, msg






#----------------------- AUDIO PROCESSING AND USER AUDIO REGISTRATION -----------------------

# Initialize model globally to avoid reloading
def initialize_model():
    logger.info("Loading pre-trained speaker recognition model...")

    try:
        # Load the pre-trained SpeakerRecognition model from hparams ecapa model works well as compare to x-vector model
        
        voice_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=model_dir)
        logger.info("Model loaded successfully!")
        return voice_model
    except Exception as e:
        logger.error(f"Failed to load the model: {e}")
        return None

# Initialize the model
voice_model = initialize_model()

# Convert WebM base64 encoded audio to waveform and resample to a standard sample rate
def decode_and_resample_audio(base64_audio, target_sr=16000):
    try:
        logger.debug("Decoding base64 audio...")
        # Decode base64 audio
        audio_data = base64.b64decode(base64_audio)
        
        # Load audio data into an in-memory file
        audio_file = io.BytesIO(audio_data)
        
        # Convert audio file to WAV format using pydub
        audio_segment = AudioSegment.from_file(audio_file, format="webm")
        temp_wav_path = 'temp.wav'
        audio_segment.export(temp_wav_path, format='wav')
        
        # Load audio with soundfile
        audio_waveform, sr = sf.read(temp_wav_path)
        
        # Resample audio to target sample rate
        if sr != target_sr:
            logger.debug(f"Resampling from {sr} to {target_sr}")
            num_samples = int(len(audio_waveform) * float(target_sr) / sr)
            audio_waveform = resample(audio_waveform, num_samples)
        
        # Clean up temporary file
        os.remove(temp_wav_path)
        logger.info("Audio decoded and resampled successfully.")
        
        return audio_waveform, target_sr
    except Exception as e:
        logger.error(f"Error decoding or resampling audio: {e}")
        return None, None


# Global variables for the model and recognizer
vosk_text_model_path = os.path.join(base_dir, "vosk-model-small-en-us-0.15")
text_model = None
recognizer = None

def load_model():
    global text_model, recognizer
    if text_model is None:
        logger.info(f"Loading Vosk model from: {vosk_text_model_path}")
        if not os.path.exists(vosk_text_model_path):
            logger.error(f"Model path does not exist: {vosk_text_model_path}")
            return False
        text_model = Model(vosk_text_model_path)
        recognizer = KaldiRecognizer(text_model, 16000)  # Assuming a sample rate of 16000
        logger.info("Vosk model loaded successfully")
    return True

def transcribe_audio(file_path):
    temp_file_path = None
    # Ensure the model is loaded before proceeding
    if not load_model(): #calling the load_model function and taking the decison on behalf of it.
        logger.error("Failed to load the Vosk model.")
        return False, None

    try:
        logger.info(f"Loading and transcribing audio from file: {file_path}")
        
        # Load the audio file using librosa
        audio, sample_rate = librosa.load(file_path, sr=None)
        logger.info(f"Audio file loaded successfully. Sample rate: {sample_rate}, Duration: {len(audio) / sample_rate:.2f} seconds")
        
        # Apply noise reduction using noisereduce
        reduced_noise_audio = nr.reduce_noise(y=audio, sr=sample_rate)
        logger.info("Noise reduction applied to the audio")

        # Normalize audio volume
        normalised_reduced_noise_audio = reduced_noise_audio / max(abs(reduced_noise_audio))  # Normalize to -1 to 1 range
        logger.info("Audio normalized successfully")

        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
            temp_file_path = temp_wav_file.name
            sf.write(temp_file_path, normalised_reduced_noise_audio, sample_rate, format='WAV')
            logger.info(f"Temporary WAV file created at: {temp_file_path}")

        # Transcribe the audio directly from the temporary file
        with open(temp_file_path, 'rb') as temp_wav:
            while True:
                data = temp_wav.read(4000)  # Read the audio in chunks from the temporary file
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
            result = recognizer.FinalResult()  # Get the final result if not already set

        # Parse the transcription result
        result_dict = json.loads(result)
        text = result_dict.get("text", "")

        if text:
            logger.info(f"Transcription successful: {text}")
            return True, text
        else:
            logger.warning("No text detected in audio.")
            return False, None

    except Exception as e:
        logger.error(f"Error in audio transcription: {e}")
        return False, None

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporary file {temp_file_path} deleted.")



    
# def transcribe_audio(file_path):  #converting the audio to text using google speech to text, for checking is human voice is present or not in audio
#     try:
#         logger.debug(f"Loading and transcribing audio from file: {file_path}")
#         # Load the WAV file using librosa
#         audio, sample_rate = librosa.load(file_path, sr=None)

#         # Apply noise reduction using noisereduce
#         reduced_noise_audio = nr.reduce_noise(y=audio, sr=sample_rate)

#         # # Save the noise-reduced audio to a temporary WAV file
#         temp_wav_file = "Audio Data/temp_noise_reduced.wav"
#         sf.write(temp_wav_file, reduced_noise_audio, sample_rate)

#         # Normalize audio volume
#         normalised_reduced_noise_audio = reduced_noise_audio / max(abs(reduced_noise_audio))  # Normalize to -1 to 1 range

#         normalized_temp_wav_file = "Audio Data/normalized_temp_noise_reduced.wav"
#         sf.write(normalized_temp_wav_file, normalised_reduced_noise_audio, sample_rate)

#         # Perform transcription using SpeechRecognition
#         recognizer = sr.Recognizer()
#         with sr.AudioFile(normalized_temp_wav_file) as source:
#             audio = recognizer.record(source)
#             text = recognizer.recognize_google(audio)
#             if text:
#                 logger.info(f"Transcription successful: {text}")
#                 return True, text
#             else:
#                 logger.warning("No text detected from audio.")
#                 return False, None
#     except sr.UnknownValueError:
#         logger.warning("Google Speech Recognition could not understand the audio")
#         return False, None
#     except sr.RequestError as e:
#         logger.error(f"Could not request results from Google Speech Recognition; {e}")
#         return False, None
#     except Exception as e:
#         logger.error(f"Error in audio transcription: {e}")
#         return False, None

def register_user_voice(user_name, base64_audio):
    logger.info(f"Registering voice of user {user_name}...")
    audio_waveform, sr = decode_and_resample_audio(base64_audio)
    if audio_waveform is None:
        logger.error("Failed to decode or resample audio.")
        msg = "Audio is not in the correct format."
        return False, msg
    
    try:
        # Save waveform temporarily for model processing
        temp_wav_path = 'temp.wav'
        sf.write(temp_wav_path, audio_waveform, sr)

        # Transcribe audio
        transcribed, _ = transcribe_audio(temp_wav_path)
        if not transcribed:
            logger.error("No text detected from audio.")
            msg = "Please speak clearly."
            return False, msg
        
        # Convert audio waveform to tensor
        audio_tensor = torch.tensor(audio_waveform).unsqueeze(0) 

        # Compute embedding
        logger.debug("Computing embedding...")
        embedding = voice_model.encode_batch(audio_tensor).detach().numpy().squeeze()

        # Save the embedding for this user
        embedding_file = os.path.join(audio_data_dir, f'{user_name}_audio_embedding.npy')
        np.save(embedding_file, embedding)

        logger.info(f"User {user_name} registered successfully. Embedding saved.")
        msg = "User voice data has been registered."
        return True, msg
    except Exception as e:
        logger.error(f"Error registering user {user_name}: {e}")
        msg = "Error during voice registration."
        return False, msg
    
    
#  ------------------------ Authenticate a user voice ----------------------------

def authenticate_user_voice(user_name, base64_audio):
    logger.info(f"Authenticating user {user_name}...")
    audio_waveform, sr = decode_and_resample_audio(base64_audio)
    if audio_waveform is None:
        logger.error("Failed to decode or resample audio.")
        msg = "Audio is not in the correct format."
        return False, msg
    
    try:
        # Save waveform temporarily for model processing
        temp_wav_path = 'temp.wav'
        sf.write(temp_wav_path, audio_waveform, sr)

        # Transcribe audio
        transcribed, _ = transcribe_audio(temp_wav_path)
        if not transcribed:
            logger.error("No text detected from audio.")
            msg = "Please speak clearly"
            return False, msg
        
        # Convert audio waveform to tensor
        audio_tensor = torch.tensor(audio_waveform).unsqueeze(0)  # Add batch dimension

        # Compute embedding
        logger.debug("Computing embedding...")
        embedding = voice_model.encode_batch(audio_tensor).detach().numpy().squeeze()

        # Load the saved embedding for this user
        embedding_file = os.path.join(audio_data_dir, f'{user_name}_audio_embedding.npy')
        if not os.path.exists(embedding_file):
            logger.error(f"No registered embedding found for user {user_name}")
            msg = "User is not registered."
            return False, msg
        
        saved_embedding = np.load(embedding_file)

        # Compute cosine similarity between the current and saved embeddings
        logger.debug("Computing cosine similarity...")
        similarity = cosine_similarity([embedding], [saved_embedding])[0][0]
        logger.debug(f"Login Embedding: {embedding}")
        logger.debug(f"Saved Embedding: {saved_embedding}")

        # Define a higher threshold, for more accuracy, -1 to 1 is a threshold limit
        user_threshold = 0.7  # 0.7 in starting for ecapa model
        
        # Compare with threshold
        authenticated = similarity > user_threshold
        logger.info(f"Similarity score: {similarity}, Threshold: {user_threshold}")
        if authenticated:
            logger.info(f"Authentication successful for {user_name}! (Similarity score > Threshold)")
            msg = "Authentication successful!"
            return True, msg
            
        else:
            logger.warning(f"Authentication failed for {user_name}. (Similarity score <= Threshold)")
            msg  = "Authentication Failed, Either you are not authorized or audio is not clear enough."
            return False, msg
        
        # return authenticated
    except Exception as e:
        logger.error(f"Error authenticating user {user_name}: {e}")
        msg = "Getting an unexpected error during authentication."
        return False, msg



# ------------------------ REGISTRATION OF A NEW USER WITH FACE AND VOICE ----------------------------------
    
def register_user_face_voice(name, image_base64_list, audio_base64):
    """Register a new user by saving face and audio data and training models."""
    try:
        # # Ensure we have exactly 3 images and 3 audio samples for training
        # if len(image_base64_list) != 3 and len(audio_base64) != 1: 
        #     raise ValueError("Exactly 3 images and 1 audio is required for registration.")
        
        logger.info(f"3 images and 1 audio has taken successfully for user '{name}'...")
        
        logger.info(f"Starting user registration for '{name}'...")
        
        # calling function for face registration
        face_registered, face_message = register_user_face(name, image_base64_list) # calling for user face registration --------
        logger.info(f'Face registered and saved -  {face_registered}')
        
        # calling function for voice registration
        voice_registered, voice_message = register_user_voice(name, audio_base64)  # calling for user voice registration --------
        logger.info(f"Voice registered and saved -  {voice_registered}")
        
        
        # logger.info(f'Face registered and saved - {face_registered}')
        # logger.info(f"Voice registered and saved -  {voice_registered}")
        
        if face_registered and voice_registered: # if both face and voice data registered and saved.
            logger.info("Both Face and Voice data is saved.")
            msg = "User has been registered successfully!"
            return True, msg
        
        elif not face_registered:  # if getting error during face registration
            logger.info("Getting an error during face registration. So, returning False to API.")
            msg = f"User Registration Failed, {face_message}"
            return False, msg
            
        else:  #if audio registration got failed.
            
            # check here if face data or audio data saved for user, delete that and return false
            try:
                face_model = os.path.join(face_data_dir, f'{name}_face_model.pkl')
                if os.path.exists(face_model):
                    os.remove(face_model)
                # return False # if face model exists, delete that and return False, otherwise return False simply  
            except Exception as e:
                logger.info(f"Error deleting data for {name}: {str(e)}")
            
            logger.info("Getting an error during voice registration. So, returning False to API.")
            msg = f"User Registration Failed, {voice_message}"
            return False, msg
            
            
        
    except Exception as e:
        logger.error(f"Error during user registration for '{name}': {e}", exc_info=True)
        msg = "Error during registration."
        return False, msg