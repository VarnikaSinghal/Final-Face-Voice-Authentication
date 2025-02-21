#this code is deployed on our server on http://122.170.111.109:7090 and it's home page is acccessible at http://122.170.111.109:7090/fv/ and http://122.170.111.109:7090/fv/api/show_names
#modified app.py after uploading the code on our server

from flask import Flask, request, jsonify
import os
import logging
import shutil
from main import *

app = Flask(__name__)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

@app.route('/')
def home():
    return 'Hello from Face Voice Authentication app!'


@app.route('/api/register', methods=['POST'])
def register_user():
    name = request.args.get('username')
    if not name:
        return jsonify({"message": "Username is required", "status": False}), 400

    try:
        data = request.get_json()
        
        #    Get image and audio base64 encodings
        image_base64_list = [data.get(f'image{i}') for i in range(1, 4)]
        audio_base64 = data.get('audio1') # Expecting a single audio sample

        #this code is to save the user image and audio data in base64 format before processing.
        # # Create a dictionary to hold the base64 encodings
        # data_to_save = {}
        # for i, image_base64 in enumerate(image_base64_list, start=1):
        #     data_to_save[f'image{i}'] = image_base64

        # data_to_save['audio'] = audio_base64

        # # Define the path for the JSON file
        # json_file_path = os.path.join(audio_data_dir, '3face_1voice_data_Samples.json')

        # # Write the dictionary to the JSON file
        # with open(json_file_path, 'w') as json_file:
        #     json.dump(data_to_save, json_file, indent=4)

        # print(f'Data saved to {json_file_path}')
        
        
        if image_base64_list is not None and audio_base64 is not None:

                # Ensure we have exactly 3 images and 3 audio samples for training
            if len(image_base64_list) != 3 and len(audio_base64) != 1: 
                return jsonify({'message': 'Exactly 3 images and 1 audio is required for registration.', "status": False}), 400
            
            # if everything is fine, the call the register user function
            result, message_returned = register_user_face_voice(name, image_base64_list, audio_base64)  # Passing 3 image and 1 audio sample to register function
            if result:
                return jsonify({'message': message_returned, 'status': True}), 200
            else:
                return jsonify({'message': message_returned, "status": False}), 500
        else:
            return jsonify({'message': 'Face and Audio data required!', "status": False}), 400

    except Exception as e:
        logger.error(f"Error in registration process: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/authenticate_face', methods=['POST'])
def authenticate_face():
    name = request.args.get('username')
    if not name:
        return jsonify({"message": "Username is required", "status": False}), 400

    try:
        data = request.get_json()
        image_base64 = data.get('image')
        
        # face_data_to_save = {}
        # import json
        # face_data_to_save['image'] = image_base64
        # json_file_path = os.path.join(face_data_dir, 'face_data_1_sample.json')
        #         # Write the dictionary to the JSON file
        # with open(json_file_path, 'w') as json_file:
        #     json.dump(face_data_to_save, json_file, indent=4)

        # print(f'Face Data saved to {json_file_path}')

        if image_base64 is not None:
            
             #check if face model present or not, if not return false
            face_model = os.path.join(face_data_dir, f"{name}_face_model.pkl")
            if not os.path.exists(face_model):
                return jsonify({"message": "User face is not registered", "status": False}), 400

            result, returned_message = authenticate_user_face(name, image_base64)
            # result = authenticate_user_face(name, image_base64)
            if result:
                return jsonify({'message': returned_message, 'status': True}), 200
            else:
                return jsonify({'message': returned_message, "status": False}), 401
        else:
            return jsonify({'message': 'Image data required!', "status": False}), 400
        
    except Exception as e:
        logger.error(f"Error in authentication process: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/authenticate_voice', methods=['POST'])
def authenticate_voice():
    name = request.args.get('username')
    if not name:
        return jsonify({"message": "Username is required", "status": False}), 400

    try:
        data = request.get_json()
        audio_base64 = data.get('audio')
        
        if audio_base64 is not None:
            audio_model = os.path.join(audio_data_dir, f'{name}_audio_embedding.npy')
            if not os.path.exists(audio_model):
                return jsonify({"message": f"User audio is not registered", "status": False}), 400
            
            result, returned_message = authenticate_user_voice(name, audio_base64)
            if result:
                return jsonify({'message': returned_message, 'status': True}), 200
            else:
                # return jsonify({'message': 'Voice Authentication failed!', "status": False}), 401
                return jsonify({'message': returned_message, "status": False}), 401
        else:
            return jsonify({'message': 'Audio data required!', "status": False}), 400
        
    except Exception as e:
        logger.error(f"Error in authentication process: {e}")
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/api/delete', methods=['DELETE'])
def delete_user():
    name = request.args.get('username')
    if not name:
        return jsonify({"message": "Username is required", "status": False}), 400

    face_file = os.path.join(face_data_dir, f"{name}_face_model.pkl")
    audio_file = os.path.join(audio_data_dir, f"{name}_audio_embedding.npy")
    
    
    errors = []

    # Delete face file
    try:
        if os.path.exists(face_file):
            os.remove(face_file)
        else:
            errors.append(f'No face data found for {name}')
    except Exception as e:
        errors.append(f"Error deleting face data for {name}: {str(e)}")

    # Delete audio model embedings
    try:
        if os.path.exists(audio_file): # model file deletion
            os.remove(audio_file)
        else:
            errors.append(f'No audio model data found for {name}')
            
    except Exception as e:
        errors.append(f"Error deleting audio data for {name}: {str(e)}")

    if errors:
        return jsonify({'message': errors, 'status': False}), 404

    return jsonify({'message': f'Deletion successful for {name}', 'status': True}), 200


@app.route('/api/delete_all_users', methods=['DELETE'])
def delete_all_users():
    errors = []  #empty list to store the errors coming in the way of deletion 
    
    # Deleting face data
    try:
        if os.path.exists(face_data_dir):
            shutil.rmtree(face_data_dir)
            
            # Recreating the face data folder after deletion for synchronization
            os.makedirs(face_data_dir)
            logging.info('All contents of the Face Data Directory has been deleted.')
        else:
            logging.info('No face data directory found to delete.')
            errors.append('No face data directory found.')
    except Exception as e:
        logging.error(f'Error deleting contents of the Face Data Directory. Reason: {str(e)}')
        errors.append(f"Error deleting face data: {str(e)}")
        

    # Deleting audio data
    try:
        if os.path.exists(audio_data_dir):
            shutil.rmtree(audio_data_dir)
            
            # Recreating the folder after deletion for synchronization
            os.makedirs(audio_data_dir)
            logging.info('All contents of the Audio Data Directory has been deleted.')
        else:
            logging.info('No audio data directory found to delete.')
            errors.append('No audio data directory found.')
    except Exception as e:
        logging.error(f'Error deleting contents of the Audio Data Directory. Reason: {str(e)}')
        errors.append(f"Error deleting audio data: {str(e)}")

        
    #returning the final results after deletion of data dir's.
    if errors:
        return jsonify({'message': errors, 'status': False}), 404

    return jsonify({'message': f'Deletion successful for all users', 'status': True}), 200



@app.route('/api/show_names', methods=['GET'])
def show_names():
    face_names = [f.split('_')[0] for f in os.listdir(face_data_dir) if f.endswith('.pkl')]
    
    audio_names = [f.split('_')[0] for f in os.listdir(audio_data_dir) if f.endswith('.npy')]
    return jsonify({'Registered_Faces': face_names,'Registered_Audios': audio_names, 'status': True}), 200
    # return jsonify({'Registered_Faces': face_names, 'status': True}), 200


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=5001, debug=True)
