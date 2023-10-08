from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import shutil

app = Flask(__name__)

# Configure the upload and output folders
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['FINAL_FRAME_FOLDER'] = 'final_frames'
app.config['INITIAL_FRAME_FOLDER'] = 'initial'
#app.config['VIDEO_OUTPUT'] = 'finalvideo'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}  # Add the allowed video file extensions

# Ensure the upload and output folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


def allowed_file(filename):
    # Check if the file extension is allowed
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_frames_from_video(video_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Read the frames and save them as images
    frame_count = 0
    while True:
        # Read the next frame
        success, frame = video.read()

        # Break the loop if no more frames are available
        if not success:
            break

        # Save the frame as an image
        frame_path = os.path.join(output_folder, f"frame_{frame_count}.png")
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    # Release the video file
    video.release()


def process_frames(model, input_folder, output_folder, final_frame_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each frame in the input folder using the model
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Load the frame
            frame_path = os.path.join(input_folder, filename)
            original_frame_path = os.path.join(input_folder, filename)
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

            # Check if the frame was successfully loaded
            if frame is None:
                print(f"Failed to load frame: {filename}")
                continue

            frame = frame / 255.0  # Normalize to [0, 1]
            frame_resized = cv2.resize(frame, (256, 256))
            input_image = np.expand_dims(frame_resized, axis=-1)
            input_image = np.repeat(input_image, 3, axis=-1)

            # Perform the prediction
            predicted_mask = model.predict(np.array([input_image]))
            predicted_mask = np.squeeze(predicted_mask)

            # Threshold the predicted mask to convert it into a binary mask
            threshold = 0.5  # Adjust the threshold as needed
            predicted_mask_binary = np.where(predicted_mask > threshold, 255, 0).astype(np.uint8)

            # Save the processed frame with the predicted mask
            processed_frame_path = os.path.join(output_folder, filename)
            cv2.imwrite(processed_frame_path, predicted_mask_binary)

            # Apply image processing functions to the processed frame
            # 1. Removing black background
            output_path = os.path.join(final_frame_folder, filename)
            add_red_border(processed_frame_path, processed_frame_path)
            
            make_white_pixels_transparent(processed_frame_path, processed_frame_path)

            make_black_pixels_transparent(processed_frame_path, processed_frame_path)
            
            # Get the corresponding original CT scan frame
            

            # If you want to overlap the processed frame with the original frame:
            # 5. Overlapping images
            overlap_images(processed_frame_path, original_frame_path, output_path)



def create_video_from_frames(input_folder, output_video_path, output_fps):
    # Get the list of image files in the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])

    # Read the first image to get the image size
    first_image_path = os.path.join(input_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Create the video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (width, height))

    # Write each image to the video file
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()


# Define the image processing functions

    
def add_red_border(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to obtain a binary mask of white pixels
    _, binary_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours of white pixels
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image
    image_with_boundary = np.copy(image)

    # Draw a thick red boundary around white pixels
    thickness = 2  # Adjust the thickness as needed
    color = (0, 255, 0)  # Red color
    cv2.drawContours(image_with_boundary, contours, -1, color, thickness)

    # Convert the image array to PIL Image
    image_pil = Image.fromarray(image_with_boundary)

    # Save the image with the red boundary
    image_pil.save(output_path)

def make_white_pixels_transparent(image_path, output_path):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to RGBA mode (with alpha channel)
    image = image.convert("RGBA")

    # Get the pixel data of the image
    pixel_data = image.load()

    # Iterate over each pixel
    for i in range(image.width):
        for j in range(image.height):
            # Get the RGBA values of the pixel
            r, g, b, a = pixel_data[i, j]

            # Check if the pixel is white or almost white
            if r >= 200 and g >= 200 and b >= 200:
                # Set the alpha value of the pixel to 0 (transparent)
                pixel_data[i, j] = (r, g, b, 0)

    # Save the modified image
    image.save(output_path)

def make_black_pixels_transparent(image_path, output_path):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to RGBA mode (with alpha channel)
    image = image.convert("RGBA")

    # Get the pixel data of the image
    pixel_data = image.load()

    # Iterate over each pixel
    for i in range(image.width):
        for j in range(image.height):
            # Get the RGBA values of the pixel
            r, g, b, a = pixel_data[i, j]

            # Check if the pixel is black or almost black
            if r <= 10 and g <= 10 and b <= 10:
                # Set the alpha value of the pixel to 0 (transparent)
                pixel_data[i, j] = (r, g, b, 0)

    # Save the modified image
    image.save(output_path)


'''def overlap_images(background_path, overlay_path, output_path, position=(0, 0)):
    # Open the background image
    background = Image.open(background_path).convert("RGBA")

    # Open the overlay image
    overlay = Image.open(overlay_path).convert("RGBA")

    # Resize the overlay image to match the background size
    overlay = overlay.resize(background.size, Image.ANTIALIAS)

    # Create a new image with transparency
    combined = Image.alpha_composite(background, overlay)

    # Save the combined image
    combined.save(output_path)'''
def overlap_images(image_path1, image_path2, output_path):
    # Load the images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Resize image2 to match the size of image1
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Blend the two images using alpha blending
    alpha =   0.9 # Adjust the alpha value for the desired blending effect
    blended_image = cv2.addWeighted(image1, 1, image2, 1, 1)

    # Save the overlapped image
    cv2.imwrite(output_path, blended_image)














@app.route('/process_video', methods=['POST'])
def process_video():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Check if the file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Save the uploaded file to the upload folder
    if file in request.files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract frames from the video
        extract_frames_from_video(file_path, app.config['INITIAL_FRAME_FOLDER'])
        import tensorflow as tf
        from tensorflow.keras import backend as K
        import numpy as np

        def dice_coefficients(y_true, y_pred):
            intersection = np.sum(y_true * y_pred)
            union = np.sum(y_true) + np.sum(y_pred)
            dice = (2.0 * intersection) / (union + 1e-7)  # Add a small constant to avoid division by zero
            return dice

        

        

        def dice_coefficients_loss(y_true, y_pred, smooth=100):
            return -dice_coefficients(y_true, y_pred, smooth)






        def iou(y_true, y_pred, smooth=100):
            intersection = K.sum(y_true * y_pred)
            sum = K.sum(y_true + y_pred)
            iou = (intersection + smooth) / (sum - intersection + smooth)
            return iou



        def jaccard_distance(y_true, y_pred):
            y_true_flatten = K.flatten(y_true)
            y_pred_flatten = K.flatten(y_pred)
            return -iou(y_true_flatten, y_pred_flatten)

        # Register the custom metric function
        custom_objects = {'iou': iou,'dice_coefficients': dice_coefficients}

        # Load the model with the custom metric function
        model = tf.keras.models.load_model('unet_small.hdf5', custom_objects=custom_objects)

            # Process the frames using the model
        process_frames(model, app.config['INITIAL_FRAME_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['FINAL_FRAME_FOLDER'])

        # Create a video from the processed frames
        output_video_path = os.path.join(app.config['FINAL_FRAME_FOLDER'], 'output.mp4')
        create_video_from_frames(app.config['FINAL_FRAME_FOLDER'], output_video_path, output_fps=30)

        # Delete the contents of the upload and output folders
        '''shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        shutil.rmtree(app.config['OUTPUT_FOLDER'])
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
        shutil.rmtree(app.config['FINAL_FRAME_FOLDER'])
        os.makedirs(app.config['FINAL_FRAME_FOLDER'], exist_ok=True)
        
        #shutil.rmtree(app.config['VIDEO_OUTPUT'])
        #os.makedirs(app.config['VIDEO_OUTPUT'], exist_ok=True)
        shutil.rmtree(app.config['INITIAL_FRAME_FOLDER'])
        os.makedirs(app.config['INITIAL_FRAME_FOLDER'], exist_ok=True)'''

        return jsonify({'message': 'Video processed successfully'}), 200


    if __name__ == '__main__':
        app.run(debug=True)
