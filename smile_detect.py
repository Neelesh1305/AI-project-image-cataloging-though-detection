#!/usr/bin/env python
import cv2
import sys
import argparse
import numpy as np
import jetson_utils
from mtcnn import MTCNN

from jetson_utils import videoSource, videoOutput, Log
from tensorflow.keras.models import load_model, save_model

sys.path.append('/usr/lib/python3.8/dist-packages/cv2/python-3.8/')

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")

# Specify the path to store images
parser.add_argument("--save_path", type=str, default="", help="Path to save images")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# create video sources and outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

# create face detector
face_detector = MTCNN()
smile_model = load_model('saved_smile_model')

# Initialize smile counter
smile_counter = 0
no_of_images=0

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None:  # timeout or error in capturing image
        print("Error: Could not capture image.")
        continue

    # remove
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    rgb_img = img.copy()

    faces = face_detector.detect_faces(rgb_img)

    if faces is None:  # error in face detection
        print("Error: Face detection failed.")
        continue

    # print the number of detected faces
    print("detected {:d} faces in image".format(len(faces)))

    # Initialize smile faces counter for each frame
    frame_smile_counter = 0

    # process each detected face
    for face in faces:
        x, y, w, h = face['box']
        face_roi = rgb_img[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
        resized_face = cv2.resize(gray_face, (64, 64))
        input_image = np.reshape(resized_face, (1, 64, 64, 1))

        # Predict smile using the model
        smile_prediction = smile_model.predict(input_image)[0][0]
        smile_label = 'Smile' if smile_prediction > 0.5 else 'No Smile'

        box_color = (0, 255, 0) if smile_prediction > 0.5 else (255, 0, 0)
        cv2.rectangle(rgb_img, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(rgb_img, f'{smile_label}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Count smile faces in the frame
        if smile_prediction > 0.5:
            frame_smile_counter += 1

    # Update the total smile counter
    smile_counter += frame_smile_counter

    img_resized = cv2.resize(rgb_img, (1024, 1024), interpolation=cv2.INTER_AREA)
    
    # render the resized image
    output.Render(jetson_utils.cudaFromNumpy(img_resized))

    # update the title bar
    output.SetStatus("Smile Detection | Network FPS: MTCNN")

    # Exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

    # Check if more than 30% of faces are smiles and save the image

    if smile_counter > 0.3 * len(faces):
    # Specify the filename to save
        no_of_images += 1
        Captured_images = f"{args.save_path}/{smile_counter}smiles_{no_of_images}.jpg"
        cv2.imwrite(Captured_images, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Save the original image 'img' instead of 'rgb_img'
        print(f"Image saved: {Captured_images}")

# Reset the counters for the next iteration
        smile_counter = 0


    
    # if smile_counter > 0.3 * len(faces):
    # # Specify the filename to save
    #     no_of_images+=1
    #     Captured_images = f"{args.save_path}/{smile_counter}smiles_{no_of_images}.jpg"
    #     cv2.imwrite(Captured_images, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Save the original image 'img' instead of 'rgb_img'
    #     print(f"Image saved: {Captured_images}")

    # # Reset the counters for the next iteration
    #     smile_counter = 0

