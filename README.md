# Image cataloging through smile detection

### Idea
The project aims to create a Multi label detection-classification model for group picture capture through face detection and smile classification.The objective is to identify and store images containing all faces within a continuous stream, specifically highlighting boxes that encapsulate smiling individuals. The model's design includes the ability to trigger image capture when over 30% of detected faces exhibit a smiling expression. 

### Data and face detection
We have obtained the data for initially distinguishing smiling/non-smiling faces from Source - https://github.com/hromi/SMILEsmileD.Our dataset contains two distinct sample sets: positive samples depicting individuals exhibiting smiling facial expressions and negative samples portraying individuals without smiling expressions.These sets are curated for the purpose of training the model, allowing it to discern and classify between facial features associated with smiling and neutral or non-smiling countenance. The relevant code for the classification model is present in the notebook smile_model.ipynb. And the model is saved as the folder saved_smile_model.
MTCNN is used for face detection and a CNN is used for classifying smiling/non-smiling faces.
![group](https://github.com/Neelesh1305/AI-project-image-cataloging-though-detection/assets/113800036/254fc7d2-5a08-4825-b50e-33fa9bc8eeec)

### Capturing the pictures
We have used the camera connected to a NVIDIA jetson-nano to perform the smile detection and catolging the images. The code is run in a docker container on the jetson and the code that is used there is present in smile_detect.py and the captured images folder contains the images that are captured which include smiling.

### References
1. Ghousethanedar. (2020, September 8). smile-detection. Kaggle. https://www.kaggle.com/code/ghousethanedar/smile-detection
2. Matsugu, M., Mori, K., Mitari, Y., & Kaneda, Y. (2003). Subject independent facial expression recognition with robust face detection using a convolutional neural network. Neural Networks, 16(5–6), 555–559. https://doi.org/10.1016/s0893-6080(03)00115-1
3. Khan, A. U. (2022). Facial emotion recognition using conventional machine learning and deep learning methods: current achievements, analysis and remaining challenges. Information, 13(6), 268. https://doi.org/10.3390/info13060268
4. Nguyen, C. K., Tran, G. S., Nghiem, T. P., Burie, J., & Luong, C. M. (2019). Real-Time Smile Detection using Deep Learning. Journal of Computer Science and Cybernetics, 35(2), 135–145. https://doi.org/10.15625/1813-9663/35/2/13315
