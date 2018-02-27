
# Artificial Intelligence Nanodegree
## Computer Vision Capstone
## Project: Facial Keypoint Detection

---

Welcome to the final Computer Vision project in the Artificial Intelligence Nanodegree program!  

In this project, you’ll combine your knowledge of computer vision techniques and deep learning to build and end-to-end facial keypoint recognition system! Facial keypoints include points around the eyes, nose, and mouth on any face and are used in many applications, from facial tracking to emotion recognition. 

There are three main parts to this project:

**Part 1** : Investigating OpenCV, pre-processing, and face detection

**Part 2** : Training a Convolutional Neural Network (CNN) to detect facial keypoints

**Part 3** : Putting parts 1 and 2 together to identify facial keypoints on any image!

---

**Here's what you need to know to complete the project:*

1. In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. 
    
    a. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 


2. In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. 
    
    a. Each section where you will answer a question is preceded by a **'Question X'** header. 
    
    b. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains **optional** suggestions for enhancing the project beyond the minimum requirements. If you decide to pursue the "(Optional)" sections, you should include the code in this IPython notebook.

Your project submission will be evaluated based on your answers to *each* of the questions and the code implementations you provide.  

### Steps to Complete the Project

Each part of the notebook is further broken down into separate steps.  Feel free to use the links below to navigate the notebook.

In this project you will get to explore a few of the many computer vision algorithms built into the OpenCV library.  This expansive computer vision library is now [almost 20 years old](https://en.wikipedia.org/wiki/OpenCV#History) and still growing! 

The project itself is broken down into three large parts, then even further into separate steps.  Make sure to read through each step, and complete any sections that begin with **'(IMPLEMENTATION)'** in the header; these implementation sections may contain multiple TODOs that will be marked in code.  For convenience, we provide links to each of these steps below.

**Part 1** : Investigating OpenCV, pre-processing, and face detection

* [Step 0](#step0): Detect Faces Using a Haar Cascade Classifier
* [Step 1](#step1): Add Eye Detection
* [Step 2](#step2): De-noise an Image for Better Face Detection
* [Step 3](#step3): Blur an Image and Perform Edge Detection
* [Step 4](#step4): Automatically Hide the Identity of an Individual

**Part 2** : Training a Convolutional Neural Network (CNN) to detect facial keypoints

* [Step 5](#step5): Create a CNN to Recognize Facial Keypoints
* [Step 6](#step6): Compile and Train the Model
* [Step 7](#step7): Visualize the Loss and Answer Questions

**Part 3** : Putting parts 1 and 2 together to identify facial keypoints on any image!

* [Step 8](#step7): Build a Robust Facial Keypoints Detector (Complete the CV Pipeline)



---
<a id='step0'></a>
## Step 0: Detect Faces Using a Haar Cascade Classifier

Have you ever wondered how Facebook automatically tags images with your friends' faces?   Or how high-end cameras automatically find and focus on a certain person's face?  Applications like these depend heavily on the machine learning task known as *face detection* -  which is the task of automatically finding faces in images containing people.  

At its root face detection is a classification problem - that is a problem of distinguishing between distinct classes of things.  With face detection these distinct classes are 1) images of human faces and 2) everything else. 

We use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `detector_architectures` directory.


### Import Resources 

In the next python cell, we load in the required libraries for this section of the project.


```python
# Import required libraries for this section

%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2                     # OpenCV library for computer vision
from PIL import Image
import time 
```

Next, we load in and display a test image for performing face detection.

*Note*: by default OpenCV assumes the ordering of our image's color channels are Blue, then Green, then Red.  This is slightly out of order with most image types we'll use in these experiments, whose color channels are ordered Red, then Green, then Blue.  In order to switch the Blue and Red channels of our test image around we will use OpenCV's ```cvtColor``` function, which you can read more about by [checking out some of its documentation located here](http://docs.opencv.org/3.2.0/df/d9d/tutorial_py_colorspaces.html).  This is a general utility function that can do other transformations too like converting a color image to grayscale, and transforming a standard color image to HSV color space.


```python
# Load in color image for face detection
image = cv2.imread('images/test_image_1.jpg')

# Convert the image to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plot our image using subplots to specify a size and title
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Original Image')
ax1.imshow(image);
```

<img src="images/output_6_0.png"/>

There are a lot of people - and faces - in this picture.  13 faces to be exact!  In the next code cell, we demonstrate how to use a Haar Cascade classifier to detect all the faces in this test image.

This face detector uses information about patterns of intensity in an image to reliably detect faces under varying light conditions. So, to use this face detector, we'll first convert the image from color to grayscale. 

Then, we load in the fully trained architecture of the face detector -- found in the file *haarcascade_frontalface_default.xml* - and use it on our image to find faces! 

To learn more about the parameters of the detector see [this post](https://stackoverflow.com/questions/20801015/recommended-values-for-opencv-detectmultiscale-parameters).


```python
# Convert the RGB  image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Extract the pre-trained face detector from an xml file
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# Detect the faces in image
faces = face_cascade.detectMultiScale(gray, 4, 6)

# Print the number of faces detected in the image
print('Number of faces detected:', len(faces))

# Make a copy of the orginal image to draw face detections on
image_with_detections = np.copy(image)

# Get the bounding box for each detected face
for (x,y,w,h) in faces:
    # Add a red bounding box to the detections image
    cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)
    

# Display the image with the detections
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Image with Face Detections')
ax1.imshow(image_with_detections);
```

    Number of faces detected: 13



<img src="images/output_8_1.png"/>


In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

---
<a id='step1'></a>

## Step 1: Add Eye Detections

There are other pre-trained detectors available that use a Haar Cascade Classifier - including full human body detectors, license plate detectors, and more.  [A full list of the pre-trained architectures can be found here](https://github.com/opencv/opencv/tree/master/data/haarcascades). 

To test your eye detector, we'll first read in a new test image with just a single face.


```python
# Load in color image for face detection
image = cv2.imread('images/james.jpg')

# Convert the image to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plot the RGB image
fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Original Image')
ax1.imshow(image);
```


<img src="images/output_12_0.png"/>


Notice that even though the image is a black and white image, we have read it in as a color image and so it will still need to be converted to grayscale in order to perform the most accurate face detection.

So, the next steps will be to convert this image to grayscale, then load OpenCV's face detector and run it with parameters that detect this face accurately.


```python
# Convert the RGB  image to grayscale
image_copy =  np.copy(image)

# Extract the pre-trained face detector from an xml file
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

def detect_faces(image, scaleFactor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor, 6)

    # Print the number of faces detected in the image
    print('Number of faces detected:', len(faces))

    # Get the bounding box for each detected face
    for (x,y,w,h) in faces:
        # Add a red bounding box to the detections image
        cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
    
detect_faces(image_copy, scaleFactor=1.25)
    
# Display the image with the detections
fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Image with Face Detection')
ax1.imshow(image_copy);
```

    Number of faces detected: 1



<img src="images/output_14_1.png"/>


### (IMPLEMENTATION) Add an eye detector to the current face detection setup.  

A Haar-cascade eye detector can be included in the same way that the face detector was and, in this first task, it will be your job to do just this.

To set up an eye detector, use the stored parameters of the eye cascade detector, called ```haarcascade_eye.xml```, located in the `detector_architectures` subdirectory.  In the next code cell, create your eye detector and store its detections.

**A few notes before you get started**: 

First, make sure to give your loaded eye detector the variable name

``eye_cascade``


and give the list of eye regions you detect the variable name 

``eyes``

Second, since we've already run the face detector over this image, you should only search for eyes *within the rectangular face regions detected in ``faces``*.  This will minimize false detections.

Lastly, once you've run your eye detector over the facial detection region, you should display the RGB image with both the face detection boxes (in red) and your eye detections (in green) to verify that everything works as expected.


```python
# Make a copy of the original image to plot rectangle detections
image_copy = np.copy(image)   

face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_eye.xml') 
    
def detect_face_and_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)  
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w] # Region of interest
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(image, (x+ex,y+ey), (x+ex+ew,y+ey+eh), (0,255,0), 2)

detect_face_and_eyes(image_copy)  
    
# Plot the image with both faces and eyes detected
fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Image with Face and Eye Detection')
ax1.imshow(image_copy);
```


<img src="images/output_17_0.png"/>


## (Optional) Add face and eye detection to your laptop camera

It's time to kick it up a notch, and add face and eye detection to your laptop's camera!  Afterwards, you'll be able to show off your creation like in the gif shown below - made with a completed version of the code!  

<img src="images/laptop_face_detector_example.gif"/>

Notice that not all of the detections here are perfect - and your result need not be perfect either.   You should spend a small amount of time tuning the parameters of your detectors to get reasonable results, but don't hold out for perfection.  If we wanted perfection we'd need to spend a ton of time tuning the parameters of each detector, cleaning up the input image frames, etc. You can think of this as more of a rapid prototype. 

The next cell contains code for a wrapper function called ``laptop_camera_face_eye_detector`` that, when called, will activate your laptop's camera.  You will place the relevant face and eye detection code in this wrapper function to implement face/eye detection and mark those detections on each image frame that your camera captures.

Before adding anything to the function, you can run it to get an idea of how it works - a small window should pop up showing you the live feed from your camera; you can press any key to close this window.

**Note:** Mac users may find that activating this function kills the kernel of their notebook every once in a while.  If this happens to you, just restart your notebook's kernel, activate cell(s) containing any crucial import statements, and you'll be good to go!


```python
# Add face and eye detection to this laptop camera function 
# Make sure to draw out all faces/eyes found in each frame on the shown video feed

import cv2
import time 

# wrapper function for face/eye detection with your laptop camera
def laptop_camera_go():
    # Create instance of video capturer
    cv2.namedWindow("face detection activated")
    vc = cv2.VideoCapture(0)

    # Try to get the first frame
    if vc.isOpened(): 
        rval, frame = vc.read()
    else:
        rval = False
    
    # Keep the video stream open
    while rval:
        
        detect_face_and_eyes(frame)
        
        # Plot the image from camera with all the face and eye detections marked
        cv2.imshow("face detection activated", frame)
        
        # Exit functionality - press any key to exit laptop video
        key = cv2.waitKey(20)
        if key > 0: # Exit by pressing any key
            # Destroy windows 
            cv2.destroyAllWindows()
            
            # Make sure window closes on OSx
            for i in range (1,5):
                cv2.waitKey(1)
            return
        
        # Read next frame
        time.sleep(0.05)             # control framerate for computation - default 20 frames per sec
        rval, frame = vc.read()    
```


```python
# Call the laptop camera face/eye detector function above

# IMPORTANT: this does not work on Ubuntu 17.10 unless you install OpenCV with the following command:
# pip install opencv-contrib-python
# More details here: https://pypi.python.org/pypi/opencv-contrib-python 

laptop_camera_go()
```

<img src="test_models/FaceAndEyeDetection.gif"/>

---
<a id='step2'></a>

## Step 2: De-noise an Image for Better Face Detection

Image quality is an important aspect of any computer vision task. Typically, when creating a set of images to train a deep learning network, significant care is taken to ensure that training images are free of visual noise or artifacts that hinder object detection.  While computer vision algorithms - like a face detector - are typically trained on 'nice' data such as this, new test data doesn't always look so nice!

When applying a trained computer vision algorithm to a new piece of test data one often cleans it up first before feeding it in.  This sort of cleaning - referred to as *pre-processing* - can include a number of cleaning phases like blurring, de-noising, color transformations, etc., and many of these tasks can be accomplished using OpenCV.

In this short subsection we explore OpenCV's noise-removal functionality to see how we can clean up a noisy image, which we then feed into our trained face detector.

### Create a noisy image to work with

In the next cell, we create an artificial noisy version of the previous multi-face image.  This is a little exaggerated - we don't typically get images that are this noisy - but [image noise](https://digital-photography-school.com/how-to-avoid-and-reduce-noise-in-your-images/), or 'grainy-ness' in a digitial image - is a fairly common phenomenon.


```python
# Load in the multi-face test image again
image = cv2.imread('images/test_image_1.jpg')

# Convert the image copy to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Make an array copy of this image
image_with_noise = np.asarray(image)

# Create noise - here we add noise sampled randomly from a Gaussian distribution: a common model for noise
noise_level = 40
noise = np.random.randn(image.shape[0],image.shape[1],image.shape[2])*noise_level

# Add this noise to the array image copy
image_with_noise = image_with_noise + noise

# Convert back to uint8 format
image_with_noise = np.asarray([np.uint8(np.clip(i,0,255)) for i in image_with_noise])

# Plot our noisy image!
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Noisy Image')
ax1.imshow(image_with_noise);
```


<img src="images/output_25_0.png"/>


In the context of face detection, the problem with an image like this is that  - due to noise - we may miss some faces or get false detections.  

In the next cell we apply the same trained OpenCV detector with the same settings as before, to see what sort of detections we get.


```python
# Convert the RGB  image to grayscale
gray_noise = cv2.cvtColor(image_with_noise, cv2.COLOR_RGB2GRAY)

# Extract the pre-trained face detector from an xml file
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# Detect the faces in image
faces = face_cascade.detectMultiScale(gray_noise, 4, 6)

# Print the number of faces detected in the image
print('Number of faces detected:', len(faces))

# Make a copy of the orginal image to draw face detections on
image_with_detections = np.copy(image_with_noise)

# Get the bounding box for each detected face
for (x,y,w,h) in faces:
    # Add a red bounding box to the detections image
    cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)
    

# Display the image with the detections
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Noisy Image with Face Detections')
ax1.imshow(image_with_detections);
```

    Number of faces detected: 12



<img src="images/output_27_1.png"/>


With this added noise we now miss one of the faces!

### (IMPLEMENTATION) De-noise this image for better face detection

Time to get your hands dirty: using OpenCV's built in color image de-noising functionality called ```fastNlMeansDenoisingColored``` - de-noise this image enough so that all the faces in the image are properly detected.  Once you have cleaned the image in the next cell, use the cell that follows to run our trained face detector over the cleaned image to check out its detections.

You can find its [official documentation here]([documentation for denoising](http://docs.opencv.org/trunk/d1/d79/group__photo__denoise.html#ga21abc1c8b0e15f78cd3eff672cb6c476) and [a useful example here](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html).


**Note:** you can keep all parameters *except* `photo_render` fixed as shown in the second link above.  Play around with the value of this parameter - see how it affects the resulting cleaned image.


```python
## TODO: Use OpenCV's built in color image de-noising function to clean up our noisy image!

# your final de-noised image (should be RGB)
# src,dst, h_luminance,photo_render, search_window, block_size
denoised_image = cv2.fastNlMeansDenoisingColored(image_with_noise,None,50,50,7,7) 

fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Denoised Image with Face Detections')
ax1.imshow(denoised_image);
```


<img src="images/output_30_0.png"/>



```python
## TODO: Run the face detector on the de-noised image to improve your detections and display the result
# Detect the faces in image
faces = face_cascade.detectMultiScale(denoised_image, 4, 6)

print('Number of faces detected:', len(faces))

image_with_detections = np.copy(denoised_image)

for (x,y,w,h) in faces:
    cv2.rectangle(image_with_detections, (x,y), (x+w,y+h), (255,0,0), 3)
    
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Denoised Image with Face Detections')
ax1.imshow(image_with_detections);
```

    Number of faces detected: 13



<img src="images/output_31_1.png"/>


---
<a id='step3'></a>

## Step 3: Blur an Image and Perform Edge Detection

Now that we have developed a simple pipeline for detecting faces using OpenCV - let's start playing around with a few fun things we can do with all those detected faces!

### Importance of Blur in Edge Detection

Edge detection is a concept that pops up almost everywhere in computer vision applications, as edge-based features (as well as features built on top of edges) are often some of the best features for e.g., object detection and recognition problems.

Edge detection is a dimension reduction technique - by keeping only the edges of an image we get to throw away a lot of non-discriminating information.  And typically the most useful kind of edge-detection is one that preserves only the important, global structures (ignoring local structures that aren't very discriminative).  So removing local structures / retaining global structures is a crucial pre-processing step to performing edge detection in an image, and blurring can do just that.  

Below is an animated gif showing the result of an edge-detected cat [taken from Wikipedia](https://en.wikipedia.org/wiki/Gaussian_blur#Common_uses), where the image is gradually blurred more and more prior to edge detection.  When the animation begins you can't quite make out what it's a picture of, but as the animation evolves and local structures are removed via blurring the cat becomes visible in the edge-detected image.

<img src="images/Edge_Image.gif"/>

Edge detection is a **convolution** performed on the image itself, and you can read about Canny edge detection on [this OpenCV documentation page](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html).

### Canny edge detection

In the cell below we load in a test image, then apply *Canny edge detection* on it.  The original image is shown on the left panel of the figure, while the edge-detected version of the image is shown on the right.  Notice how the result looks very busy - there are too many little details preserved in the image before it is sent to the edge detector.  When applied in computer vision applications, edge detection should preserve *global* structure; doing away with local structures that don't help describe what objects are in the image.


```python
# Load in the image
image = cv2.imread('images/fawzia.jpg')

# Convert to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  

# Perform Canny edge detection
edges = cv2.Canny(gray,100,200)

# Dilate the image to amplify edges
edges = cv2.dilate(edges, None)

# Plot the RGB and edge-detected image
fig = plt.figure(figsize = (15,15))
ax1 = fig.add_subplot(121)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Original Image')
ax1.imshow(image)

ax2 = fig.add_subplot(122)
ax2.set_xticks([])
ax2.set_yticks([])

ax2.set_title('Canny Edges')
ax2.imshow(edges, cmap='gray');
```


<img src="images/output_36_0.png"/>


Without first blurring the image, and removing small, local structures, a lot of irrelevant edge content gets picked up and amplified by the detector (as shown in the right panel above). 

### (IMPLEMENTATION) Blur the image *then* perform edge detection

In the next cell, you will repeat this experiment - blurring the image first to remove these local structures, so that only the important boudnary details remain in the edge-detected image.

Blur the image by using OpenCV's ```filter2d``` functionality - which is discussed in [this documentation page](http://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html) - and use an *averaging kernel* of width equal to 4.


```python
# Load in the image
image = cv2.imread('images/fawzia.jpg')

# Convert to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 

# Blur the test imageusing OpenCV's filter2d functionality, 
# Use an averaging kernel, and a kernel width equal to 4

kernel = np.ones((4,4),np.float32) / 16
blurred_image = cv2.filter2D(gray, -1, kernel)

# Then perform Canny edge detection and display the output

# Perform Canny edge detection
edges = cv2.Canny(blurred_image,100,200)

# Dilate the image to amplify edges
edges = cv2.dilate(edges, None)

# Plot the RGB and edge-detected image
fig = plt.figure(figsize = (15,15))
ax1 = fig.add_subplot(121)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Original Image')
ax1.imshow(blurred_image, cmap='gray')

ax2 = fig.add_subplot(122)
ax2.set_xticks([])
ax2.set_yticks([])

ax2.set_title('Canny Edges')
ax2.imshow(edges, cmap='gray');

```


<img src="images/output_39_0.png"/>


---
<a id='step4'></a>

## Step 4: Automatically Hide the Identity of an Individual

If you film something like a documentary or reality TV, you must get permission from every individual shown on film before you can show their face, otherwise you need to blur it out - by blurring the face a lot (so much so that even the global structures are obscured)!  This is also true for projects like [Google's StreetView maps](https://www.google.com/streetview/) - an enormous collection of mapping images taken from a fleet of Google vehicles.  Because it would be impossible for Google to get the permission of every single person accidentally captured in one of these images they blur out everyone's faces, the detected images must automatically blur the identity of detected people.  Here's a few examples of folks caught in the camera of a Google street view vehicle.

<img src="images/streetview_example_1.jpg"/>
<img src="images/streetview_example_2.jpg"/>


### Read in an image to perform identity detection

Let's try this out for ourselves.  Use the face detection pipeline built above and what you know about using the ```filter2D``` to blur and image, and use these in tandem to hide the identity of the person in the following image - loaded in and printed in the next cell. 


```python
# Load in the image
image = cv2.imread('images/gus.jpg')

# Convert the image to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Original Image')
ax1.imshow(image);
```


<img src="images/output_42_0.png"/>


### (IMPLEMENTATION) Use blurring to hide the identity of an individual in an image

The idea here is to 1) automatically detect the face in this image, and then 2) blur it out!  Make sure to adjust the parameters of the *averaging* blur filter to completely obscure this person's identity.


```python
image_copy = np.copy(image)

# Extract the pre-trained face detector from an xml file
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# Blur detected faces.
def blur_faces(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect the faces in image (image, scaleFactor, minNeighbors)
    faces = face_cascade.detectMultiScale(gray, 1.25, 5)

    # Blur the bounding box around each detected face using an averaging filter and display the result
    for (x,y,w,h) in faces:
        roi = image[y:y+h, x:x+w]
        kernel = np.ones((30,30),np.float32) / 900
        blurred_roi = cv2.filter2D(roi, -1, kernel)
        image[y:y+h, x:x+w] = blurred_roi

blur_faces(image_copy)        
        
# Display the image with the detections
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Image with Mask')
ax1.imshow(image_copy);
```


<img src="images/output_44_0.png"/>


### (Optional) Build identity protection into your laptop camera

In this optional task you can add identity protection to your laptop camera, using the previously completed code where you added face detection to your laptop camera - and the task above.  You should be able to get reasonable results with little parameter tuning - like the one shown in the gif below.

<img src="images/laptop_blurer_example.gif"/>

As with the previous video task, to make this perfect would require significant effort - so don't strive for perfection here, strive for reasonable quality.  

The next cell contains code a wrapper function called ``laptop_camera_identity_hider`` that - when called  - will activate your laptop's camera.  You need to place the relevant face detection and blurring code developed above in this function in order to blur faces entering your laptop camera's field of view.

Before adding anything to the function you can call it to get a hang of how it works - a small window will pop up showing you the live feed from your camera, you can press any key to close this window.

**Note:** Mac users may find that activating this function kills the kernel of their notebook every once in a while.  If this happens to you, just restart your notebook's kernel, activate cell(s) containing any crucial import statements, and you'll be good to go!


```python
# Insert face detection and blurring code into the wrapper below to create an identity protector on your laptop!
import cv2
import time 

def laptop_camera_go():
    # Create instance of video capturer
    cv2.namedWindow("face detection activated")
    vc = cv2.VideoCapture(0)

    # Try to get the first frame
    if vc.isOpened(): 
        rval, frame = vc.read()
    else:
        rval = False
    
    # Keep video stream open
    while rval:
        
        blur_faces(frame)
 
        # Plot image from camera with detections marked
        cv2.imshow("face detection activated", frame)
        
        # Exit functionality - press any key to exit laptop video
        key = cv2.waitKey(20)
        if key > 0: # Exit by pressing any key
            # Destroy windows
            cv2.destroyAllWindows()
            
            for i in range (1,5):
                cv2.waitKey(1)
            return
        
        # Read next frame
        time.sleep(0.05)             # control framerate for computation - default 20 frames per sec
        rval, frame = vc.read()    
        
```


```python
# Run laptop identity hider
laptop_camera_go()
```

<img src="test_models/FaceBlur.gif"/>

---
<a id='step5'></a>

## Step 5: Create a CNN to Recognize Facial Keypoints

OpenCV is often used in practice with other machine learning and deep learning libraries to produce interesting results.  In this stage of the project you will create your own end-to-end pipeline - employing convolutional networks in keras along with OpenCV - to apply a "selfie" filter to streaming video and images.  

You will start by creating and then training a convolutional network that can detect facial keypoints in a small dataset of cropped images of human faces.  We then guide you towards OpenCV to expanding your detection algorithm to more general images.  What are facial keypoints?  Let's take a look at some examples.

<img src="images/keypoints_test_results.png" width=400 height=300/>

Facial keypoints (also called facial landmarks) are the small blue-green dots shown on each of the faces in the image above - there are 15 keypoints marked in each image.  They mark important areas of the face - the eyes, corners of the mouth, the nose, etc.  Facial keypoints can be used in a variety of machine learning applications from face and emotion recognition to commercial applications like the image filters popularized by Snapchat.

Below we illustrate a filter that, using the results of this section, automatically places sunglasses on people in images (using the facial keypoints to place the glasses correctly on each face).  Here, the facial keypoints have been colored lime green for visualization purposes.

<img src="images/obamas_with_shades.png"/>

### Make a facial keypoint detector

But first things first: how can we make a facial keypoint detector?  Well, at a high level, notice that facial keypoint detection is a *regression problem*.  A single face corresponds to a set of 15 facial keypoints (a set of 15 corresponding $(x, y)$ coordinates, i.e., an output point).  Because our input data are images, we can employ a *convolutional neural network* to recognize patterns in our images and learn how to identify these keypoint given sets of labeled data.

In order to train a regressor, we need a training set - a set of facial image / facial keypoint pairs to train on.  For this we will be using [this dataset from Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/data). We've already downloaded this data and placed it in the `data` directory. Make sure that you have both the *training* and *test* data files.  The training dataset contains several thousand $96 \times 96$ grayscale images of cropped human faces, along with each face's 15 corresponding facial keypoints (also called landmarks) that have been placed by hand, and recorded in $(x, y)$ coordinates.  This wonderful resource also has a substantial testing set, which we will use in tinkering with our convolutional network.

To load in this data, run the Python cell below - notice we will load in both the training and testing sets.

The `load_data` function is in the included `utils.py` file.


```python
from utils import *

# Load training set
X_train, y_train = load_data()
print("X_train.shape == {}".format(X_train.shape))
print("y_train.shape == {}; y_train.min == {:.3f}; y_train.max == {:.3f}".format(
    y_train.shape, y_train.min(), y_train.max()))

# Load testing set
X_test, _ = load_data(test=True)
print("X_test.shape == {}".format(X_test.shape))
```

    X_train.shape == (2140, 96, 96, 1)
    y_train.shape == (2140, 30); y_train.min == -0.920; y_train.max == 0.996
    X_test.shape == (1783, 96, 96, 1)


The `load_data` function in `utils.py` originates from this excellent [blog post](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/), which you are *strongly* encouraged to read.  Please take the time now to review this function.  Note how the output values - that is, the coordinates of each set of facial landmarks - have been normalized to take on values in the range $[-1, 1]$, while the pixel values of each input point (a facial image) have been normalized to the range $[0,1]$.  

Note: the original Kaggle dataset contains some images with several missing keypoints.  For simplicity, the `load_data` function removes those images with missing labels from the dataset.  As an __*optional*__ extension, you are welcome to amend the `load_data` function to include the incomplete data points. 

### Visualize the Training Data

Execute the code cell below to visualize a subset of the training data.


```python
import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
    plot_data(X_train[i], y_train[i], ax)
```


<img src="images/output_54_0.png"/>


For each training image, there are two landmarks per eyebrow (**four** total), three per eye (**six** total), **four** for the mouth, and **one** for the tip of the nose.  

Review the `plot_data` function in `utils.py` to understand how the 30-dimensional training labels in `y_train` are mapped to facial locations, as this function will prove useful for your pipeline.

### (IMPLEMENTATION) Specify the CNN Architecture

In this section, you will specify a neural network for predicting the locations of facial keypoints.  Use the code cell below to specify the architecture of your neural network.  We have imported some layers that you may find useful for this task, but if you need to use more Keras layers, feel free to import them in the cell.

Your network should accept a $96 \times 96$ grayscale image as input, and it should output a vector with 30 entries, corresponding to the predicted (horizontal and vertical) locations of 15 facial keypoints.  If you are not sure where to start, you can find some useful starting architectures in [this blog](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/), but you are not permitted to copy any of the architectures that you find online.


```python
# Import deep learning resources from Keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, BatchNormalization
from keras.layers import Flatten, Dense

# Specify a CNN architecture
# Your model should accept 96x96 pixel graysale images in
# It should have a fully-connected output layer with 30 values (2 for each facial keypoint)

model = Sequential()

model.add(Convolution2D(filters=32, 
    kernel_size=3, 
    padding='same', 
    activation='relu', 
    input_shape=(96, 96, 1)))
model.add(MaxPooling2D(pool_size=2))

model.add(Convolution2D(filters=64, 
    kernel_size=3, 
    padding='same', 
    activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Convolution2D(filters=128, 
    kernel_size=3, 
    padding='same', 
    activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(30))

model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_4 (Conv2D)            (None, 96, 96, 32)        320       
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 48, 48, 32)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 48, 48, 64)        18496     
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 24, 24, 64)        0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 24, 24, 128)       73856     
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 12, 12, 128)       0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 18432)             0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 18432)             0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 500)               9216500   
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 500)               0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 30)                15030     
    =================================================================
    Total params: 9,324,202
    Trainable params: 9,324,202
    Non-trainable params: 0
    _________________________________________________________________


---
<a id='step6'></a>

## Step 6: Compile and Train the Model

After specifying your architecture, you'll need to compile and train the model to detect facial keypoints'

### (IMPLEMENTATION) Compile and Train the Model

Use the `compile` [method](https://keras.io/models/sequential/#sequential-model-methods) to configure the learning process.  Experiment with your choice of [optimizer](https://keras.io/optimizers/); you may have some ideas about which will work best (`SGD` vs. `RMSprop`, etc), but take the time to empirically verify your theories.

Use the `fit` [method](https://keras.io/models/sequential/#sequential-model-methods) to train the model.  Break off a validation set by setting `validation_split=0.2`.  Save the returned `History` object in the `history` variable.  

Experiment with your model to minimize the validation loss (measured as mean squared error). A very good model will achieve about 0.0015 loss (though it's possible to do even better).  When you have finished training, [save your model](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) as an HDF5 file with file path `my_model.h5`.


```python
# COMPILE THE MODEL

from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

# This is a regression problem so we use Mean Square Error loss function.
model.compile(optimizer='adamax', loss='mean_squared_error', metrics=['mse'])
```


```python
# TRAIN THE MODEL

from keras.callbacks import ModelCheckpoint  
from sklearn.model_selection import train_test_split

checkpointer = ModelCheckpoint(filepath='my_model.h5', 
    verbose=1, save_best_only=True)

history = model.fit(X_train, y_train, validation_split=0.2, 
    epochs=150, batch_size=10, callbacks=[checkpointer], verbose=1)

```

    Train on 1712 samples, validate on 428 samples
    Epoch 1/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0162 - mean_squared_error: 0.0162
    Epoch 00001: val_loss improved from inf to 0.00587, saving model to my_model.h5
    1712/1712 [==============================] - 7s 4ms/step - loss: 0.0162 - mean_squared_error: 0.0162 - val_loss: 0.0059 - val_mean_squared_error: 0.0059
    Epoch 2/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0079 - mean_squared_error: 0.0079
    Epoch 00002: val_loss improved from 0.00587 to 0.00401, saving model to my_model.h5
    1712/1712 [==============================] - 7s 4ms/step - loss: 0.0079 - mean_squared_error: 0.0079 - val_loss: 0.0040 - val_mean_squared_error: 0.0040
    Epoch 3/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0065 - mean_squared_error: 0.0065
    Epoch 00003: val_loss improved from 0.00401 to 0.00322, saving model to my_model.h5
    1712/1712 [==============================] - 7s 4ms/step - loss: 0.0065 - mean_squared_error: 0.0065 - val_loss: 0.0032 - val_mean_squared_error: 0.0032
    Epoch 4/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0056 - mean_squared_error: 0.0056
    Epoch 00004: val_loss improved from 0.00322 to 0.00305, saving model to my_model.h5
    1712/1712 [==============================] - 7s 4ms/step - loss: 0.0056 - mean_squared_error: 0.0056 - val_loss: 0.0031 - val_mean_squared_error: 0.0031
    Epoch 5/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0050 - mean_squared_error: 0.0050
    Epoch 00005: val_loss improved from 0.00305 to 0.00274, saving model to my_model.h5
    1712/1712 [==============================] - 7s 4ms/step - loss: 0.0050 - mean_squared_error: 0.0050 - val_loss: 0.0027 - val_mean_squared_error: 0.0027
    Epoch 6/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0047 - mean_squared_error: 0.0047
    Epoch 00006: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0047 - mean_squared_error: 0.0047 - val_loss: 0.0028 - val_mean_squared_error: 0.0028
    Epoch 7/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0043 - mean_squared_error: 0.0043
    Epoch 00007: val_loss improved from 0.00274 to 0.00253, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0043 - mean_squared_error: 0.0043 - val_loss: 0.0025 - val_mean_squared_error: 0.0025
    Epoch 8/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0039 - mean_squared_error: 0.0039
    Epoch 00008: val_loss improved from 0.00253 to 0.00207, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0039 - mean_squared_error: 0.0039 - val_loss: 0.0021 - val_mean_squared_error: 0.0021
    Epoch 9/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0036 - mean_squared_error: 0.0036
    Epoch 00009: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0036 - mean_squared_error: 0.0036 - val_loss: 0.0021 - val_mean_squared_error: 0.0021
    Epoch 10/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0034 - mean_squared_error: 0.0034
    Epoch 00010: val_loss improved from 0.00207 to 0.00206, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0034 - mean_squared_error: 0.0034 - val_loss: 0.0021 - val_mean_squared_error: 0.0021
    Epoch 11/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0031 - mean_squared_error: 0.0031
    Epoch 00011: val_loss improved from 0.00206 to 0.00181, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0031 - mean_squared_error: 0.0031 - val_loss: 0.0018 - val_mean_squared_error: 0.0018
    Epoch 12/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0029 - mean_squared_error: 0.0029
    Epoch 00012: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0029 - mean_squared_error: 0.0029 - val_loss: 0.0018 - val_mean_squared_error: 0.0018
    Epoch 13/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0027 - mean_squared_error: 0.0027
    Epoch 00013: val_loss improved from 0.00181 to 0.00171, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0027 - mean_squared_error: 0.0027 - val_loss: 0.0017 - val_mean_squared_error: 0.0017
    Epoch 14/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0026 - mean_squared_error: 0.0026
    Epoch 00014: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0026 - mean_squared_error: 0.0026 - val_loss: 0.0018 - val_mean_squared_error: 0.0018
    Epoch 15/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0024 - mean_squared_error: 0.0024
    Epoch 00015: val_loss improved from 0.00171 to 0.00163, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0024 - mean_squared_error: 0.0024 - val_loss: 0.0016 - val_mean_squared_error: 0.0016
    Epoch 16/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0023 - mean_squared_error: 0.0023
    Epoch 00016: val_loss improved from 0.00163 to 0.00160, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0023 - mean_squared_error: 0.0023 - val_loss: 0.0016 - val_mean_squared_error: 0.0016
    Epoch 17/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0022 - mean_squared_error: 0.0022
    Epoch 00017: val_loss improved from 0.00160 to 0.00139, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0022 - mean_squared_error: 0.0022 - val_loss: 0.0014 - val_mean_squared_error: 0.0014
    Epoch 18/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0021 - mean_squared_error: 0.0021
    Epoch 00018: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0021 - mean_squared_error: 0.0021 - val_loss: 0.0015 - val_mean_squared_error: 0.0015
    Epoch 19/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0020 - mean_squared_error: 0.0020
    Epoch 00019: val_loss improved from 0.00139 to 0.00139, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0020 - mean_squared_error: 0.0020 - val_loss: 0.0014 - val_mean_squared_error: 0.0014
    Epoch 20/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0019 - mean_squared_error: 0.0019
    Epoch 00020: val_loss improved from 0.00139 to 0.00136, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0019 - mean_squared_error: 0.0019 - val_loss: 0.0014 - val_mean_squared_error: 0.0014
    Epoch 21/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0018 - mean_squared_error: 0.0018
    Epoch 00021: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0018 - mean_squared_error: 0.0018 - val_loss: 0.0014 - val_mean_squared_error: 0.0014
    Epoch 22/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0018 - mean_squared_error: 0.0018
    Epoch 00022: val_loss improved from 0.00136 to 0.00132, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0018 - mean_squared_error: 0.0018 - val_loss: 0.0013 - val_mean_squared_error: 0.0013
    Epoch 23/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0017 - mean_squared_error: 0.0017
    Epoch 00023: val_loss improved from 0.00132 to 0.00130, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0017 - mean_squared_error: 0.0017 - val_loss: 0.0013 - val_mean_squared_error: 0.0013
    Epoch 24/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0016 - mean_squared_error: 0.0016
    Epoch 00024: val_loss improved from 0.00130 to 0.00121, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0016 - mean_squared_error: 0.0016 - val_loss: 0.0012 - val_mean_squared_error: 0.0012
    Epoch 25/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0015 - mean_squared_error: 0.0015
    Epoch 00025: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0015 - mean_squared_error: 0.0015 - val_loss: 0.0013 - val_mean_squared_error: 0.0013
    Epoch 26/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0015 - mean_squared_error: 0.0015
    Epoch 00026: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0015 - mean_squared_error: 0.0015 - val_loss: 0.0013 - val_mean_squared_error: 0.0013
    Epoch 27/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0014 - mean_squared_error: 0.0014
    Epoch 00027: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0014 - mean_squared_error: 0.0014 - val_loss: 0.0013 - val_mean_squared_error: 0.0013
    Epoch 28/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0014 - mean_squared_error: 0.0014
    Epoch 00028: val_loss improved from 0.00121 to 0.00113, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0014 - mean_squared_error: 0.0014 - val_loss: 0.0011 - val_mean_squared_error: 0.0011
    Epoch 29/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0013 - mean_squared_error: 0.0013
    Epoch 00029: val_loss improved from 0.00113 to 0.00108, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0013 - mean_squared_error: 0.0013 - val_loss: 0.0011 - val_mean_squared_error: 0.0011
    Epoch 30/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0013 - mean_squared_error: 0.0013
    Epoch 00030: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0013 - mean_squared_error: 0.0013 - val_loss: 0.0011 - val_mean_squared_error: 0.0011
    Epoch 31/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0013 - mean_squared_error: 0.0013
    Epoch 00031: val_loss improved from 0.00108 to 0.00105, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0013 - mean_squared_error: 0.0013 - val_loss: 0.0010 - val_mean_squared_error: 0.0010
    Epoch 32/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0012 - mean_squared_error: 0.0012
    Epoch 00032: val_loss improved from 0.00105 to 0.00105, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0012 - mean_squared_error: 0.0012 - val_loss: 0.0010 - val_mean_squared_error: 0.0010
    Epoch 33/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0012 - mean_squared_error: 0.0012
    Epoch 00033: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0012 - mean_squared_error: 0.0012 - val_loss: 0.0011 - val_mean_squared_error: 0.0011
    Epoch 34/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0012 - mean_squared_error: 0.0012
    Epoch 00034: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0012 - mean_squared_error: 0.0012 - val_loss: 0.0012 - val_mean_squared_error: 0.0012
    Epoch 35/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0011 - mean_squared_error: 0.0011
    Epoch 00035: val_loss improved from 0.00105 to 0.00105, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0011 - mean_squared_error: 0.0011 - val_loss: 0.0010 - val_mean_squared_error: 0.0010
    Epoch 36/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0011 - mean_squared_error: 0.0011
    Epoch 00036: val_loss improved from 0.00105 to 0.00096, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0011 - mean_squared_error: 0.0011 - val_loss: 9.6437e-04 - val_mean_squared_error: 9.6437e-04
    Epoch 37/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0011 - mean_squared_error: 0.0011
    Epoch 00037: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0011 - mean_squared_error: 0.0011 - val_loss: 9.9767e-04 - val_mean_squared_error: 9.9767e-04
    Epoch 38/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0011 - mean_squared_error: 0.0011
    Epoch 00038: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0011 - mean_squared_error: 0.0011 - val_loss: 0.0011 - val_mean_squared_error: 0.0011
    Epoch 39/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0010 - mean_squared_error: 0.0010
    Epoch 00039: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0010 - mean_squared_error: 0.0010 - val_loss: 9.8381e-04 - val_mean_squared_error: 9.8381e-04
    Epoch 40/150
    1710/1712 [============================>.] - ETA: 0s - loss: 0.0010 - mean_squared_error: 0.0010
    Epoch 00040: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0010 - mean_squared_error: 0.0010 - val_loss: 9.7045e-04 - val_mean_squared_error: 9.7045e-04
    Epoch 41/150
    1710/1712 [============================>.] - ETA: 0s - loss: 9.8858e-04 - mean_squared_error: 9.8858e-04
    Epoch 00041: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 9.8783e-04 - mean_squared_error: 9.8783e-04 - val_loss: 0.0011 - val_mean_squared_error: 0.0011
    Epoch 42/150
    1710/1712 [============================>.] - ETA: 0s - loss: 9.9978e-04 - mean_squared_error: 9.9978e-04
    Epoch 00042: val_loss improved from 0.00096 to 0.00091, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 0.0010 - mean_squared_error: 0.0010 - val_loss: 9.0724e-04 - val_mean_squared_error: 9.0724e-04
    Epoch 43/150
    1710/1712 [============================>.] - ETA: 0s - loss: 9.6749e-04 - mean_squared_error: 9.6749e-04
    Epoch 00043: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 9.6884e-04 - mean_squared_error: 9.6884e-04 - val_loss: 9.9235e-04 - val_mean_squared_error: 9.9235e-04
    Epoch 44/150
    1710/1712 [============================>.] - ETA: 0s - loss: 9.5406e-04 - mean_squared_error: 9.5406e-04
    Epoch 00044: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 9.5359e-04 - mean_squared_error: 9.5359e-04 - val_loss: 9.5000e-04 - val_mean_squared_error: 9.5000e-04
    Epoch 45/150
    1710/1712 [============================>.] - ETA: 0s - loss: 9.4127e-04 - mean_squared_error: 9.4127e-04
    Epoch 00045: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 9.4134e-04 - mean_squared_error: 9.4134e-04 - val_loss: 0.0010 - val_mean_squared_error: 0.0010
    Epoch 46/150
    1710/1712 [============================>.] - ETA: 0s - loss: 9.2109e-04 - mean_squared_error: 9.2109e-04
    Epoch 00046: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 9.2098e-04 - mean_squared_error: 9.2098e-04 - val_loss: 9.9595e-04 - val_mean_squared_error: 9.9595e-04
    Epoch 47/150
    1710/1712 [============================>.] - ETA: 0s - loss: 9.3158e-04 - mean_squared_error: 9.3158e-04
    Epoch 00047: val_loss improved from 0.00091 to 0.00087, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 9.3255e-04 - mean_squared_error: 9.3255e-04 - val_loss: 8.7300e-04 - val_mean_squared_error: 8.7300e-04
    Epoch 48/150
    1710/1712 [============================>.] - ETA: 0s - loss: 9.0838e-04 - mean_squared_error: 9.0838e-04
    Epoch 00048: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 9.0778e-04 - mean_squared_error: 9.0778e-04 - val_loss: 9.1665e-04 - val_mean_squared_error: 9.1665e-04
    Epoch 49/150
    1710/1712 [============================>.] - ETA: 0s - loss: 9.0480e-04 - mean_squared_error: 9.0480e-04
    Epoch 00049: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 9.0491e-04 - mean_squared_error: 9.0491e-04 - val_loss: 9.9974e-04 - val_mean_squared_error: 9.9974e-04
    Epoch 50/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.9893e-04 - mean_squared_error: 8.9893e-04
    Epoch 00050: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.9946e-04 - mean_squared_error: 8.9946e-04 - val_loss: 9.8183e-04 - val_mean_squared_error: 9.8183e-04
    Epoch 51/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.9503e-04 - mean_squared_error: 8.9503e-04
    Epoch 00051: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.9467e-04 - mean_squared_error: 8.9467e-04 - val_loss: 9.0819e-04 - val_mean_squared_error: 9.0819e-04
    Epoch 52/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.8474e-04 - mean_squared_error: 8.8474e-04
    Epoch 00052: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.8466e-04 - mean_squared_error: 8.8466e-04 - val_loss: 9.5982e-04 - val_mean_squared_error: 9.5982e-04
    Epoch 53/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.7546e-04 - mean_squared_error: 8.7546e-04
    Epoch 00053: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.7576e-04 - mean_squared_error: 8.7576e-04 - val_loss: 9.5488e-04 - val_mean_squared_error: 9.5488e-04
    Epoch 54/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.7368e-04 - mean_squared_error: 8.7368e-04
    Epoch 00054: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.7315e-04 - mean_squared_error: 8.7315e-04 - val_loss: 9.2066e-04 - val_mean_squared_error: 9.2066e-04
    Epoch 55/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.4463e-04 - mean_squared_error: 8.4463e-04
    Epoch 00055: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.4578e-04 - mean_squared_error: 8.4578e-04 - val_loss: 9.4393e-04 - val_mean_squared_error: 9.4393e-04
    Epoch 56/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.4793e-04 - mean_squared_error: 8.4793e-04
    Epoch 00056: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.4778e-04 - mean_squared_error: 8.4778e-04 - val_loss: 9.4325e-04 - val_mean_squared_error: 9.4325e-04
    Epoch 57/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.3112e-04 - mean_squared_error: 8.3112e-04
    Epoch 00057: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.3083e-04 - mean_squared_error: 8.3083e-04 - val_loss: 9.0477e-04 - val_mean_squared_error: 9.0477e-04
    Epoch 58/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.3027e-04 - mean_squared_error: 8.3027e-04
    Epoch 00058: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.3113e-04 - mean_squared_error: 8.3113e-04 - val_loss: 8.9126e-04 - val_mean_squared_error: 8.9126e-04
    Epoch 59/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.3106e-04 - mean_squared_error: 8.3106e-04
    Epoch 00059: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.3075e-04 - mean_squared_error: 8.3075e-04 - val_loss: 9.5625e-04 - val_mean_squared_error: 9.5625e-04
    Epoch 60/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.2567e-04 - mean_squared_error: 8.2567e-04
    Epoch 00060: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.2770e-04 - mean_squared_error: 8.2770e-04 - val_loss: 9.4038e-04 - val_mean_squared_error: 9.4038e-04
    Epoch 61/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.1007e-04 - mean_squared_error: 8.1007e-04
    Epoch 00061: val_loss improved from 0.00087 to 0.00086, saving model to my_model.h5
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.1010e-04 - mean_squared_error: 8.1010e-04 - val_loss: 8.6403e-04 - val_mean_squared_error: 8.6403e-04
    Epoch 62/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.2260e-04 - mean_squared_error: 8.2260e-04
    Epoch 00062: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.2232e-04 - mean_squared_error: 8.2232e-04 - val_loss: 9.6868e-04 - val_mean_squared_error: 9.6868e-04
    Epoch 63/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.2189e-04 - mean_squared_error: 8.2189e-04
    Epoch 00063: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.2222e-04 - mean_squared_error: 8.2222e-04 - val_loss: 9.4699e-04 - val_mean_squared_error: 9.4699e-04
    Epoch 64/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.1065e-04 - mean_squared_error: 8.1065e-04
    Epoch 00064: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.1055e-04 - mean_squared_error: 8.1055e-04 - val_loss: 9.6557e-04 - val_mean_squared_error: 9.6557e-04
    Epoch 65/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.9932e-04 - mean_squared_error: 7.9932e-04
    Epoch 00065: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.9911e-04 - mean_squared_error: 7.9911e-04 - val_loss: 9.3399e-04 - val_mean_squared_error: 9.3399e-04
    Epoch 66/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.0644e-04 - mean_squared_error: 8.0644e-04
    Epoch 00066: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.0688e-04 - mean_squared_error: 8.0688e-04 - val_loss: 8.7945e-04 - val_mean_squared_error: 8.7945e-04
    Epoch 67/150
    1710/1712 [============================>.] - ETA: 0s - loss: 8.0291e-04 - mean_squared_error: 8.0291e-04
    Epoch 00067: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 8.0257e-04 - mean_squared_error: 8.0257e-04 - val_loss: 9.5382e-04 - val_mean_squared_error: 9.5382e-04
    Epoch 68/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.9904e-04 - mean_squared_error: 7.9904e-04
    Epoch 00068: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.9923e-04 - mean_squared_error: 7.9923e-04 - val_loss: 8.7462e-04 - val_mean_squared_error: 8.7462e-04
    Epoch 69/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.8559e-04 - mean_squared_error: 7.8559e-04
    Epoch 00069: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.8514e-04 - mean_squared_error: 7.8514e-04 - val_loss: 8.9376e-04 - val_mean_squared_error: 8.9376e-04
    Epoch 70/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.8798e-04 - mean_squared_error: 7.8798e-04
    Epoch 00070: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.8780e-04 - mean_squared_error: 7.8780e-04 - val_loss: 8.9764e-04 - val_mean_squared_error: 8.9764e-04
    Epoch 71/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.9015e-04 - mean_squared_error: 7.9015e-04
    Epoch 00071: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.9033e-04 - mean_squared_error: 7.9033e-04 - val_loss: 9.6156e-04 - val_mean_squared_error: 9.6156e-04
    Epoch 72/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.6726e-04 - mean_squared_error: 7.6726e-04
    Epoch 00072: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.6670e-04 - mean_squared_error: 7.6670e-04 - val_loss: 8.8530e-04 - val_mean_squared_error: 8.8530e-04
    Epoch 73/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.7724e-04 - mean_squared_error: 7.7724e-04
    Epoch 00073: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.7775e-04 - mean_squared_error: 7.7775e-04 - val_loss: 9.3207e-04 - val_mean_squared_error: 9.3207e-04
    Epoch 74/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.6453e-04 - mean_squared_error: 7.6453e-04
    Epoch 00074: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.6511e-04 - mean_squared_error: 7.6511e-04 - val_loss: 9.3530e-04 - val_mean_squared_error: 9.3530e-04
    Epoch 75/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.8305e-04 - mean_squared_error: 7.8305e-04
    Epoch 00075: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.8299e-04 - mean_squared_error: 7.8299e-04 - val_loss: 9.3432e-04 - val_mean_squared_error: 9.3432e-04
    Epoch 76/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.6366e-04 - mean_squared_error: 7.6366e-04
    Epoch 00076: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.6336e-04 - mean_squared_error: 7.6336e-04 - val_loss: 9.1081e-04 - val_mean_squared_error: 9.1081e-04
    Epoch 77/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.4952e-04 - mean_squared_error: 7.4952e-04
    Epoch 00077: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.4979e-04 - mean_squared_error: 7.4979e-04 - val_loss: 9.3724e-04 - val_mean_squared_error: 9.3724e-04
    Epoch 78/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.6838e-04 - mean_squared_error: 7.6838e-04
    Epoch 00078: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.6825e-04 - mean_squared_error: 7.6825e-04 - val_loss: 9.2801e-04 - val_mean_squared_error: 9.2801e-04
    Epoch 79/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.6189e-04 - mean_squared_error: 7.6189e-04
    Epoch 00079: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.6135e-04 - mean_squared_error: 7.6135e-04 - val_loss: 8.9637e-04 - val_mean_squared_error: 8.9637e-04
    Epoch 80/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.6507e-04 - mean_squared_error: 7.6507e-04
    Epoch 00080: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.6466e-04 - mean_squared_error: 7.6466e-04 - val_loss: 9.4925e-04 - val_mean_squared_error: 9.4925e-04
    Epoch 81/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.3877e-04 - mean_squared_error: 7.3877e-04
    Epoch 00081: val_loss did not improve
    1712/1712 [==============================] - 7s 4ms/step - loss: 7.3846e-04 - mean_squared_error: 7.3846e-04 - val_loss: 9.2804e-04 - val_mean_squared_error: 9.2804e-04
    Epoch 82/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.4044e-04 - mean_squared_error: 7.4044e-04
    Epoch 00082: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.4069e-04 - mean_squared_error: 7.4069e-04 - val_loss: 9.2612e-04 - val_mean_squared_error: 9.2612e-04
    Epoch 83/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.4969e-04 - mean_squared_error: 7.4969e-04
    Epoch 00083: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.4958e-04 - mean_squared_error: 7.4958e-04 - val_loss: 9.4365e-04 - val_mean_squared_error: 9.4365e-04
    Epoch 84/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.5265e-04 - mean_squared_error: 7.5265e-04
    Epoch 00084: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.5245e-04 - mean_squared_error: 7.5245e-04 - val_loss: 9.6419e-04 - val_mean_squared_error: 9.6419e-04
    Epoch 85/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.4621e-04 - mean_squared_error: 7.4621e-04
    Epoch 00085: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.4627e-04 - mean_squared_error: 7.4627e-04 - val_loss: 9.3654e-04 - val_mean_squared_error: 9.3654e-04
    Epoch 86/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.5973e-04 - mean_squared_error: 7.5973e-04
    Epoch 00086: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.5990e-04 - mean_squared_error: 7.5990e-04 - val_loss: 9.6164e-04 - val_mean_squared_error: 9.6164e-04
    Epoch 87/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.2044e-04 - mean_squared_error: 7.2044e-04
    Epoch 00087: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.2027e-04 - mean_squared_error: 7.2027e-04 - val_loss: 0.0010 - val_mean_squared_error: 0.0010
    Epoch 88/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.3357e-04 - mean_squared_error: 7.3357e-04
    Epoch 00088: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.3347e-04 - mean_squared_error: 7.3347e-04 - val_loss: 9.6675e-04 - val_mean_squared_error: 9.6675e-04
    Epoch 89/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.1227e-04 - mean_squared_error: 7.1227e-04
    Epoch 00089: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.1181e-04 - mean_squared_error: 7.1181e-04 - val_loss: 9.5535e-04 - val_mean_squared_error: 9.5535e-04
    Epoch 90/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.1818e-04 - mean_squared_error: 7.1818e-04
    Epoch 00090: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.1807e-04 - mean_squared_error: 7.1807e-04 - val_loss: 9.3722e-04 - val_mean_squared_error: 9.3722e-04
    Epoch 91/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.4150e-04 - mean_squared_error: 7.4150e-04
    Epoch 00091: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.4111e-04 - mean_squared_error: 7.4111e-04 - val_loss: 9.8211e-04 - val_mean_squared_error: 9.8211e-04
    Epoch 92/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.2158e-04 - mean_squared_error: 7.2158e-04
    Epoch 00092: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.2136e-04 - mean_squared_error: 7.2136e-04 - val_loss: 8.7982e-04 - val_mean_squared_error: 8.7982e-04
    Epoch 93/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.1737e-04 - mean_squared_error: 7.1737e-04
    Epoch 00093: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.1701e-04 - mean_squared_error: 7.1701e-04 - val_loss: 9.0866e-04 - val_mean_squared_error: 9.0866e-04
    Epoch 94/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.1859e-04 - mean_squared_error: 7.1859e-04
    Epoch 00094: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.1813e-04 - mean_squared_error: 7.1813e-04 - val_loss: 9.3907e-04 - val_mean_squared_error: 9.3907e-04
    Epoch 95/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.3060e-04 - mean_squared_error: 7.3060e-04
    Epoch 00095: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.3033e-04 - mean_squared_error: 7.3033e-04 - val_loss: 9.2060e-04 - val_mean_squared_error: 9.2060e-04
    Epoch 96/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.1726e-04 - mean_squared_error: 7.1726e-04
    Epoch 00096: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.1688e-04 - mean_squared_error: 7.1688e-04 - val_loss: 9.5925e-04 - val_mean_squared_error: 9.5925e-04
    Epoch 97/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.1581e-04 - mean_squared_error: 7.1581e-04
    Epoch 00097: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.1555e-04 - mean_squared_error: 7.1555e-04 - val_loss: 9.2640e-04 - val_mean_squared_error: 9.2640e-04
    Epoch 98/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.2239e-04 - mean_squared_error: 7.2239e-04
    Epoch 00098: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.2237e-04 - mean_squared_error: 7.2237e-04 - val_loss: 9.7802e-04 - val_mean_squared_error: 9.7802e-04
    Epoch 99/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.0291e-04 - mean_squared_error: 7.0291e-04
    Epoch 00099: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.0242e-04 - mean_squared_error: 7.0242e-04 - val_loss: 9.3400e-04 - val_mean_squared_error: 9.3400e-04
    Epoch 100/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.9874e-04 - mean_squared_error: 6.9874e-04
    Epoch 00100: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.9854e-04 - mean_squared_error: 6.9854e-04 - val_loss: 9.1090e-04 - val_mean_squared_error: 9.1090e-04
    Epoch 101/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.0764e-04 - mean_squared_error: 7.0764e-04
    Epoch 00101: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.0815e-04 - mean_squared_error: 7.0815e-04 - val_loss: 9.1108e-04 - val_mean_squared_error: 9.1108e-04
    Epoch 102/150
    1710/1712 [============================>.] - ETA: 0s - loss: 7.0966e-04 - mean_squared_error: 7.0966e-04
    Epoch 00102: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 7.1083e-04 - mean_squared_error: 7.1083e-04 - val_loss: 9.1007e-04 - val_mean_squared_error: 9.1007e-04
    Epoch 103/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.8877e-04 - mean_squared_error: 6.8877e-04
    Epoch 00103: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.9118e-04 - mean_squared_error: 6.9118e-04 - val_loss: 9.2719e-04 - val_mean_squared_error: 9.2719e-04
    Epoch 104/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.9892e-04 - mean_squared_error: 6.9892e-04
    Epoch 00104: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.9876e-04 - mean_squared_error: 6.9876e-04 - val_loss: 8.9767e-04 - val_mean_squared_error: 8.9767e-04
    Epoch 105/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.6697e-04 - mean_squared_error: 6.6697e-04
    Epoch 00105: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.6670e-04 - mean_squared_error: 6.6670e-04 - val_loss: 8.8853e-04 - val_mean_squared_error: 8.8853e-04
    Epoch 106/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.8908e-04 - mean_squared_error: 6.8908e-04
    Epoch 00106: val_loss improved from 0.00086 to 0.00085, saving model to my_model.h5
    1712/1712 [==============================] - 7s 4ms/step - loss: 6.8930e-04 - mean_squared_error: 6.8930e-04 - val_loss: 8.5427e-04 - val_mean_squared_error: 8.5427e-04
    Epoch 107/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.7788e-04 - mean_squared_error: 6.7788e-04
    Epoch 00107: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.7784e-04 - mean_squared_error: 6.7784e-04 - val_loss: 9.4612e-04 - val_mean_squared_error: 9.4612e-04
    Epoch 108/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.8575e-04 - mean_squared_error: 6.8575e-04
    Epoch 00108: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.8522e-04 - mean_squared_error: 6.8522e-04 - val_loss: 8.9443e-04 - val_mean_squared_error: 8.9443e-04
    Epoch 109/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.7731e-04 - mean_squared_error: 6.7731e-04
    Epoch 00109: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.7744e-04 - mean_squared_error: 6.7744e-04 - val_loss: 9.1699e-04 - val_mean_squared_error: 9.1699e-04
    Epoch 110/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.7548e-04 - mean_squared_error: 6.7548e-04
    Epoch 00110: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.7540e-04 - mean_squared_error: 6.7540e-04 - val_loss: 9.1296e-04 - val_mean_squared_error: 9.1296e-04
    Epoch 111/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.7776e-04 - mean_squared_error: 6.7776e-04
    Epoch 00111: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.7746e-04 - mean_squared_error: 6.7746e-04 - val_loss: 9.1693e-04 - val_mean_squared_error: 9.1693e-04
    Epoch 112/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.9075e-04 - mean_squared_error: 6.9075e-04
    Epoch 00112: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.9072e-04 - mean_squared_error: 6.9072e-04 - val_loss: 9.3198e-04 - val_mean_squared_error: 9.3198e-04
    Epoch 113/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.8897e-04 - mean_squared_error: 6.8897e-04
    Epoch 00113: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.8906e-04 - mean_squared_error: 6.8906e-04 - val_loss: 9.7197e-04 - val_mean_squared_error: 9.7197e-04
    Epoch 114/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.9713e-04 - mean_squared_error: 6.9713e-04
    Epoch 00114: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.9710e-04 - mean_squared_error: 6.9710e-04 - val_loss: 9.1725e-04 - val_mean_squared_error: 9.1725e-04
    Epoch 115/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.8423e-04 - mean_squared_error: 6.8423e-04
    Epoch 00115: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.8395e-04 - mean_squared_error: 6.8395e-04 - val_loss: 8.9683e-04 - val_mean_squared_error: 8.9683e-04
    Epoch 116/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.7796e-04 - mean_squared_error: 6.7796e-04
    Epoch 00116: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.7772e-04 - mean_squared_error: 6.7772e-04 - val_loss: 9.2488e-04 - val_mean_squared_error: 9.2488e-04
    Epoch 117/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.8417e-04 - mean_squared_error: 6.8417e-04
    Epoch 00117: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.8409e-04 - mean_squared_error: 6.8409e-04 - val_loss: 9.6227e-04 - val_mean_squared_error: 9.6227e-04
    Epoch 118/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.8657e-04 - mean_squared_error: 6.8657e-04
    Epoch 00118: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.8607e-04 - mean_squared_error: 6.8607e-04 - val_loss: 9.1574e-04 - val_mean_squared_error: 9.1574e-04
    Epoch 119/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.6314e-04 - mean_squared_error: 6.6314e-04
    Epoch 00119: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.6358e-04 - mean_squared_error: 6.6358e-04 - val_loss: 9.3889e-04 - val_mean_squared_error: 9.3889e-04
    Epoch 120/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.8553e-04 - mean_squared_error: 6.8553e-04
    Epoch 00120: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.8552e-04 - mean_squared_error: 6.8552e-04 - val_loss: 9.6355e-04 - val_mean_squared_error: 9.6355e-04
    Epoch 121/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.8601e-04 - mean_squared_error: 6.8601e-04
    Epoch 00121: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.8586e-04 - mean_squared_error: 6.8586e-04 - val_loss: 9.2427e-04 - val_mean_squared_error: 9.2427e-04
    Epoch 122/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.7276e-04 - mean_squared_error: 6.7276e-04
    Epoch 00122: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.7246e-04 - mean_squared_error: 6.7246e-04 - val_loss: 9.3771e-04 - val_mean_squared_error: 9.3771e-04
    Epoch 123/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.6804e-04 - mean_squared_error: 6.6804e-04
    Epoch 00123: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.6811e-04 - mean_squared_error: 6.6811e-04 - val_loss: 9.7802e-04 - val_mean_squared_error: 9.7802e-04
    Epoch 124/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.7153e-04 - mean_squared_error: 6.7153e-04
    Epoch 00124: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.7136e-04 - mean_squared_error: 6.7136e-04 - val_loss: 9.9882e-04 - val_mean_squared_error: 9.9882e-04
    Epoch 125/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.7450e-04 - mean_squared_error: 6.7450e-04
    Epoch 00125: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.7492e-04 - mean_squared_error: 6.7492e-04 - val_loss: 9.0998e-04 - val_mean_squared_error: 9.0998e-04
    Epoch 126/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.6448e-04 - mean_squared_error: 6.6448e-04
    Epoch 00126: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.6430e-04 - mean_squared_error: 6.6430e-04 - val_loss: 9.1633e-04 - val_mean_squared_error: 9.1633e-04
    Epoch 127/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.5611e-04 - mean_squared_error: 6.5611e-04
    Epoch 00127: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.5637e-04 - mean_squared_error: 6.5637e-04 - val_loss: 9.2322e-04 - val_mean_squared_error: 9.2322e-04
    Epoch 128/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.4502e-04 - mean_squared_error: 6.4502e-04
    Epoch 00128: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.4483e-04 - mean_squared_error: 6.4483e-04 - val_loss: 9.0728e-04 - val_mean_squared_error: 9.0728e-04
    Epoch 129/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.7244e-04 - mean_squared_error: 6.7244e-04
    Epoch 00129: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.7247e-04 - mean_squared_error: 6.7247e-04 - val_loss: 9.3125e-04 - val_mean_squared_error: 9.3125e-04
    Epoch 130/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.7038e-04 - mean_squared_error: 6.7038e-04
    Epoch 00130: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.7025e-04 - mean_squared_error: 6.7025e-04 - val_loss: 9.3859e-04 - val_mean_squared_error: 9.3859e-04
    Epoch 131/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.5956e-04 - mean_squared_error: 6.5956e-04
    Epoch 00131: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.5955e-04 - mean_squared_error: 6.5955e-04 - val_loss: 9.4063e-04 - val_mean_squared_error: 9.4063e-04
    Epoch 132/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.5999e-04 - mean_squared_error: 6.5999e-04
    Epoch 00132: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.5960e-04 - mean_squared_error: 6.5960e-04 - val_loss: 9.7809e-04 - val_mean_squared_error: 9.7809e-04
    Epoch 133/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.4183e-04 - mean_squared_error: 6.4183e-04
    Epoch 00133: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.4247e-04 - mean_squared_error: 6.4247e-04 - val_loss: 8.9854e-04 - val_mean_squared_error: 8.9854e-04
    Epoch 134/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.5434e-04 - mean_squared_error: 6.5434e-04
    Epoch 00134: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.5437e-04 - mean_squared_error: 6.5437e-04 - val_loss: 8.9247e-04 - val_mean_squared_error: 8.9247e-04
    Epoch 135/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.6530e-04 - mean_squared_error: 6.6530e-04
    Epoch 00135: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.6504e-04 - mean_squared_error: 6.6504e-04 - val_loss: 9.4996e-04 - val_mean_squared_error: 9.4996e-04
    Epoch 136/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.4182e-04 - mean_squared_error: 6.4182e-04
    Epoch 00136: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.4155e-04 - mean_squared_error: 6.4155e-04 - val_loss: 9.2103e-04 - val_mean_squared_error: 9.2103e-04
    Epoch 137/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.4714e-04 - mean_squared_error: 6.4714e-04
    Epoch 00137: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.4699e-04 - mean_squared_error: 6.4699e-04 - val_loss: 9.2445e-04 - val_mean_squared_error: 9.2445e-04
    Epoch 138/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.5459e-04 - mean_squared_error: 6.5459e-04
    Epoch 00138: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.5424e-04 - mean_squared_error: 6.5424e-04 - val_loss: 9.8299e-04 - val_mean_squared_error: 9.8299e-04
    Epoch 139/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.3604e-04 - mean_squared_error: 6.3604e-04
    Epoch 00139: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.3607e-04 - mean_squared_error: 6.3607e-04 - val_loss: 8.8083e-04 - val_mean_squared_error: 8.8083e-04
    Epoch 140/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.4849e-04 - mean_squared_error: 6.4849e-04
    Epoch 00140: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.4866e-04 - mean_squared_error: 6.4866e-04 - val_loss: 9.5251e-04 - val_mean_squared_error: 9.5251e-04
    Epoch 141/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.6012e-04 - mean_squared_error: 6.6012e-04
    Epoch 00141: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.6008e-04 - mean_squared_error: 6.6008e-04 - val_loss: 9.6635e-04 - val_mean_squared_error: 9.6635e-04
    Epoch 142/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.4259e-04 - mean_squared_error: 6.4259e-04
    Epoch 00142: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.4262e-04 - mean_squared_error: 6.4262e-04 - val_loss: 9.1854e-04 - val_mean_squared_error: 9.1854e-04
    Epoch 143/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.5109e-04 - mean_squared_error: 6.5109e-04
    Epoch 00143: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.5101e-04 - mean_squared_error: 6.5101e-04 - val_loss: 9.3132e-04 - val_mean_squared_error: 9.3132e-04
    Epoch 144/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.3770e-04 - mean_squared_error: 6.3770e-04
    Epoch 00144: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.3755e-04 - mean_squared_error: 6.3755e-04 - val_loss: 8.9680e-04 - val_mean_squared_error: 8.9680e-04
    Epoch 145/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.3651e-04 - mean_squared_error: 6.3651e-04
    Epoch 00145: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.3721e-04 - mean_squared_error: 6.3721e-04 - val_loss: 9.4674e-04 - val_mean_squared_error: 9.4674e-04
    Epoch 146/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.3984e-04 - mean_squared_error: 6.3984e-04
    Epoch 00146: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.3971e-04 - mean_squared_error: 6.3971e-04 - val_loss: 9.0549e-04 - val_mean_squared_error: 9.0549e-04
    Epoch 147/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.2797e-04 - mean_squared_error: 6.2797e-04
    Epoch 00147: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.2792e-04 - mean_squared_error: 6.2792e-04 - val_loss: 9.0125e-04 - val_mean_squared_error: 9.0125e-04
    Epoch 148/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.2411e-04 - mean_squared_error: 6.2411e-04
    Epoch 00148: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.2387e-04 - mean_squared_error: 6.2387e-04 - val_loss: 9.7133e-04 - val_mean_squared_error: 9.7133e-04
    Epoch 149/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.4295e-04 - mean_squared_error: 6.4295e-04
    Epoch 00149: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.4254e-04 - mean_squared_error: 6.4254e-04 - val_loss: 9.2861e-04 - val_mean_squared_error: 9.2861e-04
    Epoch 150/150
    1710/1712 [============================>.] - ETA: 0s - loss: 6.5297e-04 - mean_squared_error: 6.5297e-04
    Epoch 00150: val_loss did not improve
    1712/1712 [==============================] - 6s 4ms/step - loss: 6.5320e-04 - mean_squared_error: 6.5320e-04 - val_loss: 9.4256e-04 - val_mean_squared_error: 9.4256e-04


---
<a id='step7'></a>

## Step 7: Visualize the Loss and Test Predictions

### (IMPLEMENTATION)  Answer a few questions and visualize the loss

__Question 1:__ Outline the steps you took to get to your final neural network architecture and your reasoning at each step.

__Answer:__

I started with a very simple network using SGD optimizer and ReLu activation functions.

<pre>
conv2d_3 (Conv2D)            (None, 96, 96, 32)
max_pooling2d_3 (MaxPooling) (None, 48, 48, 32)
dense_5 (Dense)              (None, 30)
</pre>

The validation loss is not bad but after 120 epochs we notice some overfitting and the network is not able to learn anymore. The network doesn't seem complex enough for a dropout layer to be beneficial.

<table>
    <tr>
        <td style="text-align:left;">
            <b>sgd</b> 
            <img src="test_models/model_loss_1_sgd.png" width=300 height=300/>
        </td>
    </tr>
</table>

I tried a more complex network using SGD optimizer.

<pre>
conv2d_3 (Conv2D)            (None, 96, 96, 32)
max_pooling2d_3 (MaxPooling) (None, 48, 48, 32)
conv2d_4 (Conv2D)            (None, 48, 48, 64)
max_pooling2d_4 (MaxPooling) (None, 24, 24, 64)
conv2d_5 (Conv2D)            (None, 24, 24, 128)
max_pooling2d_5 (MaxPooling) (None, 12, 12, 128)
flatten_3 (Flatten)          (None, 18432)
dense_4 (Dense)              (None, 500)
dense_5 (Dense)              (None, 30)
</pre>

The result is not as good as before but we can see that using SGD at 150 epochs the networks is still learning, unfortunately too slowly. For this reason I decided to try a different optimizer, RMSprop.

<table>
    <tr>
        <td style="text-align:left;">
            <b>SGD</b>
            <img src="test_models/model_loss_2_sgd.png" width=300 height=300/>
        </td>
        <td style="text-align:left;">
            <b>RMSprop</b> 
            <img src="test_models/model_loss_2_rmsprop.png" width=300 height=300/>
        </td>
    </tr>
    
</table>

With RMSprop the network is learning much faster and the validation loss has improved but Again we can see some overfitting so I decided to add some dropout layers.
 
<pre>
conv2d_21 (Conv2D)           (None, 96, 96, 32)
max_pooling2d_20 (MaxPooling (None, 48, 48, 32)
conv2d_22 (Conv2D)           (None, 48, 48, 64)
max_pooling2d_21 (MaxPooling (None, 24, 24, 64)
conv2d_23 (Conv2D)           (None, 24, 24, 128)
max_pooling2d_22 (MaxPooling (None, 12, 12, 128)
flatten_8 (Flatten)          (None, 18432)
dropout_11 (Dropout)         (None, 18432)
dense_14 (Dense)             (None, 500)
dropout_12 (Dropout)         (None, 500)
dense_15 (Dense)             (None, 30)
</pre>

<table>
    <tr>
        <td style="text-align:left;">
            <b>RMSprop</b>
            <img src="test_models/model_loss_3_rmsprop-dropout04.png" width=300 height=300/>
        </td>
    </tr>
</table>

Overfitting has been almost removed but there is no improvement in the validation loss and the network doesn't seem to learn anymore. 

Two options should be explored: increasing again the network complexity or increase the amount of training data.

The validation loss is 0.00094 which seems good enough so I will now explore different optimization algorithms.

__Question 2:__ Defend your choice of optimizer.  Which optimizers did you test, and how did you determine which worked best?

__Answer:__ 

I have explored different optimizers against the latest network to look for an improvement in both validation loss and in learning speed.

<table>
    <tr>
        <td style="text-align:left;">
            <b>RMSprop</b>
            <img src="test_models/model_loss_3_rmsprop-dropout04.png" width=300 height=300/>
        </td>
        <td style="text-align:left;">
            <b>Adam</b>
            <img src="test_models/model_loss_3_adam-dropout04.png" width=300 height=300/>
        </td>
    </tr>
    <tr>
        <td style="text-align:left;">
            <b>Adagrad</b>
            <img src="test_models/model_loss_3_adagrad-dropout04.png" width=300 height=300/>
        </td>
        <td style="text-align:left;">
            <b>Adamax</b>
            <img src="test_models/model_loss_3_adamax-dropout04.png" width=300 height=300/>
        </td>
    </tr>
    <tr>
        <td style="text-align:left;">
            <b>Adadelta</b>
            <img src="test_models/model_loss_3_adadelta-dropout04.png" width=300 height=300/>
        </td>
        <td style="text-align:left;">
            <b>Nadam</b>
            <img src="test_models/model_loss_3_nadam-dropout04.png" width=300 height=300/>
        </td>
    </tr>
</table>

<b>Adamax</b> is the optimizer which gives the best validation loss and this is the optimizer of choice to train the final network.

<b>Adagrad</b> is the optimizer which learns faster providing close to optimal validation loss in less than 20 epochs. This is an excellent optimizer to experiment with different network layouts before eventually using Adamax.  

Use the code cell below to plot the training and validation loss of your neural network.  You may find [this resource](http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/) useful.


```python
# Visualize the training and validation loss of your neural network
# list all data in history

best_t_loss = round(min(history.history['loss']), 5)
best_v_loss = round(min(history.history['val_loss']), 5)

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.text(40, 0.0089, 'best validation loss:' + str(best_v_loss))
plt.title('model loss')
plt.ylabel('loss')
plt.ylim((0,0.01))
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()
```


<img src="images/output_64_0.png"/>


__Question 3:__  Do you notice any evidence of overfitting or underfitting in the above plot?  If so, what steps have you taken to improve your model?  Note that slight overfitting or underfitting will not hurt your chances of a successful submission, as long as you have attempted some solutions towards improving your model (such as _regularization, dropout, increased/decreased number of layers, etc_).

__Answer:__

There is a little bit of overfitting after epoch 60. As explained in previous questions there was much more overfitting previously which has been fixed adding two DropOut layers. This didn't improve the validation loss though but opened the doors to more possibilities like increasing the network complexity or adding more training data.

I've also tried BarchNormalization between the convolutional layers but I haven't obtained any valuable improvement.

### Visualize a Subset of the Test Predictions

Execute the code cell below to visualize your model's predicted keypoints on a subset of the testing images.


```python
# Load the best model.
model.load_weights("my_model.h5")

y_test = model.predict(X_test)

fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
    plot_data(X_test[i], y_test[i], ax)
```


<img src="images/output_67_0.png"/>


---
<a id='step8'></a>

## Step 8: Complete the pipeline

With the work you did in Sections 1 and 2 of this notebook, along with your freshly trained facial keypoint detector, you can now complete the full pipeline.  That is given a color image containing a person or persons you can now 

- Detect the faces in this image automatically using OpenCV
- Predict the facial keypoints in each face detected in the image
- Paint predicted keypoints on each face detected

In this Subsection you will do just this!  

### (IMPLEMENTATION) Facial Keypoints Detector

Use the OpenCV face detection functionality you built in previous Sections to expand the functionality of your keypoints detector to color images with arbitrary size.  Your function should perform the following steps

1. Accept a color image.
2. Convert the image to grayscale.
3. Detect and crop the face contained in the image.
4. Locate the facial keypoints in the cropped image.
5. Overlay the facial keypoints in the original (color, uncropped) image.

**Note**: step 4 can be the trickiest because remember your convolutional network is only trained to detect facial keypoints in $96 \times 96$ grayscale images where each pixel was normalized to lie in the interval $[0,1]$, and remember that each facial keypoint was normalized during training to the interval $[-1,1]$.  This means - practically speaking - to paint detected keypoints onto a test face you need to perform this same pre-processing to your candidate face  - that is after detecting it you should resize it to $96 \times 96$ and normalize its values before feeding it into your facial keypoint detector.  To be shown correctly on the original image the output keypoints from your detector then need to be shifted and re-normalized from the interval $[-1,1]$ to the width and height of your detected face.

When complete you should be able to produce example images like the one below

<img src="images/obamas_with_keypoints.png"/>


```python
# Load in color image for face detection
image = cv2.imread('images/obamas4.jpg')

# Convert the image to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plot our image
fig = plt.figure(figsize = (9,9))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('image')
ax1.imshow(image);
```


<img src="images/output_70_0.png"/>



```python
# Use the face detection code we saw in Section 1 with your trained conv-net 
# Paint the predicted keypoints on the test image

image_copy = np.copy(image)

face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# Draw landmarks on detected faces.
def detect_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)

    # Iterate through all detected faces.
    for (x,y,w,h) in faces:

        # Add a red bounding box around the face.
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)

        # Extract the region of interest and resize it to a 96x96 monochrome image.
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (96, 96)) 

        # Input normalization.
        X = np.copy(roi_gray) / 255
        X = X.astype(np.float32)
        X = X.reshape(-1, 96, 96, 1)

        # Run the prediction.
        landmarks = model.predict(X)

        # Convert to points, undo normalization and offset.
        points = list(zip(landmarks[0][0::2], landmarks[0][1::2]))
        points = [(int(x + (1 + px) * w / 2), int(y + (1 + py) * h / 2)) for (px, py) in points]

        # Draw landmarks over image with OpenCV.
        for point in points:
            cv2.circle(image, center=(point[0], point[1]), radius=4, color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)
    
detect_landmarks(image_copy)    
    
# Display the main image with all detected features.   
# This is the main image.
fig = plt.figure(figsize = (9,9))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('image_copy')
ax1.imshow(image_copy);
```


<img src="images/output_71_0.png"/>


### (Optional) Further Directions - add a filter using facial keypoints to your laptop camera

Now you can add facial keypoint detection to your laptop camera - as illustrated in the gif below.

<img src="images/facial_keypoint_test.gif"/>

The next Python cell contains the basic laptop video camera function used in the previous optional video exercises.  Combine it with the functionality you developed for keypoint detection and marking in the previous exercise and you should be good to go!


```python
import cv2
import time 
from keras.models import load_model

def laptop_camera_go():
    # Create instance of video capturer
    cv2.namedWindow("face detection activated")
    vc = cv2.VideoCapture(0)

    # Try to get the first frame
    if vc.isOpened(): 
        rval, frame = vc.read()
    else:
        rval = False
    
    # keep video stream open
    while rval:

        detect_landmarks(frame)

        # plot image from camera with detections marked
        cv2.imshow("face detection activated", frame)    
            
        # exit functionality - press any key to exit laptop video
        key = cv2.waitKey(20)
        if key > 0: # exit by pressing any key
            # destroy windows
            cv2.destroyAllWindows()
            
            # hack from stack overflow for making sure window closes on osx --> https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv
            for i in range (1,5):
                cv2.waitKey(1)
            return
        
        # read next frame
        time.sleep(0.05)             # control framerate for computation - default 20 frames per sec
        rval, frame = vc.read()  
```


```python
# Run your keypoint face painter
laptop_camera_go()
```

<img src="test_models/LandmarksDetection.gif"/>

### (Optional) Further Directions - add a filter using facial keypoints

Using your freshly minted facial keypoint detector pipeline you can now do things like add fun filters to a person's face automatically.  In this optional exercise you can play around with adding sunglasses automatically to each individual's face in an image as shown in a demonstration image below.

<img src="images/obamas_with_shades.png"/>

To produce this effect an image of a pair of sunglasses shown in the Python cell below.


```python
# Load in sunglasses image - note the usage of the special option
# cv2.IMREAD_UNCHANGED, this option is used because the sunglasses 
# image has a 4th channel that allows us to control how transparent each pixel in the image is
sunglasses = cv2.imread("images/sunglasses_4.png", cv2.IMREAD_UNCHANGED)

# Plot the image
fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.imshow(sunglasses)
ax1.axis('off');
```


<img src="images/output_77_0.png"/>


This image is placed over each individual's face using the detected eye points to determine the location of the sunglasses, and eyebrow points to determine the size that the sunglasses should be for each person (one could also use the nose point to determine this).  

Notice that this image actually has *4 channels*, not just 3. 


```python
# Print out the shape of the sunglasses image
print ('The sunglasses image has shape: ' + str(np.shape(sunglasses)))
```

    The sunglasses image has shape: (1123, 3064, 4)


It has the usual red, blue, and green channels any color image has, with the 4th channel representing the transparency level of each pixel in the image.  Here's how the transparency channel works: the lower the value, the more transparent the pixel will become.  The lower bound (completely transparent) is zero here, so any pixels set to 0 will not be seen. 

This is how we can place this image of sunglasses on someone's face and still see the area around of their face where the sunglasses lie - because these pixels in the sunglasses image have been made completely transparent.

Lets check out the alpha channel of our sunglasses image in the next Python cell.  Note because many of the pixels near the boundary are transparent we'll need to explicitly print out non-zero values if we want to see them. 


```python
# Print out the sunglasses transparency (alpha) channel
alpha_channel = sunglasses[:,:,3]
print ('the alpha channel here looks like')
print (alpha_channel)

# Just to double check that there are indeed non-zero values
# Let's find and print out every value greater than zero
values = np.where(alpha_channel != 0)
print ('\n the non-zero values of the alpha channel look like')
print (values)
```

    the alpha channel here looks like
    [[0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     ...
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]]
    
     the non-zero values of the alpha channel look like
    (array([  17,   17,   17, ..., 1109, 1109, 1109]), array([ 687,  688,  689, ..., 2376, 2377, 2378]))


This means that when we place this sunglasses image on top of another image, we can use the transparency channel as a filter to tell us which pixels to overlay on a new image (only the non-transparent ones with values greater than zero).

One last thing: it's helpful to understand which keypoint belongs to the eyes, mouth, etc. So, in the image below, we also display the index of each facial keypoint directly on the image so that you can tell which keypoints are for the eyes, eyebrows, etc.

<img src="images/obamas_points_numbered.png"/>

With this information, you're well on your way to completing this filtering task!  See if you can place the sunglasses automatically on the individuals in the image loaded in / shown in the next Python cell.


```python
## (Optional) TODO: Use the face detection code we saw in Section 1 with your trained conv-net to put
## sunglasses on the individuals in our test image


```


```python
# Load in color image for face detection
image = cv2.imread('images/obamas4.jpg')

# Convert the image to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
# Function used to place a foreground image (fg), fitting the area (x, y, x + w, y + h),
# over a background image (bg).
def overlay_image(x, y, w, h, bg, fg):
    x, y, w, h = int(x), int(y), int(w), int(h)
    if (x >= 0 and y >= 0 and x + w < bg.shape[1] and y + h < bg.shape[0]):
        fg = cv2.resize(fg, (w, h))
        rgb = fg[:,:,:3]         # RGB channels
        alpha = fg[:,:,3:] / 255 # Alpha        
        bg[y:y + h, x:x + w] = bg[y:y + h, x:x + w] * (1 - alpha) + rgb * (alpha)

# Find all faces in the given face and overlay a picture of glasses
# on each of them
def overlay_glasses(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)
    for (x,y,w,h) in faces:

        # Extract the region of interest and resize it to a 96x96 monochrome image.
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (96, 96))

        # Input normalization.
        X = np.copy(roi_gray) / 255
        X = X.astype(np.float32)
        X = X.reshape(-1, 96, 96, 1)

        # Run the prediction.
        landmarks = model.predict(X)

        # Convert to points, undo normalization and offset.
        points = list(zip(landmarks[0][0::2], landmarks[0][1::2]))
        points = [(int(x + (1 + px) * w / 2), int(y + (1 + py) * h / 2)) for (px, py) in points]

        # Find center of nose averaging points 8, 6 and 10
        nx = int((points[8][0] + points[6][0] + points[10][0]) / 3)
        ny = int((points[8][1] + points[6][1] + points[10][1]) / 3)

        # Calculate width and weight of glasses using points 9 and 7 and a scale factor.
        glasses_scale = 1.2
        gw = glasses_scale * (points[7][0] - points[9][0])
        gh = gw * sunglasses.shape[0] / sunglasses.shape[1]

        # Position glasses on the nose.
        gx = int(nx - 0.5 * gw)
        gy = int(ny - 0.5 * gh)

        # Overlay the glasses over the picture.
        overlay_image(gx, gy, gw, gh, image, sunglasses)
 
overlay_glasses(image)
    
# Plot the image
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Original Image')
ax1.imshow(image);
```


<img src="images/output_84_0.png"/>


###  (Optional) Further Directions - add a filter using facial keypoints to your laptop camera 

Now you can add the sunglasses filter to your laptop camera - as illustrated in the gif below.

<img src="images/mr_sunglasses.gif" width=250 height=250/>

The next Python cell contains the basic laptop video camera function used in the previous optional video exercises.  Combine it with the functionality you developed for adding sunglasses to someone's face in the previous optional exercise and you should be good to go!


```python
import cv2
import time 
from keras.models import load_model

def laptop_camera_go():
    # Create instance of video capturer
    cv2.namedWindow("face detection activated")
    vc = cv2.VideoCapture(0)

    # try to get the first frame
    if vc.isOpened(): 
        rval, frame = vc.read()
    else:
        rval = False
    
    # Keep video stream open
    while rval:
        
        overlay_glasses(frame)
        
        # Plot image from camera with detections marked
        cv2.imshow("face detection activated", frame)
        
        # Exit functionality - press any key to exit laptop video
        key = cv2.waitKey(20)
        if key > 0: # exit by pressing any key
            # Destroy windows 
            cv2.destroyAllWindows()
            
            for i in range (1,5):
                cv2.waitKey(1)
            return
        
        # Read next frame
        time.sleep(0.05)             # control framerate for computation - default 20 frames per sec
        rval, frame = vc.read()    
        
```


```python
# Load facial landmark detector model
model = load_model('my_model.h5')

# Run sunglasses painter
laptop_camera_go()
```

<img src="test_models/Glasses.gif"/>
