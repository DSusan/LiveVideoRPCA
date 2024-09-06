# LiveVideoRPCA
This work presents a complete analysis and implementation of Robust Principal Component Analysis (RPCA) for real-time anomaly detection. The focus of the research is on enhancing the performance of RPCA in live video streams, aiding in the effective detection of dynamic objects in diverse environments.

Moving object detection is of great importance in several applications, including video surveillance, autonomous driving, and human-computer interaction.
Traditional approaches typically get overwhelmed in cases of dynamic backgrounds or when lighting conditions are changing. RPCA has to compete in this field as a robust solution for decomposing a video sequence into low-rank background and sparse moving objects with minimal hardware requirements.
Video surveillance is an area in which moving object detection and tracking are extremely important not only for security but also operational efficiency. Tasks such as frame differencing and background subtraction methods are both relevant in this context.

The implementation proposed in this document, considers 2 scenarios:
- Pre-recorded Video: The algorithm will be tested on a predefined video file to serve as point of comparison with the real-time application.
- Real-Time: Continuous video streaming from a webcam camera. The algorithm will have to perform the processing in a short time span to be able to keep up with the constant input.
The programming language used in the implementation is Python.
Python has wide applications in techniques related to anomaly detection and moving object detection, which are supported by extensive libraries and tools. Then there are libraries like OpenCV that provide robust support to image and video processing. This greatly assisted in implementing complex algorithms like RPCA.
In particular, it supports rapid development and prototyping because of the ease and readability of Python. This enables researchers and developers to rapidly iterate and test ideas. This can become very critical in real-time systems where efficiency and accuracy matter most.

# Libraries
Notably, there are a number of efficient numerical and scientific libraries available within Python; the popular ones include NumPy and SciPy.
They guarantee efficient support for matrix operations and mathematical computations that at the core of algorithms like RPCA.

OpenCV
OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library designed for computational efficiency and with a strong attention on real-time applications.
The library has more than 2500 optimized algorithms, which can be used for various image and video processing tasks.
In this work, OpenCV will be used mainly for video capturing and preprocessing, as well as visualization and file saving.

Concurrent Futures
Concurrent futures in Python provide a high-level interface for asynchronously executing callables using threads or processes. In this work, the ThreadPoolExecutor from the Concurrent Futures module is utilized to handle the computational load efficiently and improve the real-time performance of the RPCA algorithm.
By using multiple threads, video frames are processed concurrently, which significantly speeds up the frame-by-frame analysis and reduces latency.
Memorial University of Newfoundland and Labrador
Master of Data Science
Moreover, concurrent futures simplify the management of multiple threads, handling the complexities of thread creation, synchronization, and termination internally. This leads to cleaner and more maintainable code, enabling the focus to remain on the core functionality of the detection system.

Queue
The Python Queue module provides a thread-safe way to manage a queue of items, making it suitable for concurrent programming. In this work, the Queue is used to handle the buffering.
It is important to ensure that frames are processed in the correct order and that there is no data corruption or race conditions when multiple threads access the queue simultaneously.
In order to address common issues, the Queue provides a blocking mechanism, which helps manage the flow of data. If the queue is full, the producer will wait, preventing memory overflow.
Finally, if the queue is empty, the consumer (the code accessing the queue) will wait, ensuring that processing threads are not working on incomplete data.

Pyplot
Pyplot is a comprehensive library in Python for creating static, animated, and interactive visualizations.
The flexibility of this library allows for a veery simple integration with OpenCV, as it can manage a cv2.VideoCapture() frame reading input, that updates at a desired frequency.
In this work, Pyplot is used primarily for visualizing the results of the RPCA algorithm, specifically the low-rank and sparse components extracted from the video frames.

NumPy
This is one of the most used packages for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.
In this project, NumPy is extensively used for various tasks related to data manipulation and numerical operations.
It supports essential tasks such as creating and reshaping arrays for processing video frames and performing mathematical operations like SVD for the RPCA algorithm.
Its ability to manage data representation and normalization, as well as its seamless integration with many other libraries, ensures accurate analysis and interpretation of results.

## General Class Structure
The RPCA class is structured to handle moving object detection for pre-recorded video and anomaly detection from real-time videos.
The __init__ method initializes the RPCA object and sets up the following:

Input Handling: The method checks if the input is a video stream (cv2.VideoCapture). If so, it captures the video frames, converts them to grayscale, and stacks them as columns in a matrix.
Memorial University of Newfoundland and Labrador

Attributes: The method sets up attributes such as self.video, self.frame, self.height, self.width, and self.channels to store video properties. It also initializes attributes for storing the low-rank and sparse matrices (self.L and self.S).

Buffering and Multithreading: The method sets up parameters for frame buffering (self.buffer_size, self.frame_buffer), thread pooling (self.executor), and queue management (self.frame_queue, self.result_queue). These are essential for managing real-time processing.
The RPCA Algorithm is applied by the fit Method.
Then, if the algorithm converges, the user may call one of the following functions depending on the context and nature of the input.

Display Results: The display_results method normalizes and visualizes the low-rank and sparse components of the processed frames. For video input, it creates videos from these components and displays the first frames using Matplotlib.

Real-Time Display: The display_realtime method shows the low-rank and sparse frames in real-time, updating the display as new frames are processed. It also performs anomaly detection by analyzing the sparse component.
Finally, the real-time RPCA execution is handle by the method realtime_rpca and the following:
- Video Capture: It opens the video capture device and initializes video properties.
- Frame Capture and Buffering: It continuously captures frames from the video, adds them to the buffer, and manages the buffer size.
- Queue Management: When the buffer is full, it adds the buffer to the frame queue and clears it for the next batch. It also retrieves processed frames from the result queue for display.
- Threading and Synchronization: The method ensures that the processing thread runs concurrently with the main thread, enabling real-time performance. It also signals the processing thread to exit gracefully when the video capture ends.

Pre-Recorded Video Data Acquisition
Using cv2.VideoCapture, the video capture input that points to the actual video file is given to the class constructor.
It is easy to verify that the input corresponds with what’s expected with the help of a validation step. Once the input is validated, the variables contained in the capture are saved as class parameters.
These include the video frames as a NumPy array. The shape of the frames and whether the capture was read correctly or not.

## Pre-Recorded Video Preprocessing
Preprocessing involves normalizing and preparing the data for the RPCA decomposition.
Memorial University of Newfoundland and Labrador
Master of Data Science
If the input is successfully validated as a video file, the frames are read sequentially, converted to grayscale, and vectorized. This conversion ensures uniformity in processing and reduces computational complexity.
For videos, each frame is resized and vectorized to form columns in a matrix.

## Real-Time Video Data Acquisition
The acquisition process for real-time video is similar to the pre-recorded video. However, instead of providing an specific video file to cv2.VideoCapture, the function is called using a parameter specified in the OpenCV documentation.
Using the index ‘0’ the default system camera is used to capture the video stream.
Then, the variables contained in the capture are saved as class parameters in the same manner as in the pre-recorded video data acquisition section.
