import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from queue import Queue
import threading
from datetime import datetime
import os
import time
from statistics import mean

class RPCA:
    def __init__(self, input=None):
        self.wrong_input = False
        self.image_input = False
        if input is not None:
            if isinstance(input, cv2.VideoCapture):
                self.video = input
                self.ret, self.frame = input.read()
                self.height, self.width, self.channels = self.frame.shape
                self.image_input = False

                frames = []
                
                while self.ret:
                    # Convert the frame to grayscale and append it as a vector
                    frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                    frame_small = cv2.resize(frame_gray, (self.width, self.height)) 
                    frame_vector = frame_small.flatten()
                    frames.append(frame_vector)
                    self.ret, self.frame = input.read()

                input.release()

                # Stack the frames as columns in a matrix
                self.frame_matrix = np.column_stack(frames)

            elif isinstance(input, np.ndarray): 
                self.single_image = input
                self.image_array = np.asarray(self.single_image)  # Convert to array
                self.image_input = True
        
            else:
                # Wrong Input
                self.wrong_input = True
                raise ValueError("Input type is not supported")
            
            self.L = None
            self.S = None

        # Realtime PCA
        self.buffer_size = 30
        self.frame_buffer = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.frame_queue = Queue()
        self.result_queue = Queue()
        self.lock = threading.Lock()
        self.frame_counter = 0
        self.detected_num = 0
        self.detection_times = []

        # Moving object detection
        self.alpha = 0.1  # Smoothing factor for running average and std dev
        #self.std_dev_multiplier = 1.1  # Multiplier for setting the threshold
        self.std_dev_multiplier = 1.1 # Multiplier for setting the threshold
        self.running_mean = None
        self.running_std_dev = None

    def show_shape(self):
        if self.wrong_input:
            raise ValueError("Call Image_RPCA.RPCA(input) with an appropriate input first")
        if self.image_input:
            print('Height:', self.image_array.shape[0])
            print('Width:', self.image_array.shape[1])
        else:
            print('Frame dimensions')
            print('Height:', self.height)
            print('Width:', self.width)
            print('\nMatrix Shape: ', self.frame_matrix.shape)

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), 0)
    
    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))
    
    def fit(self, lambda_val=None, mu=None, tol=1E-7, max_iter=10000, verbose=True, num_iter_print=300):
        if self.wrong_input:
            raise ValueError("Call Image_RPCA.RPCA(input) with an appropriate input first")
        
        X = self.image_array if self.image_input else self.frame_matrix
        iter = 0
        err = np.inf
        Sk = np.zeros(X.shape)
        Yk = np.zeros(X.shape)
        Lk = np.zeros(X.shape)

        # Default values for lambda and mu
        if lambda_val is None:
            lambda_val = 1 / np.sqrt(np.max(X.shape))
        if mu is None:
            mu = np.prod(X.shape) / (4 * np.linalg.norm(X, ord=1))

        mu_inv = 1 / mu
        _tol = tol * self.frobenius_norm(X)

        # Main Loop
        for _ in range(max_iter):
            Lk = self.svd_threshold(X - Sk + mu_inv * Yk, mu_inv)
            Sk = self.shrink(X - Lk + mu_inv * Yk, mu_inv * lambda_val)
            Yk += mu * (X - Lk - Sk)
            err = self.frobenius_norm(X - Lk - Sk)
            iter += 1
            if verbose:
                if (iter % num_iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                    print(f'iteration: {iter}, error: {err}')
            if err < _tol:
                break

        self.L = Lk
        self.S = Sk

    def make_video(self, frames, filename='Videos/Out/out.mp4', fps=30):
        height = self.height
        width = self.width
        # Number of frames is the number of columns in the frame_matrix
        num_frames = frames.shape[1]
        video = np.reshape(frames, (height, width, num_frames))

        # Codec and writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=False)

        for i in range(num_frames):
            frame = video[:, :, i]  # Correct indexing to get the i-th frame
            frame = np.uint8(frame)
            writer.write(frame)

        writer.release()

    @staticmethod
    def get_first_frame(filename):
        cap = cv2.VideoCapture(filename)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Cannot read frame from {filename}")

    def display_results(self):
        if self.L is None or self.S is None:
            raise ValueError("RPCA has not been applied. Call apply_rpca() first.")

        # Generate a unique timestamp for the filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        first_frame_dir = 'Videos/OutFirstFrames/'

        if self.image_input:
            # Normalize L and S for display
            L_normalized = cv2.normalize(self.L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            S_normalized = cv2.normalize(self.S, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 3, 1)
            plt.title('Original Image')
            plt.imshow(self.single_image, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title('Low-Rank Component')
            plt.imshow(L_normalized, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title('Sparse Component')
            plt.imshow(S_normalized, cmap='gray')
            plt.axis('off')

            plt.show()
        
        else:
            L_normalized = cv2.normalize(self.L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            S_normalized = cv2.normalize(self.S, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Use the timestamp in the filenames to ensure uniqueness
            L_filename = f'Videos/Out/LowRank_{timestamp}.mp4'
            S_filename = f'Videos/Out/Sparse_{timestamp}.mp4'

            self.make_video(frames=L_normalized, filename=L_filename, fps=30)
            self.make_video(frames=S_normalized, filename=S_filename, fps=30)

            frame_L = self.get_first_frame(L_filename)
            frame_S = self.get_first_frame(S_filename)

            lowrank_filename = first_frame_dir + f'lowrank_first_frame_{timestamp}.jpg'
            sparse_filename = first_frame_dir + f'sparse_first_frame_{timestamp}.jpg'

            cv2.imwrite(lowrank_filename, frame_L)
            cv2.imwrite(sparse_filename, frame_S)


    def add_frame_to_buffer(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_small = cv2.resize(frame_gray, (self.width, self.height)) 
        frame_vector = frame_small.flatten().reshape(-1, 1)
        self.frame_buffer.append(frame_vector)

    def process_buffer(self, frame_buffer):
        start_time = time.time()
        self.frame_matrix = np.column_stack(frame_buffer)
        # Fit Whole MATRIX
        self.fit(tol=1E-5, verbose=False, max_iter=100)
        #return frames
        L_frames = self.L.reshape(self.height, self.width, self.buffer_size)
        S_frames = self.S.reshape(self.height, self.width, self.buffer_size)
        L_frame, S_frame = L_frames[:, :, -1], S_frames[:, :, -1]
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        #print(f"Buffer processed in: {elapsed_time} seconds")
        self.detection_times.append(elapsed_time)

        return L_frame, S_frame
    
    def process_frames(self):
        while True:
            frame_buffer = self.frame_queue.get()
            if frame_buffer is None:
                break
            L_frame, S_frame = self.process_buffer(frame_buffer)
            self.result_queue.put((L_frame, S_frame))


## Object detection
    def update_statistics(self, S_frame):
        if self.running_mean is None:
            self.running_mean = np.mean(S_frame)
            self.running_std_dev = np.std(S_frame)
        else:
            frame_mean = np.mean(S_frame)
            frame_std_dev = np.std(S_frame)
            self.running_mean = self.alpha * frame_mean + (1 - self.alpha) * self.running_mean
            self.running_std_dev = self.alpha * frame_std_dev + (1 - self.alpha) * self.running_std_dev

    def detect_object(self, S_frame):
        self.update_statistics(S_frame)
        current_std_dev = np.std(S_frame)
        threshold = self.running_mean + self.std_dev_multiplier * self.running_std_dev
        print(current_std_dev, threshold)
        if current_std_dev > threshold:
            self.detected_num += 1
            print('Object detected')

## Display 

    def display_realtime(self, L_frame, S_frame):
        if L_frame is not None and S_frame is not None:
            L_normalized = cv2.normalize(L_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            self.detect_object(S_frame)

            cv2.imshow('Low-Rank Component', L_normalized)
            cv2.imshow('Sparse Component', S_frame)

    def realtime_rpca(self, collect_data=False):
        threading.Thread(target=self.process_frames, daemon=True).start()
        video = cv2.VideoCapture(0)
        self.ret, self.frame = video.read()
        self.height, self.width, self.channels = self.frame.shape
        self.height = self.height // 2
        self.width = self.width // 2
        print('Video Capture Opened')
        object_present = 0
        processed_buffers = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            # Add frame to buffer at half frequency (15 fps)
            self.frame_counter += 1
            if self.frame_counter % 2 == 0:
                self.add_frame_to_buffer(frame)
            
            if len(self.frame_buffer) >= self.buffer_size:
                self.frame_queue.put(self.frame_buffer.copy())
                self.frame_buffer.clear()
                self.frame_counter = 0
            
            if not self.result_queue.empty():
                L_frame, S_frame = self.result_queue.get()
                self.display_realtime(L_frame, S_frame)
                processed_buffers += 1

            if cv2.waitKey(1) == ord('d'):
                object_present += 1  # 1 for detected object

            if collect_data & (len(self.detection_times)>=30):
                print('Total number of buffers processed: ', len(self.detection_times))
                print('Average buffer processing time: ', mean(self.detection_times))
                print('Total ground truth object detected: ', object_present)
                print('Total object detected', self.detected_num)
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                

        video.release()
        cv2.destroyAllWindows()

        # Signal the process_frames thread to exit
        self.frame_queue.put(None)

    def get_times(self):
        return self.detection_times



 




