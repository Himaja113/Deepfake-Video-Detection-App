# **Deepfake-Video-Detection-App**

## **Overview**
The **Deepfake Video Detection** project is a machine learning-based application designed to detect whether a given video is real or a deepfake. Using a trained deep learning model and a ResNet50-based feature extractor, the application analyzes video frames and outputs the likelihood of the video being a deepfake. This project is deployed as a web application using **Streamlit**.

## **Features**
- **Upload and Analyze**: Upload a video file to check if it's a deepfake.
- **Real-Time Feedback**: The application processes the video and provides immediate results.
- **Pretrained Model**: Utilizes a pretrained ResNet50 model for feature extraction.
- **Scalable Deployment**: Built using Streamlit for easy deployment on local or cloud platforms.

---

## **Tech Stack**
- **Programming Language**: Python
- **Libraries and Frameworks**:
  - **TensorFlow/Keras**: For deep learning model training and inference.
  - **OpenCV**: For video frame extraction and preprocessing.
  - **Streamlit**: For creating the interactive web application.
  - **NumPy**: For numerical computations.
- **Pretrained Model**: ResNet50 (ImageNet weights) for feature extraction.

---

## **How to Run the Project**

### Prerequisites
- Python 3.8 or later installed on your system.
- Virtual environment for managing dependencies.

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd deepfake-video-detection
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the pretrained ResNet50 weights are available:
   - Download from [ResNet50 Weights](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5) if not automatically downloaded.
   - Place the weights in the appropriate directory or update the code to specify the local path.

5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

6. Access the app in your browser at `http://localhost:8501`.

---

## **Folder Structure**
```
deepfake-video-detection/
├── app.py                   # Streamlit application script
├── deepfake_detection_model.h5  # Trained deepfake detection model
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── sample_videos/           # Directory for test videos
└── venv/                    # Virtual environment (optional, not included in repo)
```

---

## **Demo**
### Upload a video file (e.g., `test.mp4`) through the web interface.  
- **Result**: The app displays whether the video is a deepfake or real.  

---

## **About the Author**
- **Author**: Meesala Himaja
- **GitHub**: (https://github.com/Himaja113)

Feel free to contribute to the project or provide suggestions for improvement!

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

