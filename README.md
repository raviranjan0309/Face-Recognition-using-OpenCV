The whole process for face recognition using OpenCV can be divided in three major steps:</br>
a.	The first step to detect faces in the database images with multiple images for everyone using haarcascade classifier.</br>
b.	The next step is to build model using Local Binary Patterns Histograms Face Recognizer.</br>
c.	The last step is to test the face recognizer to recognize faces it was trained for.</br></br></br>

Technical Specification:</br>
•	Python 3.5</br>
•	Packages – numpy, cv2, os, Image</br>
•	Dataset: Yale Face Database</br></br>

How to install OpenCV with OpenCV-Contrib</br>

Step1. Download OpenCV whl file form http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv</br>
If you have python 3.5 install opencv_python‑3.2.0+contrib‑cp35‑cp35m‑win_amd64.whl</br>
If you have python 3.6 install opencv_python‑3.2.0+contrib‑cp36‑cp36m‑win_amd64.whl</br>
Verify the windows bit and python version</br>
Open Anaconda command. Type command – 
Pip install “path of whl file”
