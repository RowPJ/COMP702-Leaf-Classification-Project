# COMP702 Leaf Classification Project

Required libraries:
1. Pytorch (check https://pytorch.org/ to easily build a terminal command to install the latest version. Make sure to install with CUDA support selected if you want to use NVIDIA GPU-accelerated auto-encoder training. Auto-encoder training can still be done on the CPU, but will be a bit slower.)
2. Numpy (with pip python package manager, run "pip install numpy" in a terminal)
3. OpenCV (with pip python package manager, run "pip install opencv-python" in a terminal)
4. libSVM (with pip python package manager, run "pip install libsvm libsvm-official" in a terminal)

Steps to run:
1. Download the project and unzip it.
2. On Windows, double click "run.bat"
3. On other platforms, open a terminal and move to the directory "src", then enter "python main.py"

Notes:
* GPU acceleration can be disabled in autoencoder.py by setting useCudaP to False near the top of the file. It won't cause problems to leave it enabled though.
* The leafsnap dataset folder in this repository is not the full dataset, it is only the images used by this project to save download time. The full dataset can be used instead of the dataset in this repository, but make sure the directory structures match or the program will crash.
