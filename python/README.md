# Python library
This is a rough library api to the underlying C++ code. I used it to collect data and compare it to a CNN approach.

## CNN comparison
To run everything from the CNN comparison you need to install dependencies.
### requirements .txt
```bash
pip install -r requirements.txt
```
### Pytorch
If you want to train or run models from this project, you will need Pytorch.
I used version 2.13 with CUDA 13.2. You can use the same version:
```bash
pip install torch==2.13.0 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu132
```
### C++ bindings
To use the library you need to install it with.
```bash
cd ./python # Make sure you are in the python folder
pip install -e .
```
