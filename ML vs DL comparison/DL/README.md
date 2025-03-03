This is the code used to perform the comparison of DL algorithms (CNN-Transformer vs CNN-TransformerBimodal) to classify worms of the N2 and unc-80 strains.
## Installation
To reproduce the results you must install the dependencies.
You can do this in a conda environment:
```bash
conda create -n myenv
source activate myenv
pip install -r requirements.txt
```

For PyTorch installation, it is recommended to install the appropriate version based on your system's CUDA compatibility. Users can do this by following the official PyTorch installation guide at https://pytorch.org/get-started/locally/ and using the appropriate command. For example, to install PyTorch with CUDA 11.1, use:
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
## Usage
To train the unimodal model launch the script
```bash
python mainVideos.py
```

To train the bimodal model launch the script
```bash
python mainBimodal.py
```
