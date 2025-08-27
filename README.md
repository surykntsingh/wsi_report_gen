# WSI report generation
Report generation using multiple pretrained feature encoders along with a multi-layer encoder decoder transformer seq-to-seq model

Steps to follow:
- Setup env:
  - conda create -n wsi_rgen python=3.11
  - conda activate wsi_rgen
  - pip install -r requirements.txt
- Extract and save CONCH 1.5 and TITAN features using TRIDENT (https://github.com/mahmoodlab/TRIDENT) and save as h5 files.
- Train and extract GECKO deep and concept features (https://github.com/surykntsingh/GECKO) and save as h5 files.
- Set appropriate paths in config.yaml
- Train and test and predict the report gen model with following command: `torchrun --nproc-per-node=6 --nnodes=1 main.py train`
