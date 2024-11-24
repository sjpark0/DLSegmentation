# DLSegmentation
## Installation
### 1. pytorch설치
- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

### 2. detectron2설치
- git clone https://github.com/facebookresearch/detectron2.git
- setup.py의 79줄에 "-DWITH_CUDA", 추가 (CUDA사용시)
- python setup.py build_ext --inplace

### 3. maskdino설치
- git clone https://github.com/IDEA-Research/MaskDINO.git
- pip install -r requirements.txt
- cd maskdino/modeling/pixel_decoder/ops
- python setup.py build install

### 4. 나머지 설치
- pip install fvcore omegaconf pycocotools
  
## 실행
### maskdino실행
- demo.py파일 안에 sys.path.append로 detectron2경로 추가
