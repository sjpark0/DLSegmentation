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
- python maskdino_demo.py --config ./configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml --input ../Data/000.png --opts MODEL.WEIGHTS ../models/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth
- python mask2former_demo.py --config-file ./configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml --input ../Data/000.png --opts MODEL.WEIGHTS ../models/model_final_e5f453.pkl