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

### 4. mask2former설치
- git clone https://github.com/facebookresearch/Mask2Former.git
  
### 5. SAM설치
- git clone https://github.com/facebookresearch/segment-anything.git && cd segment-anything
- pip install -e .

### 6. SAM2설치
- git clone https://github.com/facebookresearch/sam2.git && cd sam2
- pip install -e .

- cd checkpoints
- ./download_ckpts.sh
- cd ..

### 7. 나머지 설치
- pip install fvcore omegaconf pycocotools opencv-python cloudpickle timm scipy

### 8. onnx export
- CPU : pip install onnxruntime onnx==1.16.1
- GPU : pip install onnxruntime-gpu onnx==1.16.1
- 주의 : onnx가 1.16.1로 설치하지 않으면 에러발생함
  
## 실행
- demo.py파일 안에 sys.path.append로 detectron2경로 추가
### maskdino실행
- python maskdino_demo.py --config ./configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml --input ../../Data/000.png --opts MODEL.WEIGHTS ../../models/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth
### mask2former실행
- python mask2former_demo.py --config-file ./configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml --input ../../Data/000.png --opts MODEL.WEIGHTS ../../models/model_final_e5f453.pkl
### SAM실행
- python sam_demo.py
### SAM2실행
- python sam2_demo.py