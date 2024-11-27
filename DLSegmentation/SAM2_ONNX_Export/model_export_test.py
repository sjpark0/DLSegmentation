from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
        
def original(image, input_point, input_label):
    sam_checkpoint = '../../models/sam_vit_h_4b8939.pth'
    model_type = "vit_h"

    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device='cuda')
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    start = time.time()
    masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True,)
    end = time.time()
    print(end - start, "sec")

    print(masks.shape)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show() 
        
    #mask_generator = SamAutomaticMaskGenerator(sam)
    #masks = mask_generator.generate(image)
    #print(len(masks))
    #print(masks[0].keys())

     
    #return masks

def export_test(image, input_point, input_label):
    import onnxruntime
    encoder_model_path = "../../sam2_models/sam2.1_hiera_large_encoder.onnx"
    decoder_model_path = "../../sam2_models/decoder.onnx"

    
    #EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    EP_list = ['CUDAExecutionProvider']

    ort_session_encoder = onnxruntime.InferenceSession(encoder_model_path, provider=EP_list)
    
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()

#get_input_details
    model_inputs = ort_session_encoder.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    input_width = model_inputs[0].shape[3]
    input_height = model_inputs[0].shape[2]
    print(model_inputs[0])
#get_output_details
    model_outputs = ort_session_encoder.get_outputs()    
    output_names = [model_outputs[i].name for i in range(len(model_outputs))]

#prepare_input
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (input_width, input_height))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_img = (input_img / 255.0 - mean) / std
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)    
    
    image_embedding = ort_session_encoder.run(output_names, {input_names[0] : input_tensor})

    high_res_feats_0, high_res_feats_1, image_embed = image_embedding
    
    masks = {}    

    ort_session_decoder = onnxruntime.InferenceSession(decoder_model_path, provider=EP_list)
    decoder_inputs = ort_session_decoder.get_inputs()
    decoder_input_names = [decoder_inputs[i].name for i in range(len(decoder_inputs))]
    decoder_outputs = ort_session_decoder.get_outputs()
    decoder_output_names = [decoder_outputs[i].name for i in range(len(decoder_outputs))]

    print(high_res_feats_0.shape, high_res_feats_0.dtype)
    print(high_res_feats_1.shape, high_res_feats_1.dtype)
    print(image_embed.shape, image_embed.dtype)
    print(input_names)
    print(output_names)
    print(decoder_input_names)
    print(decoder_output_names)
    
    onnx_coord = []
    onnx_label = []
    onnx_coord.append(input_point)
    onnx_label.append(input_label)
    onnx_coord = np.concatenate(onnx_coord, axis = 0)
    onnx_label = np.concatenate(onnx_label, axis = 0)

    
    num_label = 1
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.array([0], dtype=np.float32)
    original_size = np.array([image.shape[0], image.shape[1]], dtype=np.int32)

    input_point_coords = onnx_coord[np.newaxis, ...]
    input_point_labels = onnx_label[np.newaxis, ...]
    input_point_coords[..., 0] = input_point_coords[..., 0] / image.shape[1] * input_width
    input_point_coords[..., 1] = input_point_coords[..., 1] / image.shape[0] * input_height

    input_point_coords = input_point_coords.astype(np.float32)
    input_point_labels = input_point_labels.astype(np.float32)
    
    decoder_inputs = (image_embed, high_res_feats_0, high_res_feats_1, input_point_coords, input_point_labels, mask_input, has_mask_input, original_size)
    masks, _ = ort_session_decoder.run(decoder_output_names, {decoder_input_names[i]: decoder_inputs[i] for i in range(len(decoder_input_names))})

    masks = masks > 0.0
    print(masks.shape)
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show() 
    #for i, (mask, score) in enumerate(zip(masks[0,:], scores[0,:])):
    #    plt.imshow(image)
    #    show_mask(mask, plt.gca())
    #    show_points(input_point, input_label, plt.gca())
    #    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    #    plt.axis('off')
    #    plt.show()
    #plt.show() 
    
input_point = np.array([[533, 286]])
input_label = np.array([1])
image = cv2.imread("../../Data/000.png")
#original(image, input_point, input_label)
export_test(image, input_point, input_label)

