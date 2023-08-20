import numpy as np
import torch
import cv2

def loader(config):
    save_dir = config.data_path
    face_detected_resized_img = np.load('%s/face_detected_resized_img.npy' % save_dir)
    #face_detected_resized_subject_name = np.load('%s/face_detected_resized_subject_name.npy' % save_dir)
    face_detected_resized_pose = np.load('%s/face_detected_resized_pose.npy' % save_dir)
    face_detected_resized_light_info = np.load('%s/face_detected_resized_light_info.npy' % save_dir)
    face_detected_resized_file_path = np.load('%s/face_detected_resized_file_path.npy' % save_dir)

    
    #x range: [0, 1]
    resized_img_list = np.array([cv2.resize(face, dsize=(64, 64),
                                            interpolation=cv2.INTER_CUBIC) for face in face_detected_resized_img])
    resized_img_list = resized_img_list/255.0

    #c range: [-1, 1]
    float_label_light_info_list = []
    for label_light_info in face_detected_resized_light_info:
        float_label_light_info = [np.float(label_light_info[0])/180.0, np.float(label_light_info[1])/90.0]
        float_label_light_info_list.append(float_label_light_info)
    float_label_light_info_list = np.array(float_label_light_info_list)
    
    
    # Create Tensors to hold input and outputs.
    x = torch.tensor(np.expand_dims(resized_img_list, 1), dtype=torch.float32).cuda()
    c = torch.tensor(float_label_light_info_list, dtype=torch.float32).cuda()
    
    return x, c