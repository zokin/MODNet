"""
Inference ONNX model of MODNet

Arguments:
    --image-path: path of the input image (a file)
    --output-path: path for saving the predicted alpha matte (a file)
    --model-path: path of the ONNX model

Example:
python inference_onnx.py \
    --image-path=demo.jpg --output-path=matte.png --model-path=modnet.onnx
"""

import os
import cv2
import argparse
import numpy as np
import onnxruntime
import tqdm
import glob
import json
import torch
import typing
import toolz

def get_scale_factor(im_h, im_w, ref_size):
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32

    x_scale_factor = im_rw / im_w
    y_scale_factor = im_rh / im_h

    return x_scale_factor, y_scale_factor

def _read_keypoints(
    filename: str,
    load_hands=True,
    load_face=True,
    load_face_contour=False
) -> typing.Dict[str, torch.Tensor]:
    with open(filename) as keypoint_file:
        data = json.load(keypoint_file)
    keypoints, gender_pd, gender_gt = [], [], []
    for person in data['people']:
        body = np.array(person['pose_keypoints_2d'], dtype=np.float32)
        body = body.reshape([-1, 3])
        if load_hands:
            left_hand = np.array(person['hand_left_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
            right_hand = np.array(person['hand_right_keypoints_2d'], dtype=np.float32).reshape([-1, 3])
            body = np.concatenate([body, left_hand, right_hand], axis=0)
        if load_face:
            face = np.array(person['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]
            contour_keyps = np.array([], dtype=body.dtype).reshape(0, 3)
            if load_face_contour:
                contour_keyps = np.array(person['face_keypoints_2d'], dtype=np.float32).reshape([-1, 3])[:17, :]
            body = np.concatenate([body, face, contour_keyps], axis=0)

        gender_pd.append(person.get('gender_pd', None))
        gender_gt.append(person.get('gender_gt', None))
        keypoints.append(torch.from_numpy(body))
    return {
        'keypoints': keypoints,
        'gender_pd': gender_pd,
        'gender_gt': gender_gt, 
    }

if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_glob', type=str, help='wildchar path of the input images')
    parser.add_argument('--out_path', type=str, help='path for saving the predicted alpha matte (a file)')
    parser.add_argument('--kpts_path', type=str, help='path for saving the predicted alpha matte (a file)')
    parser.add_argument('--model', type=str, help='path of the ONNX model')
    parser.add_argument('--rotate', action='store_true', help='Rotate the image 90 degrees clockwise')
    parser.add_argument('--split', action='store_true', help='Rotate the image 90 degrees clockwise')
    args = parser.parse_args()
    
    
    use_keypoints = args.kpts_path is not None and os.path.exists(args.kpts_path)
    ref_size = 512
    bbox_scale_factor = 1.1
    os.makedirs(args.out_path, exist_ok=True)    
    filenames = glob.glob(args.img_glob)
    session = onnxruntime.InferenceSession(args.model, None) # Initialize session and get prediction
    for filename in tqdm.tqdm(filenames, desc='Matting'):
        name, ext = os.path.splitext(os.path.basename(filename))
        img = cv2.imread(filename)
        im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        if len(im.shape) == 2: # unify image channels to 3
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]        
        im = (im - 127.5) / 127.5   # normalize values to scale it between -1 to 1
        H, W = im.shape[:2]
        out_filename = os.path.join(args.out_path, f"{name}_silhouette{ext}")
        if use_keypoints:
            kpts_filename = os.path.join(args.kpts_path, f"{name}_keypoints.json")
            data = _read_keypoints(kpts_filename)
            keypoints = data['keypoints']            
            def _get_area(keypoints: torch.Tensor) -> float:
                min_x = keypoints[..., 0].min()
                min_y = keypoints[..., 1].min()
                max_x = keypoints[..., 0].max()
                max_y = keypoints[..., 1].max()
                return (max_x - min_x) * (max_y - min_y) * keypoints[..., 2].sum()
            if not len(keypoints):
                matte = np.zeros_like(cv2.resize(matte, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA))
                cv2.imwrite(out_filename, matte)
                cv2.imwrite(out_filename.replace('silhouette', 'background'), matte)
                cv2.imwrite(out_filename.replace('silhouette', 'foreground'), matte)
                print(f"{name} without keypoints !")
                continue
            keypoints = [max(keypoints, key=_get_area)]
            keypoints = torch.stack(keypoints, dim=0).squeeze()
            x, y = keypoints[:25, :2].split(1, dim=1)
            x = x[torch.nonzero(x)]
            y = y[torch.nonzero(y)]
            x_min, y_min = x.min()[np.newaxis], y.min()[np.newaxis]
            x_max, y_max = x.max()[np.newaxis], y.max()[np.newaxis]
            width = x_max - x_min
            height = y_max - y_min
            x_min -= bbox_scale_factor * width * 0.5
            x_max += bbox_scale_factor * width * 0.5
            y_min -= bbox_scale_factor * height * 0.5
            y_max += bbox_scale_factor * height * 0.5
            x_min = torch.clamp(x_min, 0, W)
            x_max = torch.clamp(x_max, 0, W)
            y_min = torch.clamp(y_min, 0, H)
            y_max = torch.clamp(y_max, 0, H)
            bbox = np.floor(torch.cat([x_min, y_min, x_max, y_max], dim=0).numpy()).astype(np.int32).tolist()
            im = im[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            if args.rotate:
                im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        im_h, im_w, im_c = im.shape
        x, y = get_scale_factor(im_h, im_w, ref_size) # Get x_scale_factor & y_scale_factor to resize image
        im = cv2.resize(im, None, fx=x, fy=y, interpolation=cv2.INTER_AREA)
        im = np.transpose(im)
        im = np.swapaxes(im, 1, 2)
        im = np.expand_dims(im, axis = 0).astype('float32')        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: im})        
        matte = (np.squeeze(result[0]) * 255).astype('uint8') # refine matte
        matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA)        
        if args.rotate:
            matte = cv2.rotate(matte, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if use_keypoints:
            zeros = np.zeros((H, W), dtype=np.uint8)
            zeros[bbox[1]:bbox[3], bbox[0]:bbox[2]] = matte
            matte = (zeros > 0.5).astype(np.uint8) * 255
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(matte)
            if len(stats) <= 1:
                matte = np.zeros_like(cv2.resize(matte, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA))
                cv2.imwrite(out_filename, matte)
                cv2.imwrite(out_filename.replace('silhouette', 'background'), matte)
                cv2.imwrite(out_filename.replace('silhouette', 'foreground'), matte)
                print(f"{name} without foreground !")
                continue
            max_area_label, max_area = max(toolz.drop(1, enumerate(stats)), key=lambda t: t[1][-1])
            matte[labels != max_area_label] = 0
            matte[labels == max_area_label] = 255
        cv2.imwrite(out_filename, matte)
        if args.split:
            background = img.copy()
            background[..., 0][labels == max_area_label] = 0
            background[..., 1][labels == max_area_label] = 0
            background[..., 2][labels == max_area_label] = 0
            img[..., 0][labels != max_area_label] = 0
            img[..., 1][labels != max_area_label] = 0
            img[..., 2][labels != max_area_label] = 0
            cv2.imwrite(out_filename.replace('silhouette', 'background'), background)
            cv2.imwrite(out_filename.replace('silhouette', 'foreground'), img)