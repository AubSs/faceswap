#!/usr/bin/env python3

import os
import cv2
import dlib
import argparse
import numpy as np

from helpers import warp_image_3d, mask_from_points, apply_mask, correct_colours

## Face detection
def face_detection(img):
    # Ask the detector to find the bounding boxes of each face.
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, 0)
    return faces

## Face and points detection
def face_points_detection(img, bbox, model_path):
    predictor = dlib.shape_predictor(model_path)
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, bbox)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    # return the list of (x, y)-coordinates
    return coords


def get_faces(im, model_path, r=10):
    faces = face_detection(im)

    if len(faces) < 2:
        return []

    def get_face(face_nb):
        points = np.asarray(face_points_detection(im, faces[face_nb], model_path))
        im_w, im_h = im.shape[:2]
        left, top = np.min(points, 0)
        right, bottom = np.max(points, 0)
        x, y = max(0, left-r), max(0, top-r)
        w, h = min(right+r, im_h)-x, min(bottom+r, im_w)-y
        return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y+h, x:x+w]

    return [get_face(0), get_face(1)]


if __name__ == '__main__':
    app_name = 'Face Swap'

    parser = argparse.ArgumentParser(description=app_name)
    parser.add_argument('--both', default=False, action='store_true', help='Swap both faces')
    parser.add_argument('--model', help='Model Path')
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    frame_nb = 0 # Skip frame for speedup
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_nb = frame_nb + 1
        if frame_nb % 3 == 0:
            frame_nb = 0
            continue

        # Select src face
        faces = get_faces(frame, args.model)
        if len(faces):
            first_face_points, first_face_shape, first_face = faces[0]
            second_face_points, second_face_shape, second_face = faces[1]

            ## 3D Warp Image
            def extract_face(src_points, src_shape, src_face, dst_points, dst_shape, dst_face):
                w, h = dst_face.shape[:2]
                warped_src_face = warp_image_3d(src_face, src_points[:48], dst_points[:48], (w, h))
                ## Mask for blending
                mask = mask_from_points((w, h), dst_points)
                mask_src = np.mean(warped_src_face, axis=2) > 0
                mask = np.asarray(mask*mask_src, dtype=np.uint8)
                ## Correct color
                warped_src_face = apply_mask(warped_src_face, mask)
                dst_face_masked = apply_mask(dst_face, mask)
                warped_src_face = correct_colours(dst_face_masked, warped_src_face, dst_points)
                ## Shrink the mask
                kernel = np.ones((10, 10), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                ## Poisson Blending
                r = cv2.boundingRect(mask)
                center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
                return cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)


            frame_cp = frame.copy()
            x, y, w, h = second_face_shape
            frame_cp[y:y+h, x:x+w] = extract_face(first_face_points, first_face_shape, first_face, second_face_points, second_face_shape, second_face)

            if args.both:
                x, y, w, h = first_face_shape
                frame_cp[y:y+h, x:x+w] = extract_face(second_face_points, second_face_shape, second_face, first_face_points, first_face_shape, first_face)

            cv2.imshow(app_name, frame_cp)
        else:
            cv2.imshow(app_name, frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
