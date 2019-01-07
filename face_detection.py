import dlib

## Face detection
def face_detection(img):
    # Ask the detector to find the bounding boxes of each face.
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, 0)
    return faces

## Face and points detection
def face_points_detection(img, bbox):
    PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, bbox)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    # return the list of (x, y)-coordinates
    return coords
