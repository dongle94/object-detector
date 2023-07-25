import os
import sys
import gdown
import cv2
import numpy as np
import time

from keras.models import Model, Sequential
from keras.layers import Convolution2D, Flatten, Activation
from keras.utils.image_utils import img_to_array

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

from core.gender_classification import VGGFace


class GenderModel(object):
    def __init__(self):
        self.model = self.loadModel()

    def loadModel(self):
        model = VGGFace.baseModel()
        # --------------------------
        classes = 2
        base_model_output = Sequential()
        base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation('softmax')(base_model_output)
        # --------------------------
        gender_model = Model(inputs=model.input, outputs=base_model_output)
        # --------------------------
        # load weights
        root_dir = str(ROOT)
        if os.path.isfile(root_dir+'/weights/gender_model_weights.h5') is False:
            print("gender_model_weights.h5 will be downloaded...")
            url = "https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5"
            output = root_dir+'/weights/gender_model_weights.h5'
            if not os.path.exists(os.path.dirname(output)):
                os.makedirs(os.path.dirname(output))
            gdown.download(url, output, quiet=False)
        gender_model.load_weights(root_dir+'/weights/gender_model_weights.h5')
        return gender_model

    def predict_gender(self, face_image):
        image_preprocesing = self.img_preprocess(face_image)
        gender_predictions = self.model.predict(image_preprocesing, verbose=0)[0, :]
        result_gender = "None"
        if np.argmax(gender_predictions) == 0:
            result_gender = "Woman"
        elif np.argmax(gender_predictions) == 1:
            result_gender = "Man"
        return result_gender

    def img_preprocess(self, face_array, grayscale=False, target_size=(224, 224)):
        detected_face = face_array
        if grayscale is True:
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
        detected_face = cv2.resize(detected_face, target_size)
        img_pixels = img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        # normalize input in [0, 1]
        img_pixels /= 255
        return img_pixels


if __name__ == "__main__":
    g = GenderModel()
    print("Gender Model load ok")

    import dlib

    detector = dlib.cnn_face_detection_model_v1('./weights/mmod_human_face_detector.dat')
    cv2.namedWindow("Face info")
    cam = cv2.VideoCapture(0)
    while True:
        start_time = time.time()
        ret, frame = cam.read()

        boxes_face = detector(frame, 0)
        out = []
        if len(boxes_face) != 0:
            for b in boxes_face:
                b = b.rect
                x0, y0, x1, y1 = b.left(), b.top(), b.right(), b.bottom()
                box_face = np.array([x0, y0, x1, y1])
                face_features = {
                    "gender": [],
                    "bbx_frontal_face": box_face
                }
                face_image = frame[x0:x1, y0:y1]

                face_features["gender"] = g.predict_gender(face_image)

                out.append(face_features)
        else:
            face_features = {
                "gender": [],
                "bbx_frontal_face": []
            }
            out.append(face_features)

        # paint image
        for data_face in out:
            box = data_face["bbx_frontal_face"]
            if len(box) == 0:
                continue
            else:
                x0, y0, x1, y1 = box
                img = cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                thickness = 1
                fontSize = 0.5

                try:
                    cv2.putText(frame, "gender: " + data_face["gender"], (x0, y0 - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), thickness)
                except:
                    pass
        res_img = frame

        end_time = time.time() - start_time
        FPS = 1 / end_time
        cv2.putText(res_img, f"FPS: {round(FPS, 3)}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Face info', res_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
