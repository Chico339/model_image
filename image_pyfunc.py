import mlflow.pyfunc


import mlflow.sklearn
import mlflow
import mlflow.keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
import cv2 as cv
import numpy as np
# Define the model class
class image_keras_classifier(mlflow.pyfunc.PythonModel):
    from keras.models import load_model

    def predict_from_picture(self, img_df):
        out=[]
        #img = image.load_img(img_df, target_size=(299, 299))
        #img=cv.resize(img_df,dsize=(299,299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
        images = np.vstack(out)
        keras_model1=self.load_model()
        preds = keras_model1.predict(images)
        best=decode_predictions(preds, top=1)[0]
        print('Predicted:', best )
        return best

    def load_model(self):
        keras_model = mlflow.keras.load_model("mlruns/0/00ad9cbd7cb842c5ac356a08c30707b0/artifacts/model/")
        return keras_model

    def predict(self, context, model_input):
        return self.predict_from_picture(model_input)

# Construct and save the model
model_path = "add_n_model2"
#img = cv.imread('chien.jpg')
img = image.load_img("C:/Users/Chico le bg/PycharmProjects/test3/test_image/data/01_raw/chien.jpg", target_size=(299, 299))
add5_model = image_keras_classifier()
add5_model.predict("blabla",img)


mlflow.pyfunc.save_model(path=model_path, python_model=add5_model)
#mlflow models serve -m add_n_model2 -h 0.0.0.0 -p 1238
# mlflow models serve -m f0e9f66c11c24822957817d0a3162ddc\artifacts\model\ -h 0.0.0.0 -p 1238


