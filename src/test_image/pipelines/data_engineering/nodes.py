from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from kedro.extras.datasets.pillow import ImageDataSet
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import mlflow.sklearn
import mlflow
import mlflow.keras

def get_pred(raw):
    img_path = 'C:/Users/Chico le bg/PycharmProjects/test3/test_image/data/01_raw/chien.jpg'
    out = []
    inet_model = inc_net.InceptionV3()
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = inc_net.preprocess_input(x)
    out.append(x)
    images = np.vstack(out)
    # I'm dividing by 2 and adding 0.5 because of how this Inception represents images
    plt.imshow(images[0] / 2 + 0.5)
    preds = inet_model.predict(images)
    for x in decode_predictions(preds)[0]:
        print(x)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(images[0].astype(np.double), inet_model.predict, top_labels=2,
                                             hide_color=0, num_samples=100)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2,
                                                hide_rest=True)
    img1 = mark_boundaries(temp / 2 + 0.5, mask)
    plt.imshow(img1)
    plt.savefig("C:/Users/Chico le bg/PycharmProjects/test3/test_image/data/07_model_output/img1.png")
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2,
                                                hide_rest=False)
    img2 = mark_boundaries(temp / 2 + 0.5, mask)
    plt.imshow(img2)
    plt.savefig("C:/Users/Chico le bg/PycharmProjects/test3/test_image/data/07_model_output/img2.png")
    mlflow.log_artifact("C:/Users/Chico le bg/PycharmProjects/test3/test_image/data/07_model_output/img1.png")
    mlflow.log_artifact("C:/Users/Chico le bg/PycharmProjects/test3/test_image/data/07_model_output/img2.png")
    print(type(inet_model))

    #mlflow.sklearn.log_model(inet_model, "model")
    mlflow.keras.log_model(inet_model,"model")
def affiche_image(rawda):
    data_set = ImageDataSet(filepath="data/01_raw/elephant.jpg")
    image = data_set.load()
    image.show()


