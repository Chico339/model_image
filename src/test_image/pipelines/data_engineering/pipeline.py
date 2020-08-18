from kedro.pipeline import node, Pipeline

from test_image.pipelines.data_engineering.nodes import affiche_image, get_pred

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=affiche_image,
                inputs='raw_data',
                outputs=None
            ),
            node(
                func=get_pred,
                inputs='raw_data',
                outputs=None
            )
        ]
    )
#mlflow models serve -m mlruns/0/40727d74e5a54ec7afb3bd357f9e07ab/artifacts/model/ -h 0.0.0.0 -p 1238
