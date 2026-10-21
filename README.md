# IMDB Review Sentiment Model

This project trains a binary sentiment classifier on the Kaggle IMDB Dataset of
50K Movie Reviews, exports the classifier to ONNX, converts it to OpenVINO IR,
and serves it with OpenVINO Model Server (OVMS). Reviews are transformed into
TF-IDF features before they are sent to the OVMS REST inference endpoint.

## Dataset and training

The notebook downloads the dataset from Kaggle
[`lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
into `data/`, then loads `data/IMDB Dataset.csv` locally. Configure Kaggle API
credentials in your environment before running the download cell. Do not add
Kaggle credentials to this repository.

The training workflow in [movie_review_predict_model.ipynb](./movie_review_predict_model.ipynb):

1. Loads the 50,000-row CSV and maps `positive` sentiment to `1` and
   `negative` sentiment to `0`.
2. Splits the reviews into training and test sets with an 80/20 split and
   `random_state=42`.
3. Fits a `TfidfVectorizer` on the training reviews using English stop words
   and a maximum vocabulary of 10,000 features.
4. Trains a scikit-learn `LogisticRegression` classifier
   (`max_iter=1000`) on those TF-IDF features.
5. Saves the fitted vectorizer and exports the trained classifier to ONNX,
   then converts the ONNX model to OpenVINO IR.

The vectorizer and OpenVINO model must be used together: the classifier expects
10,000 features in the same order and representation produced by the saved
vectorizer.

## Generated model files

Running the training/export cells creates:

```text
movie_review_sentiment_model/
└── 1/
    ├── logistic_regression_openvino.xml
    ├── logistic_regression_openvino.bin
    └── tfidf_vectorizer.pkl

movie_review_sentiment_model_onnx/
└── 1/
    └── logistic_pure.onnx
```

The ONNX file is an intermediate/exported model. OVMS serves the OpenVINO IR
pair (`.xml` and `.bin`) from the `movie_review_sentiment_model/1/` version
directory. The `.pkl` vectorizer is used by the client or preprocessing API;
it is not part of the model's OVMS inference graph.

## Serve the OpenVINO model with Docker

From the repository root, mount the model directory into OVMS and expose its
REST and gRPC ports:

```powershell
docker run -d --rm `
  -v "${PWD}\movie_review_sentiment_model:/models" `
  -p 9000:9000 `
  -p 8000:8000 `
  openvino/model_server:latest `
  --model_path /models `
  --model_name movie_review_sentiment_model `
  --port 9000 `
  --rest_port 8000
```

The REST inference URL is
`http://localhost:8000/v2/models/movie_review_sentiment_model/infer`.
OVMS receives numeric `FP32` features rather than raw text, so the review must
first be transformed using the matching `tfidf_vectorizer.pkl`.

## Send a review for inference

The following Python example runs locally from the repository root. It
vectorizes raw review text, constructs the OVMS REST request, and maps the
returned binary label to a sentiment:

```python
import joblib
import numpy as np
import requests

model_dir = "movie_review_sentiment_model/1"
vectorizer = joblib.load(f"{model_dir}/tfidf_vectorizer.pkl")

reviews = [
    "This movie was fantastic and thrilling!",
    "Terrible waste of time, completely boring.",
]
features = vectorizer.transform(reviews).toarray().astype(np.float32)

payload = {
    "inputs": [
        {
            "name": "float_input",
            "shape": list(features.shape),
            "datatype": "FP32",
            "data": features.tolist(),
        }
    ]
}

url = "http://localhost:8000/v2/models/movie_review_sentiment_model/infer"
response = requests.post(url, json=payload)
response.raise_for_status()
labels = response.json()["outputs"][0]["data"]

label_names = {1: "Positive", 0: "Negative"}
for review, label in zip(reviews, labels):
    print(f"{label_names[int(label)]}: {review}")
```

The model labels are `1` for positive and `0` for negative. For an API that
accepts raw text and performs this preprocessing before calling OVMS, see the
separate [IMDBReviewModelExpose project](../IMDBReviewModelExpose/README.md).

## Notebook dependencies

The notebook setup cell installs pandas, scikit-learn, skl2onnx, onnxruntime,
ONNX, OpenVINO, requests, and joblib. The training/export cells use pandas,
scikit-learn, ONNX, OpenVINO, NumPy, and joblib.
