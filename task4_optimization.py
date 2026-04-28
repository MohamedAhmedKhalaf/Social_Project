import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import TfidfVectorizer


# =====================================================
# LOAD GLOVE
# =====================================================

def load_glove_model(glove_file):

    print("Loading GloVe Model...")
    glove_dict = {}

    with open(glove_file,"r",encoding="utf-8") as f:

        for line in f:

            values = line.split()

            word = values[0]

            vector = np.asarray(
                values[1:],
                dtype="float32"
            )

            glove_dict[word] = vector

    print(f"Loaded {len(glove_dict)} words from GloVe.")

    return glove_dict


# =====================================================
# TF-IDF WEIGHTED GLOVE
# MUCH BETTER THAN SIMPLE AVERAGING
# =====================================================

def get_tfidf_glove_representation(texts, glove_dict, vector_size=100):

    print("Building TF-IDF...")

    tfidf = TfidfVectorizer()

    tfidf.fit(texts)

    vocab = tfidf.vocabulary_

    features = []

    for text in texts:

        tokens = str(text).lower().split()

        weighted_vectors = []

        weights = []

        for token in tokens:

            if token in glove_dict and token in vocab:

                idx = vocab[token]

                weight = tfidf.idf_[idx]

                weighted_vectors.append(
                    glove_dict[token] * weight
                )

                weights.append(weight)

        if len(weighted_vectors) > 0:

            vec = np.sum(
                weighted_vectors,
                axis=0
            ) / np.sum(weights)

        else:

            vec = np.zeros(vector_size)

        features.append(vec)

    return np.array(features), tfidf 


# =====================================================
# MAIN
# =====================================================

def optimize_and_evaluate():

    print("Loading datasets...")

    df = pd.read_csv("full_dataset.csv").dropna()

    X = df["text"].values
    y = df["ground_truth"].values


    # =================================================
    # TFIDF + GLOVE FEATURES
    # =================================================

    glove_dict = load_glove_model(
        "glove.6B.100d.txt"
    )

    X_glove, tfidf_vectorizer  = get_tfidf_glove_representation(
        X,
        glove_dict
    )


    # =================================================
    # SPLIT
    # =================================================

    X_train, X_test, y_train, y_test, X_text_train, X_text_test = train_test_split(

        X_glove,
        y,
        X,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    # =================================================
    # RANDOM FOREST
    # =================================================

    print("\n--- Random Forest Tuning ---")

    rf_params = {

        "n_estimators":[
            200,
            500,
            800
        ],

        "max_depth":[
            None,
            10,
            20,
            40
        ],

        "min_samples_split":[
            2,
            5,
            10
        ],

        "min_samples_leaf":[
            1,
            2,
            4
        ],

        "max_features":[
            "sqrt",
            "log2"
        ]
    }


    rf_grid = GridSearchCV(

        RandomForestClassifier(

            random_state=42,
        ),

        rf_params,

        cv=5,

        scoring="f1_weighted",

        n_jobs=-1
    )


    rf_grid.fit(
        X_train,
        y_train
    )

    print("Best RF Params:",rf_grid.best_params_)

    print(
        "Best RF CV F1:",
        rf_grid.best_score_
    )


    # =================================================
    # SVM
    # =================================================

    print("\n--- SVM Tuning ---")


    svm_params={

        "C":[
            0.01,
            0.1,
            1,
            10,
            50
        ],

        "kernel":[
            "linear",
            "rbf",
            "poly"
        ],

        "gamma":[
            "scale",
            "auto"
        ],

        "degree":[
            2,
            3
        ]
    }


    svm_grid=GridSearchCV(

        SVC(

            probability=True,

            random_state=42,
        ),

        svm_params,

        cv=5,

        scoring="f1_weighted",

        n_jobs=-1
    )


    svm_grid.fit(
        X_train,
        y_train
    )

    print(
        "Best SVM Params:",
        svm_grid.best_params_
    )

    print(
        "Best SVM CV F1:",
        svm_grid.best_score_
    )


    # =================================================
    # LOGISTIC REGRESSION
    # FIXED (NO multi_class)
    # =================================================

    print("\n--- Logistic Regression Tuning ---")


    lr_params={

        "C":[
            0.01,
            0.1,
            1,
            10
        ],

        "solver":[
            "lbfgs",
            "saga"
        ]
    }


    lr_grid=GridSearchCV(

        LogisticRegression(

            max_iter=5000,
            random_state=42
        ),

        lr_params,

        cv=5,

        scoring="f1_weighted",

        n_jobs=-1
    )


    lr_grid.fit(
        X_train,
        y_train
    )


    print(
        "Best LR Params:",
        lr_grid.best_params_
    )

    print(
        "Best LR CV F1:",
        lr_grid.best_score_
    )


    # =================================================
    # PICK BEST MODEL
    # =================================================

    candidates=[

        ("RF",rf_grid.best_estimator_,rf_grid.best_score_),

        ("SVM",svm_grid.best_estimator_,svm_grid.best_score_),

        ("LR",lr_grid.best_estimator_,lr_grid.best_score_)
    ]


    best_name,best_model,best_score=max(

        candidates,

        key=lambda x:x[2]
    )


    print("\nBest Model:",best_name)

    print("Best CV F1:",best_score)


    # =================================================
    # FINAL EVAL
    # =================================================

    y_pred=best_model.predict(
        X_test
    )


    print("\nClassification Report:\n")

    print(

        classification_report(

            y_test,

            y_pred,

            zero_division=0
        )
    )


    # =================================================
    # CONFUSION MATRIX
    # =================================================

    cm=confusion_matrix(

        y_test,

        y_pred,

        labels=[
            "Positive",
            "Negative",
            "Neutral"
        ]
    )


    plt.figure(
        figsize=(8,6)
    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        xticklabels=[
            "Positive",
            "Negative",
            "Neutral"
        ],

        yticklabels=[
            "Positive",
            "Negative",
            "Neutral"
        ]
    )


    plt.title(
        "Confusion Matrix"
    )


    plt.savefig(
        "confusion_matrix.png"
    )

    print(
        "Saved confusion_matrix.png"
    )


    # =================================================
    # ERROR ANALYSIS
    # =================================================

    failed_idx=np.where(
        y_pred!=y_test
    )[0]


    errors_df=pd.DataFrame({

        "Text":
        np.array(X_text_test)[failed_idx],

        "Actual":
        np.array(y_test)[failed_idx],

        "Predicted":
        np.array(y_pred)[failed_idx]
    })


    errors_df.to_csv(

        "error_analysis.csv",

        index=False
    )


    print(
        f"Saved error_analysis.csv ({len(errors_df)} errors)"
    )


    # =================================================
    # SAVE MODEL
    # =================================================

    joblib.dump(

        {

            "model":best_model,

            "glove":glove_dict,

            "tfidf": tfidf_vectorizer 

        },

        "sentiment_pipeline.pkl"
    )


    print(
        "Saved sentiment_pipeline.pkl"
    )



if __name__=="__main__":

    optimize_and_evaluate()