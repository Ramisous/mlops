import json
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path

import joblib
import mlflow
import optuna
import pandas as pd
import typer
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

from config import config
from tagifai import data, predict, train, utils

warnings.filterwarnings("ignore")

app = typer.Typer()


@app.command()
def load_data():
    """Load data from URLs and save to local drive."""
    # Download data assets
    projects = utils.load_json_from_url(url=config.PROJECTS_URL)
    projects_fp = Path(config.DATA_DIR, "projects.json")
    utils.save_dict(d=projects, filepath=projects_fp)

    # Download auxiliary data
    tags = utils.load_json_from_url(url=config.TAGS_URL)
    tags_fp = Path(config.DATA_DIR, "tags.json")
    utils.save_dict(d=tags, filepath=tags_fp)

    print("✅ Saved raw data!")


@app.command()
def label_data(args_fp):
    """Label data with constraints."""
    # Load projects

    projects_fp = Path(config.DATA_DIR, "projects.json")
    projects = utils.load_dict(filepath=projects_fp)
    df = pd.DataFrame(projects)

    # Load tags
    tags_dict = {}
    tags_fp = Path(config.DATA_DIR, "tags.json")
    for item in utils.load_dict(filepath=tags_fp):
        key = item.pop("tag")
        tags_dict[key] = item

    # Label with constrains
    args = Namespace(**utils.load_dict(filepath=args_fp))
    df = data.replace_oos_labels(df=df, labels=tags_dict.keys(), label_col="tag", oos_label="other")
    df = data.replace_minority_labels(
        df=df, label_col="tag", min_freq=args.min_freq, new_label="other"
    )

    # Save clean labeled data
    labeled_projects_fp = Path(config.DATA_DIR, "labeled_projects.json")
    utils.save_dict(d=df.to_dict(orient="records"), filepath=labeled_projects_fp)
    print("✅ Saved labeled data!")


@app.command()
def train_model(args_fp, experiment_name, run_name):
    """Train a model given arguments."""
    # Load labeled data
    projects_fp = Path(config.DATA_DIR, "labeled_projects.json")
    projects = utils.load_dict(filepath=projects_fp)
    df = pd.DataFrame(projects)

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        print(json.dumps(performance, indent=2))

        print("trying", vars(artifacts["args"]))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            utils.save_dict(vars(artifacts["args"]), Path(dp, "args.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


@app.command()
def optimize(args_fp, study_name, num_trials):
    """Optimize hyperparameters."""
    # Load labeled data
    projects_fp = Path(config.DATA_DIR, "labeled_projects.json")
    projects = utils.load_dict(filepath=projects_fp)
    df = pd.DataFrame(projects)

    args = Namespace(**utils.load_dict(filepath=args_fp))
    # Optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name="optimization", direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    # print(train.objective(args, df, trial))
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    utils.save_dict({**args.__dict__, **study.best_trial.params}, args_fp, cls=NumpyEncoder)
    print(f"\nBest value (f1): {study.best_trial.value}")
    print(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")


@app.command()
def predict_tag(text, run_id=None):
    """Predict tag for text."""
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    print(json.dumps(prediction, indent=2))


@app.command()
def load_artifacts(run_id):
    """Load artifacts for a given run_id."""
    # Load arguments used for this specific run
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))

    # Load objects from run
    client = mlflow.tracking.MlflowClient()
    with tempfile.TemporaryDirectory() as dp:
        client.download_artifacts(run_id=run_id, path="", dst_path=dp)
        vectorizer = joblib.load(Path(dp, "vectorizer.pkl"))
        label_encoder = data.LabelEncoder.load(fp=Path(dp, "label_encoder.json"))
        model = joblib.load(Path(dp, "model.pkl"))
        performance = utils.load_dict(filepath=Path(dp, "performance.json"))

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }


if __name__ == "__main__":
    app()
