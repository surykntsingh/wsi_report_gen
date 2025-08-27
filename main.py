import typer
import os
import torch
from datamodules.wsi_embedding_datamodule import PatchEmbeddingDataModule, PatchEmbeddingDataPredictModule
from models import ReportModel
from modules.metrics import REG_Evaluator
from report_tokenizers import Tokenizer
from trainer import Trainer, KFoldTrainer
import pandas as pd
import numpy as np
from utils.utils import save_model, get_params_for_key, copy_yaml, write_json_file, read_json_file
from datetime import datetime
from tqdm.auto import tqdm

app = typer.Typer()




@app.command()
def train(config_file_path: str='config.yaml', reg_threshold: float=0.8, save_model_pt=True):
    args = get_params_for_key(config_file_path, "train")
    split_frac = [0.8, 0.1, 0.1]
    tokenizer = Tokenizer(args.reports_json_path)
    model = ReportModel(args, tokenizer)

    datamodule = PatchEmbeddingDataModule(args, tokenizer, split_frac)
    date = datetime.now()
    args.ckpt_path +=  f'/{date.strftime("%Y%m%d")}/{date.strftime("%H%M%S")}'
    os.makedirs(args.ckpt_path, exist_ok=True)
    trainer = Trainer(args, tokenizer, split_frac)
    train_metrics, _ = trainer.train(model, datamodule, fast_dev_run=args.fast_dev_run)
    print('model training finished')

    best_model_path = trainer.best_model_path
    # if not args.fast_dev_run:
    print(f'loading best model from {best_model_path}')
    best_model = ReportModel.load_from_checkpoint(best_model_path, args=args, tokenizer=tokenizer)
    test_metrics, tr = trainer.test(best_model, datamodule, fast_dev_run=args.fast_dev_run)
    print('model testing finished')
    # save_model(args, trainer)


    metrics = {**train_metrics, **test_metrics, 'best_model_path': best_model_path}
    print(f'train_metrics: {train_metrics}, test_metrics: {test_metrics})') #, tune_metrics: {tune_metrics}')

    copy_yaml(config_file_path, args.ckpt_path)
    os.makedirs(f'{args.ckpt_path}/results', exist_ok=True)
    write_metrics(f'{args.ckpt_path}/results', metrics, date)

    os.makedirs(f'{args.results_path}/experiments', exist_ok=True)
    write_metrics(f'{args.results_path}/experiments', metrics, date)

    if test_metrics['test_reg'].item() > reg_threshold:
        print(f'Generating predictions since reg_score > {reg_threshold}')
        results = get_prediction(best_model, trainer, args, tokenizer)
        results_dir = f'{args.ckpt_path}/results_{test_metrics["test_reg"]}'
        print(f'Saving predictions at {results_dir}')
        save_results(results, results_dir)
        print(f'Predictions saved at {results_dir}')
        # os.makedirs(f'{args.ckpt_path}/saved_models', exist_ok=True)
        # save_model_path = f'{args.ckpt_path}/saved_models/reg_{test_metrics["test_reg"]}.ckpt'
        # trainer.save_model(tr, save_model_path)
    else:
        print(f'Not generating predictions since reg_score < {reg_threshold}')

    # tune_gecko_features(args, tokenizer, best_model_path, trainer, datamodule, reg_threshold, date)

    if save_model_pt:
        torch.save(model.state_dict(), f"{args.model_save_path}/model_weights.pt")



def tune_gecko_features(args, tokenizer,best_model_path, trainer, datamodule, reg_threshold, date):
    # gecko tuning
    print('tuning on gecko concept features')
    model = ReportModel.load_from_checkpoint(best_model_path, args=args, tokenizer=tokenizer)
    tune_metrics, _ = trainer.tune(model, datamodule, fast_dev_run=args.fast_dev_run)
    print('model tuning finished')
    print(f'tune_metrics: {tune_metrics}')

    best_model_path = trainer.best_model_path
    # if not args.fast_dev_run:
    print(f'loading best model from {best_model_path}')
    best_model = ReportModel.load_from_checkpoint(best_model_path, args=args, tokenizer=tokenizer)
    test_metrics, tr = trainer.test(best_model, datamodule, fast_dev_run=args.fast_dev_run)
    print('model testing finished')
    # save_model(args, trainer)

    metrics = {**tune_metrics, **test_metrics, 'best_model_path': best_model_path}
    print(f'tune_metrics: {tune_metrics}, test_metrics: {test_metrics})')
    write_metrics(f'{args.ckpt_path}/results', metrics, date)
    write_metrics(f'{args.results_path}/experiments', metrics, date)

    if test_metrics['test_reg'].item() > reg_threshold:
        print(f'Generating predictions since reg_score > {reg_threshold}')
        results = get_prediction(best_model, trainer, args, tokenizer)
        results_dir = f'{args.ckpt_path}/results_{test_metrics["test_reg"]}'
        print(f'Saving predictions at {results_dir}')
        save_results(results, results_dir)
        print(f'Predictions saved at {results_dir}')
        # os.makedirs(f'{args.ckpt_path}/saved_models', exist_ok=True)
        # save_model_path = f'{args.ckpt_path}/saved_models/reg_{test_metrics["test_reg"]}.ckpt'
        # trainer.save_model(tr, save_model_path)
    else:
        print(f'Not generating predictions since reg_score < {reg_threshold}')


@app.command()
def test(config_file_path: str='config.yaml', reg_threshold: float=0.8,
         save_model_pt=False):

    args = get_params_for_key(config_file_path, "train")
    split_frac = [0.7, 0.25, 0.05]
    tokenizer = Tokenizer(args.reports_json_path)
    datamodule = PatchEmbeddingDataModule(args, tokenizer, split_frac)
    trainer = Trainer(args, tokenizer, split_frac)

    print(f'loading best model from {args.model_load_path}')
    model = ReportModel.load_from_checkpoint(args.model_load_path, args=args, tokenizer=tokenizer)
    test_metrics, tr = trainer.test(model, datamodule, fast_dev_run=args.fast_dev_run)
    print(f'test_metrics: {test_metrics}')
    print('model testing finished')

    # if test_metrics['test_reg'].item() > reg_threshold:
    #     print(f'Generating predictions since reg_score > {reg_threshold}')
    #     results = get_prediction(model, trainer, args, tokenizer)
    #     results_dir = f'{args.results_path}/results_{test_metrics["test_reg"]}'
    #     print(f'Saving predictions at {results_dir}')
    #     save_results(results, results_dir)
    # else:
    #     print(f'Not generating predictions since reg_score < {reg_threshold}')

    if save_model_pt:
        os.makedirs(args.model_save_path, exist_ok=True)
        torch.save(model.state_dict(), f"{args.model_save_path}/model_weights.pt")



@app.command()
def trainkfold(config_file_path='config.yaml'):
    args = get_params_for_key(config_file_path, "train")
    split_frac = [0.85, 0.15]
    tokenizer = Tokenizer(args.reports_json_path)
    # model = ReportModel(args, tokenizer)
    date = datetime.now()
    args.ckpt_path +=  f'/{date.strftime("%Y%m%d")}/{date.strftime("%H%M%S")}'

    trainer = KFoldTrainer(args, tokenizer, split_frac)
    trainer.train_and_test(fast_dev_run=args.fast_dev_run)

    metrics = trainer.get_metrics()
    print(f'metrics: {metrics}')
    write_metrics(f'{args.results_path}/experiments', metrics, date)


@app.command()
def predict(config_file_path='config.yaml'):
    args = get_params_for_key(config_file_path, "train")
    tokenizer = Tokenizer(args.reports_json_path)
    model = ReportModel.load_from_checkpoint(args.model_load_path, args=args, tokenizer=tokenizer)

    split_frac = [0.85, 0.15]
    trainer = Trainer(args, tokenizer, split_frac)
    results = get_prediction(model, trainer, args, tokenizer)
    results_dir = f'{args.results_path}/predicted_results'
    os.makedirs(results_dir, exist_ok=True)

    save_results(results, results_dir)

@app.command()
def collate_predictions(
        prediction_paths: str, #comma seperated paths
        output_path: str
):
    predictions = [ read_json_file(p) for p in prediction_paths.split(',')]
    case_ids = [p['id'] for p in predictions[0]]

    r = REG_Evaluator()
    def get_similarity(t1, t2):
        return r.evaluate_dummy([(t1, t2)])

    predictions = [{p['id']:p['report']} for p in predictions]
    cases_predictions = {i: [p[i] for p in predictions] for i in case_ids}
    consensus_pred = {}
    pbar = tqdm(cases_predictions.items(), total=len(cases_predictions))
    for i, (case_id, preds) in enumerate(pbar):
        result = choose_best_prediction(case_id, preds, get_similarity)
        consensus_pred[case_id] = result

    best_prediction = [{'id': consensus_pred[i]['case_id'], 'report': consensus_pred[i]['best_prediction']} for i in case_ids]
    print(f'writing best prediction result to {output_path}')
    save_results(best_prediction, output_path)

def get_prediction(model, trainer, args, tokenizer):
    datamodule = PatchEmbeddingDataPredictModule(args, tokenizer)
    predictions = trainer.predict(model, datamodule, fast_dev_run=args.fast_dev_run)
    print('model predictions finished')
    results = []

    print(f'predictions: {predictions}')
    for slide_ids, reports in predictions:
        for i in range(args.batch_size):
            results.append({
                'id': f'{slide_ids[i]}.tiff',
                'report': reports[i]
            })

    return results

def write_metrics(results_path, metrics, date):
    metrics['date'] = date.strftime("%Y-%m-%d %H:%M:%S")
    metrics_df = pd.DataFrame([metrics])


    metrics_df.to_csv(f'{results_path}/results.csv',mode='a')

def save_results(results, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    results_path = f'{results_dir}/predictions.json'
    write_json_file(results, results_path)



def choose_best_prediction(case_id, predictions, similarity_fn):
    """
    case_id: identifier for the case
    predictions: list of str (e.g. 4 predictions for one case)
    similarity_fn: function taking (text1, text2) -> float
    returns: dict with case_id, best_prediction, index, and scores
    """
    n = len(predictions)
    sim_matrix = np.zeros((n, n))

    # Fill pairwise similarity matrix
    for i in range(n):
        for j in range(i + 1, n):
            score = similarity_fn(predictions[i], predictions[j])
            sim_matrix[i, j] = sim_matrix[j, i] = score

    # Compute average similarity for each prediction
    avg_scores = sim_matrix.sum(axis=1) / (n - 1)

    # Pick the prediction with highest average similarity
    best_idx = int(np.argmax(avg_scores))

    return {
        "case_id": case_id,
        "best_prediction": predictions[best_idx],
        "best_index": best_idx,
        "scores": avg_scores.tolist()
    }


if __name__ == "__main__":
    app()