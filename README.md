
<h1>
Automated Histology Report Generation via Concept-Level Interpretability and Foundation Models
</h1>

## Env setup
- conda create -n wsi_rgen python=3.11
- conda activate wsi_rgen
- pip install -r requirements.txt

## Extract and save CONCH, CONCH1.5 and TITAN features
Use [TRIDENT](https://github.com/mahmoodlab/TRIDENT) to create and save foundation model features for the train and test datasets in h5 format

## Train and extract GECKO deep and concept features
Use following scripts from [GECKO](https://github.com/surykntsingh/GECKO)
- Generate concept priors with CONCH features using [conch_feat_deep_to_cosine_sim_proj.py](https://github.com/surykntsingh/GECKO/blob/main/conch_feat_deep_to_cosine_sim_proj.py) and [curate_cosinesim_conch.py](https://github.com/surykntsingh/GECKO/blob/main/curate_cosinesim_conch.py)
- Train gecko model:
```  python train_gecko.py \
--keep_ratio 0.7 \
--top_k 10 \
--features_deep_path <your_conch_feature_path> \
--features_path <your_concept_prior_path> \
--save_path <your_checkpont_save_path> \
--batch_size 128 \
--warmup_epochs 5 \
--epochs 50
```
- Infer and save gecko features:
```
CUDA_VISIBLE_DEVICES=6 python infer_gecko.py \
--features_deep_path <your_conch_feature_path> \
--features_path <your_concept_prior_path> \
--out_path <your_output_path> \
--max_n_tokens 2048 \
--model_weights_path <your_trained_model_checkpoint_path>
```
## Train, test and predict Report generation
- Set appropriate paths in config.yaml
- Train and test and predict the report gen model with following command: 
```torchrun --nproc-per-node=6 --nnodes=1 main.py train```
- Repeat 3-5 experiments with perturbation by changing hyperparams like `num_layers, d_vf, d_ff, dropout etc` and generate corresponding predictions.
- Collate the generated reports to a single final report by using the maximum agreement logic using following command:
```
python main.py collate-predictions --prediction-paths <'predictions_1.json,predictions_2.json, ...'> --output_path <your_output_path>
```

## Acknowledgements
This project builds on  [CONCH](https://github.com/mahmoodlab/CONCH), [TITAN](https://github.com/mahmoodlab/TITAN), [TRIDENT](https://github.com/mahmoodlab/TRIDENT), [GECKO](https://github.com/bmi-imaginelab/GECKO) and [Wsi-Caption](https://github.com/cpystan/Wsi-Caption). We thank the authors for their contribution.
