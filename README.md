# Retroformer
This is the directory of the work [Retroformer: Pushing the Limits of Interpretable End-to-end Retrosynthesis Transformer](https://arxiv.org/abs/2201.12475).



## Dependency:
Follow the below steps for dependency installation:
```
conda create -y -n retroformer tqdm
conda activate retroformer
conda install pytorch=1.10.1 torchvision cudatoolkit=11.0 -c pytorch
conda install -y rdkit -c conda-forge
```
The `rdchiral` package is taken from [here](https://github.com/connorcoley/rdchiral) (no need to install it).

## Directory overview:
The overview of the full directory looks like this:
```
Retroformer/
├── rdchiral/
├── models/
├── utils/
└── data/
    ├── uspto50k/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    └── .../
└── intermediate/
    └── vocab_share.pk
└── checkpoint/
    ├── model_untyped_best.pt
    └── model_typed_best.pt   
├── result/
├── dataset.py
├── start.sh
├── train.py
├── translate.sh
└── translate.py     
```

## Data:
Download the raw reaction dataset from [here](https://github.com/Hanjun-Dai/GLN) and put it into your data directory. You can also create your own reaction dataset as long as the data shares the same format (columns: `id`, `class`, `reactants>reagents>production`) and the reactions are atom-mapped.  

## Train:
One can specify different model and training configurations in `start.sh`. Below is a sample code that calls `train.py`. Simply run `./start.sh` for training.

Data processing is done at the stage of building data iterator. If the training is called for the first time, it will take extra time to buildup vocab file and save it to `/intermediate/vocab.pk`.

```
python train.py \
  --num_layers 8 \
  --heads 8 \
  --max_step 150000 \
  --batch_size_token 4096 \
  --save_per_step 2500 \
  --val_per_step 2500 \
  --report_per_step 200 \
  --device cuda \
  --known_class False \
  --shared_vocab True \
  --shared_encoder False \
  --data_dir <data_folder> \
  --intermediate_dir <intermediate_folder> \
  --checkpoint_dir <checkpoint_folder> \
  --checkpoint <previous_checkpoint> 
```

## Inference:
One can specify different inference configurations in `translate.sh` as the sample code below. Simply sun `./translate.sh` for inference. 

To replicate our results, download the pre-trained checkpoints from [GoogleDrive](https://drive.google.com/drive/folders/1kiar6EhTInHBJpZLhPbrQ6dMcUuTfN39?usp=sharing).

Special arguments:
- `stepwise`: determines whether to use _naive_ strategy or _search_ strategy.
- `use_template`: determines whether to use pre-computed reaction center to accelarate the _search_ strategy (corresponds to the reaction center retrieval setting in paper).  

```
python translate.py \
  --batch_size_val 8 \
  --shared_vocab True \
  --shared_encoder False \
  --data_dir <data_folder> \
  --intermediate_dir <intermediate_folder> \
  --checkpoint_dir <checkpoint_folder> \
  --checkpoint <target_checkpoint> \
  --known_class False \
  --beam_size 10 \
  --stepwise False \
  --use_template False
```
