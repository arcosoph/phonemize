from pathlib import Path
import torch
from phonemize.utils.io import read_config, pickle_binary
from phonemize.preprocessing.text import Preprocessor
from phonemize.model.model import create_model, ModelType
from phonemize.training.trainer import Trainer


def main():
    torch.set_num_threads(1)
    cfg_path = Path('phonemize/configs/forward_config.yaml')
    config = read_config(str(cfg_path))

    # make the model small for a smoke run
    config['model']['d_model'] = 64
    config['model']['d_fft'] = 128
    config['model']['layers'] = 2
    config['model']['heads'] = 2

    # short training run
    config['training']['epochs'] = 1
    config['training']['batch_size'] = 2
    config['training']['batch_size_val'] = 2
    config['training']['generate_steps'] = 1000000
    config['training']['validate_steps'] = 1000000
    config['training']['checkpoint_steps'] = 1000000

    data_dir = Path(config['paths']['data_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = Preprocessor.from_config(config)

    # create small synthetic datasets
    # create enough synthetic items so the binned sampler can form bins
    items = []
    for i in range(24):
        w = f'word{i}'
        phon = 'aa'
        processed = preprocessor(('en_us', w, phon))
        items.append(processed)

    # save datasets and phoneme dict
    pickle_binary(items, data_dir / 'train_dataset.pkl')
    # ensure val set is large enough for the binned sampler
    pickle_binary(items[:12], data_dir / 'val_dataset.pkl')
    pickle_binary({'dummy': 'value'}, data_dir / 'phoneme_dict.pkl')

    # build model and trainer
    model_type = ModelType(config['model']['type'])
    model = create_model(model_type, config)

    checkpoint_dir = Path('checkpoints_smoke')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loss_type = 'cross_entropy' if model_type.is_autoregressive() else 'ctc'
    trainer = Trainer(checkpoint_dir=checkpoint_dir, device=torch.device('cpu'), rank=0, use_ddp=False, loss_type=loss_type)

    checkpoint = {'preprocessor': preprocessor, 'config': config}

    try:
        trainer.train(model=model, checkpoint=checkpoint, store_phoneme_dict_in_model=True)
        print('Smoke training run completed successfully')
    except Exception as e:
        print('Smoke training run failed with exception:')
        raise


if __name__ == '__main__':
    main()
