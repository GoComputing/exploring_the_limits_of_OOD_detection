"""vmmrdb_brand_model_year dataset."""

import tensorflow_datasets as tfds
import pandas as pd
import os


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for vmmrdb_brand_model_year dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  DATASET_DIR = '/mnt/homeGPU/aurrea_cpelaez/datasets/VMMRdb_splits/brand_model_year'

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # Specifies the tfds.core.DatasetInfo object
    # Load classes names
    classes_path = os.path.join(self.DATASET_DIR, 'classes.txt')
    with open(classes_path, 'r') as f:
        names = list(map(lambda line: line.strip(), f.readlines()))

    # Return
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset
            'image': tfds.features.Image(shape=(None, None, 3)),
            'label': tfds.features.ClassLabel(names=names),
        }),
        supervised_keys=('image', 'label'),
        homepage='https://github.com/faezetta/VMMRdb',
        disable_shuffling=False
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(vmmrdb_brand_model_year): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')

    generators = dict()
    for split in ['train', 'val', 'test', 'ood_test']:
        split_path = os.path.join(self.DATASET_DIR, f'{split}.csv')
        df = pd.read_csv(split_path, sep=',', index_col=0)
        if split == 'ood_test':
            df = df[df['is_ood'] == 0]
        generators[split] = self._generate_examples(df.to_dict(orient='index'))

    return generators

  def _generate_examples(self, data):
    """Yields examples."""
    for row_id, row in data.items():
      yield row_id, {
          'image': row['path'],
          'label': row['class'],
      }
