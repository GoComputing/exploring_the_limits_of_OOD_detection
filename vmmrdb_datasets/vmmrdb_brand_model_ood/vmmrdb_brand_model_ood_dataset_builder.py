"""vmmrdb_brand_model_ood dataset."""

import tensorflow_datasets as tfds
import pandas as pd
import os


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for vmmrdb_brand_model_ood dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  DATASET_DIR = '/mnt/homeGPU/aurrea_cpelaez/datasets/VMMRdb_splits/brand_model'

  def _load_df(self):
    split_path = os.path.join(self.DATASET_DIR, 'ood_test.csv')
    df = pd.read_csv(split_path, sep=',', index_col=0)
    df = df[df['is_ood'] == 1]
    return df

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # Specifies the tfds.core.DatasetInfo object
    # Load classes names
    df = self._load_df()
    names = list(df['class'].unique())
    names.sort()

    # Return
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset
            'image': tfds.features.Image(shape=(None, None, 3)),
            'label': tfds.features.ClassLabel(names=names)
        }),
        supervised_keys=('image', 'label'),
        homepage='https://github.com/faezetta/VMMRdb',
        disable_shuffling=False
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(vmmrdb_brand_model_ood): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')

    generators = dict()
    df = self._load_df()
    generators['ood_test'] = self._generate_examples(df.to_dict(orient='index'))

    return generators

  def _generate_examples(self, data):
    """Yields examples."""
    for row_id, row in data.items():
      yield row_id, {
          'image': row['path'],
          'label': row['class']
      }
