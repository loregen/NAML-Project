import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import pathlib
from typing import Any, Tuple


# Wrapper function that loads the selected dataset
def load_dataset(preprocess_params: dict[str, Any]) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray]:
    
    # Wrapper function to load audio files. This is needed because librosa is not directly compatible with TensorFlow operations.
    def load_audio(file_path, label, sample_rate):

        def _load_audio(file_path, label):
            audio, _ = librosa.load(file_path.numpy(), sr=sample_rate, mono=True)
            return audio.astype(np.float32), np.array(label).astype(np.int64)
        
        [audio, label] = tf.py_function(
            func=_load_audio,
            inp=[file_path, label],
            Tout=[tf.float32, tf.int64]
        )

        audio.set_shape([None])
        label.set_shape([None])

        return audio, label

    def load_UrbanSound8K(path: pathlib.Path, sample_rate: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray]:

        # Load the metadata
        metadata_path = path / "metadata/UrbanSound8K.csv"
        audio_path = path / "audio"
        metadata = pd.read_csv(metadata_path)

        # Get the number of classes
        num_classes = metadata['classID'].nunique()

        # Initialize lists to hold file paths and labels
        file_paths_train = []
        labels_train = []
        file_paths_test = []
        labels_test = []

        test_fold = preprocess_params['URBANSOUND8K_TESTFOLD']

        # Shuffle the metadata (optional)
        metadata = metadata.sample(frac=1, random_state=45)

        # Iterate over the metadata entries
        for _, row in metadata.iterrows():
            # Construct the file path for the current entry
            fold_number = row['fold']
            fold_name = f"fold{fold_number}"
            file_name = row['slice_file_name']
            file_path = audio_path / fold_name / file_name

            # Append the file path and label to the lists
            if fold_number == test_fold:
                file_paths_test.append(str(file_path))
                labels_test.append(row['classID'])
            else:
                file_paths_train.append(str(file_path))
                labels_train.append(row['classID'])

        # Convert lists to tensors
        file_paths_train_tensor = tf.convert_to_tensor(file_paths_train, dtype=tf.string)
        labels_train_tensor = tf.convert_to_tensor(labels_train, dtype=tf.int64)
        file_paths_test_tensor = tf.convert_to_tensor(file_paths_test, dtype=tf.string)
        labels_test_tensor = tf.convert_to_tensor(labels_test, dtype=tf.int64)

        # One-hot encode the labels
        one_hot_labels_train_tensor = tf.one_hot(labels_train_tensor, num_classes)
        one_hot_labels_test_tensor = tf.one_hot(labels_test_tensor, num_classes)

        # Create a dataset from tensors
        train_dataset = tf.data.Dataset.from_tensor_slices((file_paths_train_tensor, one_hot_labels_train_tensor))
        test_dataset = tf.data.Dataset.from_tensor_slices((file_paths_test_tensor, one_hot_labels_test_tensor))

        # Map the wrapped load_audio function to the dataset
        train_dataset = train_dataset.map(lambda file_path, label: load_audio(file_path, label, sample_rate))
        test_dataset = test_dataset.map(lambda file_path, label: load_audio(file_path, label, sample_rate))

        # Construct class names array corresponding to the one-hot labels
        unique_classes = metadata[['classID', 'class']].drop_duplicates().sort_values('classID')
        class_names = unique_classes['class'].to_numpy()

        return train_dataset, test_dataset, class_names

    def load_ESC_10(path: pathlib.Path, sample_rate: int, whole_ESC_50: bool) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray]:

        # Load the metadata
        metadata_path = path / "meta/esc50.csv"
        audio_path = path / "audio"
        metadata = pd.read_csv(metadata_path)

        # Conditionally restrict the metadata to the ESC-10 subset
        metadata = metadata if whole_ESC_50 else metadata[metadata['esc10']]

        num_classes = metadata['category'].nunique()

        # Initialize lists to hold file paths and labels
        file_paths_train = []
        labels_train = []
        file_paths_test = []
        labels_test = []

        test_fold = preprocess_params['ESC_TESTFOLD']

        # Shuffle the metadata (optional)
        metadata = metadata.sample(frac=1, random_state=42)

        # Iterate over the metadata entries
        for _, row in metadata.iterrows():
            # Construct the file path for the current entry
            fold_number = row['fold']
            file_name = row['filename']
            file_path = audio_path / file_name

            # Append the file path and label to the lists
            if fold_number == test_fold:
                file_paths_test.append(str(file_path))
                labels_test.append(row['category'])
            else:
                file_paths_train.append(str(file_path))
                labels_train.append(row['category'])

        # Construct class names array
        unique_classes = metadata['category'].unique()
        class_names = np.array(unique_classes)

        # Create a mapping from category to integer label
        category_to_int_mapping = {category: i for i, category in enumerate(class_names)}

        # Map the category to integer label
        labels_train = [category_to_int_mapping[category] for category in labels_train]
        labels_test = [category_to_int_mapping[category] for category in labels_test]

        # Convert lists to tensors
        file_paths_train_tensor = tf.convert_to_tensor(file_paths_train, dtype=tf.string)
        labels_train_tensor = tf.convert_to_tensor(labels_train, dtype=tf.int64)
        file_paths_test_tensor = tf.convert_to_tensor(file_paths_test, dtype=tf.string)
        labels_test_tensor = tf.convert_to_tensor(labels_test, dtype=tf.int64)

        # One-hot encode the labels
        one_hot_labels_train_tensor = tf.one_hot(labels_train_tensor, num_classes)
        one_hot_labels_test_tensor = tf.one_hot(labels_test_tensor, num_classes)

        # Create a dataset from tensors
        train_dataset = tf.data.Dataset.from_tensor_slices((file_paths_train_tensor, one_hot_labels_train_tensor))
        test_dataset = tf.data.Dataset.from_tensor_slices((file_paths_test_tensor, one_hot_labels_test_tensor))

        # Map the wrapped load_audio function to the dataset
        train_dataset = train_dataset.map(lambda file_path, label: load_audio(file_path, label, sample_rate))
        test_dataset = test_dataset.map(lambda file_path, label: load_audio(file_path, label, sample_rate))

        return train_dataset, test_dataset, class_names

    # Retrieve the dataset loading parameters
    dataset_name = preprocess_params['DATASET_NAME']
    sample_rate = preprocess_params['TARGET_SR']
    whole_ESC_50 = preprocess_params['WHOLE_ESC_50']

    path_dict = {
    'UrbanSound8K': pathlib.Path("UrbanSound8K"),
    'ESC-10': pathlib.Path("ESC-50-master")
    }
    if dataset_name not in path_dict:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    if dataset_name == 'UrbanSound8K':
        return load_UrbanSound8K(path_dict[dataset_name], sample_rate)
    elif dataset_name == 'ESC-10':
        return load_ESC_10(path_dict[dataset_name], sample_rate, whole_ESC_50)