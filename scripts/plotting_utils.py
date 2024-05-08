import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from IPython.display import Audio, display

def plot_waveforms(train_ds, class_names, preprocessing_params, old_num_train_samples, num_train_samples, n):

    target_sr = preprocessing_params['TARGET_SR']

    if preprocessing_params['DATA_AUGMENTATION']:
        print(f"Number of training samples before augmentation: {old_num_train_samples}")
        print(f"Number of training samples after augmentation: {num_train_samples}")
        # Compare the first 6 original and augmented audio clips
        for (audio, label), (audio_augmented, _) in zip(train_ds.take(n), train_ds.skip(old_num_train_samples).take(n)):
            print(f"Audio shape (original): {audio.shape}")
            print(f"Audio shape (augmented): {audio_augmented.shape}")
            display(Audio(audio, rate=target_sr))
            display(Audio(audio_augmented, rate=target_sr))
            plt.figure(figsize=(10, 2))
            plt.plot(audio)
            plt.title(f"Class: {class_names[np.argmax(label)]}")
            plt.show()
            plt.figure(figsize=(10, 2))
            plt.plot(audio_augmented)
            plt.title(f"Augmented audio")
            plt.show()
    else:
        print(f"Number of training samples: {num_train_samples}")
        # Inspect the first 6 audio clips in the training set
        for audio, label in train_ds.take(n):
            print(f"Audio shape: {audio.shape}")
            display(Audio(audio, rate=target_sr))
            plt.figure(figsize=(10, 2))
            plt.plot(audio)
            plt.title(f"Class: {class_names[np.argmax(label)]}")
            plt.show()

def plot_transforms(train_ds, class_names, preprocessing_params, n):
    # Determine the number of 2D and 1D transforms from the preprocessing parameters
    num_2d_transforms = len(preprocessing_params['2D_TRANSFORMS'])
    num_1d_transforms = len(preprocessing_params['1D_TRANSFORMS'])
    total_transforms = num_2d_transforms + num_1d_transforms

    # Define color maps for 2D transforms
    color_map_dict = {
        'spectrogram': 'inferno',
        'mel_spectrogram': 'inferno',
        'log_mel_spectrogram': 'inferno',
        'log_mel_spectrogram_librosa': 'inferno',
        'mfcc': 'plasma',
        'mfcc_librosa': 'plasma',
        'scalogram': 'coolwarm'
    }

    # Start plotting
    plt.figure(figsize=(18, n * 2.5 * total_transforms))
    
    for i, (transforms, label) in enumerate(train_ds.take(n)):
        class_label = class_names[np.argmax(label)]
        # Set a global title for the sample
        
        # Iterate over each transform for the current sample
        for t, transform in enumerate(transforms):
            ax = plt.subplot(n, total_transforms, i * total_transforms + t + 1)
            transform_name = preprocessing_params['2D_TRANSFORMS'][t] if t < num_2d_transforms else preprocessing_params['1D_TRANSFORMS'][t - num_2d_transforms]
            transform_title = f'{class_label} - {transform_name}'
            ax.set_title(transform_title)
            
            if t < num_2d_transforms:  # If the transform is a 2D transform
                data = tf.squeeze(transform).numpy()
                color_map = color_map_dict[transform_name]
                img = plt.imshow(data.T, aspect='equal', cmap=color_map, origin='lower')
                plt.colorbar(img, ax=ax)
                plt.xlabel('Time')
                plt.ylabel('Frequency')
            else:  # If the transform is a 1D transform
                data = tf.squeeze(transform).numpy()
                plt.plot(data)
                plt.xlabel('Samples')
                plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()