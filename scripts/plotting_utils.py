import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

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