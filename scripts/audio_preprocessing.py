import tensorflow as tf
import numpy as np
import librosa
import pywt # Import the PyWavelets package

from typing import Any


# Function that pads or truncates the audio to a fixed length. It is used for the UrbanSound8K dataset, where the audio clips have varying lengths.
def apply_resizing(dataset: tf.data.Dataset, preprocess_params: dict[str, Any]) -> tf.data.Dataset:

    def resize_audio_to_fixed_lenght(audio, preprocess_params: dict[str, Any]):

        sample_rate = preprocess_params['TARGET_SR']
        
        # Define the maximum length in seconds (4 seconds for UrbanSound8K)
        max_length_seconds = 4

        # Compute the maximum length in samples
        max_length_samples = max_length_seconds * sample_rate

        # Compute the current length and the amount of padding required
        current_length = tf.shape(audio)[0]
        padding_amount = max_length_samples - current_length

        # Use tf.cond to decide whether to pad or truncate
        padded_audio = tf.cond(
            padding_amount < 0,
            lambda: audio[:max_length_samples],  # Truncate the audio
            lambda: tf.pad(audio, paddings=[[0, padding_amount]], mode='CONSTANT', constant_values=0)  # Pad the audio
        )

        return padded_audio

    dataset = dataset.map(lambda audio, label: (resize_audio_to_fixed_lenght(audio, preprocess_params), label), num_parallel_calls=tf.data.AUTOTUNE)

    return dataset



# Function that applies the specified transform and normalization to the dataset.
def transform_normalize_dataset(audio_dataset: tf.data.Dataset, preprocess_params: dict[str, Any]) -> tf.data.Dataset:
    
    # ----------- Spectrogram functions ----------- #
    frame_length = preprocess_params['FRAME_LENGTH']
    frame_step = preprocess_params['FRAME_STEP']
    fft_length = preprocess_params['FFT_LENGTH']
    num_mel_bins = preprocess_params['NUM_MEL_BINS']
    num_mfccs = preprocess_params['NUM_MFCCS']
    target_sr = preprocess_params['TARGET_SR']

    def compute_spectrogram(waveform):

        # Convert the waveform to a spectrogram via a STFT.
        spectrogram = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    
        # Obtain the power spectrogram
        spectrogram = tf.abs(spectrogram) ** 2

        return spectrogram

    def compute_mel_spectrogram(waveform):

        spectrogram = compute_spectrogram(waveform)
        
        # Compute the mel spectrogram
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=tf.shape(spectrogram)[-1],
            sample_rate=target_sr,
            lower_edge_hertz=20.0,  # Typically 20 Hz is used for the lower edge
            upper_edge_hertz=target_sr / 2)  # Nyquist frequency
        
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(mel_spectrogram.shape[-1:]))

        # def _compute_mel_spectrogram(waveform_tf):
        #     mel_spectrogram = librosa.feature.melspectrogram(y=waveform_tf.numpy(), sr=target_sr, n_mels=num_mel_bins)
        #     return librosa.power_to_db(mel_spectrogram, ref=np.max).T

        # mel_spectrogram = tf.py_function(
        #     func=_compute_mel_spectrogram,
        #     inp=[waveform],
        #     Tout=tf.float32
        # )

        return mel_spectrogram
    
    def compute_log_mel_spectrogram(waveform):

        mel_spectrogram = compute_mel_spectrogram(waveform)

        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

        # def _tf_log10(x):
        #     numerator = tf.math.log(x)
        #     denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        #     return numerator / denominator
    
        # # Scale magnitude relative to maximum value in S. Zeros in the output 
        # # correspond to positions where S == ref.
        # ref = tf.reduce_max(mel_spectrogram)

        # amin = 1e-6
        # top_db = 80.0

        # log_mel_spectrogram = 10.0 * _tf_log10(tf.maximum(amin, mel_spectrogram))
        # log_mel_spectrogram -= 10.0 * _tf_log10(tf.maximum(amin, ref))

        # log_mel_spectrogram = tf.maximum(log_mel_spectrogram, tf.reduce_max(log_mel_spectrogram) - top_db)

        # Add a channel dimension to the log mel spectrogram
        #log_mel_spectrogram = log_mel_spectrogram[..., tf.newaxis]

        # Resize the log mel spectrogram to smaller dimensions
        #log_mel_spectrogram = tf.image.resize(log_mel_spectrogram, [128, 128])

        # Squeeze the log mel spectrogram to remove the channel dimension
        #log_mel_spectrogram = tf.squeeze(log_mel_spectrogram)

        return log_mel_spectrogram

    def compute_log_mel_spectrogram_librosa(waveform):
            
            def _compute_log_mel_spectrogram(waveform_tf):
                mel_spectrogram = librosa.feature.melspectrogram(y=waveform_tf.numpy(), sr=target_sr, n_mels=num_mel_bins)
                return librosa.power_to_db(mel_spectrogram, ref=np.max).T
            
            log_mel_spectrogram = tf.py_function(
                func=_compute_log_mel_spectrogram,
                inp=[waveform],
                Tout=tf.float32
            )
    
            return log_mel_spectrogram

    # ----------- MFCC function ----------- #
    def compute_mfcc(waveform):

        log_mel_spectrogram = compute_log_mel_spectrogram(waveform)

        # Compute MFCCs from log mel spectrograms
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_mfccs]

        return mfccs
    
        # ----------- MFCC function ----------- #
    def compute_mfcc_librosa(waveform):

        def _compute_mfcc(waveform_tf):
            return librosa.feature.mfcc(y=waveform_tf.numpy(), sr=target_sr, n_mfcc=num_mfccs).T
        
        mfccs = tf.py_function(
            func=_compute_mfcc,
            inp=[waveform],
            Tout=tf.float32
        )

        return mfccs

    # ----------- Scalogram functions ----------- #
    num_scales = preprocess_params['NUM_SCALES']
    cwt_wavelet = preprocess_params['CWT_WAVELET']
    dwt_wavelet = preprocess_params['DWT_WAVELET']

    # Compute scales array
    precomputed_scales = np.geomspace(1, 2048, num_scales)
    #precomputed_scales = np.linspace(1, 800, num_scales)
    scales_tf = tf.constant(precomputed_scales, dtype=tf.float32)

    def compute_scalogram(waveform):

        def _compute_scalogram(waveform_tf, scales):
            return pywt.cwt(waveform_tf.numpy(), scales.numpy(), cwt_wavelet)[0]

        scalogram = tf.py_function(
            func=_compute_scalogram,
            inp=[waveform, scales_tf],
            Tout=tf.float16
        )

        # Compute the absolute value of the scalogram
        scalogram = tf.abs(scalogram)

        # Add a channel dimension to the scalogram
        scalogram = scalogram[..., tf.newaxis]

        scalogram.set_shape([num_scales, waveform.shape[0], 1])

        # Resize the scalogram to smaller dimensions
        scalogram = tf.image.resize(scalogram, [num_scales, 128])

        # Squeeze the scalogram to remove the channel dimension
        scalogram = tf.squeeze(scalogram)

        # Transpose the scalogram
        scalogram = tf.transpose(scalogram)

        return scalogram

    def compute_dwt(waveform):

        def _compute_dwt(waveform_tf):
            return pywt.downcoef('a', waveform_tf.numpy(), wavelet=dwt_wavelet, level=10)
        
        dwt = tf.py_function(
            func=_compute_dwt,
            inp=[waveform],
            Tout=tf.float32
        )

        # # Normalize the DWT coefficients between 0 and 1
        # dwt = (dwt - tf.reduce_min(dwt)) / (tf.reduce_max(dwt) - tf.reduce_min(dwt))
        # Normalize the DWT coefficients using mean and standard deviation
        mean, variance = tf.nn.moments(dwt, axes=[0])
        dwt = (dwt - mean) / tf.sqrt(variance + 1e-6)

        return dwt

    def compute_mfcc_1d(waveform):
            
        mfccs = compute_mfcc(waveform)

        mfccs = per_feature_mean_std_normalize(mfccs)

        mfccs = tf.transpose(mfccs)

        # Collapse the MFCCs into a single feature dimension
        mfccs = tf.reduce_mean(mfccs, axis=0)

        return mfccs

    def compute_zcr(waveform):
        def _compute_zcr(waveform_tf):
            return librosa.feature.zero_crossing_rate(waveform_tf.numpy(), frame_length=frame_length, hop_length=frame_step)
        
        zcr = tf.py_function(
            func=_compute_zcr,
            inp=[waveform],
            Tout=tf.float32
        )

        # Normalize the ZCR with mean and standard deviation
        mean, variance = tf.nn.moments(zcr, axes=[1])
        zcr = (zcr - mean) / tf.sqrt(variance + 1e-6)

        return tf.squeeze(zcr)

    # ----------- Normalization functions ----------- #
    def min_max_normalize(image):
        # Normalize the image to the range [0, 1]
        image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
        return image

    def mean_std_normalize(image):
        # Compute mean and standard deviation
        mean = tf.reduce_mean(image)
        std = tf.math.reduce_std(image)
        # Standardize the image
        image = (image - mean) / (std + 1e-6)

        # Normalize the image to the range [0, 1]
        #image = min_max_normalize(image)
        return image

    def per_feature_mean_std_normalize(image):
        # Calculate mean and standard deviation for each feature (column)
        mean = tf.math.reduce_mean(image, axis=[1], keepdims=True)
        std = tf.math.reduce_std(image, axis=[1], keepdims=True)
        # Standardize each feature
        image = (image - mean) / (std + 1e-6)

        # Normalize the image to the range [0, 1]
        #image = min_max_normalize(image)

        return image

    # Dynamically get transform functions based on the provided parameters
    def get_transform_function(transform_name):
        # Create a dictionary to map transform names to functions
        transform_functions = {
            'mfcc': compute_mfcc,
            'mfcc_librosa': compute_mfcc_librosa,
            'spectrogram': compute_spectrogram,
            'mel_spectrogram': compute_mel_spectrogram,
            'log_mel_spectrogram': compute_log_mel_spectrogram,
            'log_mel_spectrogram_librosa': compute_log_mel_spectrogram_librosa,
            'scalogram': compute_scalogram,
            'dwt': compute_dwt,
            'mfcc_1d': compute_mfcc_1d,
            'zcr': compute_zcr
        }
        if transform_name in transform_functions:
            return transform_functions[transform_name]
        else:
            raise ValueError(f"Invalid transform value: {transform_name}")

    # Dynamically get normalization function based on the provided parameter
    def get_normalization_function(normalization_name):

        if normalization_name == 'none':
            return lambda image: image

        # Create a dictionary to map normalization names to functions
        normalization_functions = {
            'min_max': min_max_normalize,
            'mean_std': mean_std_normalize,
            'per_feature_mean_std': per_feature_mean_std_normalize
        }
        if normalization_name in normalization_functions:
            return normalization_functions[normalization_name]
        else:
            raise ValueError(f"Invalid normalization value: {normalization_name}")

    # Apply the selected transforms and normalization
    def apply_transforms_and_normalize(audio, label):
        transforms_2d = [get_transform_function(t) for t in preprocess_params['2D_TRANSFORMS']]
        transforms_1d = [get_transform_function(t) for t in preprocess_params['1D_TRANSFORMS']]
        normalization_func = get_normalization_function(preprocess_params['NORMALIZATION'])
        
        # Apply 2D transforms
        transformed_2d = [transform(audio) for transform in transforms_2d]

        # Add a channel dimension to the 2D transforms (needed for Conv2D layers)
        transformed_2d = [transform[..., tf.newaxis] for transform in transformed_2d]

        # Normalize the 2D transforms
        transformed_2d = [normalization_func(image) for image in transformed_2d]

        # Apply 1D transforms
        transformed_1d = [transform(audio) for transform in transforms_1d]
        
        # Combine the transformed data
        transformed_data = (*transformed_2d, *transformed_1d)

        return (transformed_data, label)

    # Map the apply_transforms_and_normalize function to the dataset
    dataset = audio_dataset.map(apply_transforms_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset



# Function that augments the audio dataset if specified.
def augment_audio_dataset(audio_dataset: tf.data.Dataset, preprocess_params: dict[str, Any]) -> tf.data.Dataset:

    # Function that adds the specified noises to the audio dataset.
    def add_noise_to_dataset(audio_dataset: tf.data.Dataset, noise_types: list[str], noise_levels: list[float], noise_probs: list[float], sample_rate: int) -> tf.data.Dataset:

        # Define the noise paths
        noise_paths = {
            'white': './_background_noise_/pink_noise.wav',
            'pink': './_background_noise_/white_noise.wav'
        }

        # Load the noises and convert to tf tensors
        noises = [librosa.load(noise_paths[noise_type], sr=sample_rate)[0] for noise_type in noise_types]
        noises = [tf.convert_to_tensor(noise, dtype=tf.float32) for noise in noises]

        # Get the shape of the first audio sample in the dataset
        audio_shape = tf.shape(next(iter(audio_dataset))[0])

        # Crop the noise tensors to the same length as the audio tensor
        noises = [tf.image.random_crop(noise, audio_shape) for noise in noises]

        # Scale the noise by its corresponding level
        noises = [noise * noise_level for noise, noise_level in zip(noises, noise_levels)]

        def add_noise(audio, label):

            noisy_audio = audio
            for noise, noise_prob in zip(noises, noise_probs):
                # Randomly decide whether to add this type of noise or not
                add_noise = tf.less(tf.random.uniform((), 0, 1), noise_prob)
                # Use tf.cond to decide whether to add the noise
                noisy_audio = tf.cond(add_noise, lambda: noisy_audio + noise, lambda: noisy_audio)

            # Ensure the noisy audio is within the range [-1, 1]
            noisy_audio = tf.clip_by_value(noisy_audio, -1.0, 1.0)

            return noisy_audio, label

        # Map the add_noise function to the dataset
        return audio_dataset.map(add_noise, num_parallel_calls=tf.data.AUTOTUNE)

    # Function that applies random time shifting to the audio dataset.
    def time_shift_dataset(audio_dataset: tf.data.Dataset, max_shift: float, sample_rate: float) -> tf.data.Dataset:

        # Calculate the maximum shift in samples
        max_shift_samples = int(sample_rate * max_shift)

        def time_shift(audio, label):

            # Generate a random shift from -max_shift_samples to max_shift_samples
            shift = tf.random.uniform((), -max_shift_samples, max_shift_samples, dtype=tf.int32)
            # Perform the time shift
            audio = tf.roll(audio, shift, axis=0)
            return audio, label

        # Map the time_shift function to the dataset
        return audio_dataset.map(time_shift, num_parallel_calls=tf.data.AUTOTUNE)

    # Function that applies random time stretching to the audio dataset.
    def time_stretch_dataset(audio_dataset: tf.data.Dataset, stretch_range: tuple[float, float]) -> tf.data.Dataset:

        min_stretch, max_stretch = stretch_range
        original_lenght = tf.shape(next(iter(audio_dataset))[0])[0]

        def apply_time_stretch(audio, label):

            # Generate a random time stretch factor within the specified range
            stretch_factor = tf.random.uniform([], min_stretch, max_stretch)

            # Wrap the librosa time stretch function
            def _time_stretch(waveform_tf, stretch_factor):
                return librosa.effects.time_stretch(waveform_tf.numpy(), rate=stretch_factor.numpy())

            # Apply time stretch with tf.py_function
            audio = tf.py_function(
                func=_time_stretch,
                inp=[audio, stretch_factor],
                Tout=tf.float32
            )

            #Pad or truncate the audio tensor to the original shape
            audio = tf.cond(
                tf.shape(audio)[0] < original_lenght,
                lambda: tf.pad(audio, paddings=[[original_lenght - tf.shape(audio)[0], 0]], mode='CONSTANT', constant_values=0),
                lambda: tf.image.random_crop(audio, [original_lenght])
            )

            return audio, label
        
        # Map the apply_time_stretch function to the dataset
        return audio_dataset.map(apply_time_stretch, num_parallel_calls=tf.data.AUTOTUNE)

    # Function that applies random pitch shifting to the audio dataset.

    def pitch_shift_dataset(audio_dataset: tf.data.Dataset, pitch_range: tuple[float, float], sample_rate: float) -> tf.data.Dataset:

        min_pitch, max_pitch = pitch_range

        def apply_pitch_shift(audio, label):

            # Generate a random pitch shift value within the specified range
            n_steps = np.random.uniform(min_pitch, max_pitch)

            # Wrap the librosa pitch shift function
            def _pitch_shift(waveform_tf, n_steps_tf):
                return librosa.effects.pitch_shift(waveform_tf.numpy(), sr=sample_rate, n_steps=n_steps_tf.numpy())
            # Apply pitch shift with tf.py_function
            audio = tf.py_function(
                func=_pitch_shift,
                inp=[audio, n_steps],
                Tout=tf.float32
            )

            return audio, label

        # Map the apply_pitch_shift function to the dataset
        return audio_dataset.map(apply_pitch_shift, num_parallel_calls=tf.data.AUTOTUNE)

    add_noise = preprocess_params['ADD_NOISE']
    noise_types = preprocess_params['NOISE_TYPES']
    noise_levels = preprocess_params['NOISE_LEVELS']
    noise_probs = preprocess_params['NOISE_PROBS']
    sample_rate = preprocess_params['TARGET_SR']

    time_shift = preprocess_params['TIME_SHIFT']
    max_shift = preprocess_params['MAX_SHIFT']

    time_stretch = preprocess_params['TIME_STRETCH']
    stretch_range = preprocess_params['STRETCH_RANGE']

    pitch_shift = preprocess_params['PITCH_SHIFT']
    pitch_range = preprocess_params['PITCH_RANGE']

    augmentation_factor = preprocess_params['AUGMENTATION_FACTOR']
    num_samples = len(audio_dataset)
    num_augmented_samples = int(augmentation_factor * num_samples)

    portion_to_augment = audio_dataset.take(num_augmented_samples)

    if add_noise:
        portion_to_augment = add_noise_to_dataset(portion_to_augment, noise_types, noise_levels, noise_probs, sample_rate)
    if time_shift:
        portion_to_augment = time_shift_dataset(portion_to_augment, max_shift, sample_rate)
    if time_stretch:
        portion_to_augment = time_stretch_dataset(portion_to_augment, stretch_range)
    if pitch_shift:
        portion_to_augment = pitch_shift_dataset(portion_to_augment, pitch_range, sample_rate)

    audio_dataset = audio_dataset.concatenate(portion_to_augment)

    return audio_dataset
