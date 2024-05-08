import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pathlib
import os
import json
from typing import Any

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Utility function to save a model to a directory, along with preprocessing parameters
def save_model(model: tf.keras.Model, saved_models_dir, preprocess_params: dict[str, Any], training_params: dict[str, Any], test_accuracy):
    #Extract the dataset name, transform
    dataset_name = preprocess_params['DATASET_NAME']
    transforms_2d = preprocess_params['2D_TRANSFORMS']
    transforms_1d = preprocess_params['1D_TRANSFORMS']

    transforms = '-'.join(transforms_2d + transforms_1d)

    # Compute formatted accuracy
    formatted_accuracy = "{:.2f}".format(test_accuracy * 100)

    # Construct the model save path
    model_save_path = pathlib.Path(saved_models_dir) / f"{dataset_name}_{transforms}_{formatted_accuracy}"

    # Save the model in the TensorFlow SavedModel format
    model.save(model_save_path, save_format='tf')

    # Serialize the preprocess_params dictionary to a JSON string
    preprocess_params_json = json.dumps(preprocess_params)
    training_params_json = json.dumps(training_params)

    # Create the assets directory if it does not exist
    assets_dir = model_save_path / 'assets'
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Write the JSON strings directly into the assets directory
    with open(assets_dir / 'preprocess_params.json', 'w') as f:
        f.write(preprocess_params_json)
    with open(assets_dir / 'training_params.json', 'w') as f:
        f.write(training_params_json)
    
    print(f"Model saved to: {model_save_path}")



# Utility function to load a saved model and its preprocessing parameters from a directory
def list_and_load_model(saved_models_dir) -> tuple[tf.keras.Model, dict[str, Any], dict[str, Any]]:
    # Convert to pathlib.Path object if not already
    saved_models_dir = pathlib.Path(saved_models_dir)
    
    # Check if the saved models directory exists
    if not saved_models_dir.exists():
        print(f"No saved models found in {saved_models_dir}")
    
    # List all subdirectories in the saved models directory
    model_subdirs = [d for d in os.listdir(saved_models_dir) if os.path.isdir(saved_models_dir / d)]
    
    # Print the available models with an index
    for idx, model_subdir in enumerate(model_subdirs):
        print(f"{idx}: {model_subdir}")
    
    # Ask the user to select a model to load
    selected_index = int(input("Enter the index of the model to load: "))
    selected_model_subdir = model_subdirs[selected_index]
    
    # Load the selected model
    full_model_path = saved_models_dir / selected_model_subdir
    model = tf.keras.models.load_model(full_model_path)
    
    # Load the preprocessing parameters JSON
    preprocessing_params_path = full_model_path / 'assets' / 'preprocess_params.json'
    training_params_path = full_model_path / 'assets' / 'training_params.json'

    if preprocessing_params_path.is_file():
        with open(preprocessing_params_path, 'r') as f:
            preprocessing_params = json.load(f)
    else:
        print(f"No preprocessing parameters found in {preprocessing_params_path}")
        preprocessing_params = None

    if training_params_path.is_file():
        with open(training_params_path, 'r') as f:
            training_params = json.load(f)
    else:
        print(f"No training parameters found in {training_params_path}")
        training_params = None
        
    print(f"Loaded model from {full_model_path}")
    model.summary()
    return model, preprocessing_params, training_params



# Utility function to print the confusion matrix for a model
def print_confusion_matrix(model, dataset, class_names):

    def get_true_and_predicted_labels(model, dataset):
        y_pred = []
        y_true = []

        for image_batch, label_batch in dataset:
            y_true.append(tf.argmax(label_batch, axis=-1))  # Convert one-hot encoded labels to class indices
            preds = model.predict(image_batch, verbose=0)
            y_pred.append(np.argmax(preds, axis=-1))

        correct_labels = tf.concat([item for item in y_true], axis=0)
        predicted_labels = tf.concat([item for item in y_pred], axis=0)

        return correct_labels, predicted_labels

    correct_labels, predicted_labels = get_true_and_predicted_labels(model, dataset)
    confusion_mtx = tf.math.confusion_matrix(correct_labels, predicted_labels).numpy()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    display = ConfusionMatrixDisplay(confusion_mtx, display_labels=class_names)

    display.plot(xticks_rotation='vertical', ax=ax)
    plt.show()



# Utility function to plot the feature maps of a model
from ipywidgets import interact, Dropdown
from IPython.display import clear_output

def plot_feature_maps(model, test_dataset, class_names, seed=None):
    # Set the random seed for reproducibility if provided
    tf.keras.utils.set_random_seed(seed)

    example, label = next(iter(test_dataset.rebatch(1).shuffle(32)))
    example = example[0].numpy()
    label_str = class_names[np.argmax(label)]

    feature_maps_list = [example]
    for layer in model.layers[1:]:
        last_output = feature_maps_list[-1]
        new_output = layer(last_output).numpy()
        feature_maps_list.append(new_output)

    def display_feature_maps(layer_index):
        clear_output(wait=False)

        layer_index = int(layer_index)
        current_layer = model.layers[layer_index]
        feature_maps = feature_maps_list[layer_index]

        def plot_input_layer():
            fig, _ = plt.subplots(figsize=(10, 2))
            plt.title(f'{current_layer.name}: {current_layer.input.shape[1:]} - {label_str}')
            plt.imshow(feature_maps[0, :, :, 0].T, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar()
            plt.tight_layout()
            return fig

        def plot_conv_feature():
            num_feature_maps = feature_maps.shape[-1]
            size_factor = 3.5  # Adjust this factor to change the size of the subplots
            num_cols = int(np.sqrt(num_feature_maps))
            num_rows = int(np.ceil(num_feature_maps / num_cols))
            figsize = (size_factor * num_cols, size_factor * num_rows)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
            fig.suptitle(f'{current_layer.name}: {current_layer.input.shape[1:]} -> {current_layer.output.shape[1:]} - {label_str}', fontsize=16, y=1)
            plt.subplots_adjust(top=1.5)

            for i in range(num_feature_maps):
                ax = axes.flat[i]
                feature_map = feature_maps[0, :, :, i]
                im = ax.imshow(feature_map, aspect='auto', cmap='viridis', origin='lower')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                ax.axis('off')
            plt.tight_layout()
            return fig

        def plot_1d_feature():
            fig, _ = plt.subplots(figsize=(10, 2))
            plt.title(f'{current_layer.name}: {current_layer.input.shape[1:]} -> {current_layer.output.shape[1:]} - {label_str}')
            plt.bar(range(len(feature_maps[0])), feature_maps[0])
            plt.ylabel('Value')
            if len(class_names) == len(feature_maps[0]):
                plt.xticks(range(len(feature_maps[0])), class_names, rotation=90, fontsize=8)
            plt.tight_layout()
            return fig

        if layer_index == 0:
            plot_input_layer()
        elif feature_maps.ndim == 4:
            plot_conv_feature()
        elif feature_maps.ndim == 2:
            plot_1d_feature()
        else:
            print(f"Layer {layer_index} has an unsupported output shape. Cannot plot feature maps.")

    layer_names = {layer.name: str(index) for index, layer in enumerate(model.layers)}
    interact(display_feature_maps, layer_index=Dropdown(options=layer_names, description="Select Layer:"))

# Utility function to plot the weights of Conv2D layers
def plot_conv_weights(model):
    # Iterate over all layers of the model
    for layer in model.layers:
        # Check for Conv2D layer
        if isinstance(layer, tf.keras.layers.Conv2D):
            # Get the weights of the layer
            weights = layer.get_weights()[0]
            
            # Normalize the weights
            weights_min = weights.min()
            weights_max = weights.max()
            weights = (weights - weights_min) / (weights_max - weights_min)
            
            # Get the number of filters and the size of each filter
            num_filters = weights.shape[-1]
            size_factor = 3  # Adjust this factor to change the size of the subplots
            num_cols = int(np.sqrt(num_filters))
            num_rows = int(np.ceil(num_filters / num_cols))
            figsize = (size_factor * num_cols, size_factor * num_rows)
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
            fig.suptitle(f'Filters of Layer: {layer.name} - Shape: {weights.shape}', fontsize=16)
            
            # Plot each filter
            for i in range(num_filters):
                ax = axes.flat[i]
                # Get the filter
                filter = weights[:, :, :, i]
                # Only display the first channel of the filter
                if filter.shape[2] > 1:
                    filter = filter[:, :, 0]
                im = ax.imshow(filter, aspect='auto', cmap='viridis')
                ax.axis('off')

                # Add a colorbar to each subplot
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
            
            plt.tight_layout()
            plt.show()
