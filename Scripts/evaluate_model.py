import argparse
import os
import sys

from types import ModuleType
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf

from tensorflow.keras.models import load_model

import PrognosAIs.Constants
import PrognosAIs.IO.utils as IO_utils
import PrognosAIs.Model.Losses
import PrognosAIs.Model.Parsers as ModelParsers
import PrognosAIs.Model.Metrics
import tensorflow_addons as tfa

from PrognosAIs.IO import ConfigLoader
from PrognosAIs.IO import DataGenerator


class Evaluator:
    def __init__(self, model_file, data_folder, config_file, output_folder) -> None:
        self.EVALUATION_FOLDER = "Results"
        self.data_folder = data_folder
        self.model_file = model_file
        self.predictions = None
        self.sample_predictions = None
        self.sample_labels = None
        self.sample_names = None
        self.output_folder = os.path.join(output_folder, self.EVALUATION_FOLDER)
        IO_utils.create_directory(self.output_folder)

        self.config = ConfigLoader.ConfigLoader(config_file)
        self.batch_size = self.config.get_batch_size()

        self.model = self.load_model(self.model_file, None)

        self.init_model_parameters()

        self.data_generator = self.init_data_generators()
        self.dataset_names = list(self.data_generator.keys())
        self.sample_metadata = self.data_generator[self.dataset_names[0]].get_feature_metadata()
        self.label_data_generator = self.init_label_generators()

        self.image_output_labels = self.get_image_output_labels()


    @staticmethod
    def load_model(model_file: str, custom_module: ModuleType = None) -> tf.keras.Model:
        """
        Load the model, including potential custom losses.

        Args:
            model_file (str): Location of the model file
            custom_module (ModuleType): Custom module from which to load losses or metrics

        Raises:
            error: If the model could not be loaded
                and the problem is not due to a missing loss or metric function.

        Returns:
            tf.keras.Model: The loaded model
        """
        # type hint for mypy
        model = tf.keras.models.load_model(model_file, custom_objects={"MaskedAUC": PrognosAIs.Model.Metrics.MaskedAUC, "DICE": PrognosAIs.Model.Metrics.DICE, "MaskedSensitivity": PrognosAIs.Model.Metrics.MaskedSensitivity, "MaskedSpecificity": PrognosAIs.Model.Metrics.MaskedSpecificity, "AdamW": tfa.optimizers.AdamW, "MaskedCategoricalCrossentropy": PrognosAIs.Model.Losses.MaskedCategoricalCrossentropy, "DICE_loss": PrognosAIs.Model.Losses.DICE_loss})
        # from loaded models, see https://github.com/tensorflow/tensorflow/issues/37990
        # Therefore we do a fake "fit" round to make everything available
        return model

    def _init_data_generators(self, labels_only: bool) -> dict:
        """
        Initialize data generators for all sample folders.

        Args:
            labels_only (bool): Whether to only load labels

        Returns:
            dict: initalized data generators
        """
        sub_folders = IO_utils.get_subdirectories(self.data_folder)
        data_generators = {}
        for i_sub_folder in sub_folders:
            folder_name = IO_utils.get_root_name(i_sub_folder)
            if (
                folder_name == PrognosAIs.Constants.TRAIN_DS_NAME
                and not self.config.get_evaluate_train_set()
            ):
                continue
            data_generators[folder_name] = DataGenerator.HDF5Generator(
                i_sub_folder,
                self.batch_size,
                shuffle=self.config.get_shuffle_evaluation(),
                drop_batch_remainder=False,
                labels_only=labels_only,
            )
        return data_generators

    def init_data_generators(self) -> dict:
        """
        Initialize the data generators.

        Returns:
            dict: DataGenerator for each subfolder of samples
        """

        return self._init_data_generators(False)

    def init_label_generators(self) -> dict:
        """
        Initialize the data generators which only give labels.

        Returns:
            dict: DataGenerator for each subfolder of samples
        """
        return self._init_data_generators(False)

    def init_model_parameters(self) -> None:
        """
        Initialize the parameters from the model.
        """
        self.output_names = self.model.output_names
        self.number_of_outputs = len(self.output_names)
        if self.number_of_outputs == 1:
            self.output_shapes = [self.model.output_shape]
        else:
            self.output_shapes = self.model.output_shape

        self.output_classes = {}
        self.one_hot_outputs = {}
        for i_output_index, i_output_name in enumerate(self.output_names):
            self.output_classes[i_output_name] = self.output_shapes[i_output_index][-1]

            if self.output_shapes[i_output_index][-1] > 1:
                self.one_hot_outputs[i_output_name] = True
            else:
                self.one_hot_outputs[i_output_name] = False

        self.input_names = self.model.input_names
        self.number_of_inputs = len(self.input_names)

        model_input_shape = self.model.input_shape
        if isinstance(model_input_shape, dict):
            self.input_shapes = list(model_input_shape.values())
        elif self.number_of_inputs == 1:
            self.input_shapes = [model_input_shape]
        else:
            self.input_shapes = model_input_shape.values()

    def get_image_output_labels(self) -> dict:
        """
        Whether an output label is a simple class, the label is actually an image.

        Returns:
            dict: Output labels that are image outputs
        """
        image_outputs_labels = {}

        for i_output_name, i_output_shape in zip(self.output_names, self.output_shapes):
            for i_input_name, i_input_shape in zip(self.input_names, self.input_shapes):
                # It is an image of a certain input of the output has as many dimension
                # and the size of each dimension is equal to the input size
                # minus the batch dimension and number of classes

                equal_dimensions = len(i_input_shape) == len(i_output_shape)
                equal_size = i_input_shape[1:-1] == i_output_shape[1:-1]
                if equal_dimensions and equal_size:
                    image_outputs_labels[i_output_name] = i_input_name

        return image_outputs_labels

    def _format_predictions(self, predictions: Union[list, np.ndarray]) -> dict:
        """
        Format the predictions to match them with the output names

        Args:
            predictions (Union[list, np.ndarray]): The predictions from the model

        Raises:
            ValueError: If the predictions do not match with the expected output names

        Returns:
            dict: Output predictions matched with the output names
        """
        if isinstance(predictions, np.ndarray):
            # There is only one output in this case
            predictions = [predictions]

        if len(predictions) != len(self.output_names):
            raise ValueError("The predictions do not match with the output names!")

        out_predictions = {}
        for i_output_name, i_prediction in zip(self.output_names, predictions):
            out_predictions[i_output_name] = i_prediction

        return out_predictions

    def predict(self) -> dict:
        """
        Get predictions from the model

        Returns:
            dict: Predictions for the different outputs of the model for all samples
        """
        if self.predictions is None:
            # We have not yet determined the predictions, first run
            self.predictions = {}
            predictions = {}
            for i_generator_name, i_generator in self.data_generator.items():
                # We go over all generators
                self.predictions[i_generator_name] = {}
                dataset = i_generator.get_tf_dataset()
                final_predictions = {}
                for i_output_name in self.output_names:
                    final_predictions[i_output_name] = []

                for i_batch in dataset:
                    # We have to go over the different predictions step by step
                    # Otherwise will lead to memory leak
                    # The first index in the batch is the sample (the second is the label)
                    batch_prediction = self.model.predict_on_batch(i_batch[0])

                    # Convert to list if we only have one output
                    if isinstance(batch_prediction, np.ndarray):
                        batch_prediction = [batch_prediction]
                    for i_output_name, i_prediction in zip(self.output_names, batch_prediction):
                        final_predictions[i_output_name].append(i_prediction)

                # We create one single list for all predictions that we got
                for i_output_name in self.output_names:
                    final_predictions[i_output_name] = np.concatenate(
                        final_predictions[i_output_name], axis=0
                    )

                predictions[i_generator_name] = final_predictions
            self.predictions = predictions
        return self.predictions

    def patches_to_sample_image(
        self,
        datagenerator: PrognosAIs.IO.DataGenerator.HDF5Generator,
        filenames: list,
        output_name: str,
        predictions: np.ndarray,
        labels_are_one_hot: bool,
        label_combination_type: str,
    ) -> np.ndarray:

        if not labels_are_one_hot and label_combination_type == "average":
            err_msg = (
                "Predictions can only be combined when given as probability score"
                "thus the labels must be one-hot encoded."
            )
            raise ValueError(err_msg)

        input_name = self.image_output_labels[output_name]
        image_size = self.sample_metadata[input_name]["original_size"]
        transpose_dims = np.arange(len(image_size) - 1, -1, -1)
        number_of_classes = self.output_classes[output_name]
        image_size = np.append(image_size, number_of_classes)

        original_image = np.zeros(image_size)
        number_of_hits = np.zeros(image_size)

        if isinstance(filenames, str):
            filenames = [filenames]

        # TODO REMOVE ONLY FOR IF NOT REALLY PATCHES
        if (
            len(predictions.shape) == len(original_image.shape)
            and predictions.shape[-1] == original_image.shape[-1]
        ):
            predictions = np.expand_dims(predictions, axis=0)

        for i_filename, i_prediction in zip(filenames, predictions):
            i_sample_metadata = datagenerator.get_feature_metadata_from_sample(i_filename)
            i_sample_metadata = i_sample_metadata[input_name]

            in_sample_index_start = np.copy(i_sample_metadata["index"])
            in_sample_index_end = in_sample_index_start + i_sample_metadata["size"]

            # Parts of the patch can be outside of the original image, because of padding
            # Thus here we take only the parts of the patch that are within the original image
            in_sample_index_start[in_sample_index_start < 0] = 0
            sample_indices = tuple(
                slice(*i) for i in zip(in_sample_index_start, in_sample_index_end)
            )

            patch_index_start = np.copy(i_sample_metadata["index"])
            patch_index_end = i_sample_metadata["size"]
            # We also need to cut out the part of the patch that is normally outside of the image
            # we do this here
            patch_index_start[patch_index_start > 0] = 0
            patch_index_start = -1 * patch_index_start
            patch_slices = tuple(slice(*i) for i in zip(patch_index_start, patch_index_end))

            if not labels_are_one_hot:
                i_prediction = i_prediction.astype(np.int32)
                i_prediction = np.eye(number_of_classes)[i_prediction]
            elif label_combination_type == "vote":
                i_prediction = np.round(i_prediction)

            original_image[sample_indices] += i_prediction[patch_slices]
            number_of_hits[sample_indices] += 1

        number_of_hits[number_of_hits == 0] = 1
        if label_combination_type == "vote":
            original_image = np.argmax(original_image, axis=-1)
        elif label_combination_type == "average":
            original_image = np.argmax(np.round(original_image / number_of_hits), axis=-1)
        else:
            raise ValueError("Unknown combination type")

        # Need to transpose because of different indexing between numpy and simpleitk
        original_image = np.transpose(original_image, transpose_dims)
        return original_image

    def image_array_to_sitk(self, image_array: np.ndarray, input_name: str) -> sitk.Image:
        original_image_direction = self.sample_metadata[input_name]["original_direction"]
        original_image_origin = self.sample_metadata[input_name]["original_origin"]
        original_image_spacing = self.sample_metadata[input_name]["original_spacing"]
        img = sitk.GetImageFromArray(image_array)
        img.SetDirection(original_image_direction)
        img.SetOrigin(original_image_origin)
        img.SetSpacing(original_image_spacing)
        # To ensure proper loading
        img = sitk.Cast(img, sitk.sitkFloat32)

        return img

    def _find_sample_names_from_patch_names(self, data_generator):
        filenames = data_generator.sample_files

        # Get the unique names of the files
        sample_names = np.unique([i_file.split("_patch")[0] for i_file in filenames])

        sample_indices = {}
        for i_sample_name in sample_names:
            sample_indices[i_sample_name] = np.squeeze(
                np.argwhere(
                    [i_sample_name == i_filename.split("_patch")[0] for i_filename in filenames]
                )
            )

        return sample_names, sample_indices

    def get_sample_result_from_patch_results(self, patch_results):
        sample_results = {}
        sample_names = {}
        for i_dataset_name, i_dataset_generator in self.data_generator.items():
            i_patch_results = patch_results[i_dataset_name]
            sample_results[i_dataset_name] = {}
            file_locations = np.asarray(i_dataset_generator.sample_locations)

            sample_names[i_dataset_name], sample_indices = self._find_sample_names_from_patch_names(
                i_dataset_generator
            )

            for i_output_name, i_output_prediction in i_patch_results.items():
                sample_results[i_dataset_name][i_output_name] = []

                if i_output_name in self.image_output_labels:
                    for i_sample_name, i_sample_indices in sample_indices.items():
                        patches_from_sample_results = i_output_prediction[i_sample_indices]
                        sample_results[i_dataset_name][i_output_name].append(
                            self.patches_to_sample_image(
                                i_dataset_generator,
                                file_locations[i_sample_indices],
                                i_output_name,
                                patches_from_sample_results,
                                self.one_hot_outputs[i_output_name],
                                self.config.get_label_combination_type(),
                            )
                        )
                for i_key, i_value in sample_results[i_dataset_name].items():
                    sample_results[i_dataset_name][i_key] = np.asarray(i_value)

        return sample_names, sample_results

    def get_sample_predictions_from_patch_predictions(self):
        patch_predictions = self.predict()
        sample_names, sample_predictions = self.get_sample_result_from_patch_results(
            patch_predictions
        )
        return sample_names, sample_predictions

    @staticmethod
    def one_hot_labels_to_flat_labels(labels: np.ndarray) -> np.ndarray:
        flat_labels = np.argmax(labels, axis=-1)
        flat_labels[labels[..., 0] == -1] = -1
        return flat_labels

    def make_dataframe(self, sample_names, predictions) -> pd.DataFrame:
        df_columns = ["Sample"]
        for i_output_name in self.output_names:
            if i_output_name not in self.image_output_labels:
                for i_class in range(self.output_classes[i_output_name]):
                    df_columns.append("Prediction_" + i_output_name + "_class_" + str(i_class))

        results_df = pd.DataFrame(columns=df_columns)
        results_df["Sample"] = sample_names
        for i_output_name, i_output_prediction in predictions.items():
            if i_output_name not in self.image_output_labels:
                for i_class in range(self.output_classes[i_output_name]):
                    results_df[
                        "Prediction_" + i_output_name + "_class_" + str(i_class)
                    ] = i_output_prediction[:, i_class]

        return results_df

    def write_image_predictions_to_files(self, sample_names, predictions, labels_one_hot) -> None:
        for i_output_name, i_output_prediction in predictions.items():
            if i_output_name in self.image_output_labels:
                if labels_one_hot is not None and labels_one_hot[i_output_name]:
                    i_output_prediction = self.one_hot_labels_to_flat_labels(i_output_prediction,)

                i_output_prediction_images = [
                    self.image_array_to_sitk(
                        i_sample_output_prediction, self.image_output_labels[i_output_name]
                    )
                    for i_sample_output_prediction in i_output_prediction
                ]

                for i_pred_image, i_sample_name in zip(i_output_prediction_images, sample_names):
                    out_file = os.path.join(
                        self.output_folder, i_sample_name.split(".")[0] + "_mask.nii.gz"
                    )
                    sitk.WriteImage(i_pred_image, out_file)

    def write_predictions_to_file(self) -> None:
        predictions = self.predict()

        for i_dataset_name, i_dataset_generator in self.data_generator.items():
            out_file = os.path.join(self.output_folder, "genetic_histological_predictions.csv")
            i_prediction = predictions[i_dataset_name]

            results_df = self.make_dataframe(
                i_dataset_generator.sample_files,
                i_prediction,
            )

            results_df.to_csv(out_file, index=False)

            (
                sample_names,
                sample_predictions,
            ) = self.get_sample_predictions_from_patch_predictions()

            self.write_image_predictions_to_files(
                sample_names[i_dataset_name], sample_predictions[i_dataset_name], None,
            )

    def evaluate(self):
        self.write_predictions_to_file()
