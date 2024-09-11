import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from math import ceil
import matplotlib.pyplot as plt

import tensorflow as tf
import kerastuner as kt
from sklearn import metrics

class ConvolutionalNetwork:
    """
    Convolutional Neural Network (CNN) class.
    
    Attributes
    -----------
    input_shape : tuple
        The shape of the input data.
    output_units : int
        The number of output units.
    conv_layers : list of tuples
        Each tuple represents a convolutional layer, containing:
        - number of filters
        - kernel size
        - pool size
    dense_layers : list of int
        Each integer represents the number of neurons in a fully connected (dense) layer.
    activation : Activation function
        Activation function used for layers (default: ReLU).
    dropout_rate : float
        Dropout rate for regularization.
    """

    def __init__(self, input_shape, output_units, conv_layers, dense_layers, 
                activation=tf.keras.layers.ReLU(), dropout_rate=0.0):
        """
        Initializes the Convolutional Neural Network.
        """
        
        self.input_shape = input_shape          # Shape of input (e.g., (2304, 1))
        self.output_units = output_units        # Number of output units (e.g., 17)
        self.conv_layers = conv_layers          # List of convolutional layers (filters, kernel size)
        self.dense_layers = dense_layers        # List of dense layer units
        self.activation = activation            # Activation function (default: ReLU)
        self.dropout_rate = dropout_rate        # Dropout rate

        # Build the CNN model
        self.model = self._build_model()

    def _add_conv_layer(self, x, filters, kernel_size, pool_size):
        """
        Adds a Conv1D layer to the model with optional batch normalization and dropout.
        """
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = self.activation(x)

        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = self.activation(x)

        x = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=2)(x)

        if self.dropout_rate > 0:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        return x

    def _add_dense_layer(self, x, units):
        """
        Adds a Dense layer to the model with optional dropout.
        """
        x = tf.keras.layers.Dense(units)(x)
        x = self.activation(x)
        if self.dropout_rate > 0:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        return x

    def _build_model(self):
        """
        Builds the Convolutional Neural Network model.
        """
        inputs = tf.keras.Input(shape=self.input_shape, name="Albedo")

        x = inputs
        for filters, kernel_size, pool_size in self.conv_layers:
            x = self._add_conv_layer(x, filters, kernel_size, pool_size)
        
        x = tf.keras.layers.Flatten()(x)

        for units in self.dense_layers:
            x = self._add_dense_layer(x, units)

        outputs = tf.keras.layers.Dense(self.output_units, activation='sigmoid')(x)
        model = tf.keras.Model([inputs], outputs)

        return model

    def train(self, X_train, y_train, patience=10, epochs=50, batch_size=250, checkpoint_path="../models"):
        """
        Trains the CNN model with an exponential learning rate decay schedule and early stopping.

        Parameters
        -----------
        X_train : array-like
            The input training data (features), expected to have shape (samples, timesteps, features).
        y_train : array-like
            The target training data (labels), expected to have shape (samples, output_features).
        patience : int, optional
            Number of epochs with no improvement after which training will be stopped (default is 10).
        epochs : int, optional
            Total number of epochs to train the model (default is 50).
        batch_size : int, optional
            Number of samples per batch for training (default is 250).
        learning_rate : float, optional
            The learning rate for the optimizer (default is 1e-5).

        Returns
        --------
        history : keras.callbacks.History
            History object containing training and validation metrics.
        """
        checkpoint_path = f"{checkpoint_path}/weights.weights.h5"

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=0.5)
        self.model.compile(optimizer=optimizer, loss='mse')

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True
        )

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=True,verbose=1)

        self.model.save_weights(checkpoint_path.format(epoch=0))       

        history = self.model.fit(
            X_train, y_train, validation_split=0.2,
            epochs=epochs, batch_size=batch_size,
            callbacks=[early_stopping, cp_callback],
            verbose=1
        )
        
        return history

    def summary(self):
        """Prints the model summary."""
        self.model.summary()

    def save(self, file_path):
        """Saves the model to the specified file path."""
        self.model.export(file_path)
        print(f"Model saved in {file_path}")
    
    def load_model(self, file_path):
        """Load the model from SavedModel."""
        self.model = tf.keras.models.load_model(file_path)

    def evaluate(self, X_test, y_test, feature_names, scaler=None, plot=True, additional_metrics=True):
        """
        Evaluates the CNN model, returning R² scores and optionally additional metrics.

        Parameters
        -----------
        X_test : array-like
            The input test data (features).
        y_test : array-like
            The true target labels (outputs).
        feature_names : list of str
            Names of features for labeling plots.
        scaler : sklearn.preprocessing.StandardScaler or None, optional
            Scaler used to inverse transform predictions and true values.
        plot : bool, optional
            Whether to generate plots of true vs predicted values.
        additional_metrics : bool, optional
            Whether to compute additional metrics (MAE, RMSE) alongside R² scores.

        Returns
        --------
        results : dict
            Dictionary containing R² scores and, if selected, additional metrics (MAE, RMSE) for each output feature.
        """
        predictions = self.model.predict(X_test)
        
        if scaler:
            y_test = scaler.inverse_transform(y_test)
            predictions = scaler.inverse_transform(predictions)

        r2_scores = []
        mae_scores = []
        rmse_scores = []

        for i in range(self.output_units):
            r2 = metrics.r2_score(y_test[:, i], predictions[:, i])
            r2_scores.append(round(r2, 2))
            
            if additional_metrics:
                mae = metrics.mean_absolute_error(y_test[:, i], predictions[:, i])
                rmse = np.sqrt(metrics.mean_squared_error(y_test[:, i], predictions[:, i]))
                mae_scores.append(round(mae, 2))
                rmse_scores.append(round(rmse, 2))
        
        if plot:
            num_cols = 3  
            num_rows = ceil(self.output_units / num_cols)
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows))

            for i in range(self.output_units):
                row, col = divmod(i, num_cols)
                ax = axs[row, col] if num_rows > 1 else axs[col]
                ax.scatter(y_test[:, i], predictions[:, i], label=f'R² = {r2_scores[i]}', alpha=0.6)
                ax.plot([y_test[:, i].min(), y_test[:, i].max()], [y_test[:, i].min(), y_test[:, i].max()],
                        c="tab:orange", ls="--")
                ax.set_xlabel(f'True {feature_names[i]}')
                ax.set_ylabel(f'Predicted {feature_names[i]}')
                ax.legend()

            for i in range(self.output_units, num_rows * num_cols):
                fig.delaxes(axs.flatten()[i])

            plt.tight_layout()
            plt.show()

        results = {'R² scores': r2_scores}
    
        if additional_metrics:
            results['MAE scores'] = mae_scores
            results['RMSE scores'] = rmse_scores

        return pd.DataFrame(results, index=feature_names)
    
    def plot_losses(self, history):
        """
        Plots the training and validation loss from the training history.

        Parameters
        -----------
        history : keras.callbacks.History
            History object returned from the model's fit method.
        """
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)

        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


class HyperparameterTuningBayesian:
    def __init__(self, input_shape, output_units, max_trials=10, executions_per_trial=1):
        self.input_shape = input_shape  # Shape of input (e.g., (2304, 1))
        self.output_units = output_units  # Number of output units (e.g., 17)
        self.max_trials = max_trials  # Number of different hyperparameter sets to try
        self.executions_per_trial = executions_per_trial  # Number of times to try each set

        # Keras Tuner Bayesian Optimization search
        self.tuner = kt.BayesianOptimization(
            self.build_model,  # Model-building function
            objective="val_loss",  # Objective to minimize
            max_trials=self.max_trials,  # Number of trials
            executions_per_trial=self.executions_per_trial,  # How many times to run each trial
            directory="hyperparam_search",  # Directory to store results
            project_name="cnn_bayesian_tuning"
        )

    def build_model(self, hp):
        # Define a hyperparameter search space for each element
        conv_layers = []
        for i in range(5):  # Search space for 3 convolutional layers
            filters = hp.Int(f"conv_{i}_filters", min_value=16, max_value=128, step=16)
            kernel_size = hp.Int(f"conv_{i}_kernel_size", min_value=3, max_value=17, step=2)
            pool_size = hp.Int(f"conv_{i}_pool_size", min_value=2, max_value=32, step=2)
            conv_layers.append((filters, kernel_size, pool_size))

        # Dense layer units search space
        dense_units = [
            hp.Int(f"dense_{i}_units", min_value=32, max_value=256, step=32) for i in range(3)
        ]

        # Dropout rate search space
        dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)

        # Learning rate search space
        learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")

        # Build model using the hyperparameters
        model = ConvolutionalNetwork(
            input_shape=self.input_shape,
            output_units=self.output_units,
            conv_layers=conv_layers,
            dense_layers=dense_units,
            activation=tf.keras.layers.ReLU(),
            dropout_rate=dropout_rate
        )._build_model()

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse")

        return model

    def search(self, X_train, y_train, epochs=50, batch_size=250):
        # Define early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

        # Start the hyperparameter search
        self.tuner.search(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )

        # Get the best hyperparameters
        best_hp = self.tuner.get_best_hyperparameters()[0]
        return best_hp

    def get_best_model(self):
        # Return the best model from the search
        return self.tuner.get_best_models(num_models=1)[0]

