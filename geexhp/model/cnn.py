import numpy as np
import pandas as pd
from math import ceil

import tensorflow as tf
import kerastuner as kt
from sklearn import metrics

#from geexhp.model import datasetup as dset

import matplotlib.pyplot as plt

class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

# class ManualStop(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         stop = input("Do you want to stop training? (yes/no): ")
#         if stop.lower() == 'yes':
#             print("Stopping training manually...")
#             self.model.stop_training = True

# class CustomLoss(tf.keras.losses.Loss):
#     def __init__(self, alpha=10.0):
#         super().__init__()
#         self.alpha = alpha

#     def call(self, y_true, y_pred):
#         zero_mask = tf.cast(tf.equal(y_true, 0.0), tf.float32)
#         non_zero_mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
#         zero_loss = self.alpha * tf.square(y_pred) * zero_mask
#         non_zero_loss = tf.square(y_pred - y_true) * non_zero_mask
#         return tf.reduce_mean(zero_loss + non_zero_loss)

class HyperTuningBayCNN:
    def __init__(self, input_shape, output_units, outputs_list):
        self.input_shape = input_shape
        self.output_units = output_units
        self.outputs_list = outputs_list
        self.best_hps = None
        self.best_model = None
        self.history = None

    def build_model(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs

        # Convolutional Layers
        for i in range(hp.Int('conv_layers', 1, 3)):
            x = tf.keras.layers.Conv1D(
                filters=hp.Int(f'filters_{i}', min_value=16, max_value=128, step=16),
                kernel_size=hp.Choice(f'kernel_size_{i}', values=[3, 5, 7, 9]),
                activation="swish"
            )(x)
            x = tf.keras.layers.Conv1D(
                filters=hp.Int(f'filters_{i}', min_value=16, max_value=128, step=16),
                kernel_size=hp.Choice(f'kernel_size_{i}', values=[3, 5, 7, 9]),
                activation="swish"
            )(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = MCDropout(hp.Float(f'dropout_rate_{i}', 0.1, 0.3, step=0.1))(x)
            
        x = tf.keras.layers.Flatten()(x)

        # Dense Layers
        for i in range(hp.Int('dense_layers', 1, 3)):
            x = tf.keras.layers.Dense(
                units=hp.Int(f'units_{i}', min_value=64, max_value=512, step=64),
                activation="swish",
                kernel_regularizer=tf.keras.regularizers.l2(1e-5)
            )(x)
            x = MCDropout(hp.Float(f'dropout_rate_dense_{i}', 0.1, 0.3, step=0.1))(x)

        # Output Layers
        outputs = {}
        for _, output_name in enumerate(self.outputs_list):
            output = tf.keras.layers.Dense(
                units=1,
                activation='linear',
                kernel_regularizer=tf.keras.regularizers.l1(1e-5),
                name=output_name
            )(x)
            outputs[output_name] = output

        # Compile Model with Cosine Learning Rate Schedule
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=hp.Float('learning_rate', 1e-5, 1e-2, sampling='LOG'),
            first_decay_steps=10,
            t_mul=2.0, m_mul=0.9, alpha=0.1
        )
        
        # def false_positive_rate(y_true, y_pred):
        #     zero_mask = tf.cast(tf.equal(y_true, 0.0), tf.float32)
        #     false_positives = tf.cast(tf.greater(y_pred, 0.0), tf.float32) * zero_mask
        #     return tf.reduce_sum(false_positives) / (tf.reduce_sum(zero_mask) + 1e-8)


        losses = {output_name: 'mse' for output_name in self.outputs_list}
        #metrics = {output_name: [false_positive_rate] for output_name in self.outputs_list}

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=losses,
            #metrics=metrics
        )
        return model


    def search(self, train_dataset, validation_dataset, max_trials=50, search_epochs=3):
        # Keras Tuner Bayesian Optimization search
        tuner = kt.BayesianOptimization(
            self.build_model,
            objective="val_loss",
            max_trials=max_trials,
            directory="hyperparam_search",
            project_name="cnn_bay_tuning"
        )

        tuner.search(
            train_dataset,
            validation_data=validation_dataset,
            epochs=search_epochs
        )

        # Get the best hyperparameters and store them in the class
        self.best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        return self.best_hps

    def fit_best_model(self, train_dataset, validation_dataset, epochs=100, patience=5):
        
        if not self.best_hps:
            raise ValueError("No best hyperparameters found. Please run the 'search' method first.")

        self.best_model = self.build_model(self.best_hps)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True)
        
        #manual_stop = ManualStop()  # manual stopping callback

        self.history = self.best_model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            callbacks=[early_stopping],
            verbose=1
        )

        return self.best_model, self.history


    def evaluate(self, test_dataset, feature_names, y_scalers, plot=True, additional_metrics=True):
        if self.best_model is None:
            raise ValueError("Nenhum modelo foi treinado ainda. Por favor, execute 'fit_best_model' primeiro.")

        # Inicializar listas para coletar valores verdadeiros e predições
        y_true_dict = {name: [] for name in self.outputs_list}
        y_pred_dict = {name: [] for name in self.outputs_list}

        # Iterar sobre o test_dataset para coletar predições e valores verdadeiros
        for X_batch, y_batch in test_dataset:
            # Fazer predições
            y_pred_batch = self.best_model.predict(X_batch, verbose=0)
            
            # Append to lists
            for name in self.outputs_list:
                y_true_dict[name].extend(y_batch[name].numpy())
                y_pred_dict[name].extend(y_pred_batch[name])

        # Converter listas em arrays numpy
        y_true = np.column_stack([y_true_dict[name] for name in self.outputs_list])
        y_pred = np.column_stack([y_pred_dict[name] for name in self.outputs_list])

        # Inverter a transformação dos outputs (individualmente para cada scaler)
        y_true_inv = np.column_stack([y_scalers[idx].inverse_transform(y_true[:, idx].reshape(-1, 1)).ravel()
                                    for idx in range(len(self.outputs_list))])
        y_pred_inv = np.column_stack([y_scalers[idx].inverse_transform(y_pred[:, idx].reshape(-1, 1)).ravel()
                                    for idx in range(len(self.outputs_list))])

        # Avaliar outputs
        results = self._evaluate_outputs(
            y_true_inv, y_pred_inv, feature_names, plot, additional_metrics)

        return results

    def _evaluate_outputs(self, y_true, y_pred, feature_names, plot, additional_metrics):
        r2_scores = []
        mae_scores = []
        rmse_scores = []
        
        num_outputs = y_true.shape[1]
        
        for i in range(num_outputs):
            r2 = metrics.r2_score(y_true[:, i], y_pred[:, i])
            r2_scores.append(round(r2, 2))
        
            if additional_metrics:
                mae = metrics.mean_absolute_error(y_true[:, i], y_pred[:, i])
                rmse = np.sqrt(metrics.mean_squared_error(y_true[:, i], y_pred[:, i]))
                mae_scores.append(round(mae, 2))
                rmse_scores.append(round(rmse, 2))
        
        if plot:
            num_cols = 3
            num_rows = ceil(num_outputs / num_cols)
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
        
            for i in range(num_outputs):
                row, col = divmod(i, num_cols)
                if num_rows > 1:
                    ax = axs[row, col]
                else:
                    ax = axs[col]
                ax.scatter(y_true[:, i], y_pred[:, i], label=f'R² = {r2_scores[i]}', alpha=0.6)
                ax.plot([y_true[:, i].min(), y_true[:, i].max()], [y_true[:, i].min(), y_true[:, i].max()],
                        c="tab:orange", ls="--")
                ax.set_xlabel(f'True {feature_names[i]}')
                ax.set_ylabel(f'Predicted {feature_names[i]}')
                ax.legend()
        
            # Remove any unused subplots
            total_plots = num_rows * num_cols
            if total_plots > num_outputs:
                for i in range(num_outputs, total_plots):
                    fig.delaxes(axs.flatten()[i])

            plt.tight_layout()
            plt.show()
        
        results = {'R² scores': r2_scores}
        
        if additional_metrics:
            results['MAE scores'] = mae_scores
            results['RMSE scores'] = rmse_scores
        
        return pd.DataFrame(results, index=feature_names)

    def plot_losses(self):
        """
        Plots the training and validation loss from the training history.
        """
        if self.history is None:
            raise ValueError("No training history found. Please train the model first.")

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(loss) + 1)

        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def summary(self):
        """Prints the model summary."""
        if self.best_model is None:
            print("No model is built yet. Please train or load a model first.")
        else:
            self.best_model.summary()

    def save(self, file_path):
        """Saves the model to the specified file path."""
        if self.best_model is None:
            raise ValueError("No model to save. Please train the model first.")
        self.best_model.save(file_path)
        print(f"Model saved at {file_path}")

    def load_model(self, file_path):
        """Loads the model from the specified file path."""
        self.best_model = tf.keras.models.load_model(file_path)
        print(f"Model loaded from {file_path}")
    
    def predict_with_uncertainty(self, X, n_iter=100):
        if self.best_model is None:
            raise ValueError("Nenhum modelo foi treinado ainda. Por favor, execute 'fit_best_model' primeiro.")

        predictions_list = {name: [] for name in self.outputs_list}
        for _ in range(n_iter):
            preds = self.best_model.predict(X, batch_size=32, verbose=0)
            for name in self.outputs_list:
                predictions_list[name].append(preds[name])
        
        # Converter listas em arrays numpy
        predictions_array = {name: np.array(predictions_list[name]) for name in self.outputs_list}

        # Compute mean and uncertainty
        mean_predictions = {name: np.mean(pred_array, axis=0) for name, pred_array in predictions_array.items()}
        uncertainty = {name: np.std(pred_array, axis=0) for name, pred_array in predictions_array.items()}

        return predictions_array, mean_predictions, uncertainty
    
    def inverse_transform_predictions(self, predictions, y_scalers):
        """
        Inverse transforms the scaled predictions to the original scale.
        """
        n_iter, samples, output_units = predictions.shape

        # Initialize array for inverse transformed predictions
        predictions_inv = np.zeros_like(predictions)

        # Inverse transform each output separately using its corresponding scaler
        for idx in range(output_units):
            # Reshape for inverse transform
            pred_reshaped = predictions[:, :, idx].reshape(-1, 1)
            predictions_inv[:, :, idx] = y_scalers[idx].inverse_transform(pred_reshaped).reshape(n_iter, samples)

        return predictions_inv
