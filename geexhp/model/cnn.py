import numpy as np
import pandas as pd
from math import ceil

import tensorflow as tf
import kerastuner as kt
from sklearn import metrics

import matplotlib.pyplot as plt

class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

class HyperTuningBayCNN:
    def __init__(self, input_shape, abundance_units, planetary_units):
        self.input_shape = input_shape
        self.abundance_units = abundance_units
        self.planetary_units = planetary_units
        self.best_hps = None
        self.best_model = None
        self.history = None

    def build_model(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs

        # Convolutional Layers
        for i in range(hp.Int('conv_layers', 2, 6)):
            x = tf.keras.layers.Conv1D(
                filters=hp.Int(f'filters_{i}', min_value=32, max_value=256, step=32),
                kernel_size=hp.Choice(f'kernel_size_{i}', values=[3, 5, 7]),
                activation=hp.Choice('activation_conv', ['relu', 'swish']) 
            )(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
            x = MCDropout(hp.Float(f'dropout_rate_{i}', 0.1, 0.4, step=0.1))(x)
            
        x = tf.keras.layers.Flatten()(x)

        # Dense Layers
        for i in range(hp.Int('dense_layers', 1, 4)):
            x = tf.keras.layers.Dense(
                units=hp.Int(f'units_{i}', min_value=64, max_value=512, step=64),
                activation=hp.Choice('activation_dense', ['relu', 'swish']) 
            )(x)
            x = MCDropout(hp.Float(f'dropout_rate_dense_{i}', 0.1, 0.4, step=0.1))(x)

        # Output Layers
        abundance_output = tf.keras.layers.Dense(self.abundance_units, activation='linear', name='abundance_output')(x)
        planetary_output = tf.keras.layers.Dense(self.planetary_units, activation='linear', name='planetary_output')(x)

        # Compile Model with Cosine Learning Rate Schedule
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG'),
            first_decay_steps=10,
            t_mul=2.0, m_mul=0.9, alpha=0.1
        )

        model = tf.keras.Model(inputs=inputs, outputs=[abundance_output, planetary_output])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss={'abundance_output': 'mse', 'planetary_output': 'mse'},
            loss_weights={'abundance_output': 0.6, 'planetary_output': 0.4}  # Balance losses
        )

        return model


    def search(self, X_train, y_train_abundance, y_train_planetary, max_trials=50, search_epochs=3):
        # Keras Tuner Bayesian Optimization search
        tuner = kt.BayesianOptimization(
            self.build_model,
            objective="val_loss",
            max_trials=max_trials,
            directory="hyperparam_search",
            project_name="cnn_bay_tuning"
        )

        tuner.search(
            X_train, {'abundance_output': y_train_abundance, 'planetary_output': y_train_planetary},
            validation_split=0.2,
            epochs=search_epochs
        )

        # Get the best hyperparameters and store them in the class
        self.best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        return self.best_hps

    # def show_best_parameters(self):
    #     """
    #     Displays the best hyperparameters found during the search.
    #     """
    #     if self.best_hps:
    #         print(f"Best number of convolutional layers: {self.best_hps.get('conv_layers')}")
    #         print(f"Best number of dense layers: {self.best_hps.get('dense_layers')}")
    #         print(f"Best dropout rate: {self.best_hps.get('dropout_rate')}")
    #         print(f"Best learning rate: {self.best_hps.get('learning_rate')}")
    #         print(f"Best filters: {self.best_hps.get('filters')}")
    #         print(f"Best units: {self.best_hps.get('units')}")
    #         print(f"Best kernel size: {self.best_hps.get('kernel_size')}")
    #     else:
    #         print("No hyperparameters found. Run the search method first.")
    
    def fit_best_model(self, train_dataset, validation_dataset, epochs=100, patience=20, 
                    train_steps_per_epoch=None, validation_steps=None):
        
        if not self.best_hps:
            raise ValueError("No best hyperparameters found. Please run the 'search' method first.")

        self.best_model = self.build_model(self.best_hps)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True)

        # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        self.history = self.best_model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=train_steps_per_epoch,
            validation_data=validation_dataset,
            validation_steps=validation_steps,
            callbacks=[early_stopping],
            verbose=2
        )

        return self.best_model, self.history


    def evaluate(self, test_dataset, feature_names_abundance, feature_names_planetary,
                y_scalers_abundance=None, y_scalers_planetary=None, plot=True, additional_metrics=True):
        """
        Evaluates the model using a tf.data.Dataset, returning evaluation metrics and optional plots.
        """
        if self.best_model is None:
            raise ValueError("No model is trained yet. Please run 'fit_best_model' first.")

        # Initialize lists to collect true and predicted values
        y_true_abundance_list = []
        y_pred_abundance_list = []
        y_true_planetary_list = []
        y_pred_planetary_list = []

        # Iterate over the test dataset to collect predictions and true values
        for X_batch, y_batch in test_dataset:
            # Make predictions
            y_pred_batch = self.best_model.predict(X_batch, verbose=0)
            
            # Extract true values
            y_true_abundance_batch = y_batch['abundance_output']
            y_true_planetary_batch = y_batch['planetary_output']
            
            # Extract predicted values
            y_pred_abundance_batch = y_pred_batch[0]
            y_pred_planetary_batch = y_pred_batch[1]
            
            # Append to lists
            y_true_abundance_list.append(y_true_abundance_batch.numpy())
            y_pred_abundance_list.append(y_pred_abundance_batch)
            y_true_planetary_list.append(y_true_planetary_batch.numpy())
            y_pred_planetary_list.append(y_pred_planetary_batch)
        
        # Concatenate the lists
        y_true_abundance = np.concatenate(y_true_abundance_list, axis=0)
        y_pred_abundance = np.concatenate(y_pred_abundance_list, axis=0)
        y_true_planetary = np.concatenate(y_true_planetary_list, axis=0)
        y_pred_planetary = np.concatenate(y_pred_planetary_list, axis=0)
       
        # Inverse transform abundances
        if y_scalers_abundance:
            for i, scaler in enumerate(y_scalers_abundance):
                # First, inverse the scaling
                y_true_abundance[:, i] = scaler.inverse_transform(y_true_abundance[:, i].reshape(-1, 1)).flatten()
                y_pred_abundance[:, i] = scaler.inverse_transform(y_pred_abundance[:, i].reshape(-1, 1)).flatten()
                
        # Inverse transform planetary parameters
        if y_scalers_planetary:
            for i, scaler in enumerate(y_scalers_planetary):
                y_true_planetary[:, i] = scaler.inverse_transform(y_true_planetary[:, i].reshape(-1, 1)).flatten()
                y_pred_planetary[:, i] = scaler.inverse_transform(y_pred_planetary[:, i].reshape(-1, 1)).flatten()
        
        # Evaluate abundances
        results_abundance = self._evaluate_outputs(
            y_true_abundance, y_pred_abundance, feature_names_abundance, plot, additional_metrics, title='Abundances'
        )
        
        # Evaluate planetary parameters
        results_planetary = self._evaluate_outputs(
            y_true_planetary, y_pred_planetary, feature_names_planetary, plot, additional_metrics, title='Planetary Parameters'
        )
        
        return {'Abundances': results_abundance, 'Planetary Parameters': results_planetary}


    def _evaluate_outputs(self, y_true, y_pred, feature_names, plot, additional_metrics, title):
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
        
            plt.suptitle(title)
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
        """
        Generates predictions with uncertainty estimates using Monte Carlo Dropout.
        """
        if self.best_model is None:
            raise ValueError("No model is trained yet. Please run 'fit_best_model' first.")

        predictions = {'abundance': [], 'planetary': []}
        for _ in range(n_iter):
            preds = self.best_model.predict(X, batch_size=32, verbose=0)
            # preds is a list [abundance_preds, planetary_preds]
            predictions['abundance'].append(preds[0])
            predictions['planetary'].append(preds[1])
        
        # Convert lists to numpy arrays
        predictions['abundance'] = np.array(predictions['abundance'])  # Shape: (n_iter, samples, abundance_outputs)
        predictions['planetary'] = np.array(predictions['planetary'])  # Shape: (n_iter, samples, planetary_outputs)
        
        return predictions
    
    def inverse_transform_predictions(self, predictions, y_scalers_abundance, y_scalers_planetary):
        """
        Inverse transforms the scaled predictions to the original scale.
        """
        epsilon = 1e-6
        n_iter, samples, abundance_outputs = predictions['abundance'].shape
        n_iter, samples, planetary_outputs = predictions['planetary'].shape

        # Initialize arrays
        abundance_preds_inv = np.zeros((n_iter, samples, abundance_outputs))
        planetary_preds_inv = np.zeros((n_iter, samples, planetary_outputs))

        # Inverse transform abundance predictions
        for i in range(abundance_outputs):
            scaler = y_scalers_abundance[i]
            data = predictions['abundance'][:, :, i].reshape(-1, 1)
            scaled_preds = scaler.inverse_transform(data)

        # Inverse transform planetary predictions
        for i in range(planetary_outputs):
            scaler = y_scalers_planetary[i]
            planetary_preds_inv[:, :, i] = scaler.inverse_transform(predictions['planetary'][:, :, i])

        return {'abundance': abundance_preds_inv, 'planetary': planetary_preds_inv}
