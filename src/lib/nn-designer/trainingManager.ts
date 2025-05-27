import * as tf from '@tensorflow/tfjs';
import { get } from 'svelte/store';
import { layers, trainingConfig, selectedDataset, isTraining } from './stores';
import { modelBuilder } from './modelBuilder';
import { getDataset } from '../datasets';
import type { DatasetTensors } from '../datasets';

/**
 * TrainingManager orchestrates the entire training pipeline:
 * dataset loading, model building, training execution, and evaluation.
 * Manages the lifecycle of training data tensors.
 */
export class TrainingManager {
  /** Dataset tensors for training and testing */
  private datasetTensors: DatasetTensors | null = null;

  /**
   * Loads the currently selected dataset into memory as TensorFlow.js tensors.
   * Uses the standardized dataset interface for consistent loading.
   * Disposes of any previously loaded data to prevent memory leaks.
   */
  async loadDataset() {
    // Dispose of any previously loaded data
    if (this.datasetTensors) {
      this.datasetTensors.trainData.dispose();
      this.datasetTensors.trainLabels.dispose();
      this.datasetTensors.testData.dispose();
      this.datasetTensors.testLabels.dispose();
      this.datasetTensors = null;
    }
    
    // Get the selected dataset name
    const datasetName = get(selectedDataset);
    
    // Load dataset using the standardized interface
    const dataset = getDataset(datasetName);
    this.datasetTensors = await dataset.loadTensors({
      shuffle: true,
      cache: true
    });
  }

  /**
   * Main training orchestration method.
   * Builds the model from layer configs, loads data if needed,
   * and executes the training process with progress tracking.
   * @throws Error if model configuration is invalid or training fails
   */
  async startTraining() {
    try {
      // Get current configuration
      const layerConfigs = get(layers);
      const config = get(trainingConfig);
      
      if (layerConfigs.length < 2) {
        throw new Error('Model must have at least an input and output layer');
      }
      
      // Check if the last layer is suitable as an output layer
      const lastLayer = layerConfigs[layerConfigs.length - 1];
      if (lastLayer.type !== 'dense' || lastLayer.params.units !== 10) {
        // Auto-add output layer for 10-class classification
        // This ensures the model can output predictions for all classes
        console.warn('Adding output layer for 10-class classification');
        layerConfigs.push({
          id: 'output-auto',
          type: 'dense',
          name: 'Output',
          params: {
            units: 10,
            activation: 'softmax',
            useBias: true,
            kernelInitializer: 'glorotUniform'
          }
        });
      }
      
      // Ensure we have data loaded before training
      if (!this.datasetTensors) {
        await this.loadDataset();
      }
      
      // Build and compile model from visual layer configuration
      const model = modelBuilder.buildModel(layerConfigs);
      modelBuilder.compileModel(config);
      
      // Log model summary
      console.log('Model Summary:');
      model.summary();
      
      // Train the model with epoch callbacks for progress tracking
      await modelBuilder.trainModel(
        this.datasetTensors!.trainData,
        this.datasetTensors!.trainLabels,
        config,
        (epoch, logs) => {
          // Log training progress to console
          console.log(`Epoch ${epoch + 1}:`, logs);
        }
      );
      
      // Evaluate final model performance on test set
      if (this.datasetTensors!.testData && this.datasetTensors!.testLabels) {
        // Evaluate and clean up tensors properly
        const evaluation = model.evaluate(
          this.datasetTensors!.testData,
          this.datasetTensors!.testLabels
        ) as tf.Tensor[];
        
        // Extract data from tensors
        const testLoss = await evaluation[0].data();
        const testAccuracy = await evaluation[1].data();
        
        // Dispose evaluation tensors to prevent memory leaks
        evaluation.forEach(t => t.dispose());
        
        console.log(`Test Loss: ${testLoss[0].toFixed(4)}`);
        console.log(`Test Accuracy: ${(testAccuracy[0] * 100).toFixed(2)}%`);
      }
      
    } catch (error) {
      console.error('Training error:', error);
      isTraining.set(false);
      throw error;
    }
  }

  /**
   * Stops the current training process.
   * Training will stop after the current batch completes.
   */
  stopTraining() {
    modelBuilder.stopTrainingProcess();
  }

  /**
   * Runs prediction on a single input sample.
   * @param inputData - Either a Float32Array of pixel values or a prepared tensor
   * @returns Array of class probabilities (softmax output)
   * @throws Error if no trained model is available
   */
  async predict(inputData: Float32Array | tf.Tensor): Promise<number[]> {
    const model = modelBuilder.getModel();
    if (!model) {
      throw new Error('No trained model available');
    }

    let input: tf.Tensor | null = null;
    let prediction: tf.Tensor | null = null;
    
    try {
      // Create input tensor if needed
      if (inputData instanceof Float32Array) {
        // Assume it's a flattened 28x28 grayscale image for MNIST
        // Reshape to 4D tensor with batch dimension
        input = tf.tensor4d(inputData, [1, 28, 28, 1]);
      } else {
        input = inputData;
      }

      // Run prediction
      prediction = model.predict(input) as tf.Tensor;
      const result = await prediction.data();
      
      return Array.from(result);
    } finally {
      // Clean up tensors to prevent memory leaks
      // Only dispose input if we created it (not if it was passed in)
      if (inputData instanceof Float32Array && input) {
        input.dispose();
      }
      if (prediction) {
        prediction.dispose();
      }
    }
  }

  /**
   * Disposes of all data tensors and the model.
   * Should be called when switching datasets or cleaning up.
   */
  dispose() {
    if (this.datasetTensors) {
      this.datasetTensors.trainData.dispose();
      this.datasetTensors.trainLabels.dispose();
      this.datasetTensors.testData.dispose();
      this.datasetTensors.testLabels.dispose();
      this.datasetTensors = null;
    }
    modelBuilder.dispose();
  }
}

/**
 * Singleton instance of TrainingManager.
 * Manages all training operations for the neural network designer.
 */
export const trainingManager = new TrainingManager();