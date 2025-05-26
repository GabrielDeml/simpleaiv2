import * as tf from '@tensorflow/tfjs';
import { get } from 'svelte/store';
import { layers, trainingConfig, selectedDataset, isTraining } from './stores';
import { modelBuilder } from './modelBuilder';
import { loadMNIST } from '../mnist/dataLoader';
import { loadCIFAR10 } from '../datasets/cifar10';
import { loadFashionMNIST } from '../datasets/fashionMnist';

/**
 * TrainingManager orchestrates the entire training pipeline:
 * dataset loading, model building, training execution, and evaluation.
 * Manages the lifecycle of training data tensors.
 */
export class TrainingManager {
  /** Training input data tensor */
  private trainData: tf.Tensor | null = null;
  /** Training labels tensor (one-hot encoded) */
  private trainLabels: tf.Tensor | null = null;
  /** Test/validation input data tensor */
  private testData: tf.Tensor | null = null;
  /** Test/validation labels tensor (one-hot encoded) */
  private testLabels: tf.Tensor | null = null;

  /**
   * Loads the currently selected dataset into memory as TensorFlow.js tensors.
   * Handles different dataset formats and converts them to the required shapes.
   * Disposes of any previously loaded data to prevent memory leaks.
   */
  async loadDataset() {
    const dataset = get(selectedDataset);
    
    switch (dataset) {
      case 'mnist':
        const mnistData = await loadMNIST();
        
        // Prepare training data - reshape flat array to 4D tensor [batch, height, width, channels]
        this.trainData = tf.tensor4d(
          mnistData.trainImages,
          [mnistData.trainImages.length / (28 * 28), 28, 28, 1]
        );
        
        // Labels are already one-hot encoded from the dataset loader
        // Shape: [samples, 10] where 10 is the number of classes
        this.trainLabels = tf.tensor2d(
          mnistData.trainLabels,
          [mnistData.trainLabels.length / 10, 10]
        );
        
        // Prepare test data
        this.testData = tf.tensor4d(
          mnistData.testImages,
          [mnistData.testImages.length / (28 * 28), 28, 28, 1]
        );
        
        // Labels are already one-hot encoded from the dataset
        this.testLabels = tf.tensor2d(
          mnistData.testLabels,
          [mnistData.testLabels.length / 10, 10]
        );
        break;
        
      case 'cifar10':
        const cifar10Data = await loadCIFAR10();
        
        this.trainData = tf.tensor4d(
          cifar10Data.trainImages,
          [cifar10Data.trainImages.length / (32 * 32 * 3), 32, 32, 3]
        );
        
        this.trainLabels = tf.tensor2d(
          cifar10Data.trainLabels,
          [cifar10Data.trainLabels.length / 10, 10]
        );
        
        this.testData = tf.tensor4d(
          cifar10Data.testImages,
          [cifar10Data.testImages.length / (32 * 32 * 3), 32, 32, 3]
        );
        
        this.testLabels = tf.tensor2d(
          cifar10Data.testLabels,
          [cifar10Data.testLabels.length / 10, 10]
        );
        break;
        
      case 'fashion-mnist':
        const fashionData = await loadFashionMNIST();
        
        this.trainData = tf.tensor4d(
          fashionData.trainImages,
          [fashionData.trainImages.length / (28 * 28), 28, 28, 1]
        );
        
        this.trainLabels = tf.tensor2d(
          fashionData.trainLabels,
          [fashionData.trainLabels.length / 10, 10]
        );
        
        this.testData = tf.tensor4d(
          fashionData.testImages,
          [fashionData.testImages.length / (28 * 28), 28, 28, 1]
        );
        
        this.testLabels = tf.tensor2d(
          fashionData.testLabels,
          [fashionData.testLabels.length / 10, 10]
        );
        break;
        
      default:
        throw new Error(`Unknown dataset: ${dataset}`);
    }
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
      if (!this.trainData || !this.trainLabels) {
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
        this.trainData!,
        this.trainLabels!,
        config,
        (epoch, logs) => {
          // Log training progress to console
          console.log(`Epoch ${epoch + 1}:`, logs);
        }
      );
      
      // Evaluate final model performance on test set
      if (this.testData && this.testLabels) {
        const evaluation = model.evaluate(this.testData, this.testLabels) as tf.Tensor[];
        const testLoss = await evaluation[0].data();
        const testAccuracy = await evaluation[1].data();
        
        console.log(`Test Loss: ${testLoss[0].toFixed(4)}`);
        console.log(`Test Accuracy: ${(testAccuracy[0] * 100).toFixed(2)}%`);
        
        // Clean up evaluation tensors to free memory
        evaluation.forEach(t => t.dispose());
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

    let input: tf.Tensor;
    if (inputData instanceof Float32Array) {
      // Assume it's a flattened 28x28 grayscale image for MNIST
      // Reshape to 4D tensor with batch dimension
      input = tf.tensor4d(inputData, [1, 28, 28, 1]);
    } else {
      input = inputData;
    }

    const prediction = model.predict(input) as tf.Tensor;
    const result = await prediction.data();
    
    // Clean up tensors to prevent memory leaks
    if (inputData instanceof Float32Array) {
      input.dispose();
    }
    prediction.dispose();

    return Array.from(result);
  }

  /**
   * Disposes of all data tensors and the model.
   * Should be called when switching datasets or cleaning up.
   */
  dispose() {
    if (this.trainData) this.trainData.dispose();
    if (this.trainLabels) this.trainLabels.dispose();
    if (this.testData) this.testData.dispose();
    if (this.testLabels) this.testLabels.dispose();
    modelBuilder.dispose();
  }
}

/**
 * Singleton instance of TrainingManager.
 * Manages all training operations for the neural network designer.
 */
export const trainingManager = new TrainingManager();