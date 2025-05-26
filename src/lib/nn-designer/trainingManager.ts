import * as tf from '@tensorflow/tfjs';
import { get } from 'svelte/store';
import { layers, trainingConfig, selectedDataset, isTraining } from './stores';
import { modelBuilder } from './modelBuilder';
import { loadMNIST } from '../mnist/dataLoader';

export class TrainingManager {
  private trainData: tf.Tensor | null = null;
  private trainLabels: tf.Tensor | null = null;
  private testData: tf.Tensor | null = null;
  private testLabels: tf.Tensor | null = null;

  async loadDataset() {
    const dataset = get(selectedDataset);
    
    switch (dataset) {
      case 'mnist':
        const mnistData = await loadMNIST();
        
        // Prepare training data
        this.trainData = tf.tensor4d(
          mnistData.trainImages,
          [mnistData.trainImages.length / (28 * 28), 28, 28, 1]
        );
        
        // Labels are already one-hot encoded from the dataset
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
      case 'fashion-mnist':
        throw new Error(`Dataset ${dataset} not implemented yet`);
        
      default:
        throw new Error(`Unknown dataset: ${dataset}`);
    }
  }

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
        // Auto-add output layer for MNIST (10 classes)
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
      
      // Ensure we have data loaded
      if (!this.trainData || !this.trainLabels) {
        await this.loadDataset();
      }
      
      // Build and compile model
      const model = modelBuilder.buildModel(layerConfigs);
      modelBuilder.compileModel(config);
      
      // Log model summary
      console.log('Model Summary:');
      model.summary();
      
      // Train the model
      await modelBuilder.trainModel(
        this.trainData!,
        this.trainLabels!,
        config,
        (epoch, logs) => {
          console.log(`Epoch ${epoch + 1}:`, logs);
        }
      );
      
      // Evaluate on test set if available
      if (this.testData && this.testLabels) {
        const evaluation = model.evaluate(this.testData, this.testLabels) as tf.Tensor[];
        const testLoss = await evaluation[0].data();
        const testAccuracy = await evaluation[1].data();
        
        console.log(`Test Loss: ${testLoss[0].toFixed(4)}`);
        console.log(`Test Accuracy: ${(testAccuracy[0] * 100).toFixed(2)}%`);
        
        // Clean up tensors
        evaluation.forEach(t => t.dispose());
      }
      
    } catch (error) {
      console.error('Training error:', error);
      isTraining.set(false);
      throw error;
    }
  }

  stopTraining() {
    modelBuilder.stopTrainingProcess();
  }

  async predict(inputData: Float32Array | tf.Tensor): Promise<number[]> {
    const model = modelBuilder.getModel();
    if (!model) {
      throw new Error('No trained model available');
    }

    let input: tf.Tensor;
    if (inputData instanceof Float32Array) {
      // Assume it's a flattened 28x28 image
      input = tf.tensor4d(inputData, [1, 28, 28, 1]);
    } else {
      input = inputData;
    }

    const prediction = model.predict(input) as tf.Tensor;
    const result = await prediction.data();
    
    // Clean up
    if (inputData instanceof Float32Array) {
      input.dispose();
    }
    prediction.dispose();

    return Array.from(result);
  }

  dispose() {
    if (this.trainData) this.trainData.dispose();
    if (this.trainLabels) this.trainLabels.dispose();
    if (this.testData) this.testData.dispose();
    if (this.testLabels) this.testLabels.dispose();
    modelBuilder.dispose();
  }
}

export const trainingManager = new TrainingManager();