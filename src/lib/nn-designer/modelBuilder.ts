import * as tf from '@tensorflow/tfjs';
import type { LayerConfig, TrainingConfig } from './types';
import { get } from 'svelte/store';
import { 
  layers, 
  isTraining, 
  currentEpoch, 
  trainingHistory, 
  resetTraining 
} from './stores';

export class ModelBuilder {
  private model: tf.Sequential | null = null;
  private stopTraining = false;

  buildModel(layerConfigs: LayerConfig[]): tf.Sequential {
    if (this.model) {
      this.model.dispose();
    }

    this.model = tf.sequential();
    let inputShape: number[] | undefined;

    for (let i = 0; i < layerConfigs.length; i++) {
      const layer = layerConfigs[i];

      // Handle input layer
      if (layer.type === 'input') {
        inputShape = layer.params.shape;
        continue;
      }

      // Add inputShape to first real layer
      const isFirstLayer = this.model.layers.length === 0;
      
      switch (layer.type) {
        case 'dense':
          this.model.add(tf.layers.dense({
            units: layer.params.units,
            activation: layer.params.activation as any,
            useBias: layer.params.useBias,
            kernelInitializer: layer.params.kernelInitializer,
            inputShape: isFirstLayer ? inputShape : undefined
          }));
          break;

        case 'conv2d':
          // Add channel dimension if needed
          const convInputShape = isFirstLayer && inputShape 
            ? inputShape.length === 2 ? [...inputShape, 1] : inputShape
            : undefined;
            
          this.model.add(tf.layers.conv2d({
            filters: layer.params.filters,
            kernelSize: layer.params.kernelSize,
            strides: layer.params.strides,
            padding: layer.params.padding as any,
            activation: layer.params.activation as any,
            useBias: layer.params.useBias,
            inputShape: convInputShape
          }));
          break;

        case 'maxpooling2d':
          this.model.add(tf.layers.maxPooling2d({
            poolSize: layer.params.poolSize,
            strides: layer.params.strides,
            padding: layer.params.padding as any
          }));
          break;

        case 'dropout':
          this.model.add(tf.layers.dropout({
            rate: layer.params.rate
          }));
          break;

        case 'flatten':
          this.model.add(tf.layers.flatten());
          break;
      }
    }

    return this.model;
  }

  compileModel(config: TrainingConfig): void {
    if (!this.model) {
      throw new Error('Model must be built before compiling');
    }

    let optimizer: tf.Optimizer;
    switch (config.optimizer) {
      case 'adam':
        optimizer = tf.train.adam(config.learningRate);
        break;
      case 'sgd':
        optimizer = tf.train.sgd(config.learningRate);
        break;
      case 'rmsprop':
        optimizer = tf.train.rmsprop(config.learningRate);
        break;
      default:
        optimizer = tf.train.adam(config.learningRate);
    }

    this.model.compile({
      optimizer,
      loss: config.loss,
      metrics: ['accuracy']
    });
  }

  async trainModel(
    trainData: tf.Tensor,
    trainLabels: tf.Tensor,
    config: TrainingConfig,
    onEpochEnd?: (epoch: number, logs: tf.Logs) => void
  ): Promise<tf.History> {
    if (!this.model) {
      throw new Error('Model must be built and compiled before training');
    }

    this.stopTraining = false;
    resetTraining();
    isTraining.set(true);

    try {
      const history = await this.model.fit(trainData, trainLabels, {
        epochs: config.epochs,
        batchSize: config.batchSize,
        validationSplit: config.validationSplit,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            if (logs) {
              currentEpoch.set(epoch + 1);
              
              // Update training history
              trainingHistory.update(h => ({
                loss: [...h.loss, logs.loss as number],
                valLoss: [...h.valLoss, logs.val_loss as number || 0],
                accuracy: [...h.accuracy, logs.acc as number],
                valAccuracy: [...h.valAccuracy, logs.val_acc as number || 0]
              }));

              if (onEpochEnd) {
                onEpochEnd(epoch, logs);
              }
            }

            // Check if training should stop
            if (this.stopTraining) {
              this.model!.stopTraining = true;
            }
          }
        }
      });

      return history;
    } finally {
      isTraining.set(false);
    }
  }

  stopTrainingProcess(): void {
    this.stopTraining = true;
  }

  predict(input: tf.Tensor): tf.Tensor {
    if (!this.model) {
      throw new Error('Model must be built before prediction');
    }
    return this.model.predict(input) as tf.Tensor;
  }

  getModel(): tf.Sequential | null {
    return this.model;
  }

  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }
}

// Singleton instance
export const modelBuilder = new ModelBuilder();