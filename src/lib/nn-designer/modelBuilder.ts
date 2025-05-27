import * as tf from '@tensorflow/tfjs';
import type { LayerConfig, TrainingConfig } from './types';
import { 
  isTraining, 
  currentEpoch, 
  trainingHistory, 
  resetTraining 
} from './stores';
import { 
  MultiHeadAttentionLayer, 
  PositionalEncodingLayer, 
  TransformerEncoderBlock 
} from './transformerLayers';

/**
 * ModelBuilder class handles the conversion of visual layer configurations
 * into TensorFlow.js models, along with compilation, training, and prediction.
 * Manages the lifecycle of the underlying TF.js model.
 */
export class ModelBuilder {
  /** The TensorFlow.js Sequential model instance */
  private model: tf.Sequential | null = null;
  /** Flag to signal early training termination */
  private stopTraining = false;

  /**
   * Builds a TensorFlow.js Sequential model from layer configurations.
   * Disposes of any existing model before building a new one.
   * @param layerConfigs - Array of layer configurations from the visual designer
   * @returns The compiled TensorFlow.js Sequential model
   */
  buildModel(layerConfigs: LayerConfig[]): tf.Sequential {
    // Validate input
    if (!layerConfigs || layerConfigs.length === 0) {
      throw new Error('No layers defined');
    }

    if (layerConfigs[0].type !== 'input') {
      throw new Error('First layer must be an input layer');
    }

    if (this.model) {
      this.model.dispose();
    }

    this.model = tf.sequential();
    let inputShape: number[] | undefined;

    for (let i = 0; i < layerConfigs.length; i++) {
      const layer = layerConfigs[i];

      // Handle input layer - it doesn't create a TF.js layer but defines input shape
      if (layer.type === 'input') {
        inputShape = layer.params.shape;
        continue; // Skip to next layer
      }

      // Add inputShape to first real layer (required by TF.js)
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
          // Add channel dimension if needed (e.g., [28, 28] -> [28, 28, 1])
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
          // For image data, ensure we handle the channel dimension properly
          // If inputShape is 2D (e.g., [28, 28]), add channel dimension for compatibility
          const flattenInputShape = isFirstLayer && inputShape 
            ? inputShape.length === 2 ? [...inputShape, 1] : inputShape
            : undefined;
            
          this.model.add(tf.layers.flatten({
            inputShape: flattenInputShape
          }));
          break;

        case 'output':
          this.model.add(tf.layers.dense({
            units: layer.params.units,
            activation: layer.params.activation as any,
            useBias: layer.params.useBias || true,
            kernelInitializer: layer.params.kernelInitializer || 'glorotUniform',
            inputShape: isFirstLayer ? inputShape : undefined
          }));
          break;

        case 'embedding':
          this.model.add(tf.layers.embedding({
            inputDim: layer.params.vocabSize,
            outputDim: layer.params.embeddingDim,
            inputLength: layer.params.maxLength,
            trainable: layer.params.trainable !== false,
            inputShape: isFirstLayer ? inputShape : undefined
          }));
          break;

        case 'multiHeadAttention':
          this.model.add(new MultiHeadAttentionLayer({
            numHeads: layer.params.numHeads,
            keyDim: layer.params.keyDim,
            valueDim: layer.params.valueDim,
            dropout: layer.params.dropout || 0.0,
            useBias: layer.params.useBias !== false
          }));
          break;

        case 'layerNormalization':
          this.model.add(tf.layers.layerNormalization({
            epsilon: layer.params.epsilon || 1e-6,
            center: layer.params.center !== false,
            scale: layer.params.scale !== false,
            inputShape: isFirstLayer ? inputShape : undefined
          }));
          break;

        case 'positionalEncoding':
          // Validate that we're not using positional encoding with image data
          if (this.model.layers.length > 0) {
            const prevLayer = this.model.layers[this.model.layers.length - 1];
            const outputShape = prevLayer.outputShape;
            if (Array.isArray(outputShape) && outputShape.length > 3) {
              console.warn('PositionalEncoding layer is designed for sequence data, not image data');
            }
          }
          this.model.add(new PositionalEncodingLayer({
            maxLength: layer.params.maxLength
          }));
          break;

        case 'transformerBlock':
          this.model.add(new TransformerEncoderBlock({
            numHeads: layer.params.numHeads,
            keyDim: layer.params.keyDim,
            ffDim: layer.params.ffDim,
            dropout: layer.params.dropout || 0.1
          }));
          break;

        default:
          throw new Error(`Unknown layer type: ${layer.type}`);
      }
    }

    return this.model;
  }

  /**
   * Compiles the model with optimizer, loss function, and metrics.
   * Must be called after buildModel() and before training.
   * @param config - Training configuration with optimizer settings
   * @throws Error if model hasn't been built yet
   */
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

  /**
   * Trains the model on provided data with real-time progress updates.
   * Updates global training state stores during training.
   * @param trainData - Training input data tensor
   * @param trainLabels - Training labels tensor (one-hot encoded)
   * @param config - Training configuration
   * @param onEpochEnd - Optional callback for each epoch completion
   * @returns Training history object
   * @throws Error if model hasn't been built and compiled
   */
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
              
              // Update training history store with new metrics
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

            // Check if training should stop (allows user cancellation)
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

  /**
   * Signals the training process to stop after the current batch.
   * The actual stop happens in the next epoch callback.
   */
  stopTrainingProcess(): void {
    this.stopTraining = true;
    // Immediately update the training state
    isTraining.set(false);
    // If model exists, try to stop it immediately
    if (this.model) {
      this.model.stopTraining = true;
    }
  }

  /**
   * Runs inference on input data using the trained model.
   * @param input - Input tensor matching the model's expected shape
   * @returns Prediction tensor
   * @throws Error if model hasn't been built
   */
  predict(input: tf.Tensor): tf.Tensor {
    if (!this.model) {
      throw new Error('Model must be built before prediction');
    }
    return this.model.predict(input) as tf.Tensor;
  }

  /**
   * Returns the underlying TensorFlow.js model instance.
   * @returns The Sequential model or null if not built
   */
  getModel(): tf.Sequential | null {
    return this.model;
  }

  /**
   * Exports the trained model to browser downloads.
   * Uses TensorFlow.js browser download functionality.
   * @throws Error if no model exists
   */
  async exportModel(): Promise<void> {
    if (!this.model) {
      throw new Error('No model to export. Train a model first.');
    }

    // Save model to browser downloads folder
    await this.model.save('downloads://my-neural-network');
  }

  /**
   * Exports the model as JSON artifacts for custom storage.
   * Returns the model topology and weights in a structured format.
   * @returns Object containing model topology, weight specs, and weight data
   * @throws Error if no model exists
   */
  async exportModelAsJSON(): Promise<{ modelTopology: any; weightSpecs: any; weightData: ArrayBuffer }> {
    if (!this.model) {
      throw new Error('No model to export. Train a model first.');
    }

    // Get model artifacts using a custom save handler that captures the data
    let modelArtifacts: any = null;
    await this.model.save(tf.io.withSaveHandler(async (artifacts) => {
      modelArtifacts = artifacts;
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
    }));
    
    return {
      modelTopology: modelArtifacts.modelTopology,
      weightSpecs: modelArtifacts.weightSpecs || [],
      weightData: modelArtifacts.weightData || new ArrayBuffer(0)
    };
  }

  /**
   * Disposes of the TensorFlow.js model to free GPU memory.
   * Should be called when switching models or cleaning up.
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
  }
}

/**
 * Singleton instance of ModelBuilder.
 * Ensures only one model is active at a time to prevent memory leaks.
 */
export const modelBuilder = new ModelBuilder();