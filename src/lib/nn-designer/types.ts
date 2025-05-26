/**
 * Supported layer types in the neural network designer.
 * Each type corresponds to a TensorFlow.js layer implementation.
 */
export type LayerType = 'input' | 'dense' | 'conv2d' | 'maxpooling2d' | 'dropout' | 'flatten' | 'output';

/**
 * Configuration for a single layer in the neural network.
 * This is the primary data structure used throughout the designer.
 */
export interface LayerConfig {
  /** Unique identifier for the layer, used for tracking and updates */
  id: string;
  /** The type of layer (e.g., 'dense', 'conv2d') */
  type: LayerType;
  /** User-friendly display name for the layer */
  name: string;
  /** Layer-specific parameters, structure depends on layer type */
  params: Record<string, any>;
}

/**
 * Parameters for the input layer.
 * Defines the shape of data entering the network.
 */
export interface InputLayerParams {
  /** Input tensor shape (excluding batch dimension) e.g., [28, 28, 1] for MNIST */
  shape: number[];
}

/**
 * Parameters for fully connected (dense) layers.
 * These are the most common layers in neural networks.
 */
export interface DenseLayerParams {
  /** Number of neurons/units in the layer */
  units: number;
  /** Activation function to apply to layer output */
  activation: 'relu' | 'sigmoid' | 'tanh' | 'softmax' | 'linear';
  /** Whether to include bias terms (usually true) */
  useBias: boolean;
  /** Weight initialization strategy (e.g., 'glorotUniform', 'heNormal') */
  kernelInitializer: string;
}

/**
 * Parameters for 2D convolutional layers.
 * Used for processing image data and extracting spatial features.
 */
export interface Conv2DLayerParams {
  /** Number of output filters (feature maps) */
  filters: number;
  /** Size of the convolution window (height, width) */
  kernelSize: number | [number, number];
  /** Stride of the convolution (how much to move the window) */
  strides: number | [number, number];
  /** Padding mode: 'valid' (no padding) or 'same' (preserve dimensions) */
  padding: 'valid' | 'same';
  /** Activation function for the layer output */
  activation: 'relu' | 'sigmoid' | 'tanh' | 'linear';
  /** Whether to include bias terms */
  useBias: boolean;
}

/**
 * Parameters for 2D max pooling layers.
 * Used to downsample feature maps and reduce computational load.
 */
export interface MaxPooling2DLayerParams {
  /** Size of the pooling window (height, width) */
  poolSize: number | [number, number];
  /** Stride of the pooling operation */
  strides: number | [number, number];
  /** Padding mode: 'valid' or 'same' */
  padding: 'valid' | 'same';
}

/**
 * Parameters for dropout layers.
 * Used for regularization to prevent overfitting.
 */
export interface DropoutLayerParams {
  /** Fraction of units to drop (0.0 to 1.0), e.g., 0.2 means drop 20% */
  rate: number;
}

/**
 * Parameters for flatten layers.
 * Converts multi-dimensional input to 1D, typically before dense layers.
 */
export interface FlattenLayerParams {
  // No parameters needed - this layer simply reshapes the input
}

/**
 * Parameters for output layers.
 * Similar to dense layers but specifically designed as final prediction layer.
 */
export interface OutputLayerParams {
  /** Number of output classes/units */
  units: number;
  /** Activation function, typically 'softmax' for classification */
  activation: 'relu' | 'sigmoid' | 'tanh' | 'softmax' | 'linear';
  /** Whether to include bias terms (usually true) */
  useBias: boolean;
  /** Weight initialization strategy */
  kernelInitializer: string;
}

/**
 * Configuration for model training.
 * Defines hyperparameters and training behavior.
 */
export interface TrainingConfig {
  /** Number of complete passes through the training data */
  epochs: number;
  /** Number of samples processed before updating weights */
  batchSize: number;
  /** Step size for weight updates (affects convergence speed) */
  learningRate: number;
  /** Optimization algorithm for weight updates */
  optimizer: 'adam' | 'sgd' | 'rmsprop';
  /** Loss function to minimize during training */
  loss: 'categoricalCrossentropy' | 'meanSquaredError' | 'binaryCrossentropy';
  /** Fraction of training data to use for validation (0.0 to 1.0) */
  validationSplit: number;
}

/**
 * Summary statistics for a compiled model.
 * Used to display model complexity and architecture info.
 */
export interface ModelSummary {
  /** Total number of parameters (weights + biases) in the model */
  totalParams: number;
  /** Number of parameters that will be updated during training */
  trainableParams: number;
  /** Total number of layers in the model */
  layerCount: number;
  /** Shape of the model's output tensor (excluding batch dimension) */
  outputShape: number[];
}

/**
 * Available dataset types in the designer.
 * Each dataset has specific characteristics and preprocessing requirements.
 */
export type DatasetType = 'mnist' | 'cifar10' | 'fashion-mnist' | 'custom';

/**
 * Dataset metadata and configuration.
 * Describes the characteristics of a dataset for proper model setup.
 */
export interface Dataset {
  /** Identifier for the dataset type */
  type: DatasetType;
  /** Human-readable name for display */
  name: string;
  /** Shape of a single data sample (e.g., [28, 28, 1] for MNIST) */
  shape: number[];
  /** Number of distinct classes/categories in the dataset */
  classes: number;
  /** Number of samples in the training set */
  trainSize: number;
  /** Number of samples in the test set */
  testSize: number;
}