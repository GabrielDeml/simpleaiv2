import * as tf from '@tensorflow/tfjs';

// ========== Model Configuration ==========
const INPUT_SHAPE: [number, number, number] = [28, 28, 1]; // H x W x Channels
const NUM_CLASSES = 10; // Digits 0-9

// CNN Architecture parameters
const CONV1_FILTERS = 8;
const CONV2_FILTERS = 16;
const KERNEL_SIZE = 5;
const POOL_SIZE: [number, number] = [2, 2];
const STRIDE_SIZE: [number, number] = [2, 2];

/**
 * Creates a CNN model for MNIST digit classification.
 * Architecture: Conv → Pool → Conv → Pool → Dense
 */
export function createCNNModel(): tf.Sequential {
  const model = tf.sequential();

  // ========== Feature Extraction Layers ==========
  
  // Conv Layer 1: Detect basic features (edges, curves)
  model.add(tf.layers.conv2d({
    inputShape: INPUT_SHAPE,
    kernelSize: KERNEL_SIZE,
    filters: CONV1_FILTERS,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // Pool Layer 1: Reduce dimensions 24x24 → 12x12
  model.add(tf.layers.maxPooling2d({
    poolSize: POOL_SIZE,
    strides: STRIDE_SIZE
  }));

  // Conv Layer 2: Detect complex patterns
  model.add(tf.layers.conv2d({
    kernelSize: KERNEL_SIZE,
    filters: CONV2_FILTERS,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // Pool Layer 2: Reduce dimensions 8x8 → 4x4
  model.add(tf.layers.maxPooling2d({
    poolSize: POOL_SIZE,
    strides: STRIDE_SIZE
  }));

  // ========== Classification Layers ==========
  
  // Flatten: 4x4x16 → 256
  model.add(tf.layers.flatten());

  // Output Layer: 256 → 10 classes
  model.add(tf.layers.dense({
    units: NUM_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));

  // ========== Compile Model ==========
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

/**
 * Model Architecture Flow:
 * 
 * Input Layer:      28×28×1  (784 pixels)
 *      ↓
 * Conv2D + ReLU:    24×24×8  (4,608 neurons)
 *      ↓
 * MaxPool 2×2:      12×12×8  (1,152 neurons)
 *      ↓
 * Conv2D + ReLU:    8×8×16   (1,024 neurons)
 *      ↓
 * MaxPool 2×2:      4×4×16   (256 neurons)
 *      ↓
 * Flatten:          256      (dense vector)
 *      ↓
 * Dense + Softmax:  10       (class probabilities)
 * 
 * Total parameters: ~11,000
 */

// ========== Utility Functions ==========

/**
 * Prints a summary of the model architecture.
 */
export function printModelSummary(model: tf.Sequential): void {
  model.summary();
}

/**
 * Gets the total number of trainable parameters in the model.
 */
export function getModelParameterCount(model: tf.Sequential): number {
  return model.countParams();
}