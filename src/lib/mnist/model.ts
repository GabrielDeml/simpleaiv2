import * as tf from '@tensorflow/tfjs';

/**
 * CNN Model Builder for MNIST Digit Classification
 * 
 * This module creates a Convolutional Neural Network (CNN) optimized for
 * recognizing handwritten digits from the MNIST dataset.
 * 
 * Architecture Overview:
 * - 2 Convolutional layers for feature extraction
 * - 2 MaxPooling layers for dimensionality reduction
 * - 1 Dense layer for classification
 * - Total parameters: ~11,000 (lightweight for browser execution)
 */

// ========== Model Configuration ==========
/**
 * Input shape matches MNIST image dimensions
 * [height, width, channels] where channels=1 for grayscale
 */
const INPUT_SHAPE: [number, number, number] = [28, 28, 1];

/** Number of output classes (digits 0-9) */
const NUM_CLASSES = 10;

// ========== CNN Architecture Parameters ==========
/**
 * Convolutional layer configurations:
 * - CONV1_FILTERS: 8 filters detect basic features (edges, corners)
 * - CONV2_FILTERS: 16 filters combine basic features into complex patterns
 * - KERNEL_SIZE: 5x5 sliding window for feature detection
 */
const CONV1_FILTERS = 8;   // First conv layer: few filters for basic features
const CONV2_FILTERS = 16;  // Second conv layer: more filters for complex features
const KERNEL_SIZE = 5;     // 5x5 convolution window

/**
 * Pooling layer configurations:
 * - POOL_SIZE: 2x2 window takes maximum value
 * - STRIDE_SIZE: 2x2 step size (non-overlapping pooling)
 * - Effect: Reduces spatial dimensions by half
 */
const POOL_SIZE: [number, number] = [2, 2];
const STRIDE_SIZE: [number, number] = [2, 2];

/**
 * Creates a CNN model for MNIST digit classification.
 * 
 * Network Architecture:
 * 1. Conv2D (8 filters) → ReLU activation
 * 2. MaxPooling2D (2x2)
 * 3. Conv2D (16 filters) → ReLU activation  
 * 4. MaxPooling2D (2x2)
 * 5. Flatten
 * 6. Dense (10 units) → Softmax activation
 * 
 * Design Rationale:
 * - Small filter count (8, 16) keeps model lightweight
 * - 5x5 kernels capture digit strokes effectively
 * - Two conv layers sufficient for MNIST's simple patterns
 * - No dropout needed due to small model size
 * 
 * @returns Compiled TensorFlow.js Sequential model ready for training
 */
export function createCNNModel(): tf.Sequential {
  const model = tf.sequential();

  // ========== Feature Extraction Layers ==========
  
  /**
   * Conv Layer 1: Detects basic features
   * - Input: 28x28x1 (grayscale image)
   * - Output: 24x24x8 (8 feature maps)
   * - Kernel: 5x5 sliding window
   * - Activation: ReLU (f(x) = max(0, x)) for non-linearity
   * - Initializer: VarianceScaling prevents vanishing/exploding gradients
   * 
   * This layer learns to detect edges, curves, and basic strokes
   * that form the building blocks of handwritten digits.
   */
  model.add(tf.layers.conv2d({
    inputShape: INPUT_SHAPE,
    kernelSize: KERNEL_SIZE,
    filters: CONV1_FILTERS,
    strides: 1,              // Move filter 1 pixel at a time
    activation: 'relu',      // ReLU: Rectified Linear Unit
    kernelInitializer: 'varianceScaling'  // He initialization
  }));

  /**
   * Pool Layer 1: Spatial downsampling
   * - Input: 24x24x8
   * - Output: 12x12x8
   * - Operation: Takes maximum value in each 2x2 window
   * 
   * Benefits:
   * - Reduces computation by 75% (24x24 → 12x12)
   * - Provides translation invariance (digit position flexibility)
   * - Retains strongest activations (most important features)
   */
  model.add(tf.layers.maxPooling2d({
    poolSize: POOL_SIZE,     // 2x2 window
    strides: STRIDE_SIZE     // Non-overlapping windows
  }));

  /**
   * Conv Layer 2: Detects complex patterns
   * - Input: 12x12x8
   * - Output: 8x8x16
   * - Combines features from layer 1 into higher-level patterns
   * 
   * This layer learns to detect:
   * - Loops (in 0, 6, 8, 9)
   * - Straight lines (in 1, 4, 7)
   * - Curves and intersections
   * - Digit-specific patterns
   */
  model.add(tf.layers.conv2d({
    kernelSize: KERNEL_SIZE,
    filters: CONV2_FILTERS,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  /**
   * Pool Layer 2: Further spatial reduction
   * - Input: 8x8x16
   * - Output: 4x4x16
   * - Final spatial compression before classification
   * 
   * At this stage, each of the 16 feature maps is just 4x4,
   * containing highly abstract representations of digit features.
   */
  model.add(tf.layers.maxPooling2d({
    poolSize: POOL_SIZE,
    strides: STRIDE_SIZE
  }));

  // ========== Classification Layers ==========
  
  /**
   * Flatten Layer: Convert 3D feature maps to 1D vector
   * - Input: 4x4x16 (3D tensor)
   * - Output: 256 (1D vector)
   * - Operation: Unrolls all values into a single dimension
   * 
   * This prepares the features for the final dense layer,
   * converting spatial feature maps into a feature vector.
   */
  model.add(tf.layers.flatten());

  /**
   * Output Layer: Final classification
   * - Input: 256 features
   * - Output: 10 probabilities (one per digit)
   * - Activation: Softmax ensures outputs sum to 1.0
   * 
   * Softmax formula: exp(xi) / sum(exp(x))
   * Converts raw scores (logits) into probability distribution.
   * 
   * Example output: [0.01, 0.01, 0.95, 0.01, ...] predicts digit '2'
   */
  model.add(tf.layers.dense({
    units: NUM_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'  // Multi-class probability distribution
  }));

  // ========== Compile Model ==========
  /**
   * Model compilation configures the training process:
   * 
   * - Optimizer: Adam (Adaptive Moment Estimation)
   *   - Combines benefits of AdaGrad and RMSProp
   *   - Default learning rate: 0.001
   *   - Adapts learning rate per parameter
   * 
   * - Loss: Categorical Crossentropy
   *   - Standard loss for multi-class classification
   *   - Measures distance between predicted and true probability distributions
   *   - Formula: -sum(y_true * log(y_pred))
   * 
   * - Metrics: Accuracy
   *   - Percentage of correctly classified digits
   *   - Human-interpretable performance measure
   */
  model.compile({
    optimizer: tf.train.adam(),           // Adaptive learning rate optimizer
    loss: 'categoricalCrossentropy',     // Multi-class classification loss
    metrics: ['accuracy']                 // Track classification accuracy
  });

  return model;
}

/**
 * Model Architecture Flow Diagram:
 * 
 * Input Layer:      28×28×1  (784 pixels)
 *      ↓            
 * Conv2D + ReLU:    24×24×8  (4,608 activations)
 *      ↓            [8 filters × 5×5 kernel = 200 + 8 bias = 208 params]
 * MaxPool 2×2:      12×12×8  (1,152 activations)
 *      ↓            [No parameters - just selects max values]
 * Conv2D + ReLU:    8×8×16   (1,024 activations)  
 *      ↓            [8 channels × 16 filters × 5×5 = 3,200 + 16 bias = 3,216 params]
 * MaxPool 2×2:      4×4×16   (256 activations)
 *      ↓            [No parameters]
 * Flatten:          256      (dense vector)
 *      ↓            [No parameters - just reshapes]
 * Dense + Softmax:  10       (class probabilities)
 *                   [256 × 10 weights + 10 bias = 2,570 params]
 * 
 * Total trainable parameters: 208 + 3,216 + 2,570 = 5,994
 * 
 * Memory usage (32-bit floats):
 * - Parameters: ~24 KB
 * - Activations (batch size 32): ~600 KB
 * - Suitable for browser/mobile deployment
 */

// ========== Utility Functions ==========

/**
 * Prints a detailed summary of the model architecture.
 * 
 * Shows for each layer:
 * - Layer type and name
 * - Output shape (excluding batch dimension)
 * - Number of parameters
 * - Total parameters at the end
 * 
 * @param model - The compiled TensorFlow.js model
 * 
 * @example
 * const model = createCNNModel();
 * printModelSummary(model);
 * // Output:
 * // Layer (type)          Output shape         Param #
 * // =======================================================
 * // conv2d (Conv2D)       [null,24,24,8]       208
 * // ...
 */
export function printModelSummary(model: tf.Sequential): void {
  model.summary();
}

/**
 * Gets the total number of trainable parameters in the model.
 * 
 * Useful for:
 * - Estimating model complexity
 * - Calculating memory requirements
 * - Comparing different architectures
 * 
 * @param model - The TensorFlow.js model
 * @returns Total number of trainable parameters
 * 
 * @example
 * const model = createCNNModel();
 * const params = getModelParameterCount(model);
 * console.log(`Model has ${params.toLocaleString()} trainable parameters`);
 * // Output: "Model has 5,994 trainable parameters"
 */
export function getModelParameterCount(model: tf.Sequential): number {
  return model.countParams();
}