import * as tf from '@tensorflow/tfjs';

/**
 * Creates a Convolutional Neural Network (CNN) model for MNIST digit classification
 * This architecture is based on classic CNN designs for image recognition
 * @returns A compiled TensorFlow.js Sequential model ready for training
 */
export function createCNNModel(): tf.Sequential {
  // Sequential model means layers are stacked linearly, one after another
  const model = tf.sequential();

  // First convolutional layer
  // This layer learns to detect basic features like edges and curves
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1], // MNIST images: 28x28 pixels, 1 channel (grayscale)
    kernelSize: 5, // 5x5 filter size - each filter looks at 5x5 pixel regions
    filters: 8, // Number of different filters to learn (8 different feature detectors)
    strides: 1, // Move filter by 1 pixel at a time (no skipping)
    activation: 'relu', // ReLU activation: f(x) = max(0, x) - introduces non-linearity
    kernelInitializer: 'varianceScaling' // Smart weight initialization for better training
  }));

  // First pooling layer
  // Reduces spatial dimensions while keeping important features
  // This makes the model more efficient and helps prevent overfitting
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2], // Look at 2x2 regions
    strides: [2, 2] // Move by 2 pixels (non-overlapping pooling)
    // This reduces dimensions from 24x24 to 12x12
  }));

  // Second convolutional layer
  // Learns more complex features by combining simple features from first layer
  model.add(tf.layers.conv2d({
    kernelSize: 5, // Another 5x5 filter
    filters: 16, // More filters to learn more complex patterns
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // Second pooling layer
  // Further reduces dimensions and computational load
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
    // Reduces from 8x8 to 4x4
  }));

  // Flatten layer
  // Converts 2D feature maps to 1D vector for the dense layer
  // Output shape: 4 * 4 * 16 = 256 neurons
  model.add(tf.layers.flatten());

  // Dense (fully connected) output layer
  // Maps the 256 features to 10 classes (digits 0-9)
  model.add(tf.layers.dense({
    units: 10, // 10 output neurons, one for each digit class
    kernelInitializer: 'varianceScaling',
    activation: 'softmax' // Softmax converts outputs to probabilities that sum to 1
  }));

  // Configure the model for training
  const optimizer = tf.train.adam(); // Adam optimizer: adaptive learning rate algorithm
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy', // Standard loss for multi-class classification
    metrics: ['accuracy'] // Track accuracy during training
  });

  return model;
}

/*
 * Model Architecture Summary:
 * 
 * Input: 28x28x1 (grayscale MNIST image)
 * ↓
 * Conv2D: 5x5 kernel, 8 filters → Output: 24x24x8
 * ↓
 * MaxPool2D: 2x2 pool → Output: 12x12x8
 * ↓
 * Conv2D: 5x5 kernel, 16 filters → Output: 8x8x16
 * ↓
 * MaxPool2D: 2x2 pool → Output: 4x4x16
 * ↓
 * Flatten → Output: 256
 * ↓
 * Dense: 10 units with softmax → Output: 10 (probability for each digit)
 */