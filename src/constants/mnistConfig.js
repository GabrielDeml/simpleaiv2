/**
 * MNIST Training Configuration Constants
 * Centralized configuration for MNIST digit recognition training
 */

// Dataset Configuration
export const DATASET_CONFIG = {
  IMAGE_SIZE: 784, // 28x28 pixels flattened
  NUM_CLASSES: 10, // Digits 0-9
  NUM_DATASET_ELEMENTS: 65000,
  TRAIN_TEST_RATIO: 5/6,
  IMAGE_WIDTH: 28,
  IMAGE_HEIGHT: 28
};

// Calculate derived values
export const TRAINING_CONFIG = {
  NUM_TRAIN_ELEMENTS: Math.floor(DATASET_CONFIG.TRAIN_TEST_RATIO * DATASET_CONFIG.NUM_DATASET_ELEMENTS),
  NUM_TEST_ELEMENTS: DATASET_CONFIG.NUM_DATASET_ELEMENTS - Math.floor(DATASET_CONFIG.TRAIN_TEST_RATIO * DATASET_CONFIG.NUM_DATASET_ELEMENTS),
  BATCH_SIZE: 512,
  EPOCHS: 10,
  VALIDATION_SPLIT: 0.15
};

// Model Architecture Configuration
export const MODEL_CONFIG = {
  HIDDEN_UNITS: 128,
  DROPOUT_RATE: 0.2,
  ACTIVATION: 'relu',
  OUTPUT_ACTIVATION: 'softmax',
  OPTIMIZER: 'adam',
  LOSS: 'categoricalCrossentropy',
  METRICS: ['accuracy']
};

// Data URLs
export const DATA_URLS = {
  IMAGES: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png',
  LABELS: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8'
};

// UI Configuration
export const UI_CONFIG = {
  CANVAS_DISPLAY_SIZE: 200, // Canvas display size in pixels
  CHUNK_SIZE: 5000, // Image processing chunk size
  LOG_CONTAINER_HEIGHT: 300 // Height of logs container in pixels
};

// Performance Configuration
export const PERFORMANCE_CONFIG = {
  MEMORY_CLEANUP_ENABLED: true,
  LOG_LEVEL: 'INFO' // DEBUG, INFO, WARN, ERROR
};