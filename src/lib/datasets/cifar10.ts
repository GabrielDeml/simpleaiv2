import * as tf from '@tensorflow/tfjs';

/**
 * CIFAR-10 Dataset Loader for TensorFlow.js
 * 
 * The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes,
 * with 6,000 images per class. It's a more complex dataset than MNIST,
 * featuring real-world objects in natural scenes.
 * 
 * Dataset characteristics:
 * - 60,000 RGB color images (50,000 training + 10,000 test)
 * - Image dimensions: 32x32 pixels with 3 color channels (RGB)
 * - 10 mutually exclusive classes
 * - Balanced dataset: exactly 6,000 images per class
 * 
 * NOTE: This implementation generates synthetic data that mimics CIFAR-10
 * structure for demonstration purposes. Real CIFAR-10 data requires
 * downloading and processing the original dataset files.
 */

// ========== Dataset Constants ==========
/** Image height in pixels (CIFAR-10 standard) */
const IMAGE_HEIGHT = 32;

/** Image width in pixels (CIFAR-10 standard) */
const IMAGE_WIDTH = 32;

/** Number of color channels (Red, Green, Blue) */
const IMAGE_CHANNELS = 3;

/** Total pixels per image (32 * 32 * 3 = 3,072) */
const IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS;

/** Number of object categories */
const NUM_CLASSES = 10;

/** Training set size (standard CIFAR-10 split) */
const NUM_TRAIN_ELEMENTS = 50000;

/** Test set size (standard CIFAR-10 split) */
const NUM_TEST_ELEMENTS = 10000;

/**
 * Official CIFAR-10 dataset URL (for reference)
 * The actual dataset is distributed as Python pickle files
 * compressed in a tar.gz archive (~170 MB)
 */
const CIFAR10_BASE_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz';

/**
 * CIFAR-10 class names in order (0-9)
 * These represent common objects and animals
 */
const CIFAR10_CLASSES = [
  'airplane',    // 0: Aircraft of various types
  'automobile',  // 1: Cars, trucks (but not large trucks)
  'bird',        // 2: Various bird species
  'cat',         // 3: Domestic cats
  'deer',        // 4: Various deer species
  'dog',         // 5: Domestic dogs
  'frog',        // 6: Various frog species
  'horse',       // 7: Horses
  'ship',        // 8: Large boats and ships
  'truck'        // 9: Large trucks, semis
];

/**
 * Cifar10Data - Main class for loading and managing CIFAR-10 dataset.
 * 
 * This implementation generates synthetic data for demonstration.
 * For production use, implement proper CIFAR-10 loading from:
 * - Official Python pickle files
 * - TensorFlow Datasets
 * - Pre-processed JavaScript-friendly formats
 * 
 * @example
 * const cifar10 = new Cifar10Data();
 * await cifar10.load();
 * const {xs: trainImages, ys: trainLabels} = cifar10.getTrainData();
 */
export class Cifar10Data {
  // ========== Private Properties ==========
  private trainImages: Float32Array | null = null;  // 50K images, normalized [0,1]
  private trainLabels: Uint8Array | null = null;    // 50K one-hot labels
  private testImages: Float32Array | null = null;   // 10K images, normalized [0,1]
  private testLabels: Uint8Array | null = null;     // 10K one-hot labels

  /**
   * Loads the CIFAR-10 dataset.
   * 
   * In a real implementation, this would:
   * 1. Download the tar.gz file from the official source
   * 2. Extract the Python pickle files
   * 3. Parse the binary data format
   * 4. Normalize pixel values to [0,1]
   * 5. Convert labels to one-hot encoding
   * 
   * Current implementation generates synthetic data for testing.
   */
  async load() {
    const trainData = await this.loadBatch('train');
    const testData = await this.loadBatch('test');
    
    this.trainImages = trainData.images;
    this.trainLabels = trainData.labels;
    this.testImages = testData.images;
    this.testLabels = testData.labels;
  }

  /**
   * Loads a batch of CIFAR-10 data (train or test).
   * 
   * Real CIFAR-10 format (Python pickle):
   * - Each batch file contains a dictionary with:
   *   - 'data': numpy array of uint8, shape (10000, 3072)
   *   - 'labels': list of 10000 integers (0-9)
   * - Data is stored in row-major order: [R1, G1, B1, R2, G2, B2, ...]
   * - Training data split across 5 files, test data in 1 file
   * 
   * @param type - 'train' for training data, 'test' for test data
   * @returns Object with images and one-hot encoded labels
   */
  private async loadBatch(type: 'train' | 'test'): Promise<{images: Float32Array, labels: Uint8Array}> {
    const numElements = type === 'train' ? NUM_TRAIN_ELEMENTS : NUM_TEST_ELEMENTS;
    
    // Allocate memory for images and labels
    const images = new Float32Array(numElements * IMAGE_SIZE);
    const labels = new Uint8Array(numElements * NUM_CLASSES);
    
    // Generate synthetic data (replace with real data loading)
    for (let i = 0; i < numElements; i++) {
      const syntheticImage = this.generateSyntheticImage(i);
      images.set(syntheticImage, i * IMAGE_SIZE);
      
      // Create one-hot encoded label
      const label = i % NUM_CLASSES;  // Cycles through classes
      const oneHot = new Uint8Array(NUM_CLASSES);
      oneHot[label] = 1;  // Set the correct class to 1
      labels.set(oneHot, i * NUM_CLASSES);
    }
    
    return { images, labels };
  }

  /**
   * Generates a synthetic image for testing purposes.
   * 
   * Creates class-specific color patterns:
   * - Each class has a distinct base color/brightness
   * - Adds random noise to simulate natural variation
   * - Values normalized to [0,1] range
   * 
   * Real CIFAR-10 images contain complex natural scenes with:
   * - Variable backgrounds
   * - Different object poses and angles
   * - Lighting variations
   * - Partial occlusions
   * 
   * @param index - Image index (used to determine class)
   * @returns Float32Array of shape [3072] with RGB values in [0,1]
   */
  private generateSyntheticImage(index: number): Float32Array {
    const image = new Float32Array(IMAGE_SIZE);
    const classIndex = index % NUM_CLASSES;
    
    // Generate pixels with class-specific patterns
    for (let i = 0; i < IMAGE_SIZE; i += 3) {
      // Add random noise to simulate natural variation
      const noise = Math.random() * 0.1;
      
      // Base color depends on class (evenly distributed across spectrum)
      const baseValue = (classIndex / NUM_CLASSES) * 0.8 + 0.1;
      
      // Set RGB values with slight color variation
      image[i] = Math.min(1, baseValue + noise);           // Red channel
      image[i + 1] = Math.min(1, baseValue + noise * 0.8); // Green channel (slightly less noise)
      image[i + 2] = Math.min(1, baseValue + noise * 0.6); // Blue channel (least noise)
    }
    
    return image;
  }

  /**
   * Returns training data as TensorFlow.js tensors.
   * 
   * @param numExamples - Optional limit on number of examples.
   *                      Useful for debugging or mini-batch training.
   * @returns Object containing:
   *   - xs: Tensor4D of shape [N, 32, 32, 3] - RGB images normalized to [0,1]
   *   - ys: Tensor2D of shape [N, 10] - One-hot encoded labels
   * @throws {Error} If data hasn't been loaded yet
   */
  getTrainData(numExamples?: number) {
    if (!this.trainImages || !this.trainLabels) {
      throw new Error('Data not loaded. Call load() first.');
    }
    return this.getData(this.trainImages, this.trainLabels, numExamples);
  }

  /**
   * Returns test data as TensorFlow.js tensors.
   * 
   * @param numExamples - Optional limit on number of examples.
   * @returns Object containing:
   *   - xs: Tensor4D of shape [N, 32, 32, 3] - RGB images normalized to [0,1]
   *   - ys: Tensor2D of shape [N, 10] - One-hot encoded labels
   * @throws {Error} If data hasn't been loaded yet
   */
  getTestData(numExamples?: number) {
    if (!this.testImages || !this.testLabels) {
      throw new Error('Data not loaded. Call load() first.');
    }
    return this.getData(this.testImages, this.testLabels, numExamples);
  }

  /**
   * Converts raw arrays to TensorFlow.js tensors with proper shape.
   * 
   * CIFAR-10 tensor shapes:
   * - Images: [batch_size, height, width, channels]
   *   - batch_size: Number of examples
   *   - height/width: 32x32 pixels
   *   - channels: 3 (RGB color channels)
   * - Labels: [batch_size, num_classes]
   *   - batch_size: Number of examples  
   *   - num_classes: 10 (one-hot encoded)
   * 
   * Memory layout:
   * - Images stored as flattened arrays in CHW format
   * - Each image occupies 3,072 consecutive floats (32*32*3)
   * - RGB values interleaved: [R0,G0,B0,R1,G1,B1,...]
   * 
   * @param images - Flattened image data
   * @param labels - One-hot encoded labels
   * @param numExamples - Optional subset size
   * @returns Object with xs (images) and ys (labels) tensors
   */
  private getData(images: Float32Array, labels: Uint8Array, numExamples?: number) {
    let xs: tf.Tensor4D;
    let ys: tf.Tensor2D;

    if (numExamples != null) {
      // Return requested subset
      xs = tf.tensor4d(
        images.slice(0, numExamples * IMAGE_SIZE),
        [numExamples, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
      );
      ys = tf.tensor2d(
        labels.slice(0, numExamples * NUM_CLASSES),
        [numExamples, NUM_CLASSES]
      );
    } else {
      // Return all available data
      const numImages = images.length / IMAGE_SIZE;
      xs = tf.tensor4d(
        images,
        [numImages, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
      );
      ys = tf.tensor2d(
        labels, 
        [numImages, NUM_CLASSES]
      );
    }

    return { xs, ys };
  }
}

/**
 * Convenience function to load CIFAR-10 dataset with metadata.
 * 
 * Provides a simplified interface that returns raw arrays plus
 * dataset metadata needed for model configuration.
 * 
 * @returns Promise resolving to object containing:
 *   - trainImages: Float32Array of shape [50000 * 3072] with RGB values in [0,1]
 *   - trainLabels: Uint8Array of shape [50000 * 10] with one-hot encoding
 *   - testImages: Float32Array of shape [10000 * 3072] with RGB values in [0,1]
 *   - testLabels: Uint8Array of shape [10000 * 10] with one-hot encoding
 *   - imageHeight: 32 (pixels)
 *   - imageWidth: 32 (pixels)
 *   - imageChannels: 3 (RGB)
 *   - numClasses: 10
 *   - classNames: Array of class names in order
 * 
 * @example
 * const cifar10 = await loadCIFAR10();
 * console.log(`Loaded ${cifar10.trainImages.length / 3072} training images`);
 * console.log(`Classes: ${cifar10.classNames.join(', ')}`);
 * 
 * // Configure model input shape
 * const inputShape = [cifar10.imageHeight, cifar10.imageWidth, cifar10.imageChannels];
 */
export async function loadCIFAR10() {
  const data = new Cifar10Data();
  await data.load();
  
  return {
    trainImages: data['trainImages'] as Float32Array,
    trainLabels: data['trainLabels'] as Uint8Array,
    testImages: data['testImages'] as Float32Array,
    testLabels: data['testLabels'] as Uint8Array,
    imageHeight: IMAGE_HEIGHT,
    imageWidth: IMAGE_WIDTH,
    imageChannels: IMAGE_CHANNELS,
    numClasses: NUM_CLASSES,
    classNames: CIFAR10_CLASSES
  };
}