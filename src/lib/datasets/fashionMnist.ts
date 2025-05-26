import * as tf from '@tensorflow/tfjs';

/**
 * Fashion-MNIST Dataset Loader for TensorFlow.js
 * 
 * Fashion-MNIST is a dataset of Zalando's article images, intended as a
 * drop-in replacement for MNIST. It's more challenging than MNIST while
 * maintaining the same image size and structure.
 * 
 * Dataset characteristics:
 * - 70,000 grayscale images of fashion products
 * - Each image is 28x28 pixels (same as MNIST)
 * - 10 classes of clothing and accessories
 * - Split into 60,000 training and 10,000 test images
 * - More complex patterns than handwritten digits
 * 
 * Why Fashion-MNIST?
 * - MNIST is too easy (99.7% accuracy achievable)
 * - Same format allows easy algorithm comparison
 * - More realistic computer vision challenge
 * - Represents real-world e-commerce use case
 * 
 * NOTE: This implementation generates synthetic patterns that approximate
 * Fashion-MNIST structure. For real data, download from the official source.
 */

// ========== Dataset Constants ==========
/** Image height in pixels (same as MNIST) */
const IMAGE_HEIGHT = 28;

/** Image width in pixels (same as MNIST) */
const IMAGE_WIDTH = 28;

/** Total pixels per image (784, same as MNIST) */
const IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;

/** Number of fashion categories */
const NUM_CLASSES = 10;

/** Total dataset size (standard Fashion-MNIST) */
const NUM_DATASET_ELEMENTS = 70000;

/** Training set size (standard split) */
const NUM_TRAIN_ELEMENTS = 60000;

/** Test set size (standard split) */
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

/**
 * URL paths for Fashion-MNIST data (placeholder URLs)
 * Real Fashion-MNIST is available from:
 * - GitHub: https://github.com/zalandoresearch/fashion-mnist
 * - Direct download: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/
 */
// const FASHION_MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/fashion_mnist_images.png';
// const FASHION_MNIST_LABELS_PATH = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/fashion_mnist_labels.json';

/**
 * Fashion-MNIST class names (labels 0-9)
 * Each class represents a type of clothing or accessory
 */
const FASHION_CLASSES = [
  'T-shirt/top',  // 0: Upper body garments (T-shirts, tops)
  'Trouser',      // 1: Full-length leg wear
  'Pullover',     // 2: Sweaters worn by pulling over head
  'Dress',        // 3: One-piece garments for women
  'Coat',         // 4: Outer garments (coats, jackets)
  'Sandal',       // 5: Open-toed footwear
  'Shirt',        // 6: Button-up shirts
  'Sneaker',      // 7: Athletic/casual shoes
  'Bag',          // 8: Handbags, backpacks
  'Ankle boot'    // 9: Short boots covering the ankle
];

/**
 * FashionMnistData - Main class for loading and managing Fashion-MNIST dataset.
 * 
 * Provides the same interface as MNIST for easy swapping between datasets.
 * This makes it simple to test if your model generalizes beyond digits
 * to more complex visual patterns.
 * 
 * @example
 * const fashion = new FashionMnistData();
 * await fashion.load();
 * const {xs: trainImages, ys: trainLabels} = fashion.getTrainData();
 * 
 * // Use same model architecture as MNIST
 * model.fit(trainImages, trainLabels, {
 *   epochs: 10,
 *   validationSplit: 0.1
 * });
 */
export class FashionMnistData {
  // ========== Private Properties ==========
  private datasetImages: Float32Array | null = null;  // All 70K images, normalized [0,1]
  private datasetLabels: Uint8Array | null = null;    // All labels as one-hot vectors
  private trainImages: Float32Array | null = null;    // First 60K images for training
  private testImages: Float32Array | null = null;     // Last 10K images for testing
  private trainLabels: Uint8Array | null = null;     // Training labels (one-hot)
  private testLabels: Uint8Array | null = null;      // Test labels (one-hot)

  /**
   * Loads the Fashion-MNIST dataset.
   * 
   * Process:
   * 1. Generate synthetic fashion images (or load real data)
   * 2. Create one-hot encoded labels
   * 3. Split into training and test sets
   * 
   * Real Fashion-MNIST format:
   * - Images: IDX3-UBYTE format (same as MNIST)
   * - Labels: IDX1-UBYTE format (same as MNIST)
   * - Can use same loaders as MNIST with different URLs
   */
  async load() {
    await this.loadImages();
    await this.loadLabels();
    this.splitDataset();
  }

  /**
   * Loads or generates Fashion-MNIST images.
   * 
   * For real implementation:
   * - Download IDX files from Fashion-MNIST repository
   * - Parse IDX3-UBYTE format (28x28 grayscale images)
   * - Normalize pixel values from [0,255] to [0,1]
   * 
   * Current implementation generates synthetic patterns that
   * approximate different clothing types for demonstration.
   */
  private async loadImages(): Promise<void> {
    // Generate training images
    const numElements = NUM_TRAIN_ELEMENTS;
    const images = new Float32Array(numElements * IMAGE_SIZE);
    
    for (let i = 0; i < numElements; i++) {
      const syntheticImage = this.generateSyntheticFashionImage(i);
      images.set(syntheticImage, i * IMAGE_SIZE);
    }
    
    // Generate test images (with different random seeds)
    const testImages = new Float32Array(NUM_TEST_ELEMENTS * IMAGE_SIZE);
    for (let i = 0; i < NUM_TEST_ELEMENTS; i++) {
      const syntheticImage = this.generateSyntheticFashionImage(i + NUM_TRAIN_ELEMENTS);
      testImages.set(syntheticImage, i * IMAGE_SIZE);
    }
    
    // Combine into single dataset array
    this.datasetImages = new Float32Array(NUM_DATASET_ELEMENTS * IMAGE_SIZE);
    this.datasetImages.set(images, 0);
    this.datasetImages.set(testImages, NUM_TRAIN_ELEMENTS * IMAGE_SIZE);
  }

  /**
   * Generates synthetic fashion item patterns for demonstration.
   * 
   * Each class has a distinct pattern that roughly approximates
   * the shape and structure of the fashion item:
   * 
   * Real Fashion-MNIST images contain:
   * - Centered fashion items on white/gray backgrounds
   * - Various poses and orientations
   * - Different styles within each category
   * - Some items partially cropped
   * - Grayscale values representing fabric textures
   * 
   * @param index - Image index (determines class and random seed)
   * @returns Float32Array of 784 pixels with values in [0,1]
   */
  private generateSyntheticFashionImage(index: number): Float32Array {
    const image = new Float32Array(IMAGE_SIZE);
    const classIndex = index % NUM_CLASSES;
    
    // Calculate image center for radial patterns
    const centerX = IMAGE_WIDTH / 2;
    const centerY = IMAGE_HEIGHT / 2;
    
    // Generate pixels based on class-specific patterns
    for (let y = 0; y < IMAGE_HEIGHT; y++) {
      for (let x = 0; x < IMAGE_WIDTH; x++) {
        const idx = y * IMAGE_WIDTH + x;
        const dx = x - centerX;
        const dy = y - centerY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        let value = 0;
        switch (classIndex) {
          case 0: // T-shirt pattern: horizontal stripes
            value = Math.abs(Math.sin(y * 0.3)) * 0.5;
            break;
            
          case 1: // Trouser pattern: vertical split (legs)
            value = y > IMAGE_HEIGHT / 2 ? 0.7 : 0.2;
            break;
            
          case 2: // Pullover pattern: circular (round neck)
            value = distance < 10 ? 0.8 : 0.3;
            break;
            
          case 3: // Dress pattern: flowing waves
            value = Math.abs(Math.sin(x * 0.2) * Math.cos(y * 0.1)) * 0.7;
            break;
            
          case 4: // Coat pattern: vertical panels (lapels)
            value = x < IMAGE_WIDTH / 3 || x > 2 * IMAGE_WIDTH / 3 ? 0.8 : 0.4;
            break;
            
          case 5: // Sandal pattern: horizontal straps
            value = (y > IMAGE_HEIGHT * 0.7) ? 0.9 : 0.1;
            break;
            
          case 6: // Shirt pattern: vertical lines (buttons)
            value = Math.abs(Math.cos(x * 0.3)) * 0.6;
            break;
            
          case 7: // Sneaker pattern: shoe shape
            value = distance < 8 || y > IMAGE_HEIGHT * 0.8 ? 0.9 : 0.2;
            break;
            
          case 8: // Bag pattern: rectangular shape
            value = (x > IMAGE_WIDTH * 0.3 && x < IMAGE_WIDTH * 0.7) ? 0.8 : 0.3;
            break;
            
          case 9: // Ankle boot pattern: boot silhouette
            value = y > IMAGE_HEIGHT * 0.6 ? 0.9 : (distance < 6 ? 0.7 : 0.2);
            break;
        }
        
        // Add noise to simulate fabric texture and variations
        image[idx] = value + Math.random() * 0.1;
      }
    }
    
    return image;
  }

  /**
   * Creates one-hot encoded labels for the dataset.
   * 
   * Label encoding:
   * - Each label is a 10-element vector
   * - Only one element is 1, others are 0
   * - Index of 1 indicates the class
   * 
   * Example:
   * - T-shirt (class 0): [1,0,0,0,0,0,0,0,0,0]
   * - Sneaker (class 7): [0,0,0,0,0,0,0,1,0,0]
   * 
   * For real Fashion-MNIST:
   * - Labels come as bytes (0-9) in IDX1-UBYTE format
   * - Need to convert to one-hot encoding for neural networks
   */
  private async loadLabels(): Promise<void> {
    const labels = new Uint8Array(NUM_DATASET_ELEMENTS * NUM_CLASSES);
    
    // Generate labels cycling through all classes
    for (let i = 0; i < NUM_DATASET_ELEMENTS; i++) {
      const label = i % NUM_CLASSES;  // Cycle through 0-9
      
      // Create one-hot encoded vector
      const oneHot = new Uint8Array(NUM_CLASSES);
      oneHot[label] = 1;  // Set the class index to 1
      
      // Store in the labels array
      labels.set(oneHot, i * NUM_CLASSES);
    }
    
    this.datasetLabels = labels;
  }

  /**
   * Splits the dataset into training and test sets.
   * 
   * Fashion-MNIST standard split:
   * - Training: 60,000 samples (85.7%)
   * - Test: 10,000 samples (14.3%)
   * 
   * This split is consistent with original MNIST,
   * allowing direct performance comparison.
   */
  private splitDataset(): void {
    if (!this.datasetImages || !this.datasetLabels) {
      throw new Error('Dataset not loaded');
    }

    // Split images: each image is IMAGE_SIZE (784) floats
    this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    
    // Split labels: each label is NUM_CLASSES (10) bytes
    this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }

  /**
   * Returns training data as TensorFlow.js tensors.
   * 
   * @param numExamples - Optional limit on number of examples.
   *                      Useful for quick experiments or debugging.
   * @returns Object containing:
   *   - xs: Tensor4D of shape [N, 28, 28, 1] - Grayscale images in [0,1]
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
   *   - xs: Tensor4D of shape [N, 28, 28, 1] - Grayscale images in [0,1]
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
   * Converts raw arrays to TensorFlow.js tensors.
   * 
   * Tensor shapes (same as MNIST):
   * - Images: [batch_size, height, width, channels]
   *   - batch_size: Number of examples
   *   - height/width: 28x28 pixels
   *   - channels: 1 (grayscale, not RGB)
   * - Labels: [batch_size, num_classes]
   *   - batch_size: Number of examples
   *   - num_classes: 10 (one-hot encoded)
   * 
   * The grayscale format (1 channel) is important:
   * - Reduces model parameters vs RGB (3 channels)
   * - Fashion items recognizable without color
   * - Maintains compatibility with MNIST models
   * 
   * @param images - Flattened grayscale images
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
        [numExamples, IMAGE_HEIGHT, IMAGE_WIDTH, 1]  // Note: 1 channel for grayscale
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
        [numImages, IMAGE_HEIGHT, IMAGE_WIDTH, 1]
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
 * Convenience function to load Fashion-MNIST with metadata.
 * 
 * Returns the same format as loadMNIST() for easy dataset swapping.
 * This allows you to test if models trained on MNIST generalize
 * to more complex visual patterns.
 * 
 * @returns Promise resolving to object containing:
 *   - trainImages: Float32Array of shape [60000 * 784] with values in [0,1]
 *   - trainLabels: Uint8Array of shape [60000 * 10] with one-hot encoding
 *   - testImages: Float32Array of shape [10000 * 784] with values in [0,1]
 *   - testLabels: Uint8Array of shape [10000 * 10] with one-hot encoding
 *   - imageHeight: 28 (pixels)
 *   - imageWidth: 28 (pixels) 
 *   - numClasses: 10
 *   - classNames: Array of fashion category names
 * 
 * @example
 * // Easy dataset swapping
 * const useFashion = true;
 * const dataset = useFashion ? await loadFashionMNIST() : await loadMNIST();
 * 
 * // Same preprocessing and model architecture works for both
 * const model = createCNNModel();
 * const {xs, ys} = processDataset(dataset.trainImages, dataset.trainLabels);
 * 
 * // Fashion-MNIST typically achieves 85-92% accuracy vs 98-99% for MNIST
 * await model.fit(xs, ys, {epochs: 10});
 */
export async function loadFashionMNIST() {
  const data = new FashionMnistData();
  await data.load();
  
  return {
    trainImages: data['trainImages'] as Float32Array,
    trainLabels: data['trainLabels'] as Uint8Array,
    testImages: data['testImages'] as Float32Array,
    testLabels: data['testLabels'] as Uint8Array,
    imageHeight: IMAGE_HEIGHT,
    imageWidth: IMAGE_WIDTH,
    numClasses: NUM_CLASSES,
    classNames: FASHION_CLASSES
  };
}