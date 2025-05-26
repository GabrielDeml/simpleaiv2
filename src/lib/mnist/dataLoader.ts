import * as tf from '@tensorflow/tfjs';

/**
 * MNIST Dataset Loader for TensorFlow.js
 * 
 * The MNIST (Modified National Institute of Standards and Technology) database
 * is a classic dataset of handwritten digits used for training image processing systems.
 * 
 * Dataset characteristics:
 * - 70,000 grayscale images of handwritten digits (0-9)
 * - Each image is 28x28 pixels
 * - Pixel values range from 0 (white) to 255 (black)
 * - Split into 60,000 training and 10,000 test images
 * 
 * This implementation loads a subset of 65,000 images (55,000 train + 10,000 test)
 * from Google's optimized web format for faster browser loading.
 */

// ========== Dataset Constants ==========
const IMAGE_HEIGHT = 28;          // Standard MNIST image height in pixels
const IMAGE_WIDTH = 28;           // Standard MNIST image width in pixels
const IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH; // 784 pixels per image (flattened)
const NUM_CLASSES = 10;           // Digits 0-9
const NUM_DATASET_ELEMENTS = 65000; // Total images in this web-optimized version
const NUM_TRAIN_ELEMENTS = 55000;   // Training set size (slightly less than standard 60K)
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS; // 10,000 test images

// ========== Dataset URLs ==========
/**
 * Google Cloud Storage hosts an optimized version of MNIST for web applications.
 * - Images are packed into a single PNG sprite for efficient HTTP transfer
 * - Labels are pre-encoded as one-hot vectors in a binary format
 */
const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * MnistData - Main class for loading and managing MNIST dataset in the browser.
 * 
 * Features:
 * - Efficient sprite-based loading (single HTTP request for all images)
 * - Automatic train/test splitting
 * - Memory-efficient chunked processing
 * - Returns data as TensorFlow.js tensors ready for training
 * 
 * @example
 * const mnist = new MnistData();
 * await mnist.load();
 * const {xs: trainImages, ys: trainLabels} = mnist.getTrainData();
 */
export class MnistData {
  // ========== Private Properties ==========
  private datasetImages: Float32Array | null = null;  // All 65K images as normalized floats [0,1]
  private datasetLabels: Uint8Array | null = null;    // All labels as one-hot encoded vectors
  private trainImages: Float32Array | null = null;    // First 55K images for training
  private testImages: Float32Array | null = null;     // Last 10K images for testing
  private trainLabels: Uint8Array | null = null;     // Training labels (one-hot encoded)
  private testLabels: Uint8Array | null = null;      // Test labels (one-hot encoded)

  // ========== Public Methods ==========
  
  /**
   * Loads the MNIST dataset from remote URLs.
   * This is the main entry point - call this before using getTrainData() or getTestData().
   * 
   * Process:
   * 1. Downloads and processes the image sprite (65,000 images in one PNG)
   * 2. Downloads the pre-encoded labels (binary format)
   * 3. Splits data into training and test sets
   * 
   * @throws {Error} If network requests fail
   */
  async load() {
    await this.loadImages();
    await this.loadLabels();
    this.splitDataset();
  }

  // ========== Private Methods ==========

  /**
   * Loads and processes the MNIST images from a sprite.
   * 
   * Sprite format:
   * - Single PNG image containing all 65,000 MNIST images
   * - Images are arranged vertically (one per row)
   * - Each row is 28 pixels tall, sprite width is 28 pixels
   * - Total sprite dimensions: 28 x 1,820,000 pixels
   * 
   * Processing:
   * - Loads sprite in chunks of 5,000 images to avoid memory spikes
   * - Extracts red channel (MNIST is grayscale, R=G=B)
   * - Normalizes pixel values from [0,255] to [0,1]
   */
  private async loadImages(): Promise<void> {
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    
    await new Promise<void>((resolve) => {
      img.crossOrigin = ''; // Enable CORS
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        // Allocate buffer for all images
        // Each pixel is stored as Float32 (4 bytes), total = 65000 * 784 * 4 bytes ≈ 200MB
        const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
        
        // Process sprite in chunks to avoid browser memory limits
        // Each chunk processes 5000 images = 140,000 pixels
        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        // Process sprite in chunks (13 chunks of 5000 images each)
        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer, 
            i * IMAGE_SIZE * chunkSize * 4,
            IMAGE_SIZE * chunkSize
          );
          
          // Draw chunk from sprite to canvas
          ctx.drawImage(
            img, 0, i * chunkSize, img.width, chunkSize,
            0, 0, img.width, chunkSize
          );

          // Extract pixel data from canvas
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          
          // Convert RGBA pixels to normalized grayscale
          for (let j = 0; j < imageData.data.length / 4; j++) {
            // imageData.data is Uint8ClampedArray in RGBA format
            // j * 4 = red channel (for grayscale images, R=G=B)
            // Normalize from [0,255] to [0,1] for neural network input
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        
        this.datasetImages = new Float32Array(datasetBytesBuffer);
        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });
  }

  /**
   * Loads the pre-encoded labels from the server.
   * 
   * Label format:
   * - Binary file containing one-hot encoded labels
   * - Each label is 10 bytes (one byte per class)
   * - Total size: 65,000 * 10 = 650,000 bytes
   * - Example: digit '3' = [0,0,0,1,0,0,0,0,0,0]
   */
  private async loadLabels(): Promise<void> {
    const labelsResponse = await fetch(MNIST_LABELS_PATH);
    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());
  }

  /**
   * Splits the dataset into training and test sets.
   * 
   * Split ratios:
   * - Training: 55,000 samples (84.6%)
   * - Test: 10,000 samples (15.4%)
   * 
   * Note: The standard MNIST split is 60K/10K, but this web version
   * uses 55K/10K for slightly faster loading.
   */
  private splitDataset(): void {
    if (!this.datasetImages || !this.datasetLabels) {
      throw new Error('Dataset not loaded');
    }

    // Split images
    // Each image is IMAGE_SIZE (784) floats
    this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    
    // Split labels
    // Each label is NUM_CLASSES (10) bytes for one-hot encoding
    this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }

  // ========== Data Access Methods ==========

  /**
   * Returns training data as TensorFlow.js tensors.
   * 
   * @param numExamples - Optional limit on number of examples to return.
   *                      Useful for quick testing or mini-batch training.
   * @returns Object containing:
   *   - xs: Tensor4D of shape [N, 28, 28, 1] - Images with pixel values in [0,1]
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
   * @param numExamples - Optional limit on number of examples to return.
   *                      Useful for quick evaluation during training.
   * @returns Object containing:
   *   - xs: Tensor4D of shape [N, 28, 28, 1] - Images with pixel values in [0,1]
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
   * Tensor shapes:
   * - Images: [batch_size, height, width, channels]
   *   - batch_size: Number of examples
   *   - height/width: 28x28 pixels
   *   - channels: 1 (grayscale)
   * - Labels: [batch_size, num_classes]
   *   - batch_size: Number of examples
   *   - num_classes: 10 (one-hot encoded)
   * 
   * @param images - Flattened image data (each image is 784 consecutive floats)
   * @param labels - One-hot encoded labels (each label is 10 consecutive bytes)
   * @param numExamples - Optional subset size
   * @returns Object with xs (images) and ys (labels) tensors
   */
  private getData(images: Float32Array, labels: Uint8Array, numExamples?: number) {
    let xs: tf.Tensor4D;
    let ys: tf.Tensor2D;

    if (numExamples != null) {
      // Return requested subset of data
      xs = tf.tensor4d(
        images.slice(0, numExamples * IMAGE_SIZE),
        [numExamples, IMAGE_HEIGHT, IMAGE_WIDTH, 1]  // 4D shape for Conv2D layers
      );
      ys = tf.tensor2d(
        labels.slice(0, numExamples * NUM_CLASSES),
        [numExamples, NUM_CLASSES]  // 2D shape for categorical crossentropy
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

// ========== Convenience Function ==========
/**
 * Loads MNIST dataset and returns raw arrays for easier integration.
 * 
 * This is a simplified interface that returns the raw TypedArrays instead
 * of TensorFlow.js tensors. Useful when you need direct array access or
 * want to perform custom preprocessing.
 * 
 * @returns Promise resolving to object containing:
 *   - trainImages: Float32Array of shape [55000 * 784] with values in [0,1]
 *   - trainLabels: Uint8Array of shape [55000 * 10] with one-hot encoding
 *   - testImages: Float32Array of shape [10000 * 784] with values in [0,1]
 *   - testLabels: Uint8Array of shape [10000 * 10] with one-hot encoding
 * 
 * @example
 * const {trainImages, trainLabels} = await loadMNIST();
 * // Get first image as 1D array of 784 pixels
 * const firstImage = trainImages.slice(0, 784);
 * // Get first label as one-hot vector
 * const firstLabel = trainLabels.slice(0, 10);
 */
export async function loadMNIST() {
  const data = new MnistData();
  await data.load();
  
  return {
    trainImages: data['trainImages'] as Float32Array,
    trainLabels: data['trainLabels'] as Uint8Array,
    testImages: data['testImages'] as Float32Array,
    testLabels: data['testLabels'] as Uint8Array
  };
}