import * as tf from '@tensorflow/tfjs';

// ========== Dataset Constants ==========
const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH; // 784 pixels per image
const NUM_CLASSES = 10; // Digits 0-9
const NUM_DATASET_ELEMENTS = 65000; // Total dataset size
const NUM_TRAIN_ELEMENTS = 55000; // Training set size
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS; // 10000

// ========== Dataset URLs ==========
// Google hosts the MNIST dataset as optimized web formats
const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * Class for loading and managing MNIST dataset in the browser.
 * Handles efficient loading of 65,000 handwritten digit images from a sprite.
 */
export class MnistData {
  // ========== Private Properties ==========
  private datasetImages: Float32Array | null = null;
  private datasetLabels: Uint8Array | null = null;
  private trainImages: Float32Array | null = null;
  private testImages: Float32Array | null = null;
  private trainLabels: Uint8Array | null = null;
  private testLabels: Uint8Array | null = null;

  // ========== Public Methods ==========
  
  /**
   * Loads the MNIST dataset from remote URLs.
   * Downloads a sprite containing all images and processes them into tensors.
   */
  async load() {
    await this.loadImages();
    await this.loadLabels();
    this.splitDataset();
  }

  // ========== Private Methods ==========

  /**
   * Loads and processes the MNIST images from a sprite.
   * The sprite is a single PNG containing all 65,000 images arranged vertically.
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

        // Allocate buffer for all images (4 bytes per pixel for Float32)
        const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
        const chunkSize = 5000; // Process in chunks to avoid memory spikes
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

          // Extract and normalize pixel data
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          for (let j = 0; j < imageData.data.length / 4; j++) {
            // Use red channel (MNIST is grayscale) and normalize to [0,1]
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
   */
  private async loadLabels(): Promise<void> {
    const labelsResponse = await fetch(MNIST_LABELS_PATH);
    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());
  }

  /**
   * Splits the dataset into training and test sets.
   * Training: 55,000 samples, Test: 10,000 samples
   */
  private splitDataset(): void {
    if (!this.datasetImages || !this.datasetLabels) {
      throw new Error('Dataset not loaded');
    }

    // Split images
    this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    
    // Split labels (one-hot encoded: 10 values per label)
    this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }

  // ========== Data Access Methods ==========

  /**
   * Returns training data as TensorFlow.js tensors.
   * @param numExamples - Limit number of examples (optional)
   * @returns {xs: Tensor4D, ys: Tensor2D} - Images and labels
   */
  getTrainData(numExamples?: number) {
    if (!this.trainImages || !this.trainLabels) {
      throw new Error('Data not loaded. Call load() first.');
    }
    return this.getData(this.trainImages, this.trainLabels, numExamples);
  }

  /**
   * Returns test data as TensorFlow.js tensors.
   * @param numExamples - Limit number of examples (optional)
   * @returns {xs: Tensor4D, ys: Tensor2D} - Images and labels
   */
  getTestData(numExamples?: number) {
    if (!this.testImages || !this.testLabels) {
      throw new Error('Data not loaded. Call load() first.');
    }
    return this.getData(this.testImages, this.testLabels, numExamples);
  }

  /**
   * Converts raw arrays to TensorFlow.js tensors with proper shape.
   * @param images - Flattened image data
   * @param labels - One-hot encoded labels
   * @param numExamples - Optional subset size
   * @returns Tensors ready for model training/evaluation
   */
  private getData(images: Float32Array, labels: Uint8Array, numExamples?: number) {
    let xs: tf.Tensor4D;
    let ys: tf.Tensor2D;

    if (numExamples != null) {
      // Return subset
      xs = tf.tensor4d(
        images.slice(0, numExamples * IMAGE_SIZE),
        [numExamples, IMAGE_HEIGHT, IMAGE_WIDTH, 1]
      );
      ys = tf.tensor2d(
        labels.slice(0, numExamples * NUM_CLASSES),
        [numExamples, NUM_CLASSES]
      );
    } else {
      // Return all data
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