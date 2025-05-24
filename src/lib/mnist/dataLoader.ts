import * as tf from '@tensorflow/tfjs';

// MNIST dataset constants
const IMAGE_HEIGHT = 28; // Each MNIST image is 28 pixels tall
const IMAGE_WIDTH = 28; // Each MNIST image is 28 pixels wide
const IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH; // Total pixels per image (784)
const NUM_CLASSES = 10; // Digits 0-9
const NUM_DATASET_ELEMENTS = 65000; // Total number of images in the dataset
const NUM_TRAIN_ELEMENTS = 55000; // Number of images for training

// URLs for the MNIST dataset hosted by Google
// The images are stored as a single large sprite image for efficient loading
const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * Class for loading and managing MNIST dataset in the browser
 * This is adapted from the TensorFlow.js examples to work efficiently in browsers
 */
export class MnistData {
  // TypeScript private properties to store the dataset
  private datasetImages: Float32Array | null = null; // All images as normalized float values
  private datasetLabels: Uint8Array | null = null; // All labels as one-hot encoded arrays
  private trainImages: Float32Array | null = null; // Training subset of images
  private testImages: Float32Array | null = null; // Test subset of images
  private trainLabels: Uint8Array | null = null; // Training subset of labels
  private testLabels: Uint8Array | null = null; // Test subset of labels

  /**
   * Loads the MNIST dataset from remote URLs
   * This method downloads the sprite image and labels, then processes them
   */
  async load() {
    // Create an image element to load the sprite
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    
    // Load the sprite image containing all MNIST digits
    await new Promise<void>((resolve) => {
      img.crossOrigin = ''; // Enable CORS for cross-origin image loading
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        // Create buffer to store all image data
        // 4 bytes per pixel (Float32)
        const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
        const chunkSize = 5000; // Process images in chunks to avoid memory issues
        canvas.width = img.width;
        canvas.height = chunkSize;

        // Process the sprite image in chunks
        // The sprite contains all MNIST images arranged vertically
        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          // Create a view into the buffer for this chunk
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
            IMAGE_SIZE * chunkSize
          );
          
          // Draw a chunk of the sprite to the canvas
          ctx.drawImage(
            img, 0, i * chunkSize, img.width, chunkSize,
            0, 0, img.width, chunkSize
          );

          // Extract pixel data from the canvas
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          // Convert RGBA pixel data to grayscale float values (0-1)
          // We only need one channel since MNIST is grayscale
          for (let j = 0; j < imageData.data.length / 4; j++) {
            // Take red channel (index j * 4) and normalize to 0-1
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);
        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    // Load the labels (already in the correct format)
    const labelsResponse = await fetch(MNIST_LABELS_PATH);
    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    // Split the data into training and test sets
    // Training: first 55,000 images
    // Test: remaining 10,000 images
    this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }

  /**
   * Get training data as TensorFlow.js tensors
   * @param numExamples - Optional number of examples to return (defaults to all)
   * @returns Object with xs (images) and ys (labels) tensors
   */
  getTrainData(numExamples?: number) {
    if (!this.trainImages || !this.trainLabels) {
      throw new Error('Data not loaded. Call load() first.');
    }
    return this.getData(this.trainImages, this.trainLabels, numExamples);
  }

  /**
   * Get test data as TensorFlow.js tensors
   * @param numExamples - Optional number of examples to return (defaults to all)
   * @returns Object with xs (images) and ys (labels) tensors
   */
  getTestData(numExamples?: number) {
    if (!this.testImages || !this.testLabels) {
      throw new Error('Data not loaded. Call load() first.');
    }
    return this.getData(this.testImages, this.testLabels, numExamples);
  }

  /**
   * Convert raw data arrays to TensorFlow.js tensors
   * @param images - Float32Array of image pixel data
   * @param labels - Uint8Array of one-hot encoded labels
   * @param numExamples - Optional limit on number of examples
   * @returns Object with xs and ys tensors ready for training/evaluation
   */
  private getData(images: Float32Array, labels: Uint8Array, numExamples?: number) {
    let xs: tf.Tensor4D; // 4D tensor: [batch, height, width, channels]
    let ys: tf.Tensor2D; // 2D tensor: [batch, numClasses]

    if (numExamples != null) {
      // Return a subset of the data
      xs = tf.tensor4d(
        images.slice(0, numExamples * IMAGE_SIZE),
        [numExamples, IMAGE_HEIGHT, IMAGE_WIDTH, 1] // 1 channel for grayscale
      );
      ys = tf.tensor2d(
        labels.slice(0, numExamples * NUM_CLASSES),
        [numExamples, NUM_CLASSES]
      );
    } else {
      // Return all data
      xs = tf.tensor4d(
        images,
        [images.length / IMAGE_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1]
      );
      ys = tf.tensor2d(labels, [labels.length / NUM_CLASSES, NUM_CLASSES]);
    }

    return { xs, ys };
  }
}