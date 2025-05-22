import * as tf from '@tensorflow/tfjs';
import { DATASET_CONFIG, TRAINING_CONFIG, DATA_URLS, UI_CONFIG } from '../constants/mnistConfig.js';
import { Logger } from './logger.js';

/**
 * Utility class for loading and processing MNIST data
 */
export class MNISTDataLoader {
  constructor() {
    this.trainData = null;
    this.testData = null;
  }

  /**
   * Load and process MNIST dataset
   * @returns {Promise<{trainData: Object, testData: Object}>} Processed training and test data
   */
  async loadData() {
    const startTime = performance.now();
    
    try {
      Logger.info('Loading MNIST dataset...');
      
      const [imageData, labelData] = await Promise.all([
        this._loadImages(),
        this._loadLabels()
      ]);

      const { trainImages, testImages, trainLabels, testLabels } = this._splitData(imageData, labelData);

      this.trainData = {
        images: tf.tensor2d(trainImages, [TRAINING_CONFIG.NUM_TRAIN_ELEMENTS, DATASET_CONFIG.IMAGE_SIZE]),
        labels: tf.tensor2d(trainLabels, [TRAINING_CONFIG.NUM_TRAIN_ELEMENTS, DATASET_CONFIG.NUM_CLASSES])
      };

      this.testData = {
        images: tf.tensor2d(testImages, [TRAINING_CONFIG.NUM_TEST_ELEMENTS, DATASET_CONFIG.IMAGE_SIZE]),
        labels: tf.tensor2d(testLabels, [TRAINING_CONFIG.NUM_TEST_ELEMENTS, DATASET_CONFIG.NUM_CLASSES])
      };

      const loadTime = performance.now() - startTime;
      Logger.info('MNIST dataset loaded successfully');
      Logger.performance('Data loading', loadTime);
      Logger.debug('Dataset statistics:', {
        trainSamples: TRAINING_CONFIG.NUM_TRAIN_ELEMENTS,
        testSamples: TRAINING_CONFIG.NUM_TEST_ELEMENTS,
        imageSize: DATASET_CONFIG.IMAGE_SIZE,
        numClasses: DATASET_CONFIG.NUM_CLASSES
      });

      return {
        trainData: this.trainData,
        testData: this.testData
      };
    } catch (error) {
      Logger.error('Failed to load MNIST data:', error);
      throw new Error(`Failed to load MNIST data: ${error.message}`);
    }
  }

  /**
   * Load MNIST images from sprite sheet
   * @private
   * @returns {Promise<Float32Array>} Flattened image data
   */
  async _loadImages() {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d', { willReadFrequently: true });

      img.crossOrigin = '';
      img.onload = () => {
        try {
          const datasetBytesBuffer = new ArrayBuffer(
            DATASET_CONFIG.NUM_DATASET_ELEMENTS * DATASET_CONFIG.IMAGE_SIZE * 4
          );
          
          canvas.width = img.width;
          canvas.height = UI_CONFIG.CHUNK_SIZE;

          // Process images in chunks to avoid memory issues
          for (let i = 0; i < DATASET_CONFIG.NUM_DATASET_ELEMENTS / UI_CONFIG.CHUNK_SIZE; i++) {
            const datasetBytesView = new Float32Array(
              datasetBytesBuffer,
              i * DATASET_CONFIG.IMAGE_SIZE * UI_CONFIG.CHUNK_SIZE * 4,
              DATASET_CONFIG.IMAGE_SIZE * UI_CONFIG.CHUNK_SIZE
            );

            ctx.drawImage(
              img,
              0, i * UI_CONFIG.CHUNK_SIZE,
              img.width, UI_CONFIG.CHUNK_SIZE,
              0, 0,
              img.width, UI_CONFIG.CHUNK_SIZE
            );

            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

            // Convert to grayscale and normalize (0-1)
            for (let j = 0; j < imageData.data.length / 4; j++) {
              datasetBytesView[j] = imageData.data[j * 4] / 255;
            }
          }

          resolve(new Float32Array(datasetBytesBuffer));
        } catch (error) {
          reject(error);
        }
      };

      img.onerror = () => reject(new Error('Failed to load MNIST images'));
      img.src = DATA_URLS.IMAGES;
    });
  }

  /**
   * Load MNIST labels
   * @private
   * @returns {Promise<Float32Array>} One-hot encoded labels
   */
  async _loadLabels() {
    try {
      const response = await fetch(DATA_URLS.LABELS);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const labelsBuffer = await response.arrayBuffer();
      const labelsData = new Uint8Array(labelsBuffer);
      
      // Convert to Float32Array for consistency
      const labelsArray = new Float32Array(DATASET_CONFIG.NUM_DATASET_ELEMENTS * DATASET_CONFIG.NUM_CLASSES);
      for (let i = 0; i < DATASET_CONFIG.NUM_DATASET_ELEMENTS; i++) {
        for (let j = 0; j < DATASET_CONFIG.NUM_CLASSES; j++) {
          labelsArray[i * DATASET_CONFIG.NUM_CLASSES + j] = labelsData[i * DATASET_CONFIG.NUM_CLASSES + j];
        }
      }

      return labelsArray;
    } catch (error) {
      throw new Error(`Failed to load MNIST labels: ${error.message}`);
    }
  }

  /**
   * Split data into training and test sets
   * @private
   * @param {Float32Array} imageData - All image data
   * @param {Float32Array} labelData - All label data
   * @returns {Object} Split data object
   */
  _splitData(imageData, labelData) {
    const trainImagesEnd = TRAINING_CONFIG.NUM_TRAIN_ELEMENTS * DATASET_CONFIG.IMAGE_SIZE;
    const trainLabelsEnd = TRAINING_CONFIG.NUM_TRAIN_ELEMENTS * DATASET_CONFIG.NUM_CLASSES;

    return {
      trainImages: imageData.slice(0, trainImagesEnd),
      testImages: imageData.slice(trainImagesEnd),
      trainLabels: labelData.slice(0, trainLabelsEnd),
      testLabels: labelData.slice(trainLabelsEnd)
    };
  }

  /**
   * Get random test sample for visualization
   * @returns {Promise<{imageData: Float32Array, label: number, index: number}>}
   */
  async getRandomTestSample() {
    if (!this.testData) {
      throw new Error('Test data not loaded. Call loadData() first.');
    }

    const randomIndex = Math.floor(Math.random() * this.testData.images.shape[0]);
    
    // Get image data
    const imageSlice = this.testData.images.slice([randomIndex, 0], [1, DATASET_CONFIG.IMAGE_SIZE]);
    const imageData = await imageSlice.data();
    
    // Get label
    const labelSlice = this.testData.labels.slice([randomIndex, 0], [1, DATASET_CONFIG.NUM_CLASSES]);
    const labelData = await labelSlice.argMax(1).data();
    
    // Clean up tensors
    imageSlice.dispose();
    labelSlice.dispose();

    return {
      imageData,
      label: labelData[0],
      index: randomIndex
    };
  }

  /**
   * Clean up tensors to prevent memory leaks
   */
  dispose() {
    if (this.trainData) {
      this.trainData.images?.dispose();
      this.trainData.labels?.dispose();
    }
    if (this.testData) {
      this.testData.images?.dispose();
      this.testData.labels?.dispose();
    }
  }
}