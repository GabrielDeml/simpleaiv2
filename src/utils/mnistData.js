// MNIST data loader that handles CORS issues
import * as tf from '@tensorflow/tfjs';

export class MnistData {
  constructor() {
    this.trainImages = null;
    this.trainLabels = null;
    this.testImages = null;
    this.testLabels = null;
  }

  async load() {
    // Using the CORS-friendly mirror of MNIST data
    const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
    const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

    // Alternative URLs if the above don't work
    const ALT_IMAGES_PATH = 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/mnist-core/data/mnist_images.png';
    const ALT_LABELS_PATH = 'https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/mnist-core/data/mnist_labels_uint8';

    try {
      // Try loading from the primary source first
      const img = new Image();
      img.crossOrigin = '';
      
      const imageLoadPromise = new Promise((resolve, reject) => {
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('Failed to load image'));
      });
      
      img.src = MNIST_IMAGES_SPRITE_PATH;
      
      try {
        await imageLoadPromise;
      } catch {
        // Fallback to alternative source
        console.log('Primary source failed, trying alternative...');
        img.src = ALT_IMAGES_PATH;
        await new Promise((resolve, reject) => {
          img.onload = () => resolve(img);
          img.onerror = () => reject(new Error('Failed to load image from alternative source'));
        });
      }

      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const imgWidth = img.naturalWidth;
      const imgHeight = img.naturalHeight;
      
      console.log('Image loaded successfully:', img.complete, 'Dimensions:', imgWidth, 'x', imgHeight);
      
      if (imgWidth === 0 || imgHeight === 0) {
        throw new Error('Image failed to load properly - dimensions are 0');
      }

      const datasetBytesBuffer = new ArrayBuffer(65000 * 28 * 28 * 4);
      const datasetBytesView = new Float32Array(datasetBytesBuffer);

      canvas.width = imgWidth;
      canvas.height = imgHeight;
      ctx.drawImage(img, 0, 0);

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      console.log('Image dimensions:', imgWidth, 'x', imgHeight);
      console.log('Canvas dimensions:', canvas.width, 'x', canvas.height);
      
      // Debug: Check if image data is actually loaded
      let nonZeroPixels = 0;
      for (let i = 0; i < imageData.data.length; i += 4) {
        if (imageData.data[i] > 0) nonZeroPixels++;
      }
      console.log('Non-zero pixels in loaded image:', nonZeroPixels);
      
      // The sprite sheet is a single vertical column of flattened images
      // Each image is 28x28 pixels, stored as a single row of 784 pixels
      const IMAGE_SIZE = 784; // 28 * 28
      const NUM_IMAGES = 65000;
      
      // Process the image data in chunks to avoid memory issues
      const chunkSize = 5000;
      const numChunks = Math.ceil(NUM_IMAGES / chunkSize);
      
      console.log('Processing images in', numChunks, 'chunks of', chunkSize);
      
      for (let chunk = 0; chunk < numChunks; chunk++) {
        const startIdx = chunk * chunkSize;
        const endIdx = Math.min((chunk + 1) * chunkSize, NUM_IMAGES);
        
        // Create a temporary canvas for this chunk
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = imgWidth;
        tempCanvas.height = endIdx - startIdx;
        
        // Draw the chunk
        tempCtx.drawImage(
          img,
          0, startIdx, imgWidth, endIdx - startIdx,
          0, 0, imgWidth, endIdx - startIdx
        );
        
        const chunkData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        
        // Extract pixels from the chunk
        for (let i = 0; i < (endIdx - startIdx); i++) {
          for (let j = 0; j < IMAGE_SIZE; j++) {
            // Each row in the sprite contains one flattened 28x28 image
            const srcIdx = (i * imgWidth + j) * 4;
            const dstIdx = (startIdx + i) * IMAGE_SIZE + j;
            
            // Use red channel and normalize to 0-1
            datasetBytesView[dstIdx] = chunkData.data[srcIdx] / 255;
          }
        }
      }
      
      // Debug: Check first image
      console.log('First image sample values:', Array.from(datasetBytesView.slice(0, 10)));
      let firstImageNonZero = 0;
      for (let i = 0; i < 28 * 28; i++) {
        if (datasetBytesView[i] > 0) firstImageNonZero++;
      }
      console.log('Non-zero pixels in first image:', firstImageNonZero);

      // Load labels
      let labelsResponse;
      try {
        labelsResponse = await fetch(MNIST_LABELS_PATH);
        if (!labelsResponse.ok) throw new Error('Labels fetch failed');
      } catch {
        console.log('Primary labels failed, trying alternative...');
        labelsResponse = await fetch(ALT_LABELS_PATH);
      }
      
      const labelsArrayBuffer = await labelsResponse.arrayBuffer();
      let labelsUint8Array = new Uint8Array(labelsArrayBuffer);
      
      console.log('Labels loaded, buffer size:', labelsArrayBuffer.byteLength);
      
      // Check if labels might be one-hot encoded (650000 = 65000 * 10)
      if (labelsArrayBuffer.byteLength === 650000) {
        console.log('Labels appear to be one-hot encoded (65000 * 10 bytes)');
        // Convert one-hot to class indices
        const numLabels = 65000;
        const classLabels = new Uint8Array(numLabels);
        
        for (let i = 0; i < numLabels; i++) {
          // Find the index of the 1 in the one-hot vector
          let maxIdx = 0;
          let maxVal = labelsUint8Array[i * 10];
          for (let j = 1; j < 10; j++) {
            if (labelsUint8Array[i * 10 + j] > maxVal) {
              maxVal = labelsUint8Array[i * 10 + j];
              maxIdx = j;
            }
          }
          classLabels[i] = maxIdx;
        }
        
        labelsUint8Array = classLabels;
        console.log('Converted one-hot labels to class indices');
      }
      
      // Check if the labels file has an IDX header (common in MNIST files)
      // IDX files start with magic number 0x00000801 (2049 in decimal) for label files
      if (labelsArrayBuffer.byteLength > 65000) {
        console.log('Labels file appears to have header, checking...');
        const dataView = new DataView(labelsArrayBuffer);
        // Read as big-endian
        const magic = dataView.getUint32(0, false);
        console.log('Magic number:', magic, '(hex:', magic.toString(16), ')');
        
        if (magic === 2049) {
          // This is an IDX file with header
          // Header format: magic (4 bytes), num items (4 bytes), then labels
          const numItems = dataView.getUint32(4, false);
          console.log('Number of items in file:', numItems);
          // Skip 8-byte header
          labelsUint8Array = new Uint8Array(labelsArrayBuffer, 8);
        } else {
          console.log('No IDX header detected, using raw data');
        }
      }
      
      console.log('Using', labelsUint8Array.length, 'labels');
      
      // Debug: Check first few labels AFTER processing header
      console.log('First 20 label values after header processing:', Array.from(labelsUint8Array.slice(0, 20)));
      
      // Check label distribution
      const labelCounts = new Array(10).fill(0);
      for (let i = 0; i < Math.min(1000, labelsUint8Array.length); i++) {
        const label = labelsUint8Array[i];
        if (label >= 0 && label <= 9) {
          labelCounts[label]++;
        }
      }
      console.log('Label distribution (first 1000):', labelCounts);

      // Split into train and test sets
      const trainSize = 55000;
      const testSize = 10000;

      // Create tensors
      this.trainImages = tf.tensor4d(
        datasetBytesView.slice(0, trainSize * 28 * 28),
        [trainSize, 28, 28, 1]
      );
      
      this.trainLabels = tf.oneHot(
        tf.tensor1d(labelsUint8Array.slice(0, trainSize), 'int32'),
        10
      );

      // Debug: check label distribution for train and test
      console.log('\n=== Label distributions ===');
      const trainLabelDist = new Array(10).fill(0);
      const testLabelDist = new Array(10).fill(0);
      
      for (let i = 0; i < trainSize; i++) {
        if (labelsUint8Array[i] >= 0 && labelsUint8Array[i] <= 9) {
          trainLabelDist[labelsUint8Array[i]]++;
        }
      }
      
      for (let i = trainSize; i < trainSize + testSize; i++) {
        if (labelsUint8Array[i] >= 0 && labelsUint8Array[i] <= 9) {
          testLabelDist[labelsUint8Array[i]]++;
        }
      }
      
      console.log('Train label distribution:', trainLabelDist);
      console.log('Test label distribution:', testLabelDist);
      
      this.testImages = tf.tensor4d(
        datasetBytesView.slice(trainSize * 28 * 28, (trainSize + testSize) * 28 * 28),
        [testSize, 28, 28, 1]
      );
      
      this.testLabels = tf.oneHot(
        tf.tensor1d(labelsUint8Array.slice(trainSize, trainSize + testSize), 'int32'),
        10
      );

      console.log('MNIST data loaded successfully');
      console.log(`Train images shape: ${this.trainImages.shape}`);
      console.log(`Train labels shape: ${this.trainLabels.shape}`);
      console.log(`Test images shape: ${this.testImages.shape}`);
      console.log(`Test labels shape: ${this.testLabels.shape}`);
    } catch (error) {
      console.error('Error loading MNIST data:', error);
      throw error;
    }
  }

  getTrainData(numExamples) {
    const xs = this.trainImages.slice([0, 0, 0, 0], [numExamples, 28, 28, 1]);
    const ys = this.trainLabels.slice([0, 0], [numExamples, 10]);
    return { xs, ys };
  }

  getTestData(numExamples) {
    const xs = this.testImages.slice([0, 0, 0, 0], [numExamples, 28, 28, 1]);
    const ys = this.testLabels.slice([0, 0], [numExamples, 10]);
    return { xs, ys };
  }
}