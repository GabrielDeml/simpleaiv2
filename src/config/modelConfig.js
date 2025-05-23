import * as tf from '@tensorflow/tfjs';

export const createMNISTModel = () => {
  const model = tf.sequential({
    layers: [
      tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 3,
        filters: 32,
        activation: 'relu'
      }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.conv2d({
        kernelSize: 3,
        filters: 64,
        activation: 'relu'
      }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.flatten(),
      tf.layers.dropout({ rate: 0.2 }),
      tf.layers.dense({
        units: 128,
        activation: 'relu'
      }),
      tf.layers.dense({
        units: 10,
        activation: 'softmax'
      })
    ]
  });

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
};

export const modelConfig = {
  trainingData: {
    numExamples: 5000,
    testNumExamples: 1000
  },
  training: {
    batchSize: 128,
    validationSplit: 0.2,
    shuffle: true
  }
};