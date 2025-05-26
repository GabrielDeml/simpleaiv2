export const parameterHelp = {
  // Input layer
  shape: {
    label: 'Input Shape',
    description: 'The dimensions of your input data. For images: [height, width] or [height, width, channels].',
    example: 'MNIST uses [28, 28] for 28x28 pixel grayscale images'
  },
  
  // Dense layer
  units: {
    label: 'Units (Neurons)',
    description: 'The number of neurons in this layer. More neurons = more learning capacity but slower training.',
    example: 'Try 128 for hidden layers, match your classes (e.g., 10) for output'
  },
  
  activation: {
    label: 'Activation Function',
    description: 'Adds non-linearity to let the network learn complex patterns.',
    options: {
      relu: 'ReLU: Most popular, fast, works well for hidden layers',
      sigmoid: 'Sigmoid: Outputs 0-1, good for binary classification',
      tanh: 'Tanh: Outputs -1 to 1, centered around zero',
      softmax: 'Softmax: For multi-class output (probabilities that sum to 1)',
      linear: 'Linear: No activation, rarely used except specific cases'
    }
  },
  
  useBias: {
    label: 'Use Bias',
    description: 'Adds a learnable offset to each neuron. Almost always keep this on.',
    tip: 'Bias helps the network fit data that doesn\'t pass through the origin'
  },
  
  kernelInitializer: {
    label: 'Weight Initialization',
    description: 'How to set initial weights. Good initialization helps training.',
    options: {
      glorotUniform: 'Glorot/Xavier: Default choice, works well for most cases',
      heUniform: 'He: Better for ReLU activations in deep networks',
      randomNormal: 'Random: Simple gaussian distribution'
    }
  },
  
  // Conv2D layer
  filters: {
    label: 'Number of Filters',
    description: 'Each filter learns to detect different features (edges, shapes, etc.).',
    example: 'Start with 32 or 64, increase in deeper layers'
  },
  
  kernelSize: {
    label: 'Filter Size',
    description: 'The size of the sliding window that scans the image.',
    example: '3 (3x3) is most common, 5 for larger features'
  },
  
  strides: {
    label: 'Stride',
    description: 'How many pixels the filter moves each step. Larger = smaller output.',
    example: '1 keeps size similar, 2 halves the dimensions'
  },
  
  padding: {
    label: 'Padding',
    description: 'How to handle edges of the input.',
    options: {
      valid: 'Valid: No padding, output shrinks',
      same: 'Same: Pad to keep output size same as input'
    }
  },
  
  // MaxPooling2D
  poolSize: {
    label: 'Pool Size',
    description: 'Size of the pooling window. Reduces spatial dimensions.',
    example: '2 (2x2) is standard, halves width and height'
  },
  
  // Dropout
  rate: {
    label: 'Dropout Rate',
    description: 'Fraction of neurons to randomly disable during training.',
    example: '0.2 (20%) for light regularization, 0.5 (50%) for heavy'
  },
  
  // Training parameters
  epochs: {
    label: 'Epochs',
    description: 'How many times to go through the entire dataset.',
    example: '10-50 for small datasets, adjust based on validation performance'
  },
  
  batchSize: {
    label: 'Batch Size',
    description: 'Number of samples processed before updating weights.',
    example: '32 or 64 for good balance of speed and stability'
  },
  
  learningRate: {
    label: 'Learning Rate',
    description: 'How big of steps to take when updating weights.',
    example: '0.001 is a safe default, reduce if training is unstable'
  },
  
  optimizer: {
    label: 'Optimizer',
    description: 'Algorithm for updating weights based on gradients.',
    options: {
      adam: 'Adam: Adaptive learning rate, most popular choice',
      sgd: 'SGD: Simple and reliable, good for fine-tuning',
      rmsprop: 'RMSprop: Good for RNNs and noisy gradients'
    }
  },
  
  loss: {
    label: 'Loss Function',
    description: 'Measures how wrong the predictions are.',
    options: {
      categoricalCrossentropy: 'For multi-class classification (one-hot encoded)',
      sparseCategoricalCrossentropy: 'For multi-class (integer labels)',
      binaryCrossentropy: 'For binary classification',
      meanSquaredError: 'For regression tasks'
    }
  },
  
  metrics: {
    label: 'Metrics',
    description: 'Additional measurements to track during training.',
    options: {
      accuracy: 'Percentage of correct predictions',
      loss: 'The loss value being minimized'
    }
  }
};