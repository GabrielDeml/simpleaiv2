import type { LayerType, LayerConfig } from './types';

export interface LayerDefinition {
  type: LayerType;
  displayName: string;
  icon: string;
  color: string;
  defaultParams: Record<string, any>;
  description: string;
  learnMore?: string;
}

export const layerDefinitions: Record<LayerType, LayerDefinition> = {
  input: {
    type: 'input',
    displayName: 'Input Layer',
    icon: 'I',
    color: '#8b5cf6',
    defaultParams: {
      shape: [28, 28]
    },
    description: 'The entry point for data into your neural network. Defines the shape and size of the input data (e.g., 28x28 pixels for MNIST digits).',
    learnMore: 'Input layers don\'t perform any computation - they simply pass data to the next layer. The shape must match your dataset exactly.'
  },
  dense: {
    type: 'dense',
    displayName: 'Dense',
    icon: 'D',
    color: '#22c55e',
    defaultParams: {
      units: 128,
      activation: 'relu',
      useBias: true,
      kernelInitializer: 'glorotUniform'
    },
    description: 'A fully connected layer where every neuron connects to all neurons in the previous layer. The workhorse of neural networks.',
    learnMore: 'Dense layers learn patterns by adjusting weights and biases. More units = more learning capacity but also more computation.'
  },
  conv2d: {
    type: 'conv2d',
    displayName: 'Conv2D',
    icon: 'C',
    color: '#f59e0b',
    defaultParams: {
      filters: 32,
      kernelSize: 3,
      strides: 1,
      padding: 'same',
      activation: 'relu',
      useBias: true
    },
    description: 'Convolutional layers detect features in images like edges, shapes, and textures by sliding filters across the input.',
    learnMore: 'Conv layers are the foundation of computer vision. Each filter learns to detect specific patterns, building up from simple edges to complex objects.'
  },
  maxpooling2d: {
    type: 'maxpooling2d',
    displayName: 'MaxPooling2D',
    icon: 'MP',
    color: '#06b6d4',
    defaultParams: {
      poolSize: 2,
      strides: 2,
      padding: 'valid'
    },
    description: 'Reduces spatial dimensions by taking the maximum value in each pooling window. Makes the network more efficient and robust.',
    learnMore: 'Pooling layers reduce computation and help the network focus on the most important features while providing translation invariance.'
  },
  dropout: {
    type: 'dropout',
    displayName: 'Dropout',
    icon: 'Dr',
    color: '#ef4444',
    defaultParams: {
      rate: 0.2
    },
    description: 'Randomly deactivates neurons during training to prevent overfitting. Like training multiple networks at once!',
    learnMore: 'Dropout forces the network to learn redundant representations, making it more robust. A rate of 0.2 means 20% of neurons are dropped.'
  },
  flatten: {
    type: 'flatten',
    displayName: 'Flatten',
    icon: 'F',
    color: '#a855f7',
    defaultParams: {},
    description: 'Converts multi-dimensional data (like images) into a 1D array. Required before using Dense layers after Conv2D layers.',
    learnMore: 'Flatten preserves all the data but changes its shape. For example, a 28x28 image becomes a 784-element vector.'
  },
  output: {
    type: 'output',
    displayName: 'Output',
    icon: 'O',
    color: '#f59e0b',
    defaultParams: {
      units: 10,
      activation: 'softmax'
    },
    description: 'The final layer that produces predictions. Units should match the number of classes (e.g., 10 for digits 0-9).',
    learnMore: 'Softmax activation converts raw scores into probabilities that sum to 1, perfect for classification tasks.'
  },
  embedding: {
    type: 'embedding',
    displayName: 'Embedding',
    icon: 'E',
    color: '#9333ea',
    defaultParams: {
      vocabSize: 10000,
      embeddingDim: 128,
      maxLength: 512,
      trainable: true
    },
    description: 'Converts integer token IDs into dense vector representations. Essential for text and categorical data.',
    learnMore: 'Embeddings map discrete tokens to continuous vectors, allowing neural networks to process text. Similar tokens get similar vectors.'
  },
  multiHeadAttention: {
    type: 'multiHeadAttention',
    displayName: 'Multi-Head Attention',
    icon: 'MA',
    color: '#dc2626',
    defaultParams: {
      numHeads: 8,
      keyDim: 64,
      dropout: 0.1,
      useBias: true
    },
    description: 'The core of transformers - allows the model to attend to different positions simultaneously.',
    learnMore: 'Multiple attention heads learn different relationships. Each head can focus on different aspects like syntax, semantics, or long-range dependencies.'
  },
  layerNormalization: {
    type: 'layerNormalization',
    displayName: 'Layer Norm',
    icon: 'LN',
    color: '#0891b2',
    defaultParams: {
      epsilon: 1e-6,
      center: true,
      scale: true
    },
    description: 'Normalizes inputs across features. Critical for stable transformer training.',
    learnMore: 'Unlike batch norm, layer norm works well with variable sequence lengths and small batch sizes, making it ideal for transformers.'
  },
  positionalEncoding: {
    type: 'positionalEncoding',
    displayName: 'Positional Encoding',
    icon: 'PE',
    color: '#7c3aed',
    defaultParams: {
      maxLength: 512,
      encodingType: 'sinusoidal'
    },
    description: 'Adds position information to embeddings since transformers lack inherent sequence awareness.',
    learnMore: 'Sinusoidal encodings use sine and cosine functions at different frequencies to encode positions, allowing the model to generalize to longer sequences.'
  },
  transformerBlock: {
    type: 'transformerBlock',
    displayName: 'Transformer Block',
    icon: 'TB',
    color: '#ea580c',
    defaultParams: {
      numHeads: 8,
      keyDim: 64,
      ffDim: 256,
      dropout: 0.1
    },
    description: 'Complete transformer encoder layer with multi-head attention, normalization, and feed-forward network.',
    learnMore: 'A full transformer block combines attention (for capturing relationships) with position-wise feed-forward networks (for transforming representations).'
  },
  globalAveragePooling1D: {
    type: 'globalAveragePooling1D',
    displayName: 'Global Avg Pool 1D',
    icon: 'GP',
    color: '#059669',
    defaultParams: {},
    description: 'Averages across the sequence dimension to produce a fixed-size output. Ideal for converting variable-length sequences to fixed vectors.',
    learnMore: 'Global average pooling reduces each feature map to a single value by averaging. More interpretable than flatten and reduces parameters.'
  }
};