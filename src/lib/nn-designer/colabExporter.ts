import type { LayerConfig, TrainingConfig, DatasetType } from './types';

/**
 * ColabExporter - Generates Jupyter notebooks for Google Colab training
 * 
 * Converts visual neural network designs into executable Python notebooks
 * that can be opened and run directly in Google Colab. Handles dataset loading,
 * model architecture translation, and training configuration.
 */
export class ColabExporter {
  /**
   * Generates a complete Jupyter notebook for Google Colab training.
   * 
   * @param layers - Array of layer configurations from the visual designer
   * @param trainingConfig - Training hyperparameters and settings
   * @param selectedDataset - Dataset type (mnist, cifar10, fashion-mnist)
   * @returns Jupyter notebook JSON structure ready for download
   */
  generateNotebook(
    layers: LayerConfig[],
    trainingConfig: TrainingConfig,
    selectedDataset: DatasetType
  ): any {
    const cells = [
      this.createSetupCell(),
      this.createDatasetCell(selectedDataset),
      this.createModelCell(layers),
      this.createTrainingCell(trainingConfig),
      this.createEvaluationCell()
    ];

    return {
      cells,
      metadata: {
        kernelspec: {
          display_name: "Python 3",
          language: "python",
          name: "python3"
        },
        language_info: {
          name: "python",
          version: "3.7.0"
        },
        colab: {
          provenance: []
        }
      },
      nbformat: 4,
      nbformat_minor: 0
    };
  }

  /**
   * Creates the setup cell with imports and environment configuration.
   */
  private createSetupCell(): any {
    const code = `# Neural Network Training - Exported from Visual Designer
# This notebook was automatically generated and contains the complete training pipeline

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Check for GPU availability
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Custom layer implementations for transformer architecture
class PositionalEncoding(layers.Layer):
    """Adds positional encoding to embedding vectors."""
    
    def __init__(self, max_length=5000, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        
    def build(self, input_shape):
        # Create positional encoding matrix
        position = np.arange(self.max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, input_shape[-1], 2) * -(np.log(10000.0) / input_shape[-1]))
        
        pe = np.zeros((self.max_length, input_shape[-1]))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
        super().build(input_shape)
        
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({"max_length": self.max_length})
        return config

class TransformerBlock(layers.Layer):
    """Transformer encoder block with multi-head attention and feed-forward network."""
    
    def __init__(self, num_heads, key_dim, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.key_dim
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation="relu"),
            layers.Dense(input_shape[-1])
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        # Multi-head attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate
        })
        return config

print("Environment setup complete with custom layers!")`;

    return {
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: code.split('\n').map(line => line + '\n')
    };
  }

  /**
   * Creates the dataset loading cell based on the selected dataset.
   */
  private createDatasetCell(dataset: DatasetType): any {
    let code = '';
    let description = '';

    switch (dataset) {
      case 'mnist':
        description = "# MNIST Dataset Loading\n# Handwritten digits (0-9), 28x28 grayscale images";
        code = `# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Data preprocessing
x_train = x_train.astype('float32') / 255.0  # Normalize to [0,1]
x_test = x_test.astype('float32') / 255.0

# Add channel dimension for CNN layers (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']`;
        break;
        
      case 'cifar10':
        description = "# CIFAR-10 Dataset Loading\n# Color images of objects, 32x32 RGB images";
        code = `# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Data preprocessing
x_train = x_train.astype('float32') / 255.0  # Normalize to [0,1]
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']`;
        break;
        
      case 'fashion-mnist':
        description = "# Fashion-MNIST Dataset Loading\n# Fashion items, 28x28 grayscale images";
        code = `# Load Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Data preprocessing
x_train = x_train.astype('float32') / 255.0  # Normalize to [0,1]
x_test = x_test.astype('float32') / 255.0

# Add channel dimension for CNN layers (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']`;
        break;
        
      case 'imdb':
        description = "# IMDB Dataset Loading\n# Movie reviews for sentiment analysis";
        code = `# Load IMDB dataset
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load dataset with vocabulary size limit
vocab_size = 10000
max_length = 256

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to same length
x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

# Convert labels to one-hot encoding for binary classification
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

class_names = ['Negative', 'Positive']`;
        break;
        
      case 'ag-news':
        description = "# AG News Dataset Loading\n# News article classification";
        code = `# Load AG News dataset
# Note: AG News is not built into Keras, so we'll use TensorFlow Datasets
import tensorflow_datasets as tfds

# Load the dataset
ds_train, ds_test = tfds.load('ag_news_subset', 
                               split=['train', 'test'],
                               as_supervised=True)

# Prepare tokenizer and preprocess text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
max_length = 120

# Extract texts and labels
train_texts = []
train_labels = []
for text, label in ds_train:
    train_texts.append(text.numpy().decode('utf-8'))
    train_labels.append(label.numpy())

test_texts = []
test_labels = []
for text, label in ds_test:
    test_texts.append(text.numpy().decode('utf-8'))
    test_labels.append(label.numpy())

# Tokenize and pad
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train_texts)

x_train = tokenizer.texts_to_sequences(train_texts)
x_test = tokenizer.texts_to_sequences(test_texts)

x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(train_labels, 4)
y_test = keras.utils.to_categorical(test_labels, 4)

class_names = ['World', 'Sports', 'Business', 'Sci/Tech']`;
        break;
        
      default:
        description = "# Custom Dataset Loading";
        code = "# Custom dataset loading code would go here";
    }

    const fullCode = `${description}

${code}

# Dataset information
print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"Number of classes: {len(class_names)}")

# Visualize sample images
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    if len(x_train.shape) == 4 and x_train.shape[-1] == 1:
        # Grayscale images
        plt.imshow(x_train[i].squeeze(), cmap='gray')
    else:
        # Color images
        plt.imshow(x_train[i])
    plt.title(f'Class: {class_names[np.argmax(y_train[i])]}')
    plt.axis('off')
plt.tight_layout()
plt.show()`;

    return {
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: fullCode.split('\n').map(line => line + '\n')
    };
  }

  /**
   * Creates the model definition cell by converting visual layers to Keras code.
   */
  private createModelCell(layers: LayerConfig[]): any {
    // Find input shape from input layer
    const inputLayer = layers.find(l => l.type === 'input');
    const inputShape = inputLayer ? inputLayer.params.shape : [28, 28, 1];
    
    let modelCode = "# Model Architecture - Converted from Visual Designer\n";
    modelCode += `# Input shape: ${JSON.stringify(inputShape)}\n\n`;
    modelCode += "model = keras.Sequential([\n";
    
    // Add explicit Input layer first
    const inputShapeStr = inputShape.length === 2 ? `[${inputShape.join(', ')}, 1]` : `[${inputShape.join(', ')}]`;
    modelCode += `    keras.Input(shape=${inputShapeStr}),\n`;
    
    // Convert each layer to Keras equivalent
    for (const layer of layers) {
      if (layer.type === 'input') {
        continue; // Skip input layer, it's handled by keras.Input above
      }
      
      const kerasLayer = this.convertLayerToKeras(layer, null); // No input_shape needed now
      if (kerasLayer) {
        modelCode += `    ${kerasLayer},\n`;
      }
    }
    
    modelCode += "])\n\n";
    
    // Add model summary
    modelCode += `# Model summary
model.summary()

# Visualize model architecture
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)`;

    return {
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: modelCode.split('\n').map(line => line + '\n')
    };
  }

  /**
   * Converts a visual layer configuration to Keras layer code.
   */
  private convertLayerToKeras(layer: LayerConfig, _inputShape: number[] | null): string | null {
    // Helper function to convert JavaScript boolean to Python boolean
    const toPythonBool = (value: boolean): string => value ? 'True' : 'False';
    
    switch (layer.type) {
      case 'dense':
        return `layers.Dense(${layer.params.units}, activation='${layer.params.activation}', use_bias=${toPythonBool(layer.params.useBias)})`;
        
      case 'conv2d':
        const kernelSize = Array.isArray(layer.params.kernelSize) 
          ? `(${layer.params.kernelSize.join(', ')})` 
          : layer.params.kernelSize;
        const strides = Array.isArray(layer.params.strides)
          ? `(${layer.params.strides.join(', ')})`
          : layer.params.strides;
        return `layers.Conv2D(${layer.params.filters}, ${kernelSize}, strides=${strides}, padding='${layer.params.padding}', activation='${layer.params.activation}', use_bias=${toPythonBool(layer.params.useBias)})`;
        
      case 'maxpooling2d':
        const poolSize = Array.isArray(layer.params.poolSize)
          ? `(${layer.params.poolSize.join(', ')})`
          : layer.params.poolSize;
        const poolStrides = Array.isArray(layer.params.strides)
          ? `(${layer.params.strides.join(', ')})`
          : layer.params.strides;
        return `layers.MaxPooling2D(pool_size=${poolSize}, strides=${poolStrides}, padding='${layer.params.padding}')`;
        
      case 'dropout':
        return `layers.Dropout(${layer.params.rate})`;
        
      case 'flatten':
        return `layers.Flatten()`;
        
      case 'output':
        return `layers.Dense(${layer.params.units}, activation='${layer.params.activation}')`;
        
      case 'embedding':
        return `layers.Embedding(input_dim=${layer.params.vocabSize}, output_dim=${layer.params.embeddingDim}, input_length=${layer.params.maxLength}, trainable=${toPythonBool(layer.params.trainable !== false)})`;
        
      case 'multiHeadAttention':
        return `layers.MultiHeadAttention(num_heads=${layer.params.numHeads}, key_dim=${layer.params.keyDim}, value_dim=${layer.params.valueDim || layer.params.keyDim}, dropout=${layer.params.dropout || 0.0}, use_bias=${toPythonBool(layer.params.useBias !== false)})`;
        
      case 'layerNormalization':
        return `layers.LayerNormalization(epsilon=${layer.params.epsilon || 1e-6}, center=${toPythonBool(layer.params.center !== false)}, scale=${toPythonBool(layer.params.scale !== false)})`;
        
      case 'positionalEncoding':
        // Use the custom PositionalEncoding layer defined in setup cell
        return `PositionalEncoding(max_length=${layer.params.maxLength})`;
        
      case 'transformerBlock':
        // Use the custom TransformerBlock layer defined in setup cell
        return `TransformerBlock(num_heads=${layer.params.numHeads}, key_dim=${layer.params.keyDim}, ff_dim=${layer.params.ffDim}, dropout=${layer.params.dropout || 0.1})`;
        
      case 'globalAveragePooling1D':
        return `layers.GlobalAveragePooling1D()`;
        
      default:
        return null;
    }
  }

  /**
   * Creates the training configuration and execution cell.
   */
  private createTrainingCell(config: TrainingConfig): any {
    // Convert TensorFlow.js loss names to Keras loss names
    const lossMapping: Record<string, string> = {
      'categoricalCrossentropy': 'categorical_crossentropy',
      'meanSquaredError': 'mean_squared_error',
      'binaryCrossentropy': 'binary_crossentropy'
    };
    
    const kerasLoss = lossMapping[config.loss] || config.loss;
    
    const code = `# Training Configuration - From Visual Designer
# Optimizer: ${config.optimizer}
# Learning Rate: ${config.learningRate}
# Loss Function: ${kerasLoss}
# Batch Size: ${config.batchSize}
# Epochs: ${config.epochs}
# Validation Split: ${config.validationSplit}

# Compile the model
model.compile(
    optimizer='${config.optimizer}',
    loss='${kerasLoss}',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=${config.batchSize},
    epochs=${config.epochs},
    validation_split=${config.validationSplit},
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()`;

    return {
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: code.split('\n').map(line => line + '\n')
    };
  }

  /**
   * Creates the evaluation cell with test metrics and visualizations.
   */
  private createEvaluationCell(): any {
    const code = `# Model Evaluation

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification report
print("\\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Display sample predictions
plt.figure(figsize=(15, 10))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    if len(x_test.shape) == 4 and x_test.shape[-1] == 1:
        # Grayscale images
        plt.imshow(x_test[i].squeeze(), cmap='gray')
    else:
        # Color images
        plt.imshow(x_test[i])
    
    true_label = class_names[y_true_classes[i]]
    pred_label = class_names[y_pred_classes[i]]
    confidence = np.max(y_pred[i]) * 100
    
    color = 'green' if y_true_classes[i] == y_pred_classes[i] else 'red'
    plt.title(f'True: {true_label}\\nPred: {pred_label} ({confidence:.1f}%)', 
              color=color, fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()

print("\\nModel evaluation complete!")`;

    return {
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: code.split('\n').map(line => line + '\n')
    };
  }

  /**
   * Generates a downloadable blob URL for the notebook.
   */
  generateDownloadLink(
    layers: LayerConfig[],
    trainingConfig: TrainingConfig,
    selectedDataset: DatasetType
  ): string {
    const notebook = this.generateNotebook(layers, trainingConfig, selectedDataset);
    const blob = new Blob([JSON.stringify(notebook, null, 2)], {
      type: 'application/json'
    });
    return URL.createObjectURL(blob);
  }

  /**
   * Generates a filename for the exported notebook.
   */
  generateFilename(selectedDataset: DatasetType): string {
    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
    return `neural-network-${selectedDataset}-${timestamp}.ipynb`;
  }
}

/**
 * Singleton instance of ColabExporter for use throughout the application.
 */
export const colabExporter = new ColabExporter();