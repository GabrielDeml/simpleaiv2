/**
 * Parameter validation system for neural network layers.
 * Provides comprehensive validation rules and error messages for all layer types.
 */

import type { LayerType } from './types';

/**
 * Validation rule for a single parameter.
 */
export interface ValidationRule {
  /** Check if the value is valid */
  validate: (value: any) => boolean;
  /** Error message when validation fails */
  message: string;
}

/**
 * Collection of validation rules for a parameter.
 */
export interface ParameterValidation {
  /** Whether the parameter is required */
  required?: boolean;
  /** List of validation rules to apply */
  rules: ValidationRule[];
}

/**
 * Validation schema for all parameters of a layer type.
 */
export type LayerValidationSchema = {
  [paramName: string]: ParameterValidation;
};

/**
 * Common validation rules that can be reused across different parameters.
 */
export const ValidationRules = {
  /** Value must be a positive integer */
  positiveInteger: (): ValidationRule => ({
    validate: (value) => Number.isInteger(value) && value > 0,
    message: 'Must be a positive integer'
  }),
  
  /** Value must be a positive number */
  positiveNumber: (): ValidationRule => ({
    validate: (value) => typeof value === 'number' && value > 0,
    message: 'Must be a positive number'
  }),
  
  /** Value must be between 0 and 1 (exclusive) */
  probability: (): ValidationRule => ({
    validate: (value) => typeof value === 'number' && value > 0 && value < 1,
    message: 'Must be between 0 and 1'
  }),
  
  /** Value must be between 0 and 1 (inclusive) */
  probabilityInclusive: (): ValidationRule => ({
    validate: (value) => typeof value === 'number' && value >= 0 && value <= 1,
    message: 'Must be between 0 and 1 (inclusive)'
  }),
  
  /** Value must be within a specific range */
  range: (min: number, max: number): ValidationRule => ({
    validate: (value) => typeof value === 'number' && value >= min && value <= max,
    message: `Must be between ${min} and ${max}`
  }),
  
  /** Value must be in a list of allowed values */
  oneOf: (allowed: any[]): ValidationRule => ({
    validate: (value) => allowed.includes(value),
    message: `Must be one of: ${allowed.join(', ')}`
  }),
  
  /** Array must have specific length */
  arrayLength: (length: number): ValidationRule => ({
    validate: (value) => Array.isArray(value) && value.length === length,
    message: `Must be an array with ${length} elements`
  }),
  
  /** Array must contain positive integers */
  arrayOfPositiveIntegers: (): ValidationRule => ({
    validate: (value) => 
      Array.isArray(value) && 
      value.every(v => Number.isInteger(v) && v > 0),
    message: 'Must be an array of positive integers'
  }),
  
  /** Value must be a boolean */
  boolean: (): ValidationRule => ({
    validate: (value) => typeof value === 'boolean',
    message: 'Must be true or false'
  }),
  
  /** Value must be a valid activation function */
  activation: (): ValidationRule => ({
    validate: (value) => [
      'relu', 'sigmoid', 'tanh', 'softmax', 'linear', 
      'elu', 'selu', 'softplus', 'softsign', 'swish', 'mish'
    ].includes(value),
    message: 'Must be a valid activation function'
  }),
  
  /** Value must be a valid kernel initializer */
  kernelInitializer: (): ValidationRule => ({
    validate: (value) => [
      'glorotUniform', 'glorotNormal', 'heUniform', 'heNormal',
      'leCunUniform', 'leCunNormal', 'zeros', 'ones',
      'randomUniform', 'randomNormal', 'truncatedNormal'
    ].includes(value),
    message: 'Must be a valid kernel initializer'
  }),
  
  /** Value must be a valid padding type */
  padding: (): ValidationRule => ({
    validate: (value) => ['valid', 'same'].includes(value),
    message: 'Must be "valid" or "same"'
  })
};

/**
 * Validation schemas for each layer type.
 */
export const layerValidationSchemas: Record<LayerType, LayerValidationSchema> = {
  input: {
    shape: {
      required: true,
      rules: [
        {
          validate: (value) => Array.isArray(value) && value.length >= 1 && value.length <= 3,
          message: 'Shape must be an array with 1-3 dimensions'
        },
        ValidationRules.arrayOfPositiveIntegers()
      ]
    }
  },
  
  dense: {
    units: {
      required: true,
      rules: [ValidationRules.positiveInteger()]
    },
    activation: {
      rules: [ValidationRules.activation()]
    },
    useBias: {
      rules: [ValidationRules.boolean()]
    },
    kernelInitializer: {
      rules: [ValidationRules.kernelInitializer()]
    }
  },
  
  conv2d: {
    filters: {
      required: true,
      rules: [ValidationRules.positiveInteger()]
    },
    kernelSize: {
      required: true,
      rules: [
        {
          validate: (value) => {
            if (typeof value === 'number') return value > 0 && Number.isInteger(value);
            if (Array.isArray(value)) {
              return value.length === 2 && 
                     value.every(v => Number.isInteger(v) && v > 0);
            }
            return false;
          },
          message: 'Must be a positive integer or array of 2 positive integers'
        }
      ]
    },
    strides: {
      rules: [
        {
          validate: (value) => {
            if (typeof value === 'number') return value > 0 && Number.isInteger(value);
            if (Array.isArray(value)) {
              return value.length === 2 && 
                     value.every(v => Number.isInteger(v) && v > 0);
            }
            return false;
          },
          message: 'Must be a positive integer or array of 2 positive integers'
        }
      ]
    },
    padding: {
      rules: [ValidationRules.padding()]
    },
    activation: {
      rules: [ValidationRules.activation()]
    },
    useBias: {
      rules: [ValidationRules.boolean()]
    }
  },
  
  maxpooling2d: {
    poolSize: {
      rules: [
        {
          validate: (value) => {
            if (typeof value === 'number') return value > 0 && Number.isInteger(value);
            if (Array.isArray(value)) {
              return value.length === 2 && 
                     value.every(v => Number.isInteger(v) && v > 0);
            }
            return false;
          },
          message: 'Must be a positive integer or array of 2 positive integers'
        }
      ]
    },
    strides: {
      rules: [
        {
          validate: (value) => {
            if (!value) return true; // Optional
            if (typeof value === 'number') return value > 0 && Number.isInteger(value);
            if (Array.isArray(value)) {
              return value.length === 2 && 
                     value.every(v => Number.isInteger(v) && v > 0);
            }
            return false;
          },
          message: 'Must be a positive integer or array of 2 positive integers'
        }
      ]
    },
    padding: {
      rules: [ValidationRules.padding()]
    }
  },
  
  dropout: {
    rate: {
      required: true,
      rules: [ValidationRules.probability()]
    }
  },
  
  flatten: {
    // No parameters to validate
  },
  
  output: {
    units: {
      required: true,
      rules: [ValidationRules.positiveInteger()]
    },
    activation: {
      rules: [ValidationRules.activation()]
    },
    useBias: {
      rules: [ValidationRules.boolean()]
    },
    kernelInitializer: {
      rules: [ValidationRules.kernelInitializer()]
    }
  },
  
  embedding: {
    vocabSize: {
      required: true,
      rules: [
        ValidationRules.positiveInteger(),
        ValidationRules.range(1, 1000000)
      ]
    },
    embeddingDim: {
      required: true,
      rules: [
        ValidationRules.positiveInteger(),
        ValidationRules.range(1, 1024)
      ]
    },
    maxLength: {
      required: true,
      rules: [
        ValidationRules.positiveInteger(),
        ValidationRules.range(1, 10000)
      ]
    },
    trainable: {
      rules: [ValidationRules.boolean()]
    }
  },
  
  multiHeadAttention: {
    numHeads: {
      required: true,
      rules: [
        ValidationRules.positiveInteger(),
        ValidationRules.range(1, 32)
      ]
    },
    keyDim: {
      required: true,
      rules: [
        ValidationRules.positiveInteger(),
        ValidationRules.range(1, 512)
      ]
    },
    valueDim: {
      rules: [
        ValidationRules.positiveInteger(),
        ValidationRules.range(1, 512)
      ]
    },
    dropout: {
      rules: [ValidationRules.probabilityInclusive()]
    },
    useBias: {
      rules: [ValidationRules.boolean()]
    }
  },
  
  layerNormalization: {
    epsilon: {
      rules: [
        ValidationRules.positiveNumber(),
        ValidationRules.range(1e-12, 1e-3)
      ]
    },
    center: {
      rules: [ValidationRules.boolean()]
    },
    scale: {
      rules: [ValidationRules.boolean()]
    }
  },
  
  positionalEncoding: {
    maxLength: {
      required: true,
      rules: [
        ValidationRules.positiveInteger(),
        ValidationRules.range(1, 10000)
      ]
    },
    encodingType: {
      required: true,
      rules: [ValidationRules.oneOf(['sinusoidal', 'learned'])]
    }
  },
  
  transformerBlock: {
    numHeads: {
      required: true,
      rules: [
        ValidationRules.positiveInteger(),
        ValidationRules.range(1, 32)
      ]
    },
    keyDim: {
      required: true,
      rules: [
        ValidationRules.positiveInteger(),
        ValidationRules.range(1, 512)
      ]
    },
    ffDim: {
      required: true,
      rules: [
        ValidationRules.positiveInteger(),
        ValidationRules.range(1, 4096)
      ]
    },
    dropout: {
      rules: [ValidationRules.probabilityInclusive()]
    }
  },
  
  globalAveragePooling1D: {
    // No parameters to validate
  }
};

/**
 * Validation result for a single parameter.
 */
export interface ParameterValidationResult {
  isValid: boolean;
  errors: string[];
}

/**
 * Validation result for all parameters of a layer.
 */
export interface LayerValidationResult {
  isValid: boolean;
  parameterErrors: Record<string, string[]>;
}

/**
 * Validates a single parameter value against its validation rules.
 */
export function validateParameter(
  value: any,
  validation: ParameterValidation
): ParameterValidationResult {
  const errors: string[] = [];
  
  // Check required
  if (validation.required && (value === undefined || value === null || value === '')) {
    errors.push('This field is required');
    return { isValid: false, errors };
  }
  
  // Skip validation if value is empty and not required
  if (!validation.required && (value === undefined || value === null || value === '')) {
    return { isValid: true, errors: [] };
  }
  
  // Apply validation rules
  for (const rule of validation.rules) {
    if (!rule.validate(value)) {
      errors.push(rule.message);
    }
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
}

/**
 * Validates all parameters of a layer.
 */
export function validateLayer(
  layerType: LayerType,
  parameters: Record<string, any>
): LayerValidationResult {
  const schema = layerValidationSchemas[layerType];
  const parameterErrors: Record<string, string[]> = {};
  let isValid = true;
  
  // Validate each parameter according to schema
  for (const [paramName, validation] of Object.entries(schema)) {
    const value = parameters[paramName];
    const result = validateParameter(value, validation);
    
    if (!result.isValid) {
      isValid = false;
      parameterErrors[paramName] = result.errors;
    }
  }
  
  return { isValid, parameterErrors };
}

/**
 * Gets a user-friendly description of parameter constraints.
 */
export function getParameterDescription(
  layerType: LayerType,
  parameterName: string
): string | null {
  const schema = layerValidationSchemas[layerType];
  const validation = schema[parameterName];
  
  if (!validation) return null;
  
  const descriptions: string[] = [];
  
  if (validation.required) {
    descriptions.push('Required');
  }
  
  // Extract constraints from rules
  for (const rule of validation.rules) {
    // Add specific descriptions based on rule messages
    if (rule.message.includes('positive integer')) {
      descriptions.push('Positive integer (1, 2, 3...)');
    } else if (rule.message.includes('between 0 and 1')) {
      descriptions.push('Decimal between 0 and 1');
    } else if (rule.message.includes('Must be one of')) {
      descriptions.push(rule.message);
    }
  }
  
  return descriptions.length > 0 ? descriptions.join('. ') : null;
}

/**
 * Sanitizes a parameter value to ensure it's the correct type.
 */
export function sanitizeParameterValue(
  layerType: LayerType,
  parameterName: string,
  value: any
): any {
  const schema = layerValidationSchemas[layerType];
  const validation = schema[parameterName];
  
  if (!validation) return value;
  
  // Try to convert string inputs to appropriate types
  if (typeof value === 'string') {
    // Check if it should be a number
    const hasNumberRule = validation.rules.some(rule => 
      rule.message.includes('number') || 
      rule.message.includes('integer') ||
      rule.message.includes('between')
    );
    
    if (hasNumberRule) {
      const parsed = parseFloat(value);
      if (!isNaN(parsed)) return parsed;
    }
    
    // Check if it should be a boolean
    const hasBooleanRule = validation.rules.some(rule => 
      rule.message.includes('true or false')
    );
    
    if (hasBooleanRule) {
      if (value === 'true') return true;
      if (value === 'false') return false;
    }
    
    // Check if it should be an array
    if (value.startsWith('[') && value.endsWith(']')) {
      try {
        const parsed = JSON.parse(value);
        if (Array.isArray(parsed)) return parsed;
      } catch (e) {
        // Invalid JSON, return original value
      }
    }
  }
  
  return value;
}