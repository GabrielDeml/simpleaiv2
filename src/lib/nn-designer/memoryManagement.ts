/**
 * Memory management utilities for TensorFlow.js operations.
 * Helps prevent memory leaks by tracking tensor disposal and providing
 * utility functions for safe tensor operations.
 */

import * as tf from '@tensorflow/tfjs';

/**
 * Logs current memory usage for debugging.
 * Useful during development to track memory leaks.
 */
export function logMemoryUsage(label: string = 'Memory') {
  const memInfo = tf.memory();
  console.log(`${label}:`, {
    numTensors: memInfo.numTensors,
    numDataBuffers: memInfo.numDataBuffers,
    numBytes: `${(memInfo.numBytes / 1024 / 1024).toFixed(2)} MB`,
    unreliable: memInfo.unreliable
  });
}

/**
 * Executes a function and ensures all created tensors are disposed,
 * except those explicitly returned.
 * 
 * This is a wrapper around tf.tidy that provides better error handling
 * and memory tracking in development mode.
 */
export function tidyOperation<T extends tf.TensorContainer>(
  operation: () => T,
  operationName: string = 'Operation'
): T {
  if (process.env.NODE_ENV === 'development') {
    const beforeMemory = tf.memory();
    
    try {
      const result = tf.tidy(operation);
      
      const afterMemory = tf.memory();
      const tensorDiff = afterMemory.numTensors - beforeMemory.numTensors;
      
      if (tensorDiff > 0) {
        console.debug(`${operationName} created ${tensorDiff} tensor(s)`);
      }
      
      return result;
    } catch (error) {
      console.error(`Error in ${operationName}:`, error);
      throw error;
    }
  } else {
    return tf.tidy(operation);
  }
}

/**
 * Disposes a tensor or array of tensors safely.
 * Handles null/undefined values gracefully.
 */
export function disposeTensors(tensors: tf.Tensor | tf.Tensor[] | null | undefined): void {
  if (!tensors) return;
  
  if (Array.isArray(tensors)) {
    tensors.forEach(t => {
      if (t && !t.isDisposed) {
        t.dispose();
      }
    });
  } else if (!tensors.isDisposed) {
    tensors.dispose();
  }
}

/**
 * Executes an async operation that creates tensors and ensures cleanup.
 * Unlike tf.tidy, this works with async functions.
 */
export async function tidyAsync<T>(
  asyncOperation: () => Promise<T>,
  cleanup: () => void,
  operationName: string = 'Async Operation'
): Promise<T> {
  try {
    const result = await asyncOperation();
    return result;
  } catch (error) {
    console.error(`Error in ${operationName}:`, error);
    throw error;
  } finally {
    cleanup();
  }
}

/**
 * Memory usage monitor that can be used during training or other
 * long-running operations to track memory consumption.
 */
export class MemoryMonitor {
  private intervalId: NodeJS.Timeout | null = null;
  private baselineMemory: tf.MemoryInfo | null = null;
  
  /**
   * Start monitoring memory usage at regular intervals.
   */
  start(intervalMs: number = 5000) {
    this.baselineMemory = tf.memory();
    console.log('Memory monitoring started');
    logMemoryUsage('Baseline');
    
    this.intervalId = setInterval(() => {
      const currentMemory = tf.memory();
      const tensorDiff = currentMemory.numTensors - this.baselineMemory!.numTensors;
      const byteDiff = currentMemory.numBytes - this.baselineMemory!.numBytes;
      
      if (tensorDiff > 10 || byteDiff > 10 * 1024 * 1024) { // 10MB threshold
        console.warn(`Memory increase detected: +${tensorDiff} tensors, +${(byteDiff / 1024 / 1024).toFixed(2)} MB`);
        logMemoryUsage('Current');
      }
    }, intervalMs);
  }
  
  /**
   * Stop monitoring memory usage.
   */
  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
      console.log('Memory monitoring stopped');
      logMemoryUsage('Final');
    }
  }
}

/**
 * Decorator for methods that create tensors.
 * Automatically wraps the method in tf.tidy() for memory safety.
 * 
 * Usage:
 * ```typescript
 * class MyModel {
 *   @TidyMethod
 *   processData(input: tf.Tensor): tf.Tensor {
 *     // All intermediate tensors created here will be disposed
 *     return tf.relu(tf.matMul(input, weights));
 *   }
 * }
 * ```
 */
export function TidyMethod(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  
  descriptor.value = function(...args: any[]) {
    return tidyOperation(
      () => originalMethod.apply(this, args),
      `${target.constructor.name}.${propertyKey}`
    );
  };
  
  return descriptor;
}

/**
 * Best practices for memory management in TensorFlow.js:
 * 
 * 1. Always dispose tensors when done with them
 * 2. Use tf.tidy() to automatically clean up intermediate tensors
 * 3. For async operations, use try/finally blocks to ensure cleanup
 * 4. Monitor memory usage during development with tf.memory()
 * 5. Be especially careful with tensors created in loops
 * 6. Use tf.keep() to prevent disposal of tensors you need to return from tidy
 * 7. Consider using tf.dispose() on model instances when switching models
 * 
 * Common sources of memory leaks:
 * - Forgetting to dispose evaluation results
 * - Not cleaning up intermediate tensors in data preprocessing
 * - Creating tensors in event handlers without disposal
 * - Keeping references to old models or predictions
 */