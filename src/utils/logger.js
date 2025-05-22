import { PERFORMANCE_CONFIG } from '../constants/mnistConfig.js';

/**
 * Simple logging utility with configurable levels
 */
export class Logger {
  static levels = {
    DEBUG: 0,
    INFO: 1,
    WARN: 2,
    ERROR: 3
  };

  static getCurrentLevel() {
    return Logger.levels[PERFORMANCE_CONFIG.LOG_LEVEL] || Logger.levels.INFO;
  }

  /**
   * Log debug messages (development only)
   * @param {string} message - Message to log
   * @param {...any} args - Additional arguments
   */
  static debug(message, ...args) {
    if (Logger.getCurrentLevel() <= Logger.levels.DEBUG) {
      console.log(`[DEBUG] ${message}`, ...args);
    }
  }

  /**
   * Log info messages
   * @param {string} message - Message to log
   * @param {...any} args - Additional arguments
   */
  static info(message, ...args) {
    if (Logger.getCurrentLevel() <= Logger.levels.INFO) {
      console.log(`[INFO] ${message}`, ...args);
    }
  }

  /**
   * Log warning messages
   * @param {string} message - Message to log
   * @param {...any} args - Additional arguments
   */
  static warn(message, ...args) {
    if (Logger.getCurrentLevel() <= Logger.levels.WARN) {
      console.warn(`[WARN] ${message}`, ...args);
    }
  }

  /**
   * Log error messages
   * @param {string} message - Message to log
   * @param {...any} args - Additional arguments
   */
  static error(message, ...args) {
    if (Logger.getCurrentLevel() <= Logger.levels.ERROR) {
      console.error(`[ERROR] ${message}`, ...args);
    }
  }

  /**
   * Log performance metrics
   * @param {string} operation - Operation name
   * @param {number} duration - Duration in milliseconds
   */
  static performance(operation, duration) {
    if (Logger.getCurrentLevel() <= Logger.levels.DEBUG) {
      console.log(`[PERF] ${operation}: ${duration}ms`);
    }
  }
}