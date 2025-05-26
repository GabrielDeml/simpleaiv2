import { writable } from 'svelte/store';

export interface ToastData {
  id: string;
  type: 'success' | 'error' | 'info';
  message: string;
  duration?: number;
  autoClose?: boolean;
}

export const toasts = writable<ToastData[]>([]);

export function addToast(toast: Omit<ToastData, 'id'>): string {
  const id = Math.random().toString(36).substr(2, 9);
  const newToast: ToastData = {
    id,
    duration: 4000,
    autoClose: true,
    ...toast
  };
  
  toasts.update(currentToasts => [...currentToasts, newToast]);
  return id;
}

export function removeToast(id: string) {
  toasts.update(currentToasts => currentToasts.filter(toast => toast.id !== id));
}

export function showSuccess(message: string, duration = 4000) {
  return addToast({ type: 'success', message, duration });
}

export function showError(message: string, duration = 6000) {
  return addToast({ type: 'error', message, duration });
}

export function showInfo(message: string, duration = 4000) {
  return addToast({ type: 'info', message, duration });
}