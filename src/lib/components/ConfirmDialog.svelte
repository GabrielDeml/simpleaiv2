<script lang="ts">
  export let title: string;
  export let message: string;
  export let confirmText: string = 'Confirm';
  export let cancelText: string = 'Cancel';
  export let type: 'danger' | 'warning' | 'info' = 'info';

  // Event callbacks (Svelte 5 style)
  export let onconfirm: (() => void) | undefined = undefined;
  export let oncancel: (() => void) | undefined = undefined;

  function handleConfirm() {
    onconfirm?.();
  }

  function handleCancel() {
    oncancel?.();
  }

  function handleBackdropClick(event: MouseEvent) {
    if (event.target === event.currentTarget) {
      handleCancel();
    }
  }
</script>

<div 
  class="modal-backdrop" 
  on:click={handleBackdropClick}
  on:keydown={(e) => e.key === 'Escape' && handleCancel()}
  role="dialog"
  aria-modal="true"
  aria-labelledby="dialog-title"
  tabindex="-1"
>
  <div class="confirm-dialog {type}">
    <div class="dialog-header">
      <h3 id="dialog-title" class="dialog-title">{title}</h3>
    </div>
    
    <div class="dialog-body">
      <p class="dialog-message">{message}</p>
    </div>
    
    <div class="dialog-actions">
      <button class="btn btn-secondary" on:click={handleCancel}>
        {cancelText}
      </button>
      <button class="btn btn-{type}" on:click={handleConfirm}>
        {confirmText}
      </button>
    </div>
  </div>
</div>

<style>
  .modal-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10000;
    animation: fadeIn 0.2s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  .confirm-dialog {
    background: #1a1a1a;
    border: 1px solid #333333;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    width: 90%;
    max-width: 400px;
    animation: slideIn 0.3s ease;
  }

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(-20px) scale(0.95);
    }
    to {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
  }

  .confirm-dialog.danger {
    border-color: #dc2626;
  }

  .confirm-dialog.warning {
    border-color: #f59e0b;
  }

  .confirm-dialog.info {
    border-color: #2563eb;
  }

  .dialog-header {
    padding: 24px 24px 0 24px;
  }

  .dialog-title {
    font-size: 18px;
    font-weight: 600;
    color: #ffffff;
    margin: 0;
  }

  .dialog-body {
    padding: 16px 24px 24px 24px;
  }

  .dialog-message {
    font-size: 14px;
    line-height: 1.5;
    color: #d4d4d4;
    margin: 0;
  }

  .dialog-actions {
    padding: 0 24px 24px 24px;
    display: flex;
    gap: 12px;
    justify-content: flex-end;
  }

  .btn {
    padding: 10px 20px;
    border-radius: 6px;
    border: none;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    min-width: 80px;
  }

  .btn-secondary {
    background: #262626;
    color: #d4d4d4;
    border: 1px solid #404040;
  }

  .btn-secondary:hover {
    background: #333333;
  }

  .btn-danger {
    background: #dc2626;
    color: white;
  }

  .btn-danger:hover {
    background: #b91c1c;
  }

  .btn-warning {
    background: #f59e0b;
    color: white;
  }

  .btn-warning:hover {
    background: #d97706;
  }

  .btn-info {
    background: #2563eb;
    color: white;
  }

  .btn-info:hover {
    background: #1d4ed8;
  }
</style>