import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

def calculate_metrics(y_true, y_pred, mask=None):
    # Flatten and normalize to binary (0, 1)
    y_true_f = (y_true.flatten() > 0).astype(int)
    y_pred_f = (y_pred.flatten() > 0).astype(int)
    
    if mask is not None:
        mask_f = mask.flatten() > 0
        y_true_f = y_true_f[mask_f]
        y_pred_f = y_pred_f[mask_f]
    
    cm = confusion_matrix(y_true_f, y_pred_f)
    # Handle cases where some classes might be missing in small patches (unlikely here but good practice)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback for edge cases
        tn = fp = fn = tp = 0
        if 1 in y_true_f:
             tp = np.sum((y_true_f == 1) & (y_pred_f == 1))
             fn = np.sum((y_true_f == 1) & (y_pred_f == 0))
        if 0 in y_true_f:
             tn = np.sum((y_true_f == 0) & (y_pred_f == 0))
             fp = np.sum((y_true_f == 0) & (y_pred_f == 1))

    accuracy = accuracy_score(y_true_f, y_pred_f)
    sensitivity = recall_score(y_true_f, y_pred_f) # Sensitivity = TP / (TP + FN)
    
    # Specificity = TN / (TN + FP)
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0
        
    # Geometric Mean of Sensitivity and Specificity
    g_mean = np.sqrt(sensitivity * specificity)
        
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'g_mean': g_mean,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }
