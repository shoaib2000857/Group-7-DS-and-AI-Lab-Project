"""Pixel confusion metrics; binary and semantic target spaces remain explicit."""
import numpy as np


def confusion_matrix(target, prediction, classes=3):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError('Nonempty target and prediction must have identical shapes')
    for value in (target, prediction):
        if not np.issubdtype(value.dtype, np.integer) or value.min() < 0 or value.max() >= classes:
            raise ValueError('Masks must contain integer class indices in the declared ontology')
    return np.bincount((target.astype('int64')*classes+prediction).ravel(), minlength=classes**2).reshape(classes, classes)


def metrics_from_confusion(matrix):
    cm = np.asarray(matrix, dtype='int64')
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.shape[0] < 2 or np.any(cm < 0):
        raise ValueError('Expected a nonnegative square confusion matrix')
    tp = np.diag(cm); union = cm.sum(0)+cm.sum(1)-tp
    iou = np.divide(tp, union, out=np.full(len(tp), np.nan), where=union != 0)
    btp, bfn, bfp = cm[1:, 1:].sum(), cm[1:, 0].sum(), cm[0, 1:].sum()
    def ratio(a, b): return float(a/b) if b else None
    return {'pixels': int(cm.sum()), 'class_iou': [None if np.isnan(x) else float(x) for x in iou],
            'macro_iou_present_classes': float(np.nanmean(iou)) if np.any(union) else None,
            'change_class_mean_iou': float(np.nanmean(iou[1:])) if np.any(union[1:]) else None,
            'binary_precision': ratio(btp, btp+bfp), 'binary_recall': ratio(btp, btp+bfn),
            'binary_f1': ratio(2*btp, 2*btp+bfp+bfn), 'binary_iou': ratio(btp, btp+bfp+bfn),
            'confusion_matrix': cm.tolist(), 'aggregation': 'pooled pixels; class 0 background'}
