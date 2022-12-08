from sklearn.metrics import precision_score, recall_score, f1_score


def calc_metrics(true, pred, label=None, index=0):
    return {
        'run': label,
        'precision_macro': round(precision_score(true, pred, average='macro'), 3),
        'recall_macro': round(recall_score(true, pred, average='macro'), 3),
        'f1_macro': round(f1_score(true, pred, average='macro'), 3),
        'precision_micro': round(precision_score(true, pred, average='micro'), 3),
        'recall_micro': round(recall_score(true, pred, average='micro'), 3),
        'f1_micro': round(f1_score(true, pred, average='micro'), 3),
        'index': index
    }
