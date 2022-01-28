def compare_to_groundtruth(ground_truth, predicted_tads, gap=200000):
    """
    Compare predicted TADs to ground truth TADs
    """
    gt_predicted = [False for i in range(len(ground_truth))]
    prediction_correct = [False for i in range(len(predicted_tads))]

    for i, gt_tad in enumerate(ground_truth):
        gt_start, gt_end = gt_tad
        for pred_start, pred_end in predicted_tads:
            if compare_sample_to_groundtruth(pred_tad=[pred_start, pred_end], gt_tad=[gt_start, gt_end], gap=gap):
                gt_predicted[i] = True
    
    for i, pred_tad in enumerate(predicted_tads):
        pred_start, pred_end = pred_tad
        for gt_start, gt_end in ground_truth:
            if compare_sample_to_groundtruth(pred_tad=[pred_start, pred_end], gt_tad=[gt_start, gt_end], gap=gap):
                prediction_correct[i] = True

    if len(prediction_correct) != 0:
        pred_correct_rate = sum(prediction_correct)/len(prediction_correct)
    else:
        pred_correct_rate = 0.0
    if len(gt_predicted) != 0:
        gt_predicted_rate = sum(gt_predicted)/len(gt_predicted)
    else:
        raise ValueError("No Ground Truth TADs")

    return gt_predicted, prediction_correct, gt_predicted_rate, pred_correct_rate


def compare_sample_to_groundtruth(pred_tad, gt_tad, gap=200000):
    pred_tad_start, pred_tad_end = pred_tad[0], pred_tad[1]
    gt_tad_start, gt_tad_end = gt_tad[0], gt_tad[1]

    if pred_tad_start >= gt_tad_start and pred_tad_end >= gt_tad_end: # Right shift
        if max(abs(pred_tad_start-gt_tad_start), abs(pred_tad_end-gt_tad_end)) <= gap:
            return True
    elif pred_tad_start <= gt_tad_start and pred_tad_end <= gt_tad_end: # Left shift
        if max(abs(pred_tad_start-gt_tad_start), abs(pred_tad_end-gt_tad_end)) <= gap:
            return True
    else: # Smaller / Bigger
        if abs(pred_tad_start-gt_tad_start) + abs(pred_tad_end-gt_tad_end) <= gap:
            return True
    return False
