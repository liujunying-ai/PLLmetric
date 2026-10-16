function accuracy = calc_acc_train(outputs, targets, candidates)
% CALC_ACC_TRAIN Calculates the accuracy on training set for partial label learning.
%
%   accuracy = calc_acc_train(outputs, targets, candidates)
%
%   Inputs:
%       outputs   : m x q, confidence scores.
%       targets   : m x q, one-hot ground truth labels.
%       candidates: m x q, binary mask (1=candidate, 0=non-candidate).
%
%   Output:
%       accuracy  : Scalar, prediction accuracy.

    % Mask non-candidate labels with -Inf to exclude them from max selection
    masked_outputs = outputs;
    masked_outputs(candidates == 0) = -Inf;

    % Get predicted labels (index of max confidence among candidates)
    [~, pred_labels] = max(masked_outputs, [], 2);

    % Get true labels (index of 1 in one-hot vector)
    [~, true_labels] = max(targets, [], 2);

    % Calculate accuracy: number of correct predictions / total samples
    num_correct = sum(pred_labels == true_labels);
    num_samples = size(pred_labels, 1);
    accuracy = num_correct / num_samples;
end
