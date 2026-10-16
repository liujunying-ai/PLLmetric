function accuracy = calc_acc_test(outputs, targets)
% CALC_ACC_TEST Calculates the accuracy on test set for standard classification.
%
%   accuracy = calc_acc_test(outputs, targets)
%
%   Inputs:
%       outputs : m x q, confidence scores.
%       targets : m x q, one-hot ground truth labels.
%
%   Output:
%       accuracy: Scalar, prediction accuracy.

    % Get predicted labels (index of max confidence)
    [~, pred_labels] = max(outputs, [], 2);

    % Get true labels (index of 1 in one-hot vector)
    [~, true_labels] = max(targets, [], 2);

    % Calculate accuracy: number of correct predictions / total samples
    num_correct = sum(pred_labels == true_labels);
    num_samples = size(pred_labels, 1);
    accuracy = num_correct / num_samples;
end
