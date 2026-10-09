function [accuracy] = CalAccuracy(test_outputs, test_target)
%% test_target: label by instance, test_outputs: instance by label
% [~, index1] = max(test_outputs, [], 2); % resulting a column vector containing the maximum value of each row
% [~, index2] = max(test_target, [], 2);

[~, index1] = max(test_outputs, [], 2);
[~, index2] = max(test_target, [], 2); % resulting a row vector containing the maximum value of each column.
accuracy = (sum(index1 == index2))/(size(test_outputs, 1));
end