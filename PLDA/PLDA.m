function [X_train_aug, X_test_aug] = PLDA(X_train, y_train, X_test)
    S = graph_construction(X_train, 10);
    [label_confidence, prototype] = label_propagation(X_train,y_train', S, 0.01);
    aug_feature = label_confidence * prototype;
    X_train_aug = [X_train, aug_feature];
    X_test_aug = test_data_aug_gen(X_train, label_confidence, prototype, X_test, 10);
end
