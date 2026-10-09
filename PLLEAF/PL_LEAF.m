function [label_recover, train_outputs, test_outputs] = PL_LEAF(X_train,y_train,X_test,k,ker,par,gama)
%Note that we replace the predictve model with ridge regression (see our paper for more details)
    label_recover = build_label_manifold(X_train,y_train,k);
    [train_outputs, test_outputs] = MulRegression(X_train, label_recover, X_test, gama, par, ker);
end