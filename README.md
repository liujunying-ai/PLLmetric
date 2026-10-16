# PLLmetric
calc_acc_train.m and calc_acc_test.m are dedicated to calculating metrics on training and test data, respectively. Specifically, the former computes disACCd and disACCt, while the latter yields teACC and oraACC, based on the input confidences.

# Datasets
The real-world PLL datasets are downloaded from the <https://palm.seu.edu.cn/zhangml/Resources.htm#partial_data>. Please first use demo.m in the data folder to generate ten-fold cross-validation indices for the dataset and then run the corresponding ReadMe file.

# Source code
The source codes of existing PLL methods can be downloaded from the following links respectively.

| Method   | Original Source |
|----------|-----------------|
| PL-LEAF  | <https://palm.seu.edu.cn/zhangml/files/PL-LEAF.rar>        |
| PL-AGGD  | <https://palm.seu.edu.cn/zhangml/files/PL-AGGD.rar>        |
| SURE     | <https://lfeng1995.github.io/Code/SURE.zip>                |
| PRODEN   | <https://github.com/Lvcrezia77/PRODEN>                     |
| LW       | <https://github.com/hongwei-wen/LW-loss-for-partial-label> |
| DRAW     | <https://palm.seu.edu.cn/zhangml/files/DRAW.zip>           |
| PLDA     | <https://github.com/wwangwitsel/PLDA>                      |
