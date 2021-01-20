``` r
library(tensorflow)
library(keras)
library(stringr)
library(readr)
library(purrr)
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## 
    ## Attaching package: 'caret'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

    ## The following object is masked from 'package:tensorflow':
    ## 
    ##     train

``` r
library(e1071)

cifar <- dataset_cifar10()
```

``` r
############################################################################

train_data <- scale(cifar$train$x)
dim(train_data) <- c(50000,32,32,3)

test_data <- scale(cifar$test$x)
dim(test_data) <- c(10000,32,32,3)

train_label <- as.numeric(cifar$train$y)
dim(train_label) <- c(50000)

test_label <- as.numeric(cifar$test$y)
dim(test_label) <- c(10000)
```

``` r
#######################################################################

class_names <- c('airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')

index <- 1:30

par(mfcol = c(5,6), mar = rep(1, 4), oma = rep(0.2, 4))

cifar$train$x[index,,,] %>% 
  purrr::array_tree(1) %>%
  purrr::set_names(class_names[cifar$train$y[index] + 1]) %>% 
  purrr::map(as.raster, max = 255) %>%
  purrr::iwalk(~{plot(.x); title(.y)})
```

![](Intro-to-CNN-in-R_files/figure-gfm/Data%20visualization-1.png)<!-- -->

``` r
############################################################################

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", input_shape = c(32,32,3)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) 

summary(model)
```

    ## Model: "sequential"
    ## ________________________________________________________________________________
    ## Layer (type)                        Output Shape                    Param #     
    ## ================================================================================
    ## conv2d_2 (Conv2D)                   (None, 30, 30, 32)              896         
    ## ________________________________________________________________________________
    ## conv2d_1 (Conv2D)                   (None, 28, 28, 64)              18496       
    ## ________________________________________________________________________________
    ## max_pooling2d_1 (MaxPooling2D)      (None, 14, 14, 64)              0           
    ## ________________________________________________________________________________
    ## conv2d (Conv2D)                     (None, 12, 12, 64)              36928       
    ## ________________________________________________________________________________
    ## max_pooling2d (MaxPooling2D)        (None, 6, 6, 64)                0           
    ## ================================================================================
    ## Total params: 56,320
    ## Trainable params: 56,320
    ## Non-trainable params: 0
    ## ________________________________________________________________________________

``` r
model %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

summary(model)
```

    ## Model: "sequential"
    ## ________________________________________________________________________________
    ## Layer (type)                        Output Shape                    Param #     
    ## ================================================================================
    ## conv2d_2 (Conv2D)                   (None, 30, 30, 32)              896         
    ## ________________________________________________________________________________
    ## conv2d_1 (Conv2D)                   (None, 28, 28, 64)              18496       
    ## ________________________________________________________________________________
    ## max_pooling2d_1 (MaxPooling2D)      (None, 14, 14, 64)              0           
    ## ________________________________________________________________________________
    ## conv2d (Conv2D)                     (None, 12, 12, 64)              36928       
    ## ________________________________________________________________________________
    ## max_pooling2d (MaxPooling2D)        (None, 6, 6, 64)                0           
    ## ________________________________________________________________________________
    ## flatten (Flatten)                   (None, 2304)                    0           
    ## ________________________________________________________________________________
    ## dense_3 (Dense)                     (None, 256)                     590080      
    ## ________________________________________________________________________________
    ## dense_2 (Dense)                     (None, 128)                     32896       
    ## ________________________________________________________________________________
    ## dense_1 (Dense)                     (None, 64)                      8256        
    ## ________________________________________________________________________________
    ## dense (Dense)                       (None, 10)                      650         
    ## ================================================================================
    ## Total params: 688,202
    ## Trainable params: 688,202
    ## Non-trainable params: 0
    ## ________________________________________________________________________________

``` r
model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

history <- model %>% 
  fit(
    x = train_data, y = train_label,
    epochs = 1,
    validation_split=0.2,
    use_multiprocessing=TRUE
  )

plot(history)
```

![](Intro-to-CNN-in-R_files/figure-gfm/Model%20Optimization%20and%20Training-1.png)<!-- -->

``` r
############################################################################
Prediction_train_data <- predict_classes(model, train_data)
confusionMatrix(table(Prediction_train_data,train_label))
```

    ## Confusion Matrix and Statistics
    ## 
    ##                      train_label
    ## Prediction_train_data    0    1    2    3    4    5    6    7    8    9
    ##                     0 2917   51  182   26   78    6    5   13  209   80
    ##                     1  130 3881   33   10   19    6   31    6  187  376
    ##                     2  511   41 2936  350  606  326  416  213   84   25
    ##                     3  243  120  664 3412  511 1775  773  331  183  173
    ##                     4  119   12  392  174 2763  168  287  186   23   12
    ##                     5   41   20  289  554  197 2331   85  361   19   31
    ##                     6   35   69  225  178  151   50 3289    9   19   29
    ##                     7  185   52  167  184  600  292   53 3814   33  169
    ##                     8  533  127   66   38   31   11   21   14 3919   97
    ##                     9  286  627   46   74   44   35   40   53  324 4008
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.6654          
    ##                  95% CI : (0.6612, 0.6695)
    ##     No Information Rate : 0.1             
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6282          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
    ## Sensitivity           0.58340  0.77620  0.58720  0.68240  0.55260  0.46620
    ## Specificity           0.98556  0.98227  0.94284  0.89393  0.96949  0.96451
    ## Pos Pred Value        0.81777  0.82945  0.53304  0.41686  0.66804  0.59343
    ## Neg Pred Value        0.95514  0.97531  0.95361  0.96202  0.95123  0.94207
    ## Prevalence            0.10000  0.10000  0.10000  0.10000  0.10000  0.10000
    ## Detection Rate        0.05834  0.07762  0.05872  0.06824  0.05526  0.04662
    ## Detection Prevalence  0.07134  0.09358  0.11016  0.16370  0.08272  0.07856
    ## Balanced Accuracy     0.78448  0.87923  0.76502  0.78817  0.76104  0.71536
    ##                      Class: 6 Class: 7 Class: 8 Class: 9
    ## Sensitivity           0.65780  0.76280  0.78380  0.80160
    ## Specificity           0.98300  0.96144  0.97916  0.96602
    ## Pos Pred Value        0.81130  0.68733  0.80688  0.72386
    ## Neg Pred Value        0.96276  0.97332  0.97605  0.97769
    ## Prevalence            0.10000  0.10000  0.10000  0.10000
    ## Detection Rate        0.06578  0.07628  0.07838  0.08016
    ## Detection Prevalence  0.08108  0.11098  0.09714  0.11074
    ## Balanced Accuracy     0.82040  0.86212  0.88148  0.88381

``` r
Prediction_data_test <-predict_classes(model, test_data)
confusionMatrix(table(Prediction_data_test,test_label))
```

    ## Confusion Matrix and Statistics
    ## 
    ##                     test_label
    ## Prediction_data_test   0   1   2   3   4   5   6   7   8   9
    ##                    0 563  13  38   9  14   5   0   5  47  22
    ##                    1  26 750   8   3   3   4   8   3  51  94
    ##                    2 110   9 544  96 129  82  93  42  16  10
    ##                    3  54  25 140 611 110 351 151  69  45  41
    ##                    4  19   3  79  51 505  24  51  41   4   2
    ##                    5   7   6  73 120  42 454  14  87   6   9
    ##                    6  10  14  47  37  40   9 661   0   3   7
    ##                    7  32  15  42  47 145  55  15 744   5  27
    ##                    8 116  24  17   9  10   5   3   2 767  12
    ##                    9  63 141  12  17   2  11   4   7  56 776
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.6375         
    ##                  95% CI : (0.628, 0.6469)
    ##     No Information Rate : 0.1            
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.5972         
    ##                                          
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
    ## Sensitivity            0.5630   0.7500   0.5440   0.6110   0.5050   0.4540
    ## Specificity            0.9830   0.9778   0.9348   0.8904   0.9696   0.9596
    ## Pos Pred Value         0.7863   0.7895   0.4810   0.3826   0.6483   0.5550
    ## Neg Pred Value         0.9529   0.9724   0.9486   0.9537   0.9463   0.9405
    ## Prevalence             0.1000   0.1000   0.1000   0.1000   0.1000   0.1000
    ## Detection Rate         0.0563   0.0750   0.0544   0.0611   0.0505   0.0454
    ## Detection Prevalence   0.0716   0.0950   0.1131   0.1597   0.0779   0.0818
    ## Balanced Accuracy      0.7730   0.8639   0.7394   0.7507   0.7373   0.7068
    ##                      Class: 6 Class: 7 Class: 8 Class: 9
    ## Sensitivity            0.6610   0.7440   0.7670   0.7760
    ## Specificity            0.9814   0.9574   0.9780   0.9652
    ## Pos Pred Value         0.7983   0.6602   0.7948   0.7126
    ## Neg Pred Value         0.9630   0.9711   0.9742   0.9749
    ## Prevalence             0.1000   0.1000   0.1000   0.1000
    ## Detection Rate         0.0661   0.0744   0.0767   0.0776
    ## Detection Prevalence   0.0828   0.1127   0.0965   0.1089
    ## Balanced Accuracy      0.8212   0.8507   0.8725   0.8706
