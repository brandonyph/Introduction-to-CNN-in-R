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
    epochs = 20,
    validation_split=0.2,
    use_multiprocessing=TRUE
  )

plot(history)
```

    ## `geom_smooth()` using formula 'y ~ x'

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
    ##                     0 4738   16   65   30   23   14    5   16   76   27
    ##                     1   30 4761    6    4    1    1    6    3   32   54
    ##                     2   44   10 4558   69   83   72   55   39    9    8
    ##                     3   18    8   72 4429   74  198   72   56   12   25
    ##                     4   22   10   94   66 4649   50   37   79    6   10
    ##                     5   12    7   71  200   62 4525   25   75    8    7
    ##                     6   11   25   63   89   29   46 4782   12   13   14
    ##                     7   15    6   42   57   60   75   10 4696    1    9
    ##                     8   60   32   21   32    9   10    4    2 4816   33
    ##                     9   50  125    8   24   10    9    4   22   27 4813
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9353          
    ##                  95% CI : (0.9331, 0.9375)
    ##     No Information Rate : 0.1             
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9282          
    ##                                           
    ##  Mcnemar's Test P-Value : 2.474e-09       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
    ## Sensitivity           0.94760  0.95220  0.91160  0.88580  0.92980  0.90500
    ## Specificity           0.99396  0.99696  0.99136  0.98811  0.99169  0.98962
    ## Pos Pred Value        0.94571  0.97203  0.92137  0.89222  0.92554  0.90645
    ## Neg Pred Value        0.99418  0.99470  0.99019  0.98732  0.99220  0.98945
    ## Prevalence            0.10000  0.10000  0.10000  0.10000  0.10000  0.10000
    ## Detection Rate        0.09476  0.09522  0.09116  0.08858  0.09298  0.09050
    ## Detection Prevalence  0.10020  0.09796  0.09894  0.09928  0.10046  0.09984
    ## Balanced Accuracy     0.97078  0.97458  0.95148  0.93696  0.96074  0.94731
    ##                      Class: 6 Class: 7 Class: 8 Class: 9
    ## Sensitivity           0.95640  0.93920  0.96320  0.96260
    ## Specificity           0.99329  0.99389  0.99549  0.99380
    ## Pos Pred Value        0.94060  0.94468  0.95955  0.94521
    ## Neg Pred Value        0.99515  0.99325  0.99591  0.99584
    ## Prevalence            0.10000  0.10000  0.10000  0.10000
    ## Detection Rate        0.09564  0.09392  0.09632  0.09626
    ## Detection Prevalence  0.10168  0.09942  0.10038  0.10184
    ## Balanced Accuracy     0.97484  0.96654  0.97934  0.97820

``` r
Prediction_data_test <-predict_classes(model, test_data)
confusionMatrix(table(Prediction_data_test,test_label))
```

    ## Confusion Matrix and Statistics
    ## 
    ##                     test_label
    ## Prediction_data_test   0   1   2   3   4   5   6   7   8   9
    ##                    0 772  19  64  34  20  13   6  16  66  29
    ##                    1  14 804   4  11   0   4   7   1  43  56
    ##                    2  51   5 596  73  61  54  62  47  20   6
    ##                    3  13   4  69 506  78 172  49  45  17  13
    ##                    4  31   6  77  64 670  37  37  64  10   7
    ##                    5   8   3  68 175  56 616  24  55   2  10
    ##                    6   7  12  68  56  50  21 800   7   6  10
    ##                    7  11   2  34  43  48  61   3 737   9   4
    ##                    8  44  35   9  16  11  12   7   1 802  29
    ##                    9  49 110  11  22   6  10   5  27  25 836
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7139          
    ##                  95% CI : (0.7049, 0.7227)
    ##     No Information Rate : 0.1             
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6821          
    ##                                           
    ##  Mcnemar's Test P-Value : 6.383e-08       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5
    ## Sensitivity            0.7720   0.8040   0.5960   0.5060   0.6700   0.6160
    ## Specificity            0.9703   0.9844   0.9579   0.9489   0.9630   0.9554
    ## Pos Pred Value         0.7430   0.8517   0.6113   0.5238   0.6680   0.6057
    ## Neg Pred Value         0.9746   0.9784   0.9552   0.9453   0.9633   0.9573
    ## Prevalence             0.1000   0.1000   0.1000   0.1000   0.1000   0.1000
    ## Detection Rate         0.0772   0.0804   0.0596   0.0506   0.0670   0.0616
    ## Detection Prevalence   0.1039   0.0944   0.0975   0.0966   0.1003   0.1017
    ## Balanced Accuracy      0.8712   0.8942   0.7769   0.7274   0.8165   0.7857
    ##                      Class: 6 Class: 7 Class: 8 Class: 9
    ## Sensitivity            0.8000   0.7370   0.8020   0.8360
    ## Specificity            0.9737   0.9761   0.9818   0.9706
    ## Pos Pred Value         0.7715   0.7742   0.8302   0.7593
    ## Neg Pred Value         0.9777   0.9709   0.9781   0.9816
    ## Prevalence             0.1000   0.1000   0.1000   0.1000
    ## Detection Rate         0.0800   0.0737   0.0802   0.0836
    ## Detection Prevalence   0.1037   0.0952   0.0966   0.1101
    ## Balanced Accuracy      0.8868   0.8566   0.8919   0.9033
