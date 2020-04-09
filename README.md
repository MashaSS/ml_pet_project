# Beauty estimation 

# Dataset

 [SCUT-FBP5500-Database-Release](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) 

# Weights (for caucasian female)
 Weights have to be in folder '/weights'
 
 [resnet_caucasian_female_weights](https://drive.google.com/file/d/1LJn95k3VxzK_neR9FmzF-_L0VIq0agXT/view)

# Required libraries

 * keras 2.2.5
 * tensorflow 1.14
 
# Trainig script 
 
```
  python train.py 

  -h, --help            show this help message and exit
  -r LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate.
  -e EPOCHES, --epoches EPOCHES
                        Number of epoches.
  -f LOG_FILE, --log_file LOG_FILE
                        Learning rate.
  -a {softmax,tangh,relu}, --activation {softmax,tangh,relu}
                        Activation rule.
  -o {Adam,SGD,RMSprop}, --optimizer {Adam,SGD,RMSprop}
                        Model optimizer. Default: Adam
  -l HIDDEN_LAYERS, --hidden_layers HIDDEN_LAYERS
                        Size of hidden layer before last layer
  -s SIZE, --size SIZE  Size of images. Default: 224:224
  -d DATA_FOLDER, --data_folder DATA_FOLDER
                        Path to data folder.
```

# Test script
 ```
  python test.py

  -h, --help            show this help message and exit
  -w WEIGHTS, --weights WEIGHTS
                        Weights of model.
```

