Requirements:
Python(version >= 3.6)
pytorch(version>=1.4.0)
ordered_set(version>=3.1)
numpy(version>=1.16.2)
torch_scatter(version>=2.0.4)
scikit_learn(version>=0.21.1)
We highly recommend you to use conda for package management.


Model Training:
1)Create a folder "test_data" under folder "data" and move training data, valid data, and test data to the folder. 
2)Use the following command to train the model, the model will be named as "test_model" and saved in the directory "model_saved".
  python main.py -data test_data -gpu 0 -name test_model -epoch 500


Drug-Disease Predicting:
1)Create a test file named as "ad_pre.txt" and moved the file to the folder "test_data".
2)Run the following command, predicting results will be saved in the file "pre_results.txt".
  python test.py -data test_data -gpu 1 -name test_model -save_result pre_results.txt -test_file ad_pre.txt


Parameter Note:
-data the directory of training and testing data
-gpu the GPU to use
-name the name of the model snapshot (used for storing model parameters)
-epoch the number of epochs
-save_result the filename that is used to store test results
-test_file the name of testing file


