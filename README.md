# knee_osteoarthritis_python_deep_learning
detecting the intensity of knees osteoarthritis using deep learning 
1) dataset
2) model
3) output
4) train_and_test.py
It is recomanded to run the code in google colab.
download the full datatset form kaggle (https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)
Train and test the code using train_and_test.py or knee_classify.ipynb (in a Jupyter notebook or Google Colab).
In the model folder, there is a CNN classification model. You can add a different model of CNN architecture there. It could be ResNet, VGGNet, etc.
This dataset contains knee X-ray data for both knee joint detection and knee KL grading. The grade descriptions are as follows:
      Grade 0 (Normal) : Healthy knee image.
      Grade 1 (Doubtful): Doubtful joint narrowing with possible osteophytic lipping
      Grade 2 (Minimal): Definite presence of osteophytes and possible joint space narrowing
      Grade 3 (Moderate): Multiple osteophytes, definite joint space narrowing, with mild sclerosis.
      Grade 4 (Severe): Large osteophytes, significant joint narrowing, and severe sclerosis.

By pasting the train, val, and test folder structures into the relevant folder, Mannual converts them to normal, doubtful, minimal, moderate, and severe. 
 
