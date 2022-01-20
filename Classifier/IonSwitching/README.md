### About the project
Data on 10 different ion channels was analyzed for classification purpose. The dataset contained information about time, signal and open channels.

Data was obtained from  [kaggle](https://www.kaggle.com/c/liverpool-ion-switching). For this project, following files were created:

- [IonSwitching.ipynb](IonSwitching.ipynb) provides explorative data analysis.

- [Liverpool_IonSwitching.py](Liverpool_IonSwitching.py) provides preprocessing and machine learning with different algorithms. It was designed as an automated approach to find the best suitable model for prediction. It was built to show advantages of engineering for automation and focused less on highlighting a single *Machine Learning* algorithm. It furthermore includes the calculation of recall, precision and f1-score that was suggested for time-serial data in the paper
[32nd Conference on Neural Information Processing Systems (NeurIPS 2018), Montréal, Canada](https://proceedings.neurips.cc/paper/2018/file/8f468c873a32bb0619eaeb2050ba45d1-Paper.pdf).

For a quick overview please also refer to [IonSwitching.pptx](IonSwitching.pptx). Note that a subsample of 10'000 observations was used to evaluate the scores that are shown in this .pptx document to demonstrate the procedure in principle.

**Remark**: The data is time-dependent and therefore requires different preparation as conventional classification approaches. Classical score calculations are not appropriate in this example and were therefore replaced by another method as mentioned above. However, the train test split procedure should not be performed in a randomized way as it was done in this project. In case more data is available in the future, the split procedure into train and test data shall be updated to integrate time dependencies.
