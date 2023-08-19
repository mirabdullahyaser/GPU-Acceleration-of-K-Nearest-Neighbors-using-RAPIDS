# GPU Acceleration of k-Nearest Neighbors using RAPIDS
## I.   Abstract
kNN is a simple machine learning algorithm used for classification and regression that predicts an unknown observation by comparing it with k most similar observations. CPU version of kNN is implemented on MNIST dataset using python’s scikit-learn that predicts the result taking 25 minutes While GPU version of kNN is implemented using RAPIDS cuML that takes only 2.5 seconds. This shows that 600x speedup can be achieved using cuML on kNN.

## II.  Introduction
K-nearest neighbors (kNN) is a simple, easy to implement and powerful machine learning algorithm that can be used to solve both classification and regression problems. This algorithm belongs to the supervised machine learning domain and is widely used in data science, data mining and pattern recognition. The kNN algorithm assumes that similar data points exist in close proximity. It means that similar data is close to each other. Based on this assumption, it predicts an unknown observation by comparing it with k most similar observations in the training dataset.
As kNN is an important and widely used machine learning algorithm in data science and analytical tasks therefore, providing GPU acceleration will let data scientists work with large datasets and complex models in a small amount of time. The acceleration of the GPU will let data scientists iterate through hundreds of thousands of variants required for hyperparameter optimization (HPO) in little time and will eventually help in increasing the model accuracy. RAPIDS is a suite of open source software libraries that gives you the freedom to execute end-to-end data science and analytics pipelines entirely on GPUs. GPU acceleration of the kNN algorithm will be achieved using the cuML library of RAPIDS.   

## III.   Literature Review
Python programming language is quite famous in data science and various classification and regression problems can be solved using python. The reason for the popularity of python in data science is its simple and easy to use libraries. Using these libraries, we can easily load data, train models and predict results without explicitly developing the model. Scikit-learn is a famous machine learning library in python that provides various classification, regression and cluster algorithms. kNN algorithms can also be developed using scikit learn in a few lines without making the model from scratch. Although scikit-learn is a great library because of its various built-in models yet it works on CPU and GPU acceleration cannot be achieved on it.
As kNN is a widely used algorithm hence different implementations have been made to achieve the speedup. V. Garcia, E. Debreuve, F. Nielsen and M. Barlaud implemented a GPU implementation of kNN search problem. In their paper, they proposed two fast GPU-based implementations of the brute-force kNN search algorithm using the CUDA and CUBLAS APIs. They show that their CUDA and CUBLAS implementations are up to, respectively, 64X and 189X faster on synthetic data than the highly optimized ANN C++ library, and up to, respectively,25X and 62X faster on high-dimensional SIFT matching [3]. Although this GPU implementation is fast, it is nowhere close to RAPIDS.
RAPIDS is incubated by NVIDIA. RAPIDS utilize NVIDIA CUDA primitives for low-level compute optimization, and exposes GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces. It accelerates python data science toolchain with minimal code changes and no new tools to learn. cuML from RAPIDS is a collection of machine learning libraries that will provide GPU versions of algorithms available in scikit-learn. The kNN algorithm can also be developed using cuML with minimal changes. cuML is so fast that it can improve the model accuracy by iterating through thousands of variants in little time used in hyperparameter optimization (HPO), Data Augmentation and Feature Engineering and Selection.

## IV.   Methodology
In order to infer (predict) one unknown observation, we must compute how similar that unknown observation is to each of the known training observations. An observation is a vector thus similarity is the distance between two vectors that will be calculated using Euclidean distance. Given two observations x_1 ∈ R^p and x_2 ∈ R^p, the formula is
		
    dist = √(x_1-x_2) ∙ (x_1-x_2).                      (1)
Thus, we see if we have p features, then it requires p multiplies, p subtractions, p additions, and one square root to compute the distance between two observations. Since we must compare all the test observations with all the training observations, the total number of computations ignoring square root is
	   
     3 * p * len(train) * len(test).                      (2)

We will use Kaggle CPU and GPU cloud computer services to solve the problem. We are using the famous MNIST digit image dataset. We are using scikit-learn for the CPU version and cuML for the GPU version to develop the kNN model.  First, we will load the training dataset and then we perform grid search with a 20% holdout set to find the optimal k. Alternatively, we could use full KFold cross validation shown below. We find that k=3 achieves the best validation accuracy. We are using KNeighborsClassifier () from cuML to develop a kNN classifier object and then the kNN model will be developed using the object. Later we use the test dataset to predict the result.

## V.    Experimental results
For comparison we are using scikit-learn kNN that is using Google Colab’s CPU (Intel(R) Xeon(R) CPU @ 2.20GHz) and RAPIDS cuML’s kNN that utilizes Google Columb's GPU (Nvidia Tesla P4 with 2560 CUDA cores). To find the optimal value of k we use the Grid Search. Following graph shows the validation accuracy for different k values.
From the graph we can see that k=3 has the best validation accuracy. For a more accurate grid search we used KFold instead of a single holdout set. This shows us similar results where for k=3, accuracy is

                           ACC = 0.9598479923996199.                        (3)

When we predict the test data with the GPU version of kNN then it shows a prediction accuracy of 97% with only 2.5 seconds of work. On the other hand, if we use the CPU version of kNN that uses scikit-learn, it takes 25 minutes for prediction. Hence, the comparison shows that cuML’s GPU kNN is 600x faster than scikit-learn’s CPU kNN.

## VI.   References

[1] C. Deotte “Accelerating k-nearest neighbors 600x using RAPIDS cuML” in RAPIDS-AI, Medium, 2020
[2] O. Harrison “Machine Learning Basics with the K-Nearest Neighbors Algorithm” towardsdatascience, 2018
[3] V. Garcia and E. Debreuve and F. Nielsen and M. Barlaud. k-nearest neighbor search: fast GPU-based implementations and application to high-dimensional feature matching. In Proceedings of the IEEE International Conference on Image Processing (ICIP), Hong Kong, China, September 2010
