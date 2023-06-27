
# **Machine Learning Portfolio**
### **Red Rocks Community College**, Spring 2023

---
Introduction
The following is an overview of the homework and the projects for the machine learning class at RRCC (Spring 2023). Each block below outlines what the main topic was for the homework, the current state that the notebook is in, links to other notes that I found and questions that I have. Additionally I have linked to the actual notebook for overview and reference.

---

## **K-Nearest Neighbors**

_KNN, short for K Nearest Neighbor is a set of tools used to measure the displacements between clusters taken from an imported set of data. Clusters are repeated data points in the set. Using a vareity of tools from numpy, pandas, matplot lib, and even from the KNeihborsClassifier from sklearn, data is imported into a workable space like the notebook and arranged into an organizable list. From this list, also known as an array, clusters are detected and their distance from one another measured for the KNN classifier._

LINKS:

- https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
 ---
- Joby, A. (2021, July 19). What Is K-Nearest Neighbor? An ML Algorithm to Classify Data. Learn Hub | G2. https://learn.g2.com/k-nearest-neighbor
 ---
- https://www.ibm.com/topics/knn
 ---
- Harrison, O. (2019a, July 14). Machine learning basics with the K-nearest neighbors algorithm. Medium. https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
---

## **K-Means**

_K Means is a broad category for a lot of different tools, but each of them all have to do with visualizing the actual clusters. The process almost always begins with arranging the data in question into arrays. Graphs can be made from these new lists by coordinating the number of clusters discovered inside them with the overall data contents, also known as the variance. The full amount of clusters in one data set can be sorted and visualized using silhouette graphs. These graphs can also be used to show the average amount of clusters per each label in the data set._

LINKS:

- (LEDU), E. E. (2018, September 12). Understanding K-means clustering in machine learning. Medium. https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
---
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
---
- Banoula, M. (2023, April 13). K-means clustering algorithm: Applications, types, and demos [updated]: Simplilearn. Simplilearn.com. https://www.simplilearn.com/tutorials/machine-learning-tutorial/k-means-clustering-algorithm
---
- Piech, C. (2013). CS221. Stanford University. https://stanford.edu/~cpiech/cs221/handouts/kmeans.html
---

## **Linear Regression with Gradient Descent**

_To determine gradient descent of data is to determine and visualize the trajectory of the clusters. That kind of visualization is only possible if cluster points are mapped out in relevance to time variables, which represent changes in trajectory over time. With the graphs made, a line of best fit can be formulated to depict what kind of rate of change the clusters are moving in._

LINKS:

- Mirko Stojiljkovic'. (2021, January 27). Stochastic Gradient Descent Algorithm With Python and NumPy – Real Python. Python Tutorials – Real Python. https://realpython.com/gradient-descent-algorithm-python/
---
- Gupta, S. (2022, April 17). The 7 Most Common Machine Learning Loss Functions. Built In. https://builtin.com/machine-learning/common-loss-functions
---
- Oppermann, A., Powers, J., & Pandey, P. (2022, December 14). How Loss Functions Work in Neural Networks and Deep Learning. Built In. https://builtin.com/machine-learning/loss-functions
---

## **Evolutionary Algorithms**

_Evolution of an algorithm is the progression of a machine's learning process in order to fulfill a specific task or in this case, reach a target. The process begins with the establishment of a string of numbers that two other numerical sequences or "parents" must reach by creating new sequences that provide closer approximation to the target. These new sequences, also known as descendant vectors become parents themselves and will continue to produce new sequences until they are as close as they can be to the desired target or the amount of epochs determined by the programmer runs out._

LINKS:

- Telikani, A., Tahmassebi, A., Banzhaf, W., & Gandomi, A. (2021, October). Evolutionary Machine Learning: A Survey. ACM Digital Library. https://dl.acm.org/doi/fullHtml/10.1145/3467477
---
- Karimi, A., & Baeldung CS. (2022, November 8). An Overview of Evolutionary Algorithms | Baeldung on Computer Science. Baeldung on Computer Science. https://www.baeldung.com/cs/evolutionary-algorithms-for-ai
---
- Lange, S. (2022). Evolutionary Algorithems. Albert-Ludwigs University of Freiburg.https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjX1cL1-db-AhXsAjQIHeFnC5YQFnoECC8QAQ&url=https://ml.informatik.uni-freiburg.de/former/_media/teaching/ss13/ml/riedmiller/evolution.pdf&usg=AOvVaw3RdSpzhid-2i8s46713JzG
---

## **Neural Networks**

_The most basic form of a neural net involves taking strings of numbers, called weight matrices and feeding them through an equation known as an activation function. This first exercise involves creating the function with a tool known as sigmoid. The activation function is then applied to every layer the neural net contains to process the data it will be given. This process is known as Forward Passing. A layer in a neural net is a component in the overall arrangement that takes data or information and processes it in a way the programmer determines. The processed data is then passed to the next layer, which does the same thing, but with different parameters from the previous layer in the network. Once the data has been passed through all layers, the performance and overall accuracy of the neural net's findings are evaluated using the Cost Equation or Mean Squared Error. The equation is depicted as follows: $$ \textrm{Cost} = \frac{1}{N} \sum{i=1}^N (a_{1}^{(3)} - y_i) ^ 2 $$._

_The result of the MSE is a percentage. The higher the number, the more accurate the neural nets findings are._

LINKS:

- PhD, M. S. (2019, June 17). Simple Introduction to Neural Networks. Medium. https://towardsdatascience.com/simple-introduction-to-neural-networks-ac1d7c3d7a2c
---
- Engati. (2021). Sigmoid function | Engati. https://www.engati.com/glossary/sigmoid-function#toc-what-is-a-sigmoid-function-
---
- Gupta, A. (2021, December 15). Mean Squared Error : Overview, Examples, Concepts and More | Simplilearn. Simplilearn.com. https://www.simplilearn.com/tutorials/statistics-tutorial/mean-squared-error
---
 
