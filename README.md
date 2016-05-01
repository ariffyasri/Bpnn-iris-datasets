# Backpropogation-iris-datasets

This artificial neural network program only suitable for [iris-datasets](http://archive.ics.uci.edu/ml/datasets/Iris).

The dataset used in this program has been normalized using min-max normalization.

![alt-tag](http://i.imgur.com/jmRFSAL.png)

#Input
There are 4 inputs/independant variables which are :
- sepal-length
- sepal-width
- petal-length
- petal-width 

#Output
There are 3 outputs that will be classify which are :
- Iris-Setosa
- Iris-Versicolour
- Iris-Verginica


##Sample output
![alt-tag](http://i.imgur.com/IJn73ke.png)


#How to use this program?
*Main.java

Main function where need to state your :

* learning rate
* number of epoch
* number of classes - you can ignore this since this is static, unless you change the dataset

Training process

1. need to "comment" the bpnntesting like picture below
![alt-tag](http://i.imgur.com/AN9ooST.png)
2. run the program
3. then erase the previous weight training in weight-training.txt

Testing process : 

1. need to "comment" the bpnntraining like picture below
![alt-tag](http://i.imgur.com/iohG4sT.png)
2. run the program

*Backpropogation.java
