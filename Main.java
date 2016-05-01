import java.util.*;
import java.io.*;

public class Main {   
   public static void main(String [] args) {
   
      double learningRate = 0.5;
      //range of learning rate is between 0.1 - 1.0
      
      int numberOfEpoch = 500; 
      //number of iteration, the more the iteration, the less error will become
      
      double numberOfClasses = 3.0;
      //number of classes is the categories value in class, 
      //which is iris-setosa,iris-versicolour and iris verginica
      
      Backpropogation bp = new Backpropogation();
      
      bp.inputByFile("training-dataset.csv",learningRate,numberOfEpoch);
      bp.bpnntraining();
      //to train the datasets, need to comment bp.bpnntesting first,
      //then run the program
      //training dataset - iris - normalize.csv
      
      
      bp.bpnntesting("testing-dataset.csv","weight-training.txt",numberOfClasses);
      //to test the datasets, need to comment bp.bpnntraining first,
      //then run the program
      //testing dataset = testing.txt
   }
}
