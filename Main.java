import java.util.*;
import java.io.*;

public class Main {   
	public static void main(String [] args) {
		Scanner scn = new Scanner(System.in);
		double learningRate = 0.5;
		//range of learning rate is between 0.1 - 1.0

		int numberOfEpoch = 10000; 
		//number of iteration, the more the iteration, the less error will become

		double numberOfClasses = 3.0;
		//number of classes is the categories value in class, 
		//which is iris-setosa,iris-versicolour and iris verginica

		Backpropogation bp = new Backpropogation();

		bp.inputByFile("training-dataset.csv",learningRate,numberOfEpoch);

		System.out.println("Please enter : \n1) Training Iris Dataset\n2) Testing Iris Dataset");
		int choose = scn.nextInt();
		if(choose == 1) {
			bp.bpnntraining();
		//training dataset - training-dataset.csv
		}
		else if(choose == 2) {
			bp.bpnntesting("testing-dataset.csv","weight-training.txt",numberOfClasses);	
			//testing dataset = testing-dataset.csv
		}
	}
}
