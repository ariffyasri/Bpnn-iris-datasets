import java.util.*;
import java.io.*;

public class Main {   
   public static void main(String [] args) {
      Backpropogation bp = new Backpropogation();
      bp.inputByFile("iris - normalize1.csv",0.5,5000);
      //bp.bpnntraining();
      double numberOfClasses = 3.0;
      bp.bpnntesting("testing.txt","weightraining.txt",numberOfClasses);
   }
}
