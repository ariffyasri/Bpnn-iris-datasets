import java.io.*;

public class charArray {
   public static void main(String [] args) {
      String str = new String("Welcome to the family");
      char [] st = str.toCharArray();
      for(int x=0;x<str.length();x++) {
         System.out.print(st[x]);
      }
   }  
}