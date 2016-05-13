import java.util.*;
public class testing {
   public static void main(String [] args) {
      String s = "1, 2, 4, 5, 6, 7";
      String [] splits = s.split(", ");      
      int l = splits.length;
      int [] n = new int[l];
      for(int x=0;x<l;x++) {
         n[x] = Integer.valueOf(splits[x]).intValue();
         System.out.println(n[x]);
      }
         
   }
}