package Projects.CoOccurMatrixNew; 

import java.io.IOException;
import java.util.*; 
import java.lang.String;

import java.io.DataInput;
import java.io.DataOutput;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.io.Writable;

public class CoOccurMatrixNew {

  public static class CoOccurWritable implements Writable{
    private String key;
    private int val;
    public String getKey(){
      return key;
    } 
    public void setKey(String key){
      this.key = key;
    }
    public int getVal(){
      return val;
    }
    public void setVal(int val){
      this.val = val;
    }

   @Override
   //overriding default readFields method. 
   //It de-serializes the byte stream data
   public void readFields(DataInput in) throws IOException {
    key = in.readUTF();
    val = in.readInt();
   }
 
   @Override
   //It serializes object data into byte stream data
   public void write(DataOutput out) throws IOException {
    out.writeUTF(key);
    out.writeInt(val);
   }

  }
  public static class PairTokenizerMapper
       extends Mapper<Object, Text, Text, CoOccurWritable>{

    private CoOccurWritable coOccurWritable = new CoOccurWritable();
    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();

    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      String[] str = value.toString().split(" ");
      for(int i=0;i<str.length;i++){
        for(int j=0;j<str.length;j++){
          if(!str[i].equals(str[j])){
            word.set(str[i]);
            coOccurWritable.setKey(str[j]);
            coOccurWritable.setVal(1);
            context.write(word, coOccurWritable);
          }          
        }
      }
    }
  }

  public static class IntSumReducer
       extends Reducer<Text,CoOccurWritable,Text,Text> {
    private Text finalResult = new Text();

    public void reduce(Text key, Iterable<CoOccurWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
      Map<String,Integer> map=new HashMap<String,Integer>();
      
      for (CoOccurWritable val : values) {
        if (!map.containsKey(val.getKey())){
          map.put(val.getKey(),val.getVal());
        }else{
          map.put(val.getKey(),map.get(val.getKey())+val.getVal());
        }        
      }
      String result = "{ ";
      for(Map.Entry m:map.entrySet()){  
        result = result + m.getKey()+" : "+m.getValue() + " ";  
      }  
      result = result + "}";
      finalResult.set(result);
      context.write(key, finalResult);
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "CoOccurance Matrix New");
    job.setJarByClass(CoOccurMatrixNew.class);
    job.setMapperClass(PairTokenizerMapper.class);
    job.setReducerClass(IntSumReducer.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(CoOccurWritable.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}