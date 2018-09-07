// A python script to read and write to hbase table
// Requirement : python 3
from pyspark import SparkContext
from pyspark import sql
import json

sc = SparkContext(appName="HBaseReadWrite")
sqlContext = sql.SQLContext(sc)


def get_hbase_as_rdd(host,tablename):
    """Reads and returns rdd from hbase using pyspark"""
    conf = {"hbase.zookeeper.quorum": host,
            "hbase.mapreduce.inputtable": tablename}
    print ("Connecting to host: " + conf["hbase.zookeeper.quorum"] + " table: " + conf["hbase.mapreduce.inputtable"])
    keyConv = "org.apache.spark.examples.pythonconverters.ImmutableBytesWritableToStringConverter"
    valueConv = "org.apache.spark.examples.pythonconverters.HBaseResultToStringConverter"
    hbase_rdd = sc.newAPIHadoopRDD("org.apache.hadoop.hbase.mapreduce.TableInputFormat",
                                   "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
                                   "org.apache.hadoop.hbase.client.Result",
                                   keyConverter=keyConv,
                                   valueConverter=valueConv,
                                   conf=conf)
    return hbase_rdd


def save_rdd_to_hbase(host, tablename, rdd):
    """Writes rdd to hbase using pyspark"""
    conf = {"hbase.zookeeper.quorum": host,
             "hbase.mapred.outputtable": tablename,
             "mapreduce.outputformat.class": "org.apache.hadoop.hbase.mapreduce.TableOutputFormat",
             "mapreduce.job.output.key.class": "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
             "mapreduce.job.output.value.class": "org.apache.hadoop.io.Writable"}
    print("Connecting to host: " + conf["hbase.zookeeper.quorum"] + " table: " + conf["hbase.mapred.outputtable"])
    keyConv = "org.apache.spark.examples.pythonconverters.StringToImmutableBytesWritableConverter"
    valueConv = "org.apache.spark.examples.pythonconverters.StringListToPutConverter"
    rdd = rdd.map(lambda row: (str(json.loads(row)["row"]),
                               [str(json.loads(row)["row"]), str(json.loads(row)["columnFamily"]),
                                str(json.loads(row)["qualifier"]), str(json.loads(row)["value"])]))
    rdd.saveAsNewAPIHadoopDataset(conf=conf, keyConverter=keyConv, valueConverter=valueConv)

host = "xxx.xxx.xxx.xxx" //or "xxx.xxx.xxx.xxx:xxxx"
inputTable = "input"
outTable = "output" //make sure output table exists with same schema/columnfamily
myrdd = get_hbase_as_rdd(host, inputTable)
myrdd = myrdd.flatMap(lambda row : row[1].split('\n'))
myrdd = myrdd.filter(lambda row : str(json.loads(row)["qualifier"])=="message")
save_rdd_to_hbase(host, outTable, myrdd)
