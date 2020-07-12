//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
package edu.iu.dsc.tws.examples.tset;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.logging.Logger;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import edu.iu.dsc.tws.api.JobConfig;
import edu.iu.dsc.tws.api.Twister2Job;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.resource.Twister2Worker;
import edu.iu.dsc.tws.api.resource.WorkerEnvironment;
import edu.iu.dsc.tws.api.tset.fn.MapFunc;
import edu.iu.dsc.tws.api.tset.fn.SourceFunc;
import edu.iu.dsc.tws.rsched.core.ResourceAllocator;
import edu.iu.dsc.tws.rsched.job.Twister2Submitter;
import edu.iu.dsc.tws.tset.env.BatchEnvironment;
import edu.iu.dsc.tws.tset.env.TSetEnvironment;
import edu.iu.dsc.tws.tset.fn.LoadBalancePartitioner;
import edu.iu.dsc.tws.tset.links.batch.PartitionTLink;
import edu.iu.dsc.tws.tset.links.batch.ReduceTLink;
import edu.iu.dsc.tws.tset.sets.batch.ComputeTSet;
import edu.iu.dsc.tws.tset.sets.batch.SinkTSet;
import edu.iu.dsc.tws.tset.sets.batch.SourceTSet;

public class HelloTSet implements Twister2Worker, Serializable {
  private static final Logger LOG = Logger.getLogger(HelloTSet.class.getName());

  private static final long serialVersionUID = -2;

  @Override
  public void execute(WorkerEnvironment workerEnv) {
    BatchEnvironment env = TSetEnvironment.initBatch(workerEnv);
    LOG.info("Strating Hello TSet Example");
    int para = env.getConfig().getIntegerValue("para", 4);

    SourceTSet<int[]> source = env.createSource(new SourceFunc<int[]>() {
      private int count = 0;

      @Override
      public boolean hasNext() {
        return count < para;
      }

      @Override
      public int[] next() {
        count++;
        return new int[]{1, 1, 1};
      }
    }, para).setName("source");

    PartitionTLink<int[]> partitioned = source.partition(new LoadBalancePartitioner<>());

    ComputeTSet<int[], Iterator<int[]>> mapedPartition = partitioned.map(
        (MapFunc<int[], int[]>) input -> Arrays.stream(input).map(a -> a * 2).toArray()
    );

    ReduceTLink<int[]> reduce = mapedPartition.reduce((t1, t2) -> {
      int[] ret = new int[t1.length];
      for (int i = 0; i < t1.length; i++) {
        ret[i] = t1[i] + t2[i];
      }
      return ret;
    });

    SinkTSet<int[]> sink = reduce.sink(value -> {
      LOG.info("Results " + Arrays.toString(value));
      return false;
    });
    env.run(sink);

    LOG.info("Ending  Hello TSet Example");

  }

  public static void main(String[] args) throws ParseException {
    // first load the configurations from command line and config files
    Options options = new Options();
    options.addOption("para", true, "Workers");
    CommandLineParser commandLineParser = new DefaultParser();
    CommandLine cmd = commandLineParser.parse(options, args);

    Config config = ResourceAllocator.loadConfig(new HashMap<>());
    int para = Integer.parseInt(cmd.getOptionValue("para"));
    // build JobConfig

    JobConfig jobConfig = new JobConfig();
    jobConfig.put("para", Integer.toString(para));
    submitJob(config, para, jobConfig, HelloTSet.class.getName());
  }

  private static void submitJob(Config config, int containers, JobConfig jobConfig, String clazz) {
    Twister2Job twister2Job;
    twister2Job = Twister2Job.newBuilder()
        .setJobName(clazz)
        .setWorkerClass(clazz)
        .addComputeResource(1, 512, containers)
        .setConfig(jobConfig)
        .build();
    // now submit the job
    Twister2Submitter.submitJob(twister2Job, config);
  }

}
