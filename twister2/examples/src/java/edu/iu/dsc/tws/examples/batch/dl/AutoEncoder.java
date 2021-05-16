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
package edu.iu.dsc.tws.examples.batch.dl;


import java.io.Serializable;
import java.util.HashMap;
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
import edu.iu.dsc.tws.dl.criterion.AbstractCriterion;
import edu.iu.dsc.tws.dl.criterion.MSECriterion;
import edu.iu.dsc.tws.dl.data.MiniBatch;
import edu.iu.dsc.tws.dl.data.dataset.DataSetFactory;
import edu.iu.dsc.tws.dl.graph.Sequential;
import edu.iu.dsc.tws.dl.module.Linear;
import edu.iu.dsc.tws.dl.module.ReLU;
import edu.iu.dsc.tws.dl.module.Reshape;
import edu.iu.dsc.tws.dl.module.Sigmoid;
import edu.iu.dsc.tws.dl.optim.Adam;
import edu.iu.dsc.tws.dl.optim.DistributedOptimizer;
import edu.iu.dsc.tws.dl.optim.OptimMethod;
import edu.iu.dsc.tws.dl.optim.Optimizer;
import edu.iu.dsc.tws.dl.optim.trigger.Triggers;
import edu.iu.dsc.tws.rsched.core.ResourceAllocator;
import edu.iu.dsc.tws.rsched.job.Twister2Submitter;
import edu.iu.dsc.tws.tset.env.BatchEnvironment;
import edu.iu.dsc.tws.tset.env.TSetEnvironment;
import edu.iu.dsc.tws.tset.sets.batch.SourceTSet;

/**
 * Simple AutoEncoder example
 */
public class AutoEncoder implements Twister2Worker, Serializable {
  private static final Logger LOG = Logger.getLogger(AutoEncoder.class.getName());

  @Override
  public void execute(WorkerEnvironment workerEnv) {
    long startTime = System.nanoTime();

    BatchEnvironment env = TSetEnvironment.initBatch(workerEnv);
    Config config = env.getConfig();
    int parallelism = config.getIntegerValue("parallelism");
    int dataSize = config.getIntegerValue("dataSize");
    int batchSize = config.getIntegerValue("batchSize");
    int epoch = config.getIntegerValue("epoch");
    String dataFile = config.getStringValue("data");
    if (batchSize % parallelism != 0) {
      throw new IllegalStateException("batch size should be a multiple of parallelism");
    }
    int miniBatchSize = batchSize / parallelism;

    SourceTSet<MiniBatch> source = DataSetFactory
        .createMiniBatchDataSet(env, dataFile, miniBatchSize, dataSize, parallelism, false);

    //Define model
    int features = 100;
    int classes = 12;
    Sequential model = new Sequential();
    model.add(new Reshape(new int[]{features}));
    model.add(new Linear(features, classes));
    model.add(new ReLU(false));
    model.add(new Linear(classes, features));
    model.add(new Sigmoid());
    //criterion
    AbstractCriterion criterion = new MSECriterion();

    OptimMethod optimMethod = new Adam();
    //Define Oprimizer
    Optimizer<MiniBatch> optimizer = new DistributedOptimizer(env, model, source, criterion);
    optimizer.setOptimMethod(optimMethod);
    optimizer.setEndWhen(Triggers.maxEpoch(epoch));
    optimizer.optimize();
    long endTime = System.nanoTime();
    if (env.getWorkerID() == 0) {
      System.out.println("Total Time : " + (endTime - startTime) / 1e-6 + "ms");
    }
  }

  public static void main(String[] args) throws ParseException {
    // lets take number of workers as an command line argument
    Options options = new Options();
    options.addOption("p", true, "parallelism");
    options.addOption("b", true, "batchSize");
    options.addOption("d", true, "dataSize");
    options.addOption("cpu", true, "CPU");
    options.addOption("ram", true, "RAM");
    options.addOption("data", true, "Data");
    options.addOption("e", true, "Epcoh");

    CommandLineParser commandLineParser = new DefaultParser();
    CommandLine cmd = commandLineParser.parse(options, args);
    double cpu = 2.0;
    int mem = 2048;
    int numberOfWorkers = Integer.parseInt(cmd.getOptionValue("p"));
    int batchSize = Integer.parseInt(cmd.getOptionValue("b"));
    int dataSize = Integer.parseInt(cmd.getOptionValue("d"));
    int epoch = Integer.parseInt(cmd.getOptionValue("e"));
    String data = cmd.getOptionValue("data");

    if (cmd.hasOption("cpu")) {
      cpu = Double.parseDouble(cmd.getOptionValue("cpu"));
    }

    if (cmd.hasOption("ram")) {
      mem = Integer.parseInt(cmd.getOptionValue("ram"));
    }
    // first load the configurations from command line and config files
    Config config = ResourceAllocator.loadConfig(new HashMap<>());

    // lets put a configuration here
    JobConfig jobConfig = new JobConfig();
    jobConfig.put("dnn-key", "Twister2-DNN");
    jobConfig.put("parallelism", numberOfWorkers);
    jobConfig.put("batchSize", batchSize);
    jobConfig.put("dataSize", dataSize);
    jobConfig.put("epoch", epoch);
    jobConfig.put("data", data);

    Twister2Job twister2Job = Twister2Job.newBuilder()
        .setJobName("AutoEncoder-job")
        .setWorkerClass(AutoEncoder.class)
        .addComputeResource(cpu, mem, numberOfWorkers)
        .setConfig(jobConfig)
        .build();
    // now submit the job
    Twister2Submitter.submitJob(twister2Job, config);
  }
}
