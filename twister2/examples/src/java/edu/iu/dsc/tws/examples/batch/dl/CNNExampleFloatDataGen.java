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
import edu.iu.dsc.tws.dl.criterion.CrossEntropyCriterion;
import edu.iu.dsc.tws.dl.data.MiniBatch;
import edu.iu.dsc.tws.dl.data.dataset.DataSetFactory;
import edu.iu.dsc.tws.dl.graph.Sequential;
import edu.iu.dsc.tws.dl.module.Dropout;
import edu.iu.dsc.tws.dl.module.Linear;
import edu.iu.dsc.tws.dl.module.LogSoftMax;
import edu.iu.dsc.tws.dl.module.ReLU;
import edu.iu.dsc.tws.dl.module.Reshape;
import edu.iu.dsc.tws.dl.module.SpatialConvolution;
import edu.iu.dsc.tws.dl.module.SpatialMaxPooling;
import edu.iu.dsc.tws.dl.optim.Adam;
import edu.iu.dsc.tws.dl.optim.DistributedOptimizerCustomPacker;
import edu.iu.dsc.tws.dl.optim.Optimizer;
import edu.iu.dsc.tws.dl.optim.Regularizer;
import edu.iu.dsc.tws.dl.optim.regularizer.L2Regularizer;
import edu.iu.dsc.tws.dl.optim.trigger.Triggers;
import edu.iu.dsc.tws.rsched.core.ResourceAllocator;
import edu.iu.dsc.tws.rsched.job.Twister2Submitter;
import edu.iu.dsc.tws.tset.env.BatchEnvironment;
import edu.iu.dsc.tws.tset.env.TSetEnvironment;
import edu.iu.dsc.tws.tset.sets.batch.SourceTSet;

public class CNNExampleFloatDataGen implements Twister2Worker, Serializable {

  @Override
  public void execute(WorkerEnvironment workerEnv) {
    long startTime = System.nanoTime();

    BatchEnvironment env = TSetEnvironment.initBatch(workerEnv);
    Config config = env.getConfig();
    int parallelism = config.getIntegerValue("parallelism");
    int dataSize = config.getIntegerValue("dataSize");
    int testDataSize = 100;
    int batchSize = config.getIntegerValue("batchSize");
    int epoch = config.getIntegerValue("epoch");
    if (batchSize % parallelism != 0) {
      throw new IllegalStateException("batch size should be a multiple of parallelism");
    }
    int miniBatchSize = batchSize / parallelism;
    SourceTSet<MiniBatch> source = DataSetFactory.createImageMiniBatchTestDataSet(env,
        1, 28, 28, miniBatchSize, dataSize, parallelism, true, 784);
    int featureSize = 3 * 3 * 64;

    Sequential model = new Sequential();
    model.toFloat();
    model.add(convolutionMN(1, 32, 5, 5));
    model.add(new ReLU());
    model.add(convolutionMN(32, 32, 5, 5));
    model.add(new SpatialMaxPooling(2, 2, 2, 2));
    model.add(new ReLU());
    model.add(new Dropout(0.5, true));
    model.add(convolutionMN(32, 64, 5, 5));
    model.add(new SpatialMaxPooling(2, 2, 2, 2));
    model.add(new ReLU());
    model.add(new Dropout(0.5, true));
    model.add(new Reshape(new int[]{featureSize}));
    model.add(new Linear(featureSize, 256, true));
    model.add(new ReLU());
    model.add(new Dropout(0.5, true));
    model.add(new Linear(256, 10, true));
    model.add(new LogSoftMax());


    AbstractCriterion criterion = new CrossEntropyCriterion();
    criterion.toFloat();

    //Define Oprimizer
    Optimizer<MiniBatch> optimizer =
        new DistributedOptimizerCustomPacker(env, model, source, criterion);
    optimizer.setOptimMethod(new Adam());
    optimizer.setEndWhen(Triggers.maxEpoch(epoch));
    optimizer.optimize();
//    double accuracy = model.predictAccuracy(testSrc, batchSize, testDataSize);
//    long endTime = System.nanoTime();
//    if (env.getWorkerID() == 0) {
//      System.out.println("Total Time : " + (endTime - startTime) / 1e-6 + "ms");
//      System.out.println("Accuracy : " + accuracy * 100 + "%");
//    }
  }

  private SpatialConvolution convolutionMN(int nInputPlane, int nOutputPlane, int kW, int kH) {

    double weightDecay = 1e-4;
    Regularizer wReg = new L2Regularizer(weightDecay);
    Regularizer bReg = new L2Regularizer(weightDecay);
    return new SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, 1, 1,
        0, 0, 1, true, wReg, bReg, true);
  }

  public static void main(String[] args) throws ParseException {
    // lets take number of workers as an command line argument
    Options options = new Options();
    options.addOption("p", true, "parallelism");
    options.addOption("b", true, "batchSize");
    options.addOption("d", true, "dataSize");
    options.addOption("cpu", true, "CPU");
    options.addOption("ram", true, "RAM");
    options.addOption("e", true, "Epcoh");

    CommandLineParser commandLineParser = new DefaultParser();
    CommandLine cmd = commandLineParser.parse(options, args);
    double cpu = 2.0;
    int mem = 2048;
    int numberOfWorkers = Integer.parseInt(cmd.getOptionValue("p"));
    int batchSize = Integer.parseInt(cmd.getOptionValue("b"));
    int dataSize = Integer.parseInt(cmd.getOptionValue("d"));
    int epoch = Integer.parseInt(cmd.getOptionValue("e"));

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

    Twister2Job twister2Job = Twister2Job.newBuilder()
        .setJobName("CNN-job")
        .setWorkerClass(CNNExampleFloatDataGen.class)
        .addComputeResource(cpu, mem, numberOfWorkers)
        .setConfig(jobConfig)
        .build();
    // now submit the job
    Twister2Submitter.submitJob(twister2Job, config);
  }
}
