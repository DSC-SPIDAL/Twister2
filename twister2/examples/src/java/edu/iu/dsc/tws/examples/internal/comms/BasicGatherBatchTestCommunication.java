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
package edu.iu.dsc.tws.examples.internal.comms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.JobConfig;
import edu.iu.dsc.tws.api.Twister2Submitter;
import edu.iu.dsc.tws.api.job.Twister2Job;
import edu.iu.dsc.tws.common.config.Config;
import edu.iu.dsc.tws.common.discovery.IWorkerController;
import edu.iu.dsc.tws.common.resource.AllocatedResources;
import edu.iu.dsc.tws.common.resource.WorkerComputeResource;
import edu.iu.dsc.tws.common.worker.IPersistentVolume;
import edu.iu.dsc.tws.common.worker.IVolatileVolume;
import edu.iu.dsc.tws.common.worker.IWorker;
import edu.iu.dsc.tws.comms.api.BatchReceiver;
import edu.iu.dsc.tws.comms.api.DataFlowOperation;
import edu.iu.dsc.tws.comms.api.MessageFlags;
import edu.iu.dsc.tws.comms.api.MessageType;
import edu.iu.dsc.tws.comms.core.TWSCommunication;
import edu.iu.dsc.tws.comms.core.TWSNetwork;
import edu.iu.dsc.tws.comms.core.TaskPlan;
import edu.iu.dsc.tws.comms.dfw.io.gather.GatherBatchFinalReceiver;
import edu.iu.dsc.tws.comms.dfw.io.gather.GatherBatchPartialReceiver;
import edu.iu.dsc.tws.examples.Utils;
import edu.iu.dsc.tws.examples.utils.RandomString;
import edu.iu.dsc.tws.rsched.core.ResourceAllocator;
import edu.iu.dsc.tws.rsched.core.SchedulerContext;

public class BasicGatherBatchTestCommunication implements IWorker {
  private static final Logger LOG = Logger.
      getLogger(BasicGatherBatchTestCommunication.class.getName());

  private DataFlowOperation aggregate;

  private AllocatedResources resourcePlan;

  private int id;

  private Config config;

  private static final int NO_OF_TASKS = 8;

  private int noOfTasksPerExecutor = 2;

  private RandomString randomString;

  private long startTime = 0;

  @Override
  public void init(Config cfg, int workerID, AllocatedResources resources,
                   IWorkerController workerController,
                   IPersistentVolume persistentVolume,
                   IVolatileVolume volatileVolume) {
    LOG.log(Level.INFO, "Starting the example with container id: " + resources.getWorkerId());

    this.config = cfg;
    this.resourcePlan = resources;
    this.id = workerID;
    this.noOfTasksPerExecutor = NO_OF_TASKS / resources.getNumberOfWorkers();
    this.randomString = new RandomString(128000, new Random(), RandomString.ALPHANUM);

    // lets create the task plan
    TaskPlan taskPlan = Utils.createReduceTaskPlan(cfg, resources, NO_OF_TASKS);
    //first get the communication config file
    TWSNetwork network = new TWSNetwork(cfg, taskPlan);

    TWSCommunication channel = network.getDataFlowTWSCommunication();

    Set<Integer> sources = new HashSet<>();
    for (int i = 0; i < NO_OF_TASKS; i++) {
      sources.add(i);
    }
    int dest = NO_OF_TASKS;

    Map<String, Object> newCfg = new HashMap<>();

    LOG.info("Setting up gather dataflow operation");

    try {
      // this method calls the init method
      // I think this is wrong

      aggregate = channel.gather(newCfg, MessageType.INTEGER, 0, sources,
          dest, new GatherBatchFinalReceiver(new FinalGatherReceive()),
          new GatherBatchPartialReceiver(dest));
//      aggregate = channel.gather(newCfg, MessageType.OBJECT, 0, sources,
//          dest, new FinalGatherReceive());
      for (int i = 0; i < noOfTasksPerExecutor; i++) {
        // the map thread where data is produced
        LOG.info(String.format("%d Starting %d", id, i + id * noOfTasksPerExecutor));
        Thread mapThread = new Thread(new MapWorker(i + id * noOfTasksPerExecutor));
        mapThread.start();
      }
      // we need to communicationProgress the communication
      while (true) {
        try {
          // communicationProgress the channel
          channel.progress();
          // we should communicationProgress the communication directive
          aggregate.progress();
          Thread.yield();
        } catch (Throwable t) {
          t.printStackTrace();
        }
      }
    } catch (Throwable t) {
      t.printStackTrace();
    }
  }

  /**
   * We are running the map in a separate thread
   */
  private class MapWorker implements Runnable {
    private int task = 0;
    private int sendCount = 0;

    MapWorker(int task) {
      this.task = task;
    }

    @Override
    public void run() {
      try {
        LOG.log(Level.INFO, "Starting map worker: " + id);
//      MPIBuffer data = new MPIBuffer(1024);
        startTime = System.nanoTime();
        for (int i = 0; i < 1; i++) {
          int[] data = {task, task * 100};
          // lets generate a message
//          KeyedContent mesage = new KeyedContent(task, data,
//              MessageType.INTEGER, MessageType.OBJECT);
//
          //Set the last message with the corerct flag. Since we only send one message we set it
          //on the first call itself
          int flags = MessageFlags.FLAGS_LAST;
          while (!aggregate.send(task, data, flags)) {
            // lets wait a litte and try again
            try {
              Thread.sleep(1);
            } catch (InterruptedException e) {
              e.printStackTrace();
            }
          }
          Thread.yield();
        }
        LOG.info(String.format("%d Done sending", id));
      } catch (Throwable t) {
        t.printStackTrace();
      }
    }
  }

  private class FinalGatherReceive implements BatchReceiver {
    // lets keep track of the messages
    // for each task we need to keep track of incoming messages
    private List<Integer> dataList;

    private int count = 0;

    private long start = System.nanoTime();

    @Override
    public void init(Config cfg, DataFlowOperation op, Map<Integer, List<Integer>> expectedIds) {
      dataList = new ArrayList<Integer>();
    }

    @Override
    @SuppressWarnings("unchecked")
    public void receive(int target, Iterator<Object> it) {
      int itercount = 0;
      Object temp;

      while (it.hasNext()) {
        itercount++;
        temp = it.next();
        if (temp instanceof List) {
          List<Object> datalist = (List<Object>) temp;
          for (Object o : datalist) {
            int[] data = (int[]) o;
            dataList.add(data[0]);
          }
        } else {
          int[] data = (int[]) temp;
          dataList.add(data[0]);
        }
      }
      LOG.info("Gather results (only the first int of each array)"
          + Arrays.toString(dataList.toArray()));
    }

    public void progress() {

    }
  }

  public static void main(String[] args) {
    // first load the configurations from command line and config files
    Config config = ResourceAllocator.loadConfig(new HashMap<>());

    // build JobConfig
    HashMap<String, Object> configurations = new HashMap<>();
    configurations.put(SchedulerContext.THREADS_PER_WORKER, 8);

    JobConfig jobConfig = new JobConfig();
    jobConfig.putAll(configurations);

    // build the job
    Twister2Job twister2Job = Twister2Job.newBuilder()
        .setName("basic-gather-batched")
        .setWorkerClass(BasicGatherBatchTestCommunication.class.getName())
        .setRequestResource(new WorkerComputeResource(2, 1024), 4)
        .setConfig(jobConfig)
        .build();

    // now submit the job
    Twister2Submitter.submitContainerJob(twister2Job, config);

  }
}

