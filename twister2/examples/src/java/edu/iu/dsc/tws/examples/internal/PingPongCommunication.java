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
package edu.iu.dsc.tws.examples.internal;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.common.config.Config;
import edu.iu.dsc.tws.common.discovery.IWorkerController;
import edu.iu.dsc.tws.common.resource.AllocatedResources;
import edu.iu.dsc.tws.common.worker.IPersistentVolume;
import edu.iu.dsc.tws.common.worker.IVolatileVolume;
import edu.iu.dsc.tws.common.worker.IWorker;
import edu.iu.dsc.tws.comms.api.DataFlowOperation;
import edu.iu.dsc.tws.comms.api.MessageReceiver;
import edu.iu.dsc.tws.comms.api.MessageType;
import edu.iu.dsc.tws.comms.core.TWSCommunication;
import edu.iu.dsc.tws.comms.core.TWSNetwork;
import edu.iu.dsc.tws.comms.core.TaskPlan;
import edu.iu.dsc.tws.examples.IntData;
import edu.iu.dsc.tws.examples.Utils;

public class PingPongCommunication implements IWorker {
  private static final Logger LOG = Logger.getLogger(PingPongCommunication.class.getName());

  private DataFlowOperation direct;
  private Status status;

  /**
   * Initialize the container
   */
  public void execute(Config cfg, int workerID, AllocatedResources resources,
                      IWorkerController workerController,
                      IPersistentVolume persistentVolume,
                      IVolatileVolume volatileVolume) {
    LOG.log(Level.INFO, "Starting the example with container id: " + resources.getWorkerId());

    this.status = Status.INIT;

    // lets create the task plan
    TaskPlan taskPlan = Utils.createTaskPlan(cfg, resources);
    //first get the communication config file
    TWSNetwork network = new TWSNetwork(cfg, taskPlan);

    TWSCommunication channel = network.getDataFlowTWSCommunication();

    // we are sending messages from 0th task to 1st task
    Set<Integer> sources = new HashSet<>();
    sources.add(0);
    int dests = 1;
    Map<String, Object> newCfg = new HashMap<>();

    LOG.info("Setting up reduce dataflow operation");
    // this method calls the execute method
    // I think this is wrong
    direct = channel.direct(newCfg, MessageType.OBJECT, 0, sources,
        dests, new PingPongReceive());

    if (workerID == 0) {
      // the map thread where data is produced
      Thread mapThread = new Thread(new MapWorker());

      LOG.log(Level.INFO, "Starting map thread");
      mapThread.start();

      // we need to communicationProgress the communication
      while (true) {
        // communicationProgress the channel
        channel.progress();
        // we should communicationProgress the communication directive
        direct.progress();
        Thread.yield();
      }
    } else if (workerID == 1) {
      while (status != Status.LOAD_RECEIVE_FINISHED) {
        channel.progress();
        direct.progress();
      }
    }
  }

  /**
   * Generate data with an integer array
   *
   * @return IntData
   */
  private IntData generateData() {
    int[] d = new int[10];
    for (int i = 0; i < 10; i++) {
      d[i] = i;
    }
    return new IntData(d);
  }

  private enum Status {
    INIT,
    MAP_FINISHED,
    LOAD_RECEIVE_FINISHED,
  }

  private class PingPongReceive implements MessageReceiver {
    private int count = 0;

    @Override
    public void init(Config cfg, DataFlowOperation op, Map<Integer, List<Integer>> expectedIds) {
    }

    @Override
    public boolean onMessage(int source, int path, int target, int flags, Object object) {
      count++;
      if (count % 10000 == 0) {
        LOG.info("received message: " + count);
      }
      if (count == 100000) {
        status = Status.LOAD_RECEIVE_FINISHED;
      }
      return true;
    }

    @Override
    public boolean progress() {
      return true;
    }
  }

  /**
   * We are running the map in a separate thread
   */
  private class MapWorker implements Runnable {
    private int sendCount = 0;

    @Override
    public void run() {
      LOG.log(Level.INFO, "Starting map worker");
      for (int i = 0; i < 100000; i++) {
        IntData data = generateData();
        // lets generate a message
        while (!direct.send(0, data, 0)) {
          // lets wait a litte and try again
          try {
            Thread.sleep(1);
          } catch (InterruptedException e) {
            e.printStackTrace();
          }
        }
        sendCount++;
        Thread.yield();
      }
      status = Status.MAP_FINISHED;
    }
  }

}
