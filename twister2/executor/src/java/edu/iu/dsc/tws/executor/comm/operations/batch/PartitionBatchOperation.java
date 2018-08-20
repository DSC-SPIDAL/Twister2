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
package edu.iu.dsc.tws.executor.comm.operations.batch;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

import edu.iu.dsc.tws.checkpointmanager.barrier.CheckpointBarrier;
import edu.iu.dsc.tws.common.config.Config;
import edu.iu.dsc.tws.comms.api.BatchReceiver;
import edu.iu.dsc.tws.comms.api.DataFlowOperation;
import edu.iu.dsc.tws.comms.api.MessageReceiver;
import edu.iu.dsc.tws.comms.api.MessageType;
import edu.iu.dsc.tws.comms.api.TWSChannel;
import edu.iu.dsc.tws.comms.core.TaskPlan;
import edu.iu.dsc.tws.comms.op.Communicator;
import edu.iu.dsc.tws.comms.op.LoadBalanceDestinationSelector;
import edu.iu.dsc.tws.comms.op.batch.BPartition;
import edu.iu.dsc.tws.executor.api.AbstractParallelOperation;
import edu.iu.dsc.tws.executor.api.EdgeGenerator;
import edu.iu.dsc.tws.task.api.IMessage;
import edu.iu.dsc.tws.task.api.TaskMessage;

/**
 * TODO : PartionReceiver has to implemented here. The PartitionBatchReceiver must be chnaged.
 *
 *
 * **/

public class PartitionBatchOperation extends AbstractParallelOperation {
  private static final Logger LOG = Logger.getLogger(PartitionBatchOperation.class.getName());
  private HashMap<Integer, ArrayList<Integer>> barrierMap = new HashMap<>();
  private HashMap<Integer, Integer> incommingMap = new HashMap<>();
  private HashMap<Integer, ArrayList<Object>> incommingBuffer = new HashMap<>();

  protected BPartition partition;
  private TaskPlan taskPlan;
  private Communicator communicator;

  public PartitionBatchOperation(Config config, TWSChannel network, TaskPlan tPlan) {
    super(config, network, tPlan);
    this.taskPlan = tPlan;
    this.communicator = new Communicator(config, network);
  }

  public void prepare(Set<Integer> srcs, Set<Integer> dests, EdgeGenerator e,
                      MessageType dataType, String edgeName) {
    this.edge = e;
    partition = new BPartition(communicator, taskPlan, srcs, dests, dataType,
        new PartitionBatchReceiver(), new LoadBalanceDestinationSelector());

  }

  public void prepare(Set<Integer> srcs, Set<Integer> dests, EdgeGenerator e,
                      MessageType dataType, MessageType keyType, String edgeName) {
    this.edge = e;
    partition = new BPartition(communicator, taskPlan, srcs, dests, dataType,
        new PartitionBatchReceiver(), new LoadBalanceDestinationSelector());
  }

  public void send(int source, IMessage message) {
    throw new RuntimeException("send with Message not Implemented in PartitionBatchOperation");
  }

  public boolean send(int source, IMessage message, int flags) {
    return partition.partition(source, message.getContent(), flags);
  }


  public class PartitionBatchReceiver implements BatchReceiver {
    private int count = 0;
    private int expected;

    @Override
    public void init(Config cfg, DataFlowOperation op, Map<Integer, List<Integer>> expectedIds) {
    }

    @Override
    public void receive(int target, Iterator<Object> it) {

    }
  }


  public class PartitionReceiver implements MessageReceiver {
    @Override
    public void init(Config cfg, DataFlowOperation operation,
                     Map<Integer, List<Integer>> expectedIds) {

    }

    @Override
    public boolean onMessage(int source, int destination, int target, int flags, Object object) {
      if (barrierMap.containsKey(source)) {
        if (barrierMap.get(source).size() == 0) {
          barrierMap.remove(source);
          if (object instanceof List) {
            for (Object o : (List) object) {
              TaskMessage msg = new TaskMessage(o,
                  edge.getStringMapping(communicationEdge), target);
              outMessages.get(target).offer(msg);
            }
          } else if (object instanceof CheckpointBarrier) {
            barrierMap.get(source).add(destination);
            ArrayList<Object> bufferMessege = new ArrayList<>();
            bufferMessege.add(object);
            incommingBuffer.put(source, bufferMessege);
          }
        } else if (barrierMap.get(source).size() == incommingMap.get(source)) {
          for (Object message : incommingBuffer.get(source)) {
            if (message instanceof List) {
              for (Object o : (List) message) {
                TaskMessage msg = new TaskMessage(o,
                    edge.getStringMapping(communicationEdge), target);
                outMessages.get(target).offer(msg);
              }
            }
          }
        }
      } else {
        if (object instanceof List) {
          for (Object o : (List) object) {
            TaskMessage msg = new TaskMessage(o,
                edge.getStringMapping(communicationEdge), target);
            outMessages.get(target).offer(msg);

          }
        } else if (object instanceof CheckpointBarrier) {
          ArrayList<Integer> destinationMap = new ArrayList<Integer>();
          destinationMap.add(destination);
          barrierMap.put(source, destinationMap);
        }

      }
      return true;
    }

    @Override
    public void onFinish(int target) {

    }

    @Override
    public boolean progress() {
      return true;
    }
  }

  @Override
  public boolean progress() {
    return partition.progress();
  }

  public boolean hasPending() {
    return !partition.hasPending();
  }
}
