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

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

import edu.iu.dsc.tws.common.config.Config;
import edu.iu.dsc.tws.comms.api.DataFlowOperation;
import edu.iu.dsc.tws.comms.api.MessageReceiver;
import edu.iu.dsc.tws.comms.api.MessageType;
import edu.iu.dsc.tws.comms.api.TWSChannel;
import edu.iu.dsc.tws.comms.core.TaskPlan;
import edu.iu.dsc.tws.comms.dfw.DataFlowBroadcast;
import edu.iu.dsc.tws.executor.api.AbstractParallelOperation;
import edu.iu.dsc.tws.executor.api.EdgeGenerator;
import edu.iu.dsc.tws.executor.util.Utils;
import edu.iu.dsc.tws.task.api.IMessage;
import edu.iu.dsc.tws.task.api.TaskMessage;

//TODO BBroadcast class must be implemented

public class BroadcastBatchOperation extends AbstractParallelOperation {
  private static final Logger LOG = Logger.getLogger(BroadcastBatchOperation.class.getName());
  private DataFlowBroadcast op;

  public BroadcastBatchOperation(Config config, TWSChannel network, TaskPlan tPlan) {
    super(config, network, tPlan);
  }

  public void prepare(int srcs, Set<Integer> dests, EdgeGenerator e,
                      MessageType dataType, String edgeName) {
    this.edge = e;
    LOG.info(String.format("Srcs %d dests %s", srcs, dests));
    op = new DataFlowBroadcast(channel, srcs, dests, new BcastReceiver());
    communicationEdge = e.generate(edgeName);
    LOG.info("===Communication Edge : " + communicationEdge);
    op.init(config, Utils.dataTypeToMessageType(dataType), taskPlan, communicationEdge);
  }

  @Override
  public boolean send(int source, IMessage message, int flags) {
    return op.send(source, message.getContent(), flags);
  }

  @Override
  public void send(int source, IMessage message, int dest, int flags) {
    op.send(source, message.getContent(), flags, dest);
  }

  @Override
  public boolean progress() {
    return op.progress() && hasPending();
  }

  public boolean hasPending() {
    return !op.isComplete();
  }

  public class BcastReceiver implements MessageReceiver {
    @Override
    public void init(Config cfg, DataFlowOperation operation,
                     Map<Integer, List<Integer>> expectedIds) {
    }

    @Override
    public boolean onMessage(int source, int destination, int target, int flags, Object object) {
      TaskMessage msg = new TaskMessage(object,
          edge.getStringMapping(communicationEdge), target);
      int remainingCap = outMessages.get(target).remainingCapacity();
      //LOG.info("Remaining Capacity : " + remainingCap);
      boolean status = outMessages.get(target).offer(msg);
      /*LOG.info("Message from Communication : " + msg.getContent() + ", Status : "
          + status + ", Rem Cap : " + remainingCap);*/
      return true;
    }

    @Override
    public boolean progress() {
      return true;
    }
  }
}
