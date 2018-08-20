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
package edu.iu.dsc.tws.executor.comm.operations.streaming;

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
import edu.iu.dsc.tws.comms.op.Communicator;
import edu.iu.dsc.tws.comms.op.stream.SBroadCast;
import edu.iu.dsc.tws.executor.api.AbstractParallelOperation;
import edu.iu.dsc.tws.executor.api.EdgeGenerator;
import edu.iu.dsc.tws.task.api.IMessage;
import edu.iu.dsc.tws.task.api.TaskMessage;

public class BroadcastStreamingOperation extends AbstractParallelOperation {
  private static final Logger LOG = Logger.getLogger(BroadcastStreamingOperation.class.getName());

  private SBroadCast broadCast;
  private Communicator communicator;
  private TaskPlan taskPlan;

  public BroadcastStreamingOperation(Config config, TWSChannel network, TaskPlan tPlan) {
    super(config, network, tPlan);
    this.communicator = new Communicator(config, network);
    this.taskPlan = tPlan;
  }

  public void prepare(int source, Set<Integer> dests, EdgeGenerator e,
                      MessageType dataType, String edgeName) {
    this.broadCast = new SBroadCast(communicator, taskPlan, source, dests, dataType,
        new BcastReceiver());
  }

  @Override
  public boolean send(int source, IMessage message, int flags) {
    return broadCast.bcast(source, message.getContent(), flags);
  }

  @Override
  public void send(int source, IMessage message, int dest, int flags) {
    throw new RuntimeException("send with dest in BcastStreamOps is not Implemented.");
  }

  @Override
  public boolean progress() {
    return broadCast.progress();
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
