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
import edu.iu.dsc.tws.comms.api.ReduceFunction;
import edu.iu.dsc.tws.comms.api.ReduceReceiver;
import edu.iu.dsc.tws.comms.api.TWSChannel;
import edu.iu.dsc.tws.comms.core.TaskPlan;
import edu.iu.dsc.tws.comms.op.Communicator;
import edu.iu.dsc.tws.comms.op.batch.BReduce;
import edu.iu.dsc.tws.executor.api.AbstractParallelOperation;
import edu.iu.dsc.tws.executor.api.EdgeGenerator;
import edu.iu.dsc.tws.task.api.IMessage;
import edu.iu.dsc.tws.task.api.TaskMessage;

public class ReduceBatchOperation extends AbstractParallelOperation {

  private static final Logger LOG = Logger.getLogger(ReduceBatchOperation.class.getName());

  protected BReduce reduce;
  private Communicator communicator;
  private TaskPlan taskPlan;

  public ReduceBatchOperation(Config config, TWSChannel network, TaskPlan tPlan) {
    super(config, network, tPlan);
    this.communicator = new Communicator(config, network);
    this.taskPlan = tPlan;

  }

  public void prepare(Set<Integer> sources, int dest, EdgeGenerator e,
                      MessageType dataType, String edgeName) {
    this.reduce = new BReduce(communicator, taskPlan, sources, dest, new IdentityFunction(),
        new FinalReduceReceiver(), dataType);
  }

  @Override
  public boolean send(int source, IMessage message, int flags) {
    //LOG.log(Level.INFO, String.format("Message %s", message.getContent()));
    return reduce.reduce(source, message.getContent(), flags);
      // lets wait a litte and try again
  }

  @Override
  public void send(int source, IMessage message, int dest, int flags) {
    throw new RuntimeException("ReduceBatchOperation Send with dest Not Implemented ...");
  }

  @Override
  public boolean progress() {
    return reduce.progress();
  }

  public static class IdentityFunction implements ReduceFunction {
    @Override
    public void init(Config cfg, DataFlowOperation op, Map<Integer, List<Integer>> expectedIds) {
    }

    @Override
    public Object reduce(Object t1, Object t2) {
      return t1;
    }
  }


  public class FinalReduceReceiver implements ReduceReceiver, MessageReceiver {
    private int count = 0;

    @Override
    public void init(Config cfg, DataFlowOperation operation,
                     Map<Integer, List<Integer>> expectedIds) {

    }

    @Override
    public boolean onMessage(int source, int destination, int target, int flags, Object object) {
      return false;
    }

    @Override
    public void onFinish(int target) {
      reduce.finish(target);
    }

    @Override
    public boolean progress() {
      return true;
    }

    @Override
    public boolean receive(int target, Object object) {
      TaskMessage msg = new TaskMessage(object,
          edge.getStringMapping(communicationEdge), target);
      outMessages.get(target).offer(msg);
      return true;
    }
  }

}
