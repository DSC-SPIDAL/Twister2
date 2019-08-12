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
package edu.iu.dsc.tws.executor.comms.batch;

import java.util.Iterator;
import java.util.Set;
import java.util.concurrent.BlockingQueue;

import edu.iu.dsc.tws.api.comms.BulkReceiver;
import edu.iu.dsc.tws.api.comms.Communicator;
import edu.iu.dsc.tws.api.comms.LogicalPlan;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.task.IMessage;
import edu.iu.dsc.tws.api.task.TaskMessage;
import edu.iu.dsc.tws.api.task.graph.Edge;
import edu.iu.dsc.tws.comms.batch.BAllGather;
import edu.iu.dsc.tws.executor.comms.AbstractParallelOperation;

public class AllGatherBatchOperation extends AbstractParallelOperation {

  protected BAllGather op;

  public AllGatherBatchOperation(Config config, Communicator network, LogicalPlan tPlan,
                                 Set<Integer> sources, Set<Integer> dest, Edge edge) {
    super(config, network, tPlan, edge.getName());

    if (sources.size() == 0) {
      throw new IllegalArgumentException("Sources should have more than 0 elements");
    }

    if (dest.size() == 0) {
      throw new IllegalArgumentException("Targets should have more than 0 elements");
    }

    Communicator newComm = channel.newWithConfig(edge.getProperties());
    op = new BAllGather(newComm, logicalPlan, sources, dest,
        new FinalGatherReceive(), edge.getDataType(), edge.getEdgeID().nextId(),
        edge.getEdgeID().nextId(), edge.getMessageSchema());
  }

  @Override
  public boolean send(int source, IMessage message, int flags) {
    return op.gather(source, message.getContent(), flags);
  }

  @Override
  public boolean progress() {
    return op.progress() || !op.isComplete();
  }

  @Override
  public boolean isComplete() {
    return op.isComplete();
  }

  @Override
  public void finish(int source) {
    op.finish(source);
  }

  private class FinalGatherReceive implements BulkReceiver {
    @Override
    public void init(Config cfg, Set<Integer> targets) {

    }

    @Override
    public boolean receive(int target, Iterator<Object> it) {
      TaskMessage msg = new TaskMessage<>(it, inEdge, target);
      BlockingQueue<IMessage> messages = outMessages.get(target);
      if (messages != null) {
        if (messages.offer(msg)) {
          return true;
        }
      }
      return true;
    }

    @Override
    public boolean sync(int target, byte[] message) {
      return syncs.get(target).sync(inEdge, message);
    }
  }

  @Override
  public void close() {
    op.close();
  }

  @Override
  public void reset() {
    op.reset();
  }
}
