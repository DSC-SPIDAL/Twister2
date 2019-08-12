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
package edu.iu.dsc.tws.executor.comms.streaming;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Set;
import java.util.concurrent.BlockingQueue;

import edu.iu.dsc.tws.api.comms.Communicator;
import edu.iu.dsc.tws.api.comms.LogicalPlan;
import edu.iu.dsc.tws.api.comms.SingularReceiver;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.task.IMessage;
import edu.iu.dsc.tws.api.task.TaskMessage;
import edu.iu.dsc.tws.api.task.graph.Edge;
import edu.iu.dsc.tws.comms.stream.SDirect;
import edu.iu.dsc.tws.executor.comms.AbstractParallelOperation;

public class DirectStreamingOperation extends AbstractParallelOperation {
  protected SDirect op;

  public DirectStreamingOperation(Config config, Communicator network, LogicalPlan tPlan,
                                  Set<Integer> srcs, Set<Integer> dests, Edge edge) {
    super(config, network, tPlan, edge.getName());
    if (srcs.size() == 0) {
      throw new IllegalArgumentException("Sources should have more than 0 elements");
    }

    if (dests == null) {
      throw new IllegalArgumentException("Targets should have more than 0 elements");
    }

    ArrayList<Integer> sources = new ArrayList<>(srcs);
    Collections.sort(sources);
    ArrayList<Integer> targets = new ArrayList<>(dests);
    Collections.sort(targets);

    Communicator newComm = channel.newWithConfig(edge.getProperties());
    op = new SDirect(newComm, logicalPlan, sources, targets, edge.getDataType(),
        new DirectReceiver(), edge.getEdgeID().nextId(), edge.getMessageSchema());
  }

  public boolean send(int source, IMessage message, int flags) {
    return op.partition(source, message.getContent(), flags);
  }

  public class DirectReceiver implements SingularReceiver {
    @Override
    public void init(Config cfg, Set<Integer> targets) {

    }

    @Override
    public boolean receive(int target, Object object) {
      BlockingQueue<IMessage> messages = outMessages.get(target);

      TaskMessage msg = new TaskMessage<>(object, inEdge, target);
      return messages.offer(msg);
    }

    @Override
    public boolean sync(int target, byte[] message) {
      return syncs.get(target).sync(inEdge, message);
    }
  }

  public boolean progress() {
    return op.progress();
  }

  @Override
  public void close() {
    op.close();
  }

  @Override
  public void reset() {
    op.reset();
  }

  @Override
  public boolean isComplete() {
    return op.isComplete();
  }
}
