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
package edu.iu.dsc.tws.comms.batch;

import java.util.Set;

import edu.iu.dsc.tws.api.comms.BulkReceiver;
import edu.iu.dsc.tws.api.comms.Communicator;
import edu.iu.dsc.tws.api.comms.LogicalPlan;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;
import edu.iu.dsc.tws.api.comms.packing.MessageSchema;
import edu.iu.dsc.tws.comms.dfw.BaseOperation;
import edu.iu.dsc.tws.comms.dfw.TreeBroadcast;
import edu.iu.dsc.tws.comms.dfw.io.direct.DirectBatchFinalReceiver;

public class BBroadcast extends BaseOperation {
  /**
   * Construct a Streaming Reduce operation
   *
   * @param comm the communicator
   * @param plan task plan
   * @param sources source tasks
   * @param target target tasks
   * @param rcvr receiver
   * @param dataType data type
   */
  public BBroadcast(Communicator comm, LogicalPlan plan,
                    int sources, Set<Integer> target,
                    BulkReceiver rcvr, MessageType dataType, int edgeID,
                    MessageSchema messageSchema) {
    super(comm.getChannel());
    TreeBroadcast bcast = new TreeBroadcast(comm.getChannel(), sources, target,
        new DirectBatchFinalReceiver(rcvr), messageSchema);
    bcast.init(comm.getConfig(), dataType, plan, edgeID);
    op = bcast;
  }

  /**
   * Construct a Streaming Reduce operation
   *
   * @param comm the communicator
   * @param plan task plan
   * @param sources source tasks
   * @param target target tasks
   * @param rcvr receiver
   * @param dataType data type
   */
  public BBroadcast(Communicator comm, LogicalPlan plan,
                    int sources, Set<Integer> target,
                    BulkReceiver rcvr, MessageType dataType) {
    this(comm, plan, sources, target, rcvr, dataType, comm.nextEdge(), MessageSchema.noSchema());
  }

  public BBroadcast(Communicator comm, LogicalPlan plan,
                    int sources, Set<Integer> target,
                    BulkReceiver rcvr, MessageType dataType, MessageSchema messageSchema) {
    this(comm, plan, sources, target, rcvr, dataType, comm.nextEdge(), messageSchema);
  }

  /**
   * Send a message to be reduced
   *
   * @param src source
   * @param message message
   * @param flags message flag
   * @return true if the message is accepted
   */
  public boolean bcast(int src, Object message, int flags) {
    return op.send(src, message, flags);
  }
}
