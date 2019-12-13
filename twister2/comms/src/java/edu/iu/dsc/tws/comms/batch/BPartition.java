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

import java.util.List;
import java.util.Set;

import edu.iu.dsc.tws.api.comms.BaseOperation;
import edu.iu.dsc.tws.api.comms.BulkReceiver;
import edu.iu.dsc.tws.api.comms.CommunicationContext;
import edu.iu.dsc.tws.api.comms.Communicator;
import edu.iu.dsc.tws.api.comms.DestinationSelector;
import edu.iu.dsc.tws.api.comms.LogicalPlan;
import edu.iu.dsc.tws.api.comms.messaging.MessageReceiver;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;
import edu.iu.dsc.tws.api.comms.packing.MessageSchema;
import edu.iu.dsc.tws.comms.dfw.MToNRing;
import edu.iu.dsc.tws.comms.dfw.MToNSimple;
import edu.iu.dsc.tws.comms.dfw.io.partition.DPartitionBatchFinalReceiver;
import edu.iu.dsc.tws.comms.dfw.io.partition.PartitionBatchFinalReceiver;
import edu.iu.dsc.tws.comms.dfw.io.partition.PartitionPartialReceiver;
import edu.iu.dsc.tws.comms.utils.LogicalPlanBuilder;

/**
 * Streaming Partition Operation
 */
public class BPartition extends BaseOperation {
  /**
   * Destination selector
   */
  private DestinationSelector destinationSelector;

  /**
   * Construct a Batch partition operation
   *
   * @param comm the communicator
   * @param plan task plan
   * @param sources source tasks
   * @param targets target tasks
   * @param rcvr receiver
   * @param dataType data type
   * @param destSelector destination selector
   */
  public BPartition(Communicator comm, LogicalPlan plan,
                    Set<Integer> sources, Set<Integer> targets, MessageType dataType,
                    BulkReceiver rcvr,
                    DestinationSelector destSelector, boolean shuffle,
                    int edgeId, MessageSchema messageSchema) {
    super(comm, false, CommunicationContext.PARTITION);
    this.destinationSelector = destSelector;
    List<String> shuffleDirs = comm.getPersistentDirectories();

    MessageReceiver finalRcvr;
    if (shuffle) {
      finalRcvr = new DPartitionBatchFinalReceiver(
          rcvr, shuffleDirs, null, true);
    } else {
      finalRcvr = new PartitionBatchFinalReceiver(rcvr);
    }

    if (CommunicationContext.ALLTOALL_ALGO_SIMPLE.equals(
        CommunicationContext.partitionAlgorithm(comm.getConfig()))) {
      MToNSimple p = new MToNSimple(comm.getChannel(), sources, targets,
          finalRcvr, new PartitionPartialReceiver(), dataType, messageSchema);
      p.init(comm.getConfig(), dataType, plan, edgeId);
      this.op = p;
    } else if (CommunicationContext.ALLTOALL_ALGO_RING.equals(
        CommunicationContext.partitionAlgorithm(comm.getConfig()))) {
      this.op = new MToNRing(comm.getConfig(), comm.getChannel(),
          plan, sources, targets, finalRcvr, new PartitionPartialReceiver(),
          dataType, dataType, null, null, edgeId, messageSchema);
    }
    this.destinationSelector.prepare(comm, op.getSources(), op.getTargets(), null, dataType);
  }

  public BPartition(Communicator comm, LogicalPlan plan,
                    Set<Integer> sources, Set<Integer> targets, MessageType dataType,
                    BulkReceiver rcvr,
                    DestinationSelector destSelector, boolean shuffle,
                    MessageSchema messageSchema) {
    this(comm, plan, sources, targets, dataType, rcvr, destSelector, shuffle,
        comm.nextEdge(), messageSchema);
  }

  public BPartition(Communicator comm, LogicalPlan plan,
                    Set<Integer> sources, Set<Integer> targets, MessageType dataType,
                    BulkReceiver rcvr,
                    DestinationSelector destSelector, boolean shuffle) {
    this(comm, plan, sources, targets, dataType, rcvr, destSelector, shuffle,
        comm.nextEdge(), MessageSchema.noSchema());
  }

  public BPartition(Communicator comm, LogicalPlanBuilder logicalPlanBuilder, MessageType dataType,
                    BulkReceiver rcvr,
                    DestinationSelector destSelector, boolean shuffle) {
    this(comm, logicalPlanBuilder.build(),
        logicalPlanBuilder.getSources(), logicalPlanBuilder.getTargets(),
        dataType, rcvr, destSelector, shuffle,
        comm.nextEdge(), MessageSchema.noSchema());
  }

  /**
   * Send a message to be partitioned
   *
   * @param source source
   * @param message message
   * @param flags message flag
   * @return true if the message is accepted
   */
  public boolean partition(int source, Object message, int flags) {
    int dest = destinationSelector.next(source, message);

    boolean send = op.send(source, message, flags, dest);
    if (send) {
      destinationSelector.commit(source, dest);
    }
    return send;
  }
}
