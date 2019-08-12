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
package edu.iu.dsc.tws.comms.stream;

import java.util.Set;

import edu.iu.dsc.tws.api.comms.CommunicationContext;
import edu.iu.dsc.tws.api.comms.Communicator;
import edu.iu.dsc.tws.api.comms.DestinationSelector;
import edu.iu.dsc.tws.api.comms.LogicalPlan;
import edu.iu.dsc.tws.api.comms.ReduceFunction;
import edu.iu.dsc.tws.api.comms.SingularReceiver;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;
import edu.iu.dsc.tws.api.comms.packing.MessageSchema;
import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.comms.dfw.BaseOperation;
import edu.iu.dsc.tws.comms.dfw.MToNRing;
import edu.iu.dsc.tws.comms.dfw.MToNSimple;
import edu.iu.dsc.tws.comms.dfw.io.reduce.keyed.KReduceBatchPartialReceiver;
import edu.iu.dsc.tws.comms.dfw.io.reduce.keyed.KReduceStreamingFinalReceiver;

/**
 * Streaming Keyed Partition Operation
 */
public class SKeyedReduce extends BaseOperation {
  /**
   * Destination selector
   */
  private DestinationSelector destinationSelector;

  /**
   * Key type
   */
  private MessageType keyType;

  /**
   * Data type
   */
  private MessageType dataType;

  /**
   * Construct a Streaming Key based partition operation
   *
   * @param comm the communicator
   * @param plan task plan
   * @param sources source tasks
   * @param targets target tasks
   * @param dType data type
   * @param kType key type
   * @param fnc reduce function
   * @param rcvr receiver
   * @param destSelector destination selector
   */
  public SKeyedReduce(Communicator comm, LogicalPlan plan,
                      Set<Integer> sources, Set<Integer> targets, MessageType kType,
                      MessageType dType, ReduceFunction fnc, SingularReceiver rcvr,
                      DestinationSelector destSelector, int edgeId,
                      MessageSchema messageSchema) {
    super(comm.getChannel());
    this.keyType = kType;
    this.dataType = dType;

    if (CommunicationContext.TWISTER2_PARTITION_ALGO_SIMPLE.equals(
        CommunicationContext.partitionStreamAlgorithm(comm.getConfig()))) {
      this.op = new MToNSimple(comm.getConfig(), comm.getChannel(),
          plan, sources, targets,
          new KReduceStreamingFinalReceiver(fnc, rcvr, 100),
          new KReduceBatchPartialReceiver(0, fnc), dataType, dataType,
          keyType, keyType, edgeId, messageSchema);
    } else if (CommunicationContext.TWISTER2_PARTITION_ALGO_RING.equals(
        CommunicationContext.partitionStreamAlgorithm(comm.getConfig()))) {
      this.op = new MToNRing(comm.getConfig(), comm.getChannel(),
          plan, sources, targets, new KReduceStreamingFinalReceiver(fnc, rcvr, 100),
          new KReduceBatchPartialReceiver(0, fnc),
          dataType, dataType, keyType, keyType, edgeId, messageSchema);
    }
    this.destinationSelector = destSelector;
    this.destinationSelector.prepare(comm, sources, targets);
  }

  public SKeyedReduce(Communicator comm, LogicalPlan plan,
                      Set<Integer> sources, Set<Integer> targets, MessageType kType,
                      MessageType dType, ReduceFunction fnc, SingularReceiver rcvr,
                      DestinationSelector destSelector) {
    this(comm, plan, sources, targets, kType, dType, fnc, rcvr, destSelector,
        comm.nextEdge(), MessageSchema.noSchema());
  }

  public SKeyedReduce(Communicator comm, LogicalPlan plan,
                      Set<Integer> sources, Set<Integer> targets, MessageType kType,
                      MessageType dType, ReduceFunction fnc, SingularReceiver rcvr,
                      DestinationSelector destSelector, MessageSchema messageSchema) {
    this(comm, plan, sources, targets, kType, dType, fnc, rcvr, destSelector,
        comm.nextEdge(), messageSchema);
  }

  /**
   * Send a message to be reduced
   *
   * @param src source
   * @param key key
   * @param message message
   * @param flags message flag
   * @return true if the message is accepted
   */
  public boolean reduce(int src, Object key, Object message, int flags) {
    int dest = destinationSelector.next(src, key, message);
    return op.send(src, new Tuple(key, message, keyType, dataType), flags, dest);
  }
}
