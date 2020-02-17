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

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.comms.BaseOperation;
import edu.iu.dsc.tws.api.comms.BulkReceiver;
import edu.iu.dsc.tws.api.comms.CommunicationContext;
import edu.iu.dsc.tws.api.comms.Communicator;
import edu.iu.dsc.tws.api.comms.DataFlowOperation;
import edu.iu.dsc.tws.api.comms.DestinationSelector;
import edu.iu.dsc.tws.api.comms.LogicalPlan;
import edu.iu.dsc.tws.api.comms.channel.TWSChannel;
import edu.iu.dsc.tws.api.comms.messaging.MessageFlags;
import edu.iu.dsc.tws.api.comms.messaging.MessageReceiver;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageTypes;
import edu.iu.dsc.tws.api.comms.packing.MessageSchema;
import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.comms.dfw.MToNChain;
import edu.iu.dsc.tws.comms.dfw.MToNSimple;
import edu.iu.dsc.tws.comms.dfw.io.join.BulkReceiverWrapper;
import edu.iu.dsc.tws.comms.dfw.io.join.JoinBatchCombinedReceiver;
import edu.iu.dsc.tws.comms.dfw.io.join.JoinPartitionBatchFinalReceiver;
import edu.iu.dsc.tws.comms.dfw.io.partition.DPartitionBatchFinalReceiver;
import edu.iu.dsc.tws.comms.dfw.io.partition.PartitionPartialReceiver;
import edu.iu.dsc.tws.comms.utils.JoinRelation;
import edu.iu.dsc.tws.comms.utils.KeyComparatorWrapper;
import edu.iu.dsc.tws.comms.utils.LogicalPlanBuilder;

/**
 * Batch Join operation
 */
public class BJoin extends BaseOperation {

  private static final Logger LOG = Logger.getLogger(BJoin.class.getName());

  /**
   * Left partition of the join
   */
  private DataFlowOperation partitionLeft;

  /**
   * Right partition of the join
   */
  private DataFlowOperation partitionRight;

  /**
   * Destination selector
   */
  private DestinationSelector destinationSelector;

  /**
   * THe channel
   */
  private TWSChannel channel;

  /*BARRIER RELATED FLAGS*/
  private byte[] currentBarrier = null;
  private boolean leftBarrierSent;
  private boolean rightBarrierSent;
  /*END OF BARRIER RELATED FLAGS*/

  private Set<Integer> leftSources;
  private Set<Integer> rightSources;

  /**
   * Construct a Batch partition operation
   *
   * @param comm the communicator
   * @param plan task plan
   * @param targets target tasks
   * @param rcvr receiver
   * @param leftDataType data type
   * @param destSelector destination selector
   */
  public BJoin(Communicator comm, LogicalPlan plan,
               Set<Integer> leftSources, Set<Integer> rightSources, Set<Integer> targets,
               MessageType keyType,
               MessageType leftDataType, MessageType rightDataType, BulkReceiver rcvr,
               DestinationSelector destSelector, boolean useDisk,
               Comparator<Object> comparator, int leftEdgeId, int rightEdgeId,
               CommunicationContext.JoinType joinType,
               CommunicationContext.JoinAlgorithm joinAlgorithm,
               MessageSchema leftSchema, MessageSchema rightSchema) {
    super(comm, false, CommunicationContext.JOIN);

    Map<String, Object> newConfigs = new HashMap<>();
    newConfigs.put(CommunicationContext.STREAMING, false);
    newConfigs.put(CommunicationContext.OPERATION_NAME, CommunicationContext.JOIN);
    comm.updateConfig(newConfigs);

    this.leftSources = leftSources;
    this.rightSources = rightSources;

    this.destinationSelector = destSelector;
    this.channel = comm.getChannel();

    JoinBatchCombinedReceiver commonReceiver = new JoinBatchCombinedReceiver(rcvr,
        joinAlgorithm,
        joinType,
        new KeyComparatorWrapper(comparator),
        keyType);

    MessageReceiver leftFinalReceiver;
    MessageReceiver rightFinalReceiver;
    MessageType leftRecvDataType = leftDataType;
    MessageType rightRecvDataType = rightDataType;
    if (useDisk) {
      leftFinalReceiver = new DPartitionBatchFinalReceiver(
          BulkReceiverWrapper.wrap(commonReceiver, JoinRelation.LEFT),
          comm.getPersistentDirectories(),
          joinAlgorithm.equals(CommunicationContext.JoinAlgorithm.SORT) ? comparator : null,
          false
      );
      rightFinalReceiver = new DPartitionBatchFinalReceiver(
          BulkReceiverWrapper.wrap(commonReceiver, JoinRelation.RIGHT),
          comm.getPersistentDirectories(),
          joinAlgorithm.equals(CommunicationContext.JoinAlgorithm.SORT) ? comparator : null,
          false
      );
      leftRecvDataType = MessageTypes.BYTE_ARRAY;
      rightRecvDataType = MessageTypes.BYTE_ARRAY;
    } else {
      leftFinalReceiver = new JoinPartitionBatchFinalReceiver(
          commonReceiver, JoinRelation.LEFT
      );
      rightFinalReceiver = new JoinPartitionBatchFinalReceiver(
          commonReceiver, JoinRelation.RIGHT
      );
    }

    Set<Integer> allSources = new HashSet<>(leftSources);
    allSources.addAll(rightSources);

    if (CommunicationContext.ALLTOALL_ALGO_SIMPLE.equals(
        CommunicationContext.partitionAlgorithm(comm.getConfig()))) {
      this.partitionLeft = new MToNSimple(comm.getConfig(), comm.getChannel(), plan,
          leftSources, targets, leftFinalReceiver,
          new PartitionPartialReceiver(), leftDataType, leftRecvDataType,
          keyType, keyType, leftEdgeId, leftSchema);
      this.partitionRight = new MToNSimple(comm.getConfig(),
          comm.getChannel(), plan, rightSources, targets,
          rightFinalReceiver, new PartitionPartialReceiver(),
          rightDataType, rightRecvDataType, keyType, keyType, rightEdgeId, rightSchema);
    } else {
      this.partitionLeft = new MToNChain(comm.getConfig(), comm.getChannel(), plan,
          leftSources, targets,
          leftFinalReceiver, new PartitionPartialReceiver(),
          leftDataType, leftRecvDataType, keyType, keyType, leftEdgeId, leftSchema);
      this.partitionRight = new MToNChain(comm.getConfig(), comm.getChannel(), plan,
          rightSources, targets,
          rightFinalReceiver, new PartitionPartialReceiver(),
          rightDataType, rightRecvDataType, keyType, keyType, rightEdgeId, rightSchema);
    }
    this.destinationSelector.prepare(comm, allSources, targets, keyType, null);
  }

  public BJoin(Communicator comm, LogicalPlan plan,
               Set<Integer> leftSources, Set<Integer> rightSources, Set<Integer> targets,
               MessageType keyType,
               MessageType leftDataType, MessageType rightDataType, BulkReceiver rcvr,
               DestinationSelector destSelector, boolean shuffle,
               Comparator<Object> comparator,
               CommunicationContext.JoinType joinType,
               CommunicationContext.JoinAlgorithm joinAlgorithm,
               MessageSchema leftSchema, MessageSchema rightSchema) {
    this(comm, plan, leftSources, rightSources, targets, keyType, leftDataType, rightDataType,
        rcvr, destSelector, shuffle, comparator, comm.nextEdge(), comm.nextEdge(),
        joinType, joinAlgorithm, leftSchema, rightSchema);
  }

  public BJoin(Communicator comm, LogicalPlan plan,
               Set<Integer> leftSources, Set<Integer> rightSources, Set<Integer> targets,
               MessageType keyType,
               MessageType leftDataType, MessageType rightDataType, BulkReceiver rcvr,
               DestinationSelector destSelector, boolean shuffle,
               Comparator<Object> comparator,
               CommunicationContext.JoinType joinType,
               CommunicationContext.JoinAlgorithm joinAlgorithm) {
    this(comm, plan, leftSources, rightSources, targets, keyType, leftDataType, rightDataType,
        rcvr, destSelector, shuffle, comparator, comm.nextEdge(), comm.nextEdge(),
        joinType, joinAlgorithm, MessageSchema.noSchema(), MessageSchema.noSchema());
  }

  public BJoin(Communicator comm, LogicalPlanBuilder logicalPlanBuilder, MessageType keyType,
               MessageType leftDataType, MessageType rightDataType, BulkReceiver rcvr,
               DestinationSelector destSelector, boolean shuffle,
               Comparator<Object> comparator,
               CommunicationContext.JoinType joinType,
               CommunicationContext.JoinAlgorithm joinAlgorithm) {
    this(comm, logicalPlanBuilder.build(), logicalPlanBuilder.getSources(),
        logicalPlanBuilder.getSources(),
        logicalPlanBuilder.getTargets(), keyType, leftDataType, rightDataType,
        rcvr, destSelector, shuffle, comparator, comm.nextEdge(), comm.nextEdge(),
        joinType, joinAlgorithm, MessageSchema.noSchema(), MessageSchema.noSchema());
  }

  public BJoin(Communicator comm, LogicalPlanBuilder logicalPlanBuilder,
               Set<Integer> leftSources, Set<Integer> rightSource,
               MessageType keyType,
               MessageType leftDataType, MessageType rightDataType, BulkReceiver rcvr,
               DestinationSelector destSelector, boolean shuffle,
               Comparator<Object> comparator,
               CommunicationContext.JoinType joinType,
               CommunicationContext.JoinAlgorithm joinAlgorithm) {
    this(comm, logicalPlanBuilder.build(), leftSources,
        rightSource,
        logicalPlanBuilder.getTargets(), keyType, leftDataType, rightDataType,
        rcvr, destSelector, shuffle, comparator, comm.nextEdge(), comm.nextEdge(),
        joinType, joinAlgorithm, MessageSchema.noSchema(), MessageSchema.noSchema());
  }

  /**
   * Send a data to be partitioned
   *
   * @param source source
   * @param key key for the data
   * @param data data
   * @param flags data flag
   * @return true if the data is accepted
   */
  public boolean join(int source, Object key, Object data, int flags, int tag) {
    int dest = destinationSelector.next(source, key, data);

    boolean send;
    if (tag == 0) {
      Tuple message = new Tuple<>(key, data);
      send = partitionLeft.send(source, message, flags, dest);
    } else if (tag == 1) {
      Tuple message = new Tuple<>(key, data);
      send = partitionRight.send(source, message, flags, dest);
    } else {
      throw new RuntimeException("Tag value must be either 0(left) or 1(right) for join operation");
    }

    if (send) {
      destinationSelector.commit(source, dest);
    }
    return send;
  }

  /**
   * Weather we have messages pending
   *
   * @return true if there are messages pending
   */
  @Override
  public boolean isComplete() {
    return partitionLeft.isComplete() && partitionRight.isComplete();
  }

  /**
   * Indicate the end of the communication for a given tag value
   *
   * @param source the source that is ending
   */
  @Override
  public void finish(int source) {
    if (leftSources.contains(source)) {
      partitionLeft.finish(source);
    } else {
      partitionRight.finish(source);
    }
  }

  /**
   * Progress the operation, if not called, messages will not be processed
   *
   * @return true if further progress is needed
   */
  @Override
  public boolean progress() {
    return partitionLeft.progress() | partitionRight.progress();
  }

  @Override
  /**
   * Close the operation
   */
  public void close() {
    partitionLeft.close();
    partitionRight.close();
  }

  /**
   * Clean the operation, this doesn't close it
   */
  @Override
  public void reset() {
    partitionLeft.reset();
    partitionRight.reset();
  }

  @Override
  public boolean sendBarrier(int src, byte[] barrierId) {
    if (Arrays.equals(this.currentBarrier, barrierId)) {
      // this is a retry
      if (!this.leftBarrierSent) {
        this.leftBarrierSent = partitionLeft.send(src, barrierId, MessageFlags.SYNC_BARRIER);
      }

      if (!this.rightBarrierSent) {
        this.rightBarrierSent = partitionRight.send(src, barrierId, MessageFlags.SYNC_BARRIER);
      }
    } else {
      this.leftBarrierSent = partitionLeft.send(src, barrierId, MessageFlags.SYNC_BARRIER);
      this.rightBarrierSent = partitionRight.send(src, barrierId, MessageFlags.SYNC_BARRIER);
    }

    boolean bothBarriersSent = this.leftBarrierSent && this.rightBarrierSent;

    if (bothBarriersSent) {
      //reset everything
      this.currentBarrier = null;
      this.leftBarrierSent = false;
      this.rightBarrierSent = false;
    } else {
      this.currentBarrier = barrierId;
    }

    return bothBarriersSent;
  }

  /**
   * Progress the channel and the operation
   *
   * @return true if further progress is required
   */
  @Override
  public boolean progressChannel() {
    boolean p = progress();
    channel.progress();
    return p;
  }
}
