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
package edu.iu.dsc.tws.comms.dfw;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.comms.DataFlowOperation;
import edu.iu.dsc.tws.api.comms.LogicalPlan;
import edu.iu.dsc.tws.api.comms.channel.ChannelReceiver;
import edu.iu.dsc.tws.api.comms.channel.TWSChannel;
import edu.iu.dsc.tws.api.comms.messaging.MessageFlags;
import edu.iu.dsc.tws.api.comms.messaging.MessageHeader;
import edu.iu.dsc.tws.api.comms.messaging.MessageReceiver;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;
import edu.iu.dsc.tws.api.comms.packing.MessageDeSerializer;
import edu.iu.dsc.tws.api.comms.packing.MessageSchema;
import edu.iu.dsc.tws.api.comms.packing.MessageSerializer;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.comms.dfw.io.AggregatedObjects;
import edu.iu.dsc.tws.comms.dfw.io.Deserializers;
import edu.iu.dsc.tws.comms.dfw.io.Serializers;
import edu.iu.dsc.tws.comms.utils.OperationUtils;
import edu.iu.dsc.tws.comms.utils.TaskPlanUtils;

public class MToNRing implements DataFlowOperation, ChannelReceiver {

  private static final Logger LOG = Logger.getLogger(MToNRing.class.getName());

  /**
   * Locally merged results
   */
  private Map<Integer, List<Object>> merged = new HashMap<>();

  /**
   * The data ready to be sent
   */
  private Map<Integer, List<Object>> readyToSend = new HashMap<>();

  /**
   * This is the local merger
   */
  private MessageReceiver merger;

  /**
   * Final receiver
   */
  private MessageReceiver finalReceiver;

  /**
   * The actual implementation
   */
  private ChannelDataFlowOperation delegate;

  /**
   * Lock for progressing the communication
   */
  private Lock lock = new ReentrantLock();

  /**
   * Lock for progressing the partial receiver
   */
  private Lock partialLock = new ReentrantLock();

  /**
   * The task plan
   */
  private LogicalPlan logicalPlan;

  /**
   * A map holding workerId to targets
   */
  private Map<Integer, List<Integer>> workerToTargets = new HashMap<>();

  /**
   * Targets to workers
   */
  private Map<Integer, Integer> targetsToWorkers = new HashMap<>();

  /**
   * The target routes
   */
  private Map<Integer, RoutingParameters> targetRoutes = new HashMap<>();

  /**
   * The workers for targets, sorted
   */
  private List<Integer> workers;

  /**
   * This worker id
   */
  private int thisWorker;
  private MessageSchema messageSchema;

  /**
   * The target index to send for each worker
   */
  private Map<Integer, Integer> targetIndex = new HashMap<>();

  /**
   * The next worker to send the data
   */
  private int nextWorkerIndex;

  /**
   * The data type
   */
  private MessageType dataType;

  /**
   * The key type
   */
  private MessageType keyType;

  /**
   * The sources
   */
  private Set<Integer> sources;

  /**
   * The targets
   */
  private Set<Integer> targets;

  /**
   * The source representing this
   */
  private int representSource;

  /**
   * Weather keyed operation
   */
  private boolean isKeyed;

  /**
   * Lock the data strcutures while swapping
   */
  private Lock swapLock = new ReentrantLock();

  /**
   * Sources of this worker
   */
  private Set<Integer> thisWorkerSources;

  /**
   * Keep trck of receive data type for later return
   */
  private MessageType receiveDataType;

  /**
   * Keep track of receive key type for later return
   */
  private MessageType receiveKeyType;

  /**
   * The grouping
   */
  private int groupingSize = 100;

  /**
   * Create a ring partition communication
   *
   * @param cfg configuration
   * @param channel channel
   * @param tPlan task plan
   * @param sources sources
   * @param targets targets
   * @param finalRcvr final receiver
   * @param partialRcvr partial receiver
   * @param dType data type
   * @param rcvType receive data type
   * @param kType key data type
   * @param rcvKType receive key type
   * @param edge the edge
   */
  public MToNRing(Config cfg, TWSChannel channel, LogicalPlan tPlan, Set<Integer> sources,
                  Set<Integer> targets, MessageReceiver finalRcvr,
                  MessageReceiver partialRcvr,
                  MessageType dType, MessageType rcvType,
                  MessageType kType, MessageType rcvKType, int edge,
                  MessageSchema messageSchema) {
    this.merger = partialRcvr;
    this.finalReceiver = finalRcvr;
    this.logicalPlan = tPlan;
    this.dataType = dType;
    this.keyType = kType;
    this.sources = sources;
    this.targets = targets;
    this.receiveDataType = rcvType;
    this.receiveKeyType = rcvKType;
    this.groupingSize = DataFlowContext.getNetworkPartitionBatchGroupingSize(cfg);

    // this worker
    this.thisWorker = tPlan.getThisExecutor();
    this.messageSchema = messageSchema;

    // get the tasks of this executor
    Set<Integer> targetsOfThisWorker = TaskPlanUtils.getTasksOfThisWorker(tPlan, targets);
    Set<Integer> sourcesOfThisWorker = TaskPlanUtils.getTasksOfThisWorker(tPlan, sources);
    Map<Integer, List<Integer>> mergerExpectedIds = new HashMap<>();
    for (int target : targets) {
      mergerExpectedIds.put(target, new ArrayList<>(sourcesOfThisWorker));
    }
    // initialize the merger
    merger.init(cfg, this, mergerExpectedIds);

    // final receivers ids
    Map<Integer, List<Integer>> finalExpectedIds = new HashMap<>();
    for (int target : targetsOfThisWorker) {
      finalExpectedIds.put(target, new ArrayList<>(sources));
    }
    // initialize the final receiver
    finalReceiver.init(cfg, this, finalExpectedIds);

    // now calculate the worker id to target mapping
    calculateWorkerIdToTargets();

    // calculate the workers
    workers = new ArrayList<>(workerToTargets.keySet());
    Collections.sort(workers);

    // calculate the routes
    calculateRoutingParameters();

    // lets calculate the worker as this worker
    nextWorkerIndex = workers.indexOf(thisWorker);

    // put 0 as the target index
    Set<Integer> targetWorkers = TaskPlanUtils.getWorkersOfTasks(tPlan, targets);
    for (int t : targetWorkers) {
      targetIndex.put(t, 0);
    }

    // lets set the represent source here
    if (sourcesOfThisWorker.size() > 0) {
      representSource = sourcesOfThisWorker.iterator().next();
    }

    if (keyType != null) {
      isKeyed = true;
    }

    thisWorkerSources = TaskPlanUtils.getTasksOfThisWorker(logicalPlan, sources);

    // calculate the workers from we are receiving
    Set<Integer> receiveWorkers = TaskPlanUtils.getWorkersOfTasks(tPlan, sources);
    receiveWorkers.remove(logicalPlan.getThisExecutor());

    Map<Integer, ArrayBlockingQueue<OutMessage>> pendingSendMessagesPerSource =
        new HashMap<>();
    Map<Integer, Queue<InMessage>> pendingReceiveMessagesPerSource
        = new HashMap<>();
    Map<Integer, Queue<InMessage>> pendingReceiveDeSerializations = new HashMap<>();
    Map<Integer, MessageSerializer> serializerMap = new HashMap<>();
    Map<Integer, MessageDeSerializer> deSerializerMap = new HashMap<>();

    for (int s : sources) {
      // later look at how not to allocate pairs for this each time
      pendingSendMessagesPerSource.put(s, new ArrayBlockingQueue<>(
          DataFlowContext.sendPendingMax(cfg)));
      serializerMap.put(s, Serializers.get(isKeyed, this.messageSchema));
    }

    int maxReceiveBuffers = DataFlowContext.receiveBufferCount(cfg);
    int receiveExecutorsSize = receiveWorkers.size();
    if (receiveExecutorsSize == 0) {
      receiveExecutorsSize = 1;
    }
    for (int ex : sources) {
      int capacity = maxReceiveBuffers * 2 * receiveExecutorsSize;
      pendingReceiveMessagesPerSource.put(ex, new ArrayBlockingQueue<>(capacity));
      pendingReceiveDeSerializations.put(ex, new ArrayBlockingQueue<>(capacity));
      deSerializerMap.put(ex, Deserializers.get(isKeyed, this.messageSchema));
    }
    // create the delegate
    this.delegate = new ChannelDataFlowOperation(channel);
    this.delegate.init(cfg, dataType, rcvType, kType, rcvKType, tPlan, edge, receiveWorkers,
        this, pendingSendMessagesPerSource, pendingReceiveMessagesPerSource,
        pendingReceiveDeSerializations, serializerMap, deSerializerMap, isKeyed);
  }

  private void calculateWorkerIdToTargets() {
    for (int t : targets) {
      int worker = logicalPlan.getExecutorForChannel(t);
      List<Integer> ts;
      if (workerToTargets.containsKey(worker)) {
        ts = workerToTargets.get(worker);
      } else {
        ts = new ArrayList<>();
      }
      ts.add(t);
      workerToTargets.put(worker, ts);
      targetsToWorkers.put(t, worker);
    }
  }

  private void calculateRoutingParameters() {
    this.targetRoutes = new HashMap<>();
    for (int t : targets) {
      RoutingParameters routingParameters = new RoutingParameters();

      Integer worker = targetsToWorkers.get(t);
      if (worker != thisWorker) {
        routingParameters.addExternalRoute(t);
      } else {
        routingParameters.addInteranlRoute(t);
      }
      routingParameters.setDestinationId(t);

      targetRoutes.put(t, routingParameters);
    }
  }

  @Override
  public boolean sendPartial(int source, Object message, int flags) {
    throw new UnsupportedOperationException("Operation is not supported");
  }

  @Override
  public boolean send(int source, Object message, int flags) {
    throw new UnsupportedOperationException("Operation is not supported");
  }

  @Override
  public boolean send(int source, Object message, int flags, int target) {
    partialLock.lock();
    try {
      return merger.onMessage(source, 0, target, flags, message);
    } finally {
      partialLock.unlock();
    }
  }

  @Override
  public boolean sendPartial(int source, Object message, int flags, int target) {
    swapLock.lock();
    try {
      if ((flags & MessageFlags.SYNC_EMPTY) == MessageFlags.SYNC_EMPTY) {
        List<Object> mTarget = merged.get(target);
        if ((mTarget == null || mTarget.isEmpty())
            && (readyToSend.get(target) == null || readyToSend.get(target).isEmpty())) {
          return delegate.sendMessage(source, message, target,
              MessageFlags.SYNC_EMPTY, targetRoutes.get(target));
        } else {
          return false;
        }
      }

      // we add to the merged
      List<Object> messages = merged.computeIfAbsent(target, k -> new AggregatedObjects<>());

      if (messages.size() < groupingSize) {
        if (message instanceof AggregatedObjects) {
          messages.addAll((Collection<?>) message);
        } else {
          messages.add(message);
        }
      } else {
        return false;
      }
    } finally {
      swapLock.unlock();
    }
    return true;
  }

  @Override
  public boolean progress() {
    swapLock.lock();
    try {
      int worker = workers.get(nextWorkerIndex);
      List<Integer> tgts = workerToTargets.get(worker);

      // we need to send starting from the previosu
      int i;
      int index = targetIndex.get(worker);
      for (i = index; i < tgts.size(); i++) {
        int target = tgts.get(i);
        List<Object> mergedData = merged.get(target);
        if (mergedData != null && mergedData.size() > 0) {
          swapToReady(target, mergedData);
        }

        List<Object> data = readyToSend.get(target);
        if (data != null && data.size() > 0) {
          RoutingParameters parameters = targetRoutes.get(target);
          if (!delegate.sendMessage(representSource, data, target, 0, parameters)) {
            break;
          } else {
            readyToSend.remove(target);
          }
        }
      }
      index = i;
      // if we have sent everything reset to 0 and move onto next worker index
      if (i == tgts.size()) {
        index = 0;
        incrementWorkerIndex();
      }
      targetIndex.put(worker, index);
    } finally {
      swapLock.unlock();
    }

    // now set the things
    return OperationUtils.progressReceivers(delegate, lock,
        finalReceiver, partialLock, merger);
  }

  private void incrementWorkerIndex() {
    nextWorkerIndex = (nextWorkerIndex + 1) % workers.size();
  }

  @Override
  public void close() {
    if (merged != null) {
      merger.close();
    }

    if (finalReceiver != null) {
      finalReceiver.close();
    }

    delegate.close();
  }

  @Override
  public void reset() {
    if (merged != null) {
      merger.clean();
    }

    if (finalReceiver != null) {
      finalReceiver.clean();
    }
  }

  @Override
  public LogicalPlan getLogicalPlan() {
    return logicalPlan;
  }

  @Override
  public String getUniqueId() {
    return null;
  }

  @Override
  public boolean isComplete() {
    boolean done = delegate.isComplete();
    boolean needsFurtherProgress = OperationUtils.progressReceivers(delegate, lock, finalReceiver,
        partialLock, merger);
    return done && !needsFurtherProgress;
  }

  @Override
  public MessageType getKeyType() {
    return keyType;
  }

  @Override
  public MessageType getDataType() {
    return dataType;
  }

  @Override
  public Set<Integer> getSources() {
    return sources;
  }

  @Override
  public Set<Integer> getTargets() {
    return targets;
  }

  @Override
  public boolean receiveMessage(MessageHeader header, Object object) {
    return finalReceiver.onMessage(header.getSourceId(), DataFlowContext.DEFAULT_DESTINATION,
        header.getDestinationIdentifier(), header.getFlags(), object);
  }

  @Override
  public boolean receiveSendInternally(int source, int target, int path,
                                       int flags, Object message) {
    lock.lock();
    try {
      return finalReceiver.onMessage(source, 0, target, flags, message);
    } finally {
      lock.unlock();
    }
  }

  private void swapToReady(int target, List<Object> data) {
    if (!readyToSend.containsKey(target)) {
      if (data.size() > 0) {
        readyToSend.put(target, new AggregatedObjects<>(data));
      }
    } else {
      List<Object> ready = readyToSend.get(target);
      ready.addAll(data);
    }
    data.clear();
  }

  @Override
  public boolean isDelegateComplete() {
    return delegate.isComplete();
  }

  @Override
  public void finish(int source) {
    Set<Integer> targetsOfThisWorker = TaskPlanUtils.getTasksOfThisWorker(logicalPlan, targets);
    for (int dest : targetsOfThisWorker) {
      // first we need to call finish on the partial receivers
      while (!send(source, new int[0], MessageFlags.SYNC_EMPTY, dest)) {
        // lets progress until finish
        progress();
      }
    }
  }

  @Override
  public MessageType getReceiveKeyType() {
    return receiveKeyType;
  }

  @Override
  public MessageType getReceiveDataType() {
    return receiveDataType;
  }
}
