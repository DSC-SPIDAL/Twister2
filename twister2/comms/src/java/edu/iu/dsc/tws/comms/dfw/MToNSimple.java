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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Level;
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
import edu.iu.dsc.tws.comms.dfw.io.Deserializers;
import edu.iu.dsc.tws.comms.dfw.io.Serializers;
import edu.iu.dsc.tws.comms.routing.PartitionRouter;
import edu.iu.dsc.tws.comms.utils.OperationUtils;
import edu.iu.dsc.tws.comms.utils.TaskPlanUtils;

import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;

public class MToNSimple implements DataFlowOperation, ChannelReceiver {
  private static final Logger LOG = Logger.getLogger(MToNSimple.class.getName());

  /**
   * Sources
   */
  private Set<Integer> sources;

  /**
   * Destinations
   */
  private Set<Integer> destinations;

  /**
   * Partition router
   */
  private PartitionRouter router;

  /**
   * Final receiver
   */
  private MessageReceiver finalReceiver;

  /**
   * Partial receiver
   */
  private MessageReceiver partialReceiver;

  /**
   * The actual implementation
   */
  private ChannelDataFlowOperation delegete;

  /**
   * Task plan
   */
  private LogicalPlan instancePlan;

  /**
   * Receive message type, we can receive messages as just bytes
   */
  private MessageType receiveType;

  /**
   * Receive key type, we can receive keys as just bytes
   */
  private MessageType receiveKeyType;

  /**
   * Data type
   */
  private MessageType dataType;
  /**
   * Key type
   */
  private MessageType keyType;

  /**
   * Weather this is a key based communication
   */
  private boolean isKeyed;

  /**
   * Routing parameters are cached
   */
  private Int2ObjectOpenHashMap<Int2ObjectOpenHashMap<RoutingParameters>> routingParamCache
      = new Int2ObjectOpenHashMap<>();

  /**
   * Routing parameters are cached
   */
  private Int2ObjectOpenHashMap<RoutingParameters> partialRoutingParamCache
      = new Int2ObjectOpenHashMap<>();

  /**
   * Lock for progressing the communication
   */
  private Lock lock = new ReentrantLock();

  /**
   * Lock for progressing the partial receiver
   */
  private Lock partialLock = new ReentrantLock();

  /**
   * Edge used for communication
   */
  private int edge;

  /**
   * A place holder for keeping the internal destinations
   */
  private List<Integer> internalDestinations = new ArrayList<>();

  /**
   * A place holder for keeping the external destinations
   */
  private List<Integer> externalDestinations = new ArrayList<>();

  private MessageSchema messageSchema;

  public MToNSimple(TWSChannel channel, Set<Integer> sourceTasks, Set<Integer> destTasks,
                    MessageReceiver finalRcvr, MessageReceiver partialRcvr,
                    MessageType dataType, MessageType keyType, MessageSchema messageSchema) {
    this(channel, sourceTasks, destTasks, finalRcvr, partialRcvr, messageSchema);
    this.isKeyed = true;
    this.keyType = keyType;
    this.dataType = dataType;
    this.receiveKeyType = keyType;
    this.receiveType = dataType;
  }

  public MToNSimple(TWSChannel channel, Set<Integer> sourceTasks, Set<Integer> destTasks,
                    MessageReceiver finalRcvr, MessageReceiver partialRcvr,
                    MessageType dataType, MessageSchema messageSchema) {
    this(channel, sourceTasks, destTasks, finalRcvr, partialRcvr, messageSchema);
    this.dataType = dataType;
  }

  public MToNSimple(TWSChannel channel, Set<Integer> srcs,
                    Set<Integer> dests, MessageReceiver finalRcvr,
                    MessageReceiver partialRcvr, MessageSchema messageSchema) {
    this.sources = srcs;
    this.destinations = dests;
    this.delegete = new ChannelDataFlowOperation(channel);

    this.finalReceiver = finalRcvr;
    this.partialReceiver = partialRcvr;
    this.messageSchema = messageSchema;
  }

  public MToNSimple(Config cfg, TWSChannel channel, LogicalPlan tPlan, Set<Integer> srcs,
                    Set<Integer> dests, MessageReceiver finalRcvr,
                    MessageReceiver partialRcvr,
                    MessageType dType, MessageType rcvType,
                    int e, MessageSchema messageSchema) {
    this(cfg, channel, tPlan, srcs, dests, finalRcvr, partialRcvr, dType, rcvType,
        null, null, e, messageSchema);
    this.isKeyed = false;
  }

  public MToNSimple(Config cfg, TWSChannel channel, LogicalPlan tPlan, Set<Integer> srcs,
                    Set<Integer> dests, MessageReceiver finalRcvr,
                    MessageReceiver partialRcvr,
                    MessageType dType, MessageType rcvType,
                    MessageType kType, MessageType rcvKType,
                    int e, MessageSchema messageSchema) {
    this.instancePlan = tPlan;
    this.sources = srcs;
    this.destinations = dests;
    this.delegete = new ChannelDataFlowOperation(channel);
    this.dataType = dType;
    this.receiveType = rcvType;
    this.keyType = kType;
    this.receiveKeyType = rcvKType;
    this.edge = e;
    this.messageSchema = messageSchema;

    if (keyType != null) {
      this.isKeyed = true;
    }

    this.finalReceiver = finalRcvr;
    this.partialReceiver = partialRcvr;

    init(cfg, dType, instancePlan, edge);
  }

  /**
   * Initialize
   */
  public void init(Config cfg, MessageType t, LogicalPlan logicalPlan, int ed) {
    this.edge = ed;

    Set<Integer> thisSources = TaskPlanUtils.getTasksOfThisWorker(logicalPlan, sources);
    int executor = logicalPlan.getThisExecutor();
    LOG.log(Level.FINE, String.format("%d setup loadbalance routing %s %s",
        logicalPlan.getThisExecutor(), sources, destinations));
    this.router = new PartitionRouter(logicalPlan, sources, destinations);
    Map<Integer, Set<Integer>> internal = router.getInternalSendTasks();
    Map<Integer, Set<Integer>> external = router.getExternalSendTasks();
    this.instancePlan = logicalPlan;
    this.dataType = t;
    if (this.receiveType == null) {
      this.receiveType = dataType;
    }

    LOG.log(Level.FINE, String.format("%d adding internal/external routing",
        logicalPlan.getThisExecutor()));
    for (int s : thisSources) {
      Set<Integer> integerSetMap = internal.get(s);
      if (integerSetMap != null) {
        this.internalDestinations.addAll(integerSetMap);
      }

      Set<Integer> integerSetMap1 = external.get(s);
      if (integerSetMap1 != null) {
        this.externalDestinations.addAll(integerSetMap1);
      }
      LOG.fine(String.format("%d adding internal/external routing %d",
          logicalPlan.getThisExecutor(), s));
      break;
    }

    LOG.log(Level.FINE, String.format("%d done adding internal/external routing",
        logicalPlan.getThisExecutor()));
    this.finalReceiver.init(cfg, this, receiveExpectedTaskIds());
    this.partialReceiver.init(cfg, this, router.partialExpectedTaskIds());

    Map<Integer, ArrayBlockingQueue<OutMessage>> pendingSendMessagesPerSource =
        new HashMap<>();
    Map<Integer, Queue<InMessage>> pendingReceiveMessagesPerSource
        = new HashMap<>();
    Map<Integer, Queue<InMessage>> pendingReceiveDeSerializations = new HashMap<>();
    Map<Integer, MessageSerializer> serializerMap = new HashMap<>();
    Map<Integer, MessageDeSerializer> deSerializerMap = new HashMap<>();

    Set<Integer> srcs = TaskPlanUtils.getTasksOfThisWorker(logicalPlan, sources);
    Set<Integer> tempsrcs = TaskPlanUtils.getTasksOfThisWorker(logicalPlan, sources);

    //need to set minus tasks as well
    for (Integer src : tempsrcs) {
      srcs.add((src * -1) - 1);
    }
    for (int s : srcs) {
      // later look at how not to allocate pairs for this each time
      pendingSendMessagesPerSource.put(s, new ArrayBlockingQueue<>(
          DataFlowContext.sendPendingMax(cfg)));
      serializerMap.put(s, Serializers.get(isKeyed, this.messageSchema));
    }

    int maxReceiveBuffers = DataFlowContext.receiveBufferCount(cfg);
    int receiveExecutorsSize = receivingExecutors().size();
    if (receiveExecutorsSize == 0) {
      receiveExecutorsSize = 1;
    }
    for (int ex : sources) {
      int capacity = maxReceiveBuffers * 2 * receiveExecutorsSize;
      pendingReceiveMessagesPerSource.put(ex, new ArrayBlockingQueue<>(capacity));
      pendingReceiveDeSerializations.put(ex, new ArrayBlockingQueue<>(capacity));
      deSerializerMap.put(ex, Deserializers.get(isKeyed, this.messageSchema));
    }

    for (int src : srcs) {
      routingParamCache.put(src, new Int2ObjectOpenHashMap<>());
      for (int dest : destinations) {
        sendRoutingParameters(src, dest);
      }
    }

    delegete.init(cfg, dataType, receiveType, keyType, receiveKeyType, logicalPlan, edge,
        router.receivingExecutors(), this,
        pendingSendMessagesPerSource, pendingReceiveMessagesPerSource,
        pendingReceiveDeSerializations, serializerMap, deSerializerMap, isKeyed);
  }

  @Override
  public boolean sendPartial(int source, Object message, int flags) {
    int newFlags = flags | MessageFlags.ORIGIN_PARTIAL;
    return delegete.sendMessagePartial(source, message, 0,
        newFlags, sendPartialRoutingParameters(0));
  }

  @Override
  public boolean sendPartial(int source, Object message, int flags, int target) {
    int newFlags = flags | MessageFlags.ORIGIN_PARTIAL;
    return delegete.sendMessagePartial(source, message, target, newFlags,
        sendPartialRoutingParameters(target));
  }

  @Override
  public boolean send(int source, Object message, int flags) {
    int newFlags = flags | MessageFlags.ORIGIN_SENDER;
    return delegete.sendMessage(source, message, 0, newFlags, sendRoutingParameters(source, 0));
  }

  @Override
  public boolean send(int source, Object message, int flags, int target) {
    int newFlags = flags | MessageFlags.ORIGIN_SENDER;
    return delegete.sendMessage(source, message, target, newFlags,
        sendRoutingParameters(source, target));
  }

  public boolean isComplete() {
    boolean done = delegete.isComplete();
    boolean needsFurtherProgress = OperationUtils.progressReceivers(delegete, lock, finalReceiver,
        partialLock, partialReceiver);
    return done && !needsFurtherProgress;
  }

  public boolean isDelegateComplete() {
    return delegete.isComplete();
  }

  @Override
  public boolean progress() {
    return OperationUtils.progressReceivers(delegete, lock, finalReceiver,
        partialLock, partialReceiver);
  }

  @Override
  public void close() {
    if (partialReceiver != null) {
      partialReceiver.close();
    }

    if (finalReceiver != null) {
      finalReceiver.close();
    }

    delegete.close();
  }

  @Override
  public void reset() {
    if (partialReceiver != null) {
      partialReceiver.clean();
    }

    if (finalReceiver != null) {
      finalReceiver.clean();
    }
  }

  @Override
  public void finish(int source) {
    for (int dest : destinations) {
      // first we need to call finish on the partial receivers
      while (!send(source, new byte[0], MessageFlags.SYNC_EMPTY, dest)) {
        // lets progress until finish
        progress();
      }
    }
  }

  @Override
  public LogicalPlan getLogicalPlan() {
    return instancePlan;
  }

  @Override
  public String getUniqueId() {
    return String.valueOf(edge);
  }

  private RoutingParameters sendRoutingParameters(int source, int path) {
    Int2ObjectOpenHashMap p = routingParamCache.get(source);
    if (p.containsKey(path)) {
      return (RoutingParameters) p.get(path);
    } else {
      RoutingParameters routingParameters = new RoutingParameters();
      routingParameters.setDestinationId(path);
      routingParameters.addInteranlRoute(source);
      p.put(path, routingParameters);
      return routingParameters;
    }
  }

  private RoutingParameters sendPartialRoutingParameters(int destination) {
    if (partialRoutingParamCache.containsKey(destination)) {
      return partialRoutingParamCache.get(destination);
    } else {
      RoutingParameters routingParameters = new RoutingParameters();
      routingParameters.setDestinationId(destination);
      if (externalDestinations.contains(destination)) {
        routingParameters.addExternalRoute(destination);
      } else {
        routingParameters.addInteranlRoute(destination);
      }
      partialRoutingParamCache.put(destination, routingParameters);
      return routingParameters;
    }
  }

  public boolean receiveSendInternally(int source, int path,
                                       int destination, int flags, Object message) {
    // okay this must be for the
    if ((flags & MessageFlags.ORIGIN_PARTIAL) == MessageFlags.ORIGIN_PARTIAL) {
      return finalReceiver.onMessage(source, path, destination, flags, message);
    }
    return partialReceiver.onMessage(source, path, destination, flags, message);
  }

  protected Set<Integer> receivingExecutors() {
    return router.receivingExecutors();
  }

  protected Map<Integer, List<Integer>> receiveExpectedTaskIds() {
    return router.receiveExpectedTaskIds();
  }

  public boolean receiveMessage(MessageHeader header, Object object) {
    return finalReceiver.onMessage(header.getSourceId(), DataFlowContext.DEFAULT_DESTINATION,
        header.getDestinationIdentifier(), header.getFlags(), object);
  }

  public Set<Integer> getSources() {
    return sources;
  }

  @Override
  public MessageType getKeyType() {
    return keyType;
  }

  @Override
  public MessageType getDataType() {
    return dataType;
  }

  public MessageType getReceiveDataType() {
    return receiveType;
  }

  public MessageType getReceiveKeyType() {
    return receiveKeyType;
  }

  @Override
  public Set<Integer> getTargets() {
    return destinations;
  }
}
