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
package edu.iu.dsc.tws.comms.dfw.io.partition;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.comms.BulkReceiver;
import edu.iu.dsc.tws.api.comms.CommunicationContext;
import edu.iu.dsc.tws.api.comms.DataFlowOperation;
import edu.iu.dsc.tws.api.comms.messaging.MessageFlags;
import edu.iu.dsc.tws.api.comms.messaging.MessageReceiver;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageTypes;
import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.comms.dfw.io.DFWIOUtils;
import edu.iu.dsc.tws.comms.dfw.io.ReceiverState;
import edu.iu.dsc.tws.comms.shuffle.FSKeyedMerger;
import edu.iu.dsc.tws.comms.shuffle.FSKeyedSortedMerger2;
import edu.iu.dsc.tws.comms.shuffle.FSMerger;
import edu.iu.dsc.tws.comms.shuffle.Shuffle;

import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;

/**
 * A receiver that goes to disk
 */
public class DPartitionBatchFinalReceiver implements MessageReceiver {
  private static final Logger LOG = Logger.getLogger(DPartitionBatchFinalReceiver.class.getName());

  /**
   * The receiver
   */
  private BulkReceiver bulkReceiver;

  /**
   * Sort mergers for each target
   */
  private Int2ObjectOpenHashMap<Shuffle> sortedMergers = new Int2ObjectOpenHashMap<>();

  /**
   * Comparator for sorting records
   */
  private Comparator<Object> comparator;

  /**
   * The operation
   */
  private DataFlowOperation partition;

  /**
   * Weather a keyed operation is used
   */
  private boolean keyed;

  /**
   * The worker id
   */
  private int thisWorker = 0;

  /**
   * Finished workers per target (target -> finished workers)
   */
  private Int2ObjectOpenHashMap<Set<Integer>> finishedSources = new Int2ObjectOpenHashMap<>();

  /**
   * After all the sources finished for a target we add to this set
   */
  private List<Integer> finishedTargets = new ArrayList<>();

  /**
   * We add to this set after calling receive
   */
  private Set<Integer> finishedTargetsCompleted = new HashSet<>();

  /**
   * Weather everyone finished
   */
  private Set<Integer> targets = new HashSet<>();

  /**
   * The directory in which we will be saving the shuffle objects
   */
  private List<String> shuffleDirectories;
  private boolean groupByKey;

  /**
   * Keep a refresh count to make the directories when refreshed
   */
  private int refresh = 0;

  /**
   * The expected ids
   */
  private Map<Integer, List<Integer>> expIds;

  /**
   * Keep state about the targets
   */
  protected Int2ObjectOpenHashMap<ReceiverState> targetStates = new Int2ObjectOpenHashMap<>();

  /**
   * We use a target array to iterator
   */
  private int[] targetsArray;

  private Lock lock = new ReentrantLock();

  /**
   * Weather we are complete
   */
  private boolean complete = false;

  public DPartitionBatchFinalReceiver(BulkReceiver receiver,
                                      List<String> shuffleDirs,
                                      Comparator<Object> com,
                                      boolean groupByKey) {
    this.bulkReceiver = receiver;
    this.comparator = com;
    this.shuffleDirectories = shuffleDirs;
    this.groupByKey = groupByKey;
  }

  public void init(Config cfg, DataFlowOperation op, Map<Integer, List<Integer>> expectedIds) {
    long maxBytesInMemory = CommunicationContext.getShuffleMaxBytesInMemory(cfg);
    long maxRecordsInMemory = CommunicationContext.getShuffleMaxRecordsInMemory(cfg);
    long maxFileSize = CommunicationContext.getShuffleFileSize(cfg);
    int parallelIOAllowance = CommunicationContext.getParallelIOAllowance(cfg);

    expIds = expectedIds;
    thisWorker = op.getLogicalPlan().getThisWorker();
    finishedSources = new Int2ObjectOpenHashMap<>();
    partition = op;
    keyed = partition.getKeyType() != null;
    targets = new HashSet<>(expectedIds.keySet());
    initMergers(maxBytesInMemory, maxRecordsInMemory, maxFileSize, parallelIOAllowance);
    this.bulkReceiver.init(cfg, expectedIds.keySet());

    int index = 0;
    targetsArray = new int[expectedIds.keySet().size()];
    for (Integer target : expectedIds.keySet()) {
      targetStates.put(target, ReceiverState.INIT);
      targetsArray[index++] = target;
    }
  }

  /**
   * Initialize the mergers, this happens after each refresh
   */
  private void initMergers(long maxBytesInMemory, long maxRecordsInMemory, long maxFileSize,
                           int parallelIOAllowance) {
    for (Integer target : expIds.keySet()) {
      String shuffleDirectory = this.shuffleDirectories.get(
          partition.getLogicalPlan().getIndexOfTaskInNode(target) % this.shuffleDirectories.size());
      Shuffle sortedMerger;
      if (partition.getKeyType() == null) {
        sortedMerger = new FSMerger(maxBytesInMemory, maxRecordsInMemory, shuffleDirectory,
            DFWIOUtils.getOperationName(target, partition, refresh), partition.getDataType());
      } else {
        if (comparator != null) {
          sortedMerger = new FSKeyedSortedMerger2(maxBytesInMemory, maxFileSize,
              shuffleDirectory, DFWIOUtils.getOperationName(target, partition, refresh),
              partition.getKeyType(), partition.getDataType(), comparator, target,
              groupByKey, parallelIOAllowance);
        } else {
          sortedMerger = new FSKeyedMerger(maxBytesInMemory, maxRecordsInMemory, shuffleDirectory,
              DFWIOUtils.getOperationName(target, partition, refresh), partition.getKeyType(),
              partition.getDataType());
        }
      }
      sortedMergers.put(target, sortedMerger);
      finishedSources.put(target, new HashSet<>());
    }
  }

  @Override
  @SuppressWarnings("unchecked")
  public boolean onMessage(int source, int path,
                           int target, int flags, Object object) {
    if (lock.tryLock()) {
      try {
        Shuffle sortedMerger = sortedMergers.get(target);
        if (sortedMerger == null) {
          throw new RuntimeException("Un-expected target id: " + target);
        }

        if (targetStates.get(target) == ReceiverState.INIT) {
          targetStates.put(target, ReceiverState.RECEIVING);
        }

        if ((flags & MessageFlags.SYNC_EMPTY) == MessageFlags.SYNC_EMPTY) {
          Set<Integer> finished = finishedSources.get(target);
          if (finished.contains(source)) {
            LOG.log(Level.FINE,
                String.format("%d Duplicate finish from source id %d -> %d",
                    this.thisWorker, source, target));
          } else {
            finished.add(source);
          }
          if (finished.size() == partition.getSources().size()) {
            if (!finishedTargets.contains(target)) {
              finishedTargets.add(target);
            }
            targetStates.put(target, ReceiverState.ALL_SYNCS_RECEIVED);
          }
          return true;
        }

        if (targetStates.get(target) == ReceiverState.ALL_SYNCS_RECEIVED
            || targetStates.get(target) == ReceiverState.SYNCED) {
          return false;
        }

        // add the object to the map
        if (keyed) {
          List<Tuple> tuples = (List<Tuple>) object;
          for (Tuple kc : tuples) {
            Object data = kc.getValue();
            byte[] d;
            if (partition.getReceiveDataType() != MessageTypes.BYTE_ARRAY
                || !(data instanceof byte[])
                || ((flags & MessageFlags.ORIGIN_SENDER) == MessageFlags.ORIGIN_SENDER
                && partition.getDataType() == MessageTypes.OBJECT)) {
              // 3rd case handles, when user use Object data type, but send a byte[]
              d = partition.getDataType().getDataPacker().packToByteArray(data);
              kc.setValue(d);
            }
            sortedMerger.add(kc);
          }
        } else {
          List<Object> contents = (List<Object>) object;
          for (Object kc : contents) {
            byte[] d;
            if (partition.getReceiveDataType() != MessageTypes.BYTE_ARRAY
                || !(kc instanceof byte[])
                || ((flags & MessageFlags.ORIGIN_SENDER) == MessageFlags.ORIGIN_SENDER
                && partition.getDataType() == MessageTypes.OBJECT)) {
              d = partition.getDataType().getDataPacker().packToByteArray(kc);
            } else {
              d = (byte[]) kc;
            }
            sortedMerger.add(d, d.length);
          }
        }
        return true;
      } finally {
        lock.unlock();
      }
    }
    return false;
  }

  @Override
  public boolean progress() {
    if (lock.tryLock()) {
      try {
        boolean needFurtherProgress = false;
        for (int i = 0; i < targetsArray.length; i++) {
          int target = targetsArray[i];
          Shuffle sorts = sortedMergers.get(target);
          sorts.run();

          ReceiverState state = targetStates.get(target);
          if (state != ReceiverState.SYNCED) {
            needFurtherProgress = true;
          }
        }

        if (!needFurtherProgress) {
          return needFurtherProgress;
        }

        for (int i = 0; i < finishedTargets.size(); i++) {
          int target = finishedTargets.get(i);
          if (!finishedTargetsCompleted.contains(target) && partition.isDelegateComplete()) {
            finishTarget(target);
            targetStates.put(target, ReceiverState.SYNCED);
            onSyncEvent(target, null);
            finishedTargetsCompleted.add(target);
          }
        }
      } finally {
        lock.unlock();
      }
    }
    complete = finishedTargetsCompleted.size() == targets.size();
    return !complete;
  }

  @Override
  public boolean isComplete() {
    return complete;
  }

  private void finishTarget(int target) {
    Shuffle sortedMerger = sortedMergers.get(target);
    sortedMerger.switchToReading();
    Iterator<Object> itr = sortedMerger.readIterator();
    bulkReceiver.receive(target, itr);
  }

  private void onSyncEvent(int target, byte[] value) {
    bulkReceiver.sync(target, value);
  }

  @Override
  public void close() {
    for (Shuffle s : sortedMergers.values()) {
      s.clean();
    }
  }

  @Override
  public void clean() {
    for (Shuffle s : sortedMergers.values()) {
      s.clean();
    }
    finishedTargetsCompleted.clear();
    finishedTargets.clear();
    finishedSources.forEach((k, v) -> v.clear());

    for (int taraget : targetStates.keySet()) {
      targetStates.put(taraget, ReceiverState.INIT);
    }
    complete = false;
    refresh++;
  }
}
