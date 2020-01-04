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
package edu.iu.dsc.tws.comms.dfw.io.join;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.comms.BulkReceiver;
import edu.iu.dsc.tws.api.comms.CommunicationContext;
import edu.iu.dsc.tws.api.comms.DataFlowOperation;
import edu.iu.dsc.tws.api.comms.SingularReceiver;
import edu.iu.dsc.tws.api.comms.messaging.MessageReceiver;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;
import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.comms.utils.HashJoinUtils;
import edu.iu.dsc.tws.comms.utils.KeyComparatorWrapper;
import edu.iu.dsc.tws.comms.utils.SortJoinUtils;

public class JoinBatchFinalReceiver2 implements MessageReceiver {
  private static final Logger LOG = Logger.getLogger(JoinBatchFinalReceiver2.class.getName());

  /**
   * The left receiver
   */
  private JoinPartitionBatchReceiver leftReceiver;

  /**
   * The right receiver
   */
  private JoinPartitionBatchReceiver rightReceiver;

  /**
   * The user provided receiver
   */
  private BulkReceiver bulkReceiver;
  private CommunicationContext.JoinType joinType;
  private CommunicationContext.JoinAlgorithm joinAlgorithm;
  private MessageType keyType;

  /**
   * The iterators returned by left
   */
  private Map<Integer, List<Tuple>> leftValues;

  /**
   * The iterators return by right
   */
  private Map<Integer, List<Tuple>> rightValues;

  /**
   * Comparator
   */
  private KeyComparatorWrapper comparator;

  /**
   * Right has synced
   */
  private boolean rightSynced;

  /**
   * Left has synced
   */
  private boolean leftSynced;

  public JoinBatchFinalReceiver2(BulkReceiver bulkReceiver,
                                 Comparator<Object> com,
                                 CommunicationContext.JoinType joinType,
                                 CommunicationContext.JoinAlgorithm joinAlgorithm,
                                 MessageType keyType) {
    this.bulkReceiver = bulkReceiver;
    this.joinType = joinType;
    this.joinAlgorithm = joinAlgorithm;
    this.keyType = keyType;
    this.leftReceiver = new JoinPartitionBatchReceiver(new InnerBulkReceiver(0), 0);
    this.rightReceiver = new JoinPartitionBatchReceiver(new InnerBulkReceiver(1), 1);
    this.leftValues = new HashMap<>();
    this.rightValues = new HashMap<>();
    this.comparator = new KeyComparatorWrapper(com);
    this.rightSynced = false;
    this.leftSynced = false;
  }

  @Override
  public void init(Config cfg, DataFlowOperation op, Map<Integer, List<Integer>> expectedIds) {
    leftReceiver.init(cfg, op, expectedIds);
    rightReceiver.init(cfg, op, expectedIds);
    bulkReceiver.init(cfg, expectedIds.keySet());
  }

  @Override
  public boolean onMessage(int source, int path, int target, int flags, Object object) {
    throw new RuntimeException("Not implemented");
  }

  @Override
  public void close() {
    leftReceiver.clean();
    rightReceiver.clean();
  }

  @Override
  public void clean() {
    leftReceiver.clean();
    rightReceiver.clean();

    // clean the maps
    leftValues.clear();
    rightValues.clear();
  }

  public void onSyncEvent(int target, byte[] value) {
    bulkReceiver.sync(target, value);
  }

  @Override
  public boolean onMessage(int source, int path, int target, int flags, int tag, Object object) {
    if (tag == 0) {
      return leftReceiver.onMessage(source, path, target, flags, object);
    } else {
      return rightReceiver.onMessage(source, path, target, flags, object);
    }
  }

  @Override
  public boolean progress() {
    return leftReceiver.progress() | rightReceiver.progress();
  }

  private class InnerBulkReceiver implements SingularReceiver {
    private int tag;

    InnerBulkReceiver(int tag) {
      this.tag = tag;
    }

    @Override
    public void init(Config cfg, Set<Integer> targets) {
    }

    private List doJoin(int target) {
      if (joinAlgorithm.equals(CommunicationContext.JoinAlgorithm.SORT)) {
        return SortJoinUtils.join(
            leftValues.get(target),
            rightValues.get(target), comparator, joinType);
      } else {
        return HashJoinUtils.join(
            leftValues.get(target),
            rightValues.get(target),
            joinType,
            keyType
        );
      }
    }

    @Override
    public boolean receive(int target, Object it) {
      if (tag == 0) {
        leftValues.put(target, (List<Tuple>) it);

        if (rightValues.containsKey(target)) {
          List<Object> results = doJoin(target);
          bulkReceiver.receive(target, results.iterator());
        }
      } else {
        rightValues.put(target, (List<Tuple>) it);

        if (leftValues.containsKey(target)) {
          List<Object> results = doJoin(target);
          bulkReceiver.receive(target, results.iterator());
        }
      }
      return true;
    }

    @Override
    public boolean sync(int target, byte[] message) {
      if (tag == 0) {
        leftSynced = true;
      }
      if (tag == 1) {
        rightSynced = true;
      }
      return leftSynced && rightSynced && bulkReceiver.sync(target, message);
    }
  }

  @Override
  public boolean isComplete() {
    return leftReceiver.isComplete() && rightReceiver.isComplete();
  }
}
