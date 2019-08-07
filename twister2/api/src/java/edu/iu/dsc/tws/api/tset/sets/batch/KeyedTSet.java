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

package edu.iu.dsc.tws.api.tset.sets.batch;

import edu.iu.dsc.tws.api.task.nodes.ICompute;
import edu.iu.dsc.tws.api.tset.TSetUtils;
import edu.iu.dsc.tws.api.tset.env.BatchTSetEnvironment;
import edu.iu.dsc.tws.api.tset.fn.PartitionFunc;
import edu.iu.dsc.tws.api.tset.fn.ReduceFunc;
import edu.iu.dsc.tws.api.tset.link.batch.KeyedGatherTLink;
import edu.iu.dsc.tws.api.tset.link.batch.KeyedPartitionTLink;
import edu.iu.dsc.tws.api.tset.link.batch.KeyedReduceTLink;
import edu.iu.dsc.tws.api.tset.ops.BaseComputeOp;
import edu.iu.dsc.tws.api.tset.ops.MapToTupleIterOp;
import edu.iu.dsc.tws.api.tset.ops.MapToTupleOp;
import edu.iu.dsc.tws.api.tset.sets.BaseTSet;

/**
 * Attaches a key to the oncoming data.
 *
 * @param <K> key type
 * @param <V> data (value) type
 */
public class KeyedTSet<K, V, T> extends BaseTSet<V> implements BatchTupleTSet<K, V, T> {
  private BaseComputeOp<?> mapToTupleOp;

  public KeyedTSet(BatchTSetEnvironment tSetEnv,
                   MapToTupleIterOp<K, V, T> genTupleOp, int parallelism) {
    super(tSetEnv, TSetUtils.generateName("keyed"), parallelism);
    this.mapToTupleOp = genTupleOp;
  }

  public KeyedTSet(BatchTSetEnvironment tSetEnv,
                   MapToTupleOp<K, V, T> genTupleOp, int parallelism) {
    super(tSetEnv, TSetUtils.generateName("keyed"), parallelism);
    this.mapToTupleOp = genTupleOp;
  }

  // since keyed tset is the base impl for BatchTupleTSet, it needs to override the env getter
  @Override
  public BatchTSetEnvironment getTSetEnv() {
    return (BatchTSetEnvironment) super.getTSetEnv();
  }

  @Override
  public KeyedReduceTLink<K, V> keyedReduce(ReduceFunc<V> reduceFn) {
    KeyedReduceTLink<K, V> reduce = new KeyedReduceTLink<>(getTSetEnv(), reduceFn,
        getParallelism());
    addChildToGraph(reduce);
    return reduce;
  }

  @Override
  public KeyedPartitionTLink<K, V> keyedPartition(PartitionFunc<K> partitionFn) {
    KeyedPartitionTLink<K, V> partition = new KeyedPartitionTLink<>(getTSetEnv(), partitionFn,
        getParallelism());
    addChildToGraph(partition);
    return partition;
  }

  @Override
  public KeyedGatherTLink<K, V> keyedGather() {
    KeyedGatherTLink<K, V> gather = new KeyedGatherTLink<>(getTSetEnv(), getParallelism());
    addChildToGraph(gather);
    return gather;
  }


  @Override
  public KeyedTSet<K, V, T> setName(String n) {
    rename(n);
    return this;
  }

  @Override
  public ICompute getINode() {
    return mapToTupleOp;
  }
}