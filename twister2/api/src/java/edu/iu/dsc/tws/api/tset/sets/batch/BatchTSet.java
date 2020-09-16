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

import java.util.Collection;
import java.util.Iterator;

import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.api.tset.StoringData;
import edu.iu.dsc.tws.api.tset.fn.MapFunc;
import edu.iu.dsc.tws.api.tset.fn.PartitionFunc;
import edu.iu.dsc.tws.api.tset.fn.ReduceFunc;
import edu.iu.dsc.tws.api.tset.link.TLink;
import edu.iu.dsc.tws.api.tset.link.batch.BatchTLink;
import edu.iu.dsc.tws.api.tset.sets.AcceptingData;
import edu.iu.dsc.tws.api.tset.sets.StorableTBase;
import edu.iu.dsc.tws.api.tset.sets.TSet;

/**
 * Batch extension of {@link TSet} interface. Adds methods related to
 * Batch {@link edu.iu.dsc.tws.api.compute.graph.OperationMode} and overrides generic
 * {@link TSet} and {@link TLink} return type to {@link BatchTSet}
 * and {@link BatchTLink} in all methods. Also extends the {@link AcceptingData} interface
 * because the {@link BatchTSet}s can accept data from other {@link TSet}s.
 *
 * @param <T> tset type
 */
public interface BatchTSet<T> extends TSet<T>, AcceptingData<T>, StoringData<T> {

  @Override
  BatchTSet<T> setName(String name);

  @Override
  BatchTLink<Iterator<T>, T> direct();

  BatchTLink<T, T> pipe();

  @Override
  BatchTLink<T, T> reduce(ReduceFunc<T> reduceFn);

  @Override
  BatchTLink<T, T> allReduce(ReduceFunc<T> reduceFn);

  @Override
  BatchTLink<Iterator<T>, T> partition(PartitionFunc<T> partitionFn, int targetParallelism);

  @Override
  BatchTLink<Iterator<T>, T> partition(PartitionFunc<T> partitionFn);

  @Override
  BatchTLink<Iterator<Tuple<Integer, T>>, T> gather();

  @Override
  BatchTLink<Iterator<Tuple<Integer, T>>, T> allGather();

  @Override
  <K, V> BatchTupleTSet<K, V> mapToTuple(MapFunc<T, Tuple<K, V>> mapToTupleFn);

  @Override
  BatchTLink<Iterator<T>, T> replicate(int replications);

  @Override
  BatchTSet<T> union(TSet<T> unionTSet);

  @Override
  BatchTSet<T> union(Collection<TSet<T>> tSets);

  /**
   * Adds data to this TSet
   *
   * @param key   the key used to store the given TSet
   * @param input a @{@link StorableTBase} TSet to be added as an input
   * @return this test
   */
  @Override
  BatchTSet<T> addInput(String key, StorableTBase<?> input);
}
