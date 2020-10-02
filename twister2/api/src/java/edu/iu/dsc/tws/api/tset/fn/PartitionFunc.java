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

package edu.iu.dsc.tws.api.tset.fn;

import java.io.Serializable;
import java.util.Set;

import edu.iu.dsc.tws.api.compute.TaskPartitioner;
// todo: remove task partitioner and partition function duplicate interfaces! check destination
//  selector

/**
 * Given a data, give the partition index
 *
 * @param <T> the type of data
 */
public interface PartitionFunc<T> extends Serializable, TaskPartitioner<T> {

  /**
   * Prepare the function
   *
   * @param sources      source
   * @param destinations destinations
   */
  default void prepare(Set<Integer> sources, Set<Integer> destinations) {
  }

  /**
   * Computes the partition for the given key.
   *
   * @param val value.
   * @return The partition index.
   */
  int partition(int sourceIndex, T val);

  /**
   * Commit the partition
   *
   * @param source    the source
   * @param partition partition
   */
  default void commit(int source, int partition) {
  }
}
