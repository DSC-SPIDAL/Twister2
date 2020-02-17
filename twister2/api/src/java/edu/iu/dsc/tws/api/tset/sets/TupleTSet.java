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
package edu.iu.dsc.tws.api.tset.sets;

import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;
import edu.iu.dsc.tws.api.tset.TBase;
import edu.iu.dsc.tws.api.tset.fn.PartitionFunc;
import edu.iu.dsc.tws.api.tset.link.TLink;
import edu.iu.dsc.tws.api.tset.schema.TupleSchema;

/**
 * Twister data set for keyed data. This would abstract the Task level keyed computations in a
 * more user friendly API. A {@link TupleTSet} will be followed by a Keyed {@link TLink} that
 * would expose keyed communication operations.
 *
 * Note the extensions of this interface
 * {@link edu.iu.dsc.tws.api.tset.sets.batch.BatchTupleTSet} and
 * {@link edu.iu.dsc.tws.api.tset.sets.streaming.StreamingTupleTSet}. These would intimately
 * separate out the operations based on the {@link edu.iu.dsc.tws.api.compute.graph.OperationMode}
 * of the data flow graph.
 *
 * This interface only specifies the common operations for Batch and Streaming operations.
 *
 * @param <K> key type
 * @param <V> value type
 */
public interface TupleTSet<K, V> extends TBase {
  /**
   * Set the name of the set
   *
   * @param name name
   * @return this set
   */
  TupleTSet<K, V> setName(String name);

  /**
   * Partitions data using a {@link PartitionFunc} based on keys
   *
   * @param partitionFn partition function
   * @return Keyed Partition TLink
   */
  TLink<?, ?> keyedPartition(PartitionFunc<K> partitionFn);

  /**
   * Direct/pipe communication
   *
   * @return Keyed Direct TLink
   */
  TLink<?, ?> keyedDirect();

  /**
   * Sets the data type of the {@link TupleTSet} output. This will be used in the packers for efficient
   * SER-DE operations in the following {@link TLink}s
   *
   * @param schema data type as a {@link MessageType}
   * @return this {@link TupleTSet}
   */
  TupleTSet<K, V> withSchema(TupleSchema schema);
}
