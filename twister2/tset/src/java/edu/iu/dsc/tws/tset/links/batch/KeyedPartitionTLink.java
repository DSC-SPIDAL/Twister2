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

package edu.iu.dsc.tws.tset.links.batch;

import edu.iu.dsc.tws.api.comms.CommunicationContext;
import edu.iu.dsc.tws.api.compute.OperationNames;
import edu.iu.dsc.tws.api.compute.graph.Edge;
import edu.iu.dsc.tws.api.tset.fn.PartitionFunc;
import edu.iu.dsc.tws.api.tset.schema.TupleSchema;
import edu.iu.dsc.tws.tset.env.BatchTSetEnvironment;
import edu.iu.dsc.tws.tset.links.TLinkUtils;

public class KeyedPartitionTLink<K, V> extends KeyedBatchIteratorLinkWrapper<K, V> {
  private PartitionFunc<K> partitionFunction;

  private boolean useDisk = false;

  public KeyedPartitionTLink(BatchTSetEnvironment tSetEnv, PartitionFunc<K> parFn,
                             int sourceParallelism, TupleSchema schema) {
    super(tSetEnv, "kpartition", sourceParallelism, schema);
    this.partitionFunction = parFn;
  }

  @Override
  public Edge getEdge() {
    Edge e = new Edge(getId(), OperationNames.KEYED_PARTITION, this.getSchema().getDataType());
    e.setKeyed(true);
    e.setKeyType(this.getSchema().getKeyType());
    e.setPartitioner(partitionFunction);
    e.addProperty(CommunicationContext.USE_DISK, this.useDisk);
    TLinkUtils.generateKeyedCommsSchema(getSchema(), e);
    return e;
  }

  @Override
  public KeyedPartitionTLink<K, V> setName(String n) {
    rename(n);
    return this;
  }

  public KeyedPartitionTLink<K, V> useDisk() {
    this.useDisk = true;
    return this;
  }

//  @Override
//  public CachedTSet<Tuple<K, V>> lazyCache() {
//    return (CachedTSet<Tuple<K, V>>) super.lazyCache();
//  }
//
//  @Override
//  public CachedTSet<Tuple<K, V>> cache() {
//    return (CachedTSet<Tuple<K, V>>) super.cache();
//  }
}
