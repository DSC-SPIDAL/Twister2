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


package edu.iu.dsc.tws.tset.links.streaming;

import edu.iu.dsc.tws.api.compute.OperationNames;
import edu.iu.dsc.tws.api.compute.graph.Edge;
import edu.iu.dsc.tws.api.tset.fn.PartitionFunc;
import edu.iu.dsc.tws.api.tset.schema.Schema;
import edu.iu.dsc.tws.tset.env.StreamingTSetEnvironment;
import edu.iu.dsc.tws.tset.links.TLinkUtils;

public class SPartitionTLink<T> extends StreamingSingleLink<T> {

  private PartitionFunc<T> partitionFunction;

  public SPartitionTLink(StreamingTSetEnvironment tSetEnv, int sourceParallelism,
                         Schema schema) {
    this(tSetEnv, null, sourceParallelism, schema);
  }

  public SPartitionTLink(StreamingTSetEnvironment tSetEnv, PartitionFunc<T> parFn,
                         int sourceParallelism, Schema schema) {
    this(tSetEnv, parFn, sourceParallelism, sourceParallelism, schema);
  }

  public SPartitionTLink(StreamingTSetEnvironment tSetEnv, PartitionFunc<T> parFn,
                         int sourceParallelism, int targetParallelism, Schema schema) {
    super(tSetEnv, "spartition", sourceParallelism, targetParallelism, schema);
    this.partitionFunction = parFn;
  }

  @Override
  public Edge getEdge() {
    Edge e = new Edge(getId(), OperationNames.PARTITION, this.getSchema().getDataType());
    if (partitionFunction != null) {
      e.setPartitioner(partitionFunction);
    }
    TLinkUtils.generateCommsSchema(getSchema(), e);
    return e;
  }

  @Override
  public SPartitionTLink<T> setName(String n) {
    rename(n);
    return this;
  }
}
