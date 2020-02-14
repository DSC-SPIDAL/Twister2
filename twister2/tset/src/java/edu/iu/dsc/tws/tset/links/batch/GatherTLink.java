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
import edu.iu.dsc.tws.api.tset.schema.Schema;
import edu.iu.dsc.tws.tset.env.BatchTSetEnvironment;
import edu.iu.dsc.tws.tset.links.TLinkUtils;

/**
 * Create a gather data set
 * <p>
 * Gather comms message content signature is Iterator{@literal <Tuple<Integer, T>>}.
 * Here, the tuple is artificially created by the comms layer, and users would have very little
 * idea about it. Hence, the tuple, needs to be broken apart, and only values need to be
 * considered for compute operations like map, flatmap, foreach etc.
 * <p>
 * but, for the generic compute, Iterator{@literal <Tuple<Integer, T>>} messages are valid.
 * <p>
 * Because of this reason, we can not use, IteratorLink{@literal <Tuple<Integer, T>>}
 * <p>
 * Same logic applies to AllGather TLink
 *
 * @param <T> the type of data
 */
public class GatherTLink<T> extends BatchGatherLink<T> {
  private boolean useDisk = false;

  public GatherTLink(BatchTSetEnvironment tSetEnv, int sourceParallelism, Schema schema) {
    super(tSetEnv, "gather", sourceParallelism, 1, schema);
  }

  @Override
  public Edge getEdge() {
    Edge e = new Edge(getId(), OperationNames.GATHER, getSchema().getDataType());
    e.addProperty(CommunicationContext.USE_DISK, this.useDisk);
    TLinkUtils.generateCommsSchema(getSchema(), e);
    return e;
  }

  @Override
  public GatherTLink<T> setName(String n) {
    rename(n);
    return this;
  }

  public GatherTLink<T> useDisk() {
    this.useDisk = true;
    return this;
  }
}
