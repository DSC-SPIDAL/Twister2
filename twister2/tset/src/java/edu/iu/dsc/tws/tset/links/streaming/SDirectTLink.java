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
import edu.iu.dsc.tws.api.tset.schema.Schema;
import edu.iu.dsc.tws.tset.env.StreamingTSetEnvironment;
import edu.iu.dsc.tws.tset.links.TLinkUtils;

public class SDirectTLink<T> extends StreamingSingleLink<T> {

  public SDirectTLink(StreamingTSetEnvironment tSetEnv, int sourceParallelism,
                      Schema schema) {
    super(tSetEnv, "sdirect", sourceParallelism, schema);
  }

  @Override
  public Edge getEdge() {
    Edge e = new Edge(getId(), OperationNames.DIRECT, this.getSchema().getDataType());
    TLinkUtils.generateCommsSchema(getSchema(), e);
    return e;
  }

  @Override
  public SDirectTLink<T> setName(String n) {
    rename(n);
    return this;
  }
}
