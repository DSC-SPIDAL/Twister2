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


package edu.iu.dsc.tws.tset.sets.streaming;

import java.util.Collections;

import edu.iu.dsc.tws.api.compute.nodes.INode;
import edu.iu.dsc.tws.api.tset.fn.SourceFunc;
import edu.iu.dsc.tws.api.tset.schema.PrimitiveSchemas;
import edu.iu.dsc.tws.api.tset.schema.Schema;
import edu.iu.dsc.tws.tset.env.StreamingTSetEnvironment;
import edu.iu.dsc.tws.tset.ops.SourceOp;

public class SSourceTSet<T> extends StreamingTSetImpl<T> {
  private SourceFunc<T> source;

  public SSourceTSet(StreamingTSetEnvironment tSetEnv, SourceFunc<T> src, int parallelism) {
    super(tSetEnv, "ssource", parallelism, PrimitiveSchemas.NULL);
    this.source = src;
  }

  public SSourceTSet(StreamingTSetEnvironment tSetEnv, String name, SourceFunc<T> src,
                     int parallelism) {
    super(tSetEnv, name, parallelism, PrimitiveSchemas.NULL);
    this.source = src;
  }

  @Override
  public SSourceTSet<T> setName(String n) {
    rename(n);
    return this;
  }

  public SSourceTSet<T> withSchema(Schema schema) {
    return (SSourceTSet<T>) super.withSchema(schema);
  }

  @Override
  public INode getINode() {
    return new SourceOp<>(source, this, Collections.emptyMap());
  }
}
