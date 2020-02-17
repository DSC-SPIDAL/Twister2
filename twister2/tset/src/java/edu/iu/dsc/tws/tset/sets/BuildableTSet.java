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

package edu.iu.dsc.tws.tset.sets;

import java.util.Collection;

import edu.iu.dsc.tws.api.compute.nodes.ICompute;
import edu.iu.dsc.tws.api.compute.nodes.INode;
import edu.iu.dsc.tws.api.compute.nodes.ISource;
import edu.iu.dsc.tws.api.tset.TBase;
import edu.iu.dsc.tws.api.tset.TSetConstants;
import edu.iu.dsc.tws.task.graph.GraphBuilder;
import edu.iu.dsc.tws.tset.Buildable;

public interface BuildableTSet extends TBase, Buildable {

  int getParallelism();

  INode getINode();

  @Override
  default void build(GraphBuilder graphBuilder, Collection<? extends TBase> buildSequence) {
    if (getINode() instanceof ICompute) {
      graphBuilder.addTask(getId(), (ICompute) getINode(), getParallelism());
    } else if (getINode() instanceof ISource) {
      graphBuilder.addSource(getId(), (ISource) getINode(), getParallelism());
    } else {
      throw new RuntimeException("Unknown INode " + getINode());
    }

    graphBuilder.addConfiguration(getId(), TSetConstants.INPUT_SCHEMA_KEY,
        ((BaseTSetWithSchema) this).getInputSchema());
    graphBuilder.addConfiguration(getId(), TSetConstants.OUTPUT_SCHEMA_KEY,
        ((BaseTSetWithSchema) this).getOutputSchema());
  }

}
