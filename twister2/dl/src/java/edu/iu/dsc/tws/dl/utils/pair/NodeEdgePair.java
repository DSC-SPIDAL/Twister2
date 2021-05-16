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
package edu.iu.dsc.tws.dl.utils.pair;

import java.io.Serializable;
import java.util.Objects;

import edu.iu.dsc.tws.dl.graph.Edge;
import edu.iu.dsc.tws.dl.graph.Node;

public class NodeEdgePair<K> implements Serializable {
  private Node<K> t0;
  private Edge t1;

  public NodeEdgePair(Node<K> t0, Edge t1) {
    this.t0 = t0;
    this.t1 = t1;
  }

  public Node<K> getValue0() {
    return t0;
  }

  public Edge getValue1() {
    return t1;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    NodeEdgePair<?> that = (NodeEdgePair<?>) o;
    return Objects.equals(t0, that.t0)
        && Objects.equals(t1, that.t1);
  }

  @Override
  public int hashCode() {
    return Objects.hash(t0, t1);
  }
}
