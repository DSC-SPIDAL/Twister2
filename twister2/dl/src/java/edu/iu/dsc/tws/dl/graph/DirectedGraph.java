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
package edu.iu.dsc.tws.dl.graph;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;

import edu.iu.dsc.tws.dl.utils.Util;


public class DirectedGraph<K> implements Serializable {

  // source node of the directed graph
  private Node<K> source;

  //use the original direction or the reversed direction
  private boolean reverse = false;

  public DirectedGraph(Node<K> source, boolean reverse) {
    this.source = source;
    this.reverse = reverse;
  }

  public Node<K> getSource() {
    return source;
  }

  public void setSource(Node<K> source) {
    this.source = source;
  }

  /**
   * How many nodes in the graph
   *
   * @return
   */
  int size() {
    int i = 0;
    Iterator it = BFS();
    while (it.hasNext()) {
      i++;
      it.next();
    }
    return i;
  }

  /**
   * How many edges in the graph
   *
   * @return
   */
  int edges() {
    //: Int = BFS.map(_.nextNodes.length).reduce(_ + _)
    return 0;
  }

  /**
   * Topology sort.
   *
   * @return A sequence of sorted graph nodes
   */
  public List<Node<K>> topologySort() {
    // Build indegree list, LinkedHashMap can preserve the order of the keys, so it's good to
    // write unittest.
    LinkedHashMap<Node<K>, Integer> inDegrees = new LinkedHashMap<Node<K>, Integer>();
    inDegrees.put(source, 0);
    DFS().forEachRemaining(n -> {
      List<Node<K>> nextNodes = (!reverse) ? n.nextNodes() : n.prevNodes();
      nextNodes.forEach(m -> {
        inDegrees.put(m, inDegrees.getOrDefault(m, 0) + 1);
      });
    });

    List<Node<K>> result = new ArrayList<>();
    while (!inDegrees.isEmpty()) {
      // toArray is not lazy eval, which is not affected by inDegrees - 1 operations below
      List<Node<K>> startNodes = new ArrayList<>();
      for (Node<K> key : inDegrees.keySet()) {
        if (inDegrees.get(key) == 0) {
          startNodes.add(key);
        }
      }
      Util.require(startNodes.size() != 0, "There's a cycle in the graph");
      result.addAll(startNodes);
      startNodes.forEach(n -> {
        List<Node<K>> nextNodes = (!reverse) ? n.nextNodes() : n.prevNodes();
        for (Node<K> nextNode : nextNodes) {
          inDegrees.put(nextNode, inDegrees.get(nextNode) - 1);
        }
        inDegrees.remove(n);
      });
    }
    return result;
  }

  // scalastyle:off methodName

  /**
   * Depth first search on the graph. Note that this is a directed DFS. Although eachs node
   * contains both previous and next nodes, only one direction is used.
   *
   * @return An iterator to go through nodes in the graph in a DFS order
   */
  public Iterator<Node<K>> DFS() {
    return new DFS<K>(source, reverse);
  }

  public List<Node<K>> DFSList() {
    List<Node<K>> dfs = new ArrayList<Node<K>>();
    DFS().forEachRemaining(dfs::add);
    return dfs;
  }

  /**
   * Breadth first search on the graph. Note that this is a directed BFS. Although eachs node
   * contains both previous and next nodes, only one direction is used.
   *
   * @return An iterator to go through nodes in the graph in a BFS order
   */
  public Iterator<Node<K>> BFS() {
    return new BFS<K>(source, reverse);
  }

  public List<Node<K>> BFSList() {
    List<Node<K>> bfs = new ArrayList<Node<K>>();
    DFS().forEachRemaining(bfs::add);
    return bfs;
  }

  // scalastyle:on methodName

  /**
   * Clone the graph structure, will not clone the node element
   *
   * @param reverseEdge if reverse the edge in the nodes
   * @return
   */
  public DirectedGraph<K> cloneGraph(boolean reverseEdge) {
    HashMap<Node<K>, Node<K>> oldToNew = new HashMap<>();
    List<Node<K>> bfs = new ArrayList<Node<K>>();
    BFS().forEachRemaining(bfs::add);

    bfs.forEach(kNode -> {
      oldToNew.put(kNode, new Node(kNode.getElement()));
    });
    // Keep the order in the nextNodes array and prevNodes array of the current node.
    // As we go through all node in bfs from source, the prevNodes order can be preserved.
    // For each node, we iterate and add their nextNodes, the nextNodes order can also be preserved.
    bfs.forEach(node -> {
      if (reverseEdge) {
        node.nextNodesAndEdges().forEach(nextNodeAndEdge -> {
          // Some next nodes may be not included in the graph
          if (oldToNew.containsKey(nextNodeAndEdge.getValue0())) {
            oldToNew.get(node).addPrevious(
                oldToNew.get(nextNodeAndEdge.getValue0()), nextNodeAndEdge.getValue1());
          }
        });
        node.prevNodesAndEdges().forEach(prevNodeAndEdge -> {
          if (oldToNew.containsKey(prevNodeAndEdge.getValue0())) {
            oldToNew.get(node).addNexts(
                oldToNew.get(prevNodeAndEdge.getValue0()), prevNodeAndEdge.getValue1());
          }
        });
      } else {
        node.nextNodesAndEdges().forEach(nextNodeAndEdge -> {
          if (oldToNew.containsKey(nextNodeAndEdge.getValue0())) {
            oldToNew.get(node).add(oldToNew.get(nextNodeAndEdge.getValue0()),
                nextNodeAndEdge.getValue1());
          }
        });
      }
    });

    if (reverseEdge) {
      return new DirectedGraph<K>(oldToNew.get(source), !reverse);
    } else {
      return new DirectedGraph<K>(oldToNew.get(source), reverse);
    }
  }
}
