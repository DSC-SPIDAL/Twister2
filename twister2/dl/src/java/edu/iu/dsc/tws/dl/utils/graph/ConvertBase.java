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
package edu.iu.dsc.tws.dl.utils.graph;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.iu.dsc.tws.dl.graph.Node;
import edu.iu.dsc.tws.dl.utils.pair.NodeEdgePair;

public abstract class ConvertBase<T, D> {
  /**
   * clone node relations
   *
   * @param nodeMap node element maps from T to D
   */
  public void cloneNode(List<Node<T>> allNodes, Map<Node<T>, Node<D>> nodeMap) {

    for (Node<T> node : allNodes) {

      for (NodeEdgePair<T> nextNodesAndEdge : node.nextNodesAndEdges()) {
        if (nodeMap.containsKey(nextNodesAndEdge.getValue0())) {
          nodeMap.get(node).add(nodeMap.get(nextNodesAndEdge.getValue0()),
              nextNodesAndEdge.getValue1());
        }
      }
    }

    // sort previous node
    for (Map.Entry<Node<T>, Node<D>> node : nodeMap.entrySet()) {
      // if node has more than one previous nodes, we have to consider nodes order
      if (node.getKey().prevNodesAndEdges().size() > 1) {
        node.getValue().removePrevEdges();
        node.getKey().prevNodesAndEdges().forEach(prevNodeAndEdge -> {
          if (nodeMap.containsKey(prevNodeAndEdge.getValue0())) {
            node.getValue().from(nodeMap.get(prevNodeAndEdge.getValue0()),
                prevNodeAndEdge.getValue1());
          }
        });
      }
    }
  }

  public abstract boolean convertLayerCheck(T layer);

  public abstract D convertLayer(T layer);

  public boolean convertingCheck(List<Node<T>> allNodes) {
    boolean convert = true;
    for (Node<T> node : allNodes) {
      if (!convertLayerCheck(node.getElement())) {
        //LOG.info(node.getElement() + " convertion failed");
        convert = false;
      }
    }
    return convert;
  }

  public Map<Node<T>, Node<D>> convert(List<Node<T>> allNodes) {
    Map<Node<T>, Node<D>> nodeMap = new HashMap<>();
    for (Node<T> node : allNodes) {
      nodeMap.put(node, new Node(convertLayer(node.getElement())));
    }
    cloneNode(allNodes, nodeMap);
    return nodeMap;
  }
}

