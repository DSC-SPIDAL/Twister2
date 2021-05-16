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
import java.util.List;

import edu.iu.dsc.tws.dl.utils.pair.NodeEdgePair;

/**
 * Represent a node in a graph. The connections between nodes are directed.
 */
@SuppressWarnings("NeedBraces")
public class Node<K> implements Serializable {
  private K element;


  private ArrayList<NodeEdgePair<K>> nexts = new ArrayList<NodeEdgePair<K>>();
  private ArrayList<NodeEdgePair<K>> prevs = new ArrayList<NodeEdgePair<K>>();

  public Node(K value) {
    this.element = value;
  }

  public K getElement() {
    return element;
  }

  public Node<K> setElement(K e) {
    element = e;
    return this;
  }

  public ArrayList<NodeEdgePair<K>> getNexts() {
    return nexts;
  }

  public void setNexts(ArrayList<NodeEdgePair<K>> nexts) {
    this.nexts = nexts;
  }

  public ArrayList<NodeEdgePair<K>> getPrevs() {
    return prevs;
  }

  public void setPrevs(ArrayList<NodeEdgePair<K>> prevs) {
    this.prevs = prevs;
  }

  /**
   * The nodes pointed by current node
   *
   * @return
   */
  public List<Node<K>> nextNodes() {
    List<Node<K>> result = new ArrayList<>();
    for (NodeEdgePair<K> next : nexts) {
      result.add(next.getValue0());
    }
    return result;
  }

  /**
   * The edges start from this node
   *
   * @return
   */
  public List<Edge> nextEdges() {
    List<Edge> result = new ArrayList<>();
    for (NodeEdgePair<K> next : nexts) {
      result.add(next.getValue1());
    }
    return result;
  }

  /**
   * The nodes pointed by current node with the connect edges
   *
   * @return
   */
  public List<NodeEdgePair<K>> nextNodesAndEdges() {
    return nexts;
  }

  /**
   * The nodes point to current node
   *
   * @return
   */
  public List<Node<K>> prevNodes() {
    List<Node<K>> result = new ArrayList<>();
    for (NodeEdgePair<K> pre : prevs) {
      result.add(pre.getValue0());
    }
    return result;
  }

  /**
   * The edges connect to this node
   *
   * @return
   */
  public List<Edge> prevEdges() {
    List<Edge> result = new ArrayList<>();
    for (NodeEdgePair<K> pre : prevs) {
      result.add(pre.getValue1());
    }
    return result;
  }

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket

  /**
   * The nodes pointed to current node with the connect edges
   *
   * @return
   */
  public List<NodeEdgePair<K>> prevNodesAndEdges() {
    return prevs;
  }
  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName

  /**
   * Point to another node
   *
   * @param node another node
   * @return another node
   */
  public Node<K> pointTo(Node<K> node) {
    return this.add(node, new Edge());
  }

  /**
   * Point to another node
   *
   * @param node another node
   * @return another node
   */
  public Node<K> add(Node<K> node, Edge edge) {
    Edge e = edge;
    if (e == null) {
      e = new Edge();
    }
    if (!node.prevs.contains(new NodeEdgePair<K>(this, e)))
      node.prevs.add(new NodeEdgePair<K>(this, e));
    if (!this.nexts.contains(new NodeEdgePair<K>(node, e)))
      this.nexts.add(new NodeEdgePair<K>(node, e));
    return node;
  }

  public void addPrevious(Node<K> node, Edge edge) {
    Edge e = edge;
    if (e == null) {
      e = new Edge();
    }
    if (!this.prevs.contains(new NodeEdgePair<K>(node, e)))
      this.prevs.add(new NodeEdgePair<K>(node, e));
  }

  public void addNexts(Node<K> node, Edge edge) {
    Edge e = edge;
    if (e == null) {
      e = new Edge();
    }
    if (!this.nexts.contains(new NodeEdgePair<K>(node, e)))
      this.nexts.add(new NodeEdgePair<K>(node, e));
  }

  public Node<K> from(Node<K> node, Edge edge) {
    Edge e = edge;
    if (e == null) {
      e = new Edge();
    }
    if (!node.nexts.contains(new NodeEdgePair<K>(this, e)))
      node.nexts.add(new NodeEdgePair<K>(this, e));
    if (!this.prevs.contains(new NodeEdgePair<K>(node, e)))
      this.prevs.add(new NodeEdgePair<K>(node, e));
    return node;
  }

//  /**
//   * A sugar allows user to generate the pair (n, something) via n(something)
//   * @param meta
//   * @tparam M
//   * @return
//   */
//  def apply[M](meta: M): (this.type, M) = {
//    (this, meta)
//  }

  /**
   * Remove linkage with another node
   *
   * @param node another node
   * @return current node
   */
  Node<K> delete(Node<K> node, Edge e) {
    if (e != null) {
      if (node.prevs.contains(new NodeEdgePair<K>(this, e)))
        node.prevs.remove(new NodeEdgePair<K>(this, e));
      if (this.nexts.contains(new NodeEdgePair<K>(node, e)))
        this.nexts.remove(new NodeEdgePair<K>(node, e));
    } else {
      Node<K> curNode = this;  // Because of the closure
      node.prevs.stream().filter(pair -> pair.getValue0() == curNode)
          .forEach(k -> node.prevs.remove(k));
      this.nexts.stream().filter(pair -> pair.getValue0() == node)
          .forEach(k -> this.nexts.remove(k));
    }
    return this;
  }

  /**
   * remove edges that connect previous nodes
   *
   * @return current node
   */
  public Node<K> removePrevEdges() {
    Node<K> curNode = this; // Because of the closure
    prevs.stream().map(pair -> pair.getValue0()).forEach(pn -> {
      pn.nexts.stream().filter(pnpair -> pnpair.getValue0() == curNode)
          .forEach(e -> pn.nexts.remove(e));
    });
    prevs.clear();
    return this;
  }

  /**
   * remove edges that connect next nodes
   *
   * @return current node
   */
  Node<K> removeNextEdges() {
    Node<K> curNode = this;  // Because of the closure
    nexts.stream().map(pair -> pair.getValue0()).forEach(pn -> {
      pn.prevs.stream().filter(pnpair -> pnpair.getValue0() == curNode)
          .forEach(e -> pn.prevs.remove(e));
    });
    nexts.clear();
    return this;
  }

  /**
   * Use current node as source to build a direct graph
   *
   * @param reverse
   * @return
   */
  DirectedGraph<K> graph(boolean reverse) {
    return new DirectedGraph<K>(this, reverse);
  }

  @Override
  public String toString() {
    return "Node{" + "element=" + element.toString() + '}';
  }
}

