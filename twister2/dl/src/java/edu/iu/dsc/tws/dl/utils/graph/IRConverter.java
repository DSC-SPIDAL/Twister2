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
package edu.iu.dsc.tws.dl.utils.graph;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import edu.iu.dsc.tws.dl.graph.Edge;
import edu.iu.dsc.tws.dl.graph.Graph;
import edu.iu.dsc.tws.dl.graph.Node;
import edu.iu.dsc.tws.dl.module.AbstractModule;
import edu.iu.dsc.tws.dl.module.mkldnn.BlasWrapper;
import edu.iu.dsc.tws.dl.module.mkldnn.DnnGraph;
import edu.iu.dsc.tws.dl.module.mkldnn.InputWrapper;
import edu.iu.dsc.tws.dl.module.mkldnn.Output;

public class IRConverter {

  private ArrayList<Node<IRElement>> allNodes = new ArrayList<>();
  private Node<IRElement>[] irInputs; // IRgraph.inputs.toArray
  private Node<IRElement>[] irOutputs; // = IRgraph.outputs.toArray
  private IRGraph iRgraph;

  public IRConverter(IRGraph iRgraph) {
    this.iRgraph = iRgraph;
    init();
  }

  private void init() {
    irInputs = iRgraph.inputs.toArray(new Node[0]);
    irOutputs = iRgraph.outputs.toArray(new Node[0]);
    getNodes(Arrays.asList(irInputs), allNodes);
    // reminder: some output nodes may not be searched from inputs
    for (Node<IRElement> irOutput : irOutputs) {
      if (!allNodes.contains(irOutput)) {
        allNodes.add(irOutput);
      }
    }
  }


  private void getNodes(List<Node<IRElement>> inputs, List<Node<IRElement>> nodesBuffer) {
    if (inputs.size() == 0) {
      return;
    }
    for (Node<IRElement> node : inputs) {
      if (!nodesBuffer.contains(node)) {
        nodesBuffer.add(node);
        getNodes(node.nextNodes(), nodesBuffer);
      }
    }
  }

  /**
   * convert IRgraph to blas or dnn graph according to engine type
   *
   * @return dnn graph or blas graph converted from ir graph
   */
  public Graph toGraph() {

    return toDnnGraph();
//    if (utils.Engine.getEngineType() == MklBlas) {
//      Util.require(IRToBlas[T].convertingCheck(allNodes.toArray),
//          "IR graph can not be converted to Blas layer")
//      toBlasGraph()
//    } else if (utils.Engine.getEngineType() == MklDnn) {
//      Util.require(ev.getType() == FloatType, "Mkldnn engine only supports float data")
//      Util.require(IRToDnn.convertingCheck(
//          allNodes.toArray.asInstanceOf[Array[Node[IRElement[Float]]]]),
//          "IR graph can not be converted to Dnn layer")
//      toDnnGraph()
//    } else {
//      throw new UnsupportedOperationException(
//          "Only support engineType mkldnn/mklblas, but get ${Engine.getEngineType()}");
//    }
  }

  private Graph   toDnnGraph() {
    Map<Node<IRElement>, Node<AbstractModule>> nodeMap = new IRToDnn().convert(allNodes);
    Node<AbstractModule>[] inputs = new Node[irInputs.length];
    Node<AbstractModule>[] outputs = new Node[irOutputs.length];
    Node<AbstractModule>[] realInputs = new Node[irOutputs.length];
    Node<AbstractModule>[] realOutputs = new Node[irOutputs.length];

    for (int i = 0; i < irInputs.length; i++) {
      Node<IRElement> irInput = irInputs[i];
      inputs[i] = nodeMap.get(irInput);
    }

    for (int i = 0; i < irOutputs.length; i++) {
      Node<IRElement> irOutput = irOutputs[i];
      outputs[i] = nodeMap.get(irOutput);
    }
    // add input node for dnn graph
    for (int i = 0; i < inputs.length; i++) {
      Node<AbstractModule> n = inputs[i];
      Node<AbstractModule> node = new Node(new InputWrapper());
      n.from(node, new Edge());
      realInputs[i] = node;
    }

    // add output node for graph
    for (int i = 0; i < outputs.length; i++) {
      Node<AbstractModule> model = outputs[i];
      Node<AbstractModule> node;
      if (model.getElement() instanceof BlasWrapper) {
        node = model;
      } else {
        node = model.add(new Node<AbstractModule>(new Output(iRgraph.outputFormats.get(i))),
            new Edge());
      }
      realOutputs[i] = node;
    }

    return new DnnGraph(Arrays.asList(realInputs), Arrays.asList(realOutputs), iRgraph.variables,
        iRgraph.generateBackward);
  }

//  private Graph toBlasGraph() {
//    val nodeMap = IRToBlas[T].convert(allNodes.toArray)
//    val inputs = irInputs.map(n => nodeMap.get(n).get)
//    val outputs = irOutputs.map(n => nodeMap.get(n).get)
//
//    Graph.dynamic(inputs, outputs, IRgraph.variables, IRgraph.generateBackward)
//  }
}

