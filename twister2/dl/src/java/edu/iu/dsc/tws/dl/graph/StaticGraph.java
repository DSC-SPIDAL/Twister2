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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import com.intel.analytics.bigdl.mkl.Memory;

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.module.AbstractModule;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.graph.BlasToIR;
import edu.iu.dsc.tws.dl.utils.graph.IRElement;
import edu.iu.dsc.tws.dl.utils.graph.IRGraph;

public class StaticGraph extends Graph {
  private boolean enableExcludeChecking = true;

  private List<Node<AbstractModule>> forwardExecution; // = forwardGraph.topologySort()
  private List<Node<AbstractModule>> backwardExecution;
  private Activity[] inputCache;
  private int[] backId2ForwardId;
  private Activity[] gradOutputCache;

  public StaticGraph(List<Node<AbstractModule>> starts, List<Node<AbstractModule>> endNodes) {
    super(starts, endNodes, null);
    this.enableExcludeChecking = true;
    init();
  }

  private void init() {
    forwardExecution = forwardGraph.topologySort();
    Collections.reverse(forwardExecution);

    inputCache = new Activity[forwardExecution.size()];
    if (enableExcludeChecking) {
      //TODO check
      //excludeInvalidLayers(forwardExecution.map {_.getElement()})
    }

    buildBackwardGraph();
  }

  @Override
  public void populateModules() {
    List<Node<AbstractModule>> tempGraph = forwardGraph.topologySort();
    List<AbstractModule> tempList = new ArrayList<>();
    for (int i = 0; i < tempGraph.size(); i++) {
      Node<AbstractModule> n = tempGraph.get(i);
      if (n != dummyOutput) {
        tempList.add(n.getElement());
      }
    }

    Collections.reverse(tempList);
    modules.addAll(tempList);
    checkDuplicate(new HashSet<>());
  }

  @Override
  public Activity updateOutput(Activity input) {
    int i = 0;
    while (i < forwardExecution.size()) {
      Node<AbstractModule> node = forwardExecution.get(i);
      Activity nodeInput = findInput(node, input);
      inputCache[i] = nodeInput;
      node.getElement().forward(nodeInput);
      i += 1;
    }

    output = dummyOutput.getElement().output;
    return output;
  }

  @Override
  public Activity backward(Activity input, Activity gradOutput) {
    long before = System.nanoTime();
    Activity gradients = backwardExecution(input, gradOutput, true);
    backwardTime += System.nanoTime() - before;
    return gradients;
  }

  @Override
  public Activity updateGradInput(Activity input, Activity gradOutput) {
    return backwardExecution(input, gradOutput, false);
  }

  @Override
  public Graph buildBackwardGraph() {
    super.buildBackwardGraph();
    backwardExecution = backwardGraph.topologySort();
    Collections.reverse(backwardExecution);
    backId2ForwardId = new int[backwardExecution.size()];
    gradOutputCache = new Activity[backwardExecution.size()];

    int i = 0;
    while (i < backwardExecution.size() - 1) {
      int j = 0;
      boolean find = false;
      while (j < forwardExecution.size()) {
        if (forwardExecution.get(j).getElement().getName().equals(backwardExecution
            .get(i).getElement().getName())) {
          backId2ForwardId[i] = j;
          find = true;
        }
        j += 1;
      }
      Util.require(find, "Cannot find backward layer in forward executions");
      i += 1;
    }

    return this;
  }

  @Override
  public void accGradParameters(Activity input, Activity gradOutput) {
    int i = 0;
    while (i < backwardExecution.size() - 1) {
      Node<AbstractModule> curNode = backwardExecution.get(i);
      Activity curInput = inputCache[backId2ForwardId[i]];
      curNode.getElement().accGradParameters(curInput, gradOutputCache[i]);
      i += 1;
    }
  }

  private Activity backwardExecution(Activity input, Activity gradOutput,
                                     boolean executeBackward) {
    dummyOutputGrad.getElement().gradInput = gradOutput;

    int i = 0;
    while (i < backwardExecution.size() - 1) {  // do not execute the dummy backward end
      Node<AbstractModule> curNode = backwardExecution.get(i);
      Activity curGradOutput = findGradOutput(curNode, gradOutput);
      gradOutputCache[i] = curGradOutput;
      Activity curInput = inputCache[backId2ForwardId[i]];
      if (!isStopGradient(curNode.getElement())) {
        if (executeBackward) {
          curNode.getElement().backward(curInput, curGradOutput);
        } else {
          curNode.getElement().updateGradInput(curInput, curGradOutput);
        }
      } else if (executeBackward) {
        curNode.getElement().accGradParameters(curInput, curGradOutput);
      }
      i += 1;
    }

    gradInput = fetchModelGradInput();
    return gradInput;
  }

  /**
   * convert static graph to ir graph and build according to engine type
   *
   * @return return ir graph if converted successfully, otherwise null
   */
  public IRGraph toIRgraph() {
    List<Integer> inFormats;
    if (inputsFormats == null) {
      //logger.warn("Input formats NCHW by default, Please set explicitly if needed")
      inFormats = new ArrayList<>();
      inFormats.add(Memory.Format.nchw);
    } else {
      inFormats = inputsFormats;
    }

    List<Integer> outFormats;
    if (outputsFormats == null) {
//      logger.warn("Output formats NC by default, Please set explicitly if needed")
      outFormats = new ArrayList<>();
      outFormats.add(Memory.Format.nc);
    } else {
      outFormats = outputsFormats;
    }

    List<Node<AbstractModule>> allNodes = forwardExecution;
    BlasToIR blasToIR = new BlasToIR();
    if (!(blasToIR.convertingCheck(allNodes))) {
      return null;
    }

    Map<Node<AbstractModule>, Node<IRElement>> nodeMap = blasToIR.convert(allNodes);

//    int inputNodes = inputs.toArray.map(n => nodeMap.get(n).get);
//    int outputNodes = outputs.toArray.map(n => nodeMap.get(n).get);
    ArrayList<Node<IRElement>> inputsIR = new ArrayList<>();
    ArrayList<Node<IRElement>> outputsIR = new ArrayList<>();
    for (int i = 0; i < inputs.size(); i++) {
      Node<AbstractModule> n = inputs.get(i);
      inputsIR.add(nodeMap.get(n));
    }

    for (int i = 0; i < outputs.size(); i++) {
      Node<AbstractModule> n = outputs.get(i);
      outputsIR.add(nodeMap.get(n));
    }
    IRGraph model = new IRGraph(inputsIR, outputsIR, variables, true, inFormats, outFormats);
    return model.build();
  }

  /**
   * Merge a nested StaticGraph into a non-nested one.
    */
  public StaticGraph toSingleGraph() {
    if (this.isNestedGraph()) {
      throw new UnsupportedOperationException("Nested graph not supported");
//      StaticGraph graph = this.cloneModule();
//      int fwdExecution = graph.getSortedForwardExecutions()
//      int dmOutput = fwdExecution(fwdExecution.size() - 1).nextNodes(0)
//
//      int i = 0
//      while (i < fwdExecution.size()) {
//        if (fwdExecution(i).getElement().isInstanceOf[StaticGraph[T]]) {
//          int g = fwdExecution(i).getElement().asInstanceOf[StaticGraph[T]].toSingleGraph()
//          fwdExecution(i).getElement() = g
//
//          for (inputIndex <- 0 until fwdExecution(i).prevNodes.size()) {
//            int inputNode = g.inputs(inputIndex)
//            inputNode.getElement() = Identity()
//
//            while (fwdExecution(i).prevNodes.size() != 0) {
//              int preNode = fwdExecution(i).prevNodes(0)
//              preNode.delete(fwdExecution(i))
//              preNode.add(inputNode)
//            }
//          }
//
//          for (outputIndex <- 0 until g.outputs.size()) {
//            int outputNode = g.outputs(outputIndex)
//            outputNode.removeNextEdges()
//            while (fwdExecution(i).nextNodes.size() != 0) {
//              int nextNode = fwdExecution(i).nextNodes(0)
//              fwdExecution(i).delete(nextNode)
//              outputNode.add(nextNode)
//            }
//          }
//        }
//        i += 1
//      }
//
//      int resultOutputNodes = dmOutput.prevNodes
//      resultOutputNodes.foreach(_.delete(dmOutput))
//      new StaticGraph[T](Array(graph.inputs(0)), resultOutputNodes,
//          enableExcludeChecking = this.enableExcludeChecking)
    } else {
      return this;
    }
  }

  private boolean isNestedGraph() {
    for (int i = 0; i < forwardExecution.size(); i++) {
      if (forwardExecution.get(i).getElement() instanceof StaticGraph) {
        return true;
      }
    }
    return false;
  }

}

