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
import java.util.stream.Collectors;
import java.util.stream.Stream;

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.data.EmptyGradInput;
import edu.iu.dsc.tws.dl.data.Table;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.module.AbstractModule;
import edu.iu.dsc.tws.dl.module.Container;
import edu.iu.dsc.tws.dl.module.Identity;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

/**
 * A graph container. The modules in the container are connected as a directed Graph. Each module
 * can output one tensor or multiple tensors(as table). The edges between modules in the graph
 * define how these tensors are passed. For example, if a module outputs two tensors, you can
 * pass these two tensors together to its following module, or pass only one of them
 * to its following module. If a tensor in the module output is connected to multiple modules, in
 * the back propagation, the gradients from multiple connection will be accumulated. If multiple
 * edges point to one module, the tensors from these edges will be stack as a table, then pass to
 * that module. In the back propagation, the gradients will be splited based on how the input
 * tensors stack.
 * <p>
 * The graph container has multiple inputs and multiple outputs. The order of the input tensors
 * should be same with the order of the input nodes when you construct the graph container. In the
 * back propagation, the order of the gradients tensors should be the same with the order of the
 * output nodes.
 * <p>
 * If there's one output, the module output is a tensor. If there're multiple outputs, the module
 * output is a table, which is actually an sequence of tensor. The order of the output tensors is
 * same with the order of the output modules.
 * <p>
 * All inputs should be able to connect to outputs through some paths in the graph. It is
 * allowed that some successors of the inputs node are not connect to outputs. If so, these nodes
 * will be excluded in the computation.
 */
public abstract class Graph extends Container<Activity> {
  protected Node<AbstractModule> dummyOutput;
  protected DirectedGraph<AbstractModule> forwardGraph;
  protected List<Node<AbstractModule>> forwardNodes;
  protected Node<AbstractModule> dummyOutputGrad;
  protected DirectedGraph<AbstractModule> backwardGraph;
  protected List<Node<AbstractModule>> backwardNodes;
  // For constructor
  protected List<Node<AbstractModule>> inputs;
  protected List<Node<AbstractModule>> outputs;
  //an Array of tensor containing all the weights and biases of this graph,
  //used when different nodes of this graph may share the same weight or bias.
  protected TensorArrayPair variables;
  // If the graph will generate gradInput for the input
  private Boolean[] isGradInputAvailable;


  private HashSet<String> stopGradientLayers;

  public Graph(List<Node<AbstractModule>> inputs, List<Node<AbstractModule>> outputs,
               TensorArrayPair vars) {
    this.inputs = inputs;
    this.outputs = outputs;
    this.variables = vars;
    init();
  }

  private void init() {
    // Add a dummy output node, to get an one end forward graph. So the nodes that are not dependent
    // by the outputs will be excluded
    //TODO check for correctness
    dummyOutput = new Node<AbstractModule>(new Identity());
    this.outputs.forEach(moduleNode -> moduleNode.pointTo(dummyOutput));
    forwardGraph = dummyOutput.graph(true);
    forwardNodes = new ArrayList<Node<AbstractModule>>();
    forwardGraph.BFS().forEachRemaining(forwardNodes::add);

    populateModules();

    // Check all inputs of the graph should be passed in
    checkRoots();
  }


  public List<Node<AbstractModule>> getInputs() {
    return inputs;
  }

  public void setInputs(List<Node<AbstractModule>> inputs) {
    this.inputs = inputs;
  }

  public List<Node<AbstractModule>> getOutputs() {
    return outputs;
  }

  public void setOutputs(List<Node<AbstractModule>> outputs) {
    this.outputs = outputs;
  }

  public TensorArrayPair getVariables() {
    return variables;
  }

  public void setVariables(TensorArrayPair variables) {
    this.variables = variables;
  }


  /**
   * For a multi-tensor output module, some output tensors may not contributed to the final forward
   * result. So in the back propagation, the gradient on these positions are missing. And we use
   * zero tensor to populate.
   *
   * @param output
   * @param gradOutput
   */
  protected void addZeroTensorToMissingGradOutput(Table output, Table gradOutput) {
    int i = 0;
    while (i < output.length()) {
      if (!gradOutput.contains(i + 1)) {
        Tensor tensor = output.<Tensor>get(i + 1);
        Tensor zero = new DenseTensor(tensor.size(), this.isFloat());
        gradOutput.update(i + 1, zero);
      }
      i = i + 1;
    }
  }

//    private void calcSumTimesOfAllNodes( timesOfAllNodes:
//    Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)])
//  : (Long, Long) = {
//    int sumForward = 0L
//    int sumBackward = 0L
//    timesOfAllNodes.foreach(x => {
//        sumForward += x._2
//        sumBackward += x._3
//    })
//    (sumForward, sumBackward)
//  }

//    override void getTimes():
//    Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
//    Tensor timesOfAllNodes = this.modules.flatMap(_.getTimes()).toArray
//    Tensor (sumForward, sumBackward) = calcSumTimesOfAllNodes(timesOfAllNodes)
//    timesOfAllNodes ++ Array((this,
//    this.forwardTime - sumForward, this.backwardTime - sumBackward))
//  }

  @Override
  public TensorArrayPair parameters() {
    if (variables != null) {
      return variables;
    } else {
      return super.parameters();
    }
  }


  // todo: expand the graph
  //override void toGraph(startNodes: Node<AbstractModule*): Graph = this

  /**
   * Return the corresponding node has the given name. If the given name doesn't match any node,
   * NoSuchElementException will be thrown
   *
   * @param name
   * @return
   */
  public Node<AbstractModule> node(String name) {
    List<Node<AbstractModule>> matchNodes = forwardNodes.stream()
        .filter(node -> node.getElement().getName() == name).collect(Collectors.toList());
    if (matchNodes.size() == 0) {
      throw new IllegalStateException("Can not find node with name $name");
    } else {
      return matchNodes.get(0);
    }
  }

  public abstract void populateModules();

  // Check if the graph is correct
  public void checkRoots() {
//      void duplicatedNames(names: Seq[String]): mutable.Set[String] = {
//      names.sortWith(_ < _);
//      Tensor buffer = new mutable.HashSet[String]()
//      int i = 1;
//    while(i < names.length) {
//      if (names(i) == names(i - 1)) buffer.add(names(i));
//      i += 1;
//    }
//    buffer;
//    }

    Util.require(forwardNodes.stream()
            .map(node -> node.getElement().getName()).distinct().count()
            == forwardNodes.size(),
        "the name of node in the graph should be unique, but find duplicated name "
            + "${duplicatedNames(forwardNodes.map(_.element.getName())).mkString()}");

    List<Node<AbstractModule>> roots = forwardNodes.stream()
        .filter(node -> node.prevNodes().size() == 0)
        .filter(x -> !(x.getElement() instanceof WithoutInput)).collect(Collectors.toList());
    //.filter(y -> y.getElement() instanceof ControlDependency);

    List<Node<AbstractModule>> realInputs = inputs.stream()
        .filter(x -> !(x instanceof WithoutInput)).collect(Collectors.toList());
    Util.require(roots.size() == realInputs.size(),
        "There're ${realInputs.length} inputs, "
            + "but graph has ${roots.size} roots");

    realInputs.forEach(n ->
        Util.require(roots.contains(n), "inputs and graph roots are not match"));
  }

  /**
   * Generate backward graph and apply the stopGrad
   */
  public Graph buildBackwardGraph() {
    // Clone the forward graph and reverse the edge
    DirectedGraph<AbstractModule> gradGraph = forwardGraph.cloneGraph(true);
    dummyOutputGrad = gradGraph.getSource();
    gradGraph.DFSList().stream().
        filter(x -> isStopGradient(x.getElement())).forEach(y -> removeStopNodes(y));
    backwardNodes = gradGraph.DFSList().stream()
        .filter(x -> !(x == dummyOutputGrad)).collect(Collectors.toList());
    //.filter(y -> !(y.getElement() in_.element.isInstanceOf[ControlDependency[_]]).toArray

    List<String> inputNames = inputs.stream()
        .map(x -> x.getElement().getName()).collect(Collectors.toList());
    Node<AbstractModule> dummyBackwardEnd = new Identity().inputs();
    Stream<Node<AbstractModule>> backwardTargets = backwardNodes.stream()
        .filter(n -> (n.getElement().parameters() != null
            && n.getElement().parameters().getValue0().length != 0)
            || inputNames.contains(n.getElement().getName()));

    backwardTargets.forEach(y -> y.pointTo(dummyBackwardEnd));
    backwardGraph = dummyBackwardEnd.graph(true);

    // Check if gradInput is empty for each input
    isGradInputAvailable = inputs.stream().map(x -> false).toArray(Boolean[]::new);
    backwardGraph.DFSList().forEach(curNode -> {
      for (int i = 0; i < inputs.size(); i++) {
        if (curNode.getElement().getName().equals(inputs.get(i).getElement().getName())
            && !isStopGradient(inputs.get(i).getElement())) {
          isGradInputAvailable[i] = true;
        }
      }
    });

    clearState();
    return this;
  }

  HashSet<String> getStopGradientLayers() {
    return this.stopGradientLayers;
  }

  /**
   * whether stop propagating gradInput back
   *
   * @return
   */
  protected boolean isStopGradient(AbstractModule module) {
    return null != stopGradientLayers && stopGradientLayers.contains(module.getName());
  }

  /**
   * stop the input gradient of layers that match the given ```names```
   * their input gradient are not computed.
   * And they will not contributed to the input gradient computation of
   * layers that depend on them.
   *
   * @param names an array of layer names
   * @return current graph model
   */
  Graph stopGradient(String[] names) {
    if (stopGradientLayers == null) {
      stopGradientLayers = new HashSet<String>();
    }

    for (String name : names) {
      AbstractModule layer = this.apply(name);
      Util.require(layer != null, "cannot find layer match ${name}");
      stopGradientLayers.add(layer.getName());

    }
    buildBackwardGraph();
    return this;
  }

  /**
   * set an array of layers that match the given ```names``` to be "freezed",
   * i.e. their parameters(weight/bias, if exists) are not changed in training process
   *
   * @param names an array of layer names
   * @return current graph model
   */
  public Graph freeze(String[] names) {
    for (String name : names) {
      AbstractModule layer = this.apply(name);
      Util.require(layer != null, "cannot find layer match ${name}");
      layer.setScaleW(0);
      layer.setScaleB(0);
    }
    return this;
  }

  public void removeStopNodes(Node n) {
    List<Node> nodes = n.nextNodes();
    n.removeNextEdges();
    nodes.stream().filter(x -> x.prevNodes().size() == 0).forEach(y -> removeStopNodes(y));
  }


  protected Activity getInput(Node<AbstractModule> node, Activity input) {
    if (inputs.size() == 1) {
      Util.require(inputs.get(0) == node, "input node is not in the input list");
      return input;
    } else {
      int i = inputs.indexOf(node);
      Util.require(i != -1, "input node is not in the input list");
      //TODO check correctness
      return ((Table) input).get(i + 1);
    }
  }

  public Activity findInput(Node<AbstractModule> node, Activity input) {
    if (node.getElement() instanceof WithoutInput) {
      return null;
    }

    Activity nodeInput;
    if (node.prevNodes().isEmpty()) {
      nodeInput = getInput(node, input);
    } else {
      List<Activity> prevActivities = node.prevNodesAndEdges().stream()
          //.filter(n -> n.getValue0().getElement() insn._1.element.isInstanceOf[ControlDependency])
          .map(n -> {
            if (n.getValue1().getFromIndex() != null) {
              if (n.getValue0().getElement().output == null
                  || (n.getValue1().getFromIndex() == 1
                  && n.getValue0().getElement().output.isTensor())) {
                return n.getValue0().getElement().output;
              } else {
                return n.getValue0().getElement()
                    .output.toTable().get(n.getValue1().getFromIndex());
              }
            } else {
              return n.getValue0().getElement().output;
            }
          }).collect(Collectors.toList());
      if (prevActivities.size() == 1) {
        nodeInput = prevActivities.get(0);
      } else {
        nodeInput = new Table(prevActivities.toArray());
      }
    }
    return nodeInput;
  }

  protected Activity findGradOutput(Node<AbstractModule> curNode, Activity gradOutput) {
    final Activity[] curGradOutput = new Activity[1];
    curGradOutput[0] = (curNode == dummyOutputGrad) ? gradOutput : null;

    curNode.prevNodesAndEdges().stream()
        //.filterNot(n -> n._1.element.isInstanceOf[ControlDependency])
        .forEach(n -> {
          Activity otherActivity;
          if (n.getValue0().getElement().gradInput.isTensor()
              || n.getValue0().nextEdges().size() == 1) {
            otherActivity = n.getValue0().getElement().gradInput;
          } else {
            int index = n.getValue0().nextEdges().indexOf(n.getValue1()) + 1;
            otherActivity = n.getValue0().getElement().gradInput.toTable().get(index);
          }

          if (n.getValue1().getFromIndex() != null) {
            if (n.getValue1().getFromIndex() == 1 && curNode.getElement().output.isTensor()) {
              curGradOutput[0] = accActivity(curGradOutput[0], otherActivity);
            } else {
              if (curNode.getElement().output.isTable() && curGradOutput[0] == null) {
                curGradOutput[0] = new Table();
              }
              Activity curActivity = curGradOutput[0].toTable()
                  .getOrElse(n.getValue1().getFromIndex(), null);
              curGradOutput[0].toTable()
                  .update(n.getValue1().getFromIndex(), accActivity(curActivity, otherActivity));
            }
          } else {
            curGradOutput[0] = accActivity(curGradOutput[0], otherActivity);
          }
        });

    if (curNode.getElement().output.isTable()) {
      addZeroTensorToMissingGradOutput(curNode.getElement().output.toTable(),
          curGradOutput[0].toTable());
    }

    return curGradOutput[0];
  }

  protected Activity fetchModelGradInput() {
    //TODO check
    if (inputs.size() == 1) {
      if (isGradInputAvailable[0]) {
        return inputs.get(0).getElement().gradInput;
      } else {
        return new EmptyGradInput(this.getName());
      }
    } else {
      int i = 0;
      List<Activity> results = new ArrayList<>();
      for (Node<AbstractModule> input : inputs) {
        if (isGradInputAvailable[i]) {
          results.add(input.getElement().gradInput);
        } else {
          results.add(new EmptyGradInput(this.getName()));
        }
      }
      return new Table(results.toArray());
    }
  }

  @Override
  public void reset() {
    if (null != stopGradientLayers) {
      stopGradientLayers.clear();
    }
    unFreeze(null);
    buildBackwardGraph();
  }

  /**
   * Get forward executions, the dummy node will be filtered.
   * <p>
   * This method will output an unsorted executions.
   *
   * @return
   */
  public Node<AbstractModule>[] getForwardExecutions() {
    return (Node<AbstractModule>[]) forwardNodes.stream()
        .filter(x -> !(x == dummyOutput)).toArray(Node[]::new);
  }

  /**
   * Get forward executions, the dummy nodes and control dependency nodes will be filtered.
   * <p>
   * This method will output a sorted executions. If the graph contains loop, it will throw an
   * exception
   *
   * @return
   */
  public Node<AbstractModule>[] getSortedForwardExecutions() {
    List<Node<AbstractModule>> results = forwardGraph.topologySort().stream()
        .filter(x -> x != dummyOutput).collect(Collectors.toList());
    Collections.reverse(results);
    // .filterNot(_.element.isInstanceOf[ControlDependency]).reverse
    return (Node<AbstractModule>[]) results.toArray(new Node[0]);
  }

  protected Activity accActivity(Activity activity, Activity other) {
    if (activity == null) {
      return other;
    } else {
      if (other.isTensor()) {
        Util.require(activity.isTensor(), "Cannot add a table to a tensor");
        activity.toTensor().add(other.toTensor());
        return null;
      } else {
        // if 'activity' and 'other' are both table, we need to merge 'other' to 'activity'
        // if 'other' and 'activity' both contains the index, update 'activity' by sum
        // if 'other' contains the index while 'activity' does not,
        // just insert the corresponding tensor of 'other' to 'activity'
        Table actTable = activity.toTable();
        Table otherTable = other.toTable();
        otherTable.keySet().forEach(index -> {
          if (actTable.contains(index)) {
            accActivity(actTable.get(index), otherTable.get(index));
          } else {
            actTable.insert((Integer) index, otherTable.get(index));
          }
        });
        return actTable;
      }
    }
  }

  /**
   * Save current model graph to a folder, which can be display in tensorboard by running
   * tensorboard --logdir logPath
   *
   * @param logPath
   * @param backward Draw backward graph instead of forward
   * @return
   */
  public Graph saveGraphTopology(String logPath, boolean backward) {
    throw new UnsupportedOperationException("Operation not supported");
    //      Tensor writer = new TFFileWriter(logPath)
//      Tensor graphBuilder = GraphDef.newBuilder()
//      Tensor nodes = if (backward) {
//    backwardNodes.filter(n => !n.eq(dummyOutputGrad))
//  } else {
//    forwardNodes.filter(n => !n.eq(dummyOutput))
//  }
//  nodes.map(m => {
//      Tensor nodeDef = Tensorflow.bigdlModule(m.element,
//      m.prevNodes.map(_.element.getName()).asJava)
//      graphBuilder.addNode(nodeDef)
//  })
//
//  writer.addGraphDef(graphBuilder.build())
//  writer.close()
//  this
  }

  /**
   * Clear the original module and reset with module in the graph
   */
  public void resetModules() {
    this.modules.clear();
    List<AbstractModule> temp = forwardGraph.DFSList().stream()
        //.filterNot(_.element.isInstanceOf[ControlDependency])
        .filter(n -> n != dummyOutput).map(x -> x.getElement()).collect(Collectors.toList());
    Collections.reverse(temp);
    modules.addAll(temp);
  }

}
