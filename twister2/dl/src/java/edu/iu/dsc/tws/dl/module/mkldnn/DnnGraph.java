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
package edu.iu.dsc.tws.dl.module.mkldnn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.intel.analytics.bigdl.mkl.Memory;

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.data.Table;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.format.DataFormat;
import edu.iu.dsc.tws.dl.data.format.NHWC;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.graph.Graph;
import edu.iu.dsc.tws.dl.graph.Node;
import edu.iu.dsc.tws.dl.graph.WithoutInput;
import edu.iu.dsc.tws.dl.module.AbstractModule;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.ArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.MemoryDataArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.NodeEdgePair;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

import static edu.iu.dsc.tws.dl.utils.Util.require;

@SuppressWarnings({"MemberName", "SimplifyBooleanExpression"})
public class DnnGraph extends Graph implements IMklDnnLayer, MklDnnModule {
  private boolean enableExcludeChecking = true;

  //reserse is done in init
  private List<Node<AbstractModule>> forwardExecution = forwardGraph.topologySort();
  private Node<AbstractModule>[] backwardExecution;
  private Activity[] inputCache;
  private int[] backId2ForwardId;
  private boolean[] skipPrimitiveId;


  protected transient long[] updateOutputPrimitives;
  protected transient long[] updateGradInputPrimitives;
  protected transient long[] accGradientPrimitives;

  protected MemoryData[] _inputFormats;
  protected MemoryData[] _gradInputFormats;
  protected MemoryData[] _outputFormats;
  protected MemoryData[] _gradOutputFormats;
  protected MemoryData[] _gradOutputFormatsForWeight;

  private transient long[] updateOutputMemoryPrimitives;
  private transient DenseTensor[] updateOutputTensors;
  private transient long[] updateGradInputMemoryPrimitives;
  private transient DenseTensor[] updateGradInputTensors;
  private transient Activity cachedInput;
  private transient Activity cachedGradOutput;

  protected transient ReorderManager reorderManager;

  public DnnGraph(List<Node<AbstractModule>> inputs, List<Node<AbstractModule>> outputs,
                  TensorArrayPair vars, boolean excludeChecking) {
    super(inputs, outputs, vars);
    this.enableExcludeChecking = excludeChecking;
    init();
  }

  private void init() {

    Collections.reverse(forwardExecution);
    skipPrimitiveId = new boolean[forwardExecution.size()];
    reorderManager = new ReorderManager(this);
    if (enableExcludeChecking) {
      // TODO check if needed?
      //excludeInvalidLayers(forwardExecution.map {_.getElement()})
    }

    buildBackwardGraph();
  }

  /**
   * Batch size may change when model prediction, but output size of dnn layers will not be changed.
   * So we have to compare batchSize for input and output, and do some change if not the same.
   *
   * @param input
   * @param output
   * @return
   */
  private Activity getRealOutput(Activity input, Activity output) {
    if (input.isTensor() && output.isTensor()) {
      Tensor in = input.toTensor();
      Tensor out = output.toTensor();
      // for grey image, input should be 3 dims and the first dim should be batch size
      // for non grey image, input should be 4 dims and the first dim should be batch size
      // for rnn model, input should be 2 dims and the first dim should be batch size
      require(in.nDimension() == 4 || in.nDimension() == 3 || in.nDimension() == 2,
          "only support input with 4 dimension or 3 dimension, but get ${in.nDimension()}");
      if (in.size(1) != out.size(1)) {
        return out.narrow(1, 1, in.size(1));
      } else {
        return output;
      }
    } else {
      return output;
    }
  }


  @Override
  public void populateModules() {
    List<AbstractModule> temp = forwardGraph.topologySort().stream().filter(n -> n != dummyOutput)
        .map(am -> am.getElement()).collect(Collectors.toList());
    Collections.reverse(temp);
    modules.addAll(temp);
    checkDuplicate(new HashSet<>());
  }

  @Override
  public Activity updateOutput(Activity input) {
    int i = 0;
    while (i < forwardExecution.size()) {
      Node<AbstractModule> node = forwardExecution.get(i);
      Activity nodeInput;
      if (skipPrimitiveId[i]) {
        nodeInput = findInput(node, input);
      } else {
        nodeInput = findDnnInput(node, input);
      }
      inputCache[i] = nodeInput;
      Activity output = node.getElement().forward(nodeInput);
      // resize to heap size
      if (!skipPrimitiveId[i] && output.isTensor()
          && !(node.getElement() instanceof BlasWrapper)) {
        output.toTensor().resize(((MklDnnLayer) node.getElement())
            .outputFormats()[0].getHeapShape());
      }
      i += 1;
    }
    output = getRealOutput(input, dummyOutput.getElement().output);
    return output;
  }

  @Override
  public Activity backward(Activity input, Activity gradOutput) {
    long before = System.nanoTime();
    Activity gradients = updateGradInput(input, gradOutput);
    accGradParameters(input, gradOutput);
    backwardTime += System.nanoTime() - before;
    return gradients;
  }

  @Override
  public Activity updateGradInput(Activity input, Activity gradOutput) {
    dummyOutputGrad.getElement().gradInput = gradOutput;
    int i = 0;
    while (i < backwardExecution.length - 1) { // do not execute the dummy backward end
      Node<AbstractModule> curNode = backwardExecution[i];
      Activity curGradOutput = findDnnGradOutput(curNode, gradOutput, false);
      // use input from forward
      Activity curInput = inputCache[backId2ForwardId[i]];
      if (!isStopGradient(curNode.getElement())) {
        Activity gradInput = curNode.getElement().updateGradInput(curInput, curGradOutput);
        // resize to heap size
        if (!skipPrimitiveId[i] && gradInput.isTensor()
            && !(curNode.getElement() instanceof BlasWrapper)) {
          gradInput.toTensor().resize(
              ((MklDnnLayer) curNode.getElement()).gradInputFormats()[0].getHeapShape());
        }
      }
      i += 1;
    }
    gradInput = getRealOutput(input, fetchModelGradInput());
    return gradInput;
  }


  @Override
  public void accGradParameters(Activity input, Activity gradOutput) {
    int i = 0;
    while (i < backwardExecution.length - 1) {
      Node<AbstractModule> curNode = backwardExecution[i];
      // use input from forward
      Activity curInput = inputCache[backId2ForwardId[i]];
      Activity curGradOutput = findDnnGradOutput(curNode, gradOutput, true);
      curNode.getElement().accGradParameters(curInput, curGradOutput);
      curNode.getElement().asyncGradient();
      i += 1;
    }
  }

  @Override
  public Graph buildBackwardGraph() {
    super.buildBackwardGraph();
    inputCache = new Activity[forwardExecution.size()];
    List<Node<AbstractModule>> backwardExecutionTemp;
    backwardExecutionTemp = backwardGraph.topologySort();
    Collections.reverse(backwardExecutionTemp);
    backwardExecution = new Node[backwardExecutionTemp.size()];
    backwardExecutionTemp.toArray(backwardExecution);

    backId2ForwardId = new int[backwardExecution.length];

    int i = 0;
    // do not execute the dummy backward end
    while (i < backwardExecution.length - 1) {
      int j = 0;
      boolean find = false;
      while (j < forwardExecution.size()) {
        if (forwardExecution.get(j).getElement().getName().equals(backwardExecution[i]
            .getElement().getName())) {
          AbstractModule e = forwardExecution.get(j).getElement();
          // when creating graph, there may add nn.Identity node,
          // here we have to change it to mkldnn node
          if (e instanceof edu.iu.dsc.tws.dl.module.Identity) {
            forwardExecution.get(j)
                .setElement(toDnnIdentity((edu.iu.dsc.tws.dl.module.Identity) e));
            backwardExecution[i].setElement(forwardExecution.get(j).getElement());
          } else {
            require(e instanceof MklDnnModule, "DnnGraph should only contain dnn layers,"
                + "but find ${forwardExecution.get(j).getElement().getName()}"
                + " is not a mkldnn layer");
          }
          backId2ForwardId[i] = j;
          find = true;
        }
        j += 1;
      }
      require(find, "Cannot find backward layer in forward executions");
      i += 1;
    }
    return this;
  }

  /**
   * When doing inference, we may not have to compute forward primitives for some blas layers
   */
  private void skipInitFwdPrimitives() {
    HashMap<String, Boolean> skipNodesMap = new HashMap<>();
    Arrays.fill(skipPrimitiveId, 0, skipPrimitiveId.length, false);
    if (!this.train) {
      int i = forwardExecution.size() - 1;
      while (i >= 0) {
        Node<AbstractModule> node = forwardExecution.get(i);
        skipPrimitiveId[i] = skip(node, skipNodesMap);
        skipNodesMap.put(node.getElement().getName(), skipPrimitiveId[i]);
        i -= 1;
      }
    }
  }

  /**
   * to determine whether to skip computing primitives for current node
   * Now, if current node is blaswrapper node and meets one of following cases,
   * then we will skip computing primitives for this node
   * case 1: it has no next nodes
   * case 2: all next nodes are identity node, and those next nodes has no next nodes
   * case 3: all next nodes are also skip nodes
   * In some special case, if previous nodes are not blas node, we can not skip this node,
   * but don't have to compute its output shape.
   *
   * @param node current node
   * @return
   */
  private boolean skip(Node<AbstractModule> node, HashMap<String, Boolean> skipNodesMap) {
    if (node.getElement() instanceof BlasWrapper || node.getElement() instanceof Identity) {
      if (node.nextNodes().size() == 0) {
        return true;
      }
      boolean isSkip = true;
      for (int i = 0; i < node.nextNodes().size(); i++) {
        Node<AbstractModule> n = node.nextNodes().get(i);
        if ((skipNodesMap.getOrDefault(n.getElement().getName(), false))
            || (n.getElement() instanceof Identity && n.nextNodes().size() == 0)) {
          isSkip = isSkip && true;
        } else {
          isSkip = isSkip && false;
        }
      }

      for (int i = 0; i < node.prevNodes().size(); i++) {
        Node<AbstractModule> n = node.prevNodes().get(i);
        if (!(n.getElement() instanceof BlasWrapper)
            && node.getElement() instanceof BlasWrapper
            && isSkip) {
          ((BlasWrapper) node.getElement()).setNeedOutputFormats(false);
          isSkip = false;
        }
      }
      return isSkip;
    } else {
      return false;
    }
  }

  // change nn identity to mkldnn identity
  private AbstractModule toDnnIdentity(edu.iu.dsc.tws.dl.module.Identity model) {
    return new Identity().setName(model.getName());
  }

  // if node has no previous node, then it will just use graph input as real module input
  private Activity findDnnInput(Node<AbstractModule> node, Activity input) {
    if (node.getElement() instanceof WithoutInput) {
      return null;
    }

    MemoryData[] realInputFormats = ((MklDnnModule) node.getElement()).inputFormats();

    Activity nodeInput;
    if (node.prevNodes().isEmpty()) {
      return getInput(node, input);
    } else {
      Activity[] prevActivitiesAndFormats1 = new Activity[node.prevNodesAndEdges().size()];
      MemoryData[][] prevActivitiesAndFormats2 = new MemoryData[node.prevNodesAndEdges().size()][];


      for (int i = 0; i < node.prevNodesAndEdges().size(); i++) {
        NodeEdgePair<AbstractModule> n = node.prevNodesAndEdges().get(i);
        MemoryData[] format = ((MklDnnModule) n.getValue0().getElement()).outputFormats();

        if (n.getValue1().getFromIndex() != null) {
          if (n.getValue0().getElement().output == null || (n.getValue1().getFromIndex() == 1
              && n.getValue0().getElement().output.isTensor())) {
            prevActivitiesAndFormats1[i] = n.getValue0().getElement().output;
            prevActivitiesAndFormats2[i] = format;
          } else {
            prevActivitiesAndFormats1[i] = n.getValue0().getElement()
                .output.toTable().get(n.getValue1().getFromIndex());
            prevActivitiesAndFormats2[i] = new MemoryData[]{format[i - 1]};
          }
        } else {
          prevActivitiesAndFormats1[i] = n.getValue0().getElement().output;
          prevActivitiesAndFormats2[i] = format;
        }
      }
      Activity inputAndFormat1;
      MemoryData[] inputAndFormat2;
      if (prevActivitiesAndFormats1.length == 1) {
        inputAndFormat1 = prevActivitiesAndFormats1[0];
        inputAndFormat2 = prevActivitiesAndFormats2[0];
      } else {
        inputAndFormat1 = new Table(prevActivitiesAndFormats1);
        inputAndFormat2 = Stream.of(prevActivitiesAndFormats2)
            .flatMap(Stream::of).toArray(MemoryData[]::new);
      }
      return reorderManager.infer(inputAndFormat2, realInputFormats, inputAndFormat1);
    }
  }

  private Activity findDnnGradOutput(Node<AbstractModule> curNode,
                                     Activity gradOutput, boolean isAcc) {
    Activity curGradOutput;
    if (curNode == dummyOutputGrad) {
      curGradOutput = gradOutput;
    } else {
      curGradOutput = null;
    }

    MemoryData[] realGradOutputFormats;
    if (isAcc) {
      realGradOutputFormats = ((MklDnnModule) curNode.getElement()).gradOutputWeightFormats();
    } else {
      realGradOutputFormats = ((MklDnnModule) curNode.getElement()).gradOutputFormats();
    }

    for (int i = 0; i < curNode.prevNodesAndEdges().size(); i++) {
      NodeEdgePair<AbstractModule> n = curNode.prevNodesAndEdges().get(i);
      Activity otherActivity;
      MemoryData[] format;


      if (n.getValue0().getElement().gradInput.isTensor()
          || n.getValue0().nextEdges().size() == 1) {
        otherActivity = n.getValue0().getElement().gradInput;
        format = ((MklDnnModule) n.getValue0().getElement()).gradInputFormats();
      } else {
        int index = n.getValue0().nextEdges().indexOf(n.getValue1()) + 1;
        otherActivity = n.getValue0().getElement().gradInput.toTable().get(index);
        format = new MemoryData[]{((MklDnnModule) n.getValue0()
            .getElement()).gradInputFormats()[index - 1]};
      }

      Integer index = n.getValue1().getFromIndex();

      if (index != null) {
        if (i == 1 && curNode.getElement().output.isTensor()) {
          curGradOutput = addActivity(curGradOutput, realGradOutputFormats, otherActivity, format);
        } else {
          if (curNode.getElement().output.isTable() && curGradOutput == null) {
            curGradOutput = new Table();
          }
          Activity curActivity = curGradOutput.toTable().getOrElse(i, null);
          curGradOutput.toTable().put(i, addActivity(curActivity, realGradOutputFormats,
              otherActivity, format));
        }
      } else {
        curGradOutput = addActivity(curGradOutput, realGradOutputFormats,
            otherActivity, format);
      }

    }

    if (curNode.getElement().output.isTable()) {
      addZeroTensorToMissingGradOutput(curNode.getElement()
          .output.toTable(), curGradOutput.toTable());
    }
    return curGradOutput;
  }

  private Activity addActivity(Activity activity, MemoryData[] realFormats,
                               Activity other, MemoryData[] otherFormats) {
    Activity realOthers;
    if (otherFormats.length > 0) {
      realOthers = reorderManager.infer(otherFormats, realFormats, other);
    } else {
      realOthers = other;
    }
    return super.accActivity(activity, realOthers);
  }

  final void compile(Phase phase) {
    setRuntime(new MklDnnRuntime());
    initPrimitives(phase, new MemoryData[]{});
  }

  @Override
  public void setRuntime(MklDnnRuntime runtime) {
    MklDnnModule.super.setRuntime(runtime);
    reorderManager.setRuntime(runtime);
    for (AbstractModule module : modules) {
      if (module instanceof MklDnnModule) {
        ((MklDnnModule) module).setRuntime(runtime);
      }
    }
  }

  @Override
  public long[] getUpdateOutputMemoryPrimitives() {
    long[] result = new long[inputFormats().length + outputFormats().length];
    int index = 0;
    for (MemoryData memoryData : inputFormats()) {
      result[index] = memoryData.getPrimitive(runtime, this);
      index++;
    }
    for (MemoryData memoryData : outputFormats()) {
      result[index] = memoryData.getPrimitive(runtime, this);
      index++;
    }
    return result;
  }

  private void initPrimitives(Phase phase, MemoryData[] inputFormats) {
    _outputFormats = initFwdPrimitives(inputFormats, phase).getValue1();
    if (phase == Phase.TRAINNING) {
      _gradOutputFormats = initBwdPrimitives(_outputFormats, phase).getValue0();
      _gradOutputFormatsForWeight = initGradWPrimitives(_outputFormats, phase);
    }
  }

  private MemoryData[] getInputMemoryData(Node<AbstractModule> node, MemoryData[] memoryData) {
    // the model may contain two inputs and all of them is Input.
    if (inputs.size() == 1 || memoryData == null || memoryData.length == 0) {
      require(inputs.contains(node), "input node must be in the input list");
      return memoryData;
    } else {
      int i = inputs.indexOf(node);
      require(i != -1, "input node is not in the input list");
      return new MemoryData[]{memoryData[i]};
    }
  }

  private MemoryData[] findInputFormats(Node<AbstractModule> node, MemoryData[] input) {
    if (node.prevNodes().isEmpty()) {
      return getInputMemoryData(node, input);
    } else {
      ArrayList<MemoryData> prevFormats = new ArrayList<>();

      for (int i = 0; i < node.prevNodesAndEdges().size(); i++) {
        NodeEdgePair<AbstractModule> n = node.prevNodesAndEdges().get(i);

        MemoryData[] outputFormats = ((MklDnnModule) n.getValue0().getElement()).outputFormats();
        // if outputFormats length is 1, output is a tensor
        Integer index = n.getValue1().getFromIndex();

        if (index != null) {
          if (n.getValue0().getElement().output == null || (i == 1 && outputFormats.length == 1)) {
            prevFormats.addAll(Arrays.asList(outputFormats));
          } else {
            prevFormats.add(outputFormats[index]);
          }
        } else {
          prevFormats.addAll(Arrays.asList(outputFormats));
        }

      }
      return prevFormats.toArray(new MemoryData[0]);
    }
  }

  private MemoryData[] findGradOutputFormats(Node<AbstractModule> node, MemoryData[] inputs) {
    if (node.prevNodes().isEmpty()) {
      return inputs;
    } else {
      int prevFormats;

      //change from DL, since returning first no need to loop

      NodeEdgePair<AbstractModule> n = node.prevNodesAndEdges().get(0);

      // gradInput is tensor or nextEdges number is 1
      if (((MklDnnModule) n.getValue0().getElement()).gradInputFormats().length == 1
          || n.getValue0().nextEdges().size() == 1) {
        return ((MklDnnModule) n.getValue0().getElement()).gradInputFormats();
      } else {
        int index = n.getValue0().nextEdges().indexOf(n.getValue1());
        MemoryData[] f = ((MklDnnModule) n.getValue0().getElement()).gradInputFormats();
        return new MemoryData[]{f[index]};
      }
    }
  }

  /**
   * fuse some layers when doing inference
   * first fuse layers in sequence, mainly relu with bn/conv, conv with bn.
   * after that, fuse sum operation.
   */
  private void fusion() {
//    if (!this.train) {
//
//        for (int j = 0; j < 5; j++) {
//        int i = forwardExecution.size() - 1;
//        while (i >= 0) {
//          if (j == 0) Fusion.fuseScale(forwardExecution.get(i));
//          if (j == 1) Fusion.fuseModule(forwardExecution.get(i));
//          // we should do this before sum fusion, because it will change the structure of graph
//          if (j == 2) Fusion.setNegativeInputOfConv(forwardExecution.get(i));
//          if (j == 3) Fusion.fuseCAdd(forwardExecution.get(i));
//          if (j == 4) Fusion.setScalesPrevousJoinTable(forwardExecution.get(i));
//          i -= 1
//        }
//      }
//    }
  }

  private int getHeapFormat(MemoryData[] inputs) {
    int heapFormat = -1;

    for (int i = 0; i < inputs.length; i++) {
      MemoryData m = inputs[i];
      if (m.shape().length == 4) {
        return inputs[0].layout();
      }
    }

    for (int i = 0; i < forwardExecution.size(); i++) {
      AbstractModule m = forwardExecution.get(i).getElement();
      int format = -1;
//      match {
//        case conv: mkldnn.SpatialConvolution => transferFormat(conv.format)
//        case maxPool: mkldnn.MaxPooling => transferFormat(maxPool.format)
//        case avgPool: mkldnn.AvgPooling => transferFormat(avgPool.format)
//        case sbn: mkldnn.SpatialBatchNormalization => transferFormat(sbn.format)
//        case lrn: mkldnn.LRN => transferFormat(lrn.format)
//        case _ => -1
//      }

      if (heapFormat == -1) {
        heapFormat = format;
      } else if (format != -1) {
        require(heapFormat == format,
            "layer ${m} should use format ${heapFormat}, but get ${format}");
      }
    }
    if (heapFormat == -1) {
      return Memory.Format.nchw;
    } else {
      return heapFormat;
    }
  }

  private int transferFormat(DataFormat format) {
    if (format instanceof NHWC) {
      return Memory.Format.nhwc;
    } else {
      return Memory.Format.nchw;
    }
  }

  @Override
  public long[] getUpdateGradInputMemoryPrimitives() {
    long[] result = new long[inputFormats().length + outputFormats().length];
    int index = 0;
    for (MemoryData memoryData : gradOutputFormats()) {
      result[index] = memoryData.getPrimitive(runtime, this);
      index++;
    }
    for (MemoryData memoryData : gradInputFormats()) {
      result[index] = memoryData.getPrimitive(runtime, this);
      index++;
    }
    return result;
  }

  @Override
  public ArrayPair<TensorMMap> paramsMMap() {
    return null;
  }

  @Override
  public MemoryDataArrayPair initFwdPrimitives(MemoryData[] inputs, Phase phase) {
    skipInitFwdPrimitives();
    fusion();
    MemoryData[] lastOutputFormats = inputs;
    MemoryData[] firstRealInputFormats = null;
    int heapFormat = getHeapFormat(inputs);
    for (int i = 0; i < forwardExecution.size(); i++) {
      if (!skipPrimitiveId[i]) {
        Node<AbstractModule> m = forwardExecution.get(i);
        lastOutputFormats = findInputFormats(m, inputs);

        // (realInputFormats, realOutputFormats)
        MemoryDataArrayPair realInputOutputFormats =
            ((MklDnnModule) m.getElement()).initFwdPrimitives(lastOutputFormats, phase);

        if (realInputOutputFormats.getValue1() != null) {
          for (MemoryData memoryData : realInputOutputFormats.getValue1()) {
            memoryData.setHeapFormat(heapFormat);
          }
        }

        for (int j = 0; j < lastOutputFormats.length; j++) {
          Util.copyMaskAndScales(lastOutputFormats[j], realInputOutputFormats.getValue0()[j]);
          reorderManager.register(lastOutputFormats[j], realInputOutputFormats.getValue0()[j]);
        }

        // copy the scales from the input formats to output formats, for some layers,
        // it will not copy the mask and scales automatically or generate the scales themselves
        Util.copyMaskAndScales(realInputOutputFormats.getValue0(),
            realInputOutputFormats.getValue1());

        if (i == 0) {
          firstRealInputFormats = realInputOutputFormats.getValue0();
        }
      }
    }
    _inputFormats = firstRealInputFormats;
    _outputFormats = lastOutputFormats;
    return new MemoryDataArrayPair(firstRealInputFormats, lastOutputFormats);
  }

  @Override
  public MemoryDataArrayPair initBwdPrimitives(MemoryData[] grad, Phase phase) {
    MemoryData[] lastGradInputFormats = grad;
    MemoryData[] firstRealGradOutputFormats = null;

    for (int i = 0; i < backwardExecution.length - 1; i++) {
      Node<AbstractModule> m = backwardExecution[i];
      lastGradInputFormats = findGradOutputFormats(m, grad);
      MemoryDataArrayPair realGradOutputAndInputFomrats =
          ((MklDnnModule) m.getElement()).initBwdPrimitives(lastGradInputFormats, phase);

      for (int j = 0; j < lastGradInputFormats.length; j++) {
        reorderManager.register(lastGradInputFormats[j],
            realGradOutputAndInputFomrats.getValue0()[j]);
      }

      if (i == 0) {
        firstRealGradOutputFormats = realGradOutputAndInputFomrats.getValue0();
      }
    }
    _gradOutputFormats = firstRealGradOutputFormats;
    _gradInputFormats = lastGradInputFormats;
    return new MemoryDataArrayPair(firstRealGradOutputFormats, lastGradInputFormats);
  }

  @Override
  public MemoryData[] initGradWPrimitives(MemoryData[] grad, Phase phase) {
    MemoryData[] lastGradInputFormats = grad;
    MemoryData[] firstRealGradOutputFormats = null;

    for (int i = 0; i < backwardExecution.length - 1; i++) {
      Node<AbstractModule> m = backwardExecution[i];
      lastGradInputFormats = findGradOutputFormats(m, grad);
      MemoryData[] realGradOutput =
          ((MklDnnModule) m.getElement()).initGradWPrimitives(lastGradInputFormats, phase);

      for (int j = 0; j < realGradOutput.length; j++) {
        MemoryData realOut = realGradOutput[j];
        MemoryData lastIn = lastGradInputFormats[j];
        reorderManager.register(lastIn, realOut);
      }
      if (i == 0) {
        firstRealGradOutputFormats = realGradOutput;
      }
    }
    _gradOutputFormatsForWeight = firstRealGradOutputFormats;
    return firstRealGradOutputFormats;
  }

  @Override
  public void release() {
    // do not call super.release, it will call MklDnnLayer.release()
    modules.forEach(module -> module.release());
    // we need to call releaseResources here because super.release will never be called
    this.releaseResources();
  }

  @Override
  public MklDnnModule setQuantize(boolean value) {
//    this.forwardExecution.foreach { node =>
//      if (node.getElement().isInstanceOf[MklDnnModule]) {
//        node.getElement().asInstanceOf[MklDnnModule].setQuantize(value)
//      }
//    }
    return this;
  }

  @Override
  public MemoryData[] inputFormats() {
    require(_inputFormats != null, "You should call initFwdPrimitives first");
    return _inputFormats;
  }

  @Override
  public MemoryData[] gradInputFormats() {
    require(_gradInputFormats != null, "You should call initBwdPrimitives first");
    return _gradInputFormats;
  }

  @Override
  public MemoryData[] outputFormats() {
    require(_outputFormats != null, "You should call initFwdPrimitives first");
    return _outputFormats;
  }

  @Override
  public MemoryData[] gradOutputFormats() {
    require(_gradOutputFormats != null, "You should call initBwdPrimitives first");
    return _gradOutputFormats;
  }

  @Override
  public MemoryData[] gradOutputWeightFormats() {
    return _gradOutputFormatsForWeight;
  }

  private void readObject(java.io.ObjectInputStream in)
      throws IOException, ClassNotFoundException {
    in.defaultReadObject();
    reorderManager = new ReorderManager(this);
  }
}
