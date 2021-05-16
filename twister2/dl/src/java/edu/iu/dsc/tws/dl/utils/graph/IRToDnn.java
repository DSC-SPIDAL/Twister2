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

import java.util.*;

import edu.iu.dsc.tws.dl.graph.Node;
import edu.iu.dsc.tws.dl.module.AbstractModule;
import edu.iu.dsc.tws.dl.module.Reshape;
import edu.iu.dsc.tws.dl.module.View;
import edu.iu.dsc.tws.dl.module.mkldnn.BlasWrapper;
import edu.iu.dsc.tws.dl.utils.ReflectionUtils;

@SuppressWarnings("MemberName")
public class IRToDnn extends ConvertBase<IRElement, AbstractModule> {

  private String prefix = "edu.iu.dsc.tws.dl.module.mkldnn.";
  // converter function mappings
  private Set<String> IR2DnnMap = new HashSet<>(); //[String, (IRElement) => AbstractModule]

  public IRToDnn() {
    mapInit();
  }


  private void mapInit() {
//      IR2DnnMap("IRSpatialConvolution") = fromSpatialConvolution
//      IR2DnnMap("IRSpatialShareConvolution") = fromSpatialShareConvolution
//      IR2DnnMap("IRSpatialMaxPooling") = fromSpatialMaxPooling
//      IR2DnnMap("IRSpatialAveragePooling") = fromSpatialAveragePooling
//      IR2DnnMap("IRSpatialBatchNormalization") = fromSpatialBatchNormalization
//      IR2DnnMap("IRSpatialCrossMapLRN") = fromSpatialCrossMapLRN
    IR2DnnMap.add("IRReLU"); //= fromReLU
    //IR2DnnMap.add("IRJoinTable"); //= fromJoinTable
    IR2DnnMap.add("IRGeneralModule"); //= fromBlasModule
    IR2DnnMap.add("IRInput"); //= fromInput
    //IR2DnnMap.add("IRSoftMax");// = fromSoftMax
  }

  @Override
  public boolean convertLayerCheck(IRElement layer) {
    String name = layer.getOp().name();
    if (IR2DnnMap.contains(name) && checkRequirement(layer)) {
      return true;
    }
    return ReflectionUtils.findClass(prefix + name.substring(2)) != null;
  }

  private AbstractModule getModule(String name, IRElement layer) {
    switch (name) {
      case "IRReLU":
        return fromReLU(layer);
      case "IRInput":
        return fromInput(layer);
      case "IRGeneralModule":
        return new BlasWrapper(((IRGeneralModule) layer.getOp()).getModel());
      default:
        throw new UnsupportedOperationException("Operation not supported");
    }
  }

  @Override
  public AbstractModule convertLayer(IRElement layer) {
    String name = layer.getOp().name();
    if (IR2DnnMap.contains(name)) {
      AbstractModule dnn = getModule(name, layer);
      if (!layer.getName().equals("")) {
        dnn.setName(layer.name);
      }
      return dnn;
    } else {
      try {
        return ReflectionUtils.reflectFromIR(layer, Class.forName(prefix + name.substring(2)));
      } catch (ClassNotFoundException e) {
        e.printStackTrace();
      }
    }
    return null;
  }

  @Override
  public boolean convertingCheck(List<Node<IRElement>> allNodes) {
    boolean convert = true;
    for (Node<IRElement> node : allNodes) {
      IROperator op = node.getElement().getOp();
      if (!convertLayerCheck(node.getElement())) {
        //logger.info("${node.getElement().getOp()} convertion failed");
        convert = false;
      }
    }

    return convert;
  }

  @Override
  public Map<Node<IRElement>, Node<AbstractModule>> convert(List<Node<IRElement>> allNodes) {
    Map<Node<IRElement>, Node<AbstractModule>> nodeMap = new HashMap<>();

    for (Node<IRElement> node : allNodes) {
      IROperator op = node.getElement().getOp();
      Node<AbstractModule> dnn;
      if (convertLayerCheck(node.getElement())) {
        dnn = new Node(convertLayer(node.getElement()));
      } else {
        throw new UnsupportedOperationException("can not find" + node.getElement().getOp());
      }
      // special treat for reshape -> linear and view -> linear
      if (op instanceof IRGeneralModule) {
        AbstractModule m = ((IRGeneralModule) op).getModel();
        if (m instanceof Reshape && node.nextNodes().size() == 1
            && node.nextNodes().get(0).getElement().getOp() instanceof IRLinear) {
          dnn = new Node(new edu.iu.dsc.tws.dl.module.mkldnn.Identity());
        } else if (m instanceof View && node.nextNodes().size() == 1
            && node.nextNodes().get(0).getElement().getOp() instanceof IRLinear) {
          dnn = new Node(new edu.iu.dsc.tws.dl.module.mkldnn.Identity());
        }
      }
      nodeMap.put(node, (Node<AbstractModule>) dnn);
    }

    cloneNode(allNodes, nodeMap);
    return nodeMap;
  }

  public AbstractModule fromReLU(IRElement node) {
    AbstractModule layer = new edu.iu.dsc.tws.dl.module.mkldnn.ReLU(0.0f);
    //setScales(node, layer);
    return layer;
  }

  private void setScales(IRElement fromEle, AbstractModule toELe) {
//    toELe.setInputScales(fromEle.getInputScales())
//    toELe.setOutputScales(fromEle.getOutputScales())
//    toELe.setWeightScales(fromEle.getWeightScales())
//
//    toELe.setInputDimMask(fromEle.getInputDimMask(), true)
//    toELe.setOutputDimMask(fromEle.getOutputDimMask(), true)
//    toELe.setWeightDimMask(fromEle.getWeightDimMask(), true)
  }

//    public AbstractModule fromSpatialConvolution(IRElement node){
//      val t = node.getOp().asInstanceOf[IRSpatialConvolution]
//    if (t.format == DataFormat.NCHW) {
//      ReflectionUtils.reflectFromIR(node, Class.forName(prefix + "SpatialConvolution"))
//    } else {
//      // special process for NHWC
//      require(t.nGroup == 1, "Only support nGroup is 1 for NHWC")
//      AbstractModule layer = ReflectionUtils.reflectFromIR(node,
//      Class.forName(prefix + "SpatialConvolution"))
//      val p = layer.parameters()
//      val weight = p._1(0)
//      val gradWeight = p._2(0)
//
//      val weightSize = weight.size() // for nchw
//      val newSize = Array(weightSize(2), weightSize(3), weightSize(1), weightSize(0))// for nhwc
//      val bufferNHWC = Tensor().resizeAs(weight).copy(weight).resize(newSize)
//      weight.copy(bufferNHWC.transpose(1, 4).transpose(2, 3).transpose(3, 4).contiguous())
//
//      bufferNHWC.copy(gradWeight)
//      gradWeight.copy(bufferNHWC.transpose(1, 4).transpose(2, 3).contiguous())
//
//      layer
//    }
//  }

//    public AbstractModule fromSpatialShareConvolution(IRElement node){
//      val t = node.getOp().asInstanceOf[IRSpatialShareConvolution]
//    if (t.format == DataFormat.NCHW) {
//      ReflectionUtils.reflectFromIR(node, Class.forName(prefix + "SpatialConvolution"))
//    } else {
//      // special process for NHWC
//      require(t.nGroup == 1, "Only support nGroup is 1 for NHWC")
//      AbstractModule layer = ReflectionUtils.reflectFromIR(node,
//      Class.forName(prefix + "SpatialConvolution"))
//      val p = layer.parameters()
//      val weight = p._1(0)
//      val gradWeight = p._2(0)
//
//      val weightSize = weight.size() // for nchw
//      val newSize = Array(weightSize(2), weightSize(3), weightSize(1), weightSize(0)) // for nhwc
//      val bufferNHWC = Tensor().resizeAs(weight).copy(weight).resize(newSize)
//      weight.copy(bufferNHWC.transpose(1, 4).transpose(2, 3).transpose(3, 4).contiguous())
//
//      bufferNHWC.copy(gradWeight)
//      gradWeight.copy(bufferNHWC.transpose(1, 4).transpose(2, 3).transpose(3, 4).contiguous())
//
//      layer
//    }
//  }

//    public AbstractModule fromSpatialMaxPooling(IRElement node){
//      val t = node.getOp().asInstanceOf[IRSpatialMaxPooling]
//      AbstractModule layer = ReflectionUtils.reflectFromIR(
//      node, Class.forName(prefix + "MaxPooling")).asInstanceOf[MaxPooling]
//    if (t.ceilMode) layer.ceil() else layer.floor()
//    layer
//  }

//    public AbstractModule fromSpatialAveragePooling(IRElement node){
//      val t = node.getOp().asInstanceOf[IRSpatialAveragePooling]
//      AbstractModule layer = ReflectionUtils.reflectFromIR(
//      node, Class.forName(prefix + "AvgPooling")).asInstanceOf[AvgPooling]
//    if (t.ceilMode) layer.ceil() else layer.floor()
//    layer
//  }

//    public AbstractModule fromSpatialCrossMapLRN(IRElement node){
//      val t = node.getOp().asInstanceOf[IRSpatialCrossMapLRN]
//      ReflectionUtils.reflectFromIR(node, Class.forName(prefix + "LRN"))
//  }

//    public AbstractModule fromJoinTable(IRElement node){
//      val t = node.getOp().asInstanceOf[IRJoinTable]
//      require(t.nInputDims <= 0,
//          s"Dnn JoinTable only supports nInputDims <= 0, but get ${t.nInputDims}")
//      new edu.iu.dsc.tws.dl.module.mkldnn.JoinTable(t.dimension)
//  }

//    public AbstractModule fromSpatialBatchNormalization(IRElement node){}
//      val t = node.getOp().asInstanceOf[IRSpatialBatchNormalization]
//      val nOutput = t.nOutput
//      val eps = t.eps
//      val momentum = 1 - t.momentum // meaning not same with mkldnn
//      val initWeight = t.initWeight
//      val initBias = t.initBias
//      val initGradWeight = t.initGradWeight
//      val initGradBias = t.initGradBias
//
//      AbstractModule layer = new edu.iu.dsc.tws.dl.module.mkldnn
//      .SpatialBatchNormalization(nOutput, eps, momentum,
//      true, initWeight, initBias, initGradWeight, initGradBias)
//
//      val params = node.getParameters()
//    if (params._1 != null) layer.weightAndBias.copy(params._1)
//    if (params._2 != null) layer.gradWeightAndBias.copy(params._2)
//
//    val extraParams = layer.getExtraParameter()
//    if (t.runningMean != null) extraParams(0).copy(t.runningMean.toTensor)
//    if (t.runningVar != null) extraParams(1).copy(t.runningVar.toTensor)
//
//    ReflectionUtils.setScales(node, layer)
//
//    // reminder: assume batch_norm is converted from blas
//    layer.needScale = true
//    layer
//  }

//    public AbstractModule fromSoftMax(IRElement node{
//      new edu.iu.dsc.tws.dl.module.mkldnn.SoftMax()
//  }

//    public AbstractModule fromBlasModule(IRElement node){
//      val model = node.getOp().asInstanceOf[IRGeneralModule].model
//    if (model instanceof BiRecurrent]) {
//      fromBiRecurrent(node)
//    } else if (model instanceof Recurrent]) {
//      fromRecurrent(node)
//    } else if (model instanceof TimeDistributed] &&
//        model.asInstanceOf[TimeDistributed].layer instanceof nn.SoftMax]) {
//      fromTimeDistributedWithSoftMax(node)
//    } else {
//      BlasWrapper(node.getOp().asInstanceOf[IRGeneralModule].model)
//    }
//  }

//    public AbstractModule fromRecurrent(IRElement node{
//      val model = node.getOp().asInstanceOf[IRGeneralModule]
//      .model.asInstanceOf[Recurrent]
//      AbstractModule layer = model.getCell()
//    if (layer instanceof LSTM] && model.batchNormParams ==  null) {
//      val lstm = layer.asInstanceOf[LSTM]
//      if (lstm.activation instanceof Tanh] &&
//          lstm.innerActivation instanceof Sigmoid] &&
//          lstm.p == 0.0f &&
//          lstm.wRegularizer == null &&
//          lstm.bRegularizer == null &&
//          lstm.uRegularizer == null) {
//        val f = AlgKind.EltwiseTanh
//        val direction = Direction.UnidirectionalLeft2Right
//        val inputSize = lstm.inputSize
//        val hiddenSize = lstm.hiddenSize
//        val lstmDnn = nn.new edu.iu.dsc.tws.dl.module.mkldnn.RNN(AlgKind.VanillaLstm,
//        inputSize, hiddenSize,
//            f, direction, layers = 1)
//
//        // copy weight from blas lstm to dnn lstm
//        val lstm_n_gates = 4
//
//        val blasParams = model.parameters()._1
//        val initWeight0 = blasParams(0)
//        val initBias0 = blasParams(1)
//        val initWeightIter0 = blasParams(2)
//
//        var num = initWeight0.size(1) / lstm_n_gates
//        var gate1 = initWeight0.narrow(1, 1, num)
//        var gate3 = initWeight0.narrow(1, num + 1, num)
//        var gate2 = initWeight0.narrow(1, num * 2 + 1, num)
//        var gate4 = initWeight0.narrow(1, num * 3 + 1, num)
//
//        var initWeight = Tensor(lstm_n_gates, hiddenSize, inputSize)
//        initWeight.select(1, 1).copy(gate1)
//        initWeight.select(1, 2).copy(gate2)
//        initWeight.select(1, 3).copy(gate3)
//        initWeight.select(1, 4).copy(gate4)
//        // original Array(inputSize, lstm_n_gates, hiddenSize)
//        initWeight = initWeight.transpose(1, 3).transpose(2, 3)
//
//        num = initBias0.size(1) / lstm_n_gates
//        gate1 = initBias0.narrow(1, 1, num)
//        gate3 = initBias0.narrow(1, num + 1, num)
//        gate2 = initBias0.narrow(1, num * 2 + 1, num)
//        gate4 = initBias0.narrow(1, num * 3 + 1, num)
//
//        val initBias = Tensor(lstm_n_gates, hiddenSize)
//        initBias.select(1, 1).copy(gate1)
//        initBias.select(1, 2).copy(gate2)
//        initBias.select(1, 3).copy(gate3)
//        initBias.select(1, 4).copy(gate4)
//
//        num = initWeightIter0.size(1) / lstm_n_gates
//        gate1 = initWeightIter0.narrow(1, 1, num)
//        gate3 = initWeightIter0.narrow(1, num + 1, num)
//        gate2 = initWeightIter0.narrow(1, num * 2 + 1, num)
//        gate4 = initWeightIter0.narrow(1, num * 3 + 1, num)
//
//        var initIterWeight = Tensor(lstm_n_gates, hiddenSize, hiddenSize)
//        initIterWeight.select(1, 1).copy(gate1)
//        initIterWeight.select(1, 2).copy(gate2)
//        initIterWeight.select(1, 3).copy(gate3)
//        initIterWeight.select(1, 4).copy(gate4)
//        // original Array(hiddenSize, lstm_n_gates, hiddenSize)
//        initIterWeight = initIterWeight.transpose(1, 3).transpose(2, 3)
//
//        val weights = lstmDnn.parameters()._1
//        weights(0).copy(initWeight)
//        weights(1).copy(initBias)
//        weights(2).copy(initIterWeight)
//
//        return lstmDnn
//      }
//    }
//
//    if (layer instanceof GRU] && model.batchNormParams ==  null) {
//      val gru = layer.asInstanceOf[GRU]
//      if (gru.activation instanceof Tanh] &&
//          gru.innerActivation instanceof Sigmoid] &&
//          gru.p == 0.0f &&
//          gru.wRegularizer == null &&
//          gru.bRegularizer == null &&
//          gru.uRegularizer == null) {
//        val f = AlgKind.EltwiseTanh
//        val direction = Direction.UnidirectionalLeft2Right
//        val inputSize = gru.inputSize
//        val hiddenSize = gru.outputSize
//        val gruDnn = nn.new edu.iu.dsc.tws.dl.module.mkldnn.RNN(AlgKind.VanillaGru,
//        inputSize, hiddenSize,
//            f, direction, layers = 1)
//
//        // copy weight from blas gru to dnn gru
//        val gru_n_gates = 3
//
//        val blasParams = model.parameters()._1
//        val initWeight0 = blasParams(0)
//        val initBias0 = blasParams(1)
//        // blas gru splits weight iteration into 2 tensors
//        val initWeightIter0 = blasParams(2)
//        val initWeightIter1 = blasParams(3)
//
//        var num = initWeight0.size(1) / gru_n_gates
//        var gate2 = initWeight0.narrow(1, 1, num)
//        var gate1 = initWeight0.narrow(1, num + 1, num)
//        var gate3 = initWeight0.narrow(1, num * 2 + 1, num)
//
//        var initWeight = Tensor(gru_n_gates, hiddenSize, inputSize)
//        initWeight.select(1, 1).copy(gate1)
//        initWeight.select(1, 2).copy(gate2)
//        initWeight.select(1, 3).copy(gate3)
//        // original Array(inputSize, gru_n_gates, hiddenSize)
//        initWeight = initWeight.transpose(1, 3).transpose(2, 3)
//
//        num = initBias0.size(1) / gru_n_gates
//        gate2 = initBias0.narrow(1, 1, num)
//        gate1 = initBias0.narrow(1, num + 1, num)
//        gate3 = initBias0.narrow(1, num * 2 + 1, num)
//
//        val initBias = Tensor(gru_n_gates, hiddenSize)
//        initBias.select(1, 1).copy(gate1)
//        initBias.select(1, 2).copy(gate2)
//        initBias.select(1, 3).copy(gate3)
//
//        num = initWeightIter0.size(1) / 2
//        gate2 = initWeightIter0.narrow(1, 1, num)
//        gate1 = initWeightIter0.narrow(1, num + 1, num)
//
//        num = initWeightIter1.size(1) / 1
//        gate3 = initWeightIter1.narrow(1, 1, num)
//
//        var initIterWeight = Tensor(gru_n_gates, hiddenSize, hiddenSize)
//        initIterWeight.select(1, 1).copy(gate1)
//        initIterWeight.select(1, 2).copy(gate2)
//        initIterWeight.select(1, 3).copy(gate3)
//        // original Array(hiddenSize, gru_n_gates, hiddenSize)
//        initIterWeight = initIterWeight.transpose(1, 3).transpose(2, 3)
//
//        val weights = gruDnn.parameters()._1
//        weights(0).copy(initWeight)
//        weights(1).copy(initBias)
//        weights(2).copy(initIterWeight)
//
//        return gruDnn
//      }
//    }
//
//    BlasWrapper(node.getOp().asInstanceOf[IRGeneralModule].model)
//  }

//    public AbstractModule fromBiRecurrent(IRElement node{
//      val model = node.getOp().asInstanceOf[IRGeneralModule]
//      .model.asInstanceOf[BiRecurrent]
//      AbstractModule layer = model.layer.getCell()
//      val revLayer = model.revLayer.getCell()
//      val merge = model.getMerge()
//    if ((layer equals revLayer) && layer instanceof LSTM] &&
//      model.batchNormParams ==  null && model.isSplitInput == false &&
//      (merge instanceof nn.CAddTable[Float, _]] || merge instanceof nn.ConcatTable])) {
//    val lstm = layer.asInstanceOf[LSTM]
//    if (lstm.activation instanceof Tanh] &&
//        lstm.innerActivation instanceof Sigmoid] &&
//        lstm.p == 0.0f &&
//        lstm.wRegularizer == null &&
//        lstm.bRegularizer == null &&
//        lstm.uRegularizer == null) {
//      val f = AlgKind.EltwiseTanh
//      val direction = if (merge instanceof nn.CAddTable[Float, _]]) {
//        Direction.BidirectionalSum
//      } else Direction.BidirectionalConcat
//      val inputSize = lstm.inputSize
//      val hiddenSize = lstm.hiddenSize
//      val lstmDnn = nn.new edu.iu.dsc.tws.dl.module.mkldnn.RNN(AlgKind.VanillaLstm,
//      inputSize, hiddenSize,
//          f, direction, layers = 1)
//
//      // copy weight from blas lstm to dnn lstm
//      val lstm_n_gates = 4
//
//      val blasParams = model.parameters()._1
//      val initWeight0 = Tensor(Array(2, hiddenSize * lstm_n_gates, inputSize))
//      val initWeightIter0 = Tensor(Array(2, hiddenSize * lstm_n_gates, hiddenSize))
//      val initBias0 = Tensor(Array(2, lstm_n_gates * hiddenSize))
//
//      initWeight0(1).resizeAs(blasParams(0)).copy(blasParams(0))
//      initBias0(1).resizeAs(blasParams(1)).copy(blasParams(1))
//      initWeightIter0(1).resizeAs(blasParams(2)).copy(blasParams(2))
//      initWeight0(2).resizeAs(blasParams(3)).copy(blasParams(3))
//      initBias0(2).resizeAs(blasParams(4)).copy(blasParams(4))
//      initWeightIter0(2).resizeAs(blasParams(5)).copy(blasParams(5))
//
//      val initWeight = Tensor(Array(2, lstm_n_gates, hiddenSize, inputSize))
//      val initWeightIter = Tensor(Array(2, lstm_n_gates, hiddenSize, hiddenSize))
//      val initBias = Tensor(Array(2, lstm_n_gates, hiddenSize))
//
//      for (i <- 1 to 2) {
//        var num = initWeight0(i).size(1) / lstm_n_gates
//        var gate1 = initWeight0(i).narrow(1, 1, num)
//        var gate3 = initWeight0(i).narrow(1, num + 1, num)
//        var gate2 = initWeight0(i).narrow(1, num * 2 + 1, num)
//        var gate4 = initWeight0(i).narrow(1, num * 3 + 1, num)
//        initWeight(i).select(1, 1).copy(gate1)
//        initWeight(i).select(1, 2).copy(gate2)
//        initWeight(i).select(1, 3).copy(gate3)
//        initWeight(i).select(1, 4).copy(gate4)
//
//        num = initWeightIter0(i).size(1) / 4
//        gate1 = initWeightIter0(i).narrow(1, 1, num)
//        gate3 = initWeightIter0(i).narrow(1, num + 1, num)
//        gate2 = initWeightIter0(i).narrow(1, num * 2 + 1, num)
//        gate4 = initWeightIter0(i).narrow(1, num * 3 + 1, num)
//        initWeightIter(i).select(1, 1).copy(gate1)
//        initWeightIter(i).select(1, 2).copy(gate2)
//        initWeightIter(i).select(1, 3).copy(gate3)
//        initWeightIter(i).select(1, 4).copy(gate4)
//
//        num = initBias0(i).size(1) / 4
//        gate1 = initBias0(i).narrow(1, 1, num)
//        gate3 = initBias0(i).narrow(1, num + 1, num)
//        gate2 = initBias0(i).narrow(1, num * 2 + 1, num)
//        gate4 = initBias0(i).narrow(1, num * 3 + 1, num)
//        initBias(i).select(1, 1).copy(gate1)
//        initBias(i).select(1, 2).copy(gate2)
//        initBias(i).select(1, 3).copy(gate3)
//        initBias(i).select(1, 4).copy(gate4)
//      }
//      val weights = lstmDnn.parameters()._1
//      weights(0).copy(initWeight.transpose(2, 4).transpose(3, 4))
//      weights(1).copy(initBias)
//      weights(2).copy(initWeightIter.transpose(2, 4).transpose(3, 4))
//
//      return lstmDnn
//    }
//  }
//
//    if ((layer equals revLayer) && layer instanceof GRU] &&
//      model.batchNormParams ==  null && model.isSplitInput == false &&
//      (merge instanceof nn.CAddTable[Float, _]] || merge instanceof nn.ConcatTable])) {
//    val gru = layer.asInstanceOf[GRU]
//    if (gru.activation instanceof Tanh] &&
//        gru.innerActivation instanceof Sigmoid] &&
//        gru.p == 0.0f &&
//        gru.wRegularizer == null &&
//        gru.bRegularizer == null &&
//        gru.uRegularizer == null) {
//      val f = AlgKind.EltwiseTanh
//      val direction = if (merge instanceof nn.CAddTable[Float, _]]) {
//        Direction.BidirectionalSum
//      } else Direction.BidirectionalConcat
//      val inputSize = gru.inputSize
//      val hiddenSize = gru.outputSize
//      val gruDnn = nn.new edu.iu.dsc.tws.dl.module.mkldnn.RNN(AlgKind.VanillaGru,
//      inputSize, hiddenSize,
//          f, direction, layers = 1)
//
//      // copy weight from blas gru to dnn gru
//      val gru_n_gates = 3
//
//      val blasParams = model.parameters()._1
//      val initWeight0 = Tensor(Array(2, hiddenSize * gru_n_gates, inputSize))
//      // blas gru splits weight iteration into 2 tensors
//      val initWeightIter0 = Tensor(Array(2, hiddenSize * 2, hiddenSize))
//      val initWeightIter1 = Tensor(Array(2, hiddenSize * 1, hiddenSize))
//      val initBias0 = Tensor(Array(2, gru_n_gates * hiddenSize))
//
//      initWeight0(1).resizeAs(blasParams(0)).copy(blasParams(0))
//      initBias0(1).resizeAs(blasParams(1)).copy(blasParams(1))
//      initWeightIter0(1).resizeAs(blasParams(2)).copy(blasParams(2))
//      initWeightIter1(1).resizeAs(blasParams(3)).copy(blasParams(3))
//      initWeight0(2).resizeAs(blasParams(4)).copy(blasParams(4))
//      initBias0(2).resizeAs(blasParams(5)).copy(blasParams(5))
//      initWeightIter0(2).resizeAs(blasParams(6)).copy(blasParams(6))
//      initWeightIter1(2).resizeAs(blasParams(7)).copy(blasParams(7))
//
//      val initWeight = Tensor(Array(2, gru_n_gates, hiddenSize, inputSize))
//      val initWeightIter = Tensor(Array(2, gru_n_gates, hiddenSize, hiddenSize))
//      val initBias = Tensor(Array(2, gru_n_gates, hiddenSize))
//
//      for (i <- 1 to 2) {
//        var num = initWeight0(i).size(1) / gru_n_gates
//        var gate2 = initWeight0(i).narrow(1, 1, num)
//        var gate1 = initWeight0(i).narrow(1, num + 1, num)
//        var gate3 = initWeight0(i).narrow(1, num * 2 + 1, num)
//        initWeight(i).select(1, 1).copy(gate1)
//        initWeight(i).select(1, 2).copy(gate2)
//        initWeight(i).select(1, 3).copy(gate3)
//
//        num = initWeightIter0(i).size(1) / 2
//        gate2 = initWeightIter0(i).narrow(1, 1, num)
//        gate1 = initWeightIter0(i).narrow(1, num + 1, num)
//        initWeightIter(i).select(1, 1).copy(gate1)
//        initWeightIter(i).select(1, 2).copy(gate2)
//
//        num = initWeightIter1(i).size(1) / 1
//        gate3 = initWeightIter1(i).narrow(1, 1, num)
//        initWeightIter(i).select(1, 3).copy(gate3)
//
//        num = initBias0(i).size(1) / gru_n_gates
//        gate2 = initBias0(i).narrow(1, 1, num)
//        gate1 = initBias0(i).narrow(1, num + 1, num)
//        gate3 = initBias0(i).narrow(1, num * 2 + 1, num)
//        initBias(i).select(1, 1).copy(gate1)
//        initBias(i).select(1, 2).copy(gate2)
//        initBias(i).select(1, 3).copy(gate3)
//      }
//      val weights = gruDnn.parameters()._1
//      weights(0).copy(initWeight.transpose(2, 4).transpose(3, 4))
//      weights(1).copy(initBias)
//      weights(2).copy(initWeightIter.transpose(2, 4).transpose(3, 4))
//
//      return gruDnn
//    }
//  }
//
//    BlasWrapper(node.getOp().asInstanceOf[IRGeneralModule].model)
//  }

//    public AbstractModule fromTimeDistributedWithSoftMax(IRElement node{
//      new edu.iu.dsc.tws.dl.module.mkldnn.SoftMax(axis = 2)
//  }

  public AbstractModule fromInput(IRElement node) {
    return new edu.iu.dsc.tws.dl.module.mkldnn.Identity();
  }

  public boolean checkRequirement(IRElement layer) {
    return true;
//    try {
//      layer.getOp() match {
//        case join: IRJoinTable =>
//          require(join.nInputDims <= 0)
//        case _ => null
//      }
//      true
//    } catch {
//    case e: Throwable => false
//  }
  }
}


