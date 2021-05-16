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

import java.util.ArrayList;

import com.intel.analytics.bigdl.mkl.DataType;
import com.intel.analytics.bigdl.mkl.Memory;
import com.intel.analytics.bigdl.mkl.PropKind;
import com.intel.analytics.bigdl.mkl.Query;

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.TensorNumeric;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.MklDnnMemory;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.HeapData;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.NativeData;
import edu.iu.dsc.tws.dl.optim.Initializable;
import edu.iu.dsc.tws.dl.optim.InitializationMethod;
import edu.iu.dsc.tws.dl.optim.Regularizer;
import edu.iu.dsc.tws.dl.optim.initialization.RandomUniform;
import edu.iu.dsc.tws.dl.optim.initialization.Zeros;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.ArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.MemoryDataArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;
import edu.iu.dsc.tws.dl.utils.varformat.ONE_D;
import edu.iu.dsc.tws.dl.utils.varformat.OUT_IN;

public class Linear extends MklDnnLayer implements Initializable {

  protected InitializationMethod weightInitMethod = new Zeros();
  protected InitializationMethod biasInitMethod = new Zeros();

  private int inputSize;
  private int outputSize;
  private Regularizer wRegularizer;
  private Regularizer bRegularizer;
  private Tensor initWeight;
  private Tensor initBias;
  private Tensor initGradWeight;
  private Tensor initGradBias;
  private boolean withBias = true;

  private TensorMMap weight;
  private TensorMMap bias;
  private TensorMMap gradWeight;
  private TensorMMap gradBias;

  private transient long forwardPrimDesc = 0L;
  private transient int[] weightShape = null;
  private transient int weightLayout = -1;

  private transient long[] updateOutputMemoryPrimitives;
  private transient Tensor[] updateOutputTensors;
  private transient long[] updateGradInputMemoryPrimitives;
  private transient Tensor[] updateGradInputTensors;
  private transient long[] updateGradWMemoryPrimitives;
  private transient Tensor[] updateGradWTensors;

  public Linear(int inputSize, int outputSize, Regularizer wRegularizer, Regularizer bRegularizer,
                Tensor initWeight, Tensor initBias, Tensor initGradWeight, Tensor initGradBias,
                boolean withBias) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.wRegularizer = wRegularizer;
    this.bRegularizer = bRegularizer;
    this.initWeight = initWeight;
    this.initBias = initBias;
    this.initGradWeight = initGradWeight;
    this.initGradBias = initGradBias;
    this.withBias = withBias;
    init();
  }

  private void init() {
    weight = new TensorMMap(new int[]{outputSize, inputSize}, this);
    bias = new TensorMMap(new int[]{outputSize}, this);
    gradWeight = new TensorMMap(new int[]{outputSize, inputSize}, this);
    gradBias = new TensorMMap(new int[]{outputSize}, this);

    double stdv = 1.0 / Math.sqrt(weight.size(2));
    weightInitMethod = new RandomUniform(-stdv, stdv);
    biasInitMethod = new RandomUniform(-stdv, stdv);
  }

  @Override
  public void reset() {
    if (initWeight == null) {
      weightInitMethod.init(weight.getDense(), new OUT_IN());
    } else {
      weight.getDense().copy(initWeight);
    }

    if (initBias == null) {
      biasInitMethod.init(bias.getDense(), new ONE_D());
    } else {
      bias.getDense().copy(initBias);
    }
  }

  @Override
  public MemoryDataArrayPair initFwdPrimitives(MemoryData[] inputs, Phase phase) {

    int[] weightParams1 = new int[0];
    int weightParams2 = 0;
    switch (inputs[0].shape().length) {
      case 4:
        if (inputs[0].getHeapFormat() == Memory.Format.nhwc) {
          weightParams2 = Memory.Format.nhwc;
          weightParams1 = new int[4];
          weightParams1[0] = weight.size(1);
          System.arraycopy(inputs[0].shape(), 1, weightParams1, 1, 3);
          // ohwi
        } else {
          weightParams2 = Memory.Format.nchw;
          weightParams1 = new int[4];
          weightParams1[0] = weight.size(1);
          System.arraycopy(inputs[0].shape(), 1, weightParams1, 1, 3);
          // oihw
        }
        break;
      case 2:
        weightParams1 = weight.size();
        weightParams2 = Memory.Format.nc;
        break;
      case 1:
        weightParams1 = weight.size();
        weightParams2 = Memory.Format.x;
        break;
      default:
        break;
    }

    weightShape = weightParams1;
    weightLayout = weightParams2;

    int[] inputShape = inputs[0].shape();
    Util.require(inputs[0].shape().length > 1,
        "mkldnn linear unspported input dimension");

    int[] outputShape = new int[]{inputs[0].shape()[0], outputSize};

    MklDnnMemory.MemoryDescInit(inputShape.length, inputShape,
        DataType.F32, Memory.Format.any, this);

    NativeData src = new NativeData(inputShape, Memory.Format.any);
    NativeData wei = new NativeData(weightShape, Memory.Format.any);
    NativeData bis = new NativeData(bias.size(), Memory.Format.x);
    NativeData dst = new NativeData(outputShape, Memory.Format.any);

    long desc = MklDnnMemory.LinearForwardDescInit(
        PropKind.Forward,
        src.getMemoryDescription(this),
        wei.getMemoryDescription(this),
        bis.getMemoryDescription(this),
        dst.getMemoryDescription(this), this);
    forwardPrimDesc = MklDnnMemory.PrimitiveDescCreate(desc, runtime.engine, 0, this);

    MemoryData realSrc = MemoryData.operationWant(forwardPrimDesc, Query.SrcPd);
    MemoryData realWei = MemoryData.operationWant(forwardPrimDesc, Query.WeightsPd);
    MemoryData realDst = MemoryData.operationWant(forwardPrimDesc, Query.DstPd);

    Util.require(TensorNumeric.product(weight.size()) == TensorNumeric
            .product(realWei.shape()), "${getName} weight shape is not correct.");

    weight.setMemoryData(new HeapData(weightShape, weightLayout), realWei, runtime);
    bias.setMemoryData(new HeapData(bis.shape(), Memory.Format.x), bis, runtime);

    weight.sync();
    bias.sync();

    long[] srcs = new long[]{realSrc.getPrimitive(runtime, this),
        realWei.getPrimitive(runtime, this),
        bis.getPrimitive(runtime, this)};
    int[] indexes = new int[srcs.length];
    long[] dsts = new long[]{realDst.getPrimitive(runtime, this)};

    long primitive = MklDnnMemory.PrimitiveCreate2(forwardPrimDesc, srcs, indexes, srcs.length,
        dsts, dsts.length, this);

    long[] tempSum = new long[srcs.length + dsts.length];
    System.arraycopy(srcs, 0, tempSum, 0, srcs.length);
    System.arraycopy(dsts, 0, tempSum, srcs.length, dsts.length);
    updateOutputMemoryPrimitives = tempSum;
    updateOutputPrimitives = new long[]{primitive};
    output = initTensor(realDst);

    _inputFormats = new MemoryData[]{realSrc};
    _outputFormats = new MemoryData[]{realDst};
    return new MemoryDataArrayPair(_inputFormats, _outputFormats);
  }

  @Override
  public Activity updateOutput(Activity input) {
    if (updateOutputTensors == null) {
      ArrayList<Tensor> buffer = new ArrayList<>();
      buffer.add((Tensor) input);
      buffer.add(weight.nativeDnn());
      buffer.add(bias.nativeDnn());
      buffer.add((Tensor) output);
      updateOutputTensors = new Tensor[buffer.size()];
      buffer.toArray(updateOutputTensors);
    }

    updateWithNewTensor(updateOutputTensors, 0, input);

    if (isTraining()) {
      weight.sync();
      bias.sync();
    }

    MklDnnOps.streamSubmit(runtime.stream, 1, updateOutputPrimitives,
        updateOutputPrimitives.length, updateOutputMemoryPrimitives, updateOutputTensors);

    return output;
  }

  @Override
  public MemoryDataArrayPair initBwdPrimitives(MemoryData[] grad, Phase phase) {
    int[] inputShape = inputFormats()[0].shape();

    int[] outputShape = new int[]{inputFormats()[0].shape()[0], outputSize};

    NativeData src = new NativeData(inputShape, Memory.Format.any);
    NativeData wei = new NativeData(weightShape, Memory.Format.any);
    NativeData bis = new NativeData(bias.size(), Memory.Format.x);
    NativeData dst = new NativeData(outputShape, Memory.Format.any);

    long desc = MklDnnMemory.LinearBackwardDataDescInit(
        src.getMemoryDescription(this),
        wei.getMemoryDescription(this),
        grad[0].getMemoryDescription(this), this);
    long backwardPrimDesc = MklDnnMemory.PrimitiveDescCreate(desc, runtime.engine,
        forwardPrimDesc, this);

    MemoryData realDiffSrc = MemoryData.operationWant(backwardPrimDesc, Query.DiffSrcPd);
    MemoryData realWei = MemoryData.operationWant(backwardPrimDesc, Query.WeightsPd);
    MemoryData realDiffDst = MemoryData.operationWant(backwardPrimDesc, Query.DiffDstPd);


    long[] srcs = new long[]{realDiffDst.getPrimitive(runtime, this),
        realWei.getPrimitive(runtime, this)};
    int[] indexes = new int[srcs.length];
    long[] dsts = new long[]{realDiffSrc.getPrimitive(runtime, this)};

    long primitive = MklDnnMemory.PrimitiveCreate2(backwardPrimDesc, srcs, indexes, srcs.length,
        dsts, dsts.length, this);

    long[] tempSum = new long[srcs.length + dsts.length];
    System.arraycopy(srcs, 0, tempSum, 0, srcs.length);
    System.arraycopy(dsts, 0, tempSum, srcs.length, dsts.length);
    updateGradInputMemoryPrimitives = tempSum;
    updateGradInputPrimitives = new long[]{primitive};
    gradInput = initTensor(realDiffSrc);

    _gradInputFormats = new MemoryData[]{realDiffSrc};
    _gradOutputFormats = new MemoryData[]{realDiffDst};
    return new MemoryDataArrayPair(_gradOutputFormats, _gradInputFormats);
  }

  @Override
  public MemoryData[] initGradWPrimitives(MemoryData[] grad, Phase phase) {
    int[] inputShape = inputFormats()[0].shape();

    int[] outputShape = new int[]{inputFormats()[0].shape()[0], outputSize};


    NativeData src = new NativeData(inputShape, Memory.Format.any);
    NativeData wei = new NativeData(weightShape, Memory.Format.any);
    NativeData bis = new NativeData(bias.size(), Memory.Format.x);
    NativeData dst = new NativeData(outputShape, Memory.Format.any);

    long desc = MklDnnMemory.LinearBackwardWeightsDescInit(
        src.getMemoryDescription(this), wei.getMemoryDescription(this),
        bis.getMemoryDescription(this),
        dst.getMemoryDescription(this), this);
    long gradWeightPrimDesc = MklDnnMemory.PrimitiveDescCreate(desc, runtime.engine,
        forwardPrimDesc, this);

    MemoryData realWei = MemoryData.operationWant(gradWeightPrimDesc, Query.DiffWeightsPd);
    MemoryData realDiffDst = MemoryData.operationWant(gradWeightPrimDesc, Query.DiffDstPd);

    gradWeight.setMemoryData(realWei, new HeapData(weightShape, weightLayout),
        runtime);
    gradBias.setMemoryData(bis, new HeapData(bis.shape(), Memory.Format.x), runtime);

    gradWeight.zero();
    gradBias.zero();

    long[] srcs = new long[]{inputFormats()[0].getPrimitive(runtime, this),
        realDiffDst.getPrimitive(runtime, this)};
    int[] indexes = new int[srcs.length];
    long[] dsts = new long[]{realWei.getPrimitive(runtime, this),
        bis.getPrimitive(runtime, this)};

    long primitive = MklDnnMemory.PrimitiveCreate2(gradWeightPrimDesc, srcs, indexes, srcs.length,
        dsts, dsts.length, this);

    long[] tempSum = new long[srcs.length + dsts.length];
    System.arraycopy(srcs, 0, tempSum, 0, srcs.length);
    System.arraycopy(dsts, 0, tempSum, srcs.length, dsts.length);
    updateGradWMemoryPrimitives = tempSum;
    accGradientPrimitives = new long[]{primitive};

    _gradOutputFormatsForWeight = new MemoryData[]{realDiffDst};
    return _gradOutputFormatsForWeight;
  }

  @Override
  public Activity updateGradInput(Activity input, Activity gradOutput) {
    if (updateGradInputTensors == null) {
      ArrayList<Tensor> buffer = new ArrayList<>();
      buffer.add((Tensor) gradOutput);
      buffer.add(weight.nativeDnn());
      buffer.add((Tensor) gradInput);
      updateGradInputTensors = new Tensor[buffer.size()];
      buffer.toArray(updateGradInputTensors);
    }

    updateWithNewTensor(updateGradInputTensors, 0, gradOutput);

    MklDnnOps.streamSubmit(runtime.stream, 1, updateGradInputPrimitives,
        updateGradInputPrimitives.length, updateGradInputMemoryPrimitives, updateGradInputTensors);

    return gradInput;
  }

  @Override
  public void accGradParameters(Activity input, Activity gradOutput) {
    if (updateGradWTensors == null) {
      ArrayList<Tensor> buffer = new ArrayList<>();
      buffer.add((Tensor) input);
      buffer.add((Tensor) gradOutput);
      buffer.add(gradWeight.nativeDnn());
      buffer.add(gradBias.nativeDnn());
      updateGradWTensors = new Tensor[buffer.size()];
      buffer.toArray(updateGradWTensors);
    }

    // do not use the updateGradInputTensors for acc
    updateWithNewTensor(updateGradWTensors, 0, input);
    updateWithNewTensor(updateGradWTensors, 1, gradOutput);

    MklDnnOps.streamSubmit(runtime.stream, 1, accGradientPrimitives,
        accGradientPrimitives.length, updateGradWMemoryPrimitives, updateGradWTensors);

    gradWeight.sync();
    gradBias.sync();

    if (null != wRegularizer && scaleW != 0) {
      wRegularizer.accRegularization(weight.getDense(), gradWeight.getDense(), scaleW);
    }
    if (null != bRegularizer && scaleB != 0) {
      bRegularizer.accRegularization(bias.getDense(), gradBias.getDense(), scaleB);
    }
  }

  @Override
  public TensorArrayPair parameters() {
    return new TensorArrayPair(new Tensor[]{weight.getDense(), bias.getDense()},
        new Tensor[]{gradWeight.getDense(), gradBias.getDense()});
  }

  @Override
  public ArrayPair<TensorMMap> paramsMMap() {
    return new ArrayPair<TensorMMap>(new TensorMMap[]{weight, bias},
        new TensorMMap[]{gradWeight, gradBias});
  }

  @Override
  public void zeroGradParameters() {
  }

  @Override
  public Initializable setInitMethod(InitializationMethod weightMethod,
                                     InitializationMethod biasMethod) {
    if (weightInitMethod != null) {
      this.weightInitMethod = weightMethod;
    }

    if (biasInitMethod != null) {
      this.biasInitMethod = biasMethod;
    }
    reset();
    return this;
  }
}

