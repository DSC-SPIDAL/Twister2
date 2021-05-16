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

import com.intel.analytics.bigdl.mkl.Memory;

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.module.AbstractModule;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.HeapData;
import edu.iu.dsc.tws.dl.utils.MultiShape;
import edu.iu.dsc.tws.dl.utils.Shape;
import edu.iu.dsc.tws.dl.utils.SingleShape;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.MemoryDataArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

public class BlasWrapper extends MklDnnLayer {

  private boolean needOutputFormats = true;

  private transient AbstractModule[] subModels;
  private transient int subModelNumber = 1;
  private transient boolean withMultiThread = false;
  private transient Activity[] inputBuffer;
  private transient Tensor[] tensorBuffer;
  private transient int batchSize;
  private transient boolean initEnv = false;
  private AbstractModule<Activity> module;

  public BlasWrapper(AbstractModule<Activity> modal) {
    super();
    Util.require(!(module instanceof MklDnnModule),
        "Only support wrapper blas layer to dnn layer");
    this.module = modal;
    this.output = module.output;
    this.gradInput = module.gradInput;
  }

  public boolean isNeedOutputFormats() {
    return needOutputFormats;
  }

  public void setNeedOutputFormats(boolean needOutputFormats) {
    this.needOutputFormats = needOutputFormats;
  }


  @Override
  public TensorArrayPair parameters() {
    return module.parameters();
  }

  @Override
  public void reset() {
  }

  @Override
  public MemoryDataArrayPair initFwdPrimitives(MemoryData[] inputs, Phase phase) {
    _inputFormats = inferInputFormats(inputs);
    if (needOutputFormats) {
      _outputFormats = inferOutputFormats(_inputFormats);
    } else {
      _outputFormats = null;
    }
    if (_outputFormats != null) {
      for (MemoryData outputFormat : _outputFormats) {
        outputFormat.getPrimitive(runtime, this);
      }
    }
    return new MemoryDataArrayPair(_inputFormats, _outputFormats);
  }

  @Override
  public MemoryDataArrayPair initBwdPrimitives(MemoryData[] grad, Phase phase) {
    _gradOutputFormats = _outputFormats;
    _gradInputFormats = _inputFormats;
    return new MemoryDataArrayPair(_gradOutputFormats, _gradInputFormats);
  }

  @Override
  public MemoryData[] initGradWPrimitives(MemoryData[] grad, Phase phase) {
    _gradOutputFormatsForWeight = _outputFormats;
    return _gradOutputFormatsForWeight;
  }

  // reminder: for dim 3, there may be ntc or tnc, now we just support ntc
  private int getFormats(int dims) {
    switch (dims) {
      case 4:
        return Memory.Format.nchw;
      case 3:
        return Memory.Format.ntc;
      case 2:
        return Memory.Format.nc;
      case 1:
        return Memory.Format.x;
      default:
        throw new UnsupportedOperationException("Operation not supported");
    }
  }

  private int getHeapFormats(MemoryData in) {
    if (in.getHeapFormat() == -1 || in.shape().length != 4) {
      return getFormats(in.shape().length);
    } else {
      return in.getHeapFormat();
    }
  }

  private MemoryData[] inferInputFormats(MemoryData[] inputs) {
    MemoryData[] results = new MemoryData[inputs.length];
    for (int i = 0; i < inputs.length; i++) {
      MemoryData in = inputs[i];
      HeapData heap;
      if (in.layout() == Memory.Format.tnc) {
        int[] size = in.shape();
        heap = new HeapData(new int[]{size[1], size[0], size[2]}, Memory.Format.ntc);
      } else {
        heap = new HeapData(in.shape(), getHeapFormats(in));
      }
      heap.setHeapFormat(in.getHeapFormat());
      results[i] = heap;
    }
    return results;
  }

  private MemoryData[] inferOutputFormats(MemoryData[] inputs) {
    Shape[] inputShape = new Shape[inputs.length];
    for (int i = 0; i < inputs.length; i++) {
      MemoryData in = inputs[i];
      inputShape[i] = new SingleShape(in.getHeapShape());
    }

    Shape[] outputShape;
    if (inputShape.length == 1) {
      outputShape = new Shape[]{module.computeOutputShape(inputShape[0])};
    } else {
      // multi shape
      Shape out = module.computeOutputShape(new MultiShape(inputShape));
      if (out instanceof MultiShape) {
        outputShape = out.toMulti();
      } else {
        outputShape = new Shape[]{out};
      }
    }

    MemoryData[] results = new MemoryData[outputShape.length];
    for (int i = 0; i < outputShape.length; i++) {
      Shape in = outputShape[i];
      int[] size = in.toSingle();
      int f;
      if (size.length == 4 && inputs[0].getHeapFormat() == Memory.Format.nhwc) {
        f = Memory.Format.nhwc;
      } else {
        f = getFormats(size.length);
      }
      int[] outSize;
      if (f == Memory.Format.nhwc) {
        outSize = new int[]{size[0], size[3], size[1], size[2]};
      } else {
        outSize = size;
      }
      results[i] = new HeapData(outSize, f).setHeapFormat(f);
    }

    return results;
  }

  /**
   * Blas layers normally do not have competitive performance when running under mkldnn.
   * So we can leverage multi-threading to resolve bottleneck introduced by one model only
   * for mkl-dnn backend. The parallelism is determined by both bath size and core number,
   * with restrictions that both input and output format must be batched.
   */
  private void setMultiThreadEnv(Activity input) {
//    this.initEnv = true;
//    val multiThread = System.getProperty("multiThread", "false").toBoolean
//    if (this.train && multiThread) {
//      throw new IllegalArgumentException("Please not set multiThread to true for model training")
//    }
//    if (this.train
//        || !multiThread
//        || (_outputFormats != null && _outputFormats.length != 1)
//        || (_outputFormats != null && _inputFormats != null
//        && _inputFormats(0).shape(0) != _outputFormats(0).shape(0))
//        || !flattenInput(input)
//    ) {
//      return
//    }
//    batchSize = tensorBuffer(0).size(1)
//    val residue = batchSize % Engine.coreNumber()
//    if (residue != 0 || batchSize < 2 || Engine.coreNumber() < 2) {
//      logger.warn("If you want to use multiThread property to speed up, " +
//          "please attention core number should be greater than 1, " +
//          s"batch size should be greater than 1 and divided by core number, " +
//          s"but now get core number ${Engine.coreNumber()} batch size ${batchSize}")
//      return
//    }
//    subModelNumber = Engine.coreNumber()
//    initModules()
//    withMultiThread = true
  }


//  private def flattenInput(input: Activity): Boolean = {
//    val inputDepth = if (input.isTensor) 1 else input.toTable.length()
//    if (tensorBuffer == null) tensorBuffer = new Array[Tensor[Float]](inputDepth)
//        var batch : Int = 0
//    if (inputDepth == 1) {
//      tensorBuffer(0) = input.toTensor[Float]
//    } else {
//      val in = input.toTable
//      for (i <- 1 to in.length()) {
//        if (in.get(i).get.isInstanceOf[Table]) return false
//        tensorBuffer(i - 1) = in.get[Tensor[Float]](i).get
//        if (i == 1) batch = tensorBuffer(i - 1).size(1)
//        // reminder: inputs for DetectionOutputSSD are not all in batch,
//        // but the non-batched input can be shared in all batch. So this layer can be paralleled.
//        if (batch != tensorBuffer(i - 1).size(1)
//            && !module.isInstanceOf[DetectionOutputSSD[Float]]) {
//          return false
//        }
//      }
//    }
//    true
//  }
//  private def initModules(): Unit = {
//    subModels = if (module.parameters() != null) {
//      val wb = NNUtils.getAndClearWeightBias(module.parameters())
//      val models = (1 to subModelNumber).map(i => {
//          val m = module.cloneModule()
//          NNUtils.putWeightBias(wb, m)
//          m.asInstanceOf[Module[Float]]
//      }).toArray
//      NNUtils.putWeightBias(wb, module)
//      models
//    } else {
//      val models = (1 to subModelNumber).map(i => {
//          val m = module.cloneModule()
//          m.asInstanceOf[Module[Float]]
//      }).toArray
//          models
//    }
//  }
//
//  private def getInput(dim: Int, index: Int, size: Int): Activity = {
//    if (tensorBuffer.length == 1) {
//      tensorBuffer(0).narrow(dim, index, size)
//    } else {
//      // the third tensor of inputs for DetectionOutputSSD is not in batch,
//      // but it can be shared with all batch.
//      if (module.isInstanceOf[DetectionOutputSSD[Float]]) {
//        T(tensorBuffer(0).narrow(dim, index, size),
//            tensorBuffer(1).narrow(dim, index, size), tensorBuffer(2))
//      } else {
//        T.array(tensorBuffer.map(_.narrow(dim, index, size)))
//      }
//    }
//  }
//  private def forwardInParallel(input: Activity): Activity = {
//    if (inputBuffer == null) inputBuffer = new Array[Activity](subModelNumber)
//        val stackSize = batchSize / subModelNumber
//
//    val tasks = Engine.wrapperComputing.invoke((0 until subModelNumber).map(i =>
//        () => inputBuffer(i) = getInput(1, i * stackSize + 1, stackSize)))
//    Engine.wrapperComputing.sync(tasks)
//
//    val forwardThreads = Engine.wrapperComputing.invoke((0 until subModelNumber).map(i =>
//        () => subModels(i).forward(inputBuffer(i)).toTensor[Float]))
//    Engine.wrapperComputing.sync(forwardThreads)
//
//    if (subModels(0).output.isTable) {
//      withMultiThread = false
//      module.forward(input)
//    } else {
//      val subOutSize = subModels(0).output.toTensor[Float].size()
//      if (subOutSize(0) != stackSize) {
//        withMultiThread = false
//        module.forward(input)
//      } else {
//        subOutSize(0) = batchSize
//        if (output == null || output.toTensor[Float].isEmpty) {
//          output = Tensor[Float]().resize(subOutSize)
//        }
//        val copyThreads = Engine.wrapperComputing.invoke((0 until subModelNumber).map(i =>
//            () => {
//          output.toTensor[Float].narrow(1, i * stackSize + 1, stackSize)
//              .copy(subModels(i).output.toTensor[Float])
//        }))
//        Engine.wrapperComputing.sync(copyThreads)
//
//        output
//      }
//    }
//  }

  @Override
  public Activity updateOutput(Activity input) {
    if (!initEnv) {
      setMultiThreadEnv(input);
    }
    if (withMultiThread) {
      // output = forwardInParallel(input);
    } else {
      output = module.forward(input);
    }
    return output;
  }

  @Override
  public Activity updateGradInput(Activity input, Activity gradOutput) {
    gradInput = module.updateGradInput(input, gradOutput);
    return gradInput;
  }

  @Override
  public void accGradParameters(Activity input, Activity gradOutput) {
    module.accGradParameters(input, gradOutput);
  }

  @Override
  public AbstractModule clearState() {
    super.clearState();
    module.clearState();
    return this;
  }

  @Override
  public int hashCode() {
    int seed = 37;
    int hash = super.hashCode();
    hash = hash * seed + module.hashCode();
    return hash;
  }

  @Override
  public AbstractModule training() {
    train = true;
    module.training();
    return this;
  }

  @Override
  public AbstractModule evaluate() {
    train = false;
    module.evaluate();
    return this;
  }

  @Override
  public void release() {
    super.release();
  }
}
