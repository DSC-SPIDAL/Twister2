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

import com.intel.analytics.bigdl.mkl.DataType;
import com.intel.analytics.bigdl.mkl.Memory;

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.TensorNumeric;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.MklDnnMemory;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.HeapData;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.NativeData;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.MemoryDataArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

public class ReorderMemory extends MklDnnLayer implements Releasable {
  private MemoryData inputFormat;
  private MemoryData outputFormat;
  private MemoryData gradInputFormat;
  private MemoryData gradOutputFormat;
  private MemoryOwner memoryOwner;

  private MemoryData[] realInput = null;
  private MemoryData[] realOutput = null;
  private MemoryData[] realgradInput = null;
  private MemoryData[] realgradOutput = null;

  // ReorderMemory is a special layer. It can be owned by other layers.
  // So there is an optional MemoryOwner that can be null.
  // If it is null, this means the ReorderMemory is a normal layer.
  // If it is not null, it means ReorderMemory is owned by another layer


  public ReorderMemory(MemoryData outputFormat, MemoryOwner memoryOwner) {
    this(null, outputFormat, null, null,
        memoryOwner);
  }

  public ReorderMemory(MemoryData outputFormat, MemoryData gradInputFormat,
                       MemoryOwner memoryOwner) {
    this(null, outputFormat, gradInputFormat, null,
        memoryOwner);
  }

  public ReorderMemory(MemoryData inputFormat, MemoryData outputFormat, MemoryData gradInputFormat,
                       MemoryData gradOutputFormat, MemoryOwner memoryOwner) {
    this.inputFormat = inputFormat;
    this.outputFormat = outputFormat;
    this.gradInputFormat = gradInputFormat;
    this.gradOutputFormat = gradOutputFormat;
    this.memoryOwner = memoryOwner;

    if (memoryOwner != null) {
      memoryOwner.registerResource(this);
    }
    this._outputFormats = new MemoryData[]{outputFormat};
    this._gradInputFormats = new MemoryData[]{gradInputFormat};
  }

  private MemoryData[] initMemory(MemoryData src, int[] shape, int layout) {
    MemoryData[] ret;

    if (src instanceof HeapData) {
      ret = new HeapData[]{new HeapData(shape, layout, src.dataType())};
    } else if (src instanceof NativeData) {
      ret = new NativeData[]{new NativeData(shape, layout, src.dataType())};
    } else {
      throw new UnsupportedOperationException("Not support such memory format");
    }

    ret[0].setMask(src.mask());
    ret[0].setScales(src.scales);
    return ret;
  }

  private String shapeToString(int[] shape) {
    String name = "";
    for (int i : shape) {
      name += i + ",";
    }
    return name;
  }

  private void reshapeOutputIfNeeded(MemoryData format, Tensor tensor) {
    // must pay attention to the shape of tensor when format is nhwc,
    // the Tensor's shape in BigDL always be relevant with the format, such as
    // [4, 3, 224, 224] will be nchw and [4, 224, 224, 3] will be nhwc.
    // but for mkldnn, it always uses the nchw format shape, library will use
    // correct shape by the format.
    if (format.layout() == Memory.Format.nhwc && format instanceof HeapData) {
      tensor.toTensor().resize(format.shape());
    }
    // for mkldnn, it always use tnc format shape even though format is ntc
    if (format.layout() == Memory.Format.ntc && format instanceof HeapData) {
      tensor.toTensor().resize(format.shape());
    }
  }

//  private long createInt8PrimDesc() {
//    long attr = MklDnnMemory.CreateAttr(memoryOwner);
//    MklDnn.AttrSetIntOutputRoundMode(attr, 1);
//
//    if (realOutput[0].scales == null || realOutput[0].scales.isEmpty) {
//      realOutput[0].setMask(realInput[0].mask());
//      realOutput[0].setScales(realInput[0].scales);
//    }
//
//    // if convert s8/u8 to f32, we should set the scale factor to 1.0f/x
//    if (realOutput[0].dataType == DataType.F32) {
//      realOutput[0].setScales(realOutput[0].scales.map(1.0f / _));
//    }
//
//    // copy the scales back to outputFormats if not equal
//    if (realOutput[0] ne _outputFormats(0)) {
//      _outputFormats[0].setMask(realOutput[0].mask());
//      _outputFormats[0].setScales(realOutput[0].scales);
//    }
//
//    Util.require(realOutput[0].scales.nonEmpty);
//    MklDnn.AttrSetOutputScales(attr, realOutput[0].scales.length, realOutput[0].mask(),
//        realOutput[0].scales);
//    MklDnnMemory.ReorderPrimitiveDescCreateV2(
//        realInput[0].getPrimitiveDescription(runtime),
//        realOutput[0].getPrimitiveDescription(runtime),
//        attr);
//  }

  @Override
  public Activity updateGradInput(Activity input, Activity gradOutput) {
    gradInput = super.updateGradInput(input, gradOutput);
    return gradInput;
  }

  @Override
  public TensorArrayPair parameters() {
    return null;
  }

  @Override
  public void reset() {

  }

  @Override
  public long[] getUpdateGradInputMemoryPrimitives() {
    long[] result = new long[inputFormats().length + outputFormats().length];
    int index = 0;
    for (MemoryData memoryData : realgradOutput) {
      result[index] = memoryData.getPrimitive(runtime, this);
      index++;
    }
    for (MemoryData memoryData : realgradInput) {
      result[index] = memoryData.getPrimitive(runtime, this);
      index++;
    }
    return result;
  }

  @Override
  public long[] getUpdateOutputMemoryPrimitives() {
    long[] result = new long[inputFormats().length + outputFormats().length];
    int index = 0;
    for (MemoryData memoryData : realInput) {
      result[index] = memoryData.getPrimitive(runtime, this);
      index++;
    }
    for (MemoryData memoryData : realOutput) {
      result[index] = memoryData.getPrimitive(runtime, this);
      index++;
    }
    return result;
  }

  @Override
  public MemoryDataArrayPair initFwdPrimitives(MemoryData[] inputs, Phase phase) {
    if (memoryOwner == null) {
      memoryOwner = this;
    }
    if (inputFormat == null) {
      _inputFormats = inputs;
    } else {
      _inputFormats = new MemoryData[]{inputFormat};
    }
    Util.require(_inputFormats.length == 1, "Only accept one tensor as input");

    if (outputFormat == null) {
      _outputFormats = _inputFormats;
    }

    shapeToString(_inputFormats[0].shape());

    Util.require(TensorNumeric.product(_inputFormats[0].shape())
            == TensorNumeric.product(_outputFormats[0].shape()),
        "input output memory not match, input shape " + shapeToString(_inputFormats[0].shape())
            + "output shape " + shapeToString(_outputFormats[0].shape()));

    int[] inputShape = _inputFormats[0].shape();
    int[] outputShape = _outputFormats[0].shape();
    int inputLayout = _inputFormats[0].layout();
    int outputLayout = _outputFormats[0].layout();
    realInput = _inputFormats;
    realOutput = _outputFormats;

    if (inputLayout != outputLayout) {
      if (inputLayout == Memory.Format.nhwc || inputLayout == Memory.Format.ntc) {
        // remind: if format of input MemoryData is nhwc or ntc,
        // its shape should be output shape
        realInput = initMemory(_inputFormats[0], outputShape, inputLayout);
      } else if (outputLayout == Memory.Format.nhwc || outputLayout == Memory.Format.ntc) {
        // remind: if format of output MemoryData is nhwc or ntc,
        // its shape should be input shape
        realOutput = initMemory(_outputFormats[0], inputShape, outputLayout);
      }
    }

    boolean noInt8Formats = inputFormats()[0].dataType() == DataType.F32
        && outputFormats()[0].dataType() == DataType.F32;

    long fwdReorderPrimDesc;
    if (noInt8Formats) {
      fwdReorderPrimDesc = MklDnnMemory.ReorderPrimitiveDescCreate(
          realInput[0].getPrimitiveDescription(runtime, memoryOwner),
          realOutput[0].getPrimitiveDescription(runtime, memoryOwner), memoryOwner);
    } else {
      throw new UnsupportedOperationException("Int8 not supported");
    }

    long fwdReorderPrim = MklDnnMemory.PrimitiveCreate2(fwdReorderPrimDesc,
        new long[]{realInput[0].getPrimitive(runtime, memoryOwner)}, new int[]{0}, 1,
        new long[]{realOutput[0].getPrimitive(runtime, memoryOwner)}, 1, memoryOwner);

    updateOutputPrimitives = new long[]{fwdReorderPrim};

    // recover to original data
    output = initTensor(realOutput[0]);

    reshapeOutputIfNeeded(_outputFormats[0], output.toTensor());

    return new MemoryDataArrayPair(_inputFormats, _outputFormats);
  }


  @Override
  public MemoryDataArrayPair initBwdPrimitives(MemoryData[] grad, Phase phase) {
    if (memoryOwner == null) {
      memoryOwner = this;
    }
    if (gradInputFormat == null && inputFormat == null) {
      _gradInputFormats = inputFormats();
    } else if (gradInputFormat == null && inputFormat != null) {
      _gradInputFormats = new MemoryData[]{inputFormat};
    } else if (gradInputFormat != null) {
      _gradInputFormats = new MemoryData[]{gradInputFormat};
    }

    if (gradOutputFormat == null) {
      _gradOutputFormats = grad;
    } else {
      _gradOutputFormats = new MemoryData[]{gradOutputFormat};
    }
    Util.require(_gradOutputFormats.length == 1, "Only accept one tensor as input");
    Util.require(TensorNumeric.product(_gradOutputFormats[0].shape())
            == TensorNumeric.product(_gradInputFormats[0].shape()),
        "gradInput and gradOutput memory not match,"
            + "gradInput shape " + shapeToString(_gradInputFormats[0].shape())
            + "gradOutput shape " + shapeToString(_gradOutputFormats[0].shape()));

    int[] gradInputShape = _gradInputFormats[0].shape();
    int[] gradOutputShape = _gradOutputFormats[0].shape();
    int gradInputLayout = _gradInputFormats[0].layout();
    int gradOutputLayout = _gradOutputFormats[0].layout();
    realgradInput = _gradInputFormats;
    realgradOutput = _gradOutputFormats;

    if (gradInputLayout != gradOutputLayout) {
      if (gradOutputLayout == Memory.Format.nhwc || gradOutputLayout == Memory.Format.ntc) {
        // remind: if format of gradOutput MemoryData is nhwc or ntc,
        // its shape should be gradInput shape
        realgradOutput = initMemory(_gradOutputFormats[0], gradInputShape, gradOutputLayout);
      } else if (gradInputLayout == Memory.Format.nhwc || gradInputLayout == Memory.Format.ntc) {
        // remind: if format of gradInput MemoryData is nhwc or ntc,
        // its shape should be gradOutput shape
        realgradInput = initMemory(_gradInputFormats[0], gradOutputShape, gradInputLayout);
      }
    }

    long bwdReorderPrimDesc = MklDnnMemory.ReorderPrimitiveDescCreate(
        realgradOutput[0].getPrimitiveDescription(runtime, memoryOwner),
        realgradInput[0].getPrimitiveDescription(runtime, memoryOwner), memoryOwner);

    long[] rgOutputPrim = new long[realgradOutput.length];
    long[] rgInputPrim = new long[realgradInput.length];

    for (int i = 0; i < realgradOutput.length; i++) {
      rgOutputPrim[i] = realgradOutput[i].getPrimitive(runtime, memoryOwner);
    }

    for (int i = 0; i < realgradInput.length; i++) {
      rgInputPrim[i] = realgradInput[i].getPrimitive(runtime, memoryOwner);
    }

    long bwdReorderPrim = MklDnnMemory.PrimitiveCreate2(bwdReorderPrimDesc,
        rgOutputPrim, new int[]{0}, 1,
        rgInputPrim, 1, memoryOwner);

    updateGradInputPrimitives = new long[]{bwdReorderPrim};
    gradInput = initTensor(realgradInput[0]);

    reshapeOutputIfNeeded(_gradInputFormats[0], gradInput.toTensor());

    return new MemoryDataArrayPair(_gradOutputFormats, _gradInputFormats);
  }

  @Override
  public String toString() {
    if (_inputFormats != null) {
      return "nn.mkl.ReorderMemory(${_inputFormats[0]} -> ${outputFormat})";
    } else {
      return "nn.mkl.ReorderMemory(_ -> ${outputFormat})";
    }
  }

}
