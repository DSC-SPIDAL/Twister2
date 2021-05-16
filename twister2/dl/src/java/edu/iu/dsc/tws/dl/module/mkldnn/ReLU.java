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

import com.intel.analytics.bigdl.mkl.AlgKind;
import com.intel.analytics.bigdl.mkl.PropKind;
import com.intel.analytics.bigdl.mkl.Query;

import edu.iu.dsc.tws.dl.module.mkldnn.memory.MklDnnMemory;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.MemoryDataArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

@SuppressWarnings({"StaticVariableName", "ParameterName", "MemberName"})
public class ReLU extends MklDnnLayer {

  private float valueInternal = 0.0f;
  private long UNDEFINED = 0L;
  private transient long fwdPrimDesc = UNDEFINED;

  public ReLU(float valueInternal) {
    this.valueInternal = valueInternal;
  }

  @Override
  public TensorArrayPair parameters() {
    return null;
  }

  @Override
  public void reset() {

  }

  @Override
  public MemoryDataArrayPair initFwdPrimitives(MemoryData[] inputs, Phase phase) {
    _inputFormats = singleNativeData(inputs);
    long description = MklDnnMemory.EltwiseForwardDescInit(
        PropKind.Forward, AlgKind.EltwiseRelu, _inputFormats[0].getMemoryDescription(this),
        valueInternal, 0, this);
    fwdPrimDesc = MklDnnMemory.PrimitiveDescCreate(description, runtime.engine,
        0L, this);
    _outputFormats = new MemoryData[]{MemoryData.primitiveOutput(fwdPrimDesc)};
    long[] temp = new long[_outputFormats.length];
    for (int i = 0; i < temp.length; i++) {
      temp[i] = _outputFormats[i].getPrimitive(runtime, this);
    }
    updateOutputPrimitives = new long[]{MklDnnMemory.PrimitiveCreate2(fwdPrimDesc,
        new long[]{_inputFormats[0].getPrimitive(runtime, this)}, new int[]{0},
        _inputFormats.length, temp, _outputFormats.length, this)};
    output = initTensor(_outputFormats[0]);
    return new MemoryDataArrayPair(_inputFormats, _outputFormats);
  }

  @Override
  public MemoryDataArrayPair initBwdPrimitives(MemoryData[] grad, Phase phase) {
    _gradOutputFormats = singleNativeData(grad);
    _gradOutputFormatsForWeight = _gradOutputFormats;
    long description = MklDnnMemory.EltwiseBackwardDescInit(AlgKind.EltwiseRelu,
        _gradOutputFormats[0].getMemoryDescription(this), _inputFormats[0].
            getMemoryDescription(this),
        valueInternal, 0, this);
    Util.require(fwdPrimDesc != UNDEFINED, "You should call initFwdPrimitives first");
    long primDesc = MklDnnMemory.PrimitiveDescCreate(description, runtime.engine,
        fwdPrimDesc, this);
    _gradInputFormats = new MemoryData[]{MemoryData.operationWant(primDesc, Query.DiffSrcPd)};

    long[] temp1 = new long[2];
    temp1[0] = _inputFormats[0].getPrimitive(runtime, this);
    temp1[1] = _gradOutputFormats[0].getPrimitive(runtime, this);

    long[] temp2 = new long[_gradInputFormats.length];
    for (int i = 0; i < temp2.length; i++) {
      temp2[i] = _gradInputFormats[i].getPrimitive(runtime, this);
    }

    updateGradInputPrimitives = new long[]{MklDnnMemory.PrimitiveCreate2(primDesc, temp1,
        new int[]{0}, 2, temp2, _gradInputFormats.length, this)};
    gradInput = initTensor(_gradInputFormats[0]);
    return new MemoryDataArrayPair(_gradOutputFormats, _gradInputFormats);
  }
}
