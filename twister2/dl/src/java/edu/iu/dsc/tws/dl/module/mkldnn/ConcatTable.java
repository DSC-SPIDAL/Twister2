//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS I" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
package edu.iu.dsc.tws.dl.module.mkldnn;

import java.util.ArrayList;
import java.util.Arrays;

import com.intel.analytics.bigdl.mkl.DataType;
import com.intel.analytics.bigdl.mkl.Memory;

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.data.Table;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.MklDnnMemory;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.NativeData;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.MemoryDataArrayPair;

@SuppressWarnings({"StaticVariableName", "ParameterName", "MemberName"})
public class ConcatTable extends MklDnnContainer {

  private transient long[] sumPrimitive = null;
  private transient Tensor[] tensors = null;
  private transient long[] tensorPrimitives = null;

  private MemoryData[] _inputFormats;
  private MemoryData[] _gradInputFormats;
  private MemoryData[] _outputFormats;
  private MemoryData[] _gradOutputFormats;
  private MemoryData[] _gradOutputWeightFormats;

  public ConcatTable() {
    output = new Table();
  }

  @Override
  public Activity updateOutput(Activity input) {
    Util.require(modules.size() > 0, "empty modules of concat table");
    int i = 0;
    while (i < modules.size()) {
      Activity currentOutput = modules.get(i).forward(
          reorderManager.infer(_inputFormats, mklDnnModules[i].inputFormats(), input));
      ((Table) output).update(i + 1, currentOutput);
      i += 1;
    }
    return output;
  }

  @Override
  public Activity updateGradInput(Activity input, Activity gradOutput) {
    Util.require(modules.size() > 0, "empty modules of concat table");

    int i = 0;
    while (i < modules.size()) {
      tensors[i] = (Tensor) modules.get(i).updateGradInput(input, ((Table) gradOutput).get(i + 1));
      i += 1;
    }
    MklDnnOps.streamSubmit(runtime.stream, 1, sumPrimitive, 1, tensorPrimitives, tensors);
    return gradInput;
  }

  @Override
  public void accGradParameters(Activity input, Activity gradOutput) {
    int i = 0;
    while (i < modules.size()) {
      modules.get(i).accGradParameters(input, ((Table) gradOutput).get(i + 1));
      modules.get(i).asyncGradient();
      i += 1;
    }
  }

  @Override
  public MemoryDataArrayPair initFwdPrimitives(MemoryData[] inputs, Phase phase) {
    Util.require(mklDnnModules != null, "You should call compile first");
    Util.require(inputs.length == 1, "Concat only accept one tensor");
    ArrayList<MemoryData> buffer = new ArrayList();

    for (int i = 0; i < mklDnnModules.length; i++) {
      MklDnnModule m = mklDnnModules[i];
      MemoryDataArrayPair realInputOut = m.initFwdPrimitives(inputs, phase);
      Util.require(realInputOut.getValue1().length == 1, "output should be one tensor");
      for (int j = 0; j < inputs.length; j++) {
        MemoryData input = inputs[j];
        reorderManager.register(inputs[i], realInputOut.getValue0()[i]);
      }
      buffer.add(realInputOut.getValue1()[0]);
    }
    _outputFormats = (MemoryData[]) buffer.toArray();
    _inputFormats = inputs;
    return new MemoryDataArrayPair(inputs, _outputFormats);
  }

  @Override
  public MemoryDataArrayPair initBwdPrimitives(MemoryData[] grad, Phase phase) {
    Util.require(grad.length == mklDnnModules.length, "grad tensor number is not correct");
    _gradOutputFormats = new MemoryData[grad.length];
    MemoryData[] subGradInputs = new MemoryData[grad.length];
    tensorPrimitives = new long[grad.length + 1];
    int[] shape = null;

    for (int i = 0; i < grad.length; i++) {
      MklDnnModule m = mklDnnModules[i];
      MemoryDataArrayPair realGradsInput = m.initBwdPrimitives(new MemoryData[]{grad[i]}, phase);
      Util.require(realGradsInput.getValue0().length == 1, "real grad length should be 1");
      _gradOutputFormats[i] = realGradsInput.getValue0()[0];
      Util.require(realGradsInput.getValue1().length == 1, "real grad length should be 1");
      subGradInputs[i] = realGradsInput.getValue1()[0];
      tensorPrimitives[i] = realGradsInput.getValue1()[0].getPrimitive(runtime, this);
      if (shape == null) {
        shape = realGradsInput.getValue1()[0].shape().clone();
      } else {
        Util.require(shape.length == realGradsInput.getValue1()[0].shape().length,
            "backward grad shape should be same");
        for (int j = 0; j < shape.length; j++) {
          Util.require(shape[j] == realGradsInput.getValue1()[0].shape()[j],
              "backward grad shape size should be same");
        }
      }
    }

    long outputMD = MklDnnMemory.MemoryDescInit(shape.length, shape,
        DataType.F32, Memory.Format.any, this);
    float[] scales = new float[grad.length];
    Arrays.fill(scales, 1.0f);
    long[] temp = new long[subGradInputs.length];
    for (int i = 0; i < temp.length; i++) {
      temp[i] = subGradInputs[i].getPrimitiveDescription(runtime, this);

    }

    long[] temp2 = new long[subGradInputs.length];
    for (int i = 0; i < temp.length; i++) {
      temp2[i] = subGradInputs[i].getPrimitive(runtime, this);

    }

    long[] temp3 = new long[subGradInputs.length];
    for (int i = 0; i < temp.length; i++) {
      temp3[i] = _gradInputFormats[i].getPrimitive(runtime, this);

    }
    long pd = MklDnnMemory.SumPrimitiveDescCreate(outputMD, grad.length, scales,
        temp, this);
    _gradInputFormats = new NativeData[]{MemoryData.primitiveOutput(pd)};
    tensorPrimitives[grad.length] = _gradInputFormats[0].getPrimitive(runtime, this);
    sumPrimitive = new long[]{MklDnnMemory.PrimitiveCreate2(pd,
        temp2, new int[]{grad.length}, grad.length, temp3, 1, this)};
    gradInput = initTensor(_gradInputFormats[0]);
    tensors = new Tensor[grad.length + 1];
    tensors[grad.length] = (Tensor) gradInput;
    return new MemoryDataArrayPair(_gradOutputFormats, _gradInputFormats);
  }

  @Override
  public MemoryData[] initGradWPrimitives(MemoryData[] grad, Phase phase) {
    ArrayList<MemoryData> realGradsBuffer = new ArrayList();
    for (int i = 0; i < grad.length; i++) {
      MklDnnModule m = mklDnnModules[i];
      MemoryData[] realGradOutput = m.initGradWPrimitives(new MemoryData[]{grad[i]}, phase);
      Util.require(realGradOutput.length == 1, "real grad length should be 1, "
          + "but it's ${realGradOutput.length}");
      realGradsBuffer.add(realGradOutput[0]);
    }
    _gradOutputWeightFormats = (MemoryData[]) realGradsBuffer.toArray();
    return _gradOutputWeightFormats;
  }

  @Override
  public MemoryData[] inputFormats() {
    Util.require(_inputFormats != null, "You should call initFwdPrimitives first");
    return _inputFormats;
  }

  @Override
  public MemoryData[] gradInputFormats() {
    Util.require(_gradInputFormats != null, "You should call initBwdPrimitives first");
    return _gradInputFormats;
  }

  @Override
  public MemoryData[] outputFormats() {
    Util.require(_outputFormats != null, "You should call initFwdPrimitives first");
    return _outputFormats;
  }

  @Override
  public MemoryData[] gradOutputFormats() {
    Util.require(_gradOutputFormats != null, "You should call initBwdPrimitives first");
    return _gradOutputFormats;
  }

  @Override
  public MemoryData[] gradOutputWeightFormats() {
    return _gradOutputWeightFormats;
  }

  @Override
  public String toString() {
    return "ConcatTable{"
        + "sumPrimitive=" + Arrays.toString(sumPrimitive)
        + ", tensors=" + Arrays.toString(tensors)
        + ", tensorPrimitives=" + Arrays.toString(tensorPrimitives)
        + ", _inputFormats=" + Arrays.toString(_inputFormats)
        + ", _gradInputFormats=" + Arrays.toString(_gradInputFormats)
        + ", _outputFormats=" + Arrays.toString(_outputFormats)
        + ", _gradOutputFormats=" + Arrays.toString(_gradOutputFormats)
        + ", _gradOutputWeightFormats=" + Arrays.toString(_gradOutputWeightFormats)
        + '}';
  }

  private void reconstruct() {
    mklDnnModules = new MklDnnModule[modules.size()];
    for (int i = 0; i < modules.size(); i++) {
      mklDnnModules[i] = (MklDnnModule) modules.get(i);
    }
  }

}
