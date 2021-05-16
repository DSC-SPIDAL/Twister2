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

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.data.Table;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.TensorType;
import edu.iu.dsc.tws.dl.module.AbstractModule;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.ArrayPair;

@SuppressWarnings("MemberName")
public abstract class MklDnnLayer extends AbstractModule<Activity> implements MklDnnModule,
    IMklDnnLayer {
  /**
   * MKL-DNN primitives of the module. Note you should only initialize this field by calling
   * initPrimitives method. This field will be erased when sending model to remote worker. So you
   * need to reinitialize it after sending the model.
   */
  protected transient long[] updateOutputPrimitives;
  protected transient long[] updateGradInputPrimitives;
  protected transient long[] accGradientPrimitives;

  protected MemoryData[] _inputFormats;
  protected MemoryData[] _gradInputFormats;
  protected MemoryData[] _outputFormats;
  protected MemoryData[] _gradOutputFormats;
  protected MemoryData[] _gradOutputFormatsForWeight;

  private transient long[] updateOutputMemoryPrimitives;
  private transient Tensor[] updateOutputTensors;
  private transient long[] updateGradInputMemoryPrimitives;
  private transient Tensor[] updateGradInputTensors;
  private transient Activity cachedInput;
  private transient Activity cachedGradOutput;

  @Override
  public MemoryData[] initGradWPrimitives(MemoryData[] grad, Phase phase) {
    _gradOutputFormatsForWeight = grad;
    return grad;
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
  public Activity updateOutput(Activity input) {
    if (updateOutputMemoryPrimitives == null) {
      updateOutputMemoryPrimitives = getUpdateOutputMemoryPrimitives();
    }
    if (updateOutputTensors == null || cachedInput == null || cachedInput != input) {
      ArrayList<Tensor> buffer = new ArrayList<Tensor>();
      if (input.isTensor()) {
        buffer.add((Tensor) input);
      } else {
        Table table = input.toTable();
        int i = 1;
        while (i <= table.length()) {
          buffer.add(table.get(i));
          i += 1;
        }
      }
      if (output.isTensor()) {
        buffer.add((Tensor) output);
      } else {
        Table table = output.toTable();
        int i = 1;
        while (i <= table.length()) {
          buffer.add(table.get(i));
          i += 1;
        }
      }
      updateOutputTensors = new Tensor[buffer.size()];
      buffer.toArray(updateOutputTensors);
      cachedInput = input;
    }
    MklDnnOps.streamSubmit(
        runtime.stream, 1, updateOutputPrimitives, updateOutputPrimitives.length,
        updateOutputMemoryPrimitives,
        updateOutputTensors
    );
    return output;
  }

  @Override
  public Activity updateGradInput(Activity input, Activity gradOutput) {
    if (updateGradInputMemoryPrimitives == null) {
      updateGradInputMemoryPrimitives = getUpdateGradInputMemoryPrimitives();
    }
    if (updateGradInputTensors == null || cachedInput == null || cachedInput != input
        || cachedGradOutput == null || cachedGradOutput != gradOutput) {
      ArrayList<Tensor> buffer = new ArrayList();
      if (gradOutput.isTensor()) {
        buffer.add((Tensor) gradOutput);
      } else {
        Table table = gradOutput.toTable();
        int i = 1;
        while (i <= table.length()) {
          buffer.add(table.get(i));
          i += 1;
        }
      }
      if (gradInput.isTensor()) {
        buffer.add((Tensor) gradInput);
      } else {
        Table table = gradInput.toTable();
        int i = 1;
        while (i <= table.length()) {
          buffer.add(table.get(i));
          i += 1;
        }
      }

      updateGradInputTensors = new Tensor[buffer.size()];
      buffer.toArray(updateGradInputTensors);
      cachedInput = input;
      cachedGradOutput = gradOutput;
    }
    MklDnnOps.streamSubmit(runtime.stream, 1, updateGradInputPrimitives,
        updateGradInputPrimitives.length,
        updateGradInputMemoryPrimitives, updateGradInputTensors);
    return gradInput;
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
    return _gradOutputFormatsForWeight;
  }

  public void updateWithNewTensor(Tensor[] from, int index, Activity value) {


    for (int i = 0; i < from.length; i++) {
      if (from[i].getTensorType() == TensorType.DenseType) {
        from[i] = value.toTensor();
      }
    }
  }

  @Override
  public void release() {
    this.releaseResources();
  }

  @Override
  public MklDnnLayer setQuantize(boolean value) {
    return this;
  }

  @Override
  public ArrayPair<TensorMMap> paramsMMap() {
    // return null for weight and gradWeight by default
    return new ArrayPair<TensorMMap>(new TensorMMap[0], new TensorMMap[0]);
  }
}

