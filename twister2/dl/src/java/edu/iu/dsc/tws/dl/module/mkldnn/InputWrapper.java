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

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.MemoryDataArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

public class InputWrapper extends MklDnnLayer {

  private Input inputLayer = null;

  @Override
  public TensorArrayPair parameters() {
    return null;
  }

  @Override
  public void reset() {

  }

  @Override
  public Activity updateOutput(Activity input) {
    output = inputLayer.forward(input);
    return output;
  }

  @Override
  public Activity updateGradInput(Activity input, Activity gradOutput) {
    gradInput = inputLayer.backward(input, gradOutput);
    return gradInput;
  }

  @Override
  public MemoryDataArrayPair initFwdPrimitives(MemoryData[] inputs, Phase phase) {
    Util.require(inputs.length == 1, "Only accept one tensor as input");
    inputLayer = new Input(inputs[0].shape(), inputs[0].layout());
    inputLayer.setRuntime(this.runtime);
    inputLayer.initFwdPrimitives(inputs, phase);
    _inputFormats = inputLayer.inputFormats();
    _outputFormats = inputLayer.outputFormats();
    return new MemoryDataArrayPair(_inputFormats, _outputFormats);
  }

  @Override
  public MemoryDataArrayPair initBwdPrimitives(MemoryData[] grad, Phase phase) {
    Util.require(grad.length == 1, "Only accept one tensor as input");
    inputLayer.initBwdPrimitives(grad, phase);
    _gradInputFormats = inputLayer.gradInputFormats();
    _gradOutputFormats = inputLayer.gradOutputFormats();
    return new MemoryDataArrayPair(_gradOutputFormats, _gradInputFormats);
  }

  @Override
  public void release() {
    super.release();
    if (inputLayer != null) {
      inputLayer.release();
    }
  }

  @Override
  public String toString() {
    return "nn.mkl.InputWrapper";
  }
}
