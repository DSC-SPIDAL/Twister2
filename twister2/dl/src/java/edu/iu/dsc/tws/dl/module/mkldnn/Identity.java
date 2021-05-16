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
import edu.iu.dsc.tws.dl.utils.pair.MemoryDataArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

public class Identity extends MklDnnLayer {

  @Override
  public TensorArrayPair parameters() {
    return null;
  }

  @Override
  public void reset() {

  }

  @Override
  public Activity updateOutput(Activity input) {
    output = input;
    return output;
  }

  @Override
  public Activity updateGradInput(Activity input, Activity gradOutput) {
    gradInput = gradOutput;
    return gradInput;
  }

  @Override
  public MemoryDataArrayPair initFwdPrimitives(MemoryData[] inputs, Phase phase) {
    _inputFormats = inputs;
    _outputFormats = inputs;
    return new MemoryDataArrayPair(inputs, inputs);
  }

  @Override
  public MemoryDataArrayPair initBwdPrimitives(MemoryData[] grad, Phase phase) {
    _gradOutputFormats = grad;
    _gradOutputFormatsForWeight = grad;
    _gradInputFormats = grad;
    return new MemoryDataArrayPair(grad, grad);
  }
}
