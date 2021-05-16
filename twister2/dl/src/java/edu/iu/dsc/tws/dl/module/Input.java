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
package edu.iu.dsc.tws.dl.module;

import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

/**
 * Input layer do nothing to the input tensors, just pass them. It should be used as input node
 * when the first layer of your module accepts multiple tensors as inputs.
 * <p>
 * Each input node of the graph container should accept one tensor as input. If you want a module
 * accepting multiple tensors as input, you should add some Input module before it and connect
 * the outputs of the Input nodes to it.
 * <p>
 * Please note that the return is not a layer but a Node containing input layer.
 */
public class Input extends AbstractModule<DenseTensor> {

  public Input() {
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
  }

  public Input(String name) {
    this.setName(name);
  }

  @Override
  public DenseTensor updateOutput(DenseTensor input) {
    output = input;
    return (DenseTensor) output;
  }

  @Override
  public DenseTensor updateGradInput(DenseTensor input, DenseTensor gradOutput) {
    gradInput = gradOutput;
    return (DenseTensor) gradInput;
  }

  @Override
  public TensorArrayPair parameters() {
    return null;
  }

  @Override
  public void reset() {

  }
}
