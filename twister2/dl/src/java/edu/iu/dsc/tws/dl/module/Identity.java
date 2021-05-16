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

import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

public class Identity extends AbstractModule<DenseTensor> {

  public Identity() {
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
  }

  @Override
  public DenseTensor updateOutput(DenseTensor input) {
    this.output = input;
    return (DenseTensor) this.output;
  }

  @Override
  public DenseTensor updateGradInput(DenseTensor input, DenseTensor gradOutput) {
    this.gradInput = gradOutput;
    return (DenseTensor) this.gradInput;
  }

  @Override
  public TensorArrayPair parameters() {
    return null;
  }

  @Override
  public Tensor[] getExtraParameter() {
    return null;
  }

  @Override
  public void reset() {

  }
}
