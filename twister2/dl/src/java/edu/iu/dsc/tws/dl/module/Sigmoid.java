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
import edu.iu.dsc.tws.dl.graph.TensorModule;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

public class Sigmoid extends TensorModule<DenseTensor> {

  private DenseTensor buffer = new DenseTensor(false);

  public Sigmoid() {
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
  }

  @Override
  public void toFloat() {
    super.toFloat();
    buffer = new DenseTensor(true);
    this.output = new DenseTensor(this.isFloat);
    this.gradInput = new DenseTensor(this.isFloat);
  }

  @Override
  public DenseTensor updateOutput(DenseTensor input) {
    if (this.isFloat) {
      ((DenseTensor) output).resizeAs(input).fill(1.0f);
      buffer.resizeAs(input).copy(input).mul(-1.0f);
      buffer.exp().add(1.0f);
      ((DenseTensor) output).cdiv(buffer);
    } else {
      ((DenseTensor) output).resizeAs(input).fill(1.0);
      buffer.resizeAs(input).copy(input).mul(-1.0);
      buffer.exp().add(1.0);
      ((DenseTensor) output).cdiv(buffer);
    }

    return (DenseTensor) output;
  }

  @Override
  public DenseTensor updateGradInput(DenseTensor input, DenseTensor gradOutput) {
    return updateGradInputInternal((Tensor) output, gradOutput);
  }

  private DenseTensor updateGradInputInternal(Tensor output, DenseTensor gradOutput) {
    if (this.isFloat) {
      ((DenseTensor) gradInput).resizeAs(gradOutput).copy(gradOutput);
      buffer.resizeAs(gradOutput);
      buffer.fill(1.0f).sub(output);
      ((DenseTensor) gradInput).cmul(output).cmul(buffer);
    } else {
      ((DenseTensor) gradInput).resizeAs(gradOutput).copy(gradOutput);
      buffer.resizeAs(gradOutput);
      buffer.fill(1.0).sub(output);
      ((DenseTensor) gradInput).cmul(output).cmul(buffer);
    }
    return (DenseTensor) gradInput;
  }

  @Override
  public TensorArrayPair parameters() {
    return null;
  }

  @Override
  public void reset() {

  }

  @Override
  public AbstractModule clearState() {
    super.clearState();
    buffer.set();
    return this;
  }
}
