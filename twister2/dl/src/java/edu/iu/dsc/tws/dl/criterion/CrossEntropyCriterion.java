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
package edu.iu.dsc.tws.dl.criterion;

import java.util.ArrayList;
import java.util.List;

import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.module.LogSoftMax;

/**
 * This criterion combines LogSoftMax and ClassNLLCriterion in one single class.
 * weights A tensor assigning weight to each of the classes
 */

@SuppressWarnings("LocalVariableName")
public class CrossEntropyCriterion extends TensorCriterion {

  private DenseTensor weights;
  private boolean sizeAverage;
  private LogSoftMax lsm = new LogSoftMax();
  private ClassNLLCriterion nll = new ClassNLLCriterion(weights, sizeAverage);

  public CrossEntropyCriterion(DenseTensor weights, boolean sizeAverage) {
    this.weights = weights;
    this.sizeAverage = sizeAverage;
  }

  public CrossEntropyCriterion() {
    this(null, true);
  }

  @Override
  public void toFloat() {
    super.toFloat();
    nll.toFloat();
    lsm.toFloat();
  }

  @Override
  public double updateOutput(Tensor input, Tensor target) {
    lsm.updateOutput((DenseTensor) input);
    nll.updateOutput((Tensor) lsm.output, target);
    output = nll.output;
    return output;
  }

  @Override
  public float updateOutputf(Tensor input, Tensor target) {
    lsm.updateOutput((DenseTensor) input);
    nll.updateOutputf((Tensor) lsm.output, target);
    outputf = nll.outputf;
    return outputf;
  }

  @Override
  public Tensor updateGradInput(Tensor input, Tensor target) {
    int[] size = input.size();
    DenseTensor _gradInput = new DenseTensor(this.isFloat);
    _gradInput = (DenseTensor) nll.updateGradInput((Tensor) lsm.output, target);
    lsm.updateGradInput((DenseTensor) input, _gradInput);
    gradInput.resizeAs((Tensor) lsm.gradInput).copy((Tensor) lsm.gradInput).view(size);
    return gradInput;
  }

  @Override
  public boolean canEqual(Object other) {
    return other instanceof CrossEntropyCriterion;
  }

  @Override
  public int hashCode() {
    List state = new ArrayList();
    state.add(super.hashCode());
    state.add(this.weights);
    return state.stream().mapToInt(o ->
        (o == null) ? 0 : o.hashCode()).reduce(0, (a, b) -> 37 * a + b);
  }

  @Override
  public boolean equals(Object other) {
    if (other instanceof CrossEntropyCriterion) {
      return weights == ((CrossEntropyCriterion) other).weights
          && sizeAverage == ((CrossEntropyCriterion) other).sizeAverage;
    }
    return false;
  }

  @Override
  public String toString() {
    return "nn.CrossEntropyCriterion";
  }
}
