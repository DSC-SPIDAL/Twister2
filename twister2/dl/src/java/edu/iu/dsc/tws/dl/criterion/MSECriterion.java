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

import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.TensorNumeric;

public class MSECriterion extends TensorCriterion {

  private boolean sizeAverage = true;

  public MSECriterion() {
  }

  public MSECriterion(boolean sizeAverage) {
    this.sizeAverage = sizeAverage;
  }

  @Override
  public double updateOutput(Tensor input, Tensor target) {
    gradInput.resizeAs(input).copy(input);
    gradInput.sub(target);
    output = gradInput.dot(gradInput);
    if (sizeAverage) {
      output = TensorNumeric.divide(output, input.nElement());
    }
    return output;
  }

  @Override
  public float updateOutputf(Tensor input, Tensor target) {
    gradInput.resizeAs(input).copy(input);
    gradInput.sub(target);
    outputf = gradInput.dotf(gradInput);
    if (sizeAverage) {
      outputf = TensorNumeric.divide(outputf, input.nElement());
    }
    return outputf;
  }

  @Override
  public Tensor updateGradInput(Tensor input, Tensor target) {
    double norm = 2.0;
    if (sizeAverage) {
      norm = 2.0 / input.nElement();
    }
    if (this.isFloat) {
      gradInput.mul((float) norm);
    } else {
      gradInput.mul(norm);
    }
    return gradInput;
  }

  public boolean isSizeAverage() {
    return sizeAverage;
  }

  public void setSizeAverage(boolean sizeAverage) {
    this.sizeAverage = sizeAverage;
  }
}
