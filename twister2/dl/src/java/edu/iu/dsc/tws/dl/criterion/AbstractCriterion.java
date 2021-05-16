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

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;

/**
 * [[AbstractCriterion]] is an abstract class the concrete criterion should extend.
 * `Criterion`s are helpful to train a neural network. Given an input and a target,
 * they compute the gradient according to a loss function.
 * <p>
 * It provides some important method such as `forward`, `backward`, `updateOutput`,
 * `updateGradInput` frequently used as a criteria. Some of them need to be override
 * in a concrete criterion class.
 *
 * @tparam A represents the input type of the criterion, which an be abstract type [[Activity]],
 * or concrete type [[Tensor]] or [[Table]]
 * @tparam B represents the output type of the criterion
 */
public abstract class AbstractCriterion<I extends Activity, O extends Activity>
    implements Criterion {
  protected I gradInput;
  protected double output;
  protected float outputf;
  protected boolean isFloat;
  protected SizeAverageStatus sizeAverageStatus = SizeAverageStatus.NONE;

  public AbstractCriterion() {
    gradInput = (I) new DenseTensor(false);
    output = 0.0;
    outputf = 0.0f;
    isFloat = false;
  }

  public void toFloat() {
    isFloat = true;
    gradInput = (I) new DenseTensor(true);
  }

  public Activity getGradInput() {
    return gradInput;
  }

  public void setGradInput(I gradInput) {
    this.gradInput = gradInput;
  }

  public double getOutput() {
    return output;
  }

  public float getOutputf() {
    return outputf;
  }

  public void setOutput(double output) {
    this.output = output;
  }

  public void setOutput(float output) {
    this.outputf = output;
  }

  /**
   * Takes an input object, and computes the corresponding loss of the criterion,
   * compared with `target`.
   *
   * @param input  input data
   * @param target target
   * @return the loss of criterion
   */
  public double forward(I input, O target) {
    return updateOutput(input, target);
  }

  public float forwardf(I input, O target) {
    return updateOutputf(input, target);
  }

  /**
   * Performs a back-propagation step through the criterion, with respect to the given input.
   *
   * @param input  input data
   * @param target target
   * @return gradient corresponding to input data
   */
  public I backward(I input, O target) {
    return updateGradInput(input, target);
  }

  /**
   * Computes the loss using input and objective function. This function
   * returns the result which is stored in the output field.
   *
   * @param input  input of the criterion
   * @param target target or labels
   * @return the loss of the criterion
   */
  public double updateOutput(I input, O target) {
    return this.output;
  }

  public float updateOutputf(I input, O target) {
    return this.outputf;
  }

  /**
   * Computing the gradient of the criterion with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   *
   * @param input  input data
   * @param target target data / labels
   * @return gradient of input
   */
  public abstract I updateGradInput(I input, O target);

  /**
   * Deep copy this criterion
   *
   * @return a deep copied criterion
   */
  AbstractCriterion<I, O> cloneCriterion() {
    throw new UnsupportedOperationException("Clone not supported");
  }

  @Override
  public int hashCode() {
    //TODO check correctness
    if (this.isFloat) {
      return Double.valueOf(outputf).hashCode();
    } else {
      return Double.valueOf(output).hashCode();
    }
  }

  public boolean canEqual(Object other) {
    return other instanceof AbstractCriterion;
  }

  @Override
  public boolean equals(Object other) {
    if (this.isFloat) {
      return other instanceof AbstractCriterion && other.getClass() == this.getClass()
          && this.outputf == ((AbstractCriterion<?, ?>) other).getOutput();
    } else {
      return other instanceof AbstractCriterion && other.getClass() == this.getClass()
          && this.output == ((AbstractCriterion<?, ?>) other).getOutput();
    }
  }

  enum SizeAverageStatus {
    TRUE,
    FALSE,
    NONE
  }
}
