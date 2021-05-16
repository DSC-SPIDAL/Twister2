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
package edu.iu.dsc.tws.dl.optim;

import edu.iu.dsc.tws.dl.data.Table;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.utils.pair.DoubleTensorPair;
import edu.iu.dsc.tws.dl.utils.pair.TensorAndArrayPair;

/**
 * An implementation of RMSprop
 *
 * @param learningRate      learning rate
 * @param learningRateDecay learning rate decay
 * @param decayRate         decayRate, also called rho
 * @param Epsilon           for numerical stability
 * @tparam T
 */
@SuppressWarnings({"LocalVariableName", "HiddenField", "NeedBraces"})
public class RMSprop implements OptimMethod {
  private double learningRate = 1e-2;
  private double learningRateDecay = 0.0;
  private double decayRate = 0.99;
  private double epsilon = 1e-8;

  public RMSprop() {
    initState();
  }

  public RMSprop(double learningRate, double learningRateDecay, double decayRate, double epsilon) {
    this.learningRate = learningRate;
    this.learningRateDecay = learningRateDecay;
    this.decayRate = decayRate;
    this.epsilon = epsilon;
  }

  /**
   * An implementation of RMSprop
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   * @return the new x vector and the function list, evaluated before the update
   */
  @Override
  public TensorAndArrayPair optimize(OptimFunction feval, Tensor parameter) {

    int nevals = state.<Integer>getOrDefault("evalCounter", 0);
    if (!parameter.isFloat()) {
      double lr = this.learningRate;
      double lrd = this.learningRateDecay;
      double dr = this.decayRate;
      double eps = this.epsilon;

      //double (fx, eval.getValue1()) = feval(parameter)
      DoubleTensorPair eval = feval.apply(parameter);

      double clr = lr / (1 + nevals * lrd);

      Tensor _sumofsquare;
      Tensor _rms;

      if (state.<Tensor>get("sumSquare") != null) {
        _sumofsquare = state.<Tensor>get("sumSquare");
        _rms = state.<Tensor>get("rms");
      } else {
        _sumofsquare = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
        _rms = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
      }

      _sumofsquare.mul(dr).addcmul(1 - dr, eval.getValue1(), eval.getValue1());
      _rms.sqrt(_sumofsquare).add(eps);
      parameter.addcdiv(-clr, eval.getValue1(), _rms);
      state.put("evalCounter", nevals + 1);
      state.put("sumSquare", _sumofsquare);
      state.put("rms", _rms);

      return new TensorAndArrayPair(parameter, new double[]{eval.getValue0()});
    } else {
      float lr = (float) this.learningRate;
      float lrd = (float) this.learningRateDecay;
      float dr = (float) this.decayRate;
      float eps = (float) this.epsilon;

      //double (fx, eval.getValue1()) = feval(parameter)
      DoubleTensorPair eval = feval.apply(parameter);

      float clr = lr / (1f + nevals * lrd);

      Tensor _sumofsquare;
      Tensor _rms;

      if (state.<Tensor>get("sumSquare") != null) {
        _sumofsquare = state.<Tensor>get("sumSquare");
        _rms = state.<Tensor>get("rms");
      } else {
        _sumofsquare = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
        _rms = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
      }

      _sumofsquare.mul(dr).addcmul(1.0f - dr, eval.getValue1(), eval.getValue1());
      _rms.sqrt(_sumofsquare).add(eps);
      parameter.addcdiv(-clr, eval.getValue1(), _rms);
      state.put("evalCounter", nevals + 1);
      state.put("sumSquare", _sumofsquare);
      state.put("rms", _rms);

      return new TensorAndArrayPair(parameter, new double[]{eval.getValue0()});
    }
  }

  @Override
  public void clearHistory() {
    state.delete("sumSquare");
    state.delete("rms");
  }

  @Override
  public void updateHyperParameter() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public String getHyperParameter() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double getLearningRate() {
    return this.learningRate;
  }

  @Override
  public OptimMethod loadFromTable(Table config) {
    throw new UnsupportedOperationException("Operation not supported");
  }
}
