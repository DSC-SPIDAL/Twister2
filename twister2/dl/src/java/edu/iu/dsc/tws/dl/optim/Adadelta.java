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
 * Adadelta implementation for SGD: http://arxiv.org/abs/1212.5701
 *
 * @param decayRate decayRate, also called interpolation parameter rho
 * @param Epsilon   for numerical stability
 * @tparam T
 */
@SuppressWarnings({"LocalVariableName", "HiddenField", "NeedBraces"})
public class Adadelta implements OptimMethod {
  private double decayRate = 0.9;
  private double epsilon = 1e-10;

  public Adadelta() {
    initState();
  }

  public Adadelta(double decayRate, double epsilon) {
    initState();
    this.decayRate = decayRate;
    this.epsilon = epsilon;
  }

  /**
   * Adadelta implementation for SGD: http://arxiv.org/abs/1212.5701
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   *                  state("paramVariance") : vector of temporal variances of parameters
   *                  state("accDelta"): vector of accumulated delta of gradients
   * @return the new x vector and the function list {fx}, evaluated before the update
   */
  @Override
  public TensorAndArrayPair optimize(OptimFunction feval, Tensor parameter) {
    int nevals = state.<Integer>getOrDefault("evalCounter", 0);
    if (!parameter.isFloat()) {
      double dr = this.decayRate;
      double eps = this.epsilon;

      DoubleTensorPair eval = feval.apply(parameter);

      Tensor _paramVariance;
      Tensor _paramStd;
      Tensor _delta;
      Tensor _accDelta;

      if (state.<Tensor>get("paramVariance") != null) {
        _paramVariance = state.<Tensor>get("paramVariance");
        _paramStd = state.<Tensor>get("paramStd");
        _delta = state.<Tensor>get("delta");
        _accDelta = state.<Tensor>get("accDelta");
      } else {
        _paramVariance = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
        _paramStd = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
        _delta = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
        _accDelta = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
      }

      _paramVariance.mul(dr).addcmul(1.0 - dr, eval.getValue1(), eval.getValue1());
      _paramStd.copy(_paramVariance).add(eps).sqrt();
      _delta.copy(_accDelta).add(eps).sqrt()
          .cdiv(_paramStd).cmul(eval.getValue1());
      parameter.add(-1.0, _delta);
      _accDelta.mul(dr).addcmul(1.0 - dr, _delta, _delta);
      state.put("evalCounter", nevals + 1);
      state.put("paramVariance", _paramVariance);
      state.put("paramStd", _paramStd);
      state.put("delta", _delta);
      state.put("accDelta", _accDelta);

      return new TensorAndArrayPair(parameter, new double[]{eval.getValue0()});
    } else {
      float dr = (float) this.decayRate;
      float eps = (float) this.epsilon;

      DoubleTensorPair eval = feval.apply(parameter);

      Tensor _paramVariance;
      Tensor _paramStd;
      Tensor _delta;
      Tensor _accDelta;

      if (state.<Tensor>get("paramVariance") != null) {
        _paramVariance = state.<Tensor>get("paramVariance");
        _paramStd = state.<Tensor>get("paramStd");
        _delta = state.<Tensor>get("delta");
        _accDelta = state.<Tensor>get("accDelta");
      } else {
        _paramVariance = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
        _paramStd = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
        _delta = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
        _accDelta = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
      }

      _paramVariance.mul(dr).addcmul(1.0f - dr, eval.getValue1(), eval.getValue1());
      _paramStd.copy(_paramVariance).add(eps).sqrt();
      _delta.copy(_accDelta).add(eps).sqrt()
          .cdiv(_paramStd).cmul(eval.getValue1());
      parameter.add(-1.0f, _delta);
      _accDelta.mul(dr).addcmul(1.0f - dr, _delta, _delta);
      state.put("evalCounter", nevals + 1);
      state.put("paramVariance", _paramVariance);
      state.put("paramStd", _paramStd);
      state.put("delta", _delta);
      state.put("accDelta", _accDelta);

      return new TensorAndArrayPair(parameter, new double[]{eval.getValue0()});

    }
  }

  @Override
  public void clearHistory() {
    state.delete("paramVariance");
    state.delete("paramStd");
    state.delete("delta");
    state.delete("accDelta");
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
    return 0.0;
  }

  @Override
  public OptimMethod loadFromTable(Table config) {
    throw new UnsupportedOperationException("Operation not supported");
  }
}
