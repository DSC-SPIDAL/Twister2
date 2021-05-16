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
 * An implementation of Adagrad. See the original paper:
 * http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
 *
 * @param learningRate      learning rate
 * @param learningRateDecay learning rate decay
 * @param weightDecay       weight decay
 * @tparam T
 */
@SuppressWarnings({"LocalVariableName", "HiddenField", "NeedBraces"})
public class Adagrad implements OptimMethod {
  private double learningRate = 1e-3;
  private double learningRateDecay = 0.0;
  private double weightDecay = 0.0;

  public Adagrad() {
    initState();
  }

  public Adagrad(double learningRate, double learningRateDecay, double weightDecay) {
    this.learningRate = learningRate;
    this.learningRateDecay = learningRateDecay;
    this.weightDecay = weightDecay;
  }

  /**
   * Adagrad implementation for Adagrad
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
      double wd = this.weightDecay;

      //double (fx, eval.getValue1()) = feval(parameter)
      DoubleTensorPair eval = feval.apply(parameter);

      if (wd != 0) {
        eval.getValue1().add(wd, parameter);
      }

      double clr = lr / (1 + nevals * lrd);

      Tensor _paramVariance;
      Tensor _paramStd;

      if (state.<Tensor>get("paramVariance") != null) {
        _paramVariance = state.<Tensor>get("paramVariance");
        _paramStd = state.<Tensor>get("paramStd");
      } else {
        _paramVariance = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
        _paramStd = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
      }

      _paramVariance.addcmul(1.0, eval.getValue1(), eval.getValue1());
      _paramStd.resizeAs(_paramVariance).copy(_paramVariance).sqrt();
      parameter.addcdiv(-clr, eval.getValue1(), _paramStd.add(1e-10));

      state.put("evalCounter", nevals + 1);
      state.put("paramVariance", _paramVariance); // vector of temporal variances of parameters
      state.put("paramStd", _paramStd);

      return new TensorAndArrayPair(parameter, new double[]{eval.getValue0()});
    } else {
      float lr = (float) this.learningRate;
      float lrd = (float) this.learningRateDecay;
      float wd = (float) this.weightDecay;

      //double (fx, eval.getValue1()) = feval(parameter)
      DoubleTensorPair eval = feval.apply(parameter);

      if (wd != 0) {
        eval.getValue1().add(wd, parameter);
      }
      float clr = lr / (1f + nevals * lrd);

      Tensor _paramVariance;
      Tensor _paramStd;

      if (state.<Tensor>get("paramVariance") != null) {
        _paramVariance = state.<Tensor>get("paramVariance");
        _paramStd = state.<Tensor>get("paramStd");
      } else {
        _paramVariance = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
        _paramStd = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
      }

      _paramVariance.addcmul(1.0f, eval.getValue1(), eval.getValue1());
      _paramStd.resizeAs(_paramVariance).copy(_paramVariance).sqrt();
      parameter.addcdiv(-clr, eval.getValue1(), _paramStd.add(1e-10f));

      state.put("evalCounter", nevals + 1);
      state.put("paramVariance", _paramVariance); // vector of temporal variances of parameters
      state.put("paramStd", _paramStd);

      return new TensorAndArrayPair(parameter, new double[]{eval.getValue0()});
    }
  }

  @Override
  public void clearHistory() {
    state.delete("paramVariance");
    state.delete("paramStd");
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
