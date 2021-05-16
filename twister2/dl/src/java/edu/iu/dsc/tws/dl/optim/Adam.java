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

@SuppressWarnings({"LocalVariableName", "HiddenField", "NeedBraces"})
public class Adam implements OptimMethod {
  private transient Tensor buffer;
  private double learningRate = 1e-3;
  private double learningRateDecay = 0.0;
  private double beta1 = 0.9;
  private double beta2 = 0.999;
  private double epsilon = 1e-8;

  public Adam() {
    initState();
  }

  public Adam(double learningRate, double learningRateDecay, double beta1,
              double beta2, double epsilon) {
    initState();
    this.learningRate = learningRate;
    this.learningRateDecay = learningRateDecay;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
  }

  public double getLearningRateDecay() {
    return learningRateDecay;
  }

  public void setLearningRateDecay(double learningRateDecay) {
    this.learningRateDecay = learningRateDecay;
  }

  public double getBeta1() {
    return beta1;
  }

  public void setBeta1(double beta1) {
    this.beta1 = beta1;
  }

  public double getBeta2() {
    return beta2;
  }

  public void setBeta2(double beta2) {
    this.beta2 = beta2;
  }

  public double getEpsilon() {
    return epsilon;
  }

  public void setEpsilon(double epsilon) {
    this.epsilon = epsilon;
  }

  /**
   * An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   * @return the new x vector and the function list {fx}, evaluated before the update
   */
  @Override
  public TensorAndArrayPair optimize(OptimFunction feval, Tensor parameter) {
    if (buffer == null) buffer = new DenseTensor(parameter.isFloat());
    if (!parameter.isFloat()) {
      double lr = this.learningRate;
      double lrd = this.learningRateDecay;
      double beta1 = this.beta1;
      double beta2 = this.beta2;
      double eps = this.epsilon;

      DoubleTensorPair eval = feval.apply(parameter);

      int timestep = state.<Integer>getOrDefault("evalCounter", 0);

      Tensor _s;
      Tensor _r;
      Tensor _denom;

      if (state.<Tensor>get("s") != null) {
        _s = state.<Tensor>get("s");
        _r = state.<Tensor>get("r");
        _denom = state.<Tensor>get("denom").resizeAs(eval.getValue1());
      } else {
        _s = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
        _r = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
        _denom = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
      }
      double clr = lr / (1 + timestep * lrd);

      timestep = timestep + 1;

      /**
       * m_t = beta_1 * m_t-1 + (1 - beta_1) * g_t
       * v_t = beta_2 * v_t-1 + (1 - beta_2) * g_t * g_t
       */
      _s.mul(beta1).add(1 - beta1, eval.getValue1());
      // buffer = eval.getValue1() * eval.getValue1()
      buffer.resizeAs(eval.getValue1()).cmul(eval.getValue1(), eval.getValue1());
      _r.mul(beta2).add(1 - beta2, buffer);
      _denom.sqrt(_r);

      // used as MKL.axpy: 1 * a + y = y, and fill buffer with one
      buffer.fill(1.0);
      _denom.add(eps, buffer);

      // efficiency improved upon by changing the order of computation, at expense of clarity
      double biasCorrection1 = 1 - Math.pow(beta1, timestep);
      double biasCorrection2 = 1 - Math.pow(beta2, timestep);
      double stepSize = clr * Math.sqrt(biasCorrection2) / biasCorrection1;
      parameter.addcdiv(-stepSize, _s, _denom);

      state.put("evalCounter", timestep); // A tmp tensor to hold the sqrt(v) + epsilon
      state.put("s", _s); // 1st moment variables
      state.put("r", _r); // 2nd moment variables
      state.put("denom", _denom); // 3nd moment variables

      return new TensorAndArrayPair(parameter, new double[]{eval.getValue0()});
    } else {
      float lr = (float) this.learningRate;
      float lrd = (float) this.learningRateDecay;
      float beta1 = (float) this.beta1;
      float beta2 = (float) this.beta2;
      float eps = (float) this.epsilon;

      DoubleTensorPair eval = feval.apply(parameter);

      int timestep = state.<Integer>getOrDefault("evalCounter", 0);

      Tensor _s;
      Tensor _r;
      Tensor _denom;

      if (state.<Tensor>get("s") != null) {
        _s = state.<Tensor>get("s");
        _r = state.<Tensor>get("r");
        _denom = state.<Tensor>get("denom").resizeAs(eval.getValue1());
      } else {
        _s = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
        _r = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
        _denom = new DenseTensor(parameter.isFloat()).resizeAs(eval.getValue1()).zero();
      }
      float clr = lr / (1 + timestep * lrd);

      timestep = timestep + 1;

      /**
       * m_t = beta_1 * m_t-1 + (1 - beta_1) * g_t
       * v_t = beta_2 * v_t-1 + (1 - beta_2) * g_t * g_t
       */
      _s.mul(beta1).add(1 - beta1, eval.getValue1());
      // buffer = eval.getValue1() * eval.getValue1()
      buffer.resizeAs(eval.getValue1()).cmul(eval.getValue1(), eval.getValue1());
      _r.mul(beta2).add(1 - beta2, buffer);
      _denom.sqrt(_r);

      // used as MKL.axpy: 1 * a + y = y, and fill buffer with one
      buffer.fill(1.0f);
      _denom.add(eps, buffer);

      // efficiency improved upon by changing the order of computation, at expense of clarity
      float biasCorrection1 = (float) (1 - Math.pow(beta1, timestep));
      float biasCorrection2 = (float) (1 - Math.pow(beta2, timestep));
      float stepSize = (float) (clr * Math.sqrt(biasCorrection2) / biasCorrection1);
      parameter.addcdiv(-stepSize, _s, _denom);

      state.put("evalCounter", timestep); // A tmp tensor to hold the sqrt(v) + epsilon
      state.put("s", _s); // 1st moment variables
      state.put("r", _r); // 2nd moment variables
      state.put("denom", _denom); // 3nd moment variables

      return new TensorAndArrayPair(parameter, new float[]{(float) eval.getValue0()});
    }
  }

  @Override
  public void clearHistory() {
    state.delete("s");
    state.delete("r");
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

  public void setLearningRate(double learningRate) {
    this.learningRate = learningRate;
  }

  @Override
  public OptimMethod loadFromTable(Table config) {
    throw new UnsupportedOperationException("Operation not supported");
  }
}
