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
package edu.iu.dsc.tws.dl.optim.regularizer;

import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.optim.Regularizer;

/**
 * Apply both L1 and L2 regularization
 */
@SuppressWarnings("NeedBraces")
public class L1L2Regularizer extends Regularizer {

  //l1 regularization rate
  private double l1;

  //l2 regularization rate
  private double l2;

  private transient Tensor l1SignBuffer = null;

  public L1L2Regularizer(double l1, double l2) {
    this.l1 = l1;
    this.l2 = l2;
  }

  @Override
  public void accRegularization(Tensor parameter, Tensor gradParameter, double scale) {
    if (!preCheck(parameter, gradParameter)) {
      return;
    }
    accL1L2Regularization(l1, l2, parameter, gradParameter, scale);
  }

  @Override
  public void accRegularization(Tensor parameter, Tensor gradParameter, float scale) {
    if (!preCheck(parameter, gradParameter)) {
      return;
    }
    accL1L2Regularization(l1, l2, parameter, gradParameter, scale);
  }
  /**
   * Accumulates the gradient of the l1, l2 regularization of `parameter`
   * to `gradParameter`
   *
   * @param l1Alpha       l1 regularization rate
   * @param l2Alpha       l2 regularization rate
   * @param parameter     the parameter that is regularized
   * @param gradParameter the gradient of the parameter
   * @param scale         scale of gradParameters
   */
  private void accL1L2Regularization(
      double l1Alpha,
      double l2Alpha,
      Tensor parameter,
      Tensor gradParameter,
      double scale
  ) {
    accL1Regularization(l1Alpha, parameter, gradParameter, scale);
    accL2Regularization(l2Alpha, parameter, gradParameter, scale);
  }

  private void accL1L2Regularization(
      float l1Alpha,
      float l2Alpha,
      Tensor parameter,
      Tensor gradParameter,
      float scale
  ) {
    accL1Regularization(l1Alpha, parameter, gradParameter, scale);
    accL2Regularization(l2Alpha, parameter, gradParameter, scale);
  }

  /**
   * Accumulates the gradient of the l1 regularization of `parameter`
   * to `gradParameter`
   *
   * @param alpha         l1 regularization rate
   * @param parameter     the parameter that is regularized
   * @param gradParameter the gradient of the parameter
   * @param scale         scale of gradParameters
   */
  private void accL1Regularization(
      double alpha,
      Tensor parameter,
      Tensor gradParameter,
      double scale
  ) {
    if (alpha != 0 && scale != 0) {
      if (null == l1SignBuffer) l1SignBuffer = new DenseTensor(false);
      gradParameter.add(alpha * scale,
          l1SignBuffer.resizeAs(parameter).copy(parameter).sign());
    }
  }

  private void accL1Regularization(
      float alpha,
      Tensor parameter,
      Tensor gradParameter,
      float scale
  ) {
    if (alpha != 0 && scale != 0) {
      if (null == l1SignBuffer) l1SignBuffer = new DenseTensor(true);
      gradParameter.add(alpha * scale,
          l1SignBuffer.resizeAs(parameter).copy(parameter).sign());
    }
  }

  /**
   * Accumulates the gradient of the l2 regularization of `parameter`
   * to `gradParameter`
   *
   * @param alpha         l2 regularization rate
   * @param parameter     the parameter that is regularized
   * @param gradParameter the gradient of the parameter
   * @param scale         scale of gradParameters
   */
  private void accL2Regularization(
      double alpha,
      Tensor parameter,
      Tensor gradParameter,
      double scale
  ) {
    if (alpha != 0 && scale != 0) gradParameter.add(alpha * scale, parameter);
  }

  private void accL2Regularization(
      float alpha,
      Tensor parameter,
      Tensor gradParameter,
      float scale
  ) {
    if (alpha != 0 && scale != 0) gradParameter.add(alpha * scale, parameter);
  }
}
