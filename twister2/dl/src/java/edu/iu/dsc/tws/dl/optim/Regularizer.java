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
import java.io.Serializable;

import edu.iu.dsc.tws.dl.data.Tensor;

/**
 * abstract class for all regularizers.
 * Any regularizers need to inherit the result.
 */
public abstract class Regularizer implements Serializable {

  private boolean isRegualrized = false;

  /**
   * Enable the regularization feature
   */
  public void enable() {
    isRegualrized = true;
  }

  /**
   * Disable the regularization feature
   */
  public void disable() {
    isRegualrized = false;
  }

  /**
   * The method need to be override by the concrete regularizer class
   * It accumulates the gradient of the regularization of `parameter` to `gradParameter`
   *
   * @param parameter     the parameter that is regularized
   * @param gradParameter the gradient of the parameter
   * @param scale         the scale of gradParameters
   */
  public abstract void accRegularization(Tensor parameter, Tensor gradParameter, double scale);

  public abstract void accRegularization(Tensor parameter, Tensor gradParameter, float scale);

  /**
   * Check the regularization is applied or not
   *
   * @param parameter     the parameter that is regularized
   * @param gradParameter the gradient of the parameter
   * @return a boolean, if true, accumulates the gradient of regularization,
   * otherwise not.
   */
  protected boolean preCheck(Tensor parameter, Tensor gradParameter) {
    return null != parameter && null != gradParameter && isRegualrized;
  }
}
