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
package edu.iu.dsc.tws.dl.utils.graph;

import java.io.Serializable;

import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.utils.pair.TensorPair;

public class IRElement implements Serializable {

  protected String name;
  private IROperator op;
  private Tensor weights = null;
  private Tensor gradWeights = null;

  public IRElement(String name, IROperator op, Tensor weights, Tensor gradWeights) {
    this.name = name;
    this.op = op;
    this.weights = weights;
    this.gradWeights = gradWeights;
  }

  /**
   * set weight and bias
   */
  public void setWeights(Tensor weightsAndBias) {
    weights = weightsAndBias;
  }

  /**
   * set gradWeight and gradbias
   */
  public void setGradWeights(Tensor gradWeightsAndGradBias) {
    gradWeights = gradWeightsAndGradBias;
  }

  public TensorPair getParameters() {
    return new TensorPair(weights, gradWeights);
  }

  public String getName() {
    return this.name;
  }

  public IROperator getOp() {
    return this.op;
  }
}
