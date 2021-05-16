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

import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.optim.Regularizer;

public class IRLinear extends IROperator {
  //the size the each input sample
  private int inputSize;
  //the size of the module ((DenseTensor output) of each sample
  private int outputSize;
  //instance of [[Regularizer]]
  // (eg. L1 or L2 regularization), applied to the input weights matrices.
  private Regularizer wRegularizer;
  //instance of [[Regularizer]]
  //applied to the bias.
  private Regularizer bRegularizer;
  private boolean withBias = true;
  private Tensor initWeight;
  private Tensor initBias;
  private Tensor initGradWeight;
  private Tensor initGradBias;

  public IRLinear(int inputSize, int outputSize, Regularizer wRegularizer, Regularizer bRegularizer,
                  boolean withBias, Tensor initWeight, Tensor initBias, Tensor initGradWeight,
                  Tensor initGradBias) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.wRegularizer = wRegularizer;
    this.bRegularizer = bRegularizer;
    this.withBias = withBias;
    this.initWeight = initWeight;
    this.initBias = initBias;
    this.initGradWeight = initGradWeight;
    this.initGradBias = initGradBias;
  }
}
