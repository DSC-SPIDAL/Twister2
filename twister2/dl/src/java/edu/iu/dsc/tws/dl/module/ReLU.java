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
package edu.iu.dsc.tws.dl.module;

/**
 * Applies the rectified linear unit (ReLU) function element-wise to the input Tensor
 * Thus the output is a Tensor of the same dimension
 * ReLU function is defined as:
 * f(x) = max(0, x)
 */
public class ReLU extends Threshold {

  public ReLU(double threshold, double value, boolean inPlace) {
    super(threshold, value, inPlace);
  }

  public ReLU() {
    this(false);
  }

  public ReLU(boolean inPlace) {
    super(0.0, 0.0, inPlace);
  }
}
