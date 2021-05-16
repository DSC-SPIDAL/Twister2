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
import edu.iu.dsc.tws.dl.utils.VariableFormat;

public interface InitializationMethod extends Serializable {
  /**
   * Initialize the given weight and bias.
   *
   * @param variable   the weight to initialize
   * @param dataFormat the data format of weight indicating the dimension order of
   *                   the weight. "output_first" means output is in the lower dimension
   *                   "input_first" means input is in the lower dimension.
   */
  void init(Tensor variable, VariableFormat dataFormat);
}
