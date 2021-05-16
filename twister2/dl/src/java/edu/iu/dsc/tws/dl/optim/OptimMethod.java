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

import edu.iu.dsc.tws.dl.data.Table;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.utils.pair.TensorAndArrayPair;

@SuppressWarnings("ConstantName")
public interface OptimMethod extends Serializable {

  // a table describing the state of the optimizer; after each call the state is modified
  Table state = new Table();

  /**
   * Optimize the model parameter
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation,
   *                  and returns f(X) and df/dX
   * @param parameter the initial point
   * @return the new x vector and the function list, evaluated before the update
   */
  TensorAndArrayPair optimize(OptimFunction feval, Tensor parameter);

  /**
   * Clear the history information in the OptimMethod state
   *
   * @return
   */
  void clearHistory();

  /**
   * Update hyper parameter.
   * We have updated hyper parameter in method optimize(). But in DistriOptimizer, the method
   * optimize() is only called on the executor side, the driver's hyper parameter is unchanged.
   * So this method is using to update hyper parameter on the driver side.
   *
   */
  void updateHyperParameter();


  /**
   * Get hyper parameter from config table.
   */
  String getHyperParameter();

//  /**
//   * clone OptimMethod
//   *
//   * @return
//   */
//  @override
//  OptimMethod clone(){
//    //SerializationUtils.clone(this)
//  };

  /**
   * get learning rate
   *
   * @return
   */
  double getLearningRate();

  /**
   * save OptimMethod
   *
   * @param path      path
   * @param overWrite whether to overwrite
   * @return
   */
  default OptimMethod save(String path, boolean overWrite) {
    //TODO: complete save
    this.clearHistory();
    //File.save(this, path, overWrite);
    return this;
  }

  /**
   * load optimMethod parameters from Table
   *
   * @param config
   * @return
   */
  OptimMethod loadFromTable(Table config);

  default void initState() {
    state.put("epoch", 1);
    state.put("neval", 1);
  }
}

