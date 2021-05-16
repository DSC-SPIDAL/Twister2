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
package edu.iu.dsc.tws.dl.data;

import java.io.Serializable;

/**
 * Class that represents the features and labels of a data sample.
 *
 * @tparam T numeric type
 */
public interface Sample extends Serializable {

  /**
   * First dimension length of index-th feature.
   * This function could be used to sort samples in [[DataSet]].
   *
   * @return
   */
  int featureLength(int index);

  /**
   * First dimension length of index-th label.
   * This function could be used to find the longest label.
   *
   * @return
   */
  int labelLength(int index);

  /**
   * Number of tensors in feature
   *
   * @return number of tensors in feature
   */
  int numFeature();

  /**
   * Number of tensors in label
   *
   * @return number of tensors in label
   */
  int numLabel();

  //TODO: implement clone
//    /**
//     *@return A deep clone
//     */
//    override def clone(): this.type =
//      SerializationUtils.clone(this)

  /**
   * Get feature tensor, for one feature Sample only.
   * You don't need to override this, because we have add
   * a default implement to throw exception.
   *
   * @return feature tensor
   */
  Tensor feature();

  /**
   * Get feature tensor for given index
   *
   * @param index index of specific sample
   */
  Tensor feature(int index);

  /**
   * Get label tensor, for one label Sample only.
   * You don't need to override this, because we have add
   * a default implement to throw exception.
   *
   * @return label tensor
   */
  Tensor label();

  /**
   * Get label tensor for given index
   *
   * @param index index of specific sample
   */
  Tensor label(int index);

  /**
   * Get feature sizes
   *
   * @return feature sizes
   */
  int[][] getFeatureSize();


  /**
   * Get label sizes
   *
   * @return label sizes
   */
  int[][] getLabelSize();

  /**
   * Get data
   *
   * @return data
   */
  double[] getData();
}
