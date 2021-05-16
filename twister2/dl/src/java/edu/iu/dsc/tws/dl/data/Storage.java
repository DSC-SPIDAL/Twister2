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
 * Storage defines a simple storage interface that controls the underlying storage for
 * any tensor object.
 */
public interface Storage extends Serializable, Iterable {

  /**
   * Returns the number of elements in the storage. The original method name in torch is size,
   * which is conflict with
   * Iterable
   *
   * @return
   */
  int length();

//  /**
//   * Returns the element at position index in the storage.
//   Valid range of index is 0 to length() -1.
//   *
//   * @param index
//   * @return
//   */
//  T apply(int index);

  /**
   * Set the element at position index in the storage. Valid range of index is 1 to length()
   *
   * @param index
   * @param value
   */
  void update(int index, double value);

  /**
   * Set the element at position index in the storage. Valid range of index is 1 to length()
   *
   * @param index
   * @param value
   */
  void update(int index, float value);

  /**
   * Copy another storage. The types of the two storages might be different: in that case a
   * conversion of types occur
   * (which might result, of course, in loss of precision or rounding). This method returns itself.
   *
   * @param source
   * @return
   */
  Storage copy(Storage source, int offset, int sourceOffset, int length);

  default Storage copy(Storage source) {
    return copy(source, 0, 0, length());
  }

  /**
   * Fill the Storage with the given value. This method returns itself.
   *
   * @param value
   * @param offset offset start from 1
   * @param length length of fill part
   * @return
   */
  Storage fill(double value, int offset, int length);

  /**
   * Fill the Storage with the given value. This method returns itself.
   *
   * @param value
   * @param offset offset start from 1
   * @param length length of fill part
   * @return
   */
  Storage fill(float value, int offset, int length);

  /**
   * Resize the storage to the provided size. The new contents are undetermined.
   * This function returns itself
   *
   * @param size
   * @return
   */
  Storage resize(int size);

  /**
   * Convert the storage to an array
   *
   * @return
   */
  double[] toDoubleArray();

  /**
   * Convert the storage to an array
   *
   * @return
   */
  float[] toFloatArray();

  /**
   * Get the element type in the storage
   *
   * @return
   */
  //  def getType() : DataType

  /**
   * Share the same underlying storage
   *
   * @param other
   * @return
   */
  Storage set(Storage other);
}

