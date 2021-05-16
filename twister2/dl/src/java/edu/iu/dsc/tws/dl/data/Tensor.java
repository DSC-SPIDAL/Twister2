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
 * Tensor class
 */
@SuppressWarnings("NoClone")
public interface Tensor extends Activity, Serializable, TensorMath {

  /**
   * Check if Tensor type is float
   * @return true if float tensor
   */
  boolean isFloat();

  /**
   * Check if empty
   *
   * @return whether this tensor is an empty tensor. Note that nDimension == 0 is not
   * sufficient to determine a tensor is empty, because a scalar tensor's nDimension
   * is also 0.
   */
  boolean isEmpty();

  /**
   * check if scalar
   *
   * @return whether this tensor is a scalar
   */
  boolean isScalar();

  /**
   * Dimension number of the tensor. For empty tensor, its dimension number is 0
   *
   * @return dimension number
   */
  int nDimension();

  /**
   * A shortcut of nDimension()
   *
   * @see Tensor#nDimension()
   */
  int dim();

  /**
   * Size of tensor. Return an array of which each value represents the size on the
   * dimension(i + 1), i is the index of the corresponding value.
   * It will generate a new array each time method is invoked.
   *
   * @return size array
   */
  int[] size();

  /**
   * size of the tensor on the given dimension
   *
   * @param dim dimension, count from 1
   * @return size
   */
  int size(int dim);

  /**
   * Jumps between elements on the each dimension in the storage.
   * It will generate a new array each time method is invoked.
   *
   * @return strides array
   */
  int[] stride();

  /**
   * Jumps between elements on the given dimension in the storage.
   *
   * @param dim dimension, count from 1
   * @return jump
   */
  int stride(int dim);

  /**
   * Fill with a given value. It will change the value of the current tensor and return itself
   *
   * @param v value to fill the tensor
   * @return current tensor
   */
  Tensor fill(double v);

  Tensor fill(float v);

  /**
   * Fill with a given value. It will change the value of the current tensor and return itself
   * <p>
   * Note the value should be an instance of double
   *
   * @param v value to fill the tensor
   * @return current tensor
   */
  Tensor forceFill(double v);

  Tensor forceFill(float v);

  /**
   * Fill with zero. It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  Tensor zero();

  /**
   * Fill with random value(normal gaussian distribution).
   * It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  Tensor randn();

  /**
   * Fill with random value(normal gaussian distribution with the specified mean
   * and stdv).
   * It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  Tensor randn(double mean, double stdv);

  Tensor randn(float mean, float stdv);

  /**
   * Fill with random value(uniform distribution).
   * It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  Tensor rand();

  /**
   * Fill with random value(uniform distribution between [lowerBound, upperBound])
   * It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  Tensor rand(double lowerBound, double upperBound);

  Tensor rand(float lowerBound, float upperBound);

  /**
   * Fill with random value(bernoulli distribution).
   * It will change the value of the current tensor and return itself
   *
   * @return current tensor
   */
  Tensor bernoulli(double p);

  Tensor bernoulli(float p);

  /**
   * Create a new tensor which exchanges the given dimensions of the current tensor
   *
   * @param dim1 dimension to be exchanged, count from one
   * @param dim2 dimension to be exchanged, count from one
   * @return new tensor
   */
  Tensor transpose(int dim1, int dim2);

  /**
   * Shortcut of transpose(1, 2) for 2D tensor
   *
   * @see Tensor#transpose(int, int)
   */
  Tensor t();

  /**
   * Query tensor on a given index. Tensor should not be empty
   *
   * @param index count from 1
   * @return
   */
  Tensor apply(int index);

  /**
   * Query the value on a given index. Tensor should not be empty
   *
   * @param indexes the indexes length should be same as the tensor dimension length and each
   *                value count from 1
   * @return the value on the given index
   */
  double apply(int[] indexes);

  float applyf(int[] indexes);
  /**
   * Get value
   *
   * @return the value of a scalar. Requires the tensor to be a scalar.
   */
  double value();

  /**
   * Query the value on a given position. The number of parameters
   * should be equal to the dimension number of the tensor.
   * Tensor should not be empty.
   *
   * @param d1 the given position
   * @return the value on a given position
   */
  double valueAt(int d1);

  double valueAt(int d1, int d2);

  double valueAt(int d1, int d2, int d3);

  double valueAt(int d1, int d2, int d3, int d4);

  double valueAt(int d1, int d2, int d3, int d4, int d5);

  float valueAtf(int d1);

  float valueAtf(int d1, int d2);

  float valueAtf(int d1, int d2, int d3);

  float valueAtf(int d1, int d2, int d3, int d4);

  float valueAtf(int d1, int d2, int d3, int d4, int d5);
  /**
   * Subset the tensor by apply the elements of the given table to the corresponding dimension
   * of the tensor. The elements of the given table can be an int or another Table.
   * An int means select on current dimension; A table means narrow on current dimension,
   * the table should have two elements, of which the first is the start index and
   * the second is the end index. An empty table is equal to Table(1, size_of_current_dimension)
   * If the table length is less than the tensor dimension, each missing dimension is token up by
   * an empty table
   *
   * @param t The table length should be less than or equal to the tensor dimensions
   * @return
   */
  Tensor apply(Table t);

  /**
   * For tensor(i) = value. If tensor(i) is another tensor, it will fill the selected subset by
   * the given value
   *
   * @param index index
   * @param value value to write
   */
  void update(int index, double value);

  void update(int index, float value);

  /**
   * Copy the give tensor value to the select subset of the current tensor by the given index.
   * The subset should have the same size of the given tensor
   *
   * @param index index
   * @param src   tensor to write
   */
  void update(int index, Tensor src);

  /**
   * Write the value to the positions indexed by the given index array
   *
   * @param indexes index array. It should has same length with the tensor dimension
   * @param value   value to write
   */
  void update(int[] indexes, double value);

  void update(int[] indexes, float value);


  /**
   * Set value for a scalar tensor
   *
   * @param value the written value
   * @return
   */
  Tensor setValue(double value);

  Tensor setValue(float value);

  /**
   * Write the value on a given position. The number of parameters
   * should be equal to the dimension number of the tensor.
   *
   * @param d1    the given position
   * @param value the written value
   * @return
   */
  Tensor setValue(int d1, double value);

  Tensor setValue(int d1, float value);

  Tensor setValue(int d1, int d2, double value);

  Tensor setValue(int d1, int d2, float value);

  Tensor setValue(int d1, int d2, int d3, double value);

  Tensor setValue(int d1, int d2, int d3, float value);

  Tensor setValue(int d1, int d2, int d3, int d4, double value);

  Tensor setValue(int d1, int d2, int d3, int d4, float value);

  Tensor setValue(int d1, int d2, int d3, int d4, int d5, double value);

  Tensor setValue(int d1, int d2, int d3, int d4, int d5, float value);

  /**
   * Fill the select subset of the current tensor with the given value.
   * The element of the given table can be an int or another Table. An int means select on current
   * dimension; A table means narrow on the current dimension, the table should has two elements,
   * of which the first is the start index and the second is the end index. An empty table is equal
   * to Table(1, size_of_current_dimension) If the table length is less than the tensor dimension,
   * each missing dimension is applied by an empty table
   *
   * @param t     subset table
   * @param value value to write
   */
  void update(Table t, double value);

  void update(Table t, float value);

  /**
   * Copy the given tensor values to the selected subset of the current tensor
   * Each element of the given table can be an int or another Table. An int means select on current
   * dimension; A table means narrow on current dimension, the table should has two elements,
   * of which the first is start index and the second is the end index. An empty table is equal
   * to Table(1, size_of_current_dimension). If the table's length is smaller than the tensor's
   * dimension, the missing dimension is applied by an empty table.
   *
   * @param t   subset table
   * @param src tensor to copy
   */
  void update(Table t, Tensor src);

  /**
   * Check if the tensor is contiguous on the storage
   *
   * @return true if it's contiguous
   */
  boolean isContiguous();

  /**
   * Get a contiguous tensor from current tensor
   *
   * @return the current tensor if it's contiguous; or a new contiguous tensor with separated
   * storage
   */
  Tensor contiguous();

  /**
   * Check if the size is same with the give tensor
   *
   * @param other tensor to be compared
   * @return true if they have same size
   */
  boolean isSameSizeAs(Tensor other);

  /**
   * return a new empty tensor of the same type
   *
   * @return new tensor
   */
  Tensor emptyInstance();

  /**
   * Resize the current tensor to the same size of the given tensor. It will still use the same
   * storage if the storage
   * is sufficient for the new size
   *
   * @param src target tensor
   * @return current tensor
   */
  Tensor resizeAs(Tensor src);

  /**
   * Resize the current tensor to the give shape
   *
   * @param sizes   Array describe the size
   * @param strides Array describe the jumps
   * @return
   */
  Tensor resize(int[] sizes, int[] strides);

  Tensor resize(int[] sizes);

  Tensor resize(int size1);

  Tensor resize(int size1, int size2);

  Tensor resize(int size1, int size2, int size3);

  Tensor resize(int size1, int size2, int size3, int size4);

  Tensor resize(int size1, int size2, int size3, int size4, int size5);

  //  def repeatTensor(result: Tensor, tensor: Tensor, int size*)

  /**
   * Element number
   *
   * @return element number
   */
  int nElement();

  /**
   * Remove the dim-th dimension and return the subset part. For instance
   * tensor =
   * 1 2 3
   * 4 5 6
   * tensor.select(1, 1) is [1 2 3]
   * tensor.select(1, 2) is [4 5 6]
   * tensor.select(2, 3) is [3 6]
   *
   * @param dim
   * @param index
   * @return
   */
  Tensor select(int dim, int index);

  Tensor set(Tensor other);

  /**
   * The Tensor is now going to "view" the given storage, starting at position storageOffset (>=1)
   * with the given dimension sizes and the optional given strides. As the result, any
   * modification in the elements of the Storage will have an impact on the elements of the Tensor,
   * and vice-versa. This is an efficient method, as there is no memory copy!
   * <p>
   * If only storage is provided, the whole storage will be viewed as a 1D Tensor.
   *
   * @param storage
   * @param storageOffset default 1
   * @param sizes
   * @param strides
   * @return current tensor
   */
  Tensor set(Storage storage, int storageOffset, int[] sizes, int[] strides);

  /**
   * Shrunk the size of the storage to 0, and also the tensor size
   *
   * @return
   */
  Tensor set();

  /**
   * Get the storage
   *
   * @return storage
   */
  Storage storage();

  /**
   * tensor offset on the storage
   *
   * @return storage offset, count from 1
   */
  int storageOffset();

  /**
   * Get a subset of the tensor on dim-th dimension. The offset is given by index, and length is
   * given by size. The important difference with select is that it will not reduce the dimension
   * number. For Instance
   * tensor =
   * 1 2 3
   * 4 5 6
   * tensor.narrow(1, 1, 1) is [1 2 3]
   * tensor.narrow(2, 2, 2) is
   * 2 3
   * 5 6
   *
   * @param dim
   * @param index
   * @param size
   * @return
   */
  Tensor narrow(int dim, int index, int size);

  /**
   * Copy the value of the given tensor to the current. They should have same size. It will use
   * the old storage (storage is not used for now)
   *
   * @param other source tensor
   * @return current tensor
   */
  Tensor copy(Tensor other);

  /**
   * Removes all singleton dimensions of the tensor
   *
   * @return current tensor
   */
  Tensor squeeze();

  /**
   * Removes given dimensions of the tensor if it's singleton
   *
   * @return current tensor
   */
  Tensor squeeze(int dim);

  /**
   * Create a new tensor that removes all singleton dimensions of the tensor
   *
   * @return create a new tensor
   */
  Tensor squeezeNewTensor();

  /**
   * Return a new tensor with specified sizes. The input tensor must be contiguous, and the
   * elements number in the given sizes must be equal to the current tensor
   *
   * @param sizes
   * @return new tensor
   */

  Tensor view(int[] sizes);

  /**
   * Returns a tensor which contains all slices of size @param size
   * in the dimension. Step between two slices is given by @param step.
   *
   * @param dim  dimension
   * @param step Step between two slices
   * @return new tensor
   */
  Tensor unfold(int dim, int size, int step);

  /**
   * Repeating a tensor allocates new memory, unless result is provided, in which case its memory
   * is resized. sizes specify the number of times the tensor is repeated in each dimension.
   *
   * @param sizes
   * @return
   */
  Tensor repeatTensor(int[] sizes);

  /**
   * This is equivalent to this.expand(template.size())
   *
   * @param template the given tensor
   * @return
   */
  Tensor expandAs(Tensor template);

  /**
   * Expanding a tensor allocates new memory, tensor where singleton dimensions can be expanded
   * to multiple ones by setting the stride to 0. Any dimension that has size 1 can be expanded
   * to arbitrary value with new memory allocation. Attempting to expand along a dimension that
   * does not have size 1 will result in an error.
   *
   * @param sizes the size that tensor will expend to
   * @return
   */
  Tensor expand(int[] sizes);

  /**
   * Splits current tensor along dimension dim into a result table of Tensors of size size
   * (a number) or less (in the case of the last Tensor). The sizes of the non-dim dimensions
   * remain unchanged. Internally, a series of narrows are performed along dimensions dim.
   * Argument dim defaults to 1.
   *
   * @param size
   * @param dim
   * @return
   */
  Tensor[] split(int size, int dim);

  /**
   * spilt one tensor into multi tensor along the `dim` dimension
   *
   * @param dim the specific dimension
   * @return
   */
  Tensor[] split(int dim);

  /**
   * Compare and print differences between two tensors
   *
   * @param other
   * @param count
   * @return true if there's difference, vice versa
   */
  boolean diff(Tensor other, int count, boolean reverse);

  /**
   * view this.tensor and add a Singleton Dimension to `dim` dimension
   *
   * @param t   source tensor
   * @param dim the specific dimension, default is 1
   * @return this
   */
  Tensor addSingletonDimension(Tensor t, int dim);

  /**
   * view this.tensor and add multiple Dimensions to `dim` dimension
   *
   * @param t   source tensor
   * @param dim the specific dimension array, default is [1]
   * @return this
   */
  Tensor addMultiDimension(Tensor t, int[] dim);

  /**
   * create a new tensor without any change of the tensor
   *
   * @param sizes the size of the new Tensor
   * @return
   */
  Tensor reshape(int[] sizes);

  /**
   * Save the tensor to given path
   *
   * @param path
   * @param overWrite
   * @return
   */
  Tensor save(String path, boolean overWrite);


  /**
   * Return true because it's a Tensor implemented from [[Activity]]
   *
   * @return true
   */
  default boolean isTensor() {
    return true;
  }


  /**
   * Return false because it's not a Table
   *
   * @return false
   */
  default boolean isTable() {
    return false;
  }

  /**
   * Return tensor numeric
   *
   * @return
   */
  TensorNumeric getTensorNumeric();

  /**
   * Return tensor type
   * @return Dense / Quant
   */
  TensorType getTensorType();

  /**
   * Convert 1D tensor to an array. If the tensor is not 1D, an exception will be thrown out.
   *
   * @return
   */
  double[] toArray();

  /**
   * Convert 1D tensor to an float array. If the tensor is not 1D, an exception will be thrown out.
   *
   * @return
   */
  float[] toFloatArray();

  /**
   * Get a new tensor with same value and different storage
   *
   * @return new tensor
   */
  default Tensor clone() {
    return this;
  }
}

