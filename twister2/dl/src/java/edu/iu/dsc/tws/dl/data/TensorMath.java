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

import edu.iu.dsc.tws.dl.utils.pair.TensorPair;

/**
 * It provides multiple math operation functions for manipulating Tensor objects.
 * All functions support both allocating a new Tensor to return the result
 * and treating the caller as a target Tensor, in which case the target Tensor(s)
 * will be resized accordingly and filled with the result. This property is especially
 * useful when one wants to have tight control over when memory is allocated.
 *
 * @tparam double should be double or float
 */
public interface TensorMath {
  // scalastyle:off methodName

  /**
   * Add all elements of this with value not in place.
   * It will allocate new memory.
   *
   * @param s
   * @return
   */

  Tensor addCopy(double s);

  Tensor addCopy(float s);

  /**
   * Add a Tensor to another one, return the result in new allocated memory.
   * The number of elements in the Tensors must match, but the sizes do not matter.
   * The size of the returned Tensor will be the size of the first Tensor
   *
   * @param t
   * @return
   */
  Tensor addCopy(Tensor t);

  /**
   * subtract all elements of this with the value not in place.
   * It will allocate new memory.
   *
   * @param s
   * @return
   */
  Tensor subCopy(double s);

  Tensor subCopy(float s);

  /**
   * Subtract a Tensor from another one, return the result in new allocated memory.
   * The number of elements in the Tensors must match, but the sizes do not matter.
   * The size of the returned Tensor will be the size of the first Tensor
   *
   * @param t
   * @return
   */
  Tensor subCopy(Tensor t);

  Tensor negative();

  /**
   * divide all elements of this with value not in place.
   * It will allocate new memory.
   *
   * @param s
   * @return
   */
  Tensor divCopy(double s);

  Tensor divCopy(float s);

  /**
   * Divide a Tensor by another one, return the result in new allocated memory.
   * The number of elements in the Tensors must match, but the sizes do not matter.
   * The size of the returned Tensor will be the size of the first Tensor
   *
   * @param t
   * @return
   */
  Tensor divCopy(Tensor t);

  /**
   * multiply all elements of this with value not in place.
   * It will allocate new memory.
   *
   * @param s
   * @return
   */
  Tensor mulCopy(double s);

  Tensor mulCopy(float s);

  /**
   * Multiply a Tensor by another one, return the result in new allocated memory.
   * The number of elements in the Tensors must match, but the sizes do not matter.
   * The size of the returned Tensor will be the size of the first Tensor
   *
   * @param t
   * @return
   */
  Tensor mulCopy(Tensor t);
  // scalastyle:on methodName

  /**
   * returns the sum of the elements of this
   *
   * @return
   */
  double sum();

  float sumf();

  /**
   * returns the product of the elements of this
   *
   * @return
   */
  double prod();

  float prodf();

  Tensor prod(Tensor x, int dim);

  /**
   * performs the sum operation over the dimension dim
   *
   * @param dim
   * @return
   */
  Tensor sum(int dim);

  Tensor sum(Tensor x, int dim);

  /**
   * returns the mean of all elements of this.
   *
   * @return
   */
  double mean();

  float meanf();

  /**
   * performs the mean operation over the dimension dim.
   *
   * @param dim
   * @return
   */
  Tensor mean(int dim);

  /**
   * returns the single biggest element of x
   *
   * @return
   */
  double max();

  float maxf();

  /**
   * performs the max operation over the dimension n
   *
   * @param dim
   * @return
   */
  TensorPair max(int dim);

  /**
   * performs the max operation over the dimension n
   *
   * @param values
   * @param indices
   * @param dim
   * @return
   */
  TensorPair max(Tensor values, Tensor indices, int dim);

  /**
   * returns the single minimum element of x
   *
   * @return
   */
  double min();

  float minf();

  /**
   * performs the min operation over the dimension n
   *
   * @param dim
   * @return
   */
  TensorPair min(int dim);

  /**
   * performs the min operation over the dimension n
   *
   * @param values
   * @param indices
   * @param dim
   * @return
   */
  TensorPair min(Tensor values, Tensor indices, int dim);

  /**
   * Writes all values from tensor src into this tensor at the specified indices
   *
   * @param dim
   * @param index
   * @param src
   * @return this
   */
  Tensor scatter(int dim, Tensor index, Tensor src);

  /**
   * change this tensor with values from the original tensor by gathering a number of values
   * from each "row", where the rows are along the dimension dim.
   *
   * @param dim
   * @param index
   * @param src
   * @return this
   */
  Tensor gather(int dim, Tensor index, Tensor src);

  /**
   * This function computes 2 dimensional convolution of a single image
   * with a single kernel (2D output). the dimensions of input and kernel
   * need to be 2, and Input image needs to be bigger than kernel. The
   * last argument controls if the convolution is a full ('F') or valid
   * ('V') convolution. The default is valid convolution.
   *
   * @param kernel
   * @param vf     full ('F') or valid ('V') convolution.
   * @return
   */
  Tensor conv2(Tensor kernel, char vf);

  /**
   * This function operates with same options and input/output configurations as conv2,
   * but performs cross-correlation of the input with the kernel k.
   *
   * @param kernel
   * @param vf     full ('F') or valid ('V') convolution.
   * @return
   */
  Tensor xcorr2(Tensor kernel, char vf);

  /**
   * replaces all elements in-place with the square root of the elements of this.
   *
   * @return
   */
  Tensor sqrt();

  /**
   * replaces all elements in-place with the tanh root of the elements of this.
   *
   * @return
   */
  Tensor tanh();

  /**
   * replaces all elements in-place with the absolute values of the elements of this.
   *
   * @return
   */
  Tensor abs();

  /**
   * x.add(value,y) multiply-accumulates values of y into x.
   *
   * @param value scalar
   * @param y     other tensor
   * @return current tensor
   */
  Tensor add(double value, Tensor y);

  Tensor add(float value, Tensor y);

  /**
   * accumulates all elements of y into this
   *
   * @param y other tensor
   * @return current tensor
   */
  Tensor add(Tensor y);
  // Puts the result of x + value * y in current tensor

  /**
   * z.add(x, value, y) puts the result of x + value * y in z.
   *
   * @param x
   * @param value
   * @param y
   * @return
   */
  Tensor add(Tensor x, double value, Tensor y);

  Tensor add(Tensor x, float value, Tensor y);

  /**
   * x.add(value) : add value to all elements of x in place.
   *
   * @param value
   * @return
   */
  Tensor add(double value);

  Tensor add(float value);

  Tensor add(Tensor x, Tensor y);

  /**
   * Performs the dot product. The number of elements must match: both Tensors are seen as a 1D
   * vector.
   *
   * @param y
   * @return
   */
  double dot(Tensor y);

  float dotf(Tensor y);


  /**
   * For each elements of the tensor, performs the max operation compared with the given value
   * vector.
   *
   * @param value
   * @return
   */
  Tensor cmax(double value);

  Tensor cmax(float value);

  /**
   * Performs the p-norm distance calculation between two tensors
   *
   * @param y    the secode Tensor
   * @param norm the norm of distance
   * @return
   */
  double dist(Tensor y, int norm);

  float distf(Tensor y, int norm);

  /**
   * Performs the element-wise multiplication of tensor1 by tensor2, multiply the result by the
   * scalar value (1 if not present) and add it to x. The number of elements must match, but sizes
   * do not matter.
   *
   * @param value
   * @param tensor1
   * @param tensor2
   */
  Tensor addcmul(double value, Tensor tensor1, Tensor tensor2);

  Tensor addcmul(float value, Tensor tensor1, Tensor tensor2);

  Tensor addcmul(Tensor tensor1, Tensor tensor2);

  /**
   * Performs the element-wise division of tensor1 by tensor2, multiply the result by the scalar
   * value and add it to x.
   * The number of elements must match, but sizes do not matter.
   *
   * @param value
   * @param tensor1
   * @param tensor2
   * @return
   */
  Tensor addcdiv(double value, Tensor tensor1, Tensor tensor2);

  Tensor addcdiv(float value, Tensor tensor1, Tensor tensor2);

  Tensor sub(double value, Tensor y);

  Tensor sub(float value, Tensor y);

  // Puts the result of x - value * y in current tensor
  Tensor sub(Tensor x, double value, Tensor y);

  Tensor sub(Tensor x, float value, Tensor y);

  /**
   * subtracts all elements of y from this
   *
   * @param y other tensor
   * @return current tensor
   */
  Tensor sub(Tensor y);

  Tensor sub(Tensor x, Tensor y);

  Tensor sub(double value);

  Tensor sub(float value);

  /**
   * Element-wise multiply
   * x.cmul(y) multiplies all elements of x with corresponding elements of y.
   * x = x * y
   *
   * @param y tensor
   * @return current tensor
   */
  Tensor cmul(Tensor y);

  /**
   * Element-wise multiply
   * z.cmul(x, y) equals z = x * y
   *
   * @param x tensor
   * @param y tensor
   * @return current tensor
   */
  Tensor cmul(Tensor x, Tensor y);

  /**
   * Element-wise divide
   * x.cdiv(y) all elements of x divide all elements of y.
   * x = x / y
   *
   * @param y tensor
   * @return current tensor
   */
  Tensor cdiv(Tensor y);

  /**
   * Element-wise divide
   * z.cdiv(x, y) means z = x / y
   *
   * @param x tensor
   * @param y tensor
   * @return current tensor
   */
  Tensor cdiv(Tensor x, Tensor y);

  /**
   * multiply all elements of this with value in-place.
   *
   * @param value
   * @return
   */
  Tensor mul(double value);

  Tensor mul(float value);

  /**
   * divide all elements of this with value in-place.
   *
   * @param value
   * @return
   */
  Tensor div(double value);

  Tensor div(float value);

  /**
   * Element-wise divide
   * x.div(y) all elements of x divide all elements of y.
   * x = x / y
   *
   * @param y tensor
   * @return current tensor
   */
  Tensor div(Tensor y);

  /**
   * put the result of x * value in current tensor
   *
   * @param value
   * @return
   */
  Tensor mul(Tensor x, double value);

  Tensor mul(Tensor x, float value);

  /**
   * Performs a matrix-matrix multiplication between mat1 (2D tensor) and mat2 (2D tensor).
   * Optional values v1 and v2 are scalars that multiply m and mat1 * mat2 respectively.
   * Optional value beta is a scalar that scales the result tensor, before accumulating the result
   * into the tensor. Defaults to 1.0.
   * If mat1 is a n x m matrix, mat2 a m x p matrix, m must be a n x p matrix.
   * <p>
   * res = (v1 * m) + (v2 * mat1*mat2)
   *
   * @param v1
   * @param m
   * @param v2
   * @param mat1
   * @param mat2
   */
  Tensor addmm(double v1, Tensor m, double v2, Tensor mat1, Tensor mat2);

  Tensor addmm(float v1, Tensor m, float v2, Tensor mat1, Tensor mat2);

  /**
   * res = m + (mat1*mat2)
   */
  Tensor addmm(Tensor m, Tensor mat1, Tensor mat2);

  /**
   * res = res + mat1 * mat2
   */
  Tensor addmm(Tensor mat1, Tensor mat2);

  /**
   * res = res + v2 * mat1 * mat2
   */
  Tensor addmm(double v2, Tensor mat1, Tensor mat2);

  Tensor addmm(float v2, Tensor mat1, Tensor mat2);

  /**
   * res = v1 * res + v2 * mat1*mat2
   */
  Tensor addmm(double v1, double v2, Tensor mat1, Tensor mat2);

  Tensor addmm(float v1, float v2, Tensor mat1, Tensor mat2);

  /**
   * res = mat1*mat2
   */
  Tensor mm(Tensor mat1, Tensor mat2);

  /**
   * Performs the outer-product between vec1 (1D tensor) and vec2 (1D tensor).
   * Optional values v1 and v2 are scalars that multiply mat and vec1 [out] vec2 respectively.
   * In other words,
   * res_ij = (v1 * mat_ij) + (v2 * vec1_i * vec2_j)
   *
   * @param t1
   * @param t2
   * @return
   */
  Tensor addr(Tensor t1, Tensor t2);

  Tensor addr(double v1, Tensor t1, Tensor t2);

  Tensor addr(float v1, Tensor t1, Tensor t2);

  Tensor addr(double v1, Tensor t1, double v2, Tensor t2);

  Tensor addr(float v1, Tensor t1, float v2, Tensor t2);

  /**
   * Performs the outer-product between vec1 (1D Tensor) and vec2 (1D Tensor).
   * Optional values v1 and v2 are scalars that multiply mat and vec1 [out] vec2 respectively.
   * In other words,res_ij = (v1 * mat_ij) + (v2 * vec1_i * vec2_j)
   *
   * @param v1
   * @param t1
   * @param v2
   * @param t2
   * @param t3
   * @return
   */
  Tensor addr(double v1, Tensor t1, double v2, Tensor t2, Tensor t3);

  Tensor addr(float v1, Tensor t1, float v2, Tensor t2, Tensor t3);

  /**
   * return pseudo-random numbers, require 0<=args.length<=2
   * if args.length = 0, return [0, 1)
   * if args.length = 1, return [1, args(0)] or [args(0), 1]
   * if args.length = 2, return [args(0), args(1)]
   *
   * @param args
   */
  double uniform(double... args);

  float uniform(float... args);

  /**
   * Performs a matrix-vector multiplication between mat (2D Tensor) and vec2 (1D Tensor) and add
   * it to vec1. Optional values v1 and v2 are scalars that multiply vec1 and vec2 respectively.
   * <p>
   * In other words,
   * res = (beta * vec1) + alpha * (mat * vec2)
   * <p>
   * Sizes must respect the matrix-multiplication operation: if mat is a n × m matrix,
   * vec2 must be vector of size m and vec1 must be a vector of size n.
   */
  Tensor addmv(double beta, Tensor vec1, double alpha, Tensor mat, Tensor vec2);

  Tensor addmv(float beta, Tensor vec1, float alpha, Tensor mat, Tensor vec2);

  /**
   * res = beta * res + alpha * (mat * vec2)
   */
  Tensor addmv(double beta, double alpha, Tensor mat, Tensor vec2);

  Tensor addmv(float beta, float alpha, Tensor mat, Tensor vec2);

  /**
   * res = res + alpha * (mat * vec2)
   */
  Tensor addmv(double alpha, Tensor mat, Tensor vec2);

  Tensor addmv(float alpha, Tensor mat, Tensor vec2);

  /**
   * res = res + (mat * vec2)
   */
  Tensor mv(Tensor mat, Tensor vec2);

  /**
   * Perform a batch matrix matrix multiplication of matrices and stored in batch1 and batch2
   * with batch add. batch1 and batch2 must be 3D Tensors each containing the same number of
   * matrices. If batch1 is a b × n × m Tensor, batch2 a b × m × p Tensor, res will be a
   * b × n × p Tensor.
   * <p>
   * In other words,
   * res_i = (beta * M_i) + (alpha * batch1_i * batch2_i)
   */
  Tensor baddbmm(double beta, Tensor m, double alpha, Tensor batch1, Tensor batch2);

  Tensor baddbmm(float beta, Tensor m, float alpha, Tensor batch1, Tensor batch2);

  /**
   * res_i = (beta * res_i) + (alpha * batch1_i * batch2_i)
   */
  Tensor baddbmm(double beta, double alpha, Tensor batch1, Tensor batch2);

  Tensor baddbmm(float beta, float alpha, Tensor batch1, Tensor batch2);

  /**
   * res_i = res_i + (alpha * batch1_i * batch2_i)
   */
  Tensor baddbmm(double alpha, Tensor batch1, Tensor batch2);

  Tensor baddbmm(float alpha, Tensor batch1, Tensor batch2);

  /**
   * res_i = res_i + batch1_i * batch2_i
   */
  Tensor bmm(Tensor batch1, Tensor batch2);

  /**
   * Replaces all elements in-place with the elements of x to the power of n
   *
   * @param y
   * @param n
   * @return current tensor reference
   */
  Tensor pow(Tensor y, double n);

  Tensor pow(Tensor y, float n);

  Tensor pow(double n);

  Tensor pow(float n);

  /**
   * Replaces all elements in-place with the elements of x squared
   *
   * @return current tensor reference
   */
  Tensor square();

  /**
   * Populate the given tensor with the floor result of elements
   *
   * @param y
   * @return
   */
  Tensor floor(Tensor y);

  /**
   * Replaces all elements in-place with the floor result of elements
   *
   * @return
   */
  Tensor floor();

  /**
   * Replaces all elements in-place with the ceil result of elements
   *
   * @return
   */
  Tensor ceil();

  /**
   * Computes the reciprocal of this tensor element-wise and update the content inplace
   *
   * @return
   */
  Tensor inv();

  /**
   * Computes the reciprocal of this tensor element-wise and update the content inplace
   *
   * @return
   */
  Tensor erf();

  /**
   * Computes the reciprocal of this tensor element-wise and update the content inplace
   *
   * @return
   */
  Tensor erfc();

  /**
   * Computes the log of the absolute value of `Gamma(x)` element-wise,
   * and update the content inplace
   *
   * @return
   */
  Tensor logGamma();

  /**
   * Computes Psi, the derivative of Lgamma (the log of the absolute value of
   * `Gamma(x)`), element-wise and update the content inplace
   *
   * @return
   */
  Tensor digamma();

  /**
   * Get the top k smallest values and their indices.
   *
   * @param result       result buffer
   * @param indices      indices buffer
   * @param k
   * @param dim          dimension, default is the last dimension (default -1)
   * @param increase     sort order, set it to true if you want to get the smallest
   *                     top k values (default true)
   * @param sortedResult default value true
   * @return
   */
  TensorPair topk(int k, int dim, boolean increase, Tensor result,
                  Tensor indices, boolean sortedResult);

  /**
   * Replaces all elements in-place with the elements of lnx
   *
   * @param y
   * @return current tensor reference
   */
  Tensor log(Tensor y);

  Tensor exp(Tensor y);

  Tensor sqrt(Tensor y);

  Tensor tanh(Tensor y);

  Tensor log1p(Tensor y);

  Tensor log();

  Tensor exp();

  Tensor log1p();

  Tensor abs(Tensor x);

  /**
   * returns the p-norms of the Tensor x computed over the dimension dim.
   *
   * @param y     result buffer
   * @param value
   * @param dim
   * @return
   */
  Tensor norm(Tensor y, int value, int dim);

  /**
   * Implements > operator comparing each element in x with y
   *
   * @param x
   * @param y
   * @return current tensor reference
   */
  Tensor gt(Tensor x, Tensor y);

  /**
   * Implements < operator comparing each element in x with y
   *
   * @param x
   * @param y
   * @return current tensor reference
   */
  Tensor lt(Tensor x, Tensor y);

  /**
   * Implements <= operator comparing each element in x with y
   *
   * @param x
   * @param y
   * @return current tensor reference
   */
  Tensor le(Tensor x, Tensor y);

  /**
   * Implements == operator comparing each element in x with y
   *
   * @param y
   * @return current tensor reference
   */
  Tensor eq(Tensor x, double y);

  Tensor eq(Tensor x, float y);

  /**
   * Fills the masked elements of itself with value val
   *
   * @param mask
   * @param e
   * @return current tensor reference
   */
  Tensor maskedFill(Tensor mask, double e);

  Tensor maskedFill(Tensor mask, float e);

  /**
   * Copies the elements of tensor into mask locations of itself.
   *
   * @param mask
   * @param y
   * @return current tensor reference
   */
  Tensor maskedCopy(Tensor mask, Tensor y);

  /**
   * Returns a new Tensor which contains all elements aligned to a 1 in the corresponding mask.
   *
   * @param mask
   * @param y
   * @return current tensor reference
   */
  Tensor maskedSelect(Tensor mask, Tensor y);

  /**
   * returns the sum of the n-norms on the Tensor x
   *
   * @param value the n-norms
   * @return
   */
  double norm(int value);

  float normf(int value);

  /**
   * returns a new Tensor with the sign (+/- 1 or 0) of the elements of x.
   *
   * @return
   */
  Tensor sign();

  /**
   * Implements >= operator comparing each element in x with value
   *
   * @param x
   * @param value
   * @return
   */
  Tensor ge(Tensor x, double value);

  Tensor ge(Tensor x, float value);

  /**
   * Accumulate the elements of tensor into the original tensor by adding to the indices
   * in the order given in index. The shape of tensor must exactly match the elements indexed
   * or an error will be thrown.
   *
   * @param dim
   * @param index
   * @param y
   * @return
   */
  Tensor indexAdd(int dim, Tensor index, Tensor y);

  /**
   * Accumulate the elements of tensor into the original tensor by adding to the indices
   * in the order given in index. The shape of tensor must exactly match the elements indexed
   * or an error will be thrown.
   *
   * @param dim
   * @param index
   * @param y
   * @return
   */
  Tensor index(int dim, Tensor index, Tensor y);

  /**
   * stores the element-wise maximum of x and y in x.
   * x.cmax(y) = max(x, y)
   *
   * @param y tensor
   * @return current tensor
   */
  Tensor cmax(Tensor y);

  /**
   * stores the element-wise maximum of x and y in x.
   * x.cmin(y) = min(x, y)
   *
   * @param y tensor
   * @return current tensor
   */
  Tensor cmin(Tensor y);

  /**
   * stores the element-wise maximum of x and y in z.
   * z.cmax(x, y) means z = max(x, y)
   *
   * @param x tensor
   * @param y tensor
   */
  Tensor cmax(Tensor x, Tensor y);

  /**
   * stores the element-wise maximum of x and y in z.
   * z.cmin(x, y) means z = min(x, y)
   *
   * @param x tensor
   * @param y tensor
   */
  Tensor cmin(Tensor x, Tensor y);

  /**
   * resize this tensor size to floor((xmax - xmin) / step) + 1 and set values from
   * xmin to xmax with step (default to 1).
   *
   * @param xmin
   * @param xmax
   * @param step default 1
   * @return this tensor
   */
  Tensor range(double xmin, double xmax, int step);

  Tensor range(float xmin, float xmax, int step);

  /**
   * Computes numerical negative value element-wise. y = -x
   *
   * @param x
   * @return this tensor
   */
  Tensor negative(Tensor x);

  //TODO: implement for reduce
// /**
//  * Reduce along the given dimension with the given reducer, and copy the result to the result
//  * tensor
//  * @param dim
//  * @param result
//  * @param reducer
//  */
// Tensor reduce(int dim,Tensor result, reducer: (double, double) => double);
  double sumSquare();

  float sumSquaref();

  Tensor clamp(double min, double max);

  Tensor clamp(float min, float max);
}

