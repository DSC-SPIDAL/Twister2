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

import edu.iu.dsc.tws.dl.utils.Util;

@SuppressWarnings("LocalVariableName")
public final class DenseTensorBLAS {

  private static long time = 0L;

  private DenseTensorBLAS() {
  }

  public static long getTime() {
    return time;
  }

  public static void setTime(long time) {
    DenseTensorBLAS.time = time;
  }

  /**
   * The gemm routines compute a scalar-matrix-matrix product and
   * add the result to a scalar-matrix product, with general matrices.
   * C := alpha*op(A)*op(B) + beta*C,
   * where:
   * op(X) is one of op(X) = X, or op(X) = XT,
   * alpha and beta are scalars,
   * A, B and C are matrices:
   * op(A) is an m-by-k matrix,
   * op(B) is a k-by-n matrix,
   * C is an m-by-n matrix.
   * <p>
   * this interface treat the input array as column-major array.
   *
   * @param transa  Specifies the form of op(A) used in the matrix multiplication:
   *                if transa=CblasNoTrans, then op(A) = A;
   *                if transa=CblasTrans, then op(A) = AT;
   * @param transb  Specifies the form of op(B) used in the matrix multiplication:
   *                if transb=CblasNoTrans, then op(B) = B;
   *                if transb=CblasTrans, then op(B) = BT;
   * @param m       Specifies the number of rows of the matrix op(A) and of the matrix C.
   *                The value of m must be at least zero.
   * @param n       Specifies the number of columns of the matrix op(B) and the number of
   *                columns of the matrix C. The value of n must be at least zero.
   * @param k       Specifies the number of columns of the matrix op(A) and the number of
   *                rows of the matrix op(B). The value of k must be at least zero.
   * @param alpha   Specifies the scalar alpha.
   * @param a       Array. if transa=CblasNoTrans, size lda*k. if transa=CblasTrans, size lda*m.
   * @param aOffset a offset
   * @param lda     Specifies the leading dimension of a as declared in the calling (sub)program.
   *                if transa=CblasNoTrans, lda must be at least max(1, m).
   *                if transa=CblasTrans, lda must be at least max(1, k).
   * @param b       Array. if transb=CblasNoTrans, size ldb by n. if transb=CblasTrans, size ldb by k.
   * @param bOffset b offset
   * @param ldb     Specifies the leading dimension of b as declared in the calling (sub)program.
   *                if transb=CblasNoTrans, ldb must be at least max(1, m).
   *                if transb=CblasTrans, ldb must be at least max(1, k).
   * @param beta    Specifies the scalar beta.
   *                When beta is equal to zero, then c need not be set on input.
   * @param c       Array, size ldc by n. Before entry, the leading m-by-n part of the array c must
   *                contain the matrix C, except when beta is equal to zero,
   *                in which case c need not be set on entry.
   * @param cOffset c offset
   * @param ldc     ldc must be at least max(1, m).
   * @tparam T
   */
  public static void gemm(char transa, char transb, int m, int n, int k, double alpha,
                          double[] a, int aOffset, int lda, double[] b, int bOffset, int ldb,
                          double beta, double[] c, int cOffset, int ldc) {

    boolean _transa = transa == 't' || transa == 'T';
    boolean _transb = transb == 't' || transb == 'T';

    int _ldc = ldc;
    if (n == 1) {
      _ldc = m;
    }

    int _lda = lda;
    if (_transa) {
      if (m == 1) {
        _lda = k;
      }
    } else {
      if (k == 1) {
        _lda = m;
      }
    }

    int _ldb = ldb;
    if (_transb) {
      if (k == 1) {
        _ldb = n;
      }
    } else {
      if (n == 1) {
        _ldb = k;
      }
    }

    long start = System.nanoTime();
    TensorNumeric.gemm(transa, transb, m, n, k, alpha, a, aOffset, _lda, b, bOffset, _ldb, beta, c,
        cOffset, _ldc);
    time += System.nanoTime() - start;
  }

  public static void gemm(char transa, char transb, int m, int n, int k, float alpha,
                          float[] a, int aOffset, int lda, float[] b, int bOffset, int ldb,
                          float beta, float[] c, int cOffset, int ldc) {

    boolean _transa = transa == 't' || transa == 'T';
    boolean _transb = transb == 't' || transb == 'T';

    int _ldc = ldc;
    if (n == 1) {
      _ldc = m;
    }

    int _lda = lda;
    if (_transa) {
      if (m == 1) {
        _lda = k;
      }
    } else {
      if (k == 1) {
        _lda = m;
      }
    }

    int _ldb = ldb;
    if (_transb) {
      if (k == 1) {
        _ldb = n;
      }
    } else {
      if (n == 1) {
        _ldb = k;
      }
    }

    long start = System.nanoTime();
    TensorNumeric.gemm(transa, transb, m, n, k, alpha, a, aOffset, _lda, b, bOffset, _ldb, beta, c,
        cOffset, _ldc);
    time += System.nanoTime() - start;
  }

  /**
   * to be fixed: this interface treat the input tensor as row-major array.
   *
   * @param alpha
   * @param matrix
   * @param vector
   * @param beta
   * @param r
   * @tparam T
   */
  public static void gemv(double alpha, Tensor matrix, Tensor vector, double beta, Tensor r) {
    Util.require(matrix.size(2) == vector.size(1), "matrix vector size doesn't match");
    Util.require(matrix.size(1) == r.size(1), "matrix result size doesn't match");
    if (matrix.stride(1) == 1) {
      TensorNumeric.gemv('N', matrix.size(1), matrix.size(2), alpha,
          matrix.storage().toDoubleArray(), matrix.storageOffset() - 1, matrix.stride(2),
          vector.storage().toDoubleArray(), vector.storageOffset() - 1, vector.stride(1),
          beta, r.storage().toDoubleArray(),
          r.storageOffset() - 1, r.stride(1));
    } else if (matrix.stride(2) == 1) {
      TensorNumeric.gemv('T', matrix.size(2), matrix.size(1), alpha,
          matrix.storage().toDoubleArray(), matrix.storageOffset() - 1, matrix.stride(1),
          vector.storage().toDoubleArray(), vector.storageOffset() - 1,
          vector.stride(1), beta, r.storage().toDoubleArray(),
          r.storageOffset() - 1, r.stride(1));
    } else {
      Tensor mat = matrix.contiguous();
      TensorNumeric.gemv('T', mat.size(2), mat.size(1), alpha,
          mat.storage().toDoubleArray(), mat.storageOffset() - 1, mat.stride(1),
          vector.storage().toDoubleArray(), vector.storageOffset() - 1, vector.stride(1),
          beta, r.storage().toDoubleArray(), r.storageOffset() - 1, r.stride(1));
    }
  }

  public static void gemv(float alpha, Tensor matrix, Tensor vector, float beta, Tensor r) {
    Util.require(matrix.size(2) == vector.size(1), "matrix vector size doesn't match");
    Util.require(matrix.size(1) == r.size(1), "matrix result size doesn't match");
    if (matrix.stride(1) == 1) {
      TensorNumeric.gemv('N', matrix.size(1), matrix.size(2), alpha,
          matrix.storage().toFloatArray(), matrix.storageOffset() - 1, matrix.stride(2),
          vector.storage().toFloatArray(), vector.storageOffset() - 1, vector.stride(1),
          beta, r.storage().toFloatArray(),
          r.storageOffset() - 1, r.stride(1));
    } else if (matrix.stride(2) == 1) {
      TensorNumeric.gemv('T', matrix.size(2), matrix.size(1), alpha,
          matrix.storage().toFloatArray(), matrix.storageOffset() - 1, matrix.stride(1),
          vector.storage().toFloatArray(), vector.storageOffset() - 1,
          vector.stride(1), beta, r.storage().toFloatArray(),
          r.storageOffset() - 1, r.stride(1));
    } else {
      Tensor mat = matrix.contiguous();
      TensorNumeric.gemv('T', mat.size(2), mat.size(1), alpha,
          mat.storage().toFloatArray(), mat.storageOffset() - 1, mat.stride(1),
          vector.storage().toFloatArray(), vector.storageOffset() - 1, vector.stride(1),
          beta, r.storage().toFloatArray(), r.storageOffset() - 1, r.stride(1));
    }
  }
}
