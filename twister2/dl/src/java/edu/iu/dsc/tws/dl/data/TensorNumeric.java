//  Licensed under the Apache License, Version 2.0 (the "License"){};
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
import java.util.Arrays;

import com.intel.analytics.bigdl.mkl.MKL;

import edu.iu.dsc.tws.dl.utils.RandomGenerator;

public final class TensorNumeric implements Serializable {

  private TensorNumeric() {
  }

  //T one: T = fromType<int>(1)

  //T zero: T = fromType[Int](0)

  //  T fromType[K](k: K)(implicit c: ConvertableFrom[K]){};
  //  T toType[K](t: T)(implicit c: ConvertableTo[K]){}; K
  // TensorDataType getType(){};


  //double opertions
  public static double plus(double x, double y) {
    return x + y;
  }

  public static double minus(double x, double y) {
    return x - y;
  }

  public static double times(double x, double y) {
    return x * y;
  }

  public static double divide(double x, double y) {
    return x / y;
  }

  public static double exp(double x) {
    return Math.exp(x);
  }

  public static double log(double x) {
    return Math.log(x);
  }

  public static double max(double x, double y) {
    return Math.max(x, y);
  }

  public static double min(double x, double y) {
    return Math.min(x, y);
  }

  public static double sqrt(double x) {
    return Math.sqrt(x);
  }

  public static double tanh(double x) {
    return Math.tanh(x);
  }

  public static double abs(double x) {
    return Math.abs(x);
  }

  public static double or(double x, double y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  public static double and(double x, double y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  public static double negative(double x) {
    return -x;
  }

  public static double pow(double x) {
    return Math.pow(x, -1);
  }

  public static double pow(double x, double y) {
    return Math.pow(x, y);
  }

  public static double log1p(double x) {
    return Math.log1p(x);
  }

  public static boolean isGreater(double x, double y) {
    return x > y;
  }

  public static boolean isGreaterEq(double x, double y) {
    return x >= y;
  }

  public static double rand() {
    return RandomGenerator.RNG().uniform(0, 1);
  }

  public static double randn() {
    return RandomGenerator.RNG().normal(0, 1);
  }

  public static void gemm(char transa, char transb, int m, int n, int k, double alpha, double[] a,
                          int aOffset, int lda, double[] b, int bOffset, int ldb,
                          double beta, double[] c, int cOffset, int ldc) {
    checkMKL();
    MKL.vdgemm(transa, transb, m, n, k, alpha, a, aOffset, lda, b,
        bOffset, ldb, beta, c, cOffset, ldc);
  }

  public static void gemv(char trans, int m, int n, double alpha, double[] a, int aoffset, int lda,
                          double[] x, int xOffset, int incx, double beta, double[] y, int yOffset,
                          int incy) {
    checkMKL();

    MKL.vdgemv(trans, m, n, alpha, a, aoffset, lda, x, xOffset,
        incx, beta, y, yOffset, incy);
  }

  public static void axpy(int n, double da, double[] dx, int dxOffset, int incx, double[] dy,
                          int dyOffset, int incy) {
    checkMKL();

    MKL.vdaxpy(n, da, dx, dxOffset, incx, dy, dyOffset, incy);
  }

  public static double dot(int n, double[] dx, int dxOffset, int incx, double[] dy, int dyOffset,
                           int incy) {
    checkMKL();
    return MKL.vddot(n, dx, dxOffset, incx, dy, dyOffset, incy);
  }

  public static void ger(int m, int n, double alpha, double[] x, int xOffset, int incx, double[] y,
                         int yOffset, int incy, double[] a, int aOffset, int lda) {
    checkMKL();
    MKL.vdger(m, n, alpha, x, xOffset, incx, y, yOffset,
        incy, a, aOffset, lda);
  }

  public static void fill(double[] data, int fromIndex, int toIndex, double value) {
    Arrays.fill(data, fromIndex, toIndex, value);
  }

  public static void vPowx(int n, double[] a, int aOffset, double b, double[] y, int yOffset) {
    checkMKL();
    MKL.vdPowx(n, a, aOffset, b, y, yOffset);
  }

  public static void vLn(int n, double[] a, int aOffset, double[] y, int yOffset) {

    checkMKL();
    MKL.vdLn(n, a, aOffset, y, yOffset);

  }

  public static void vExp(int n, double[] a, int aOffset, double[] y, int yOffset) {
    checkMKL();
    MKL.vdExp(n, a, aOffset, y, yOffset);

  }

  public static void vSqrt(int n, double[] a, int aOffset, double[] y, int yOffset) {
    checkMKL();
    MKL.vdSqrt(n, a, aOffset, y, yOffset);

  }

  public static void vTanh(int n, double[] a, int aOffset, double[] y, int yOffset) {
    checkMKL();
    MKL.vdTanh(n, a, aOffset, y, yOffset);

  }

  public static void vAbs(int n, double[] a, int aOffset, double[] y, int yOffset) {
    checkMKL();
    MKL.vdAbs(n, a, aOffset, y, yOffset);

  }

  public static void vLog1p(int n, double[] a, int aOffset, double[] y, int yOffset) {
    checkMKL();
    MKL.vdLog1p(n, a, aOffset, y, yOffset);

  }

  public static void scal(int n, double sa, double[] sx, int offset, int incx) {
    checkMKL();
    MKL.vdscal(n, sa, sx, offset, incx);

  }

  public static double inv(double v) {
    return 1 / v;
  }

  public static double erf(double v) {
    return org.apache.commons.math3.special.Erf.erf(v);
  }

  public static double erfc(double v) {
    return org.apache.commons.math3.special.Erf.erfc(v);
  }

  public static double logGamma(double v) {
    return org.apache.commons.math3.special.Gamma.logGamma(v);
  }

  public static double digamma(double v) {
    return org.apache.commons.math3.special.Gamma.digamma(v);
  }

  public static void add(int n, double[] a, int offset, double v, int stride) {
    int i = 0;
    while (i < n) {
      a[offset + i * stride] += v;
      i += 1;
    }
  }

  public static void sub(int n, double[] a, int offset, double v, int stride) {
    int i = 0;
    while (i < n) {
      a[offset + i * stride] -= v;
      i += 1;
    }
  }

  public static void vAdd(int n, double[] a, int aOffset, double[] b, int bOffset, double[] y,
                          int yOffset) {
    MKL.vdAdd(n, a, aOffset, b, bOffset, y, yOffset);

  }

  public static void vSub(int n, double[] a, int aOffset, double[] b, int bOffset, double[] y,
                          int yOffset) {
    MKL.vdSub(n, a, aOffset, b, bOffset, y, yOffset);

  }

  public static void vMul(int n, double[] a, int aOffset, double[] b, int bOffset, double[] y,
                          int yOffset) {
    if (checkMKL()) {
      MKL.vdMul(n, a, aOffset, b, bOffset, y, yOffset);
    } else {
      int i = 0;
      while (i < n) {
        y[yOffset + i] = a[aOffset + i] * b[bOffset + i];
        i += 1;
      }
    }
  }

  public static void vDiv(int n, double[] a, int aOffset, double[] b, int bOffset, double[] y,
                          int yOffset) {
    if (checkMKL()) {
      MKL.vdDiv(n, a, aOffset, b, bOffset, y, yOffset);
    } else {
      int i = 0;
      while (i < n) {
        y[yOffset + i] = a[aOffset + i] / b[bOffset + i];
        i += 1;
      }
    }
  }

  public static double sum(int n, double[] a, int aOffset, int stride) {
    int i = 0;
    double r = 0.0;
    while (i < n) {
      r += a[aOffset + i * stride];
      i += 1;
    }
    return r;
  }

  public static double prod(int n, double[] a, int aOffset, int stride) {
    int i = 0;
    double r = 1.0;
    while (i < n) {
      r *= a[aOffset + i * stride];
      i += 1;
    }
    return r;
  }

  public static void arraycopy(double[] src, int srcPos, double[] dest, int destPos, int length) {
    System.arraycopy(src, srcPos, dest, destPos, length);
  }

  public static void addcmul(double value, int n,
                             double[] self, int selfOffset,
                             double[] a, int aOffset,
                             double[] b, int bOffset) {
    double v = value;
    int i = 0;
    while (i < n) {
      self[i + selfOffset] += a[aOffset + i] * b[bOffset + i] * v;
      i += 1;
    }
  }

  public static void addcdiv(double value, int n,
                             double[] self, int selfOffset,
                             double[] a, int aOffset,
                             double[] b, int bOffset) {
    double v = value;
    int i = 0;

    while (i < n) {
      self[i + selfOffset] += a[aOffset + i] / b[bOffset + i] * v;
      i += 1;
    }
  }

  public static boolean nearlyEqual(double a, double b, double epsilon) {
    double absA = Math.abs(a);
    double absB = Math.abs(b);
    double diff = Math.abs(a - b);

    boolean result = false;
    if (a == b) {
      result = true;
    } else if (a == 0 || b == 0 || diff < Double.MIN_NORMAL) {
      result = diff < (epsilon * Double.MIN_NORMAL);
    } else {
      result = diff / (absA + absB) < epsilon;
    }
    return result;
  }

  public static double floor(double a) {
    return Math.floor(a);
  }

  public static double ceil(double a) {
    return Math.ceil(a);
  }

  public static boolean isFinite(double a) {
    return Double.isFinite(a);
  }

  public static boolean isNan(double a) {
    return Double.isNaN(a);
  }

  public static boolean isInf(double a) {
    return Double.isInfinite(a);
  }

  public static double round(double a) {
    return Math.round(a);
  }

  public static double truncate(double a) {
    if (a >= 0) {
      return Math.floor(a);
    } else if (a == Math.floor(a)) {
      return a;
    } else {
      return Math.floor(a) + 1;
    }
  }

  public static double floorDiv(double a, double b) {
    return Math.floor(a / b);
  }

  public static double clip(double a, double lower, double upper) {
    if (lower <= upper) {
      throw new IllegalStateException("lower bound must be less or equal than upper bound");
    }
    return Math.min(Math.max(a, lower), upper);
  }

  /**
   * returns the product of each element in the array
   *
   * @return product result
   */
  public static double product(double[] data) {
    double result = 1.0;
    for (double datum : data) {
      result = result * datum;
    }
    return result;
  }

  public static int product(int[] data) {
    int result = 1;
    for (int datum : data) {
      result = result * datum;
    }
    return result;
  }

  // Float operations
  public static float plus(float x, float y) {
    return x + y;
  }

  public static float minus(float x, float y) {
    return x - y;
  }

  public static float times(float x, float y) {
    return x * y;
  }

  public static float divide(float x, float y) {
    return x / y;
  }

  public static float exp(float x) {
    return (float) Math.exp(x);
  }

  public static float log(float x) {
    return (float) Math.log(x);
  }

  public static float max(float x, float y) {
    return Math.max(x, y);
  }

  public static float min(float x, float y) {
    return Math.min(x, y);
  }

  public static float sqrt(float x) {
    return (float) Math.sqrt(x);
  }

  public static float tanh(float x) {
    return (float) Math.tanh(x);
  }

  public static float abs(float x) {
    return Math.abs(x);
  }

  public static float or(float x, float y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  public static float and(float x, float y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  public static float negative(float x) {
    return -x;
  }

  public static float pow(float x) {
    return (float) Math.pow(x, -1);
  }

  public static float pow(float x, float y) {
    return (float) Math.pow(x, y);
  }

  public static float log1p(float x) {
    return (float) Math.log1p(x);
  }

  public static boolean isGreater(float x, float y) {
    return x > y;
  }

  public static boolean isGreaterEq(float x, float y) {
    return x >= y;
  }

  public static void gemm(char transa, char transb, int m, int n, int k, float alpha, float[] a,
                          int aOffset, int lda, float[] b, int bOffset, int ldb,
                          float beta, float[] c, int cOffset, int ldc) {
    checkMKL();
    //System.out.println("###### gemm: " + transa + " : " + transb + " : " + m + " : " + n
     //   + k + " : " + alpha + " : " + a.length + " : " + aOffset + " : " + lda + " : " + b.length
      // + bOffset + " : " + ldb + " : " + beta + " : " + c.length + " : " + cOffset + " : " + ldc);
    //System.out.println("a : " + sum(a) + " b : " + sum(b) + " c : " + sum(c));
    //long time = System.nanoTime();
    MKL.vsgemm(transa, transb, m, n, k, alpha, a, aOffset, lda, b,
        bOffset, ldb, beta, c, cOffset, ldc);
    //System.out.println("vsgemm Time : " + (System.nanoTime() - time) / 1e6);
  }

  public static float sum(float[] data) {
    float result = 0L;
    for (float datum : data) {
      result += datum;
    }
    return result;
  }

  public static void gemv(char trans, int m, int n, float alpha, float[] a, int aoffset, int lda,
                          float[] x, int xOffset, int incx, float beta, float[] y, int yOffset,
                          int incy) {
    checkMKL();

    MKL.vsgemv(trans, m, n, alpha, a, aoffset, lda, x, xOffset,
        incx, beta, y, yOffset, incy);
  }

  public static void axpy(int n, float da, float[] dx, int dxOffset, int incx, float[] dy,
                          int dyOffset, int incy) {
    checkMKL();

    MKL.vsaxpy(n, da, dx, dxOffset, incx, dy, dyOffset, incy);
  }

  public static float dot(int n, float[] dx, int dxOffset, int incx, float[] dy, int dyOffset,
                         int incy) {
    checkMKL();
    return MKL.vsdot(n, dx, dxOffset, incx, dy, dyOffset, incy);
  }

  public static void ger(int m, int n, float alpha, float[] x, int xOffset, int incx, float[] y,
                         int yOffset, int incy, float[] a, int aOffset, int lda) {
    checkMKL();
    MKL.vsger(m, n, alpha, x, xOffset, incx, y, yOffset,
        incy, a, aOffset, lda);
  }

  public static void fill(float[] data, int fromIndex, int toIndex, float value) {
    Arrays.fill(data, fromIndex, toIndex, value);
  }

  public static void vPowx(int n, float[] a, int aOffset, float b, float[] y, int yOffset) {
    checkMKL();
    MKL.vsPowx(n, a, aOffset, b, y, yOffset);
  }

  public static void vLn(int n, float[] a, int aOffset, float[] y, int yOffset) {

    checkMKL();
    MKL.vsLn(n, a, aOffset, y, yOffset);

  }

  public static void vExp(int n, float[] a, int aOffset, float[] y, int yOffset) {
    checkMKL();
    MKL.vsExp(n, a, aOffset, y, yOffset);
  }

  public static void vSqrt(int n, float[] a, int aOffset, float[] y, int yOffset) {
    checkMKL();
    MKL.vsSqrt(n, a, aOffset, y, yOffset);
  }

  public static void vTanh(int n, float[] a, int aOffset, float[] y, int yOffset) {
    checkMKL();
    MKL.vsTanh(n, a, aOffset, y, yOffset);

  }

  public static void vAbs(int n, float[] a, int aOffset, float[] y, int yOffset) {
    checkMKL();
    MKL.vsAbs(n, a, aOffset, y, yOffset);

  }

  public static void vLog1p(int n, float[] a, int aOffset, float[] y, int yOffset) {
    checkMKL();
    MKL.vsLog1p(n, a, aOffset, y, yOffset);

  }

  public static void scal(int n, float sa, float[] sx, int offset, int incx) {
    checkMKL();
    MKL.vsscal(n, sa, sx, offset, incx);

  }

  public static float inv(float v) {
    return 1 / v;
  }

  public static float erf(float v) {
    return (float) org.apache.commons.math3.special.Erf.erf(v);
  }

  public static float erfc(float v) {
    return (float) org.apache.commons.math3.special.Erf.erfc(v);
  }

  public static float logGamma(float v) {
    return (float) org.apache.commons.math3.special.Gamma.logGamma(v);
  }

  public static float digamma(float v) {
    return (float) org.apache.commons.math3.special.Gamma.digamma(v);
  }

  public static void add(int n, float[] a, int offset, float v, int stride) {
    int i = 0;
    while (i < n) {
      a[offset + i * stride] += v;
      i += 1;
    }
  }

  public static void sub(int n, float[] a, int offset, float v, int stride) {
    int i = 0;
    while (i < n) {
      a[offset + i * stride] -= v;
      i += 1;
    }
  }

  public static void vAdd(int n, float[] a, int aOffset, float[] b, int bOffset,
                          float[] y, int yOffset) {
    MKL.vsAdd(n, a, aOffset, b, bOffset, y, yOffset);

  }

  public static void vSub(int n, float[] a, int aOffset, float[] b, int bOffset, float[] y,
                          int yOffset) {
    MKL.vsSub(n, a, aOffset, b, bOffset, y, yOffset);

  }

  public static void vMul(int n, float[] a, int aOffset, float[] b, int bOffset, float[] y,
                          int yOffset) {
    if (checkMKL()) {
      MKL.vsMul(n, a, aOffset, b, bOffset, y, yOffset);
    } else {
      int i = 0;
      while (i < n) {
        y[yOffset + i] = a[aOffset + i] * b[bOffset + i];
        i += 1;
      }
    }
  }

  public static void vDiv(int n, float[] a, int aOffset, float[] b, int bOffset, float[] y,
                          int yOffset) {
    if (checkMKL()) {
      MKL.vsDiv(n, a, aOffset, b, bOffset, y, yOffset);
    } else {
      int i = 0;
      while (i < n) {
        y[yOffset + i] = a[aOffset + i] / b[bOffset + i];
        i += 1;
      }
    }
  }

  public static float sum(int n, float[] a, int aOffset, int stride) {
    int i = 0;
    float r = 0.0f;
    while (i < n) {
      r += a[aOffset + i * stride];
      i += 1;
    }
    return r;
  }

  public static float prod(int n, float[] a, int aOffset, int stride) {
    int i = 0;
    float r = 1.0f;
    while (i < n) {
      r *= a[aOffset + i * stride];
      i += 1;
    }
    return r;
  }

  public static void arraycopy(float[] src, int srcPos, float[] dest, int destPos, int length) {
    System.arraycopy(src, srcPos, dest, destPos, length);
  }

  public static void addcmul(float value, int n,
                             float[] self, int selfOffset,
                             float[] a, int aOffset,
                             float[] b, int bOffset) {
    float v = value;
    int i = 0;
    while (i < n) {
      self[i + selfOffset] += a[aOffset + i] * b[bOffset + i] * v;
      i += 1;
    }
  }

  public static void addcdiv(float value, int n,
                             float[] self, int selfOffset,
                             float[] a, int aOffset,
                             float[] b, int bOffset) {
    float v = value;
    int i = 0;

    while (i < n) {
      self[i + selfOffset] += a[aOffset + i] / b[bOffset + i] * v;
      i += 1;
    }
  }

  public static boolean nearlyEqual(float a, float b, float epsilon) {
    float absA = Math.abs(a);
    float absB = Math.abs(b);
    float diff = Math.abs(a - b);

    boolean result = false;
    if (a == b) {
      result = true;
    } else if (a == 0 || b == 0 || diff < Double.MIN_NORMAL) {
      result = diff < (epsilon * Double.MIN_NORMAL);
    } else {
      result = diff / (absA + absB) < epsilon;
    }
    return result;
  }

  public static float floor(float a) {
    return (float) Math.floor(a);
  }

  public static float ceil(float a) {
    return (float) Math.ceil(a);
  }

  public static boolean isFinite(float a) {
    return Double.isFinite(a);
  }

  public static boolean isNan(float a) {
    return Double.isNaN(a);
  }

  public static boolean isInf(float a) {
    return Double.isInfinite(a);
  }

  public static float round(float a) {
    return Math.round(a);
  }

  public static float truncate(float a) {
    if (a >= 0) {
      return (float) Math.floor(a);
    } else if (a == Math.floor(a)) {
      return a;
    } else {
      return (float) Math.floor(a) + 1;
    }
  }

  public static float floorDiv(float a, float b) {
    return (float) Math.floor(a / b);
  }

  public static float clip(float a, float lower, float upper) {
    if (lower <= upper) {
      throw new IllegalStateException("lower bound must be less or equal than upper bound");
    }
    return Math.min(Math.max(a, lower), upper);
  }

  /**
   * If MKL is not set throw exception
   */
  private static boolean checkMKL() {
    if (!MKL.isMKLLoaded()) {
      throw new IllegalStateException("mkl isn't loaded");
    }

    return true;
  }
}

