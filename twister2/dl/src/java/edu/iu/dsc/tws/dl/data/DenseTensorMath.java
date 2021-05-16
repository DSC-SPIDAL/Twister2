//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use self file except in compliance with the License.
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

import com.intel.analytics.bigdl.mkl.MKL;

import edu.iu.dsc.tws.dl.data.function.TensorFunc2;
import edu.iu.dsc.tws.dl.data.function.TensorFunc4;
import edu.iu.dsc.tws.dl.data.storage.ArrayDoubleStorage;
import edu.iu.dsc.tws.dl.data.storage.ArrayFloatStorage;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.utils.Util;

@SuppressWarnings({"LocalVariableName", "ParameterName"})
public final class DenseTensorMath {

  private DenseTensorMath() {
  }

  public static Tensor mul(DenseTensor self, Tensor x, double value) {
    if (x != null) {
      Util.require(self.nElement() == x.nElement(), "incorrect element numbers");
      self.copy(x);
    }

    if (self.isContiguous()) {
      TensorNumeric.scal(self.nElement(), value, self.storage().toDoubleArray(),
          self.storageOffset() - 1, 1);
    } else {
      TensorFunc2<double[]> func = (data, index) -> data[index] = TensorNumeric.times(data[index],
          value);
      DenseTensorApply.apply1(self, func);
    }
    return self;
  }

  public static Tensor mul(DenseTensor self, Tensor x, float value) {
    if (x != null) {
      Util.require(self.nElement() == x.nElement(), "incorrect element numbers");
      self.copy(x);
    }

    if (self.isContiguous()) {
      TensorNumeric.scal(self.nElement(), value, self.storage().toFloatArray(),
          self.storageOffset() - 1, 1);
    } else {
      TensorFunc2<float[]> func = (data, index) -> data[index] = TensorNumeric.times(data[index],
          value);
      DenseTensorApply.apply1(self, func);
    }
    return self;
  }

  /**
   * cmul.
   * @param self
   * @param x
   * @param y
   * @return
   */
  public static Tensor cmul(DenseTensor self, DenseTensor x, DenseTensor y) {
    if (x.nElement() != y.nElement() && DenseTensor.canFastBroadcast(x, y)) {
      Util.require(self.nElement() == x.nElement(),
          "the self tensor nElement is not same as x self(${self.nElement()}) x(${x.nElement()})");
      // recursive cmul
      int i = 0;
      while (i < x.size(1)) {
        cmul((DenseTensor) self.select(1, i + 1),
            (DenseTensor) x.select(1, i + 1), y);
        i += 1;
      }
    } else if (x.nElement() != y.nElement() && DenseTensor.canFastBroadcast(y, x)) {
      Util.require(self.nElement() == y.nElement(),
          "the self tensor nElement is not same as y self(${self.nElement()}) y(${y.nElement()})");
      // recursive cmul
      int i = 0;
      while (i < y.size(1)) {
        cmul((DenseTensor) self.select(1, i + 1), x,
            (DenseTensor) y.select(1, i + 1));
        i += 1;
      }
    } else if (x.nElement() != y.nElement()) {
      self.resizeAs(x).copy(x);
      self.cmul(self.expandTensor(y));
    } else {
      Util.require(self.nElement() == y.nElement(), "element number doesn't match "
          + "self(${self.nElement()}) y(${y.nElement()}) x(${x.nElement()})");
      if (self.isFloat()) {
        if (self.isContiguous() && x.isContiguous() && y.isContiguous() && MKL.isMKLLoaded()) {

          TensorNumeric.vMul(self.nElement(), x.storage().toFloatArray(), x.storageOffset() - 1,
              y.storage().toFloatArray(), y.storageOffset() - 1,
              self.storage().toFloatArray(), self.storageOffset() - 1);
        } else {
          throw new UnsupportedOperationException("operation not supported");
        }
      } else {
        if (self.isContiguous() && x.isContiguous() && y.isContiguous() && MKL.isMKLLoaded()) {

          TensorNumeric.vMul(self.nElement(), x.storage().toDoubleArray(), x.storageOffset() - 1,
              y.storage().toDoubleArray(), y.storageOffset() - 1,
              self.storage().toDoubleArray(), self.storageOffset() - 1);
        } else {
          throw new UnsupportedOperationException("operation not supported");
        }
      }

    }
    return self;
  }

  public static Tensor cdiv(DenseTensor self, Tensor x, Tensor y) {
    Util.require(self.nElement() == y.nElement() && self.nElement() == x.nElement(),
        "element number doesn't match");
    if (self.isContiguous() && y.isContiguous() && x.isContiguous() && MKL.isMKLLoaded()) {
      if (self.isFloat()) {
        TensorNumeric.vDiv(self.nElement(), x.storage().toFloatArray(), x.storageOffset() - 1,
            y.storage().toFloatArray(), y.storageOffset() - 1,
            self.storage().toFloatArray(), self.storageOffset() - 1);
      } else {
        TensorNumeric.vDiv(self.nElement(), x.storage().toDoubleArray(), x.storageOffset() - 1,
            y.storage().toDoubleArray(), y.storageOffset() - 1,
            self.storage().toDoubleArray(), self.storageOffset() - 1);
      }
    } else {
      throw new UnsupportedOperationException("operation not supported");
    }
    return self;
  }

  public static Tensor cadd(
      DenseTensor self, Tensor x, double value, Tensor y) {
    Util.require(x != null && y.nElement() == x.nElement(), "Incorrect elements");

    if (self != x && self != y) {
      self.resizeAs(x).copy(x);
    }

    if (self == x && self.isContiguous() && y.isContiguous()) {
      TensorNumeric.axpy(y.nElement(), value, y.storage().toDoubleArray(),
          y.storageOffset() - 1, 1,
          self.storage().toDoubleArray(), self.storageOffset() - 1, 1);
    } else {
      throw new UnsupportedOperationException("operation not supported");
    }
    return self;
  }

  public static Tensor csub(DenseTensor self, Tensor x, double value, Tensor y) {
    Util.require(x != null && x.nElement() == y.nElement());
    if (self != x) {
      self.resizeAs(x).copy(x);
    }

    if (self == x && self.isContiguous() && y.isContiguous()) {
      TensorNumeric.axpy(y.nElement(), value, y.storage().toDoubleArray(),
          y.storageOffset() - 1, 1, self.storage().toDoubleArray(),
          self.storageOffset() - 1, 1);
    } else {
      TensorFunc4<double[]> func2 = (data1, offset1, data2, offset2)
          -> data1[offset1] = TensorNumeric.minus(data1[offset1],
          TensorNumeric.times(value, data2[offset2]));
      DenseTensorApply.apply2(self, y, func2);
    }
    return self;
  }

  public static Tensor add(double s, DenseTensor t) {
    DenseTensor result = new DenseTensor(false);
    result.resizeAs(t);
    result.copy(t);
    TensorFunc2<double[]> func = (data, index) -> data[index] = TensorNumeric.plus(data[index], s);
    DenseTensorApply.apply1(result, func);

    return result;
  }

  public static Tensor cadd(
      DenseTensor self, Tensor x, float value, Tensor y) {
    Util.require(x != null && y.nElement() == x.nElement(), "Incorrect elements");

    if (self != x && self != y) {
      self.resizeAs(x).copy(x);
    }

    if (self == x && self.isContiguous() && y.isContiguous()) {
      TensorNumeric.axpy(y.nElement(), value, y.storage().toFloatArray(),
          y.storageOffset() - 1, 1,
          self.storage().toFloatArray(), self.storageOffset() - 1, 1);
    } else {
      throw new UnsupportedOperationException("operation not supported");
    }
    return self;
  }

  public static Tensor csub(DenseTensor self, Tensor x, float value, Tensor y) {
    Util.require(x != null && x.nElement() == y.nElement());
    if (self != x) {
      self.resizeAs(x).copy(x);
    }

    if (self == x && self.isContiguous() && y.isContiguous()) {
      TensorNumeric.axpy(y.nElement(), value, y.storage().toFloatArray(),
          y.storageOffset() - 1, 1, self.storage().toFloatArray(),
          self.storageOffset() - 1, 1);
    } else {
      TensorFunc4<float[]> func2 = (data1, offset1, data2, offset2)
          -> data1[offset1] = TensorNumeric.minus(data1[offset1],
          TensorNumeric.times(value, data2[offset2]));
      DenseTensorApply.apply2(self, y, func2);
    }
    return self;
  }

  public static Tensor add(float s, DenseTensor t) {
    DenseTensor result = new DenseTensor(true);
    result.resizeAs(t);
    result.copy(t);
    TensorFunc2<float[]> func = (data, index) -> data[index] = TensorNumeric.plus(data[index], s);
    DenseTensorApply.apply1(result, func);

    return result;
  }

  public static Tensor add(DenseTensor self, Tensor t) {
    DenseTensor result = new DenseTensor(self.isFloat());
    result.resizeAs(self);
    result.copy(self);
    int n = result.nElement();
    if (self.isFloat()) {
      if (result.isContiguous() && t.isContiguous() && n == t.nElement()) {
        TensorNumeric.axpy(n, 1.0f, t.storage().toFloatArray(), t.storageOffset() - 1, 1,
            result.storage().toFloatArray(),
            result.storageOffset() - 1, 1);
        return result;
      } else {
        TensorFunc4<float[]> func2 = (data1, offset1, data2, offset2)
            -> data1[offset1] = TensorNumeric.plus(data1[offset1], data2[offset2]);
        DenseTensorApply.apply2(self, t, func2);
        return result;
      }
    } else {
      if (result.isContiguous() && t.isContiguous() && n == t.nElement()) {
        TensorNumeric.axpy(n, 1.0, t.storage().toDoubleArray(), t.storageOffset() - 1, 1,
            result.storage().toDoubleArray(),
            result.storageOffset() - 1, 1);
        return result;
      } else {
        TensorFunc4<double[]> func2 = (data1, offset1, data2, offset2)
            -> data1[offset1] = TensorNumeric.plus(data1[offset1], data2[offset2]);
        DenseTensorApply.apply2(self, t, func2);
        return result;
      }
    }
  }

  public static Tensor sub(double s, DenseTensor t) {
    DenseTensor result = new DenseTensor(false);
    result.resizeAs(t);
    result.copy(t);
    TensorFunc2<double[]> func = (data, index) -> data[index] = TensorNumeric.minus(data[index], s);
    DenseTensorApply.apply1(result, func);
    return result;
  }

  public static Tensor sub(float s, DenseTensor t) {
    DenseTensor result = new DenseTensor(true);
    result.resizeAs(t);
    result.copy(t);
    TensorFunc2<float[]> func = (data, index) -> data[index] = TensorNumeric.minus(data[index], s);
    DenseTensorApply.apply1(result, func);
    return result;
  }

  public static Tensor sub(DenseTensor self, Tensor t) {
    DenseTensor result = new DenseTensor(self.isFloat());
    result.resizeAs(self);
    result.copy(self);
    if (self.isFloat()) {
      TensorFunc4<float[]> func2 = (data1, offset1, data2, offset2)
          -> data1[offset1] = TensorNumeric.minus(data1[offset1], data2[offset2]);
      DenseTensorApply.apply2(result, t, func2);
    } else {
      TensorFunc4<double[]> func2 = (data1, offset1, data2, offset2)
          -> data1[offset1] = TensorNumeric.minus(data1[offset1], data2[offset2]);
      DenseTensorApply.apply2(result, t, func2);
    }

    return result;
  }

  public static Tensor neg(DenseTensor self) {
    DenseTensor result = new DenseTensor(self.isFloat());
    result.resizeAs(self);
    result.copy(self);
    if (self.isFloat()) {
      TensorFunc2<double[]> func = (data, index)
          -> data[index] = TensorNumeric.negative(data[index]);
      DenseTensorApply.apply1(result, func);
    } else {
      TensorFunc2<double[]> func = (data, index)
          -> data[index] = TensorNumeric.negative(data[index]);
      DenseTensorApply.apply1(result, func);
    }

    return result;
  }

  public static Tensor divide(double s, DenseTensor t) {
    DenseTensor result = new DenseTensor(false);
    result.resizeAs(t);
    result.copy(t);
    TensorFunc2<double[]> func = (data, index)
        -> data[index] = TensorNumeric.divide(data[index], s);
    DenseTensorApply.apply1(result, func);
    return result;
  }

  public static Tensor divide(float s, DenseTensor t) {
    DenseTensor result = new DenseTensor(true);
    result.resizeAs(t);
    result.copy(t);
    TensorFunc2<float[]> func = (data, index)
        -> data[index] = TensorNumeric.divide(data[index], s);
    DenseTensorApply.apply1(result, func);
    return result;
  }

  public static Tensor divide(DenseTensor self, Tensor t) {
    DenseTensor result = new DenseTensor(self.isFloat());
    result.resizeAs(self);
    result.copy(self);
    if (self.isFloat()) {
      TensorFunc4<float[]> func2 = (data1, offset1, data2, offset2)
          -> data1[offset1] = TensorNumeric.divide(data1[offset1], data2[offset2]);
      DenseTensorApply.apply2(result, t, func2);
    } else {
      TensorFunc4<double[]> func2 = (data1, offset1, data2, offset2)
          -> data1[offset1] = TensorNumeric.divide(data1[offset1], data2[offset2]);
      DenseTensorApply.apply2(result, t, func2);
    }

    return result;
  }

  public static Tensor mul(double s, DenseTensor t) {
    DenseTensor result = new DenseTensor(false);
    result.resizeAs(t);
    result.copy(t);
    TensorFunc2<double[]> func = (data, index) -> data[index] = TensorNumeric.times(data[index], s);
    DenseTensorApply.apply1(result, func);
    return result;
  }

  public static Tensor mul(float s, DenseTensor t) {
    DenseTensor result = new DenseTensor(true);
    result.resizeAs(t);
    result.copy(t);
    TensorFunc2<float[]> func = (data, index) -> data[index] = TensorNumeric.times(data[index], s);
    DenseTensorApply.apply1(result, func);
    return result;
  }

  public static Tensor mul(Tensor self, Tensor t) {
    if (self.nDimension() == 1 && t.nDimension() == 1) {
      Util.require(self.size(1) == t.size(1), "vector size not match");
      if (self.isFloat()) {
        float result = TensorNumeric.dot(self.nElement(), self.storage().toFloatArray(),
            self.storageOffset() - 1, self.stride(1), t.storage().toFloatArray(),
            t.storageOffset() - 1, t.stride(1));
        new DenseTensor(new ArrayFloatStorage(new float[]{result}));
      } else {
        double result = TensorNumeric.dot(self.nElement(), self.storage().toDoubleArray(),
            self.storageOffset() - 1, self.stride(1), t.storage().toDoubleArray(),
            t.storageOffset() - 1, t.stride(1));
        new DenseTensor(new ArrayDoubleStorage(new double[]{result}));
      }
    } else if (self.nDimension() == 2 && t.nDimension() == 1) {
      DenseTensor result = new DenseTensor(self.size(1), self.isFloat());
      if (self.isFloat()) {
        DenseTensorBLAS.gemv(1.0f, self, t, 0.0f, result);
      } else {
        DenseTensorBLAS.gemv(1.0, self, t, 0.0, result);
      }
      return result;
    } else if (self.nDimension() == 2 && t.nDimension() == 2) {
      Tensor result = new DenseTensor(t.size(2), self.size(1), self.isFloat()).t();
      if (self.isFloat()) {
        addmm(result, 0.0f, result, 1.0f, self, t);
      } else {
        addmm(result, 0.0, result, 1.0, self, t);
      }
      return result;
    } else {
      throw new UnsupportedOperationException("multiplication between ${self.nDimension()}D and "
          + "${t.nDimension()}D not yet supported");
    }
    return null;
  }

  public static Tensor pow(DenseTensor self, Tensor x, double n) {
    Util.require(self.nElement() == x.nElement());
    if (MKL.isMKLLoaded() && self.isContiguous() && x.isContiguous()) {
      TensorNumeric.vPowx(self.nElement(), x.storage().toDoubleArray(), x.storageOffset() - 1, n,
          self.storage().toDoubleArray(), self.storageOffset() - 1);
    } else {
      TensorFunc4<double[]> func2 = (data1, offset1, data2, offset2)
          -> data1[offset1] = TensorNumeric.pow(data2[offset2], n);
      DenseTensorApply.apply2(self, x, func2);
    }
    return self;
  }

  public static Tensor pow(DenseTensor self, Tensor x, float n) {
    Util.require(self.nElement() == x.nElement());
    if (MKL.isMKLLoaded() && self.isContiguous() && x.isContiguous()) {
      TensorNumeric.vPowx(self.nElement(), x.storage().toFloatArray(), x.storageOffset() - 1, n,
          self.storage().toFloatArray(), self.storageOffset() - 1);
    } else {
      TensorFunc4<float[]> func2 = (data1, offset1, data2, offset2)
          -> data1[offset1] = TensorNumeric.pow(data2[offset2], n);
      DenseTensorApply.apply2(self, x, func2);
    }
    return self;
  }

  public static Tensor exp(DenseTensor self, Tensor x) {
    if (self.nElement() != x.nElement()) {
      self.resizeAs(x);
    }

    if (self.isFloat()) {
      if (MKL.isMKLLoaded() && self.isContiguous() && x.isContiguous()) {
        TensorNumeric.vExp(self.nElement(), x.storage().toFloatArray(), x.storageOffset() - 1,
            self.storage().toFloatArray(), self.storageOffset() - 1);
      } else {
        TensorFunc4<float[]> func = (data1, offset1, data2, offset2)
            -> data1[offset1] = TensorNumeric.exp(data2[offset2]);
        DenseTensorApply.apply2(self, x, func);
      }
    } else {
      if (MKL.isMKLLoaded() && self.isContiguous() && x.isContiguous()) {
        TensorNumeric.vExp(self.nElement(), x.storage().toDoubleArray(), x.storageOffset() - 1,
            self.storage().toDoubleArray(), self.storageOffset() - 1);
      } else {
        TensorFunc4<double[]> func = (data1, offset1, data2, offset2)
            -> data1[offset1] = TensorNumeric.exp(data2[offset2]);
        DenseTensorApply.apply2(self, x, func);
      }
    }

    return self;
  }

  public static Tensor log(DenseTensor self, Tensor x) {
    Util.require(self.nElement() == x.nElement());

    if (self.isFloat()) {
      if (MKL.isMKLLoaded() && self.isContiguous() && x.isContiguous()) {
        TensorNumeric.vLn(self.nElement(), x.storage().toFloatArray(), x.storageOffset() - 1,
            self.storage().toFloatArray(), self.storageOffset() - 1);
      } else {
        TensorFunc4<float[]> func = (data1, offset1, data2, offset2)
            -> data1[offset1] = TensorNumeric.log(data2[offset2]);
        DenseTensorApply.apply2(self, x, func);
      }
    } else {
      if (MKL.isMKLLoaded() && self.isContiguous() && x.isContiguous()) {
        TensorNumeric.vLn(self.nElement(), x.storage().toDoubleArray(), x.storageOffset() - 1,
            self.storage().toDoubleArray(), self.storageOffset() - 1);
      } else {
        TensorFunc4<double[]> func = (data1, offset1, data2, offset2)
            -> data1[offset1] = TensorNumeric.log(data2[offset2]);
        DenseTensorApply.apply2(self, x, func);
      }
    }
    return self;
  }

  public static Tensor sqrt(DenseTensor self, Tensor x) {
    Util.require(self.nElement() == x.nElement());
    if (self.isFloat()) {
      if (MKL.isMKLLoaded() && self.isContiguous() && x.isContiguous()) {
        TensorNumeric.vSqrt(self.nElement(), x.storage().toFloatArray(), x.storageOffset() - 1,
            self.storage().toFloatArray(), self.storageOffset() - 1);
      } else {
        TensorFunc4<float[]> func = (data1, offset1, data2, offset2)
            -> data1[offset1] = TensorNumeric.sqrt(data2[offset2]);
        DenseTensorApply.apply2(self, x, func);
      }
    } else {
      if (MKL.isMKLLoaded() && self.isContiguous() && x.isContiguous()) {
        TensorNumeric.vSqrt(self.nElement(), x.storage().toDoubleArray(), x.storageOffset() - 1,
            self.storage().toDoubleArray(), self.storageOffset() - 1);
      } else {
        TensorFunc4<double[]> func = (data1, offset1, data2, offset2)
            -> data1[offset1] = TensorNumeric.sqrt(data2[offset2]);
        DenseTensorApply.apply2(self, x, func);
      }
    }
    return self;
  }

  public static Tensor tanh(DenseTensor self, Tensor x) {
    Util.require(self.nElement() == x.nElement());
    if (self.isFloat()) {
      if (MKL.isMKLLoaded() && self.isContiguous() && x.isContiguous()) {
        TensorNumeric.vTanh(self.nElement(), x.storage().toFloatArray(), x.storageOffset() - 1,
            self.storage().toFloatArray(), self.storageOffset() - 1);
      } else {
        TensorFunc4<float[]> func = (data1, offset1, data2, offset2)
            -> data1[offset1] = TensorNumeric.tanh(data2[offset2]);
        DenseTensorApply.apply2(self, x, func);
      }
    } else {
      if (MKL.isMKLLoaded() && self.isContiguous() && x.isContiguous()) {
        TensorNumeric.vTanh(self.nElement(), x.storage().toDoubleArray(), x.storageOffset() - 1,
            self.storage().toDoubleArray(), self.storageOffset() - 1);
      } else {
        TensorFunc4<double[]> func = (data1, offset1, data2, offset2)
            -> data1[offset1] = TensorNumeric.tanh(data2[offset2]);
        DenseTensorApply.apply2(self, x, func);
      }
    }
    return self;
  }

  public static Tensor log1p(DenseTensor self, Tensor x) {
    Util.require(self.nElement() == x.nElement());
    if (self.isFloat()) {
      if (MKL.isMKLLoaded() && self.isContiguous() && x.isContiguous()) {
        TensorNumeric.vLog1p(self.nElement(), x.storage().toFloatArray(), x.storageOffset() - 1,
            self.storage().toFloatArray(), self.storageOffset() - 1);
      } else {
        TensorFunc4<float[]> func = (data1, offset1, data2, offset2)
            -> data1[offset1] = TensorNumeric.log1p(data2[offset2]);
        DenseTensorApply.apply2(self, x, func);
      }
    } else {
      if (MKL.isMKLLoaded() && self.isContiguous() && x.isContiguous()) {
        TensorNumeric.vLog1p(self.nElement(), x.storage().toDoubleArray(), x.storageOffset() - 1,
            self.storage().toDoubleArray(), self.storageOffset() - 1);
      } else {
        TensorFunc4<double[]> func = (data1, offset1, data2, offset2)
            -> data1[offset1] = TensorNumeric.log1p(data2[offset2]);
        DenseTensorApply.apply2(self, x, func);

      }
    }

    return self;
  }

  public static double prodAll(DenseTensor self) {
    double[] product = new double[1];
    TensorFunc2<double[]> func = (data, index) ->
        product[0] = TensorNumeric.times(data[index], product[0]);
    DenseTensorApply.apply1(self, func);
    return product[0];
  }

  public static float prodAllf(DenseTensor self) {
    float[] product = new float[1];
    TensorFunc2<float[]> func = (data, index) ->
        product[0] = TensorNumeric.times(data[index], product[0]);
    DenseTensorApply.apply1(self, func);
    return product[0];
  }

  public static double sumAll(DenseTensor self) {
    double[] sum = new double[1];
    TensorFunc2<double[]> func = (data, index) -> sum[0] = TensorNumeric.plus(data[index], sum[0]);
    DenseTensorApply.apply1(self, func);
    return sum[0];
  }

  public static float sumAllf(DenseTensor self) {
    float[] sum = new float[1];
    TensorFunc2<float[]> func = (data, index) -> sum[0] = TensorNumeric.plus(data[index], sum[0]);
    DenseTensorApply.apply1(self, func);
    return sum[0];
  }

  public static Tensor prod(DenseTensor self, Tensor x, int _dim) {
    Util.require(_dim >= 0 && _dim < x.nDimension(), "dimension ${_dim + 1} out of range");
    DenseTensor result = (self == null) ? new DenseTensor(self.isFloat()) : self;
    int[] sizes = x.size();
    sizes[_dim] = 1;
    result.resize(sizes);
    throw new UnsupportedOperationException("operation not supported");
  }

  public static Tensor sum(DenseTensor self, Tensor x, int _dim) {
    Util.require(_dim >= 0 && _dim < x.nDimension(), "dimension ${_dim + 1} out of range");
    DenseTensor result = (self == null) ? new DenseTensor(self.isFloat()) : self;
    int[] sizes = x.size();
    sizes[_dim] = 1;
    result.resize(sizes);
    throw new UnsupportedOperationException("operation not supported");
  }

  public static double maxAll(DenseTensor self) {
    double[] max = new double[1];
    boolean[] first = new boolean[]{true};

    TensorFunc2<double[]> func = (data, index) -> {
      if (first[0]) {
        first[0] = false;
        max[0] = data[index];
      } else if (TensorNumeric.isGreater(data[index], max[0])) {
        max[0] = data[index];
      }
    };
    DenseTensorApply.apply1(self, func);
    return max[0];
  }

  public static float maxAllf(DenseTensor self) {
    float[] max = new float[1];
    boolean[] first = new boolean[]{true};

    TensorFunc2<float[]> func = (data, index) -> {
      if (first[0]) {
        first[0] = false;
        max[0] = data[index];
      } else if (TensorNumeric.isGreater(data[index], max[0])) {
        max[0] = data[index];
      }
    };
    DenseTensorApply.apply1(self, func);
    return max[0];
  }

  public static double minAll(DenseTensor self) {
    double[] min = new double[1];
    boolean[] first = new boolean[]{true};
    TensorFunc2<double[]> func = (data, index) -> {
      if (first[0]) {
        first[0] = false;
        min[0] = data[index];
      } else if (TensorNumeric.isGreater(min[0], data[index])) {
        min[0] = data[index];
      }
    };
    DenseTensorApply.apply1(self, func);
    return min[0];
  }

  public static float minAllf(DenseTensor self) {
    float[] min = new float[1];
    boolean[] first = new boolean[]{true};
    TensorFunc2<float[]> func = (data, index) -> {
      if (first[0]) {
        first[0] = false;
        min[0] = data[index];
      } else if (TensorNumeric.isGreater(min[0], data[index])) {
        min[0] = data[index];
      }
    };
    DenseTensorApply.apply1(self, func);
    return min[0];
  }

  /**
   * addmm
   *
   * @param r
   * @param beta
   * @param t
   * @param alpha
   * @param m1
   * @param m2
   * @return
   */
  public static Tensor addmm(Tensor r, double beta, Tensor t,
                             double alpha, Tensor m1, Tensor m2) {
    Util.require(m1.dim() == 2 && m2.dim() == 2,
        "matrices expected, got ${m1.dim()}, ${m2.dim()} tensors");
    Util.require(m1.size(2) == m2.size(1),
        "size mismatch, m1:${m1.size().mkString()} m2:${m2.size().mkString()}");
    Util.require(t.dim() == 2,
        "matrix expected, got ${t.dim()} tensor for t");
    Util.require(t.size(1) == m1.size(1) && t.size(2) == m2.size(2),
        "size mismatch. t:${t.size().mkString()}, "
            + "m1:${m1.size().mkString()} + m2:${m2.size().mkString()}");

    if (r != t) {
      r.resizeAs(t);
      r.copy(t);
    }

    Tensor _r = null;
    Tensor _m1 = m1;
    Tensor _m2 = m2;
    int transpose_r = ' ';
    if (r.stride(1) == 1 && r.stride(2) != 0) {
      transpose_r = 'n';
      _r = r;
    } else if (r.stride(2) == 1 && r.stride(1) != 0) {
      Tensor swap = _m2;
      _m2 = _m1;
      _m1 = swap;
      transpose_r = 't';
      _r = r;
    } else {
      transpose_r = 'n';
      _r = new DenseTensor(r.size(2), r.size(1), r.isFloat());
      _r.copy(r);
      _r = _r.transpose(1, 2);
    }

    int index1 = (transpose_r == 'n') ? 1 : 2;
    int index2 = (transpose_r == 'n') ? 2 : 1;
    char transpose_m1 = ' ';
    Tensor __m1 = null;
    if (_m1.stride(index1) == 1 && _m1.stride(index2) != 0) {
      transpose_m1 = 'n';
      __m1 = _m1;
    } else if (_m1.stride(index2) == 1 && _m1.stride(index1) != 0) {
      transpose_m1 = 't';
      __m1 = _m1;
    } else {
      transpose_m1 = (transpose_r == 'n') ? 't' : 'n';
      __m1 = _m1.contiguous();
    }

    char transpose_m2 = ' ';
    Tensor __m2 = null;
    if (_m2.stride(index1) == 1 && _m2.stride(index2) != 0) {
      transpose_m2 = 'n';
      __m2 = _m2;
    } else if (_m2.stride(index2) == 1 && _m2.stride(index1) != 0) {
      transpose_m2 = 't';
      __m2 = _m2;
    } else {
      transpose_m2 = (transpose_r == 'n') ? 't' : 'n';
      __m2 = _m2.contiguous();
    }
    int temp1 = (transpose_m1 == 'n') ? __m1.stride(index2) : __m1.stride(index1);
    int temp2 = (transpose_m2 == 'n') ? __m2.stride(index2) : __m2.stride(index1);
    DenseTensorBLAS.gemm(transpose_m1, transpose_m2, _r.size(index1), _r.size(index2),
        __m1.size(index2), alpha, __m1.storage().toDoubleArray(), __m1.storageOffset() - 1,
        temp1,
        __m2.storage().toDoubleArray(), __m2.storageOffset() - 1,
        temp2,
        beta,
        _r.storage().toDoubleArray(), _r.storageOffset() - 1,
        _r.stride(index2));
    if (_r != r) {
      r.copy(_r);
    }
    return r;
  }

  /**
   * addmm.
   * @param r
   * @param beta
   * @param t
   * @param alpha
   * @param m1
   * @param m2
   * @return
   */
  public static Tensor addmm(Tensor r, float beta, Tensor t,
                             float alpha, Tensor m1, Tensor m2) {
    Util.require(m1.dim() == 2 && m2.dim() == 2,
        "matrices expected, got ${m1.dim()}, ${m2.dim()} tensors");
    Util.require(m1.size(2) == m2.size(1),
        "size mismatch, m1:${m1.size().mkString()} m2:${m2.size().mkString()}");
    Util.require(t.dim() == 2,
        "matrix expected, got ${t.dim()} tensor for t");
    Util.require(t.size(1) == m1.size(1) && t.size(2) == m2.size(2),
        "size mismatch. t:${t.size().mkString()}, "
            + "m1:${m1.size().mkString()} + m2:${m2.size().mkString()}");
    long startTime2d = System.nanoTime();

    if (r != t) {
      r.resizeAs(t);
      r.copy(t);
    }

    Tensor _r = null;
    Tensor _m1 = m1;
    Tensor _m2 = m2;
    int transpose_r = ' ';
    if (r.stride(1) == 1 && r.stride(2) != 0) {
      transpose_r = 'n';
      _r = r;
    } else if (r.stride(2) == 1 && r.stride(1) != 0) {
      Tensor swap = _m2;
      _m2 = _m1;
      _m1 = swap;
      transpose_r = 't';
      _r = r;
    } else {
      transpose_r = 'n';
      _r = new DenseTensor(r.size(2), r.size(1), r.isFloat());
      _r.copy(r);
      _r = _r.transpose(1, 2);
    }

    int index1 = (transpose_r == 'n') ? 1 : 2;
    int index2 = (transpose_r == 'n') ? 2 : 1;
    char transpose_m1 = ' ';
    Tensor __m1 = null;
    if (_m1.stride(index1) == 1 && _m1.stride(index2) != 0) {
      transpose_m1 = 'n';
      __m1 = _m1;
    } else if (_m1.stride(index2) == 1 && _m1.stride(index1) != 0) {
      transpose_m1 = 't';
      __m1 = _m1;
    } else {
      transpose_m1 = (transpose_r == 'n') ? 't' : 'n';
      __m1 = _m1.contiguous();
    }

    char transpose_m2 = ' ';
    Tensor __m2 = null;
    if (_m2.stride(index1) == 1 && _m2.stride(index2) != 0) {
      transpose_m2 = 'n';
      __m2 = _m2;
    } else if (_m2.stride(index2) == 1 && _m2.stride(index1) != 0) {
      transpose_m2 = 't';
      __m2 = _m2;
    } else {
      transpose_m2 = (transpose_r == 'n') ? 't' : 'n';
      __m2 = _m2.contiguous();
    }
    int temp1 = (transpose_m1 == 'n') ? __m1.stride(index2) : __m1.stride(index1);
    int temp2 = (transpose_m2 == 'n') ? __m2.stride(index2) : __m2.stride(index1);

    DenseTensorBLAS.gemm(transpose_m1, transpose_m2, _r.size(index1), _r.size(index2),
        __m1.size(index2), alpha, __m1.storage().toFloatArray(), __m1.storageOffset() - 1,
        temp1,
        __m2.storage().toFloatArray(), __m2.storageOffset() - 1,
        temp2,
        beta,
        _r.storage().toFloatArray(), _r.storageOffset() - 1,
        _r.stride(index2));
    if (_r != r) {
      r.copy(_r);
    }
   // System.out.println("Update Frame addmm: " + (System.nanoTime() - startTime2d) / 1e6);
    return r;
  }

  public static Tensor addr(Tensor r, double beta, Tensor t,
                            double alpha, Tensor vec1, Tensor vec2) {
    Util.require(vec1.dim() == 1 && vec2.dim() == 1);
    Util.require(t.dim() == 2);
    Util.require(t.size(1) == vec1.size(1) && t.size(2) == vec2.size(1));

    if (r != t) {
      r.resizeAs(t).copy(t);
    }

    if (beta != 1) {
      r.mul(beta);
    }

    if (r.stride(1) == 1) {
      int lda = (t.stride(2) == 1) ? r.size(1) : r.stride(2);
      TensorNumeric.ger(vec1.size(1), vec2.size(1), alpha, vec1.storage().toDoubleArray(),
          vec1.storageOffset() - 1, vec1.stride(1), vec2.storage().toDoubleArray(),
          vec2.storageOffset() - 1, vec2.stride(1), r.storage().toDoubleArray(),
          r.storageOffset() - 1, lda);
    } else if (r.stride(2) == 1) {
      TensorNumeric.ger(vec2.size(1), vec1.size(1), alpha, vec2.storage().toDoubleArray(),
          vec2.storageOffset() - 1, vec2.stride(1), vec1.storage().toDoubleArray(),
          vec1.storageOffset() - 1, vec1.stride(1), r.storage().toDoubleArray(),
          r.storageOffset() - 1, r.stride(1));
    } else {
      Tensor cr = r.contiguous();
      TensorNumeric.ger(vec2.size(1), vec1.size(1), alpha, vec2.storage().toDoubleArray(),
          vec2.storageOffset() - 1, vec2.stride(1), vec1.storage().toDoubleArray(),
          vec1.storageOffset() - 1, vec1.stride(1), cr.storage().toDoubleArray(),
          cr.storageOffset() - 1, cr.stride(1));
      r.copy(cr);
    }

    return r;
  }

  public static Tensor addr(Tensor r, float beta, Tensor t,
                            float alpha, Tensor vec1, Tensor vec2) {
    Util.require(vec1.dim() == 1 && vec2.dim() == 1);
    Util.require(t.dim() == 2);
    Util.require(t.size(1) == vec1.size(1) && t.size(2) == vec2.size(1));

    if (r != t) {
      r.resizeAs(t).copy(t);
    }

    if (beta != 1) {
      r.mul(beta);
    }

    if (r.stride(1) == 1) {
      int lda = (t.stride(2) == 1) ? r.size(1) : r.stride(2);
      TensorNumeric.ger(vec1.size(1), vec2.size(1), alpha, vec1.storage().toFloatArray(),
          vec1.storageOffset() - 1, vec1.stride(1), vec2.storage().toFloatArray(),
          vec2.storageOffset() - 1, vec2.stride(1), r.storage().toFloatArray(),
          r.storageOffset() - 1, lda);
    } else if (r.stride(2) == 1) {
      TensorNumeric.ger(vec2.size(1), vec1.size(1), alpha, vec2.storage().toFloatArray(),
          vec2.storageOffset() - 1, vec2.stride(1), vec1.storage().toFloatArray(),
          vec1.storageOffset() - 1, vec1.stride(1), r.storage().toFloatArray(),
          r.storageOffset() - 1, r.stride(1));
    } else {
      Tensor cr = r.contiguous();
      TensorNumeric.ger(vec2.size(1), vec1.size(1), alpha, vec2.storage().toFloatArray(),
          vec2.storageOffset() - 1, vec2.stride(1), vec1.storage().toFloatArray(),
          vec1.storageOffset() - 1, vec1.stride(1), cr.storage().toFloatArray(),
          cr.storageOffset() - 1, cr.stride(1));
      r.copy(cr);
    }

    return r;
  }

  public static Tensor baddbmm(Tensor result, double beta, Tensor M, double alpha,
                               Tensor batch1, Tensor batch2) {
    Util.require(batch1.dim() == 3, "expected 3D tensor, got ${batch1.dim()}D");
    Util.require(batch2.dim() == 3, "expected 3D tensor, got ${batch2.dim()}D");
    Util.require(batch1.size(1) == batch2.size(1),
        "equal number of batches expected, got "
        + "${batch1.size(1)}, ${batch2.size(1)}");
    Util.require(batch1.size(3) == batch2.size(2), "wrong matrix size, batch1: "
        + "${batch1.size(2)}${batch1.size(3)}, batch2: "
        + "${batch2.size(2)}${batch2.size(3)}");

    int bs = batch1.size(1);
    int dim1 = batch1.size(2);
    int dim2 = batch2.size(3);
    Util.require(M.size(1) == bs, "output tensor of incorrect size");
    Util.require(M.size(2) == dim1, "output tensor of incorrect size");
    Util.require(M.size(3) == dim2, "output tensor of incorrect size");

    if (M != result) {
      return result.resizeAs(M).copy(M);
    }

    int batch = 1;
    while (batch <= batch1.size(1)) {
      Tensor m1 = batch1.select(1, batch);
      Tensor m2 = batch2.select(1, batch);
      Tensor resultMatrix = result.select(1, batch);

      addmm(resultMatrix, beta, resultMatrix, alpha, m1, m2);
      batch += 1;
    }

    return result;
  }

  public static Tensor baddbmm(Tensor result, float beta, Tensor M, float alpha,
                               Tensor batch1, Tensor batch2) {
    Util.require(batch1.dim() == 3, "expected 3D tensor, got ${batch1.dim()}D");
    Util.require(batch2.dim() == 3, "expected 3D tensor, got ${batch2.dim()}D");
    Util.require(batch1.size(1) == batch2.size(1),
        "equal number of batches expected, got "
        + "${batch1.size(1)}, ${batch2.size(1)}");
    Util.require(batch1.size(3) == batch2.size(2), "wrong matrix size, batch1: "
        + "${batch1.size(2)}${batch1.size(3)}, batch2: "
        + "${batch2.size(2)}${batch2.size(3)}");

    int bs = batch1.size(1);
    int dim1 = batch1.size(2);
    int dim2 = batch2.size(3);
    Util.require(M.size(1) == bs, "output tensor of incorrect size");
    Util.require(M.size(2) == dim1, "output tensor of incorrect size");
    Util.require(M.size(3) == dim2, "output tensor of incorrect size");

    if (M != result) {
      return result.resizeAs(M).copy(M);
    }

    int batch = 1;
    while (batch <= batch1.size(1)) {
      Tensor m1 = batch1.select(1, batch);
      Tensor m2 = batch2.select(1, batch);
      Tensor resultMatrix = result.select(1, batch);

      addmm(resultMatrix, beta, resultMatrix, alpha, m1, m2);
      batch += 1;
    }

    return result;
  }

  public static Tensor addmv(Tensor r, double beta, Tensor t, double alpha,
                             Tensor mat, Tensor vec) {
    Util.require(mat.nDimension() == 2 && vec.nDimension() == 1);
    Util.require(mat.size(2) == vec.size(1));
    Util.require(t.nDimension() == 1);
    Util.require(t.size(1) == mat.size(1), "${t.size(1)} == ${mat.size(1)}");

    if (r != t) {
      r.resizeAs(t).copy(t);
    }

    if (mat.stride(1) == 1) {
      int lda = (mat.size(2) == 1) ? mat.size(1) : mat.stride(2);
      TensorNumeric.gemv('N', mat.size(1), mat.size(2), alpha,
          mat.storage().toDoubleArray(), mat.storageOffset() - 1, lda,
          vec.storage().toDoubleArray(), vec.storageOffset() - 1, vec.stride(1),
          beta, r.storage().toDoubleArray(), r.storageOffset() - 1, r.stride(1));
    } else if (mat.stride(2) == 1) {
      TensorNumeric.gemv('T', mat.size(2), mat.size(1), alpha, mat.storage().toDoubleArray(),
          mat.storageOffset() - 1, mat.stride(1), vec.storage().toDoubleArray(),
          vec.storageOffset() - 1, vec.stride(1), beta,
          r.storage().toDoubleArray(), r.storageOffset() - 1, r.stride(1));
    } else {
      Tensor cmat = mat.contiguous();
      TensorNumeric.gemv('T', cmat.size(2), cmat.size(1), alpha,
          cmat.storage().toDoubleArray(), cmat.storageOffset() - 1, cmat.stride(1),
          vec.storage().toDoubleArray(), vec.storageOffset() - 1, vec.stride(1), beta,
          r.storage().toDoubleArray(), r.storageOffset() - 1, r.stride(1));
    }

    return r;
  }

  public static Tensor addmv(Tensor r, float beta, Tensor t, float alpha,
                             Tensor mat, Tensor vec) {
    Util.require(mat.nDimension() == 2 && vec.nDimension() == 1);
    Util.require(mat.size(2) == vec.size(1));
    Util.require(t.nDimension() == 1);
    Util.require(t.size(1) == mat.size(1), "${t.size(1)} == ${mat.size(1)}");

    if (r != t) {
      r.resizeAs(t).copy(t);
    }

    if (mat.stride(1) == 1) {
      int lda = (mat.size(2) == 1) ? mat.size(1) : mat.stride(2);
      TensorNumeric.gemv('N', mat.size(1), mat.size(2), alpha,
          mat.storage().toFloatArray(), mat.storageOffset() - 1, lda,
          vec.storage().toFloatArray(), vec.storageOffset() - 1, vec.stride(1),
          beta, r.storage().toFloatArray(), r.storageOffset() - 1, r.stride(1));
    } else if (mat.stride(2) == 1) {
      TensorNumeric.gemv('T', mat.size(2), mat.size(1), alpha, mat.storage().toFloatArray(),
          mat.storageOffset() - 1, mat.stride(1), vec.storage().toFloatArray(),
          vec.storageOffset() - 1, vec.stride(1), beta,
          r.storage().toFloatArray(), r.storageOffset() - 1, r.stride(1));
    } else {
      Tensor cmat = mat.contiguous();
      TensorNumeric.gemv('T', cmat.size(2), cmat.size(1), alpha,
          cmat.storage().toFloatArray(), cmat.storageOffset() - 1, cmat.stride(1),
          vec.storage().toFloatArray(), vec.storageOffset() - 1, vec.stride(1), beta,
          r.storage().toFloatArray(), r.storageOffset() - 1, r.stride(1));
    }

    return r;
  }

  public static double meanAll(DenseTensor self) {
    double[] sum = {0.0};

    TensorFunc2<double[]> func = (data, index) -> sum[0] = TensorNumeric.plus(data[index], sum[0]);
    DenseTensorApply.apply1(self, func);
    return TensorNumeric.divide(sum[0], self.nElement());
  }

  public static float meanAllf(DenseTensor self) {
    float[] sum = {0.0f};

    TensorFunc2<float[]> func = (data, index) -> sum[0] = TensorNumeric.plus(data[index], sum[0]);
    DenseTensorApply.apply1(self, func);
    return TensorNumeric.divide(sum[0], self.nElement());
  }

  public static Tensor mean(DenseTensor self, int _dim) {
    Util.require(_dim >= 0 && _dim < self.nDimension(), "dimension ${_dim + 1} out of range");
    DenseTensor result = new DenseTensor(self.isFloat());
    int[] sizes = self.size();
    sizes[_dim] = 1;
    //DenseTensor.resize(result, sizes, null);
    throw new UnsupportedOperationException("operation not supported");
  }

  /**
   * returns the p-norms of the Tensor x computed over the dimension dim.
   *
   * @param self
   * @param value value-norms
   * @param _dim  the dimension dim
   * @return
   */
  public static Tensor norm(DenseTensor self, Tensor result, int value, int _dim) {
    Util.require(_dim >= 0 && _dim < self.nDimension(), "invalid dimension");
    int[] sizes = self.size();
    sizes[_dim] = 1;
    result.resize(sizes);
    throw new UnsupportedOperationException("operation not supported");
  }

  public static boolean nearlyEqual(double a, double b, double epsilon) {
    return TensorNumeric.nearlyEqual(a, b, epsilon);
  }

  public static Tensor cmax(DenseTensor self, Tensor x, Tensor y) {
    Util.require(self.nElement() == y.nElement() && self.nElement() == x.nElement(),
        "element number doesn't match");
    // todo: the performance of contiguous tensor should be optimized
    throw new UnsupportedOperationException("operation not supported");
  }

  public static Tensor cmin(DenseTensor self, Tensor x, Tensor y) {
    Util.require(self.nElement() == y.nElement() && self.nElement() == x.nElement(),
        "element number doesn't match");
    // todo: the performance of contiguous tensor should be optimized
    throw new UnsupportedOperationException("operation not supported");
  }
}

