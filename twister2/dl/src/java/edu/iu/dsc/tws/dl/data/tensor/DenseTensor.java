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
package edu.iu.dsc.tws.dl.data.tensor;

import java.util.Arrays;

import com.intel.analytics.bigdl.mkl.MKL;

import edu.iu.dsc.tws.dl.data.DenseTensorApply;
import edu.iu.dsc.tws.dl.data.DenseTensorDimApply;
import edu.iu.dsc.tws.dl.data.DenseTensorMath;
import edu.iu.dsc.tws.dl.data.Storage;
import edu.iu.dsc.tws.dl.data.Table;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.TensorMath;
import edu.iu.dsc.tws.dl.data.TensorNumeric;
import edu.iu.dsc.tws.dl.data.TensorType;
import edu.iu.dsc.tws.dl.data.function.TensorDimFunc3;
import edu.iu.dsc.tws.dl.data.function.TensorFunc2;
import edu.iu.dsc.tws.dl.data.function.TensorFunc4;
import edu.iu.dsc.tws.dl.data.function.TensorFunc6;
import edu.iu.dsc.tws.dl.data.storage.ArrayDoubleStorage;
import edu.iu.dsc.tws.dl.data.storage.ArrayFloatStorage;
import edu.iu.dsc.tws.dl.data.storage.ArrayStorage;
import edu.iu.dsc.tws.dl.utils.RandomGenerator;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.TensorPair;

@SuppressWarnings({"ChainingConstructorIgnoresParameter", "NeedBraces",
    "LocalVariableName", "NoClone", "SuperClone"})
public class DenseTensor implements Tensor, TensorMath {

  private ArrayStorage storageInternal;
  private int storageOffsetInternal;
  private int[] sizeInternal;
  private int[] strideInternal;
  private int nDimensionInternal;
  private boolean isFloat = false;

  public DenseTensor(boolean isFloat) {
    this(null, 0, null, null, 0);
    this.isFloat = isFloat;
  }

  public DenseTensor(int dim1, boolean isFloat) {
    this(Util.buildStorage(dim1, isFloat), 0, new int[]{dim1}, new int[]{1}, 1);
  }

  public DenseTensor(int dim1, int dim2, boolean isFloat) {
    this(Util.buildStorage(dim1 * dim2, isFloat), 0, new int[]{dim1, dim2},
        new int[]{dim2, 1}, 2);
  }

  public DenseTensor(int dim1, int dim2, int dim3, boolean isFloat) {
    this(Util.buildStorage(dim1, isFloat), 0, new int[]{dim1, dim2, dim3},
        new int[]{dim3 * dim2, dim3, 1}, 3);
  }

  public DenseTensor(ArrayStorage storage) {
    this.storageInternal = storage;
    initWithStorage(storage, 0, new int[]{storage.length()}, new int[]{1});
    if (storage instanceof ArrayFloatStorage) {
      this.isFloat = true;
    }
  }

  public DenseTensor(ArrayStorage storage, int[] sizes, int[] stride) {
    this.storageInternal = storage;
    initWithStorage(storage, 0, sizes, stride);
    if (storage instanceof ArrayFloatStorage) {
      this.isFloat = true;
    }
  }

  public DenseTensor(double[] values) {
    this(new ArrayDoubleStorage(values));
  }

  public DenseTensor(float[] values) {
    this(new ArrayFloatStorage(values));
  }

  public DenseTensor(int[] sizes, boolean isFloat) {
    this(Util.buildStorage(TensorNumeric.product(sizes), isFloat),
        0, sizes.clone(), sizeToStride(sizes), sizes.length);
  }

  private DenseTensor(ArrayStorage storage, int storageOffset, int[] size,
                      int[] stride, int nDimension) {
    this.storageInternal = storage;
    this.storageOffsetInternal = storageOffset;
    this.sizeInternal = size;
    this.strideInternal = stride;
    this.nDimensionInternal = nDimension;
    if (storage instanceof ArrayFloatStorage) {
      this.isFloat = true;
    }
  }

  private DenseTensor(ArrayStorage storage, int storageOffset, int[] size, int[] stride) {
    this.storageInternal = storage;
    this.storageOffsetInternal = storageOffset;
    this.sizeInternal = size;
    this.strideInternal = stride;
    if (storage instanceof ArrayFloatStorage) {
      this.isFloat = true;
    }
  }

  public DenseTensor(ArrayStorage newStorage, int storageOffset, int[] size) {
    this(newStorage, 0, null, null, 0);
    if (newStorage != null) {
      int tempstorageOffset = storageOffset - 1;
      int[] tempSize = (size == null) ? new int[newStorage.length()] : size;
      int[] tempStride = (size == null) ? null : strideInternal;
      initWithStorage(newStorage, tempstorageOffset, tempSize, tempStride);
    }
  }

  private static int[] sizeToStride(int[] sizes) {
    int[] strides = new int[sizes.length];
    int jump = 1;
    int i = strides.length - 1;
    while (i >= 0) {
      strides[i] = jump;
      jump = jump * sizes[i];
      i -= 1;
    }
    return strides;
  }

  public static boolean canFastBroadcast(DenseTensor tensor, DenseTensor other) {
    if (tensor.nDimension() < other.nDimension()) return false;

    int delta = tensor.nDimension() - other.nDimension();
    int d = other.nDimension();
    // Check dimensions
    boolean broadcasting = false;
    while (d > 0) {
      if (broadcasting) {
        if (other.size(d) != 1) return false;
      } else if (tensor.size(delta + d) != other.size(d)) {
        if (other.size(d) != 1) return false;
        broadcasting = true;
      }
      d -= 1;
    }

    return true;
  }

  private void initWithStorage(ArrayStorage newStorage, int newStorageOffset,
                               int[] newSize, int[] newStride) {
    if (newSize != null && newStride != null) {
      if (newSize.length != newStride.length) {
        throw new IllegalArgumentException("inconsistent size");
      }
    }
    this.storageInternal = newStorage;
    this.storageOffsetInternal = newStorageOffset;

    if (newSize != null) {
      this.nDimensionInternal = newSize.length;
    } else if (newStride != null) {
      this.nDimensionInternal = newStride.length;
    } else {
      this.nDimensionInternal = 0;
    }
    rawResize(this, nDimensionInternal, newSize, newStride);
  }

  @Override
  public Tensor toTensor() {
    return this;
  }

  @Override
  public Table toTable() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public boolean isFloat() {
    return isFloat;
  }

  @Override
  public boolean isEmpty() {
    return this.storage() == null || this.storage().length() == 0;
  }

  @Override
  public boolean isScalar() {
    return !this.isEmpty() && this.nDimension() == 0;
  }

  @Override
  public int nDimension() {
    return this.nDimensionInternal;
  }

  @Override
  public int dim() {
    return this.nDimensionInternal;
  }

  @Override
  public int[] size() {
    if (sizeInternal == null) {
      throw new UnsupportedOperationException("Operation not supported");
    } else {
      return Arrays.copyOfRange(sizeInternal, 0, nDimensionInternal);
    }
  }

  @Override
  public int size(int dim) {
    return sizeInternal[dim - 1];
  }

  @Override
  public int[] stride() {
    if (strideInternal == null) {
      throw new UnsupportedOperationException("Operation not supported");
    } else {
      return Arrays.copyOfRange(strideInternal, 0, nDimensionInternal);
    }
  }

  @Override
  public int stride(int dim) {
    return strideInternal[dim - 1];
  }

  @Override
  public Tensor fill(double v) {
    if (this.storage() == null) return this;

    if (this.isContiguous()) {
      this.storage().fill(v, this.storageOffset(), this.nElement());
    } else {
      throw new UnsupportedOperationException("non contiguous not supported");
    }
    return this;
  }

  @Override
  public Tensor fill(float v) {
    if (this.storage() == null) return this;

    if (this.isContiguous()) {
      this.storage().fill(v, this.storageOffset(), this.nElement());
    } else {
      throw new UnsupportedOperationException("non contiguous not supported");
    }
    return this;
  }

  @Override
  public Tensor forceFill(double v) {
    return this.fill(v);
  }

  @Override
  public Tensor forceFill(float v) {
    return this.fill(v);
  }

  @Override
  public Tensor zero() {
    if (this.isFloat) {
      return this.fill(0.0f);
    } else {
      return this.fill(0.0);
    }
  }

  @Override
  public Tensor randn() {
    return this.randn(0.0, 1.0);
  }

  @Override
  public Tensor randn(double mean, double stdv) {
    if (this.isFloat) {
      return randn((float) mean, (float) stdv);
    }

    if (this.isContiguous()) {
      int i = 0;
      int total = this.nElement();
      double[] data = this.storage().toDoubleArray();
      int offset = this.storageOffset() - 1;
      while (i < total) {
        data[offset + i] = RandomGenerator.RNG().normal(mean, stdv);
        i += 1;
      }
    } else {
      throw new UnsupportedOperationException("non contiguous not supported");
    }
    return this;
  }

  @Override
  public Tensor randn(float mean, float stdv) {
    if (this.isContiguous()) {
      int i = 0;
      int total = this.nElement();
      float[] data = this.storage().toFloatArray();
      int offset = this.storageOffset() - 1;
      while (i < total) {
        data[offset + i] = (float) RandomGenerator.RNG().normal(mean, stdv);
        i += 1;
      }
    } else {
      throw new UnsupportedOperationException("non contiguous not supported");
    }
    return this;
  }

  @Override
  public Tensor rand() {
    return this.rand(0.0, 1.0);
  }

  @Override
  public Tensor rand(float lowerBound, float upperBound) {
    if (this.isContiguous()) {
      int i = 0;
      int total = this.nElement();
      float[] data = this.storage().toFloatArray();
      int offset = this.storageOffset() - 1;
      while (i < total) {
        data[offset + i] = (float) RandomGenerator.RNG().uniform(lowerBound, upperBound);
        i += 1;
      }
    } else {
      throw new UnsupportedOperationException("non contiguous not supported");
    }
    return this;
  }

  @Override
  public Tensor rand(double lowerBound, double upperBound) {
    if (this.isFloat) {
      return rand((float) lowerBound, (float) upperBound);
    }

    if (this.isContiguous()) {
      int i = 0;
      int total = this.nElement();
      double[] data = this.storage().toDoubleArray();
      int offset = this.storageOffset() - 1;
      while (i < total) {
        data[offset + i] = RandomGenerator.RNG().uniform(lowerBound, upperBound);
        i += 1;
      }
    } else {
      throw new UnsupportedOperationException("non contiguous not supported");
    }
    return this;
  }

  @Override
  public Tensor bernoulli(double p) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public Tensor bernoulli(float p) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public Tensor transpose(int dim1, int dim2) {
    DenseTensor result = newWithTensor(this);
    transpose(result, null, dim1 - 1, dim2 - 1);
    return result;
  }

  private void transpose(DenseTensor self, DenseTensor source, int dimension1, int dimension2) {
    DenseTensor src = source;
    if (src == null) src = self;
    Util.require(dimension1 >= 0 && dimension1 < src.nDimension(), "out of range");
    Util.require(dimension2 >= 0 && dimension2 < src.nDimension(), "out of range");

    set(self, src);
    if (dimension1 == dimension2) {
      return;
    }
    int z = self.strideInternal[dimension1];
    self.strideInternal[dimension1] = self.strideInternal[dimension2];
    self.strideInternal[dimension2] = z;
    z = self.sizeInternal[dimension1];
    self.sizeInternal[dimension1] = self.sizeInternal[dimension2];
    self.sizeInternal[dimension2] = z;
  }

  private DenseTensor set(DenseTensor self, DenseTensor other) {
    if (self != other) {
      return rawSet(self, other.storage(), other.storageOffset(),
          other.nDimension(), other.size(), other.stride());
    } else {
      return self;
    }
  }

  private DenseTensor newWithTensor(DenseTensor old) {
    DenseTensor newTensor = new DenseTensor(old.isFloat());
    return newTensor.rawSet(newTensor, old.storage(), old.storageOffset() - 1, old.nDimension(),
        old.size(), old.stride());
  }

  @Override
  public Tensor t() {
    Util.require(this.nDimension() == 2, "t() is only for 2D tensor");
    return transpose(1, 2);
  }

  @Override
  public Tensor apply(int index) {
    Util.require(this.nDimension() > 0, "empty or scalar tensor");
    int _index = index - 1;
    if (_index < 0) _index = this.sizeInternal[0] + _index + 1;
    Util.require(_index >= 0 && _index < this.sizeInternal[0],
        "out of range, ${_index}: 0 to ${this._size(0)}");

    DenseTensor result = newWithTensor(this);
    select(result, null, 0, _index);
    return result;
  }

  @Override
  public double apply(int[] indexes) {
    Util.require(indexes.length == this.nDimension(), "invalid size");
    int offset = this.storageOffsetInternal;
    int d = 0;
    while (d < indexes.length) {
      offset += getOffset(indexes[d] - 1, d + 1);
      d += 1;
    }
    return this.storageInternal.getDouble(offset);
  }

  @Override
  public float applyf(int[] indexes) {
    Util.require(indexes.length == this.nDimension(), "invalid size");
    int offset = this.storageOffsetInternal;
    int d = 0;
    while (d < indexes.length) {
      offset += getOffset(indexes[d] - 1, d + 1);
      d += 1;
    }
    return this.storageInternal.getFloat(offset);
  }

  private int getOffset(int z, int dim) {
    int _z = z;
    if (_z < 0) {
      _z = this.size(dim) + _z + 1;
    }
    Util.require(_z >= 0 && _z < this.size(dim), "index out of bound");
    return _z * this.stride(dim);
  }

  @Override
  public double value() {
    Util.require(1 == this.nElement(), "invalid size: 1 == ${this.nElement()}");
    int offset = this.storageOffsetInternal;
    return this.storageInternal.getDouble(offset);
  }

  @Override
  public double valueAt(int d1) {
    Util.require(1 == this.nDimension(), "invalid size: 1 == ${this.nDimension}");
    int offset = this.storageOffsetInternal;
    offset += getOffset(d1 - 1, 1);
    return this.storageInternal.getDouble(offset);
  }

  @Override
  public double valueAt(int d1, int d2) {
    Util.require(2 == this.nDimension(), "invalid size");
    int offset = this.storageOffsetInternal;
    offset += getOffset(d1 - 1, 1);
    offset += getOffset(d2 - 1, 2);
    return this.storageInternal.getDouble(offset);
  }

  @Override
  public double valueAt(int d1, int d2, int d3) {
    Util.require(3 == this.nDimension(), "invalid size");
    int offset = this.storageOffsetInternal;
    offset += getOffset(d1 - 1, 1);
    offset += getOffset(d2 - 1, 2);
    offset += getOffset(d3 - 1, 3);
    return this.storageInternal.getDouble(offset);
  }

  @Override
  public double valueAt(int d1, int d2, int d3, int d4) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public double valueAt(int d1, int d2, int d3, int d4, int d5) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public float valueAtf(int d1) {
    Util.require(1 == this.nDimension(), "invalid size: 1 == ${this.nDimension}");
    int offset = this.storageOffsetInternal;
    offset += getOffset(d1 - 1, 1);
    return this.storageInternal.getFloat(offset);
  }

  @Override
  public float valueAtf(int d1, int d2) {
    Util.require(2 == this.nDimension(), "invalid size");
    int offset = this.storageOffsetInternal;
    offset += getOffset(d1 - 1, 1);
    offset += getOffset(d2 - 1, 2);
    return this.storageInternal.getFloat(offset);
  }

  @Override
  public float valueAtf(int d1, int d2, int d3) {
    Util.require(3 == this.nDimension(), "invalid size");
    int offset = this.storageOffsetInternal;
    offset += getOffset(d1 - 1, 1);
    offset += getOffset(d2 - 1, 2);
    offset += getOffset(d3 - 1, 3);
    return this.storageInternal.getFloat(offset);
  }

  @Override
  public float valueAtf(int d1, int d2, int d3, int d4) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public float valueAtf(int d1, int d2, int d3, int d4, int d5) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public Tensor apply(Table t) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public void update(int index, double value) {
    Util.require(this.nDimension() > 0, "empty tensor");
    int _index = index - 1;
    if (_index < 0) _index = this.sizeInternal[0] + _index + 1;
    Util.require(_index >= 0 && _index < this.sizeInternal[0], "out of range");
    if (this.nDimension() == 1) {
      this.storageInternal.update(this.storageOffsetInternal + _index * this.strideInternal[0],
          value);
    } else {
      DenseTensor tensor = newWithTensor(this);
      narrow(tensor, null, 0, _index, 1);
      tensor.fill(value);
    }
  }

  @Override
  public void update(int index, float value) {
    Util.require(this.nDimension() > 0, "empty tensor");
    int _index = index - 1;
    if (_index < 0) _index = this.sizeInternal[0] + _index + 1;
    Util.require(_index >= 0 && _index < this.sizeInternal[0], "out of range");
    if (this.nDimension() == 1) {
      this.storageInternal.update(this.storageOffsetInternal + _index * this.strideInternal[0],
          value);
    } else {
      DenseTensor tensor = newWithTensor(this);
      narrow(tensor, null, 0, _index, 1);
      tensor.fill(value);
    }
  }

  @Override
  public void update(int index, Tensor src) {
    Util.require(this.nDimension() > 0, "empty or scalar tensor");
    int _index = index - 1;
    if (_index < 0) _index = this.sizeInternal[0] + _index + 1;
    Util.require(_index >= 0 && _index < this.sizeInternal[0], "out of range");
    DenseTensor tensor = newWithTensor(this);
    narrow(tensor, null, 0, _index, 1);
    tensor.copy(src);
  }

  @Override
  public void update(int[] indexes, double value) {
    Util.require(indexes.length == this.nDimension(), "invalid size");
    int offset = this.storageOffsetInternal;
    int d = 0;
    while (d < indexes.length) {
      offset += getOffset(indexes[d] - 1, d + 1);
      d += 1;
    }
    this.storageInternal.update(offset, value);
  }

  @Override
  public void update(int[] indexes, float value) {
    Util.require(indexes.length == this.nDimension(), "invalid size");
    int offset = this.storageOffsetInternal;
    int d = 0;
    while (d < indexes.length) {
      offset += getOffset(indexes[d] - 1, d + 1);
      d += 1;
    }
    this.storageInternal.update(offset, value);
  }

  @Override
  public Tensor setValue(double value) {
    Util.require(0 == this.nDimension(), "invalid size, you can only call this on a scalar");
    int offset = this.storageOffsetInternal;
    this.storageInternal.update(offset, value);
    return this;
  }

  @Override
  public Tensor setValue(float value) {
    Util.require(0 == this.nDimension(), "invalid size, you can only call this on a scalar");
    int offset = this.storageOffsetInternal;
    this.storageInternal.update(offset, value);
    return this;
  }

  @Override
  public Tensor setValue(int d1, double value) {
    Util.require(1 == this.nDimension(), "invalid size");
    int offset = this.storageOffsetInternal;
    offset += getOffset(d1 - 1, 1);
    this.storageInternal.update(offset, value);
    return this;
  }

  @Override
  public Tensor setValue(int d1, float value) {
    Util.require(1 == this.nDimension(), "invalid size");
    int offset = this.storageOffsetInternal;
    offset += getOffset(d1 - 1, 1);
    this.storageInternal.update(offset, value);
    return this;
  }

  @Override
  public Tensor setValue(int d1, int d2, double value) {
    Util.require(2 == this.nDimension(), "invalid size");
    int offset = this.storageOffsetInternal;
    offset += getOffset(d1 - 1, 1);
    offset += getOffset(d2 - 1, 2);
    this.storageInternal.update(offset, value);
    return this;
  }

  @Override
  public Tensor setValue(int d1, int d2, float value) {
    Util.require(2 == this.nDimension(), "invalid size");
    int offset = this.storageOffsetInternal;
    offset += getOffset(d1 - 1, 1);
    offset += getOffset(d2 - 1, 2);
    this.storageInternal.update(offset, value);
    return this;
  }

  @Override
  public Tensor setValue(int d1, int d2, int d3, double value) {
    Util.require(3 == this.nDimension(), "invalid size");
    int offset = this.storageOffsetInternal;
    offset += getOffset(d1 - 1, 1);
    offset += getOffset(d2 - 1, 2);
    offset += getOffset(d3 - 1, 3);
    this.storageInternal.update(offset, value);
    return this;
  }

  @Override
  public Tensor setValue(int d1, int d2, int d3, float value) {
    Util.require(3 == this.nDimension(), "invalid size");
    int offset = this.storageOffsetInternal;
    offset += getOffset(d1 - 1, 1);
    offset += getOffset(d2 - 1, 2);
    offset += getOffset(d3 - 1, 3);
    this.storageInternal.update(offset, value);
    return this;
  }

  @Override
  public Tensor setValue(int d1, int d2, int d3, int d4, double value) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public Tensor setValue(int d1, int d2, int d3, int d4, float value) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public Tensor setValue(int d1, int d2, int d3, int d4, int d5, double value) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public Tensor setValue(int d1, int d2, int d3, int d4, int d5, float value) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public void update(Table t, double value) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public void update(Table t, float value) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public void update(Table t, Tensor src) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public boolean isContiguous() {
    int s = 1;
    int d = this.nDimension() - 1;
    while (d >= 0) {
      if (this.sizeInternal[d] != 1) {
        if (s != this.strideInternal[d]) {
          return false;
        } else {
          s = s * this.sizeInternal[d];
        }
      }
      d -= 1;
    }
    return true;
  }

  @Override
  public Tensor contiguous() {
    if (!this.isContiguous()) {
      return this.clone();
    } else {
      return this;
    }
  }


  @Override
  public Tensor clone() {
    DenseTensor tensor = new DenseTensor(this.isFloat);
    resizeAs(tensor, this);
    copy(tensor, this);
    return tensor;
  }

  @Override
  public boolean isSameSizeAs(Tensor other) {
    return false;
  }

  @Override
  public Tensor emptyInstance() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor resizeAs(Tensor src) {
    resizeAs(this, src);
    return this;
  }

  @Override
  public Tensor resize(int[] sizes, int[] strides) {
    return resize(this, sizes, strides);
  }

  @Override
  public Tensor resize(int[] sizes) {
    return resize(this, sizes, null);
  }

  @Override
  public Tensor resize(int size1) {
    if (this.nDimensionInternal != 1 || this.size(1) != size1) {
      return resize(this, new int[]{size1}, null);
    } else {
      return this;
    }
  }

  @Override
  public Tensor resize(int size1, int size2) {
    if (this.nDimensionInternal != 2 || this.size(1) != size1 || this.size(2) != size2) {
      return resize(this, new int[]{size1, size2}, null);
    } else {
      return this;
    }
  }

  @Override
  public Tensor resize(int size1, int size2, int size3) {
    if (this.nDimensionInternal != 3 || this.size(1) != size1 || this.size(2) != size2
        || this.size(3) != size3) {
      return resize(this, new int[]{size1, size2, size3}, null);
    } else {
      return this;
    }
  }

  @Override
  public Tensor resize(int size1, int size2, int size3, int size4) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor resize(int size1, int size2, int size3, int size4, int size5) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public int nElement() {
    if (this.isEmpty()) {
      return 0;
    } else {
      int n = 1;
      int d = 0;
      while (d < this.nDimensionInternal) {
        n = n * this.sizeInternal[d];
        d += 1;
      }
      return n;
    }
  }

  @Override
  public Tensor select(int dim, int index) {
    int _dimension = dim - 1;
    int _sliceIndex = index - 1;

    Util.require(this.nDimension() > 0, "empty or scalar tensor cannot be selected");
    DenseTensor result = newWithTensor(this);
    select(result, null, _dimension, _sliceIndex);
    return result;
  }

  private void select(DenseTensor self, DenseTensor source, int dimension, int sliceIndex) {
    DenseTensor src = source;
    if (src == null) src = self;
    Util.require(src.nDimension() > 0, "cannot select on a scalar");
    Util.require(dimension >= 0 && dimension < src.nDimension(), "out of range");
    Util.require(sliceIndex >= 0 && sliceIndex < src.size(dimension + 1),
        "{_sliceIndex} out of range 0 to ${src.size(_dimension + 1) - 1}");

    set(self, src);
    narrow(self, null, dimension, sliceIndex, 1);

    int d = dimension;
    while (d < self.nDimension() - 1) {
      self.sizeInternal[d] = self.sizeInternal[d + 1];
      self.strideInternal[d] = self.strideInternal[d + 1];
      d += 1;
    }

    self.nDimensionInternal = self.nDimension() - 1;
  }

  private void narrow(DenseTensor self, DenseTensor source, int dimension,
                      int firstIndex, int size) {
    DenseTensor src = source;
    if (src == null) {
      src = self;
    }

    Util.require(dimension >= 0 && dimension < src.nDimension(), "dimension out of range");
    Util.require(firstIndex >= 0 && firstIndex < src.size(dimension + 1),
        "firstIndex(${_firstIndex}) out of range [0, ${src.size(_dimension + 1)})");
    Util.require(size > 0 && firstIndex + size <= src.size(dimension + 1),
        "size out of range $size (0, ${src.size(_dimension + 1)} - ${_firstIndex}]");

    set(self, src);

    if (firstIndex > 0) {
      self.storageOffsetInternal = self.storageOffsetInternal
          + firstIndex * self.strideInternal[dimension];
    }
    self.sizeInternal[dimension] = size;
  }

  @Override
  public Tensor set(Tensor other) {
    this.storageInternal = (ArrayStorage) other.storage();
    this.storageOffsetInternal = other.storageOffset() - 1;
    return rawResize(this, other.nDimension(), other.size(), other.stride());
  }

  @Override
  public Tensor set(Storage storage, int storageOffset, int[] sizes, int[] strides) {
    if (sizes != null && strides != null) {
      Util.require(sizes.length == strides.length, "Invalid Size");
    }
    this.storageInternal = (ArrayStorage) storage;
    this.storageOffsetInternal = storageOffset - 1;
    if (sizes == null) {
      return rawResize(this, 0, sizes, strides);
    } else {
      return rawResize(this, sizes.length, sizes, strides);
    }
  }

  @Override
  public Tensor set() {
    if (this.storageInternal != null) {
      this.storageInternal.resize(0);
    }
    this.nDimensionInternal = 0;
    this.sizeInternal = new int[0];
    return this;
  }

  @Override
  public ArrayStorage storage() {
    return storageInternal;
  }

  @Override
  public int storageOffset() {
    return this.storageOffsetInternal + 1;
  }

  @Override
  public Tensor narrow(int dim, int index, int size) {
    DenseTensor result = newWithTensor(this);
    narrow(result, null, dim - 1, index - 1, size);
    return result;
  }

  @Override
  public Tensor copy(Tensor other) {
    copy(this, other);
    return this;
  }

  private void copy(DenseTensor self, Tensor src) {
    Util.require(self.nElement() == src.nElement(), "self element"
        + " number(${self.nElement()}) is not"
        + " equal to source element number(${src.nElement()})");

    if (self.isEmpty()) {
      return;
    }

    if (self.isContiguous() && src.isContiguous() && sameStride(self.stride(), src.stride())) {
      if (self.isFloat()) {
        System.arraycopy(src.storage().toFloatArray(), src.storageOffset() - 1,
            self.storage().toFloatArray(), self.storageOffset() - 1, self.nElement());
        return;
      } else {
        System.arraycopy(src.storage().toDoubleArray(), src.storageOffset() - 1,
            self.storage().toDoubleArray(), self.storageOffset() - 1, self.nElement());
        return;
      }
    } else {
      throw new UnsupportedOperationException("non Contiguous not supported");
    }
  }

  private boolean sameStride(int[] l, int[] r) {
    if (l.length != r.length) {
      return false;
    }

    int i = 0;
    while (i < l.length) {
      if (l[i] != r[i]) {
        return false;
      }
      i += 1;
    }
    return true;
  }

  @Override
  public Tensor squeeze() {
    int ndim = 0;
    int d = 0;
    while (d < this.nDimensionInternal) {
      if (this.sizeInternal[d] != 1) {
        if (d != ndim) {
          this.sizeInternal[ndim] = this.sizeInternal[d];
          this.strideInternal[ndim] = this.strideInternal[d];
        }
        ndim += 1;
      }
      d += 1;
    }

    if (ndim == 0 && this.nDimensionInternal > 0) {
      this.sizeInternal[0] = 1;
      this.strideInternal[0] = 1;
      ndim = 1;
    }

    this.nDimensionInternal = ndim;
    return this;
  }

  @Override
  public Tensor squeeze(int dim) {
    if (dim >= 0 && dim < this.nDimensionInternal) {
      throw new IllegalArgumentException("dimension out of range");
    }
    if (this.size(dim) == 1 && this.nDimensionInternal > 1) {
      int d = dim;
      while (d < this.nDimensionInternal - 1) {
        this.sizeInternal[d] = this.sizeInternal[d + 1];
        this.strideInternal[d] = this.strideInternal[d + 1];
        d += 1;
      }

      this.nDimensionInternal -= 1;
    }
    return this;
  }

  @Override
  public Tensor squeezeNewTensor() {
    DenseTensor result = new DenseTensor(this.storageInternal, this.storageOffset(),
        this.sizeInternal, this.strideInternal);
    return result.squeeze();
  }

  @Override
  public Tensor view(int[] sizes) {
    if (!this.isContiguous()) {
      throw new IllegalStateException("current tensor is not contiguous");
    }
    if (TensorNumeric.product(sizes) != this.nElement()) {
      throw new IllegalStateException("invalid size eElement");
    }
    return new DenseTensor(this.storageInternal, this.storageOffset(), sizes.clone());
  }

  @Override
  public Tensor unfold(int dim, int size, int step) {
    Util.require(this.nDimensionInternal > 0, "cannot unfold an empty tensor");
    Util.require(dim > 0 && dim <= this.nDimensionInternal, "out of range");
    Util.require(size <= this.size(dim), "out of range");
    Util.require(step > 0, "invalid step");

    int[] newSize = new int[this.nDimensionInternal + 1];
    int[] newStride = new int[this.nDimensionInternal + 1];

    newSize[this.nDimensionInternal] = size;
    newStride[this.nDimensionInternal] = this.stride(dim);

    int d = 0;
    while (d < this.nDimensionInternal) {
      if (d + 1 == dim) {
        newSize[d] = (this.size(d + 1) - size) / step + 1;
        newStride[d] = step * this.stride(d + 1);
      } else {
        newSize[d] = this.size(d + 1);
        newStride[d] = this.stride(d + 1);
      }
      d = d + 1;
    }

    return new DenseTensor(this.storageInternal, this.storageOffsetInternal, newSize,
        newStride, this.dim() + 1);
  }

  @Override
  public Tensor repeatTensor(int[] sizes) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor expandAs(Tensor template) {
    return this.expand(template.size());
  }

  @Override
  public Tensor expand(int[] sizes) {
    Util.require(sizes.length == this.dim(),
        "the number of dimensions provided must equal ${this.dim()}");
    int tensorDim = this.dim();
    int[] tensorStride = this.stride();
    int[] tensorSize = this.size();

    int i = 0;
    while (i < tensorDim) {
      if (tensorSize[i] == 1) {
        tensorSize[i] = sizes[i];
        tensorStride[i] = 0;
      } else if (tensorSize[i] != sizes[i]) {
        throw new UnsupportedOperationException(
            "incorrect size: only supporting singleton expansion (size=1)");
      }
      i += 1;
    }

    return set(this.storage(), this.storageOffset(), tensorSize, tensorStride);
  }

  @Override
  public Tensor[] split(int size, int dim) {
    return new Tensor[0];
  }

  @Override
  public Tensor[] split(int dim) {
    return new Tensor[0];
  }

  @Override
  public boolean diff(Tensor other, int count, boolean reverse) {
    return false;
  }

  @Override
  public Tensor addSingletonDimension(Tensor t, int dim) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addMultiDimension(Tensor t, int[] dim) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor reshape(int[] sizes) {
    Util.require(TensorNumeric.product(sizes) == this.nElement(),
        "DenseTensor: nElement of this tensor is not equal to nElement specified by sizes");
    DenseTensor result = new DenseTensor(this.isFloat);
    result.resize(sizes);
    result.copy(this);
    return result;
  }

  @Override
  public Tensor save(String path, boolean overWrite) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public TensorNumeric getTensorNumeric() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public TensorType getTensorType() {
    return TensorType.DenseType;
  }

  @Override
  public double[] toArray() {
    Util.require(this.dim() == 1, "toArray only support 1D tensor");
    int n = this.nElement();
    double[] array = new double[n];
    int i = 0;
    while (i < n) {
      array[i] = this.valueAt(i + 1);
      i += 1;
    }

    return array;
  }

  @Override
  public float[] toFloatArray() {
    Util.require(this.dim() == 1, "toArray only support 1D tensor");
    int n = this.nElement();
    float[] array = new float[n];
    int i = 0;
    while (i < n) {
      array[i] = this.valueAtf(i + 1);
      i += 1;
    }

    return array;
  }

  @Override
  public Tensor addCopy(double s) {
    DenseTensor result = new DenseTensor(false);
    result.resizeAs(this);
    result.copy(this);

    TensorFunc2<double[]> addFunc = (data, index) -> data[index] = data[index] + s;

    DenseTensorApply.apply1(result, addFunc);
    return result;
  }

  @Override
  public Tensor addCopy(float s) {
    DenseTensor result = new DenseTensor(true);
    result.resizeAs(this);
    result.copy(this);

    TensorFunc2<float[]> addFunc = (data, index) -> data[index] = data[index] + s;

    DenseTensorApply.apply1(result, addFunc);
    return result;
  }

  @Override
  public Tensor addCopy(Tensor t) {
    DenseTensor result = new DenseTensor(t.isFloat());
    result.resizeAs(this);
    result.copy(this);
    int n = result.nDimension();
    if (t.isFloat()) {
      if (result.isContiguous() && t.isContiguous() && n == t.nElement()) {
        TensorNumeric.axpy(n, 1.0f, t.storage().toFloatArray(), t.storageOffset() - 1, 1,
            result.storage().toFloatArray(), result.storageOffset() - 1, 1);
        return result;
      } else {
        TensorFunc4<float[]> addFunc = (data1, offset1, data2, offset2)
            -> data1[offset1] = data1[offset1] + data2[offset2];
        DenseTensorApply.apply2(result, t, addFunc);
        return result;
      }
    } else {
      if (result.isContiguous() && t.isContiguous() && n == t.nElement()) {
        TensorNumeric.axpy(n, 1.0, t.storage().toDoubleArray(), t.storageOffset() - 1, 1,
            result.storage().toDoubleArray(), result.storageOffset() - 1, 1);
        return result;
      } else {
        TensorFunc4<double[]> addFunc = (data1, offset1, data2, offset2)
            -> data1[offset1] = data1[offset1] + data2[offset2];
        DenseTensorApply.apply2(result, t, addFunc);
        return result;
      }
    }
  }

  @Override
  public Tensor subCopy(double s) {
    DenseTensor result = new DenseTensor(false);
    result.resizeAs(this);
    result.copy(this);
    TensorFunc2<double[]> subFunc = (data, index) -> data[index] = data[index] - s;

    DenseTensorApply.apply1(result, subFunc);
    return result;
  }

  @Override
  public Tensor subCopy(float s) {
    DenseTensor result = new DenseTensor(true);
    result.resizeAs(this);
    result.copy(this);
    TensorFunc2<float[]> subFunc = (data, index) -> data[index] = data[index] - s;

    DenseTensorApply.apply1(result, subFunc);
    return result;
  }

  @Override
  public Tensor subCopy(Tensor t) {
    DenseTensor result = new DenseTensor(t.isFloat());
    result.resizeAs(this);
    result.copy(this);
    if (t.isFloat()) {
      TensorFunc4<float[]> subFunc = (data1, offset1, data2, offset2)
          -> data1[offset1] = data1[offset1] - data2[offset2];
      DenseTensorApply.apply2(result, t, subFunc);
    } else {
      TensorFunc4<double[]> subFunc = (data1, offset1, data2, offset2)
          -> data1[offset1] = data1[offset1] - data2[offset2];
      DenseTensorApply.apply2(result, t, subFunc);
    }
    return result;
  }

  @Override
  public Tensor negative() {
    DenseTensor result = new DenseTensor(this.isFloat);
    result.resizeAs(this);
    result.copy(this);
    if (this.isFloat()) {
      TensorFunc2<float[]> negFunc = (data, index) -> data[index] = -data[index];
      DenseTensorApply.apply1(result, negFunc);
    } else {
      TensorFunc2<double[]> negFunc = (data, index) -> data[index] = -data[index];
      DenseTensorApply.apply1(result, negFunc);
    }
    return result;
  }

  @Override
  public Tensor divCopy(double s) {
    DenseTensor result = new DenseTensor(false);
    result.resizeAs(this);
    result.copy(this);
    TensorFunc2<double[]> func = (data, index) -> data[index] = data[index] / s;
    DenseTensorApply.apply1(result, func);
    return result;
  }

  @Override
  public Tensor divCopy(float s) {
    DenseTensor result = new DenseTensor(true);
    result.resizeAs(this);
    result.copy(this);
    TensorFunc2<float[]> func = (data, index) -> data[index] = data[index] / s;
    DenseTensorApply.apply1(result, func);
    return result;
  }

  @Override
  public Tensor divCopy(Tensor t) {
    DenseTensor result = new DenseTensor(t.isFloat());
    result.resizeAs(this);
    result.copy(this);
    if (t.isFloat()) {
      TensorFunc4<float[]> func = (data1, offset1, data2, offset2)
          -> data1[offset1] = data1[offset1] / data2[offset2];
      DenseTensorApply.apply2(result, t, func);
    } else {
      TensorFunc4<double[]> func = (data1, offset1, data2, offset2)
          -> data1[offset1] = data1[offset1] / data2[offset2];
      DenseTensorApply.apply2(result, t, func);
    }
    return result;
  }

  @Override
  public Tensor mulCopy(double s) {
    DenseTensor result = new DenseTensor(false);
    result.resizeAs(this);
    result.copy(this);
    TensorFunc2<double[]> func = (data, index) -> data[index] = data[index] * s;

    DenseTensorApply.apply1(result, func);
    return result;
  }

  @Override
  public Tensor mulCopy(float s) {
    DenseTensor result = new DenseTensor(true);
    result.resizeAs(this);
    result.copy(this);
    TensorFunc2<float[]> func = (data, index) -> data[index] = data[index] * s;

    DenseTensorApply.apply1(result, func);
    return result;
  }

  @Override
  public Tensor mulCopy(Tensor t) {
    return DenseTensorMath.mul(this, t);
  }

  @Override
  public double sum() {
    return DenseTensorMath.sumAll(this);
  }

  @Override
  public float sumf() {
    return DenseTensorMath.sumAllf(this);
  }

  @Override
  public double prod() {
    return DenseTensorMath.prodAll(this);
  }

  @Override
  public float prodf() {
    return DenseTensorMath.prodAllf(this);
  }

  @Override
  public Tensor prod(Tensor x, int dim) {
    return DenseTensorMath.prod(this, x, dim - 1);
  }

  @Override
  public Tensor sum(int dim) {
    return DenseTensorMath.sum(null, this, dim - 1);
  }

  @Override
  public Tensor sum(Tensor x, int dim) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double mean() {
    return DenseTensorMath.meanAll(this);
  }

  @Override
  public float meanf() {
    return DenseTensorMath.meanAllf(this);
  }

  @Override
  public Tensor mean(int dim) {
    return DenseTensorMath.mean(this, dim - 1);
  }

  @Override
  public double max() {
    return DenseTensorMath.maxAll(this);
  }

  @Override
  public float maxf() {
    return DenseTensorMath.maxAllf(this);
  }

  @Override
  public TensorPair max(int dim) {
    Util.require(dim > 0 && dim <= this.nDimension(), "dimension out of range");
    return max(new DenseTensor(this.isFloat), new DenseTensor(this.isFloat), dim);
  }

  @Override
  public TensorPair max(Tensor values, Tensor indices, int dim) {
    Util.require(dim > 0 && dim <= this.nDimension(), "dimension out of range");
    int[] sizes = this.size(); // here slice
    sizes[dim - 1] = 1;
    values.resize(sizes);
    indices.resize(sizes);

    if (this.isFloat()) {
// TODO: the performance of contiguous tensor should be optimize
      TensorDimFunc3<float[]> func3 = new TensorDimFunc3<float[]>() {
        @Override
        public void apply(float[] tdata, int toffset, int tstride, int tsize, float[] vdata,
                          int voffset, int vstride, int vsize, float[] idata, int ioffset,
                          int istride, int isize) {
          float max = tdata[toffset];
          int index = 1;
          int i = 0;
          while (i < tsize) {
            if (TensorNumeric.minus(tdata[toffset + i * tstride], max) > 0) {
              index = i + 1;
              max = tdata[toffset + i * tstride];
            }
            i += 1;
          }
          vdata[voffset] = max;
          idata[ioffset] = index;
        }
      };
      DenseTensorDimApply.dimApply3(this, (DenseTensor) values, (DenseTensor) indices, dim, func3);
    } else {
// TODO: the performance of contiguous tensor should be optimize
      TensorDimFunc3<double[]> func3 = new TensorDimFunc3<double[]>() {
        @Override
        public void apply(double[] tdata, int toffset, int tstride, int tsize, double[] vdata,
                          int voffset, int vstride, int vsize, double[] idata, int ioffset,
                          int istride, int isize) {
          double max = tdata[toffset];
          int index = 1;
          int i = 0;
          while (i < tsize) {
            if (TensorNumeric.minus(tdata[toffset + i * tstride], max) > 0) {
              index = i + 1;
              max = tdata[toffset + i * tstride];
            }
            i += 1;
          }
          vdata[voffset] = max;
          idata[ioffset] = index;
        }
      };
      DenseTensorDimApply.dimApply3(this, (DenseTensor) values, (DenseTensor) indices, dim, func3);
    }

    return new TensorPair(values, indices);
  }

  @Override
  public double min() {
    return DenseTensorMath.minAll(this);
  }

  @Override
  public float minf() {
    return DenseTensorMath.minAllf(this);
  }

  @Override
  public TensorPair min(int dim) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public TensorPair min(Tensor values, Tensor indices, int dim) {
    throw new UnsupportedOperationException("operation not supported");
  }

  @Override
  public Tensor scatter(int dim, Tensor index, Tensor src) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor gather(int dim, Tensor index, Tensor src) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor conv2(Tensor kernel, char vf) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor xcorr2(Tensor kernel, char vf) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor sqrt() {
    return DenseTensorMath.sqrt(this, this);
  }

  @Override
  public Tensor tanh() {
    return DenseTensorMath.tanh(this, this);
  }

  @Override
  public Tensor abs() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor add(double value, Tensor y) {
    return DenseTensorMath.cadd(this, this, value, y);
  }

  @Override
  public Tensor add(float value, Tensor y) {
    return DenseTensorMath.cadd(this, this, value, y);
  }

  @Override
  public Tensor add(Tensor x) {
    Util.require(x instanceof DenseTensor, "Only support dense tensor in this operation");
    if (this.nElement() == x.nElement()) {
      if (x.isFloat()) {
        if (MKL.isMKLLoaded() && this.isContiguous() && x.isContiguous()) {
          TensorNumeric.vAdd(this.nElement(), this.storage().toFloatArray(),
              this.storageOffset() - 1,
              x.storage().toFloatArray(), x.storageOffset() - 1,
              this.storage().toFloatArray(), this.storageOffset() - 1);
        } else {
          TensorFunc4<float[]> subFunc = (data1, offset1, data2, offset2)
              -> data1[offset1] = TensorNumeric.plus(data1[offset1], data2[offset2]);
          DenseTensorApply.apply2(this, x, subFunc);
        }
      } else {
        if (MKL.isMKLLoaded() && this.isContiguous() && x.isContiguous()) {
          TensorNumeric.vAdd(this.nElement(), this.storage().toDoubleArray(),
              this.storageOffset() - 1,
              x.storage().toDoubleArray(), x.storageOffset() - 1,
              this.storage().toDoubleArray(), this.storageOffset() - 1);
        } else {
          TensorFunc4<double[]> subFunc = (data1, offset1, data2, offset2)
              -> data1[offset1] = TensorNumeric.plus(data1[offset1], data2[offset2]);
          DenseTensorApply.apply2(this, x, subFunc);
        }
      }
    } else if (DenseTensor.canFastBroadcast(this, (DenseTensor) x)) {
      // recursive add
      int i = 0;
      while (i < this.size(1)) {
        this.select(1, i + 1).add(x);
        i += 1;
      }
    } else {
      return this.add(expandTensor((DenseTensor) x));
    }
    return this;
  }

  @Override
  public Tensor add(Tensor x, double value, Tensor y) {
    return DenseTensorMath.cadd(this, x, value, y);
  }

  @Override
  public Tensor add(Tensor x, float value, Tensor y) {
    return DenseTensorMath.cadd(this, x, value, y);
  }

  @Override
  public Tensor add(double value) {
    if (this.isContiguous()) {
      TensorNumeric.add(this.nElement(), this.storage().toDoubleArray(),
          this.storageOffset() - 1, value, 1);
      return this;
    } else {
      TensorFunc2<double[]> addFunc = (data, index) -> data[index] = data[index] + value;
      DenseTensorApply.apply1(this, addFunc);
      return this;
    }
  }

  @Override
  public Tensor add(float value) {
    if (this.isContiguous()) {
      TensorNumeric.add(this.nElement(), this.storage().toFloatArray(),
          this.storageOffset() - 1, value, 1);
      return this;
    } else {
      TensorFunc2<float[]> addFunc = (data, index) -> data[index] = data[index] + value;
      DenseTensorApply.apply1(this, addFunc);
      return this;
    }
  }


  @Override
  public Tensor add(Tensor x, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
//    Util.require(this.nElement() == x.nElement() && this.nElement() == y.nElement());
//    if (MKL.isMKLLoaded && this.isContiguous() && x.isContiguous() && y.isContiguous()) {
//      TensorNumeric.vAdd(this.nElement(), y.storage().toDoubleArray(), y.storageOffset() - 1,
//          x.storage().toDoubleArray(), x.storageOffset() - 1,
//          this.storage().toDoubleArray(), this.storageOffset() - 1);
//    } else {
//      TensorFunc6 func = (data, offset, data1, offset1, data2, offset2) -> data[o]
//      val func = new TensorFunc6[T] {
//        override def apply(data: Array[T], offset: Int, data1: Array[T],
//            offset1: Int, data2: Array[T], offset2: Int): Unit = {
//            data(offset1) = ev.plus(data1(offset1), data2(offset2))
//        }
//      }
//      DenseTensorApply.apply3[T](this, x, y, func)
//    }
//    this
  }

  @Override
  public double dot(Tensor y) {
    Util.require(this.nElement() == y.nElement());
    if (MKL.isMKLLoaded() && this.isContiguous() && y.isContiguous()) {
      return TensorNumeric.dot(this.nElement(), this.storage().toDoubleArray(),
          this.storageOffset() - 1, 1,
          y.storage().toDoubleArray(), y.storageOffset() - 1, 1);
    } else {
      double[] sum = new double[1];
      TensorFunc4<double[]> func = (data1, offset1, data2, offset2)
          -> sum[0] = sum[0] + (data1[offset1] * data2[offset2]);
      DenseTensorApply.apply2(this, y, func);

      return sum[0];
    }
  }

  @Override
  public float dotf(Tensor y) {
    if (MKL.isMKLLoaded() && this.isContiguous() && y.isContiguous()) {
      return TensorNumeric.dot(this.nElement(), this.storage().toFloatArray(),
          this.storageOffset() - 1, 1,
          y.storage().toFloatArray(), y.storageOffset() - 1, 1);
    } else {
      float[] sum = new float[1];
      TensorFunc4<float[]> func = (data1, offset1, data2, offset2)
          -> sum[0] = sum[0] + (data1[offset1] * data2[offset2]);
      DenseTensorApply.apply2(this, y, func);

      return sum[0];
    }
  }

  @Override
  public Tensor cmax(double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor cmax(float value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double dist(Tensor y, int norm) {
    return 0.0;
  }

  @Override
  public float distf(Tensor y, int norm) {
    return 0.0f;
  }

  @Override
  public Tensor addcmul(double value, Tensor tensor1, Tensor tensor2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addcmul(float value, Tensor tensor1, Tensor tensor2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addcmul(Tensor tensor1, Tensor tensor2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addcdiv(double value, Tensor tensor1, Tensor tensor2) {
    if (this.isContiguous() && tensor1.isContiguous() && tensor2.isContiguous()) {
      TensorNumeric.addcdiv(value, this.nElement(), this.storage().toDoubleArray(),
          this.storageOffset() - 1, tensor1.storage().toDoubleArray(), tensor1.storageOffset() - 1,
          tensor2.storage().toDoubleArray(), tensor2.storageOffset() - 1);
    } else {
      TensorFunc6<double[]> func = (data1, offset1, data2, offset2, data3, offset3)
          -> data1[offset1] = data1[offset1] + (data2[offset2] / data3[offset3]) * value;
      DenseTensorApply.apply3(this, tensor1, tensor2, func);
    }
    return this;
  }

  @Override
  public Tensor addcdiv(float value, Tensor tensor1, Tensor tensor2) {
    if (this.isContiguous() && tensor1.isContiguous() && tensor2.isContiguous()) {
      TensorNumeric.addcdiv(value, this.nElement(), this.storage().toFloatArray(),
          this.storageOffset() - 1, tensor1.storage().toFloatArray(), tensor1.storageOffset() - 1,
          tensor2.storage().toFloatArray(), tensor2.storageOffset() - 1);
    } else {
      TensorFunc6<float[]> func = (data1, offset1, data2, offset2, data3, offset3)
          -> data1[offset1] = data1[offset1] + (data2[offset2] / data3[offset3]) * value;
      DenseTensorApply.apply3(this, tensor1, tensor2, func);
    }
    return this;
  }

  @Override
  public Tensor sub(double value, Tensor y) {
    return DenseTensorMath.csub(this, this, TensorNumeric.negative(value), y);
  }

  @Override
  public Tensor sub(float value, Tensor y) {
    return DenseTensorMath.csub(this, this, TensorNumeric.negative(value), y);
  }

  @Override
  public Tensor sub(Tensor x, double value, Tensor y) {
    return DenseTensorMath.csub(this, x, value, y);
  }

  @Override
  public Tensor sub(Tensor x, float value, Tensor y) {
    return DenseTensorMath.csub(this, x, value, y);
  }

  @Override
  public Tensor sub(Tensor x) {
    Util.require(x instanceof DenseTensor, "Only dense tensor is supported in this operation");
    if (this.nElement() == x.nElement()) {
      if (((DenseTensor) x).isFloat) {
        if (MKL.isMKLLoaded() && this.isContiguous() && x.isContiguous()) {
          TensorNumeric.vSub(this.nElement(), this.storage().toFloatArray(),
              this.storageOffset() - 1, x.storage().toFloatArray(), x.storageOffset() - 1,
              this.storage().toFloatArray(), this.storageOffset() - 1);
        } else {
          TensorFunc4<double[]> subFunc = (data1, offset1, data2, offset2)
              -> data1[offset1] = TensorNumeric.minus(data1[offset1], data2[offset2]);
          DenseTensorApply.apply2(this, x, subFunc);
        }
      } else {
        if (MKL.isMKLLoaded() && this.isContiguous() && x.isContiguous()) {
          TensorNumeric.vSub(this.nElement(), this.storage().toDoubleArray(),
              this.storageOffset() - 1, x.storage().toDoubleArray(), x.storageOffset() - 1,
              this.storage().toDoubleArray(), this.storageOffset() - 1);
        } else {
          TensorFunc4<double[]> subFunc = (data1, offset1, data2, offset2)
              -> data1[offset1] = TensorNumeric.minus(data1[offset1], data2[offset2]);
          DenseTensorApply.apply2(this, x, subFunc);
        }
      }

    } else if (DenseTensor.canFastBroadcast(this, (DenseTensor) x)) {
      // recursive add
      int i = 0;
      while (i < this.size(1)) {
        this.select(1, i + 1).sub(x);
        i += 1;
      }
    } else {
      this.sub(expandTensor((DenseTensor) x));
    }

    return this;
  }

  @Override
  public Tensor sub(Tensor x, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
//    Util.require(this.nElement() == x.nElement() && this.nElement() == y.nElement());
//    if (MKL.isMKLLoaded() && this.isContiguous() && x.isContiguous() && y.isContiguous()) {
//      TensorNumeric.vSub(this.nElement(), x.storage().toDoubleArray(), x.storageOffset() - 1,
//          y.storage().toDoubleArray(), y.storageOffset() - 1,
//          this.storage().toDoubleArray(), this.storageOffset() - 1);
//    } else {
//      TensorFunc4 subFunc = (data1, offset1, data2, offset2)
//          -> data1[offset1] = TensorNumeric.minus(data1[offset1], data2[offset2]);
//      DenseTensorApply.apply2(this, x, subFunc);
//      val func = new TensorFunc6[T] {
//        override def apply (data: Array[T], offset: Int, data1: Array[T],
//            offset1: Int, data2: Array[T], offset2: Int): Unit = {
//            data(offset) = ev.minus(data1(offset1), data2(offset2))
//        }
//      }
//      DenseTensorApply.apply3[T](this, x, y, func)
//    }
//    this
  }

  @Override
  public Tensor sub(double value) {
    if (this.isContiguous()) {
      TensorNumeric.sub(this.nElement(), this.storage().toDoubleArray(),
          this.storageOffset() - 1, value, 1);
      return this;
    } else {
      TensorFunc2<double[]> addFunc = (data, index) -> data[index] = data[index] - value;
      DenseTensorApply.apply1(this, addFunc);
      return this;
    }
  }

  @Override
  public Tensor sub(float value) {
    if (this.isContiguous()) {
      TensorNumeric.sub(this.nElement(), this.storage().toFloatArray(),
          this.storageOffset() - 1, value, 1);
      return this;
    } else {
      TensorFunc2<float[]> addFunc = (data, index) -> data[index] = data[index] - value;
      DenseTensorApply.apply1(this, addFunc);
      return this;
    }
  }

  @Override
  public Tensor cmul(Tensor y) {
    return DenseTensorMath.cmul(this, this, (DenseTensor) y);
  }

  @Override
  public Tensor cmul(Tensor x, Tensor y) {
    return DenseTensorMath.cmul(this, (DenseTensor) x, (DenseTensor) y);
  }

  @Override
  public Tensor cdiv(Tensor y) {
    return DenseTensorMath.cdiv(this, this, y);
  }

  @Override
  public Tensor cdiv(Tensor x, Tensor y) {
    return DenseTensorMath.cdiv(this, x, y);
  }

  @Override
  public Tensor mul(double value) {
    return DenseTensorMath.mul(this, null, value);
  }

  @Override
  public Tensor mul(float value) {
    return DenseTensorMath.mul(this, null, value);
  }

  @Override
  public Tensor div(double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor div(float value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor div(Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor mul(Tensor x, double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor mul(Tensor x, float value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addmm(double v1, Tensor m, double v2, Tensor mat1, Tensor mat2) {
    return DenseTensorMath.addmm(this, v1, m, v2, mat1, mat2);
  }

  @Override
  public Tensor addmm(float v1, Tensor m, float v2, Tensor mat1, Tensor mat2) {
    return DenseTensorMath.addmm(this, v1, m, v2, mat1, mat2);
  }

  @Override
  public Tensor addmm(Tensor m, Tensor mat1, Tensor mat2) {
    if (this.isFloat) {
      return DenseTensorMath.addmm(this, 1.0f, m, 1.0f, mat1, mat2);
    } else {
      return DenseTensorMath.addmm(this, 1.0, m, 1.0, mat1, mat2);
    }
  }

  @Override
  public Tensor addmm(Tensor mat1, Tensor mat2) {
    if (this.isFloat) {
      return DenseTensorMath.addmm(this, 1.0f, this, 1.0f, mat1, mat2);
    } else {
      return DenseTensorMath.addmm(this, 1.0, this, 1.0, mat1, mat2);
    }
  }

  @Override
  public Tensor addmm(double v2, Tensor mat1, Tensor mat2) {
    return DenseTensorMath.addmm(this, 1.0, this, v2, mat1, mat2);
  }

  @Override
  public Tensor addmm(float v2, Tensor mat1, Tensor mat2) {
    return DenseTensorMath.addmm(this, 1.0f, this, v2, mat1, mat2);
  }

  @Override
  public Tensor addmm(double v1, double v2, Tensor mat1, Tensor mat2) {
    return DenseTensorMath.addmm(this, v1, this, v2, mat1, mat2);
  }

  @Override
  public Tensor addmm(float v1, float v2, Tensor mat1, Tensor mat2) {
    return DenseTensorMath.addmm(this, v1, this, v2, mat1, mat2);
  }

  @Override
  public Tensor mm(Tensor mat1, Tensor mat2) {
    if (this.isFloat) {
      return DenseTensorMath.addmm(this, 0.0f, this, 1.0f, mat1, mat2);
    } else {
      return DenseTensorMath.addmm(this, 0.0, this, 1.0, mat1, mat2);
    }
  }

  @Override
  public Tensor addr(Tensor t1, Tensor t2) {
    if (this.isFloat) {
      return DenseTensorMath.addr(this, 1.0f, this, 1.0f, t1, t2);
    } else {
      return DenseTensorMath.addr(this, 1.0, this, 1.0, t1, t2);
    }
  }

  @Override
  public Tensor addr(double v1, Tensor t1, Tensor t2) {
    return DenseTensorMath.addr(this, 1.0, this, v1, t1, t2);
  }

  @Override
  public Tensor addr(float v1, Tensor t1, Tensor t2) {
    return DenseTensorMath.addr(this, 1.0f, this, v1, t1, t2);
  }

  @Override
  public Tensor addr(double v1, Tensor t1, double v2, Tensor t2) {
    return DenseTensorMath.addr(this, v1, this, v2, t1, t2);
  }

  @Override
  public Tensor addr(float v1, Tensor t1, float v2, Tensor t2) {
    return DenseTensorMath.addr(this, v1, this, v2, t1, t2);
  }

  @Override
  public Tensor addr(double v1, Tensor t1, double v2, Tensor t2, Tensor t3) {
    return DenseTensorMath.addr(this, v1, t1, v2, t2, t3);
  }

  @Override
  public Tensor addr(float v1, Tensor t1, float v2, Tensor t2, Tensor t3) {
    return DenseTensorMath.addr(this, v1, t1, v2, t2, t3);
  }

  @Override
  public double uniform(double... args) {
    return 0.0;
  }

  @Override
  public float uniform(float... args) {
    return 0.0f;
  }

  @Override
  public Tensor addmv(double beta, Tensor vec1, double alpha, Tensor mat, Tensor vec2) {
    return DenseTensorMath.addmv(this, beta, vec1, alpha, mat, vec2);
  }

  @Override
  public Tensor addmv(float beta, Tensor vec1, float alpha, Tensor mat, Tensor vec2) {
    return DenseTensorMath.addmv(this, beta, vec1, alpha, mat, vec2);
  }

  @Override
  public Tensor addmv(double beta, double alpha, Tensor mat, Tensor vec2) {
    return DenseTensorMath.addmv(this, beta, this, alpha, mat, vec2);
  }

  @Override
  public Tensor addmv(float beta, float alpha, Tensor mat, Tensor vec2) {
    return DenseTensorMath.addmv(this, beta, this, alpha, mat, vec2);
  }

  @Override
  public Tensor addmv(double alpha, Tensor mat, Tensor vec2) {
    return DenseTensorMath.addmv(this, 1.0, this, alpha, mat, vec2);
  }

  @Override
  public Tensor addmv(float alpha, Tensor mat, Tensor vec2) {
    return DenseTensorMath.addmv(this, 1.0f, this, alpha, mat, vec2);
  }

  @Override
  public Tensor mv(Tensor mat, Tensor vec2) {
    if (this.isFloat) {
      return DenseTensorMath.addmv(this, 1.0f, this, 1.0f, mat, vec2);
    } else {
      return DenseTensorMath.addmv(this, 1.0, this, 1.0, mat, vec2);
    }
  }

  @Override
  public Tensor baddbmm(double beta, Tensor m, double alpha, Tensor batch1, Tensor batch2) {
    return DenseTensorMath.baddbmm(this, beta, m, alpha, batch1, batch2);
  }

  @Override
  public Tensor baddbmm(float beta, Tensor m, float alpha, Tensor batch1, Tensor batch2) {
    return DenseTensorMath.baddbmm(this, beta, m, alpha, batch1, batch2);
  }

  @Override
  public Tensor baddbmm(double beta, double alpha, Tensor batch1, Tensor batch2) {
    return DenseTensorMath.baddbmm(this, beta, this, alpha, batch1, batch2);
  }

  @Override
  public Tensor baddbmm(float beta, float alpha, Tensor batch1, Tensor batch2) {
    return DenseTensorMath.baddbmm(this, beta, this, alpha, batch1, batch2);
  }

  @Override
  public Tensor baddbmm(double alpha, Tensor batch1, Tensor batch2) {
    return DenseTensorMath.baddbmm(this, 1.0, this, alpha, batch1, batch2);
  }

  @Override
  public Tensor baddbmm(float alpha, Tensor batch1, Tensor batch2) {
    return DenseTensorMath.baddbmm(this, 1.0f, this, alpha, batch1, batch2);
  }

  @Override
  public Tensor bmm(Tensor batch1, Tensor batch2) {
    if (this.isFloat) {
      return DenseTensorMath.baddbmm(this, 1.0f, this, 1.0f, batch1, batch2);
    } else {
      return DenseTensorMath.baddbmm(this, 1.0, this, 1.0, batch1, batch2);
    }
  }

  @Override
  public Tensor pow(Tensor y, double n) {
    return DenseTensorMath.pow(this, y, n);
  }

  @Override
  public Tensor pow(Tensor y, float n) {
    return DenseTensorMath.pow(this, y, n);
  }

  @Override
  public Tensor pow(double n) {
    return DenseTensorMath.pow(this, this, n);
  }

  @Override
  public Tensor pow(float n) {
    return DenseTensorMath.pow(this, this, n);
  }

  @Override
  public Tensor square() {
    return pow(2.0);
  }

  @Override
  public Tensor floor(Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor floor() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor ceil() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor inv() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor erf() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor erfc() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor logGamma() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor digamma() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public TensorPair topk(int k, int dim, boolean increase,
                         Tensor result, Tensor indices, boolean sortedResult) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor log(Tensor y) {
    return DenseTensorMath.log(this, y);
  }

  @Override
  public Tensor exp(Tensor y) {
    return DenseTensorMath.exp(this, y);
  }

  @Override
  public Tensor sqrt(Tensor y) {
    return DenseTensorMath.sqrt(this, y);
  }

  @Override
  public Tensor tanh(Tensor y) {
    return DenseTensorMath.tanh(this, y);
  }

  @Override
  public Tensor log1p(Tensor y) {
    return DenseTensorMath.log1p(this, y);
  }

  @Override
  public Tensor log() {
    return DenseTensorMath.log(this, this);
  }

  @Override
  public Tensor exp() {
    return DenseTensorMath.exp(this, this);
  }

  @Override
  public Tensor log1p() {
    return DenseTensorMath.log1p(this, this);
  }

  @Override
  public Tensor abs(Tensor x) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor norm(Tensor y, int value, int dim) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor gt(Tensor x, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor lt(Tensor x, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor le(Tensor x, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor eq(Tensor x, double y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor eq(Tensor x, float y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor maskedFill(Tensor mask, double e) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor maskedFill(Tensor mask, float e) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor maskedCopy(Tensor mask, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor maskedSelect(Tensor mask, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double norm(int value) {
    return 0.0;
  }

  @Override
  public float normf(int value) {
    return 0;
  }

  @Override
  public Tensor sign() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor ge(Tensor x, double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor ge(Tensor x, float value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor indexAdd(int dim, Tensor index, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor index(int dim, Tensor index, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor cmax(Tensor y) {
    return DenseTensorMath.cmax(this, this, y);
  }

  @Override
  public Tensor cmin(Tensor y) {
    return DenseTensorMath.cmin(this, this, y);
  }

  @Override
  public Tensor cmax(Tensor x, Tensor y) {
    return DenseTensorMath.cmax(this, x, y);
  }

  @Override
  public Tensor cmin(Tensor x, Tensor y) {
    return DenseTensorMath.cmin(this, x, y);
  }

  @Override
  public Tensor range(double xmin, double xmax, int step) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor range(float xmin, float xmax, int step) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor negative(Tensor x) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double sumSquare() {
    return 0.0;
  }

  @Override
  public float sumSquaref() {
    return 0;
  }

  @Override
  public Tensor clamp(double min, double max) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor clamp(float min, float max) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  private void resizeAs(DenseTensor self, Tensor src) {
    if (!isSameSizeAs(self, src)) rawResize(self, src.nDimension(), src.size(), null);
  }

  private Tensor resize(DenseTensor self, int[] sizes, int[] strides) {
    if (sizes == null) {
      throw new IllegalStateException("Sizes cannot be null");
    }
    if (strides != null) {
      if (sizes.length == strides.length) {
        throw new IllegalStateException("invalid stride");
      }
    }
    return rawResize(self, sizes.length, sizes, strides);
  }

  private boolean isSameSizeAs(DenseTensor self, Tensor src) {
    if (self.nDimensionInternal != src.nDimension()) {
      return false;
    }

    if (self.isEmpty() != src.isEmpty()) {
      return false;
    }

    int d = 0;
    while (d < self.nDimensionInternal) {
      if (self.size(d + 1) != src.size(d + 1)) {
        return false;
      }
      d += 1;
    }
    return true;
  }

  private DenseTensor rawSet(DenseTensor self, ArrayStorage newStorage, int newStorageOffset,
                             int nDim, int[] newSize, int[] newStride) {
    Util.require(newStorageOffset >= 0, "Tensor: invalid storage offset");
    self.storageInternal = newStorage;
    self.storageOffsetInternal = newStorageOffset;
    return rawResize(self, nDim, newSize, newStride);
  }

  private DenseTensor rawResize(DenseTensor self, int nDim, int[] newSize, int[] newStride) {
    // resize as a scalar
    if (nDim == 0 && newSize.length == 0) {
      self.sizeInternal = new int[0];
      self.strideInternal = new int[0];
      int totalSize = 1;
      if (self.storageInternal == null) {
        self.storageInternal = Util.buildStorage(totalSize + self.storageOffsetInternal,
            self.isFloat);
      } else if (totalSize + self.storageOffsetInternal > self.storageInternal.length()) {
        self.storageInternal.resize(totalSize + self.storageOffsetInternal);
      }
      return self;
    }

    boolean hasCorrectSize = true;
    int nDim_ = 0;
    int d = 0;
    while (d < nDim) {
      nDim_ = nDim_ + 1;
      if (self.sizeInternal == null
          || (self.nDimensionInternal > d && newSize[d] != self.sizeInternal[d])) {
        hasCorrectSize = false;
      }
      if (self.strideInternal == null
          || (self.nDimensionInternal > d && newStride != null && newStride[d] >= 0
          && newStride[d] != self.strideInternal[d])) {
        hasCorrectSize = false;
      }
      d += 1;
    }

    if (nDim_ != self.nDimensionInternal) hasCorrectSize = false;

    if (hasCorrectSize) return self;

    if (nDim_ > 0) {
      if (!hasCorrectSize) {
        self.sizeInternal = new int[nDim];
        self.strideInternal = new int[nDim];
        self.nDimensionInternal = nDim;
      }

      int totalSize = 1;
      d = self.nDimensionInternal - 1;
      while (d >= 0) {
        self.sizeInternal[d] = newSize[d];
        if (newStride != null && newStride[d] >= 0) {
          self.strideInternal[d] = newStride[d];
        } else {
          if (d == self.nDimensionInternal - 1) {
            self.strideInternal[d] = 1;
          } else {
            self.strideInternal[d] = self.sizeInternal[d + 1] * self.strideInternal[d + 1];
          }
        }
        totalSize = totalSize + (self.sizeInternal[d] - 1) * self.strideInternal[d];

        d -= 1;
      }
      if (totalSize + self.storageOffsetInternal > 0) {
        if (self.storageInternal == null) {
          self.storageInternal = Util.buildStorage(totalSize + self.storageOffsetInternal,
              self.isFloat);
        } else if (totalSize + self.storageOffsetInternal > self.storageInternal.length()) {
          self.storageInternal.resize(totalSize + self.storageOffsetInternal);
        }
      }
    } else {
      self.nDimensionInternal = 0;
    }

    return self;
  }

  public Tensor expandTensor(DenseTensor x) {
    int[] targetSize = DenseTensor.expandSize(this, x);
    int[] expandStrides = new int[targetSize.length];

    int[] expandStridesX = new int[targetSize.length];
    int i = targetSize.length - 1;
    int delta2 = targetSize.length - x.nDimension();
    while (i >= delta2) {
      if (x.size(i + 1 - delta2) != 1) expandStridesX[i] = x.stride(i + 1 - delta2);
      i -= 1;
    }
    DenseTensor expandX = new DenseTensor(
        x.storage(),
        x.storageOffset(),
        targetSize,
        expandStridesX
    );
    if (TensorNumeric.product(targetSize) != this.nElement()) {
      i = targetSize.length - 1;
      int delta1 = targetSize.length - this.nDimension();
      while (i >= delta1) {
        if (this.size(i + 1 - delta1) != 1) expandStrides[i] = this.stride(i + 1 - delta1);
        i -= 1;
      }
      DenseTensor tensor1 = new DenseTensor(
          this.storageInternal,
          this.storageOffset(),
          targetSize,
          expandStrides
      );
      DenseTensor newTensor = (DenseTensor) new DenseTensor(x.isFloat())
          .resize(targetSize).add(tensor1);
      this.set(newTensor);
    }
    return expandX;
  }

  private static int[] expandSize(DenseTensor tensor, DenseTensor other) {
    String errorMsg = "tensor size not match ${tensor.size.mkString()} "
        + "${other.size.mkString()}";
    DenseTensor longTensor = (tensor.dim() > other.dim()) ? tensor : other;
    DenseTensor shortTensor = (tensor.dim() > other.dim()) ? other : tensor;
    int ndim = longTensor.nDimension();
    int delta = longTensor.nDimension() - shortTensor.nDimension();
    int[] size = new int[ndim];
    int i = ndim - 1;
    while (i >= delta) {
      Util.require(longTensor.size(i + 1) == shortTensor.size(i + 1 - delta)
          || longTensor.size(i + 1) == 1 || shortTensor.size(i + 1 - delta) == 1, errorMsg);
      size[i] = Math.max(longTensor.size(i + 1), shortTensor.size(i + 1 - delta));
      i -= 1;
    }

    while (i >= 0) {
      size[i] = longTensor.size(i + 1);
      i -= 1;
    }

    return size;
  }
}
