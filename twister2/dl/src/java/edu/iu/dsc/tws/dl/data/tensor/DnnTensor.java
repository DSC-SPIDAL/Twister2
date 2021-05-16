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
import java.util.Objects;

import com.intel.analytics.bigdl.mkl.Memory;

import edu.iu.dsc.tws.dl.data.Storage;
import edu.iu.dsc.tws.dl.data.Table;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.TensorNumeric;
import edu.iu.dsc.tws.dl.data.TensorType;
import edu.iu.dsc.tws.dl.module.mkldnn.DnnStorage;
import edu.iu.dsc.tws.dl.module.mkldnn.MemoryOwner;
import edu.iu.dsc.tws.dl.module.mkldnn.Releasable;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.TensorPair;

@SuppressWarnings({"ChainingConstructorIgnoresParameter", "NeedBraces",
    "LocalVariableName", "NoClone", "SuperClone"})
public class DnnTensor implements Tensor, Releasable {

  private DnnStorage storageInternal;
  private int[] sizes;
  private MemoryOwner owner;

  private int numElements;

  public DnnTensor(int[] size, MemoryOwner owner) {
    this(new DnnStorage(TensorNumeric.product(size)), size, owner);
  }

  public DnnTensor(int[] size, long realSize, MemoryOwner owner) {
    this(new DnnStorage((int) realSize), size, owner);
  }

  public DnnTensor(int d1, MemoryOwner owner) {
    this(new DnnStorage(d1), new int[]{d1}, owner);
  }

  public DnnTensor(int d1, int d2, MemoryOwner owner) {
    this(new DnnStorage(d1 * d2), new int[]{d1, d2}, owner);
  }

  public DnnTensor(int d1, int d2, int d3, MemoryOwner owner) {
    this(new DnnStorage(d1 * d2 * d3), new int[]{d1, d2, d3}, owner);
  }

  public DnnTensor(DnnStorage storageInternal, int[] sizes, MemoryOwner owner) {
    this.storageInternal = storageInternal;
    this.sizes = sizes;
    this.owner = owner;
    this.numElements = TensorNumeric.product(sizes);
  }

  public DnnStorage getStorageInternal() {
    return storageInternal;
  }

  @Override
  public boolean isFloat() {
    //Only Float is supported for DnnTensor
    return true;
  }

  @Override
  public boolean isEmpty() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public boolean isScalar() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public int nDimension() {
    return sizes.length;
  }

  @Override
  public int dim() {
    return sizes.length;
  }

  @Override
  public int[] size() {
    return sizes.clone();
  }

  @Override
  public int size(int dim) {
    return sizes[dim - 1];
  }

  @Override
  public int[] stride() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public int stride(int dim) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor fill(double v) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor fill(float v) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor forceFill(double v) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor forceFill(float v) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor zero() {
    Memory.Zero(this.storageInternal.getPtrInternal().getPtr(), this.nElement(),
        DnnStorage.FLOAT_BYTES);
    return this;
  }

  public void axpby(float a, float b, DnnTensor to) {
    long x = this.storageInternal.getPtrInternal().getPtr();
    long y = to.getStorageInternal().getPtrInternal().getPtr();
    Memory.Axpby(this.nElement(), a, x, b, y);
  }

  public void scale(DnnTensor from, float scal) {
    int length = this.nElement();
    Memory.Scale(length, scal, from.getStorageInternal().getPtrInternal().getPtr(),
        this.storageInternal.getPtrInternal().getPtr());
  }

  @Override
  public Tensor randn() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor randn(double mean, double stdv) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor randn(float mean, float stdv) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor rand() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor rand(double lowerBound, double upperBound) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor rand(float lowerBound, float upperBound) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor bernoulli(double p) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor bernoulli(float p) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor transpose(int dim1, int dim2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor t() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor apply(int index) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double apply(int[] indexes) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float applyf(int[] indexes) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double value() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double valueAt(int d1) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double valueAt(int d1, int d2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double valueAt(int d1, int d2, int d3) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double valueAt(int d1, int d2, int d3, int d4) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double valueAt(int d1, int d2, int d3, int d4, int d5) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float valueAtf(int d1) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float valueAtf(int d1, int d2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float valueAtf(int d1, int d2, int d3) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float valueAtf(int d1, int d2, int d3, int d4) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float valueAtf(int d1, int d2, int d3, int d4, int d5) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor apply(Table t) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public void update(int index, double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public void update(int index, float value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public void update(int index, Tensor src) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public void update(int[] indexes, double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public void update(int[] indexes, float value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor setValue(double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor setValue(float value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor setValue(int d1, double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor setValue(int d1, float value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor setValue(int d1, int d2, double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor setValue(int d1, int d2, float value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor setValue(int d1, int d2, int d3, double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor setValue(int d1, int d2, int d3, float value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor setValue(int d1, int d2, int d3, int d4, double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor setValue(int d1, int d2, int d3, int d4, float value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor setValue(int d1, int d2, int d3, int d4, int d5, double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor setValue(int d1, int d2, int d3, int d4, int d5, float value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public void update(Table t, double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public void update(Table t, float value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public void update(Table t, Tensor src) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public boolean isContiguous() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor contiguous() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public boolean isSameSizeAs(Tensor other) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor emptyInstance() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor resizeAs(Tensor src) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor resize(int[] size, int[] strides) {
    Util.require(strides == null, "dnn tensor doesn't have stride");
    if (TensorNumeric.product(size) > nElement()) {
      storageInternal.release();
      storageInternal = new DnnStorage(TensorNumeric.product(size));
    }
    this.sizes = size.clone();
    return this;
  }

  @Override
  public Tensor resize(int[] size) {
    if (TensorNumeric.product(size) > nElement()) {
      storageInternal.release();
      storageInternal = new DnnStorage(TensorNumeric.product(size));
    }
    this.sizes = size.clone();
    return this;
  }

  @Override
  public Tensor resize(int size1) {
    if (size1 > nElement()) {
      storageInternal.release();
      storageInternal = new DnnStorage(size1);
    }
    this.sizes = new int[]{size1};
    return this;
  }

  @Override
  public Tensor resize(int size1, int size2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor resize(int size1, int size2, int size3) {
    throw new UnsupportedOperationException("Operation not supported");
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
    return numElements;
  }

  @Override
  public Tensor select(int dim, int index) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor set(Tensor other) {
    Util.require(other instanceof DnnTensor, "only support to set DnnTensor");
    this.storageInternal.release();
    this.storageInternal = (DnnStorage) other.storage();
    return this;
  }

  @Override
  public Tensor set(Storage storage, int storageOffset, int[] size, int[] strides) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor set() {
    return this;
  }

  @Override
  public Storage storage() {
    return storageInternal;
  }

  @Override
  public int storageOffset() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor narrow(int dim, int index, int size) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor copy(Tensor other) {
    if (other instanceof DenseTensor) {
      Util.require(noTransposed((DenseTensor) other), "dense tensor should not be transposed");
      Util.require(this.nElement() == other.nElement(), "tensor elements number must be same");
      this.storageInternal.copy(other.storage(), 0, other.storageOffset() - 1, this.nElement());
    } else if (other instanceof DnnTensor) {
      Util.require(this.nElement() == other.nElement(), "tensor elements number must be same");
      this.storageInternal.copy(other.storage(), 0, 0, this.nElement());
    } else {
      throw new UnsupportedOperationException("Only support copy from dense tensor and dnn tensor");
    }
    return this;
  }

  private boolean noTransposed(DenseTensor t) {
    int product = 1;
    int i = t.dim();
    while (i > 0) {
      if (product != t.stride(i)) {
        return false;
      }
      product *= t.size(i);
      i -= 1;
    }
    return true;
  }

  @Override
  public Tensor squeeze() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor squeeze(int dim) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor squeezeNewTensor() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor view(int[] size) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor unfold(int dim, int size, int step) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor repeatTensor(int[] size) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor expandAs(Tensor template) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor expand(int[] size) {
    throw new UnsupportedOperationException("Operation not supported");
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
    throw new UnsupportedOperationException("Operation not supported");
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
  public Tensor reshape(int[] size) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor save(String path, boolean overWrite) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public TensorNumeric getTensorNumeric() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public TensorType getTensorType() {
    return TensorType.MklDnnType;
  }

  @Override
  public double[] toArray() {
    return new double[0];
  }

  @Override
  public float[] toFloatArray() {
    return new float[0];
  }

  @Override
  public DnnTensor toTensor() {
    return this;
  }

  @Override
  public Table toTable() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addCopy(double s) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addCopy(float s) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addCopy(Tensor t) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor subCopy(double s) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor subCopy(float s) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor subCopy(Tensor t) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor negative() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor divCopy(double s) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor divCopy(float s) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor divCopy(Tensor t) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor mulCopy(double s) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor mulCopy(float s) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor mulCopy(Tensor t) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double sum() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float sumf() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double prod() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float prodf() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor prod(Tensor x, int dim) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor sum(int dim) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor sum(Tensor x, int dim) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double mean() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float meanf() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor mean(int dim) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double max() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float maxf() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public TensorPair max(int dim) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public TensorPair max(Tensor values, Tensor indices, int dim) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double min() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float minf() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public TensorPair min(int dim) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public TensorPair min(Tensor values, Tensor indices, int dim) {
    throw new UnsupportedOperationException("Operation not supported");
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
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor tanh() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor abs() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor add(double value, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor add(float value, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor add(Tensor x) {
    Util.require(x instanceof DnnTensor, "Just support two dnn tensor add");
    Memory.SAdd(this.nElement(), this.storageInternal.getPtrInternal().getPtr(), 0,
        ((DnnTensor) x).storageInternal.getPtrInternal().getPtr(),
        0, this.storageInternal.getPtrInternal().getPtr(), 0);
    return this;
  }

  @Override
  public Tensor add(Tensor x, double value, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor add(Tensor x, float value, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor add(double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor add(float value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor add(Tensor x, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double dot(Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float dotf(Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
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
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float distf(Tensor y, int norm) {
    throw new UnsupportedOperationException("Operation not supported");
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
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addcdiv(float value, Tensor tensor1, Tensor tensor2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor sub(double value, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor sub(float value, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor sub(Tensor x, double value, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor sub(Tensor x, float value, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor sub(Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor sub(Tensor x, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor sub(double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor sub(float value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor cmul(Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor cmul(Tensor x, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor cdiv(Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor cdiv(Tensor x, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor mul(double value) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor mul(float value) {
    throw new UnsupportedOperationException("Operation not supported");
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
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addmm(float v1, Tensor m, float v2, Tensor mat1, Tensor mat2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addmm(Tensor m, Tensor mat1, Tensor mat2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addmm(Tensor mat1, Tensor mat2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addmm(double v2, Tensor mat1, Tensor mat2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addmm(float v2, Tensor mat1, Tensor mat2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addmm(double v1, double v2, Tensor mat1, Tensor mat2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addmm(float v1, float v2, Tensor mat1, Tensor mat2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor mm(Tensor mat1, Tensor mat2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addr(Tensor t1, Tensor t2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addr(double v1, Tensor t1, Tensor t2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addr(float v1, Tensor t1, Tensor t2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addr(double v1, Tensor t1, double v2, Tensor t2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addr(float v1, Tensor t1, float v2, Tensor t2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addr(double v1, Tensor t1, double v2, Tensor t2, Tensor t3) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addr(float v1, Tensor t1, float v2, Tensor t2, Tensor t3) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public double uniform(double... args) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float uniform(float... args) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addmv(double beta, Tensor vec1, double alpha, Tensor mat, Tensor vec2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addmv(float beta, Tensor vec1, float alpha, Tensor mat, Tensor vec2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addmv(double beta, double alpha, Tensor mat, Tensor vec2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addmv(float beta, float alpha, Tensor mat, Tensor vec2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addmv(double alpha, Tensor mat, Tensor vec2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor addmv(float alpha, Tensor mat, Tensor vec2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor mv(Tensor mat, Tensor vec2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor baddbmm(double beta, Tensor m, double alpha, Tensor batch1, Tensor batch2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor baddbmm(float beta, Tensor m, float alpha, Tensor batch1, Tensor batch2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor baddbmm(double beta, double alpha, Tensor batch1, Tensor batch2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor baddbmm(float beta, float alpha, Tensor batch1, Tensor batch2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor baddbmm(double alpha, Tensor batch1, Tensor batch2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor baddbmm(float alpha, Tensor batch1, Tensor batch2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor bmm(Tensor batch1, Tensor batch2) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor pow(Tensor y, double n) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor pow(Tensor y, float n) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor pow(double n) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor pow(float n) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor square() {
    throw new UnsupportedOperationException("Operation not supported");
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
  public TensorPair topk(int k, int dim, boolean increase, Tensor result,
                         Tensor indices, boolean sortedResult) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor log(Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor exp(Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor sqrt(Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor tanh(Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor log1p(Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor log() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor exp() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor log1p() {
    throw new UnsupportedOperationException("Operation not supported");
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
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float normf(int value) {
    throw new UnsupportedOperationException("Operation not supported");
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
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor cmin(Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor cmax(Tensor x, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor cmin(Tensor x, Tensor y) {
    throw new UnsupportedOperationException("Operation not supported");
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
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public float sumSquaref() {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor clamp(double min, double max) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public Tensor clamp(float min, float max) {
    throw new UnsupportedOperationException("Operation not supported");
  }

  @Override
  public void release() {
  }

  public long storageAddress() {
    return storageInternal.getPtrInternal().getPtr();
  }

  public boolean isReleased() {
    return storageInternal.isReleased();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    DnnTensor dnnTensor = (DnnTensor) o;
    return numElements == dnnTensor.numElements
        && Objects.equals(storageInternal, dnnTensor.storageInternal)
        && Arrays.equals(sizes, dnnTensor.sizes)
        && Objects.equals(owner, dnnTensor.owner);
  }

  @Override
  public int hashCode() {
    int seed = 37;
    int hash = 1;
    hash = hash * seed + this.nDimension();
    int d = 1;
    while (d <= this.nDimension()) {
      hash = hash * seed + this.size(d);
      d += 1;
    }

    hash = hash * seed + this.storageInternal.getPtrInternal().hashCode();

    return hash;
  }

  @Override
  public String toString() {
    if (TensorNumeric.product(size()) != this.nElement()) {
      DenseTensor dense = new DenseTensor(new int[]{this.nElement()}, true);
      Memory.CopyPtr2Array(this.storageAddress(), 0, dense.storage().toFloatArray(),
          0, nElement(), 4);
      return dense.toString();
    } else {
      DenseTensor dense = new DenseTensor(size(), true);
      dense.copy(this);
      return dense.toString();
    }
  }

  @Override
  public Tensor clone() {
    return null;
  }
}
