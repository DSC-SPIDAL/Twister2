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
package edu.iu.dsc.tws.dl.module.mkldnn;

import java.io.Serializable;

import com.intel.analytics.bigdl.mkl.Memory;

import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.data.tensor.DnnTensor;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.HeapData;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.NativeData;
import edu.iu.dsc.tws.dl.utils.Util;

/**
 * `TensorMMap` contains two tensors, dense and native, which are a map of each other.
 * It's used in the layer which contains weights. For the weight, we should sync the
 * dense tensor to native tensor before `submit`. For the gradient, we should sync the native
 * tensor to dense tensor after `submit`.
 *
 * @param _size the shape of Tensor, such as Array(4, 3, 224, 224)
 */
@SuppressWarnings({"MemberName", "ParameterName"})
public class TensorMMap implements Serializable {
  private int[] _size;
  private MemoryOwner owner;
  // dense weight on heap is used to optimizer and so on, which is exposed to
  // AbstractModule level.
  private Tensor dense;

  // the native DnnTensor. It's allocate at runtime when do primitive initialization.
  // it has two benefits, the first is the clone will only clone one copy of weights and gradients
  // before primitive initialized and the second is that we can determined the size, type, format
  // when do initializing primitive.
  private transient DnnTensor _native = null;

  private transient MemoryData _from = null;
  private transient MemoryData _to = null;
  private transient ReorderMemory _reorder = null;

  private transient HeapData _heapData = null;

  public TensorMMap(int[] _size, MemoryOwner owner) {
    this._size = _size;
    this.owner = owner;
    dense = new DenseTensor(_size, true);
  }

  public DnnTensor nativeDnn() {
    return (DnnTensor) _native;
  }

  public HeapData heapData() {
    return _heapData;
  }

  public void sync() {
    Util.require(_reorder != null && _native != null,
        "you should initialize the native relevant resources first");

    if (_from instanceof HeapData) {
      _reorder.forward(this.dense);
    } else if (_from instanceof NativeData) {
      _reorder.forward(this.nativeDnn());
    }
  }

  public Tensor getDense() {
    return dense;
  }

  public void setDense(Tensor dense) {
    this.dense = dense;
  }

  /**
   * set the dense <-> native map, maintain the format to reorder
   * <p>
   * Note, it will only create the native tensor based on the size and will not
   * do the reorder. So you should call `sync()` by manual.
   *
   * @param from    the source tensor memory data, could be HeapData or NativeData
   * @param to      the dest tensor memory data, could be HeapData or NativeData
   * @param runtime the mkldnn runtime for reorder operation
   */
  public void setMemoryData(MemoryData from, MemoryData to, MklDnnRuntime runtime) {
    Util.require(_from == null && _to == null, "you only can set once the memory data");
    _from = from;
    _to = to;

    _reorder = new ReorderMemory(to, owner);
    _reorder.setRuntime(runtime);
    _reorder.initFwdPrimitives(new MemoryData[]{_from}, Phase.INFERENCE);

    if (_from instanceof HeapData) {
      this._native = (DnnTensor) _reorder.output;
      _heapData = (HeapData) _from;
    } else if (_from instanceof NativeData) {
      // the native tensor size should be determined by the memory description
      // other wise will be segment fault
      this._native = new DnnTensor(Memory.GetPaddingShape(_from.getMemoryDescription(owner)),
          owner);
      // the native initialize value should be all zeros.
      this._native.zero();
      ((Tensor) _reorder.output).set(this.dense);
      _heapData = (HeapData) _to;
    } else {
      throw new UnsupportedOperationException("Not support such memory format");
    }
  }

  public void zero() {
    dense.zero();
    if (nativeDnn() != null) {
      nativeDnn().zero();
    }
  }

  public void copy(Tensor t) {
    dense.copy(t);
  }

  public int[] size() {
    return dense.size();
  }

  public int size(int index) {
    return dense.size(index);
  }

  public void release() {
    if (nativeDnn() != null) {
      nativeDnn().release();
    }
  }

  public void setNative(TensorMMap another) {
    if (nativeDnn() != null && another.nativeDnn() != null) {
      nativeDnn().set(another.nativeDnn());
    }
  }

}
