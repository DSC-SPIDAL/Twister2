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
package edu.iu.dsc.tws.dl.module.mkldnn.memory.data;

import java.util.Arrays;

import com.intel.analytics.bigdl.mkl.DataType;

import edu.iu.dsc.tws.dl.module.mkldnn.MemoryData;

@SuppressWarnings({"LocalVariableName", "ParameterName", "MemberName"})
public class HeapData extends MemoryData {
  private int[] _shape;
  private int _layout;
  private int _dataType = DataType.F32;

  public HeapData(int[] _shape, int _layout) {
    this._shape = _shape;
    this._layout = _layout;
  }

  public HeapData(int[] _shape, int _layout, int _dataType) {
    this._shape = _shape;
    this._layout = _layout;
    this._dataType = _dataType;
  }

  @Override
  public int[] shape() {
    return _shape.clone();
  }

  @Override
  public int layout() {
    return _layout;
  }

  @Override
  public int dataType() {
    return _dataType;
  }

  @Override
  public MemoryData cloneFormat() {
    return new HeapData(_shape, _layout, _dataType);
  }

  @Override
  public int hashCode() {
    int seed = 37;
    int hash = 1;
    hash = hash * seed + this.layout();
    int d = 0;
    while (d < this.shape().length) {
      hash = hash * seed + this.shape()[d];
      d += 1;
    }

    hash = hash * seed + this.dataType();

    return hash;
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof HeapData)) {
      return false;
    }
    HeapData other = (HeapData) obj;
    if (this == other) {
      return true;
    }
    if (this.layout() != other.layout()) {
      return false;
    }
    if (this.shape() == null && other.shape() == null) {
      return true;
    }
    if (this.shape() != null && other.shape() != null) {
      if (this.shape().length != other.shape().length) {
        return false;
      }
      int i = 0;
      while (i < this.shape().length) {
        if (this.shape()[i] != other.shape()[i]) {
          return false;
        }
        i += 1;
      }
      return true;
    } else {
      return false;
    }
  }

  @Override
  public String toString() {
    return "HeapData([" + Arrays.toString(_shape) + "], " + layout() + ")";
  }

  public NativeData toNative() {
    return new NativeData(shape(), layout(), dataType());
  }
}
