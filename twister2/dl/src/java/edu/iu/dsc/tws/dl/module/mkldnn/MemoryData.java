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

import com.intel.analytics.bigdl.mkl.DataType;
import com.intel.analytics.bigdl.mkl.Memory;
import com.intel.analytics.bigdl.mkl.MklDnn;
import com.intel.analytics.bigdl.mkl.Query;

import edu.iu.dsc.tws.dl.module.mkldnn.memory.MklDnnMemory;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.NativeData;
import edu.iu.dsc.tws.dl.utils.Util;

@SuppressWarnings({"LocalVariableName", "ParameterName", "MemberName"})
public abstract class MemoryData implements Serializable {

  private int heapFormat = -1;

  private int _mask = -1;
  private float[] _scales = new float[0];


  private long UNDEFINED = -1;
  private long ERROR = 0;

  private transient long primitive = UNDEFINED;
  private transient long primitiveDesc = UNDEFINED;
  private transient long description = UNDEFINED;

  public static NativeData primitiveOutput(long pd) {
    return operationWant(pd, Query.DstPd, 0);
  }

  public static NativeData operationWant(long primDesc, int queryType) {
    return operationWant(primDesc, queryType, 0);
  }

  public static NativeData operationWant(long primDesc, int queryType, int index) {
    long memoryPrimDesc = MklDnn.PrimitiveDescQueryPd(primDesc, queryType, index);
    long memoryDesc = MklDnn.PrimitiveDescQueryMemory(memoryPrimDesc);
    int[] shape = Memory.GetShape(memoryDesc);
    int[] paddingShape = Memory.GetPaddingShape(memoryDesc);
    int layout = Memory.GetLayout(memoryDesc);
    int dataType = Memory.GetDataType(memoryDesc);

    NativeData memory = new NativeData(shape, layout, dataType);
    memory.setMemoryDescription(memoryDesc);
    memory.setPrimitiveDescription(memoryPrimDesc);
    return memory;
  }

  public int getHeapFormat() {
    return heapFormat;
  }

  public abstract int[] shape();

  public abstract int layout();

  public abstract int dataType();

  public int mask() {
    return _mask;
  }

  public void setMask(int s) {
    _mask = s;
  }

  public float[] scales = _scales;

  public void setScales(float[] f) {
    _scales = f;
  }

  public MemoryData setHeapFormat(int f) {
    heapFormat = f;
    return this;
  }

  public int[] getHeapShape() {
    if (layout() == Memory.Format.nhwc) { // native shape is nchw
      return new int[]{shape()[0], shape()[2], shape()[3], shape()[1]};
    } else {
      return shape();
    }
  }

  public abstract MemoryData cloneFormat();

  public long getMemoryDescription(MemoryOwner owner) {
    if (description == UNDEFINED || description == ERROR) {
      checkConsistency(shape(), layout());
      description = MklDnnMemory.MemoryDescInit(shape().length, shape(), dataType(), layout(),
          owner);
    }
    return description;
  }

  public long getPrimitiveDescription(MklDnnRuntime runtime, MemoryOwner owner) {
    Util.require(runtime != null, "Have you initialized the MklDnnRuntime?");
    if (primitiveDesc == UNDEFINED || primitiveDesc == ERROR) {
      primitiveDesc =
          MklDnnMemory.MemoryPrimitiveDescCreate(getMemoryDescription(owner), runtime.engine,
              owner);
    }
    return primitiveDesc;
  }

  public long getPrimitive(MklDnnRuntime runtime, MemoryOwner owner) {
    Util.require(runtime != null, "Have you initialized the MklDnnRuntime?");
    if (primitive == UNDEFINED || primitive == ERROR) {
      primitive =
          MklDnnMemory.PrimitiveCreate0(getPrimitiveDescription(runtime, owner), owner);
    }
    return primitive;
  }

  public void setPrimitiveDescription(long desc) {
    primitiveDesc = desc;
  }

  public void setMemoryDescription(long desc) {
    description = desc;
  }

  public long getRealSize() {
    Util.require(primitiveDesc != UNDEFINED && primitiveDesc != ERROR);
    return MklDnn.PrimitiveDescGetSize(primitiveDesc) / getDataTypeBytes();
  }

  public int[] getPaddingShape() {
    Util.require(description != UNDEFINED && description != ERROR);
    return Memory.GetPaddingShape(description);
  }

  private int getDataTypeBytes() {
    switch (dataType()) {
      case DataType.F32:
        return DnnStorage.FLOAT_BYTES;
      case DataType.S32:
        return DnnStorage.INT_BYTES;
      case DataType.S8:
        return DnnStorage.INT8_BYTES;
      case DataType.U8:
        return DnnStorage.INT8_BYTES;
      default:
        throw new UnsupportedOperationException("unsupported data type");
    }
  }

  private void checkConsistency(int[] shape, int layout) {
    boolean switchCheck = false;
    switch (shape.length) {
      case 1:
        switchCheck = layout == Memory.Format.x;
        break;
      case 2:
        switchCheck = layout == Memory.Format.nc || layout == Memory.Format.io
            || layout == Memory.Format.oi;
        break;
      case 3:
      case 4:
      case 5:
        switchCheck = layout != Memory.Format.nc || layout != Memory.Format.x;
        break;
      default:
        switchCheck = false;
    }
    boolean isConsistency = Memory.Format.any == layout || switchCheck;
    Util.require(isConsistency,
        "the shape([${shape.mkString(',')}]) of tensor is different from layout(${layout})");
  }
}

