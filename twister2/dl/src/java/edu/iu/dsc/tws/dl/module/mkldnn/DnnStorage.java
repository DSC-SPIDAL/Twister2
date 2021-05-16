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

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import com.intel.analytics.bigdl.mkl.Memory;

import edu.iu.dsc.tws.dl.data.Storage;
import edu.iu.dsc.tws.dl.data.storage.ArrayDoubleStorage;
import edu.iu.dsc.tws.dl.utils.Pointer;
import edu.iu.dsc.tws.dl.utils.Util;

/**
 * Represent a native array which is needed by mkl-dnn
 *
 * @param size Storage size
 * @tparam T data type, only support float now
 */
@SuppressWarnings({"StaticVariableName", "ParameterName", "MemberName"})
public class DnnStorage implements Storage {

  private int sizeInternal;
  private int bytes = 4;
  private boolean _isReleased = false;
  // Hold the address of the native array
  private transient Pointer ptrInternal;

  public static int CACHE_LINE_SIZE = Integer.parseInt(System.getProperty("bigdl.cache.line",
      "64"));
  public static int FLOAT_BYTES = 4;
  public static int INT8_BYTES = 1;
  public static int INT_BYTES = 4;

  private static ConcurrentHashMap<Long, Boolean> nativeStorages = new ConcurrentHashMap();

  public DnnStorage(int size) {
    this.sizeInternal = size;
    this.bytes = DnnStorage.FLOAT_BYTES;
    this.ptrInternal = new Pointer(allocate(size));
  }

  public static boolean checkAndSet(long pointer) {
    return nativeStorages.replace(pointer, false, true);
  }

  public static void add(long key) {
    nativeStorages.put(key, false);
  }

  public static Map<Long, Boolean> get() {
    return nativeStorages;
  }

  public Pointer getPtrInternal() {
    return ptrInternal;
  }

  @Override
  public int length() {
    return sizeInternal;
  }

  @Override
  public void update(int index, double value) {
    throw new UnsupportedOperationException("Not support this operation in DnnStorage");
  }

  @Override
  public void update(int index, float value) {
    throw new UnsupportedOperationException("Not support this operation in DnnStorage");
  }

  @Override
  public Storage copy(Storage source, int offset, int sourceOffset, int length) {
    //TODO Need to complete copy
    if (source instanceof ArrayDoubleStorage) {
      //Util.require(checkIsInstanceOf(ClassTag.Float), "copy from float storage not supported");
//        Memory.CopyArray2Ptr(s.array().asInstanceOf[Array[Float]], sourceOffset,
//            ptr.address, offset, length, bytes)
    } else if (source instanceof DnnStorage) {
//        Memory.CopyPtr2Ptr(s.ptr.address, sourceOffset, ptr.address, offset, length,
//            bytes)
    } else {
      throw new UnsupportedOperationException("Only support copy from ArrayStorage or DnnStorage");
    }
    return null;
  }

  @Override
  public Storage fill(double value, int offset, int length) {
    throw new UnsupportedOperationException("Not support this operation in DnnStorage");
  }

  @Override
  public Storage fill(float value, int offset, int length) {
    throw new UnsupportedOperationException("Not support this operation in DnnStorage");
  }

  @Override
  public Storage resize(int size) {
    throw new UnsupportedOperationException("Not support this operation in DnnStorage");
  }

  @Override
  public double[] toDoubleArray() {
    return new double[0];
  }

  @Override
  public float[] toFloatArray() {
    return new float[0];
  }

  @Override
  public Storage set(Storage other) {
    throw new UnsupportedOperationException("Not support this operation in DnnStorage");
  }

  @Override
  public Iterator iterator() {
    throw new UnsupportedOperationException("Not support this operation in DnnStorage");
  }

  private boolean checkIsInstanceOf(Object that) {
    //TODO check if type check is needed
    //scala.reflect.classTag[T] == that
    return true;
  }

  /**
   * Release the native array, the storage object is useless
   */
  public synchronized void release() {
    if (!this.isReleased() && ptrInternal.address() != 0L) {
      Memory.AlignedFree(ptrInternal.address());
      DnnStorage.checkAndSet(ptrInternal.address());
      _isReleased = true;
      ptrInternal = null;
    }
  }

  public boolean isReleased() {
    return _isReleased;
  }

  private long allocate(int capacity) {
    Util.require(capacity > 0, "capacity should be larger than 0");
    long ptr = Memory.AlignedMalloc(capacity * bytes, DnnStorage.CACHE_LINE_SIZE);
    Util.require(ptr != 0L, "allocate native aligned memory failed");
    _isReleased = false;
    DnnStorage.add(ptr);
    return ptr;
  }

  private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
    in.defaultReadObject();
    if (!_isReleased) {
      ptrInternal = new Pointer(allocate(this.sizeInternal));
      float[] elements = (float[]) in.readObject();
      Memory.CopyArray2Ptr(elements, 0, ptrInternal.address(), 0, sizeInternal,
          DnnStorage.FLOAT_BYTES);
    }
  }

  private void writeObject(ObjectOutputStream out) throws IOException, ClassNotFoundException {
    out.defaultWriteObject();
    if (!_isReleased) {
      float[] elements = new float[this.length()];
      Memory.CopyPtr2Array(this.ptrInternal.address(), 0, elements, 0, sizeInternal,
          DnnStorage.FLOAT_BYTES);
      out.writeObject(elements);
    }
  }
}
