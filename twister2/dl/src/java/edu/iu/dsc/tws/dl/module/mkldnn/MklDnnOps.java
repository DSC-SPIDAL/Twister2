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

import com.intel.analytics.bigdl.mkl.Memory;
import com.intel.analytics.bigdl.mkl.MklDnn;
import com.intel.analytics.bigdl.mkl.Stream;

import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.tensor.DnnTensor;
import edu.iu.dsc.tws.dl.utils.Util;

@SuppressWarnings("ParameterName")
public final class MklDnnOps {

  private MklDnnOps() {
  }

  public static long memorySetDataHandle(long memory, Tensor data, int offset) {
    Util.require(MklDnn.isLoaded(), "mkldnn isn't loaded");
    return MklDnn.MemorySetDataHandle(memory, data.storage().toFloatArray(), offset);
  }

  public static void memoryReleaseDataHandle(Tensor data, long ptr) {
    Util.require(MklDnn.isLoaded(), "mkldnn isn't loaded");
    MklDnn.MemoryReleaseDataHandle(data.storage().toFloatArray(), ptr);
  }

  public static void streamSubmit(long loc, int block, long[] primitives, int length,
                                  long[] memory_primitives,
                                  Tensor[] buffers) {
    // the tensor maybe Tensor[Byte]. so use the unchecked to handle this
    Util.require(MklDnn.isLoaded(), "mkldnn isn't loaded");
    Util.require(memory_primitives.length == buffers.length);

    long[] handle = new long[memory_primitives.length];
    for (int i = 0; i < memory_primitives.length; i++) {
      if (memory_primitives[i] != 0L) {
        if (buffers[i] instanceof DnnTensor) {
          Memory.SetDataHandle(memory_primitives[i],
              ((DnnTensor) buffers[i]).storageAddress(), 0);
        } else {
          handle[i] = MklDnnOps.memorySetDataHandle(
              memory_primitives[i], buffers[i], buffers[i].storageOffset() - 1);
        }
      }
    }

    Stream.Submit(loc, block, primitives);

    for (int i = 0; i < memory_primitives.length; i++) {
      if (handle[i] != 0L) {
        MklDnnOps.memoryReleaseDataHandle(buffers[i], handle[i]);
      }
    }
  }
}

