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

import java.util.HashMap;
import java.util.Map;

import com.intel.analytics.bigdl.mkl.DataType;

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.data.Table;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.HeapData;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.NativeData;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.ReorderManagerKey;

public class ReorderManager {
  private MemoryOwner owner;
  // (MemoryFormatId, TargetFormat) -> Reorder

  private Map<ReorderManagerKey, ReorderMemory> reorders = new HashMap<>();
  // ReorderId -> RefCount
  private Map<Integer, Integer> refCounts = new HashMap<>();
  private Map<Integer, Integer> useCounts = new HashMap<>();

  private MklDnnRuntime runtime;

  public ReorderManager(MemoryOwner owner) {
    this.owner = owner;
  }

  public void register(MemoryData from, MemoryData to) {
    Util.require(runtime != null, "Please call setRuntime first");
    int mId = System.identityHashCode(from);
    ReorderManagerKey tempKey = new ReorderManagerKey(mId, to);
    if (needReorder(from, to)) {
      if (reorders.containsKey(new ReorderManagerKey(mId, to))) {
        int temp = refCounts.get(System.identityHashCode(reorders.get(tempKey)));
        refCounts.put(System.identityHashCode(reorders.get(tempKey)), temp + 1);
      } else {
        ReorderMemory reorder = new ReorderMemory(to, owner);
        reorder.setRuntime(runtime);
        reorder.initFwdPrimitives(new MemoryData[]{from}, Phase.INFERENCE);
        reorders.put(tempKey, reorder);
        int reorderId = System.identityHashCode(reorder);
        refCounts.put(reorderId, 1);
        useCounts.put(reorderId, 0);
      }
    }
  }

  public void setRuntime(MklDnnRuntime runtim) {
    this.runtime = runtim;
  }

  public Activity infer(MemoryData[] from, MemoryData[] to, Activity output) {
    if (from.length == 1) {
      Util.require(output.isTensor(), "output activity should be a tensor");
      return inferTensor(from[0], to[0], (Tensor) output);
    } else {
      Util.require(output.toTable().length() == from.length,
          "output activity length doesn't match");
      Table outputTable = new Table();
      int i = 0;
      while (i < from.length) {
        outputTable.put(i + 1, inferTensor(from[i], to[i], output.toTable().get(i + 1)));
        i += 1;
      }
      return outputTable;
    }
  }

  private Tensor inferTensor(MemoryData from, MemoryData to, Tensor output) {
    int mId = System.identityHashCode(from);
    ReorderManagerKey tempKey = new ReorderManagerKey(mId, to);

    if (reorders.containsKey(tempKey)) {
      ReorderMemory reorder = reorders.get(tempKey);
      int reorderId = System.identityHashCode(reorder);
      Activity result;
      if (useCounts.get(reorderId) == 0) {
        result = reorder.forward(output);
      } else {
        result = reorder.output;
      }
      useCounts.put(reorderId, useCounts.get(reorderId) + 1);
      if (useCounts.get(reorderId) == refCounts.get(reorderId)) {
        useCounts.put(reorderId, 0);
      }
      return (Tensor) result;
    } else {
      return output;
    }
  }

  private boolean needReorder(MemoryData from, MemoryData to) {
    if (from instanceof HeapData) {
      if (to instanceof HeapData) {
        return from.layout() != to.layout();
      } else if (to instanceof NativeData) {
        return true;
      } else {
        throw new UnsupportedOperationException("Not support such memory format");
      }
    } else if (from instanceof NativeData) {
      if (to instanceof HeapData) {
        return true;
      } else if (to instanceof NativeData) {
        boolean doNotReorderIt = from.layout() == to.layout()
            && (from.dataType() == to.dataType()
            || (from.dataType() == DataType.S8 && to.dataType() == DataType.U8)
            || (from.dataType() == DataType.U8 && to.dataType() == DataType.S8)); // skip the s8->u8

        return !doNotReorderIt;
      } else {
        throw new UnsupportedOperationException("Not support such memory format");
      }
    } else {
      throw new UnsupportedOperationException("Not support such memory format");
    }
  }

  public void release() {

  }
}
