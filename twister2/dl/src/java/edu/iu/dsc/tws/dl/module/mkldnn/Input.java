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

import java.util.Arrays;

import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.HeapData;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.NativeData;

public class Input extends ReorderMemory {

  private int[] shapeInternal;
  private int layoutInternal;

  public Input(int[] shape, int layout) {
    super(new HeapData(shape, layout), new NativeData(shape, layout),
        new HeapData(shape, layout), new NativeData(shape, layout), null);
  }

  public Input(int[] shape, int layout, MemoryOwner owner) {
    super(new HeapData(shape, layout), new NativeData(shape, layout),
        new HeapData(shape, layout), new NativeData(shape, layout), owner);
  }

  @Override
  public String toString() {
    return "Input{"
        + "shapeInternal=" + Arrays.toString(shapeInternal)
        + ", layoutInternal=" + layoutInternal
        + '}';
  }
}
