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
package edu.iu.dsc.tws.dl.module.mkldnn.memory;

import com.intel.analytics.bigdl.mkl.DataType;

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.data.Table;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.data.tensor.DnnTensor;
import edu.iu.dsc.tws.dl.module.mkldnn.MemoryData;
import edu.iu.dsc.tws.dl.module.mkldnn.MemoryOwner;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.HeapData;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.NativeData;
import edu.iu.dsc.tws.dl.utils.Util;

public interface MklDnnModuleHelper extends MemoryOwner {

  default Activity initActivity(MemoryData[] formats) {
    if (formats.length == 1) {
      return initTensor(formats[0]);
    } else {
      Tensor[] tensors = new Tensor[formats.length];
      for (int i = 0; i < tensors.length; i++) {
        tensors[i] = initTensor(formats[i]);
      }
      return new Table(tensors);
    }
  }

  default Tensor initTensor(MemoryData format) {
    int[] paddingShape = format.getPaddingShape();
    long realSize = format.getRealSize();

    if (format instanceof NativeData) {
      switch (format.dataType()) {
        case DataType.S8:
          return new DnnTensor(paddingShape, realSize, this); //Byte
        case DataType.U8:
          return new DnnTensor(paddingShape, realSize, this); //Byte
        case DataType.S32:
          return new DnnTensor(paddingShape, realSize, this); //Int
        case DataType.F32:
          return new DnnTensor(paddingShape, realSize, this);
        default:
          break;
      }
    } else if (format instanceof MemoryData) {
      return new DenseTensor(paddingShape, true);
    } else {
      throw new UnsupportedOperationException("memory format is not supported");
    }
    return null;
  }

  default MemoryData[] singleNativeData(MemoryData[] formats) {
    Util.require(formats.length == 1, "Only accept one tensor as input");
    return nativeData(formats);
  }

  default MemoryData[] nativeData(MemoryData[] formats) {
    MemoryData[] nativeDataArray = new MemoryData[formats.length];
    for (int i = 0; i < formats.length; i++) {
      MemoryData format = formats[i];
      if (!(format instanceof HeapData || format instanceof NativeData)) {
        throw new UnsupportedOperationException("Not support memory format");
      }
      if (format instanceof HeapData) {
        format = ((HeapData) formats[i]).toNative();
      }
      nativeDataArray[i] = format;
    }
    return nativeDataArray;
  }
}

