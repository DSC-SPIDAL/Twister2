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
package edu.iu.dsc.tws.dl.module;

import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.storage.ArrayDoubleStorage;
import edu.iu.dsc.tws.dl.data.storage.ArrayFloatStorage;
import edu.iu.dsc.tws.dl.data.storage.ArrayStorage;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.utils.Util;

public final class ModuleUtil {

  private ModuleUtil() {
  }

  public static Tensor flatten(Tensor[] parameters) {
    Tensor compactedTensor = isCompact(parameters);
    if (compactedTensor != null) {
      return compactedTensor;
    }
    int i = 0;
    int length = 0;
    while (i < parameters.length) {
      Util.require(parameters[i].isContiguous(), "parameters should be contiguous");
      length += parameters[i].nElement();
      i += 1;
    }

    DenseTensor result = new DenseTensor(length, parameters[0].isFloat());
    ArrayStorage resultStorage = result.storage();

    i = 0;
    int offset = 0;
    if (result.isFloat()) {
      while (i < parameters.length) {
        System.arraycopy(parameters[i].storage().toFloatArray(), parameters[i].storageOffset() - 1,
            resultStorage.toFloatArray(), offset, parameters[i].nElement());
        parameters[i].set(resultStorage, offset + 1, parameters[i].size(), parameters[i].stride());
        offset += parameters[i].nElement();
        i += 1;
      }
    } else {
      while (i < parameters.length) {
        System.arraycopy(parameters[i].storage().toDoubleArray(), parameters[i].storageOffset() - 1,
            resultStorage.toDoubleArray(), offset, parameters[i].nElement());
        parameters[i].set(resultStorage, offset + 1, parameters[i].size(), parameters[i].stride());
        offset += parameters[i].nElement();
        i += 1;
      }
    }
    return result;
  }

  private static Tensor isCompact(Tensor[] parameters) {
    Util.require(parameters.length > 0,
        "The length of paramters should >= 0"
            + "parameter length"
            + " ${parameters.length}");

    int i = 1;
    ArrayStorage storage;
    int length;
    int offset;
    if (parameters[0].isFloat()) {
      storage = (ArrayFloatStorage) parameters[0].storage();
      length = parameters[0].nElement();
      offset = parameters[0].storageOffset();
      // make sure parameters is shared and contiguous
      while (i < parameters.length) {
        if (storage != parameters[i].storage()) {
          return null;
        }
        if (offset + length != parameters[i].storageOffset()) {
          return null;
        }
        length += parameters[i].nElement();
        i += 1;
      }
    } else {
      storage = (ArrayDoubleStorage) parameters[0].storage();
      length = parameters[0].nElement();
      offset = parameters[0].storageOffset();
      // make sure parameters is shared and contiguous
      while (i < parameters.length) {
        if (storage != parameters[i].storage()) {
          return null;
        }
        if (offset + length != parameters[i].storageOffset()) {
          return null;
        }
        length += parameters[i].nElement();
        i += 1;
      }
    }


    return new DenseTensor(storage, offset, new int[]{length});
  }
}
