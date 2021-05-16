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

import edu.iu.dsc.tws.dl.module.mkldnn.memory.MklDnnModuleHelper;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.MemoryDataArrayPair;

/**
 * Helper utilities when integrating Module with MKL-DNN
 */
@SuppressWarnings("ConstantName")
public interface MklDnnModule extends MklDnnModuleHelper {
  /**
   * MklDnn runtime, which includes a MKL-DNN engine and a MKL-DNN stream.
   * Note that this instance will be erased when send to remote worker, so you
   * should recreate a MklDnnRuntime.
   */
  MklDnnRuntime runtime = new MklDnnRuntime();

  default void setRuntime(MklDnnRuntime runtime) {
    // TODO need to check how this is used and address this
    //this.runtime = runtime;
  }

  default MklDnnRuntime getRuntime() {
    Util.require(runtime != null, "you should init the mkldnn runtime first");
    return runtime;
  }

  /**
   * Init the MKL-DNN primitives for the layer. Note that these primitives will be erased when
   * sent to a remote worker.
   */
  MemoryDataArrayPair initFwdPrimitives(MemoryData[] inputs, Phase phase);

  MemoryDataArrayPair initBwdPrimitives(MemoryData[] grad, Phase phase);

  default MemoryData[] initGradWPrimitives(MemoryData[] grad, Phase phase) {
    return grad;
  }

  default MemoryDataArrayPair initFwdPrimitives(MemoryData[] inputs) {
    return initFwdPrimitives(inputs, null);
  }

  default MemoryDataArrayPair initBwdPrimitives(MemoryData[] grad) {
    return initBwdPrimitives(grad, null);
  }

  default MemoryData[] initGradWPrimitives(MemoryData[] grad) {
    return initGradWPrimitives(grad, null);
  }

  MemoryData[] inputFormats();

  MemoryData[] gradInputFormats();

  MemoryData[] outputFormats();

  MemoryData[] gradOutputFormats();

  MemoryData[] gradOutputWeightFormats();

  default MklDnnModule setQuantize(boolean value) {
    return this;
  }
}

