//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless Util.required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
package edu.iu.dsc.tws.dl.module.mkldnn;

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.module.AbstractModule;
import edu.iu.dsc.tws.dl.module.DynamicContainer;
import edu.iu.dsc.tws.dl.utils.Util;

/**
 * Helper utilities when integrating containers with MKL-DNN
 */
public abstract class MklDnnContainer extends DynamicContainer<Activity> implements MklDnnModule {

  protected transient ReorderManager reorderManager = new ReorderManager(this);
  protected MklDnnModule[] mklDnnModules;

  @Override
  public DynamicContainer add(AbstractModule module) {
    Util.require(mklDnnModules == null, "You should not call add after compilation");
    Util.require(module instanceof MklDnnModule, "layer should be MklDnnModule");
    return super.add(module);
  }

  private boolean checkInputs() {
    return getAllInputs(this);
  }

  private boolean getAllInputs(AbstractModule module) {
    if (module instanceof Sequential) {
      return getAllInputs(((Sequential) module).modules.get(0));
    } else if (module instanceof ConcatTable) {
      boolean result = true;
      for (int i = 0; i < ((ConcatTable) module).modules.size(); i++) {
        result = result && getAllInputs(((ConcatTable) module).modules.get(i));
        if (!result) {
          return false;
        }
      }
      return result;
    } else {
      return module instanceof Input;
    }
  }

  final void compile(Phase phase) {
    Util.require(checkInputs(), "You should add Input for the container.");
    compile(phase, new MklDnnRuntime(), new MemoryData[0]);
  }

  /**
   * Create MklDnnRuntime and compile the model
   *
   * @param phase
   */
  private void compile(Phase phase, MemoryData[] formats) {
    compile(phase, new MklDnnRuntime(), formats);
  }

  /**
   * Compile the model, which includes infer memory shapes, allocate memory, optimize computing
   * path and create MKL-DNN primitives
   *
   * @param phase
   * @param runtime
   */
  private void compile(Phase phase, MklDnnRuntime runtime,
                             MemoryData[] formats) {
    freeze();
    fusion(phase);
    initPrimitives(phase, runtime, formats);
  }

  final void initPrimitives(Phase phase, MklDnnRuntime runtime, MemoryData[] formats) {
    setRuntime(runtime);
    MemoryData[] outputFormats = initFwdPrimitives(formats, phase).getValue1();
    if (phase == Phase.TRAINNING) {
      initBwdPrimitives(outputFormats, phase);
      initGradWPrimitives(outputFormats, phase);
    }
  }

  @Override
  public void setRuntime(MklDnnRuntime runtime) {
    MklDnnModule.super.setRuntime(runtime);
    reorderManager.setRuntime(runtime);
    for (AbstractModule module : modules) {
      if (module instanceof MklDnnModule) {
        ((MklDnnModule) module).setRuntime(runtime);
      }
    }
  }

  /**
   * Modify the computing path by fuse some layers into one to improve the performance
   */
  protected void fusion(Phase phase) {

    for (AbstractModule module : modules) {
      if (module instanceof MklDnnContainer) {
        ((MklDnnContainer) module).fusion(phase);
      }
    }
  }

  protected void freeze() {
    if (mklDnnModules == null) {
      mklDnnModules = new MklDnnModule[modules.size()];
      for (int i = 0; i < mklDnnModules.length; i++) {
        mklDnnModules[i] = (MklDnnModule) modules.get(i);
      }
    }
    for (AbstractModule module : modules) {
      if (module instanceof MklDnnContainer) {
        ((MklDnnContainer) module).freeze();
      }
    }
  }

  @Override
  public void release() {
    super.release();
    this.releaseResources();
  }

  @Override
  public MklDnnContainer setQuantize(boolean value) {
    this.modules.forEach(module -> {
      if (module instanceof MklDnnModule) {
        ((MklDnnModule) module).setQuantize(value);
      }
    });
    return this;
  }
}
