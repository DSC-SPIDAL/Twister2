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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.graph.Operation;
import edu.iu.dsc.tws.dl.utils.Util;

public abstract class DynamicContainer<A extends Activity> extends Container<A> {

  /**
   * Add a sub-module to the contained `modules`
   *
   * @param module module to be add
   * @return this container
   */
  public DynamicContainer add(AbstractModule module) {
    Util.require(!(module instanceof Operation),
        "Add operations to dynamic container is not allowed, as operations don't have backward. "
            + "Operation can only be used in Graph");
    List<AbstractModule> moduleList = new ArrayList<>();
    if (this.isFloat()) {
      module.toFloat();
    }
    moduleList.add(module);
    validateInput(moduleList);
    this.modules.add(module);
    checkDuplicate(new HashSet<Integer>());
    return this;
  }

}
