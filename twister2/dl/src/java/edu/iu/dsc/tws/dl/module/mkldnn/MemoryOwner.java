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

import java.util.ArrayList;
import java.util.List;

/**
 * This trait is a owner of the resources that need to be released.
 * It will track all Releasable resources (Primitives, tensor, ReorderMemory).
 * You can call releaseResources to release all the
 * resources at once. These resources will require an implicit MemoryOwner at
 * the constructors. The constructors of the resources will register themselves to the MemoryOwner.
 * For DNN Layer classes, they extends MemoryOwner and have a implicit value of "this" as a
 * MemoryOwner. ReorderMemory is a kind of special resource. They can be a normal layer or a
 * resource of another layer.
 */
@SuppressWarnings({"MemberName", "ParameterName", "ConstantName"})
public interface MemoryOwner {
  List<Releasable> _resources = new ArrayList();

  default void registerResource(Releasable m) {
    _resources.add(m);
  }

  default void releaseResources() {
    _resources.forEach(o -> o.release());
    _resources.clear();
  }
}
