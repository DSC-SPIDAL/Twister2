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
package edu.iu.dsc.tws.dl.utils.graph;

import edu.iu.dsc.tws.dl.module.AbstractModule;
import edu.iu.dsc.tws.dl.utils.ReflectionUtils;

public class BlasToIR extends ConvertBase<AbstractModule, IRElement> {
  @Override
  public boolean convertLayerCheck(AbstractModule layer) {
    return ReflectionUtils.findClass(layer.getClass().getSimpleName()) != null
        || layer instanceof AbstractModule;
  }

  @Override
  public IRElement convertLayer(AbstractModule layer) {
    Class cls = ReflectionUtils.findClass(className(layer.getClass().getSimpleName()));
    if (cls != null) {
      return ReflectionUtils.reflectToIR(layer, cls);
    } else if (layer instanceof AbstractModule) {
      IRGeneralModule op = new IRGeneralModule((AbstractModule) layer);
      return new IRElement(layer.getName(), op, null, null);
    } else {
      throw new UnsupportedOperationException("can not convert $layer to IRelement ");
    }
  }

  private String className(String name) {
    return "edu.iu.dsc.tws.dl.utils.graph.IR" + name;
  }
}
