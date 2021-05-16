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
package edu.iu.dsc.tws.dl.utils.varformat;

import edu.iu.dsc.tws.dl.utils.VariableFormat;

@SuppressWarnings("TypeName")
public class GP_OUT_IN_KW_KH implements VariableFormat {
  @Override
  public int getFanIn(int[] shape) {
    int receptiveFieldSize = shape[0] * shape[3] * shape[4];
    return shape[2] * receptiveFieldSize;
  }

  @Override
  public int getFanOut(int[] shape) {
    int receptiveFieldSize = shape[0] * shape[3] * shape[4];
    return shape[1] * receptiveFieldSize;
  }
}
