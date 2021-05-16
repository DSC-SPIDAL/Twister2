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
package edu.iu.dsc.tws.dl.utils;

public interface Shape {
  /**
   * Use this method if its only a single Shape
   */
  default int[] toSingle() {
    throw new RuntimeException("Invalid operation");
  }

  /**
   * Use this method if the current Shape consist of multiple value
   */
  default Shape[] toMulti() {
    throw new RuntimeException("Invalid operation");
  }

  /**
   * Update the given dim and return a new copy
   */
  default Shape copyAndUpdate(int dim, int v) {
    throw new RuntimeException("Invalid operation");
  }

  /**
   * Update the given dim and return a new copy
   */
  default Shape copyAndUpdate(int dim, Shape v) {
    throw new RuntimeException("Invalid operation");
  }


  default int getDim(int dim, int length) {
    int rdim = (dim < 0) ? length + dim : dim;
    Util.require(rdim < length && rdim >= 0, "out of range");
    return rdim;
  }
}

