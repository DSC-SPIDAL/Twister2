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
package edu.iu.dsc.tws.dl.utils.pair;

import java.io.Serializable;

import edu.iu.dsc.tws.dl.data.Tensor;

/**
 * Holds a pair of Tensors
 */
public class TensorArrayPair implements Serializable {

  private Tensor[] t0;
  private Tensor[] t1;

  public TensorArrayPair(Tensor[] t0, Tensor[] t1) {
    this.t0 = t0;
    this.t1 = t1;
  }

  public Tensor[] getValue0() {
    return t0;
  }

  public Tensor[] getValue1() {
    return t1;
  }
}
