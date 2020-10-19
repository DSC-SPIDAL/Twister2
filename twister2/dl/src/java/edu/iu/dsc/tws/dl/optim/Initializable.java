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
package edu.iu.dsc.tws.dl.optim;

import edu.iu.dsc.tws.dl.optim.initialization.Zeros;

public abstract class Initializable {

  protected InitializationMethod weightInitMethod = new Zeros();
  protected InitializationMethod biasInitMethod = new Zeros();

  public Initializable setInitMethod(InitializationMethod weightInitMethod,
                              InitializationMethod biasInitMethod{
    if (weightInitMethod != null) {
      this.weightInitMethod = weightInitMethod;
    }

    if (biasInitMethod != null) {
      this.biasInitMethod = biasInitMethod;
    }
    reset();
    return this;
  }

  public Initializable setInitMethod(InitializationMethod[] initMethod){
    throw new UnsupportedOperationException("setInitMethod with a array of InitializationMethod"
        + " does not support for ${this.toString}");
  }

  public abstract void reset();
}
