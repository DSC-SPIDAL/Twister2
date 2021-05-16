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

import java.util.Arrays;

public class SingleShape implements Shape {

  private int[] values;

  public SingleShape(int[] values) {
    this.values = values;
  }

  @Override
  public int[] toSingle() {
    return this.values;
  }

  @Override
  public Shape copyAndUpdate(int dim, int v) {
    //TODO check if this needs to be immutable could be perf improvement
    int[] updated = values.clone();
    updated[getDim(dim, values.length)] = v;
    return new SingleShape(updated);
  }

  @Override
  public int hashCode() {
    int prime = 31;
    int result = 1;
    result = prime * Arrays.hashCode(values);
    return result;
  }

  @Override
  public boolean equals(Object o) {
    return o instanceof SingleShape && this.hashCode() == o.hashCode();
  }
}
