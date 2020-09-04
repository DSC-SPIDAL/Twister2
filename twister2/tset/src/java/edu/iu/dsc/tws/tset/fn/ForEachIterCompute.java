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

package edu.iu.dsc.tws.tset.fn;

import java.util.Iterator;

import edu.iu.dsc.tws.api.tset.TSetContext;
import edu.iu.dsc.tws.api.tset.fn.ApplyFunc;
import edu.iu.dsc.tws.api.tset.fn.ComputeFunc;

public class ForEachIterCompute<T> implements ComputeFunc<Iterator<T>, Object> {
  private ApplyFunc<T> applyFn;

  public ForEachIterCompute() {

  }

  public ForEachIterCompute(ApplyFunc<T> applyFunction) {
    this.applyFn = applyFunction;
  }

  public Object compute(Iterator<T> input) {
    while (input.hasNext()) {
      applyFn.apply(input.next());
    }
    return null;
  }

  @Override
  public void prepare(TSetContext context) {
    applyFn.prepare(context);
  }

  @Override
  public void close() {
    applyFn.close();
  }
}
