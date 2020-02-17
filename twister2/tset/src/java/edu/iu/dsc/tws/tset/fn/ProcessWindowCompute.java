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
import edu.iu.dsc.tws.api.tset.fn.MapFunc;

public class ProcessWindowCompute<O, I> implements WindowComputeFunc<O, Iterator<I>> {

  private MapFunc<O, Iterator<I>> processFn;

  public ProcessWindowCompute(MapFunc<O, Iterator<I>> procFn) {
    this.processFn = procFn;
  }

  @Override
  public void prepare(TSetContext ctx) {
    this.processFn.prepare(ctx);
  }

  @Override
  public void close() {
    this.processFn.close();
  }

  @Override
  public O compute(Iterator<I> input) {
    return processFn.map(input);
  }

}
