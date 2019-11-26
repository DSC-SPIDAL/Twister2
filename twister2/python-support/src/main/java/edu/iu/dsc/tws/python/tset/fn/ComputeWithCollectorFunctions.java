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
package edu.iu.dsc.tws.python.tset.fn;

import java.io.Serializable;

import edu.iu.dsc.tws.api.tset.RecordCollector;
import edu.iu.dsc.tws.api.tset.TSetContext;
import edu.iu.dsc.tws.api.tset.fn.ComputeCollectorFunc;
import edu.iu.dsc.tws.python.processors.PythonLambdaProcessor;

public class ComputeWithCollectorFunctions extends TFunc<ComputeCollectorFunc> {

  private static final ComputeWithCollectorFunctions INSTANCE = new ComputeWithCollectorFunctions();

  static ComputeWithCollectorFunctions getInstance() {
    return INSTANCE;
  }

  public static class ComputeFuncImpl implements ComputeCollectorFunc, Serializable {

    private PythonLambdaProcessor lambdaProcessor;
    private TSetContext ctx;

    ComputeFuncImpl(byte[] byBinary) {
      this.lambdaProcessor = new PythonLambdaProcessor(byBinary);
    }

    @Override
    public void prepare(TSetContext context) {
      this.ctx = context;
    }

    @Override
    public void compute(Object input, RecordCollector output) {
      this.lambdaProcessor.invoke(input, output, ctx);
    }
  }

  @Override
  public ComputeCollectorFunc build(byte[] pyBinary) {
    return new ComputeFuncImpl(pyBinary);
  }
}
