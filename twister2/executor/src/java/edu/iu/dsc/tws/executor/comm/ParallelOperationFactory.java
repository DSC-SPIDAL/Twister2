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
package edu.iu.dsc.tws.executor.comm;

import java.util.Set;

import edu.iu.dsc.tws.comms.core.TWSNetwork;
import edu.iu.dsc.tws.data.api.DataType;

public class ParallelOperationFactory {
  private TWSNetwork network;

  public ParallelOperationFactory(TWSNetwork network) {
    this.network = network;
  }

  public IParallelOperation build(String operation, Set<Integer> sources, Set<Integer> dests,
                                 DataType dataType) {
    return null;
  }

  public IParallelOperation build(String operation, Set<Integer> sources, Set<Integer> dests,
                                 DataType dataType, DataType keyType) {
    return null;
  }
}
