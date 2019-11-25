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
package edu.iu.dsc.tws.api.tset.link.streaming;

import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.api.tset.TBase;
import edu.iu.dsc.tws.api.tset.fn.ApplyFunc;
import edu.iu.dsc.tws.api.tset.fn.ComputeCollectorFunc;
import edu.iu.dsc.tws.api.tset.fn.ComputeFunc;
import edu.iu.dsc.tws.api.tset.fn.FlatMapFunc;
import edu.iu.dsc.tws.api.tset.fn.MapFunc;
import edu.iu.dsc.tws.api.tset.fn.SinkFunc;
import edu.iu.dsc.tws.api.tset.link.TLink;
import edu.iu.dsc.tws.api.tset.sets.streaming.StreamingTSet;
import edu.iu.dsc.tws.api.tset.sets.streaming.StreamingTupleTSet;

public interface StreamingTLink<T1, T0> extends TLink<T1, T0> {
  @Override
  StreamingTLink<T1, T0> setName(String name);

  @Override
  <O> StreamingTSet<O> compute(ComputeFunc<O, T1> computeFunction);

  @Override
  <O> StreamingTSet<O> compute(ComputeCollectorFunc<O, T1> computeFunction);

  @Override
  <O> StreamingTSet<O> map(MapFunc<O, T0> mapFn);

  @Override
  <O> StreamingTSet<O> flatmap(FlatMapFunc<O, T0> mapFn);

  @Override
  <K, V> StreamingTupleTSet<K, V> mapToTuple(MapFunc<Tuple<K, V>, T0> genTupleFn);

  @Override
  void forEach(ApplyFunc<T0> applyFunction);

  @Override
  TBase sink(SinkFunc<T1> sinkFunction);
}
