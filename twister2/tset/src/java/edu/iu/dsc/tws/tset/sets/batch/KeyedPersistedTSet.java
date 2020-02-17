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
package edu.iu.dsc.tws.tset.sets.batch;

import java.util.Iterator;

import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.api.tset.fn.SinkFunc;
import edu.iu.dsc.tws.api.tset.schema.KeyedSchema;
import edu.iu.dsc.tws.api.tset.schema.TupleSchema;
import edu.iu.dsc.tws.api.tset.sets.StorableTBase;
import edu.iu.dsc.tws.tset.env.BatchTSetEnvironment;

public class KeyedPersistedTSet<K, V> extends KeyedStoredTSet<K, V> {

  public KeyedPersistedTSet(BatchTSetEnvironment tSetEnv,
                            SinkFunc<Iterator<Tuple<K, V>>> storingSinkFn, int parallelism,
                            KeyedSchema inputSchema) {
    super(tSetEnv, "kpersisted", storingSinkFn, parallelism, inputSchema);
  }

  @Override
  public KeyedPersistedTSet<K, V> setName(String n) {
    return (KeyedPersistedTSet<K, V>) super.setName(n);
  }

  @Override
  public KeyedPersistedTSet<K, V> addInput(String key, StorableTBase<?> input) {
    return (KeyedPersistedTSet<K, V>) super.addInput(key, input);
  }

  @Override
  public KeyedPersistedTSet<K, V> withSchema(TupleSchema schema) {
    return (KeyedPersistedTSet<K, V>) super.withSchema(schema);
  }

  @Override
  public KeyedCachedTSet<K, V> cache() {
    throw new UnsupportedOperationException("Cache on PersistedTSet is undefined!");
  }

  @Override
  public KeyedCachedTSet<K, V> lazyCache() {
    throw new UnsupportedOperationException("Cache on PersistedTSet is undefined!");
  }

  @Override
  public KeyedPersistedTSet<K, V> persist() {
    return this;
  }

  @Override
  public KeyedPersistedTSet<K, V> lazyPersist() {
    return this;
  }
}
