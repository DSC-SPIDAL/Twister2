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
package edu.iu.dsc.tws.tset.links.batch;

import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.api.tset.schema.KeyedSchema;
import edu.iu.dsc.tws.api.tset.schema.TupleSchema;
import edu.iu.dsc.tws.tset.env.BatchChkPntEnvironment;
import edu.iu.dsc.tws.tset.env.BatchEnvironment;
import edu.iu.dsc.tws.tset.sets.batch.KeyedCachedTSet;
import edu.iu.dsc.tws.tset.sets.batch.KeyedCheckpointedTSet;
import edu.iu.dsc.tws.tset.sets.batch.KeyedPersistedTSet;
import edu.iu.dsc.tws.tset.sinks.CacheIterSink;
import edu.iu.dsc.tws.tset.sinks.DiskPersistIterIterSink;
import edu.iu.dsc.tws.tset.sources.DiskPartitionBackedSource;

public abstract class KeyedBatchIteratorLinkWrapper<K, V> extends BatchIteratorLink<Tuple<K, V>> {
  KeyedBatchIteratorLinkWrapper(BatchEnvironment env, String n, int sourceP,
                                TupleSchema schema) {
    super(env, n, sourceP, schema);
  }

  KeyedBatchIteratorLinkWrapper(BatchEnvironment env, String n, int sourceP, int targetP,
                                TupleSchema schema) {
    super(env, n, sourceP, targetP, schema);
  }

  protected KeyedBatchIteratorLinkWrapper() {
  }

  @Override
  public KeyedCachedTSet<K, V> lazyCache() {
    KeyedCachedTSet<K, V> cacheTSet = new KeyedCachedTSet<>(getTSetEnv(), new CacheIterSink<>(),
        getTargetParallelism(), getSchema());
    addChildToGraph(cacheTSet);

    return cacheTSet;
  }

  @Override
  public KeyedCachedTSet<K, V> cache() {
    return (KeyedCachedTSet<K, V>) super.cache();
  }

  @Override
  public KeyedPersistedTSet<K, V> lazyPersist() {
    KeyedPersistedTSet<K, V> persistedTSet = new KeyedPersistedTSet<>(getTSetEnv(),
        new DiskPersistIterIterSink<>(this.getId()), getTargetParallelism(), getSchema());
    addChildToGraph(persistedTSet);

    return persistedTSet;
  }

  @Override
  public KeyedPersistedTSet<K, V> persist() {
    // handling checkpointing
    if (getTSetEnv().isCheckpointingEnabled()) {
      String persistVariableName = this.getId() + "-persisted";
      BatchChkPntEnvironment chkEnv = (BatchChkPntEnvironment) getTSetEnv();
      Boolean persisted = chkEnv.initVariable(persistVariableName, false);

      if (persisted) {
        // create a source function with the capability to read from disk
        DiskPartitionBackedSource<Tuple<K, V>> sourceFn =
            new DiskPartitionBackedSource<>(this.getId());

        // pass the source fn to the checkpointed tset (that would create a source tset from the
        // source function, the same way as a persisted tset. This preserves the order of tsets
        // that are being created in the checkpointed env)
        KeyedCheckpointedTSet<K, V> checkTSet = new KeyedCheckpointedTSet<>(getTSetEnv(), sourceFn,
            this.getTargetParallelism(), getSchema());

        // adding checkpointed tset to the graph, so that the IDs would not change
        addChildToGraph(checkTSet);

        // run only the checkpointed tset so that it would populate the inputs in the executor
        getTSetEnv().runOne(checkTSet);

        return checkTSet;
      } else {
        KeyedPersistedTSet<K, V> storable = this.doPersist();
        chkEnv.updateVariable(persistVariableName, true);
        chkEnv.commit();
        return storable;
      }
    }
    return doPersist();
  }

  @Override
  protected KeyedSchema getSchema() {
    return (KeyedSchema) super.getSchema();
  }

  private KeyedPersistedTSet<K, V> doPersist() {
    KeyedPersistedTSet<K, V> lazyPersist = lazyPersist();
    getTSetEnv().run(lazyPersist);
    return lazyPersist;
  }
}
