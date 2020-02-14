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

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.api.compute.nodes.INode;
import edu.iu.dsc.tws.api.dataset.DataObject;
import edu.iu.dsc.tws.api.dataset.DataPartition;
import edu.iu.dsc.tws.api.dataset.DataPartitionConsumer;
import edu.iu.dsc.tws.api.tset.fn.MapFunc;
import edu.iu.dsc.tws.api.tset.fn.PartitionFunc;
import edu.iu.dsc.tws.api.tset.fn.ReduceFunc;
import edu.iu.dsc.tws.api.tset.fn.SinkFunc;
import edu.iu.dsc.tws.api.tset.schema.Schema;
import edu.iu.dsc.tws.api.tset.sets.StorableTBase;
import edu.iu.dsc.tws.api.tset.sets.TSet;
import edu.iu.dsc.tws.tset.env.BatchTSetEnvironment;
import edu.iu.dsc.tws.tset.links.batch.AllGatherTLink;
import edu.iu.dsc.tws.tset.links.batch.AllReduceTLink;
import edu.iu.dsc.tws.tset.links.batch.DirectTLink;
import edu.iu.dsc.tws.tset.links.batch.GatherTLink;
import edu.iu.dsc.tws.tset.links.batch.PartitionTLink;
import edu.iu.dsc.tws.tset.links.batch.ReduceTLink;
import edu.iu.dsc.tws.tset.links.batch.ReplicateTLink;
import edu.iu.dsc.tws.tset.ops.SinkOp;
import edu.iu.dsc.tws.tset.sources.DataPartitionSourceFunc;

/**
 * Cached tset
 *
 * @param <T> base type of the tset
 */
public abstract class StoredTSet<T> extends BatchTSetImpl<T> implements StorableTBase<T> {
  private SinkFunc<?> storingSinkFunc;
  private String storedSourcePrefix;
  protected SourceTSet<T> storedSource;

  /*
  Sink function type is unknown as we need to preserve the output datao bject type to T. In doing
  so, we would need to have several types of sink functions that can convert the comms message to
   T. example: for direct, sink func would convert Iterator<T> to T.
   */
  StoredTSet(BatchTSetEnvironment tSetEnv, String name, SinkFunc<?> sinkFunc,
             int parallelism, Schema inputSchema) {
    super(tSetEnv, name, parallelism, inputSchema);
    this.storingSinkFunc = sinkFunc;
    this.storedSourcePrefix = "stored(" + getId() + ")";
  }

  @Override
  public INode getINode() {
    return new SinkOp<>(storingSinkFunc, this, getInputs());
  }

  @Override
  public DirectTLink<T> direct() {
    return getStoredSourceTSet().direct();
  }

  @Override
  public ReduceTLink<T> reduce(ReduceFunc<T> reduceFn) {
    return getStoredSourceTSet().reduce(reduceFn);
  }

  @Override
  public PartitionTLink<T> partition(PartitionFunc<T> partitionFn, int targetParallelism) {
    return getStoredSourceTSet().partition(partitionFn, targetParallelism);
  }

  @Override
  public PartitionTLink<T> partition(PartitionFunc<T> partitionFn) {
    return partition(partitionFn, getParallelism());
  }

  @Override
  public GatherTLink<T> gather() {
    return getStoredSourceTSet().gather();
  }

  @Override
  public AllReduceTLink<T> allReduce(ReduceFunc<T> reduceFn) {
    return getStoredSourceTSet().allReduce(reduceFn);
  }

  @Override
  public AllGatherTLink<T> allGather() {
    return getStoredSourceTSet().allGather();
  }

  @Override
  public <K, V> KeyedTSet<K, V> mapToTuple(MapFunc<Tuple<K, V>, T> generateTuple) {
    return getStoredSourceTSet().mapToTuple(generateTuple);
  }

  @Override
  public ReplicateTLink<T> replicate(int replications) {
    return getStoredSourceTSet().replicate(replications);
  }

  @Override
  public StoredTSet<T> addInput(String key, StorableTBase<?> input) {
    return (StoredTSet<T>) super.addInput(key, input);
  }

  @Override
  public ComputeTSet<T, Iterator<T>> union(TSet<T> other) {
    throw new UnsupportedOperationException("Union on StoredTSet is not supported");
  }

  @Override
  public ComputeTSet<T, Iterator<T>> union(Collection<TSet<T>> tSets) {
    throw new UnsupportedOperationException("Union on StoredTSet is not supported");
  }

  /**
   * Once a cache TSet is run, it will produce data and store data in the data executor with the
   * tset ID as the key. This will be exposed to other operations as a {@link SourceTSet}
   *
   * @return {@link SourceTSet} source tset from data
   * <p>
   */
  @Override
  public SourceTSet<T> getStoredSourceTSet() {
    if (this.storedSource == null) {
      // this cache source will consume the data object created by the execution of this tset.
      // hence this tset ID needs to be set as an input to the cache source
      this.storedSource = getTSetEnv().createSource(storedSourcePrefix,
          new DataPartitionSourceFunc<>(storedSourcePrefix), getParallelism());
      this.storedSource.addInput(storedSourcePrefix, this);
    }

    return this.storedSource;
  }

  @Override
  public List<T> getData() {
    List<T> results = new ArrayList<>();

    DataObject<T> data = getTSetEnv().getData(getId());

    if (data != null) {
      for (DataPartition<T> partition : data.getPartitions()) {
        DataPartitionConsumer<T> consumer = partition.getConsumer();
        while (consumer.hasNext()) {
          results.add(consumer.next());
        }
      }
    }

    return results;
  }

  @Override
  public DataObject<T> getDataObject() {
    return getTSetEnv().getData(getId());
  }
}
