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
package edu.iu.dsc.tws.api.tset;

import edu.iu.dsc.tws.api.task.ComputeConnection;
import edu.iu.dsc.tws.api.tset.link.BaseTLink;
import edu.iu.dsc.tws.api.tset.ops.SinkOp;
import edu.iu.dsc.tws.common.config.Config;
import edu.iu.dsc.tws.dataset.DataObject;
import edu.iu.dsc.tws.dataset.DataObjectImpl;
import edu.iu.dsc.tws.dataset.impl.EntityPartition;

public class CachedTSet<T> extends BaseTSet<T> {

  private BaseTLink<T> parent;
  // todo: This dataobject should bind to the executor, I think! because tsets would not be
  //  visible to the executor
  private DataObject<T> datapoints = null;

//  public CachedTSet(Config cfg, TaskGraphBuilder bldr, BaseTLink<T> prnt) {
//    super(cfg, bldr);
//    this.parent = prnt;
//    this.name = "cache-" + parent.getName();
//    datapoints = new DataObjectImpl<>(config);
//
//  }

  public CachedTSet(Config cfg, TSetBuilder bldr, BaseTLink<T> prnt) {
    super(cfg, bldr);
    this.parent = prnt;
    this.name = "cache-" + parent.getName();
    datapoints = new DataObjectImpl<>(config);

  }

  // todo: operations like map is different on a cached tset, because map will be done on data in
  //  the execution runtime, rather than a source task

  @Override
  public boolean baseBuild() {
    boolean isIterable = TSetUtils.isIterableInput(parent, builder.getOpMode());
    boolean keyed = TSetUtils.isKeyedInput(parent);
    // lets override the parallelism
    int p = calculateParallelism(parent);
    ComputeConnection connection = builder.getTaskGraphBuilder().addSink(getName(),
        new SinkOp<>(new CacheSink(), isIterable, keyed), p);
    parent.buildConnection(connection);
    return true;
  }

  public <P> MapTSet<P, T> map(MapFunction<T, P> mapFn) {
    TSetBuilder cacheBuilder = TSetBuilder.newBuilder(config);
    cacheBuilder.setMode(builder.getOpMode());
    SourceTSet<T> cacheSource = (SourceTSet<T>) cacheBuilder.createSource(new CacheSource());
    return cacheSource.map(mapFn);
  }

  public <P> FlatMapTSet<P, T> flatMap(FlatMapFunction<T, P> mapFn) {
    TSetBuilder cacheBuilder = TSetBuilder.newBuilder(config);
    cacheBuilder.setMode(builder.getOpMode());
    SourceTSet<T> cacheSource = (SourceTSet<T>) cacheBuilder.createSource(new CacheSource());
    return cacheSource.flatMap(mapFn);
  }

  @Override
  public void buildConnection(ComputeConnection connection) {

  }

  private class CacheSink implements Sink<T> {

    @Override
    public boolean add(T value) {
      // todo every time add is called, a new partition will be made! how to handle that?
      datapoints.addPartition(new EntityPartition<T>(0, value)); //
      return true;
    }

    @Override
    public void close() {

    }
  }

  private class CacheSource implements Source<T> {
    //TODO: need to check this codes logic developed now just based on the data object API
    private int count = datapoints.getPartitions().length;
    private int current = 0;

    @Override
    public boolean hasNext() {
      boolean hasNext = (current < count) ? datapoints.getPartitions(current)
          .getConsumer().hasNext() : false;

      while (++current < count && !hasNext) {
        hasNext = datapoints.getPartitions(current).getConsumer().hasNext();
      }
      return hasNext;
    }

    @Override
    public T next() {
      return datapoints.getPartitions(current).getConsumer().next();
    }
  }
}
