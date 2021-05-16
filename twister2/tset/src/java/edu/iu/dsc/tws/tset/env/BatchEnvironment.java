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
package edu.iu.dsc.tws.tset.env;

import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.InputFormat;

import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.api.compute.graph.OperationMode;
import edu.iu.dsc.tws.api.dataset.DataObject;
import edu.iu.dsc.tws.api.dataset.EmptyDataObject;
import edu.iu.dsc.tws.api.resource.WorkerEnvironment;
import edu.iu.dsc.tws.api.tset.fn.MapFunc;
import edu.iu.dsc.tws.api.tset.fn.SourceFunc;
import edu.iu.dsc.tws.api.tset.sets.StorableTBase;
import edu.iu.dsc.tws.common.table.Row;
import edu.iu.dsc.tws.tset.TSetUtils;
import edu.iu.dsc.tws.tset.fn.impl.ArrowBasedSourceFunction;
import edu.iu.dsc.tws.tset.fn.impl.CSVBasedSourceFunction;
import edu.iu.dsc.tws.tset.fn.impl.TextBasedSourceFunction;
import edu.iu.dsc.tws.tset.sets.BaseTSet;
import edu.iu.dsc.tws.tset.sets.batch.KeyedSourceTSet;
import edu.iu.dsc.tws.tset.sets.batch.SourceTSet;
import edu.iu.dsc.tws.tset.sets.batch.row.RowSourceTSet;
import edu.iu.dsc.tws.tset.sources.HadoopSource;
import edu.iu.dsc.tws.tset.sources.HadoopSourceWithMap;

/**
 * Implementation of {@link TSetEnvironment} for batch {@link OperationMode}.
 * <p>
 * There are 2 ways a tset be executed.
 * 1. running a tset (at the completion, execution will be closed)
 * 2. evaluating a tset (execution will be kept alive until 'finishEval' method is called)
 * <p>
 * And there are 3 execution options.
 * 1. Running/ evaluating a subgraph/ DAG from a specified TSet
 * 2. Running/ evaluating a tset and update another with the results
 * 3. Running just a single source TSet
 */
public class BatchEnvironment extends TSetEnvironment {
  private static final Logger LOG = Logger.getLogger(BatchEnvironment.class.getName());

  /**
   * private cache of {@link BuildContext}s to be reused during execution
   */
  private Map<String, BuildContext> buildCtxCache = new HashMap<>();

  public BatchEnvironment(WorkerEnvironment wEnv) {
    super(wEnv);
  }

  public BatchEnvironment() {
    super();
  }

  @Override
  public OperationMode getOperationMode() {
    return OperationMode.BATCH;
  }

  @Override
  public <T> SourceTSet<T> createSource(SourceFunc<T> source, int parallelism) {
    SourceTSet<T> sourceT = new SourceTSet<>(this, source, parallelism);
    getGraph().addSourceTSet(sourceT);

    return sourceT;
  }

  public SourceTSet<String[]> createCSVSource(String filePath, int datasize, int parallelism,
                                              String type) {
    return createSource(new CSVBasedSourceFunction(filePath, datasize, parallelism, type),
        parallelism);
  }

  //Todo: let's support generic types next
  public SourceTSet<Object> createArrowSource(String filePath, String fileName,
                                              int parallelism, String schema) {
    return createSource(new ArrowBasedSourceFunction(filePath, fileName, parallelism, schema),
        parallelism);
  }

  public SourceTSet<String> createTextSource(String filePath, int dataSize, int parallelism,
                                             String type) {
    return createSource(new TextBasedSourceFunction(filePath, dataSize, parallelism, type),
        parallelism);
  }

  @Override
  public BaseTSet<Integer> createArrowSource(String filePath, int parallelism) {
    return null;
  }

  @Override
  public <T> SourceTSet<T> createSource(String name, SourceFunc<T> source, int parallelism) {
    SourceTSet<T> sourceT = new SourceTSet<>(this, name, source, parallelism);
    getGraph().addSourceTSet(sourceT);

    return sourceT;
  }

  @Override
  public <K, V> KeyedSourceTSet<K, V> createKeyedSource(SourceFunc<Tuple<K, V>> source,
                                                        int parallelism) {
    KeyedSourceTSet<K, V> sourceT = new KeyedSourceTSet<>(this, source, parallelism);
    getGraph().addSourceTSet(sourceT);

    return sourceT;
  }

  @Override
  public <K, V> KeyedSourceTSet<K, V> createKeyedSource(String name, SourceFunc<Tuple<K, V>> source,
                                                        int parallelism) {
    KeyedSourceTSet<K, V> sourceT = new KeyedSourceTSet<>(this, name, source, parallelism);
    getGraph().addSourceTSet(sourceT);

    return sourceT;
  }


  public RowSourceTSet createRowSource(String name, SourceFunc<Row> source,
                                       int parallelism) {
    RowSourceTSet sourceT = new RowSourceTSet(this, name, source, parallelism, null);
    getGraph().addSourceTSet(sourceT);

    return sourceT;
  }

  public <K, V, F extends InputFormat<K, V>> SourceTSet<Tuple<K, V>> createHadoopSource(
      Configuration configuration, Class<F> inputFormat, int parallel) {
    SourceTSet<Tuple<K, V>> sourceT = new SourceTSet<>(this,
        new HadoopSource<>(configuration, inputFormat), parallel);
    getGraph().addSourceTSet(sourceT);

    return sourceT;
  }

  public <K, V, F extends InputFormat<K, V>, I> SourceTSet<I> createHadoopSource(
      Configuration configuration, Class<F> inputFormat, int parallel,
      MapFunc<Tuple<K, V>, I> mapFunc) {
    SourceTSet<I> sourceT = new SourceTSet<>(this,
        new HadoopSourceWithMap<>(configuration, inputFormat, mapFunc), parallel);
    getGraph().addSourceTSet(sourceT);

    return sourceT;
  }

  public <K, V,
      F extends InputFormat<K, V>, K2, V2> KeyedSourceTSet<K2, V2> createKeyedHadoopSource(
      Configuration configuration, Class<F> inputFormat, int parallel,
      MapFunc<Tuple<K, V>, Tuple<K2, V2>> mapFunc) {
    KeyedSourceTSet<K2, V2> sourceT = new KeyedSourceTSet<>(this,
        new HadoopSourceWithMap<>(configuration, inputFormat, mapFunc), parallel);
    getGraph().addSourceTSet(sourceT);

    return sourceT;
  }

  // get data from a tset and update the another
  public <T, ST extends BaseTSet<T> & StorableTBase<T>> void updateTSet(ST tSet, ST updateTSet) {
    // get the data from the evaluation
    DataObject<T> data = getData(tSet.getId());

    // update the data mapping for targetTSet
    addData(updateTSet.getId(), data);
  }

  public <T, ST extends BaseTSet<T> & StorableTBase<T>> void updateTSet(DataObject<T> data,
                                                                        ST updateTSet) {
    // update the data mapping for targetTSet
    addData(updateTSet.getId(), data);
  }

  // adds the data into the task executor
  private <T> void addData(String key, DataObject<T> data) {
    getTaskExecutor().addInput(key, data);
  }

  public <T> DataObject<T> getData(String key) {
    DataObject<T> result = getTaskExecutor().getOutput(key);
    if (result != null) {
      return result;
    } else {
      return EmptyDataObject.getInstance();
    }
  }

  /**
   * Runs a single TSet (NO subgraph execution!)
   *
   * @param tSet tset to run
   */
  public void runOne(BaseTSet tSet) {
    BuildContext buildContext = getTSetGraph().buildOne(tSet);
    executeBuildContext(buildContext);
  }

  /**
   * Runs a subgraph of TSets from the specified TSet
   *
   * @param leafTset TSet to be run
   */
  public void run(BaseTSet leafTset) {
    BuildContext buildContext = getTSetGraph().build(leafTset);
    executeBuildContext(buildContext);
  }

  /**
   * Runs a subgraph of TSets from the specified TSet and output results as a tset
   *
   * @param <T>        type of the output data object
   * @param runTSet    TSet to be run
   * @param updateTSet TSet to be updated
   */
  public <T, ST extends BaseTSet<T> & StorableTBase<T>> void runAndUpdate(ST runTSet,
                                                                          ST updateTSet) {
    run(runTSet);
    updateTSet(runTSet, updateTSet);
  }

  /**
   * Evaluates the TSet using iterative execution in the executor. The task graph generated by the
   * evaluation will be cached and reused in subsequent evaluation calls of that particular TSet. To
   * complete the iterative execution, call @finishEval method
   *
   * @param evalTSet TSet to be evaluated
   */
  public void eval(BaseTSet evalTSet) {
    BuildContext buildCtx;
    String buildId = TSetUtils.generateBuildId(evalTSet);
    if (buildCtxCache.containsKey(buildId)) {
      buildCtx = buildCtxCache.get(buildId);
    } else {
      buildCtx = getTSetGraph().build(evalTSet);
      buildCtxCache.put(buildId, buildCtx);
    }

    // build the context which will create compute graph and execution plan
    buildCtx.build(getTaskExecutor());

    LOG.fine(buildCtx.getComputeGraph()::toString);
    LOG.fine(() -> "edges: " + buildCtx.getComputeGraph().getDirectedEdgesSet());
    LOG.fine(() -> "vertices: " + buildCtx.getComputeGraph().getTaskVertexSet());

    // we execute using the associated executor
    buildCtx.getExecutor().execute();
  }

  /**
   * Similar to eval, but here, the data produced by the evaluation will be passed on to the
   * updateTSet
   *
   * @param evalTSet   TSet to be evaluated
   * @param updateTSet TSet to be updated
   * @param <T>        type
   */
  public <T, ST extends BaseTSet<T> & StorableTBase<T>> void evalAndUpdate(ST evalTSet,
                                                                           ST updateTSet) {
    eval(evalTSet);
    updateTSet(evalTSet, updateTSet);
  }

  /**
   * Completes iterative execution for evaluated TSet
   *
   * @param evalTset TSet to be evaluated
   */
  public void finishEval(BaseTSet evalTset) {
    BuildContext ctx = buildCtxCache.remove(TSetUtils.generateBuildId(evalTset));
    ctx.getExecutor().closeExecution();
  }
}
