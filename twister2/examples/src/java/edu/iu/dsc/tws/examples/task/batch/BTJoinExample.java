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
package edu.iu.dsc.tws.examples.task.batch;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import edu.iu.dsc.tws.api.comms.CommunicationContext;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageType;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageTypes;
import edu.iu.dsc.tws.api.comms.structs.JoinedTuple;
import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.api.compute.TaskContext;
import edu.iu.dsc.tws.api.compute.TaskPartitioner;
import edu.iu.dsc.tws.api.compute.nodes.BaseSource;
import edu.iu.dsc.tws.api.compute.nodes.ICompute;
import edu.iu.dsc.tws.api.compute.schedule.elements.TaskInstancePlan;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.comms.utils.JoinRelation;
import edu.iu.dsc.tws.comms.utils.KeyComparatorWrapper;
import edu.iu.dsc.tws.comms.utils.SortJoinUtils;
import edu.iu.dsc.tws.examples.task.BenchTaskWorker;
import edu.iu.dsc.tws.examples.utils.bench.BenchmarkConstants;
import edu.iu.dsc.tws.examples.utils.bench.BenchmarkUtils;
import edu.iu.dsc.tws.examples.utils.bench.Timing;
import edu.iu.dsc.tws.examples.verification.ResultsVerifier;
import edu.iu.dsc.tws.examples.verification.comparators.IteratorComparator;
import edu.iu.dsc.tws.task.impl.ComputeGraphBuilder;
import edu.iu.dsc.tws.task.typed.batch.BJoinCompute;

import static edu.iu.dsc.tws.examples.utils.bench.BenchmarkConstants.TIMING_ALL_SEND;

public class BTJoinExample extends BenchTaskWorker {

  private static final Logger LOG = Logger.getLogger(BTJoinExample.class.getName());
  private static final String RIGHT_EDGE = "right";
  private static final String LEFT_EDGE = "left";
  private static final String SOURCE2 = "source-2";

  @Override
  public ComputeGraphBuilder buildTaskGraph() {
    List<Integer> taskStages = jobParameters.getTaskStages();
    int sourceParallelism = taskStages.get(0);
    int sinkParallelism = taskStages.get(1);
    MessageType keyType = MessageTypes.INTEGER;
    MessageType dataType = MessageTypes.INTEGER_ARRAY;


    BaseSource source1 = new JoinSource(JoinRelation.LEFT);
    BaseSource source2 = new JoinSource(JoinRelation.RIGHT);
    ICompute r = new JoinSinkTask();

    computeGraphBuilder.addSource(SOURCE, source1, sourceParallelism);
    computeGraphBuilder.addSource(SOURCE2, source2, sourceParallelism);
    computeConnection = computeGraphBuilder.addCompute(SINK, r, sinkParallelism);
    computeConnection.innerJoin(SOURCE, SOURCE2, CommunicationContext.JoinAlgorithm.SORT)
        .viaLeftEdge(LEFT_EDGE)
        .viaRightEdge(RIGHT_EDGE)
        .withKeyType(keyType)
        .withLeftDataType(dataType).withRightDataType(dataType)
        .withTaskPartitioner(new TaskPartitioner() {
          private List<Integer> dst;

          @Override
          public void prepare(Set sources, Set destinations) {
            this.dst = new ArrayList<>(destinations);
            Collections.sort(this.dst);
          }

          @Override
          public int partition(int source, Object data) {
            return dst.get((Integer) data % dst.size());
          }

          @Override
          public void commit(int source, int partition) {

          }
        })
        .withComparator(Integer::compareTo);
    return computeGraphBuilder;
  }

  protected static class JoinSinkTask
      extends BJoinCompute<Integer, int[], int[]> {

    private static final long serialVersionUID = -254264903510284798L;

    private ResultsVerifier<int[], Iterator<JoinedTuple>> resultsVerifier;
    private boolean verified = true;
    private boolean timingCondition;

    @Override
    public void prepare(Config cfg, TaskContext ctx) {
      super.prepare(cfg, ctx);
      this.timingCondition = getTimingCondition(SINK, context);
      resultsVerifier = new ResultsVerifier<>(inputDataArray, (ints, args) -> {
        List<Integer> sinkIds = ctx.getTasksByName(SINK).stream()
            .map(TaskInstancePlan::getTaskIndex)
            .sorted()
            .collect(Collectors.toList());

        long sources = ctx.getTasksByName(SOURCE).stream()
            .map(TaskInstancePlan::getTaskIndex)
            .count();

        List<Tuple> onLeftEdge = new ArrayList<>();
        List<Tuple> onRightEdge = new ArrayList<>();

        int iterations = jobParameters.getIterations() + jobParameters.getWarmupIterations();
        for (int i = 0; i < sources; i++) {
          for (int key = 0; key < iterations; key++) {
            if (sinkIds.get(key % sinkIds.size()) == ctx.taskIndex()) {
              onLeftEdge.add(Tuple.of(key, inputDataArray));
            }
            if (sinkIds.get((key / 2) % sinkIds.size()) == ctx.taskIndex()) {
              onRightEdge.add(Tuple.of(key / 2, inputDataArray));
            }
          }
        }

        Iterator<JoinedTuple> objects = SortJoinUtils.join(onLeftEdge, onRightEdge,
            new KeyComparatorWrapper(Comparator.naturalOrder()),
            CommunicationContext.JoinType.INNER);

        return objects;
      }, new IteratorComparator<>(
          (d1, d2) -> d1.getKey().equals(d2.getKey())
      ));
    }

    @Override
    public boolean join(Iterator<JoinedTuple<Integer, int[], int[]>> content) {
      LOG.info("Received joined tuple");
      Timing.mark(BenchmarkConstants.TIMING_ALL_RECV, this.timingCondition);
      BenchmarkUtils.markTotalTime(resultsRecorder, this.timingCondition);
      resultsRecorder.writeToCSV();
      this.verified = verifyResults(resultsVerifier, content, null, verified);
      return true;
    }
  }

  protected static class JoinSource extends BaseSource {
    private int count = 0;

    private int iterations;

    private boolean timingCondition;
    private boolean endNotified;
    private JoinRelation joinRelation;

    public JoinSource(JoinRelation joinRelation) {
      this.joinRelation = joinRelation;
    }

    @Override
    public void prepare(Config cfg, TaskContext ctx) {
      super.prepare(cfg, ctx);
      this.iterations = jobParameters.getIterations() + jobParameters.getWarmupIterations();
      this.timingCondition = getTimingCondition(SOURCE, ctx)
          && this.joinRelation == JoinRelation.LEFT;
      sendersInProgress.incrementAndGet();
    }

    private void notifyEnd() {
      if (endNotified) {
        return;
      }
      sendersInProgress.decrementAndGet();
      endNotified = true;
      LOG.info(String.format("Source : %d done sending.", context.taskIndex()));
    }

    @Override
    public void execute() {
      if (count < iterations) {
        if (count == jobParameters.getWarmupIterations()) {
          Timing.mark(TIMING_ALL_SEND, this.timingCondition);
        }
        if (joinRelation == JoinRelation.LEFT) {
          context.write(LEFT_EDGE, count, inputDataArray);
        } else {
          context.write(RIGHT_EDGE, count / 2, inputDataArray);
        }
        count++;
      } else if (!this.endNotified) {
        if (joinRelation == JoinRelation.LEFT) {
          context.end(LEFT_EDGE);
        } else {
          context.end(RIGHT_EDGE);
        }
        this.notifyEnd();
      }
    }
  }
}
