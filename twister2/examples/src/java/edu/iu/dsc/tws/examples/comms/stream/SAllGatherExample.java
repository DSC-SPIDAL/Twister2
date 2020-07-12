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
package edu.iu.dsc.tws.examples.comms.stream;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.comms.BulkReceiver;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageTypes;
import edu.iu.dsc.tws.api.comms.structs.Tuple;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.resource.WorkerEnvironment;
import edu.iu.dsc.tws.comms.stream.SAllGather;
import edu.iu.dsc.tws.comms.utils.LogicalPlanBuilder;
import edu.iu.dsc.tws.examples.comms.BenchWorker;
import edu.iu.dsc.tws.examples.utils.bench.BenchmarkConstants;
import edu.iu.dsc.tws.examples.utils.bench.BenchmarkUtils;
import edu.iu.dsc.tws.examples.utils.bench.Timing;
import edu.iu.dsc.tws.examples.verification.ResultsVerifier;
import edu.iu.dsc.tws.examples.verification.comparators.IntArrayComparator;
import edu.iu.dsc.tws.examples.verification.comparators.IntComparator;
import edu.iu.dsc.tws.examples.verification.comparators.IteratorComparator;
import edu.iu.dsc.tws.examples.verification.comparators.TupleComparator;

import static edu.iu.dsc.tws.examples.utils.bench.BenchmarkConstants.TIMING_ALL_RECV;

public class SAllGatherExample extends BenchWorker {
  private static final Logger LOG = Logger.getLogger(SAllGatherExample.class.getName());

  private SAllGather gather;

  private volatile boolean gatherDone = true;

  private ResultsVerifier<int[], Iterator<Tuple<Integer, int[]>>> resultsVerifier;

  private int receiverInWorker0 = -1; //any recv scheduled in worker 0

  @Override
  protected void compute(WorkerEnvironment workerEnv) {
    LogicalPlanBuilder logicalPlanBuilder = LogicalPlanBuilder.plan(
        jobParameters.getSources(),
        jobParameters.getTargets(),
        workerEnv
    ).withFairDistribution();

    // create the communication
    gather = new SAllGather(workerEnv.getCommunicator(), logicalPlanBuilder,
        new FinalReduceReceiver(),
        MessageTypes.INTEGER_ARRAY);

    Set<Integer> sourceTasksOfExecutor = logicalPlanBuilder.getSourcesOnThisWorker();
    for (int t : sourceTasksOfExecutor) {
      finishedSources.put(t, false);
    }
    if (sourceTasksOfExecutor.size() == 0) {
      sourcesDone = true;
    }

    Set<Integer> targetTasksOfExecutor = logicalPlanBuilder.getTargetsOnThisWorker();
    for (int taskId : targetTasksOfExecutor) {
      if (logicalPlanBuilder.getTargets().contains(taskId)) {
        gatherDone = false;

        if (workerId == 0) {
          receiverInWorker0 = taskId;
        }
      }
    }

    this.resultsVerifier = new ResultsVerifier<>(inputDataArray, (dataArray, args) -> {
      List<Tuple<Integer, int[]>> listOfArrays = new ArrayList<>();
      for (int i = 0; i < logicalPlanBuilder.getSources().size(); i++) {
        listOfArrays.add(new Tuple<>(i, dataArray));
      }
      return listOfArrays.iterator();
    }, new IteratorComparator<>(
        new TupleComparator<>(
            IntComparator.getInstance(),
            IntArrayComparator.getInstance()
        )
    ));

    // now initialize the workers
    for (int t : sourceTasksOfExecutor) {
      // the map thread where data is produced
      Thread mapThread = new Thread(new BenchWorker.MapWorker(t));
      mapThread.start();
    }
  }

  @Override
  protected boolean progressCommunication() {
    gather.progress();
    return !gather.isComplete();
  }

  @Override
  protected boolean sendMessages(int task, Object data, int flag) {
    while (!gather.gather(task, data, flag)) {
      // lets wait a litte and try again
      gather.progress();
    }
    return true;
  }

  @Override
  protected boolean isDone() {
    return gatherDone && sourcesDone && gather.isComplete();
  }

  public class FinalReduceReceiver implements BulkReceiver {

    private int count = 0;
    private int countToLowest = 0;

    private int totalExpectedCount = 0;

    @Override
    public void init(Config cfg, Set<Integer> targets) {
      this.totalExpectedCount = targets.size() * jobParameters.getTotalIterations();
    }

    @Override
    public boolean receive(int target, Iterator<Object> itr) {
      count++;
      if (receiverInWorker0 == target) {
        this.countToLowest++;
        if (countToLowest > jobParameters.getWarmupIterations()) {
          Timing.mark(BenchmarkConstants.TIMING_MESSAGE_RECV, workerId == 0
              && target == receiverInWorker0);
        }

        if (countToLowest == jobParameters.getTotalIterations()) {
          Timing.mark(TIMING_ALL_RECV, workerId == 0
              && target == receiverInWorker0);
          BenchmarkUtils.markTotalAndAverageTime(resultsRecorder, workerId == 0
              && target == receiverInWorker0);
          resultsRecorder.writeToCSV();
          LOG.info(() -> String.format("Target %d received ALL %d", target, count));
        }
      }

      LOG.info(() -> String.format("Target %d received count %d", target, count));

      verifyResults(resultsVerifier, itr, null);

      if (count == totalExpectedCount) {
        gatherDone = true;
      }
      return true;
    }
  }

  @Override
  protected void finishCommunication(int src) {
    gather.finish(src);
  }

  @Override
  public void close() {
    gather.close();
  }
}
