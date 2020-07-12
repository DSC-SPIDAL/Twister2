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
package edu.iu.dsc.tws.examples.comms.batch;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.comms.BulkReceiver;
import edu.iu.dsc.tws.api.comms.messaging.types.MessageTypes;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.resource.WorkerEnvironment;
import edu.iu.dsc.tws.comms.batch.BBroadcast;
import edu.iu.dsc.tws.comms.utils.LogicalPlanBuilder;
import edu.iu.dsc.tws.examples.comms.BenchWorker;
import edu.iu.dsc.tws.examples.utils.bench.BenchmarkConstants;
import edu.iu.dsc.tws.examples.utils.bench.BenchmarkUtils;
import edu.iu.dsc.tws.examples.utils.bench.Timing;
import edu.iu.dsc.tws.examples.verification.ResultsVerifier;
import edu.iu.dsc.tws.examples.verification.comparators.IntArrayComparator;
import edu.iu.dsc.tws.examples.verification.comparators.IteratorComparator;

public class BBroadcastExample extends BenchWorker {
  private static final Logger LOG = Logger.getLogger(BBroadcastExample.class.getName());

  private BBroadcast bcast;

  private ResultsVerifier<int[], Iterator<int[]>> resultsVerifier;

  @Override
  protected void compute(WorkerEnvironment workerEnv) {
    if (jobParameters.getTaskStages().get(0) != 1) {
      LOG.warning("Setting no of senders to 1");
      jobParameters.getTaskStages().set(0, 1);
    }

    int source = 0;

    LogicalPlanBuilder logicalPlanBuilder = LogicalPlanBuilder.plan(
        jobParameters.getSources(),
        jobParameters.getTargets(),
        workerEnv
    ).withFairDistribution();

    // create the communication
    bcast = new BBroadcast(workerEnv.getCommunicator(), logicalPlanBuilder,
        new BCastReceiver(), MessageTypes.INTEGER_ARRAY, true);

    Set<Integer> tasksOfExecutor = logicalPlanBuilder.getSourcesOnThisWorker();
    for (int t : tasksOfExecutor) {
      finishedSources.put(t, false);
    }
    if (tasksOfExecutor.size() == 0) {
      sourcesDone = true;
    }

    this.resultsVerifier = new ResultsVerifier<>(inputDataArray, (ints, args) -> {
      List<int[]> expectedData = new ArrayList<>();
      for (int i = 0; i < jobParameters.getTotalIterations(); i++) {
        expectedData.add(ints);
      }
      return expectedData.iterator();
    }, new IteratorComparator<>(
        IntArrayComparator.getInstance()
    ));

    // the map thread where data is produced
    if (workerId == 0) {
      Thread mapThread = new Thread(new MapWorker(source));
      mapThread.start();
    }
  }

  @Override
  protected boolean progressCommunication() {
    bcast.progress();
    return !bcast.isComplete();
  }

  protected void finishCommunication(int src) {
    bcast.finish(src);
  }

  @Override
  public void close() {
    bcast.close();
  }

  @Override
  protected boolean isDone() {
    return sourcesDone && bcast.isComplete();
  }

  @Override
  protected boolean sendMessages(int task, Object data, int flag) {
    while (!bcast.bcast(task, data, flag)) {
      // lets wait a litte and try again
      bcast.progress();
    }
    return true;
  }

  public class BCastReceiver implements BulkReceiver {
    private int lowestTarget = 0;

    @Override
    public void init(Config cfg, Set<Integer> targets) {
      if (targets.isEmpty()) {
        return;
      }
      this.lowestTarget = targets.stream().min(Comparator.comparingInt(o -> (Integer) o)).get();
    }

    @Override
    public boolean receive(int target, Iterator<Object> object) {
      Timing.mark(BenchmarkConstants.TIMING_ALL_RECV,
          workerId == 0 && target == lowestTarget);
      BenchmarkUtils.markTotalTime(resultsRecorder, workerId == 0
          && target == lowestTarget);
      resultsRecorder.writeToCSV();
      verifyResults(resultsVerifier, object, null);
      return true;
    }
  }
}
