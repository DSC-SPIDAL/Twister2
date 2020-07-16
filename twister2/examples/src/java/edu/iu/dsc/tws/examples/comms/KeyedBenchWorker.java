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

package edu.iu.dsc.tws.examples.comms;

import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.comms.LogicalPlan;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.exceptions.TimeoutException;
import edu.iu.dsc.tws.api.resource.Twister2Worker;
import edu.iu.dsc.tws.api.resource.WorkerEnvironment;
import edu.iu.dsc.tws.examples.Utils;
import edu.iu.dsc.tws.examples.utils.bench.BenchmarkResultsRecorder;
import edu.iu.dsc.tws.examples.utils.bench.Timing;
import edu.iu.dsc.tws.examples.utils.bench.TimingUnit;
import edu.iu.dsc.tws.examples.verification.ExperimentData;
import edu.iu.dsc.tws.examples.verification.ResultsVerifier;
import static edu.iu.dsc.tws.examples.utils.bench.BenchmarkConstants.TIMING_ALL_SEND;
import static edu.iu.dsc.tws.examples.utils.bench.BenchmarkConstants.TIMING_MESSAGE_SEND;

/**
 * BenchWorker class that works with keyed operations
 */
public abstract class KeyedBenchWorker implements Twister2Worker {

  private static final Logger LOG = Logger.getLogger(KeyedBenchWorker.class.getName());

  protected int workerId;

  protected LogicalPlan logicalPlan;

  protected JobParameters jobParameters;

  protected final Map<Integer, Boolean> finishedSources = new HashMap<>();

  protected boolean sourcesDone = false;

  protected ExperimentData experimentData;

  //for verification
  protected int[] inputDataArray;
  private boolean verified = true;

  //to capture benchmark results
  protected BenchmarkResultsRecorder resultsRecorder;

  private long streamWait = 0;

  private WorkerEnvironment workerEnv;

  @Override
  public void execute(WorkerEnvironment workerEnvironment) {

    this.workerEnv = workerEnvironment;
    workerId = workerEnv.getWorkerId();
    Config cfg = workerEnv.getConfig();

    Timing.setDefaultTimingUnit(TimingUnit.NANO_SECONDS);
    this.resultsRecorder = new BenchmarkResultsRecorder(
        cfg,
        workerId == 0
    );

    // create the job parameters
    this.jobParameters = JobParameters.build(cfg);

    // lets create the task plan
    this.logicalPlan = Utils.createStageLogicalPlan(workerEnv, jobParameters.getTaskStages());

    this.inputDataArray = DataGenerator.generateIntData(jobParameters.getSize());

    //collect experiment data
    experimentData = new ExperimentData();
    // now lets execute
    compute(workerEnv);
    // now progress
    progress();
    // wait for the sync
    try {
      workerEnv.getWorkerController().waitOnBarrier();
    } catch (TimeoutException timeoutException) {
      LOG.log(Level.SEVERE, timeoutException.getMessage(), timeoutException);
    }
    // let allows the specific example to close
    close();
    // lets terminate the communicator
    workerEnv.close();
  }

  protected abstract void compute(WorkerEnvironment wEnv);

  protected void progress() {
    // we need to progress the communication
    boolean needProgress = true;

    while (true) {
      boolean seemsDone = !needProgress && isDone();
      if (seemsDone) {
        if (jobParameters.isStream()) {
          if (streamWait == 0) {
            streamWait = System.currentTimeMillis();
          }
          if (streamWait > 0 && (System.currentTimeMillis() - streamWait) > 5000) {
            break;
          }
        } else {
          break;
        }
      } else {
        streamWait = 0;
      }
      // communicationProgress the channel
      workerEnv.getChannel().progress();
      // we should communicationProgress the communication directive
      needProgress = progressCommunication();
    }
  }

  /**
   * This method will verify results and append the output to the results recorder
   */
  protected void verifyResults(ResultsVerifier resultsVerifier, Object results,
                               Map<String, Object> args) {
    if (jobParameters.isDoVerify()) {
      verified = verified && resultsVerifier.verify(results, args);
      //this will record verification failed if any of the iteration fails to verify
      this.resultsRecorder.recordColumn("Verified", verified);
    } else {
      this.resultsRecorder.recordColumn("Verified", "Not Performed");
    }
  }

  public void close() {
  }

  protected abstract boolean progressCommunication();

  protected abstract boolean isDone();

  protected abstract boolean sendMessages(int task, Object key, Object data, int flag);

  protected void finishCommunication(int src) {
  }

  protected class MapWorker implements Runnable {

    private final boolean timingCondition;
    private int task;

    private boolean timingForLowestTargetOnly = false;

    public MapWorker(int task) {
      this.task = task;
      this.timingCondition = workerId == 0 && task == 0;
      Timing.defineFlag(
          TIMING_MESSAGE_SEND,
          jobParameters.getIterations(),
          this.timingCondition
      );
    }

    public void setTimingForLowestTargetOnly(boolean timingForLowestTargetOnly) {
      this.timingForLowestTargetOnly = timingForLowestTargetOnly;
    }

    @Override
    public void run() {
      LOG.info(() -> "Starting map worker: " + workerId + " task: " + task);
      experimentData.setInput(inputDataArray);
      experimentData.setTaskStages(jobParameters.getTaskStages());

      Integer key;
      for (int i = 0; i < jobParameters.getTotalIterations(); i++) {
        // lets generate a message
        key = i;
        int flag = 0;

        if (i == jobParameters.getWarmupIterations()) {
          Timing.mark(TIMING_ALL_SEND, this.timingCondition);
        }

        if (i >= jobParameters.getWarmupIterations()) {
          Timing.mark(TIMING_MESSAGE_SEND,
              this.timingCondition && (!timingForLowestTargetOnly || key == 0));
        }

        sendMessages(task, key, inputDataArray, flag);
      }

      LOG.info(() -> String.format("%d Done sending", workerId));

      synchronized (finishedSources) {
        finishedSources.put(task, true);
        boolean allDone = !finishedSources.values().contains(false);
        finishCommunication(task);
        sourcesDone = allDone;
      }
    }
  }
}
