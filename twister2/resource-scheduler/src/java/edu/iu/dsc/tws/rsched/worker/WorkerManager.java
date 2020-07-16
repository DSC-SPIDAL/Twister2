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
package edu.iu.dsc.tws.rsched.worker;

import java.util.Map;
import java.util.TreeMap;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.exceptions.JobFaultyException;
import edu.iu.dsc.tws.api.exceptions.TimeoutException;
import edu.iu.dsc.tws.api.exceptions.Twister2RuntimeException;
import edu.iu.dsc.tws.api.faulttolerance.FaultToleranceContext;
import edu.iu.dsc.tws.api.faulttolerance.JobProgress;
import edu.iu.dsc.tws.api.resource.IJobMasterFailureListener;
import edu.iu.dsc.tws.api.resource.IPersistentVolume;
import edu.iu.dsc.tws.api.resource.IVolatileVolume;
import edu.iu.dsc.tws.api.resource.IWorker;
import edu.iu.dsc.tws.api.resource.IWorkerController;
import edu.iu.dsc.tws.api.resource.IWorkerFailureListener;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI;
import edu.iu.dsc.tws.proto.system.job.JobAPI;
import edu.iu.dsc.tws.rsched.core.WorkerRuntime;

/**
 * Keep information about a managed environment where workers can get restarted.
 */
public class WorkerManager implements IWorkerFailureListener, IJobMasterFailureListener {
  private static final Logger LOG = Logger.getLogger(WorkerManager.class.getName());

  /**
   * The IWorker we are working with
   */
  private IWorker managedWorker;

  /**
   * The configuration
   */
  private Config config;

  /**
   * The worker id
   */
  private int workerID;

  /**
   * Job object
   */
  private JobAPI.Job job;

  /**
   * The worker controller
   */
  private IWorkerController workerController;

  /**
   * Persistant volume
   */
  private IPersistentVolume persistentVolume;

  /**
   * The volatile volume
   */
  private IVolatileVolume volatileVolume;

  // job does not become faulty, if faults occur before proceeding the first barrier
  private boolean firstInitBarrierProceeded = false;

  /**
   * Maximum retries
   */
  private final int maxRetries;

  private Map<Integer, JobMasterAPI.WorkerInfo> restartedWorkers = new TreeMap<>();

  public WorkerManager(Config config,
                       JobAPI.Job job,
                       IWorkerController workerController,
                       IPersistentVolume persistentVolume,
                       IVolatileVolume volatileVolume,
                       IWorker worker) {
    this.config = config;
    this.job = job;
    this.workerID = workerController.getWorkerInfo().getWorkerID();
    this.workerController = workerController;
    this.persistentVolume = persistentVolume;
    this.volatileVolume = volatileVolume;
    this.managedWorker = worker;

    // we default to three retries
    this.maxRetries = FaultToleranceContext.maxReExecutes(config);

    WorkerRuntime.addWorkerFailureListener(this);
    WorkerRuntime.addJMFailureListener(this);
    JobProgressImpl.init();
  }

  /**
   * Execute IWorker
   * return false if IWorker fails fully after retries
   * return true if execution successful
   * throw an exception if execution fails and the worker needs to be restarted from jvm
   */
  public boolean execute() {
    while (JobProgress.getWorkerExecuteCount() < maxRetries) {

      LOG.info("Waiting on the init barrier before starting IWorker: " + workerID
          + " with restartCount: " + workerController.workerRestartCount()
          + " and with re-executionCount: " + JobProgress.getWorkerExecuteCount());
      try {
        workerController.waitOnInitBarrier();
        firstInitBarrierProceeded = true;
      } catch (TimeoutException e) {
        throw new Twister2RuntimeException("Could not pass through the init barrier", e);
      }

      LOG.fine("Proceeded through INIT barrier. Starting Worker: " + workerID);
      JobProgressImpl.setJobStatus(JobProgress.JobStatus.EXECUTING);
      JobProgressImpl.increaseWorkerExecuteCount();
      JobProgressImpl.setRestartedWorkers(restartedWorkers.values());
      try {
        managedWorker.execute(
            config, job, workerController, persistentVolume, volatileVolume);
      } catch (JobFaultyException cue) {
        // a worker in the cluster should have failed
        // we will try to re-execute this worker
        JobProgressImpl.setJobStatus(JobProgress.JobStatus.FAULTY);
        LOG.warning("thrown JobFaultyException. Some workers should have failed.");
      }

      // if the job is healthy, it means this worker has finished successfully.
      // we need to make sure whether that all workers finished successfully also
      if (JobProgress.isJobHealthy()) {

        try {
          // wait on the barrier indefinitely until all workers arrive
          // or the barrier is broken with with a job fault
          LOG.info("Worker completed, waiting for other workers to finish at the final barrier.");
          workerController.waitOnBarrier(Long.MAX_VALUE);
          LOG.info("Worker finished successfully");
          return true;
        } catch (TimeoutException e) {
          // this should never happen
          throw new Twister2RuntimeException("Could not pass through the final barrier", e);
        } catch (JobFaultyException e) {
          JobProgressImpl.setJobStatus(JobProgress.JobStatus.FAULTY);
          LOG.warning("thrown JobFaultyException. Some workers failed before finishing.");
        }

      }
    }

    LOG.info(String.format("Re-executed IWorker %d times and failed, we are exiting", maxRetries));
    return false;
  }

  @Override
  public void failed(int wID) {

    // ignore failure events if the first INIT barrier is not proceeded
    if (!firstInitBarrierProceeded) {
      LOG.fine("Worker failure event received before first INIT barrier. Failed worker: " + wID);
      return;
    }

    // job is becoming faulty
    if (JobProgress.isJobHealthy()) {
      faultOccurred(wID);
    }
  }

  @Override
  public void restarted(JobMasterAPI.WorkerInfo workerInfo) {

    // ignore failure events if the first INIT barrier is not proceeded
    if (!firstInitBarrierProceeded) {
      LOG.fine("Worker restarted event received before first INIT barrier. Restarted worker: "
          + workerInfo.getWorkerID());
      return;
    }

    // job is becoming faulty
    if (JobProgress.isJobHealthy()) {
      faultOccurred(workerInfo.getWorkerID());
    }

    restartedWorkers.put(workerInfo.getWorkerID(), workerInfo);
  }

  /**
   * this method must be called exactly once for each fault
   * if there are multiple faults without restoring back to the normal in the middle,
   * it must be called once only
   */
  private void faultOccurred(int wID) {

    // set the status to FAULTY
    JobProgressImpl.setJobStatus(JobProgress.JobStatus.FAULTY);

    // job is becoming faulty, clear restartedWorkers set
    LOG.warning("A fault occurred. Job moves into the FAULTY stage.");
    restartedWorkers.clear();

    JobProgressImpl.faultOccurred(wID);
  }

  @Override
  public void jmFailed() {
    // jm failed events are not propagated currently
  }

  @Override
  public void jmRestarted(String jobMasterAddress) {
    // ignore failure events if the first INIT barrier is not proceeded
    if (!firstInitBarrierProceeded) {
      LOG.warning("Job Master restarted event received before the first INIT barrier. Ignoring");
      return;
    }

    // job is becoming faulty
    if (JobProgress.isJobHealthy()) {
      faultOccurred(-1);
    }
  }
}
