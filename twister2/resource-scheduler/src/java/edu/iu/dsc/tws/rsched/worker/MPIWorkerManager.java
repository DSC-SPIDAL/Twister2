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

import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.exceptions.JobFaultyException;
import edu.iu.dsc.tws.api.exceptions.TimeoutException;
import edu.iu.dsc.tws.api.exceptions.Twister2RuntimeException;
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
import edu.iu.dsc.tws.rsched.utils.NetworkUtils;

public class MPIWorkerManager implements IWorkerFailureListener, IJobMasterFailureListener {
  private static final Logger LOG = Logger.getLogger(MPIWorkerManager.class.getName());

  // job does not become faulty, if faults occur before proceeding the first barrier
  private boolean firstInitBarrierProceeded = false;

  public MPIWorkerManager() {
    WorkerRuntime.addWorkerFailureListener(this);
    WorkerRuntime.addJMFailureListener(this);
    JobProgressImpl.init();
  }

  public boolean execute(Config config,
                         JobAPI.Job job,
                         IWorkerController workerController,
                         IPersistentVolume persistentVolume,
                         IVolatileVolume volatileVolume,
                         IWorker managedWorker) {

    int workerID = workerController.getWorkerInfo().getWorkerID();

    /*LOG.info("Waiting on the init barrier before starting IWorker: " + workerID
        + " with restartCount: " + workerController.workerRestartCount()
        + " and with re-executionCount: " + JobProgress.getWorkerExecuteCount());*/
    try {
      workerController.waitOnInitBarrier();
      firstInitBarrierProceeded = true;
    } catch (TimeoutException e) {
      throw new Twister2RuntimeException("Could not pass through the init barrier", e);
    }

    // if it is executing for the first time, release worker ports
    if (JobProgress.getWorkerExecuteCount() == 0) {
      NetworkUtils.releaseWorkerPorts();
    }

    JobProgressImpl.setJobStatus(JobProgress.JobStatus.EXECUTING);
    JobProgressImpl.increaseWorkerExecuteCount();

    try {
      managedWorker.execute(
          config, job, workerController, persistentVolume, volatileVolume);
      return true;

    } catch (JobFaultyException jfe) {
      // a worker in the cluster should have failed
      JobProgressImpl.setJobStatus(JobProgress.JobStatus.FAULTY);
      throw jfe;
    }
  }

  @Override
  public void jmFailed() {
    // ignore failure events if the first INIT barrier is not proceeded
    if (!firstInitBarrierProceeded) {
      LOG.warning("Job Master failed event received before the first INIT barrier. Ignoring");
      return;
    }

    faultOccurred(-1);
  }

  @Override
  public void jmRestarted(String jobMasterAddress) {
    // ignore failure events if the first INIT barrier is not proceeded
    if (!firstInitBarrierProceeded) {
      LOG.warning("Job Master restarted event received before the first INIT barrier. Ignoring");
      return;
    }

    faultOccurred(-1);
  }

  @Override
  public void failed(int wID) {
    // ignore failure events if the first INIT barrier is not proceeded
    if (!firstInitBarrierProceeded) {
      LOG.fine("Worker failure event received before first INIT barrier. Failed worker: " + wID);
      return;
    }

    faultOccurred(wID);
  }

  @Override
  public void restarted(JobMasterAPI.WorkerInfo workerInfo) {
    // ignore failure events if the first INIT barrier is not proceeded
    if (!firstInitBarrierProceeded) {
      LOG.fine("Worker restart event received before first INIT barrier. Restarted worker: "
          + workerInfo.getWorkerID());
      return;
    }

    faultOccurred(workerInfo.getWorkerID());
  }

  /**
   * this method must be called exactly once for each fault
   * if there are multiple faults without restoring back to the normal in the middle,
   * it must be called once only
   */
  private void faultOccurred(int wID) {
    // job is becoming faulty
    if (JobProgress.isJobHealthy()) {
      JobProgressImpl.setJobStatus(JobProgress.JobStatus.FAULTY);
      JobProgressImpl.faultOccurred(wID);
      throw new JobFaultyException("Worker[" + wID + "] failed");
    }
  }

}
