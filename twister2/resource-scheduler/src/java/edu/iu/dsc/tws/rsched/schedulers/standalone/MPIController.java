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
package edu.iu.dsc.tws.rsched.schedulers.standalone;

import java.io.File;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.MPIContext;
import edu.iu.dsc.tws.api.faulttolerance.FaultToleranceContext;
import edu.iu.dsc.tws.api.scheduler.IController;
import edu.iu.dsc.tws.proto.system.job.JobAPI;
import edu.iu.dsc.tws.rsched.utils.ProcessUtils;

public class MPIController implements IController {
  private static final Logger LOG = Logger.getLogger(MPIController.class.getName());

  private final boolean isVerbose;

  MPIController(boolean isVerbose) {
    this.isVerbose = isVerbose;
  }

  private Config config;
  private String workingDirectory;
  private boolean standalone = true;
  private MPICommand command;

  @Override
  public void initialize(Config mConfig) {
    this.config = Config.transform(mConfig);

    // get the job working directory
    this.workingDirectory = MPIContext.workingDirectory(config);
    LOG.log(Level.INFO, "Working directory: " + workingDirectory);

    // lets set the mode
    this.standalone = "standalone".equals(MPIContext.mpiMode(mConfig));

    // lets creaete the command accordingly
    if (standalone) {
      command = new StandaloneCommand(mConfig, workingDirectory);
    } else {
      command = new SlurmCommand(mConfig, workingDirectory);
    }
  }

  @Override
  public void close() {
    // Nothing to do here
  }

  @Override
  public boolean start(JobAPI.Job job) {
    if (job == null || job.getNumberOfWorkers() == 0) {
      LOG.log(Level.SEVERE, "No worker requested. Can't deploy the job");
      return false;
    }
    LOG.log(Level.INFO, String.format("Launching job in %s scheduler with no of workers = %d",
        MPIContext.clusterType(config), job.getNumberOfWorkers()));

    String jobDirectory = Paths.get(this.workingDirectory, job.getJobId()).toString();
    boolean jobCreated = createJob(this.workingDirectory, jobDirectory, job);

    if (!jobCreated) {
      LOG.log(Level.SEVERE, "Failed to create job");
    } else {
      LOG.log(Level.FINE, "Job created successfully");
    }
    return jobCreated;
  }

  @Override
  public boolean kill(JobAPI.Job job) {
    String[] killCommand = command.killCommand();

    StringBuilder stderr = new StringBuilder();
    runProcess(workingDirectory, killCommand, stderr);

    if (!stderr.toString().equals("")) {
      LOG.log(Level.SEVERE, "Failed to kill the job");
    }
    return false;
  }

  /**
   * Create a slurm job. Use the slurm scheduler's sbatch command to submit the job.
   * sbatch allocates the nodes and runs the script specified by slurmScript.
   * This script runs the twister2 executor on each of the nodes allocated.
   *
   * @param jobWorkingDirectory working directory
   * @return true if the job creation is successful
   */
  public boolean createJob(String jobWorkingDirectory, String twister2Home, JobAPI.Job job) {
    String[] mpiCmd = command.mpiCommand(jobWorkingDirectory, job);
    LOG.fine("Executing job [" + jobWorkingDirectory + "]: " + Arrays.toString(mpiCmd));
    StringBuilder stderr = new StringBuilder();
    return runProcess(twister2Home, mpiCmd, stderr);
  }

  /**
   * Submit the job
   * resubmit it if it fails
   */
  protected boolean runProcess(String jobWorkingDirectory, String[] cmd, StringBuilder stderr) {
    File workingDir = jobWorkingDirectory == null ? null : new File(jobWorkingDirectory);

    int tryCount = 0;

    while (tryCount++ < FaultToleranceContext.maxMpiJobRestarts(config)) {
      int statusCode = ProcessUtils.runSyncProcess(false, cmd, stderr, workingDir, true);
      if (statusCode == 0) {
        LOG.info("MPI job succeeded.");
        return true;
      } else if (tryCount < FaultToleranceContext.maxMpiJobRestarts(config)) {
        LOG.severe(
            "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                + "\nFailed to execute mpirun. Will try again. STDERR: " + stderr.toString()
                + "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");

        // clear error buffer
        stderr.setLength(0);

        // update restartCount at the mpi command
        command.updateRestartCount(cmd, tryCount);
      }
    }

    return false;
  }
}
