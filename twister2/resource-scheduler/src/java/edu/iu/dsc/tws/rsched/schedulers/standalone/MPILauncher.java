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
import java.io.FileFilter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.io.filefilter.WildcardFileFilter;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.Context;
import edu.iu.dsc.tws.api.config.SchedulerContext;
import edu.iu.dsc.tws.api.driver.DriverJobState;
import edu.iu.dsc.tws.api.driver.IScalerPerCluster;
import edu.iu.dsc.tws.api.driver.NullScalar;
import edu.iu.dsc.tws.api.exceptions.Twister2Exception;
import edu.iu.dsc.tws.api.scheduler.IController;
import edu.iu.dsc.tws.api.scheduler.ILauncher;
import edu.iu.dsc.tws.api.scheduler.Twister2JobState;
import edu.iu.dsc.tws.common.util.NetworkUtils;
import edu.iu.dsc.tws.common.zk.ZKContext;
import edu.iu.dsc.tws.master.JobMasterContext;
import edu.iu.dsc.tws.master.server.JobMaster;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI;
import edu.iu.dsc.tws.proto.system.job.JobAPI;
import edu.iu.dsc.tws.proto.utils.NodeInfoUtils;
import edu.iu.dsc.tws.rsched.core.ResourceRuntime;
import edu.iu.dsc.tws.rsched.schedulers.nomad.NomadTerminator;
import edu.iu.dsc.tws.rsched.utils.FileUtils;
import edu.iu.dsc.tws.rsched.utils.ProcessUtils;
import edu.iu.dsc.tws.rsched.utils.ResourceSchedulerUtils;

public class MPILauncher implements ILauncher {

  private static final Logger LOG = Logger.getLogger(MPILauncher.class.getName());

  private Config config;

  private String jobWorkingDirectory;

  @Override
  public void initialize(Config mConfig) {
    this.config = mConfig;

    // get the job working directory
    this.jobWorkingDirectory = MPIContext.workingDirectory(mConfig);

    if (ZKContext.isZooKeeperServerUsed(config)) {
      LOG.warning("ZooKeeper is not supported in Standalone mode. Ignoring it.");
      config = Config.newBuilder()
          .putAll(config)
          .put(ZKContext.SERVER_ADDRESSES, null)
          .build();
    }
  }

  @Override
  public void close() {

  }

  @Override
  public boolean killJob(String jobID) {
    // not implemented yet
    return false;
  }

  /**
   * This method performs following tasks in order to distribute file among worker nodes
   * <li>Picks job file and core file from local source root</li>
   * <li>Calculate MD5s of both files</li>
   * <li>Create a temporary hostfile to submit with mpirun command. This host file grantees to
   * spawn just 1 process per worker node (setting slots=1). This prevents concurrent writes
   * to the disk of the worker nodes</li>
   * <li>Spawn a new mpi job by calling bootstrap.sh. This job runs
   * {@link edu.iu.dsc.tws.rsched.schedulers.standalone.bootstrap.MPIBootstrap} on all workers</li>
   */
  private void distributeJobFiles(JobAPI.Job job) throws IOException {
    File localSourceRoot = new File(
        this.config.getStringValue(SchedulerContext.TEMPORARY_PACKAGES_PATH)
    );

    File jobFile = new File(
        localSourceRoot,
        "twister2-job.tar.gz"
    );

    LOG.info(String.format("Found Job file : %s", jobFile.getAbsolutePath()));

    String jobFileMD5 = FileUtils.md5(jobFile);
    String jobFilePath = jobFile.getAbsolutePath();

    String coreFileMD5 = "NA";
    String coreFilePath = "NA";

    /**
     * Not copying core in bootstrap. ResourceAllocator will do this instead.
     * todo remove additional if/else if copyCore = false seems okay
     */
    boolean copyCore = false; //SchedulerContext.copySystemPackage(config);

    if (copyCore) {
      //finding twister2 core
      FileFilter fileFilter = new WildcardFileFilter("twister2-core-*.*.*.tar.gz");
      File[] files = localSourceRoot.listFiles(fileFilter);

      if (files == null || files.length == 0) {
        throw new RuntimeException("Couldn't find twister2 core at "
            + localSourceRoot.getAbsolutePath());
      }

      File coreFile = files[0];

      LOG.info(String.format("Found Core file : %s", coreFile.getAbsolutePath()));

      coreFilePath = coreFile.getAbsolutePath();
    }

    Path tempHotsFile = Files.createTempFile("hosts-" + job.getJobId(), "");
    int np = this.createOneSlotPerNodeFile(tempHotsFile);

    String mpiRunFile = MPIContext.mpiRunFile(config);

    StringBuilder stringBuilder = new StringBuilder();
    int status = ProcessUtils.runSyncProcess(
        false,
        new String[]{
            "conf/standalone/bootstrap.sh",
            Integer.toString(np),
            tempHotsFile.toAbsolutePath().toString(),
            job.getJobId(),
            this.jobWorkingDirectory,
            jobFilePath,
            jobFileMD5,
            Boolean.toString(copyCore),
            coreFilePath,
            coreFileMD5,
            mpiRunFile,
            SchedulerContext.twister2Home(config)
        },
        stringBuilder,
        new File("."),
        true
    );

    if (status != 0) {
      LOG.severe("Failed to execute bootstrap procedure : " + status);
      throw new RuntimeException("Bootstrap procedure failed with status " + status);
    } else {
      if (stringBuilder.length() != 0) {
        LOG.severe("Bootstrap procedure failed with error : " + stringBuilder.toString());
        throw new RuntimeException("Bootstrap procedure failed with error "
            + stringBuilder.toString());
      } else {
        LOG.info("Bootstrap procedure executed successfully.");
      }
    }
  }

  private int createOneSlotPerNodeFile(Path tempHostFile) throws IOException {
    List<String> hosts = Files.readAllLines(new File("./conf/standalone/nodes").toPath());
    StringBuilder hostFileBuilder = new StringBuilder();
    int ipCount = 0;
    for (String host : hosts) {
      String[] parts = host.split(" ");
      if (parts.length > 0 && !parts[0].trim().isEmpty()) {
        ipCount++;
        hostFileBuilder
            .append(parts[0])
            .append(" ")
            .append("slots=1")
            .append(System.getProperty("line.separator"));
      }
    }
    Files.write(tempHostFile, hostFileBuilder.toString().getBytes());
    return ipCount;
  }

  @Override
  public Twister2JobState launch(JobAPI.Job job) {
    LOG.log(Level.INFO, "Launching job for cluster {0}",
        MPIContext.clusterType(config));
    Twister2JobState state = new Twister2JobState(false);

    //distributing bundle if not running in shared file system
    if (!MPIContext.isSharedFs(config)) {
      LOG.info("Configured as NON SHARED file system. "
          + "Running bootstrap procedure to distribute files...");
      try {
        this.distributeJobFiles(job);
      } catch (IOException e) {
        LOG.log(Level.SEVERE, "Error in distributing job files", e);
        throw new RuntimeException("Error in distributing job files");
      }
    } else {
      LOG.info("Configured as SHARED file system. "
          + "Skipping bootstrap procedure & setting up working directory");
      if (!setupWorkingDirectory(job.getJobId())) {
        throw new RuntimeException("Failed to setup the directory");
      }
    }

    config = Config.newBuilder().putAll(config).put(
        SchedulerContext.WORKING_DIRECTORY, jobWorkingDirectory).build();

    JobMaster jobMaster = null;
    Thread jmThread = null;
    if (JobMasterContext.isJobMasterUsed(config)
        && JobMasterContext.jobMasterRunsInClient(config)) {
      // Since the job master is running on client we can collect job information
      state.setDetached(false);

      try {
        int port = NetworkUtils.getFreePort();
        String hostAddress = JobMasterContext.jobMasterIP(config);
        if (hostAddress == null) {
          hostAddress = ResourceSchedulerUtils.getHostIP(config);
        }
        // add the port and ip to config
        config = Config.newBuilder().putAll(config).put("__job_master_port__", port).
            put("__job_master_ip__", hostAddress).build();

        LOG.log(Level.INFO, String.format("Starting the job master: %s:%d", hostAddress, port));
        JobMasterAPI.NodeInfo jobMasterNodeInfo = NodeInfoUtils.createNodeInfo(hostAddress,
            "default", "default");

        IScalerPerCluster nullScaler = new NullScalar();
        JobMasterAPI.JobMasterState initialState = JobMasterAPI.JobMasterState.JM_STARTED;
        NomadTerminator nt = new NomadTerminator();

        jobMaster = new JobMaster(
            config, "0.0.0.0", port, nt, job, jobMasterNodeInfo, nullScaler, initialState);
        jobMaster.addShutdownHook(true);
        jmThread = jobMaster.startJobMasterThreaded();
        ResourceRuntime.getInstance().setJobMasterHostPort(hostAddress, port);
      } catch (Twister2Exception e) {
        LOG.log(Level.SEVERE, "Exception when starting Job master: ", e);
        throw new RuntimeException(e);
      }
    }

    final boolean[] start = {false};
    // now start the controller, which will get the resources and start
    Thread controllerThread = new Thread(() -> {
      IController controller = new MPIController(true);
      controller.initialize(config);
      start[0] = controller.start(job);
    });
    controllerThread.start();
    // wait until the controller finishes
    try {
      controllerThread.join();
    } catch (InterruptedException ignore) {
    }

    // now lets wait on client
    if (jmThread != null && JobMasterContext.isJobMasterUsed(config)
        && JobMasterContext.jobMasterRunsInClient(config)) {
      try {
        jmThread.join();
      } catch (InterruptedException ignore) {
      }
    }
    if (jobMaster != null && jobMaster.getDriver() != null) {
      if (jobMaster.getDriver().getState() != DriverJobState.FAILED) {
        state.setJobstate(DriverJobState.COMPLETED);
      } else {
        state.setJobstate(jobMaster.getDriver().getState());
      }
      state.setFinalMessages(jobMaster.getDriver().getMessages());
    }
    state.setRequestGranted(start[0]);
    return state;
  }

  /**
   * setup the working directory mainly it downloads and extracts the job package
   * to the working directory
   *
   * @return false if setup fails
   */
  protected boolean setupWorkingDirectory(String jobId) {
    // get the path of core release URI
    String corePackage = MPIContext.corePackageFileName(config);

    // Form the job package's URI
    String jobPackageURI = MPIContext.jobPackageUri(config).toString();

    // copy the files to the working directory
    return ResourceSchedulerUtils.setupWorkingDirectory(
        jobId,
        jobWorkingDirectory,
        corePackage,
        jobPackageURI,
        Context.verbose(config),
        SchedulerContext.copySystemPackage(config));
  }
}
