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
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.google.protobuf.InvalidProtocolBufferException;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.MPIContext;
import edu.iu.dsc.tws.api.config.SchedulerContext;
import edu.iu.dsc.tws.api.driver.IScalerPerCluster;
import edu.iu.dsc.tws.api.driver.NullScaler;
import edu.iu.dsc.tws.api.exceptions.JobFaultyException;
import edu.iu.dsc.tws.api.exceptions.Twister2Exception;
import edu.iu.dsc.tws.api.exceptions.Twister2RuntimeException;
import edu.iu.dsc.tws.api.faulttolerance.FaultToleranceContext;
import edu.iu.dsc.tws.api.resource.FSPersistentVolume;
import edu.iu.dsc.tws.api.resource.IPersistentVolume;
import edu.iu.dsc.tws.api.resource.IWorker;
import edu.iu.dsc.tws.api.resource.IWorkerController;
import edu.iu.dsc.tws.api.resource.IWorkerStatusUpdater;
import edu.iu.dsc.tws.checkpointing.util.CheckpointingContext;
import edu.iu.dsc.tws.common.config.ConfigLoader;
import edu.iu.dsc.tws.common.logging.LoggingContext;
import edu.iu.dsc.tws.common.logging.LoggingHelper;
import edu.iu.dsc.tws.common.zk.ZKContext;
import edu.iu.dsc.tws.master.JobMasterContext;
import edu.iu.dsc.tws.master.server.JobMaster;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI;
import edu.iu.dsc.tws.proto.system.job.JobAPI;
import edu.iu.dsc.tws.proto.utils.NodeInfoUtils;
import edu.iu.dsc.tws.proto.utils.WorkerInfoUtils;
import edu.iu.dsc.tws.rsched.core.WorkerRuntime;
import edu.iu.dsc.tws.rsched.schedulers.NullTerminator;
import edu.iu.dsc.tws.rsched.utils.JobUtils;
import edu.iu.dsc.tws.rsched.utils.NetworkUtils;
import edu.iu.dsc.tws.rsched.utils.ResourceSchedulerUtils;
import edu.iu.dsc.tws.rsched.worker.MPIWorkerManager;

import mpi.Intracomm;
import mpi.MPI;
import mpi.MPIException;

/**
 * This is the base process started by the resource scheduler.
 * This process will launch Twister2 IWorker
 */
public final class MPIWorkerStarter {
  private static final Logger LOG = Logger.getLogger(MPIWorkerStarter.class.getName());

  /**
   * Configuration
   */
  private Config config;

  /**
   * Information of this worker
   */
  private JobMasterAPI.WorkerInfo wInfo;

  /**
   * Job object describing job resources
   */
  private JobAPI.Job job;

  /**
   * global MPI rank
   *
   * if JM is used and it is not running at the submitting client,
   * then, JM is running at globalRank zero
   */
  private int globalRank;

  /**
   * if this is the MPI process running Twister2 JobMaster,
   * we create a JobMaster object
   */
  private JobMaster jobMaster;

  /**
   * number of time the job is restarted after failures
   * initially, it is zero for the first submission
   */
  private int restartCount;

  public static void main(String[] args) {
    new MPIWorkerStarter(args);
  }

  /**
   * Construct the MPIWorkerStarter
   *
   * @param args the main args
   */
  private MPIWorkerStarter(String[] args) {

    // initialize MPI
    try {
      MPI.InitThread(args, MPI.THREAD_MULTIPLE);
      globalRank = MPI.COMM_WORLD.getRank();
    } catch (MPIException e) {
      LOG.log(Level.SEVERE, "Failed to initialize MPI process", e);
      throw new Twister2RuntimeException("Failed to initialize MPI process", e);
    }

    setUncaughtExceptionHandler();

    // parse command line parameters
    Options cmdOptions = setupOptions();
    CommandLineParser parser = new DefaultParser();
    CommandLine cmd;
    try {
      cmd = parser.parse(cmdOptions, args);
    } catch (ParseException e) {
      HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("MPIWorkerStarter", cmdOptions);
      throw new Twister2RuntimeException("Error parsing command line options: ", e);
    }

    // load the configurations and read Job object
    config = loadConfigurations(cmd);
    LOG.log(Level.FINE, "An MPI worker process is starting with the rank: " + globalRank);

    // set the thread name
    setThreadName();

    if (JobMasterContext.isJobMasterUsed(config)) {
      startWorkerWithJM();
    } else {
      startWorkerWithoutJM(config, MPI.COMM_WORLD);
    }

    finalizeMPI();
  }

  @SuppressWarnings("RegexpSinglelineJava")
  private void setUncaughtExceptionHandler() {
    // on any uncaught exception, we will send worker status message to JM and exit
    // We use system exit to signal MPI master that this worker has failed
    Thread.setDefaultUncaughtExceptionHandler((thread, throwable) -> {
      LOG.log(Level.SEVERE, "Uncaught exception in thread "
          + thread + ". Finalizing this worker...", throwable);

      // if JM is not used, exit JVM
      if (!JobMasterContext.isJobMasterUsed(config)) {
        System.exit(1);
      }

      // if this is the JobMaster
      if (wInfo != null && wInfo.getWorkerID() == -1 && jobMaster != null) {
        jobMaster.jmFailed();
        LOG.severe("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! JM Exiting with failure.");
        System.exit(1);
      }

      // if JM runs in the client, send a FAILED message,
      // if RESTART count reached the limit, send FULLY_FAILED
      if (JobMasterContext.jobMasterRunsInClient(config)) {
        updateWorkerState(JobMasterAPI.WorkerState.FAILED);
        if (restartCount >= FaultToleranceContext.maxMpiJobRestarts(config) - 1) {
          sendWorkerFinalStateToJM(JobMasterAPI.WorkerState.FULLY_FAILED);
        }
        System.exit(1);
      }

      //
      // if JM is running in the first MPI worker
      //
      // it means another worker already failed
      // just exit. Since JM is already done.
      if (throwable instanceof JobFaultyException) {
        WorkerRuntime.close();
        LOG.severe("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Worker Exiting with JobFaultyException failure.");
        System.exit(1);
      } else {
        // first send a FAILED message
        // this message will be broadcasted to all workers
        // then send a FULLY_FAILED, to end the job
        updateWorkerState(JobMasterAPI.WorkerState.FAILED);
        sendWorkerFinalStateToJM(JobMasterAPI.WorkerState.FULLY_FAILED);
        LOG.severe("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Worker Exiting with failure.");
        System.exit(1);
      }
    });
  }

  private void setThreadName() {
    if (JobMasterContext.isJobMasterUsed(config)
        && !JobMasterContext.jobMasterRunsInClient(config)) {

      if (globalRank == 0) {
        Thread.currentThread().setName("Twister2MPIWorker-JM");
      } else {
        Thread.currentThread().setName("Twister2MPIWorker-" + (globalRank - 1));
      }
    } else {
      Thread.currentThread().setName("Twister2MPIWorker-" + globalRank);
    }
  }

  /**
   * Start the worker without JM
   *
   * @param cfg configuration
   * @param intracomm communication
   */
  private void startWorkerWithoutJM(Config cfg, Intracomm intracomm) {
    wInfo = createWorkerInfo(config, globalRank);
    Map<Integer, JobMasterAPI.WorkerInfo> infos = createWorkerInfoMap(intracomm);
    MPIWorkerController wc = new MPIWorkerController(globalRank, infos, restartCount);
    WorkerRuntime.init(cfg, wc);
    startWorker(intracomm);
  }

  private void startWorkerWithJM() {

    if (JobMasterContext.jobMasterRunsInClient(config)) {
      wInfo = createWorkerInfo(config, globalRank);
      WorkerRuntime.init(config, job, wInfo, restartCount);
      startWorker(MPI.COMM_WORLD);
    } else {
      // lets broadcast the worker info
      // broadcast the port of job master

      // when JM is not running in the submitting client,
      // it is running at rank 0 of MPI world
      // Split JM MPI world and worker MPI worlds
      int color = globalRank == 0 ? 0 : 1;

      int splittedRank;
      Intracomm splittedComm;
      try {
        splittedComm = MPI.COMM_WORLD.split(color, globalRank);
        splittedRank = splittedComm.getRank();
      } catch (MPIException e) {
        throw new Twister2RuntimeException("Can not split MPI.COMM_WORLD", e);
      }

      if (globalRank == 0) {
        wInfo = createWorkerInfo(config, -1);
      } else {
        wInfo = createWorkerInfo(config, splittedRank);
      }

      // broadcast the job master information to all workers
      broadCastMasterInformation(globalRank);

      if (globalRank == 0) {
        startMaster();
      } else {
        // init WorkerRuntime
        WorkerRuntime.init(config, job, wInfo, restartCount);
        startWorker(splittedComm);
      }
    }
  }

  /**
   * Start the worker
   *
   * @param intracomm communication
   */
  private void startWorker(Intracomm intracomm) {
    try {
      // initialize the logger
      initWorkerLogger(config, intracomm.getRank());

      // now create the worker
      IWorkerController wc = WorkerRuntime.getWorkerController();
      IPersistentVolume persistentVolume = initPersistenceVolume(config, globalRank);

      MPIContext.addRuntimeObject("comm", intracomm);
      IWorker worker = JobUtils.initializeIWorker(job);
      MPIWorkerManager workerManager = new MPIWorkerManager();
      workerManager.execute(config, job, wc, persistentVolume, null, worker);
    } catch (MPIException e) {
      LOG.log(Level.SEVERE, "Failed to synchronize the workers at the start");
      throw new RuntimeException(e);
    }
  }

  /**
   * Start the JobMaster
   */
  private void startMaster() {
    try {
      // init the logger
      initJMLogger(config);

      // release the port for JM
      NetworkUtils.releaseWorkerPorts();

      int port = JobMasterContext.jobMasterPort(config);
      String hostAddress = ResourceSchedulerUtils.getHostIP(config);
      LOG.log(Level.INFO, String.format("Starting the job master: %s:%d", hostAddress, port));
      JobMasterAPI.NodeInfo jobMasterNodeInfo = null;
      IScalerPerCluster clusterScaler = new NullScaler();
      JobMasterAPI.JobMasterState initialState = JobMasterAPI.JobMasterState.JM_STARTED;
      NullTerminator nt = new NullTerminator();

      jobMaster = new JobMaster(
          config, "0.0.0.0", port, nt, job, jobMasterNodeInfo, clusterScaler, initialState);
      jobMaster.startJobMasterBlocking();
      LOG.log(Level.INFO, "JobMaster done... ");
    } catch (Twister2Exception e) {
      LOG.log(Level.SEVERE, "Exception when starting Job master: ", e);
      throw new RuntimeException(e);
    }
  }

  /**
   * Broadcast the job master information to workers
   *
   * @param rank rank
   */
  private void broadCastMasterInformation(int rank) {
    byte[] workerBytes = wInfo.toByteArray();
    int length = workerBytes.length;
    IntBuffer countSend = MPI.newIntBuffer(1);
    if (rank == 0) {
      countSend.put(length);
    }

    try {
      MPI.COMM_WORLD.bcast(countSend, 1, MPI.INT, 0);
      length = countSend.get(0);

      ByteBuffer sendBuffer = MPI.newByteBuffer(length);
      if (rank == 0) {
        sendBuffer.put(workerBytes);
      }
      MPI.COMM_WORLD.bcast(sendBuffer, length, MPI.BYTE, 0);
      byte[] jmInfoBytes = new byte[length];
      if (rank != 0) {
        sendBuffer.get(jmInfoBytes);
        JobMasterAPI.WorkerInfo masterInfo = JobMasterAPI.WorkerInfo.
            newBuilder().mergeFrom(jmInfoBytes).build();
        config = Config.newBuilder().putAll(config).
            put(JobMasterContext.JOB_MASTER_PORT, masterInfo.getPort()).
            put(JobMasterContext.JOB_MASTER_IP, masterInfo.getNodeInfo().getNodeIP()).build();
      } else {
        config = Config.newBuilder().putAll(config).
            put(JobMasterContext.JOB_MASTER_PORT, wInfo.getPort()).
            put(JobMasterContext.JOB_MASTER_IP, wInfo.getNodeInfo().getNodeIP()).build();
      }
    } catch (MPIException mpie) {
      throw new Twister2RuntimeException("Error when broadcasting Job Master information", mpie);
    } catch (InvalidProtocolBufferException ipbe) {
      throw new Twister2RuntimeException("Error when decoding Job Master information", ipbe);
    }
  }

  /**
   * Setup the command line options for the MPI process
   *
   * @return cli options
   */
  private Options setupOptions() {
    Options options = new Options();

    Option configDirectory = Option.builder("d")
        .desc("The class name of the container to launch")
        .longOpt("config_dir")
        .hasArgs()
        .argName("configuration directory")
        .required()
        .build();

    Option twister2Home = Option.builder("t")
        .desc("The class name of the container to launch")
        .longOpt("twister2_home")
        .hasArgs()
        .argName("twister2 home")
        .required()
        .build();

    Option clusterType = Option.builder("n")
        .desc("The clustr type")
        .longOpt("cluster_type")
        .hasArgs()
        .argName("cluster type")
        .required()
        .build();

    Option jobId = Option.builder("j")
        .desc("Job Id")
        .longOpt("job_id")
        .hasArgs()
        .argName("job id")
        .required()
        .build();

    Option jobMasterIP = Option.builder("i")
        .desc("Job master ip")
        .longOpt("job_master_ip")
        .hasArgs()
        .argName("job master ip")
        .required()
        .build();

    Option jobMasterPort = Option.builder("p")
        .desc("Job master ip")
        .longOpt("job_master_port")
        .hasArgs()
        .argName("job master port")
        .required()
        .build();

    Option restoreJob = Option.builder("r")
        .desc("Whether the job is being restored")
        .longOpt("restore_job")
        .hasArgs()
        .argName("restore job")
        .required()
        .build();

    Option restartCountOption = Option.builder("x")
        .desc("number of time the job is restarted after failure")
        .longOpt("restart_count")
        .hasArgs()
        .argName("restart count")
        .required()
        .build();

    options.addOption(twister2Home);
    options.addOption(configDirectory);
    options.addOption(clusterType);
    options.addOption(jobId);
    options.addOption(jobMasterIP);
    options.addOption(jobMasterPort);
    options.addOption(restoreJob);
    options.addOption(restartCountOption);

    return options;
  }

  private Config loadConfigurations(CommandLine cmd) {
    String twister2Home = cmd.getOptionValue("twister2_home");
    String configDir = cmd.getOptionValue("config_dir");
    String clusterType = cmd.getOptionValue("cluster_type");
    String jobId = cmd.getOptionValue("job_id");
    String jIp = cmd.getOptionValue("job_master_ip");
    int jPort = Integer.parseInt(cmd.getOptionValue("job_master_port"));
    boolean restoreJob = Boolean.parseBoolean(cmd.getOptionValue("restore_job"));
    restartCount = Integer.parseInt(cmd.getOptionValue("restart_count"));

    Config cfg = ConfigLoader.loadConfig(twister2Home, configDir, clusterType);

    Config workerConfig = Config.newBuilder()
        .putAll(cfg)
        .put(MPIContext.TWISTER2_HOME.getKey(), twister2Home)
        .put(MPIContext.TWISTER2_CONTAINER_ID, globalRank)
        .put(MPIContext.TWISTER2_CLUSTER_TYPE, clusterType)
        .build();

    String jobDescFile = JobUtils.getJobDescriptionFilePath(jobId, workerConfig);
    job = JobUtils.readJobFile(jobDescFile);

    Config updatedConfig = JobUtils.overrideConfigs(job, cfg);

    updatedConfig = Config.newBuilder()
        .putAll(updatedConfig)
        .put(MPIContext.TWISTER2_HOME.getKey(), twister2Home)
        .put(MPIContext.WORKER_CLASS, job.getWorkerClassName())
        .put(MPIContext.TWISTER2_CONTAINER_ID, globalRank)
        .put(MPIContext.JOB_ID, jobId)
        .put(MPIContext.JOB_OBJECT, job)
        .put(MPIContext.TWISTER2_CLUSTER_TYPE, clusterType)
        .put(JobMasterContext.JOB_MASTER_IP, jIp)
        .put(JobMasterContext.JOB_MASTER_PORT, jPort)
        .put(ZKContext.SERVER_ADDRESSES, null)
        .put(CheckpointingContext.CHECKPOINTING_RESTORE_JOB, restoreJob)
        .build();

    LOG.log(Level.FINE, String.format("Initializing process with "
            + "twister_home: %s worker_class: %s config_dir: %s cluster_type: %s",
        twister2Home, job.getWorkerClassName(), configDir, clusterType));

    return updatedConfig;
  }

  /**
   * create a AllocatedResources
   *
   * @return a map of rank to hostname
   */
  public Map<Integer, JobMasterAPI.WorkerInfo> createWorkerInfoMap(Intracomm intracomm) {
    try {
      byte[] workerBytes = wInfo.toByteArray();
      int length = workerBytes.length;

      IntBuffer countSend = MPI.newIntBuffer(1);
      int worldSize = intracomm.getSize();
      IntBuffer countReceive = MPI.newIntBuffer(worldSize);
      // now calculate the total number of characters
      countSend.put(length);
      intracomm.allGather(countSend, 1, MPI.INT, countReceive, 1, MPI.INT);

      int[] receiveSizes = new int[worldSize];
      int[] displacements = new int[worldSize];
      int sum = 0;
      for (int i = 0; i < worldSize; i++) {
        receiveSizes[i] = countReceive.get(i);
        displacements[i] = sum;
        sum += receiveSizes[i];
      }
      // now we need to send this to all the nodes
      ByteBuffer sendBuffer = MPI.newByteBuffer(length);
      ByteBuffer receiveBuffer = MPI.newByteBuffer(sum);
      sendBuffer.put(workerBytes);

      // now lets receive the process names of each rank
      intracomm.allGatherv(sendBuffer, length, MPI.BYTE, receiveBuffer,
          receiveSizes, displacements, MPI.BYTE);

      Map<Integer, JobMasterAPI.WorkerInfo> workerInfoMap = new HashMap<>();
      for (int i = 0; i < receiveSizes.length; i++) {
        byte[] c = new byte[receiveSizes[i]];
        receiveBuffer.get(c);
        JobMasterAPI.WorkerInfo info = JobMasterAPI.WorkerInfo.newBuilder().mergeFrom(c).build();
        workerInfoMap.put(i, info);
        LOG.log(Level.FINE, String.format("Worker %d info: %s", i, workerInfoMap.get(i)));
      }
      return workerInfoMap;
    } catch (MPIException e) {
      throw new RuntimeException("Failed to communicate", e);
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException("Failed to create worker info", e);
    }
  }

  /**
   * Create worker information
   *
   * @param cfg configuration
   * @param workerId communicator
   * @return the worker information
   * @throws MPIException if an error occurs
   */
  private JobMasterAPI.WorkerInfo createWorkerInfo(Config cfg, int workerId) {
    InetAddress workerIP;
    List<String> networkInterfaces = SchedulerContext.networkInterfaces(cfg);
    if (networkInterfaces == null) {
      try {
        workerIP = InetAddress.getLocalHost();
      } catch (UnknownHostException e) {
        throw new RuntimeException("Failed to get ip address", e);
      }
    } else {
      workerIP = ResourceSchedulerUtils.getLocalIPFromNetworkInterfaces(networkInterfaces);
      if (workerIP == null) {
        throw new RuntimeException("Failed to get ip address from network interfaces: "
            + networkInterfaces);
      }
    }

    JobMasterAPI.NodeInfo nodeInfo =
        NodeInfoUtils.createNodeInfo(workerIP.getHostAddress(), "default", "default");
    JobAPI.ComputeResource computeResource = JobUtils.getComputeResource(job, workerId);
    List<String> portNames = SchedulerContext.additionalPorts(cfg);
    if (portNames == null) {
      portNames = new ArrayList<>();
    }
    portNames.add("__worker__");

    Map<String, Integer> freePorts = NetworkUtils.findFreePorts(portNames, config, workerIP);
    Integer workerPort = freePorts.remove("__worker__");
    LOG.fine("Worker info host:" + workerIP + ":" + workerPort);
    return WorkerInfoUtils.createWorkerInfo(
        workerId, workerIP.getHostAddress(), workerPort, nodeInfo, computeResource, freePorts);
  }

  /**
   * Initialize the loggers to log into the task local directory
   *
   * @param cfg the configuration
   * @param workerID worker id
   */
  public static void initWorkerLogger(Config cfg, int workerID) {
    // we can not initialize the logger fully yet,
    // but we need to set the format as the first thing
    // LoggingHelper.setLoggingFormat(LoggingHelper.DEFAULT_FORMAT);

    // set logging level
    // LoggingHelper.setLogLevel(LoggingContext.loggingLevel(cfg));

    String logDir = LoggingContext.loggingDir(cfg);
    File directory = new File(logDir);
    if (!directory.exists()) {
      if (!directory.mkdirs()) {
        // this worker may have failed to created the directory,
        // but another worker may have succeeded.
        // test it to make sure
        if (!directory.exists()) {
          throw new RuntimeException("Failed to create log directory: " + logDir);
        }
      }
    }
    LoggingHelper.setupLogging(cfg, logDir, "worker-" + workerID);
    LOG.fine(String.format("Logging is setup with the file %s", logDir));
  }

  /**
   * Initialize the logger for job master
   *
   * @param cfg the configuration
   */
  public static void initJMLogger(Config cfg) {
    // we can not initialize the logger fully yet,
    // but we need to set the format as the first thing
    // LoggingHelper.setLoggingFormat(LoggingHelper.DEFAULT_FORMAT);

    // set logging level
    // LoggingHelper.setLogLevel(LoggingContext.loggingLevel(cfg));

    String logDir = LoggingContext.loggingDir(cfg);
    File directory = new File(logDir);
    if (!directory.exists()) {
      if (!directory.mkdirs()) {
        throw new RuntimeException("Failed to create log directory: " + logDir);
      }
    }
    LoggingHelper.setupLogging(cfg, logDir, "job-master");
    LOG.fine(String.format("Logging is setup with the file %s", logDir));
  }

  private IPersistentVolume initPersistenceVolume(Config cfg, int rank) {
    File baseDir = new File(MPIContext.fileSystemMount(cfg));

    // if the base dir does not exist
    while (!baseDir.exists() && !baseDir.mkdirs()) {
      try {
        Thread.sleep(100);
      } catch (InterruptedException e) {
        throw new RuntimeException("Thread interrupted", e);
      }
    }

    return new FSPersistentVolume(baseDir.getAbsolutePath(), rank);
  }

  public void finalizeMPI() {
    try {
      sendWorkerFinalStateToJM(JobMasterAPI.WorkerState.COMPLETED);
      MPI.Finalize();
    } catch (MPIException ignore) {
    }
  }

  /**
   * if Job Master is used and this is not JM,
   * propagate worker state to Job Master
   */
  private void updateWorkerState(JobMasterAPI.WorkerState workerState) {

    // if JM is not used, nothing to send
    if (!JobMasterContext.isJobMasterUsed(config)) {
      return;
    }

    // if this is JM, nothing to send
    if (wInfo.getWorkerID() == -1) {
      return;
    }

    // if JM is used and this is a worker,
    // send worker final state to the Job Master and close WorkerRuntime
    IWorkerStatusUpdater workerStatusUpdater = WorkerRuntime.getWorkerStatusUpdater();
    if (workerStatusUpdater != null) {
      workerStatusUpdater.updateWorkerStatus(workerState);
    }
  }

  /**
   * if Job Master is used,
   * propagate final worker state to Job Master
   */
  private void sendWorkerFinalStateToJM(JobMasterAPI.WorkerState workerState) {
//    LOG.info(String.format("Worker-%d finished executing with the final status: %s",
//        wInfo.getWorkerID(), workerState.name()));

    updateWorkerState(workerState);
    WorkerRuntime.close();
  }

/* //todo: change this and include this in the checkpoint mgr
  private Config updateCheckpointInfo(Config cfg) {

    String checkpointDir = CheckpointingContext.checkpointDir(cfg);

    new File(checkpointDir).mkdirs();

    File checkpointFile = new File(checkpointDir + File.separator
        + CheckpointingContext.LAST_SNAPSHOT_FILE);

    long checkpointID = 0;
    if (checkpointFile.exists()) {
      try {
        checkpointID = Long.valueOf(new String(Files.readAllBytes(checkpointFile.toPath())).trim());
      } catch (IOException e) {
        throw new RuntimeException("Unable to read the checkpoint file", e);
      }
    }

    return Config.newBuilder()
        .putAll(cfg)
        .put(CheckpointingContext.RESTORE_SNAPSHOT, checkpointID)
        .build();
  }*/
}
