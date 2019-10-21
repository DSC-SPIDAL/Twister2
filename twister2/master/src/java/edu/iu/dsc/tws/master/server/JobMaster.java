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

package edu.iu.dsc.tws.master.server;

import java.io.IOException;
import java.nio.channels.SocketChannel;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.checkpointing.StateStore;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.exceptions.Twister2RuntimeException;
import edu.iu.dsc.tws.api.faulttolerance.FaultToleranceContext;
import edu.iu.dsc.tws.api.net.StatusCode;
import edu.iu.dsc.tws.api.net.request.ConnectHandler;
import edu.iu.dsc.tws.checkpointing.master.CheckpointManager;
import edu.iu.dsc.tws.checkpointing.util.CheckpointUtils;
import edu.iu.dsc.tws.checkpointing.util.CheckpointingConfigurations;
import edu.iu.dsc.tws.common.driver.IDriver;
import edu.iu.dsc.tws.common.driver.IScalerPerCluster;
import edu.iu.dsc.tws.common.net.tcp.Progress;
import edu.iu.dsc.tws.common.net.tcp.request.RRServer;
import edu.iu.dsc.tws.common.util.ReflectionUtils;
import edu.iu.dsc.tws.common.zk.ZKContext;
import edu.iu.dsc.tws.common.zk.ZKMasterController;
import edu.iu.dsc.tws.master.IJobTerminator;
import edu.iu.dsc.tws.master.JobMasterContext;
import edu.iu.dsc.tws.master.dashclient.DashboardClient;
import edu.iu.dsc.tws.master.dashclient.models.JobState;
import edu.iu.dsc.tws.master.driver.DriverMessenger;
import edu.iu.dsc.tws.master.driver.Scaler;
import edu.iu.dsc.tws.master.driver.ZKJobUpdater;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI.ListWorkersRequest;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI.ListWorkersResponse;
import edu.iu.dsc.tws.proto.system.job.JobAPI;

/**
 * JobMaster class
 * It is started for each Twister2 job
 * It provides:
 * worker discovery
 * barrier method
 * Ping service
 * <p>
 * It can be started in two different modes:
 * Threaded and Blocking
 * <p>
 * If the user calls:
 * startJobMasterThreaded()
 * It starts as a Thread and the call to this method returns
 * <p>
 * If the user calls:
 * startJobMasterBlocking()
 * It uses the calling thread and this call does not return unless the JobMaster completes
 * <p>
 * JobMaster to Dashboard messaging
 * JobMaster reports to Dashboard server when dashboard address is provided in the config
 * If dashboard host address is not provided, it does not try to connect to dashboard server
 */

public class JobMaster {
  private static final Logger LOG = Logger.getLogger(JobMaster.class.getName());

  /**
   * an id to be used when comminicating with workers and the client
   */
  public static final int JOB_MASTER_ID = -10;

  /**
   * A singleton Progress object monitors network channel
   */
  private static Progress looper;

  /**
   * config object for the Job Master
   */
  private Config config;

  /**
   * the ip address of this job master
   */
  private String masterAddress;

  /**
   * port number of this job master
   */
  private int masterPort;

  /**
   * name of the job this Job Master will manage
   */
  private JobAPI.Job job;

  /**
   * the network object to receive and send messages
   */
  private RRServer rrServer;

  /**
   * the object to monitor workers
   */
  private WorkerMonitor workerMonitor;

  /**
   * a flag to show that whether the job is done
   * when it is converted to true, the job master exits
   */
  private boolean jobCompleted = false;

  /**
   * Job Terminator object.
   * it will the terminate all workers and cleanup job resources.
   */
  private IJobTerminator jobTerminator;

  /**
   * NodeInfo object for Job Master
   * location of Job Master
   */
  private JobMasterAPI.NodeInfo nodeInfo;

  /**
   * the scaler for the cluster in that Job Master is running
   */
  private IScalerPerCluster clusterScaler;

  /**
   * the driver object
   */
  private IDriver driver;

  /**
   * host address of Dashboard server
   * if it is set, job master will report to Dashboard
   * otherwise, it will ignore Dashboard
   */
  private String dashboardHost;

  /**
   * the client that will handle Dashboard messaging
   */
  private DashboardClient dashClient;

  /**
   * WorkerHandler object to communicate with workers
   */
  private WorkerHandler workerHandler;

  /**
   * PingHandler object
   */
  private PingHandler pingHandler;

  /**
   * BarrierHandler object
   */
  private BarrierHandler barrierHandler;

  /**
   * JobMaster to ZooKeeper connection
   * TODO: need to close zk connection, need to integrate with zk-con in scaler, job terminator.
   */
  private ZKMasterController zkMasterController;

  /**
   * a variable that shows whether JobMaster will run jobTerminate
   * when it is killed with a shutdown hook
   */
  private boolean clearResourcesWhenKilled;

  private CheckpointManager checkpointManager;

  private JobMasterAPI.JobMasterState initialState;

  /**
   * JobMaster constructor
   *
   * @param config configuration
   * @param masterAddress master host
   * @param port the port number
   * @param jobTerminator terminator
   * @param job the job in proto format
   * @param nodeInfo node info of master
   */
  public JobMaster(Config config,
                   String masterAddress,
                   int port,
                   IJobTerminator jobTerminator,
                   JobAPI.Job job,
                   JobMasterAPI.NodeInfo nodeInfo,
                   IScalerPerCluster clusterScaler,
                   JobMasterAPI.JobMasterState initialState) {
    this.config = config;
    this.masterAddress = masterAddress;
    this.jobTerminator = jobTerminator;
    this.job = job;
    this.nodeInfo = nodeInfo;
    this.masterPort = port;
    this.clusterScaler = clusterScaler;
    this.initialState = initialState;

    this.dashboardHost = JobMasterContext.dashboardHost(config);
    if (dashboardHost == null) {
      LOG.warning("Dashboard host address is null. Not connecting to Dashboard");
      this.dashClient = null;
    } else {
      this.dashClient = new DashboardClient(dashboardHost, job.getJobId());
    }
  }

  /**
   * JobMaster constructor to create a job master, the port of job master is read from config
   * file
   *
   * @param config configuration
   * @param masterAddress master host
   * @param jobTerminator terminator
   * @param job the job in proto format
   * @param nodeInfo node info of master
   */
  public JobMaster(Config config,
                   String masterAddress,
                   IJobTerminator jobTerminator,
                   JobAPI.Job job,
                   JobMasterAPI.NodeInfo nodeInfo,
                   IScalerPerCluster clusterScaler,
                   JobMasterAPI.JobMasterState initialState) {

    this(config, masterAddress, JobMasterContext.jobMasterPort(config),
        jobTerminator, job, nodeInfo, clusterScaler, initialState);
  }


  /**
   * initialize the Job Master
   */
  private void init() {

    looper = new Progress();

    // if Dashboard is used, register this job with that
    if (dashClient != null) {
      boolean registered = dashClient.registerJob(job, nodeInfo);
      if (!registered) {
        LOG.warning("Not using Dashboard since it can not register with it.");
        dashClient = null;
      }
    }

    ServerConnectHandler connectHandler = new ServerConnectHandler();
    rrServer =
        new RRServer(config, masterAddress, masterPort, looper, JOB_MASTER_ID, connectHandler);

    // init Driver if it exists
    // this ha to be done before WorkerMonitor initialization
    initDriver();

    boolean faultTolerant = FaultToleranceContext.faultTolerant(config);
    workerMonitor = new WorkerMonitor(this, rrServer, dashClient, job, driver, faultTolerant);

    workerHandler = new WorkerHandler(workerMonitor, rrServer);
    pingHandler = new PingHandler(workerMonitor, rrServer, dashClient);
    barrierHandler = new BarrierHandler(workerMonitor, rrServer);

    // if ZoKeeper server is used for this job, initialize that
    initZKMasterController(workerMonitor);

    JobMasterAPI.Ping.Builder pingBuilder = JobMasterAPI.Ping.newBuilder();

    JobMasterAPI.RegisterWorker.Builder registerWorkerBuilder =
        JobMasterAPI.RegisterWorker.newBuilder();
    JobMasterAPI.RegisterWorkerResponse.Builder registerWorkerResponseBuilder
        = JobMasterAPI.RegisterWorkerResponse.newBuilder();

    JobMasterAPI.WorkerStateChange.Builder stateChangeBuilder =
        JobMasterAPI.WorkerStateChange.newBuilder();
    JobMasterAPI.WorkerStateChangeResponse.Builder stateChangeResponseBuilder
        = JobMasterAPI.WorkerStateChangeResponse.newBuilder();

    ListWorkersRequest.Builder listWorkersBuilder = ListWorkersRequest.newBuilder();
    ListWorkersResponse.Builder listResponseBuilder = ListWorkersResponse.newBuilder();
    JobMasterAPI.BarrierRequest.Builder barrierRequestBuilder =
        JobMasterAPI.BarrierRequest.newBuilder();
    JobMasterAPI.BarrierResponse.Builder barrierResponseBuilder =
        JobMasterAPI.BarrierResponse.newBuilder();

    JobMasterAPI.WorkersScaled.Builder scaledMessageBuilder =
        JobMasterAPI.WorkersScaled.newBuilder();

    JobMasterAPI.DriverMessage.Builder driverMessageBuilder =
        JobMasterAPI.DriverMessage.newBuilder();

    JobMasterAPI.WorkerMessage.Builder workerMessageBuilder =
        JobMasterAPI.WorkerMessage.newBuilder();
    JobMasterAPI.WorkerMessageResponse.Builder workerResponseBuilder
        = JobMasterAPI.WorkerMessageResponse.newBuilder();

    JobMasterAPI.WorkersJoined.Builder joinedBuilder = JobMasterAPI.WorkersJoined.newBuilder();

    rrServer.registerRequestHandler(pingBuilder, pingHandler);

    rrServer.registerRequestHandler(registerWorkerBuilder, workerHandler);
    rrServer.registerRequestHandler(registerWorkerResponseBuilder, workerHandler);

    rrServer.registerRequestHandler(stateChangeBuilder, workerHandler);
    rrServer.registerRequestHandler(stateChangeResponseBuilder, workerHandler);

    rrServer.registerRequestHandler(listWorkersBuilder, workerHandler);
    rrServer.registerRequestHandler(listResponseBuilder, workerHandler);

    rrServer.registerRequestHandler(barrierRequestBuilder, barrierHandler);
    rrServer.registerRequestHandler(barrierResponseBuilder, barrierHandler);

    rrServer.registerRequestHandler(scaledMessageBuilder, workerMonitor);
    rrServer.registerRequestHandler(driverMessageBuilder, workerMonitor);

    rrServer.registerRequestHandler(workerMessageBuilder, workerMonitor);
    rrServer.registerRequestHandler(workerResponseBuilder, workerMonitor);

    rrServer.registerRequestHandler(joinedBuilder, workerMonitor);

    //initialize checkpoint manager
    if (CheckpointingConfigurations.isCheckpointingEnabled(config)) {
      StateStore stateStore = CheckpointUtils.getStateStore(config);
      stateStore.init(config, job.getJobId());
      this.checkpointManager = new CheckpointManager(
          this.rrServer,
          stateStore,
          job.getJobId()
      );
      LOG.info("Checkpoint manager initialized");
      this.checkpointManager.init();
    }
    //done initializing checkpoint manager

    rrServer.start();
    looper.loop();

  }

  /**
   * start the Job Master in a Thread
   */
  public Thread startJobMasterThreaded() {
    // first call the init method
    init();

    // start Driver thread if the driver exists
    startDriverThread();

    Thread jmThread = new Thread() {
      public void run() {
        startLooping();
      }
    };

    jmThread.setName("JM");
    jmThread.start();

    return jmThread;
  }

  /**
   * start the Job Master in a blocking call
   */
  public void startJobMasterBlocking() {
    // first call the init method
    init();

    // start Driver thread if the driver exists
    startDriverThread();

    startLooping();
  }

  /**
   * Job Master loops until all workers in the job completes
   */
  private void startLooping() {
    LOG.info("JobMaster [" + masterAddress + "] started and waiting worker messages on port: "
        + masterPort);

    while (!jobCompleted) {
      looper.loopBlocking();
    }

    // send the remaining messages if any and stop
    rrServer.stopGraceFully(2000);

    if (zkMasterController != null) {
      zkMasterController.close();
    }

    if (jobTerminator != null) {
      jobTerminator.terminateJob(job.getJobName());
    }
  }

  private void initDriver() {

    // if Driver is not set, can not initialize Driver
    if (job.getDriverClassName().isEmpty()) {
      return;
    }

    // first construct the driver
    String driverClass = job.getDriverClassName();
    try {
      Object object = ReflectionUtils.newInstance(driverClass);
      driver = (IDriver) object;
      LOG.info("loaded driver class: " + driverClass);
    } catch (ClassNotFoundException | InstantiationException | IllegalAccessException e) {
      LOG.severe(String.format("failed to load the driver class %s", driverClass));
      throw new RuntimeException(e);
    }
  }

  /**
   * start Driver in a Thread
   */
  public Thread startDriverThread() {

    if (driver == null) {
      return null;
    }

    Thread driverThread = new Thread() {
      public void run() {
        ZKJobUpdater zkJobUpdater = new ZKJobUpdater(config);
        Scaler scaler = new Scaler(job, clusterScaler, workerMonitor, zkJobUpdater);
        DriverMessenger driverMessenger = new DriverMessenger(workerMonitor);
        driver.execute(config, scaler, driverMessenger);
      }
    };
    driverThread.setName("driver");
    driverThread.start();

    return driverThread;
  }

  /**
   * initialize ZKMasterController if ZooKeeper used
   */
  private void initZKMasterController(WorkerMonitor wMonitor) {
    if (ZKContext.isZooKeeperServerUsed(config)) {
      zkMasterController = new ZKMasterController(config, job.getJobName(),
          job.getNumberOfWorkers(), ZKContext.serverAddresses(config));

      try {
        zkMasterController.initialize(initialState);
      } catch (Exception e) {
        throw new Twister2RuntimeException(e);
      }

      zkMasterController.addFailureListener(wMonitor);
      zkMasterController.addWorkerStatusListener(wMonitor);
    }

  }

  /**
   * this method is called when all workers became RUNNING
   * we let the dashboard know that the job STARTED
   */
  public void allWorkersBecameRunning() {
    // if Dashboard is used, tell it that the job has STARTED
    if (dashClient != null) {
      dashClient.jobStateChange(JobState.STARTED);
    }
  }

  /**
   * this method finishes the job
   * It is executed when the worker completed message received from all workers
   */
  public void completeJob(JobState jobFinalState) {

    // if Dashboard is used, tell it that the job has completed or killed
    if (dashClient != null) {
      dashClient.jobStateChange(jobFinalState);
    }

    jobCompleted = true;
    looper.wakeup();
  }

  /**
   * A job can be terminated in two ways:
   * a) successful completion: all workers complete their work and send a COMPLETED message
   * to the Job Master. Job master clears all job resources from the cluster and
   * informs Dashboard. This is handled in the method: completeJob()
   * b) forced killing: user chooses to terminate the job explicitly by executing job kill command
   * not all workers successfully complete in this case.
   * JobMaster does not get COMPLETED message from all workers.
   * So completeJob() method is not called.
   * Instead, the JobMaster pod or the JobMaster process in the submitting client is deleted.
   * ShutDown Hook gets executed.
   * <p>
   * In this case, it can either clear job resources or just send a message to Dashboard
   * based on the parameter that is provided when the shut down hook is registered.
   * If the job master runs in the client, it should clear resources.
   * when the job master runs in the cluster, it should not clear resources.
   * The resources should be cleared by the job killing process.
   */
  public void addShutdownHook(boolean clearJobResourcesOnKill) {
    clearResourcesWhenKilled = clearJobResourcesOnKill;

    Thread hookThread = new Thread() {
      public void run() {

        // if Job completed successfully, do nothing
        if (jobCompleted) {
          return;
        }

        // if Dashboard is used, tell it that the job is killed
        if (dashClient != null) {
          dashClient.jobStateChange(JobState.KILLED);
        }

        if (clearResourcesWhenKilled) {
          jobCompleted = true;
          looper.wakeup();

          if (zkMasterController != null) {
            zkMasterController.close();
          }

          if (jobTerminator != null) {
            jobTerminator.terminateJob(job.getJobName());
          }
        }

      }
    };

    Runtime.getRuntime().addShutdownHook(hookThread);
  }

  public class ServerConnectHandler implements ConnectHandler {
    @Override
    public void onError(SocketChannel channel) {
    }

    @Override
    public void onConnect(SocketChannel channel, StatusCode status) {
      try {
        LOG.fine("Client connected from:" + channel.getRemoteAddress());
      } catch (IOException e) {
        LOG.log(Level.SEVERE, "Exception when getting RemoteAddress", e);
      }
    }

    @Override
    public void onClose(SocketChannel channel) {
    }
  }

}
