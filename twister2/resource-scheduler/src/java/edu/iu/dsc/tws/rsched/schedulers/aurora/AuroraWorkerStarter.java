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
package edu.iu.dsc.tws.rsched.schedulers.aurora;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.file.Paths;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.Context;
import edu.iu.dsc.tws.api.config.SchedulerContext;
import edu.iu.dsc.tws.api.exceptions.TimeoutException;
import edu.iu.dsc.tws.api.resource.IWorker;
import edu.iu.dsc.tws.common.config.ConfigLoader;
import edu.iu.dsc.tws.common.util.ReflectionUtils;
import edu.iu.dsc.tws.common.zk.ZKWorkerController;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI;
import edu.iu.dsc.tws.proto.system.job.JobAPI;
import edu.iu.dsc.tws.proto.utils.NodeInfoUtils;
import edu.iu.dsc.tws.proto.utils.WorkerInfoUtils;
import edu.iu.dsc.tws.rsched.utils.JobUtils;
import static edu.iu.dsc.tws.api.config.Context.JOB_ARCHIVE_DIRECTORY;

public final class AuroraWorkerStarter {
  public static final Logger LOG = Logger.getLogger(AuroraWorkerStarter.class.getName());

  private InetAddress workerAddress;
  private int workerPort;
  private String mesosTaskID;
  private Config config;
  private JobAPI.Job job;
  private ZKWorkerController zkWorkerController;

  private AuroraWorkerStarter() {
  }

  public static void main(String[] args) {

    // create the worker
    AuroraWorkerStarter workerStarter = createAuroraWorker();

    // get the number of workers from some where
    // wait for all of them
    // print their list and exit
    workerStarter.waitAndGetAllWorkers();

    String workerClass = SchedulerContext.workerClass(workerStarter.config);
    IWorker worker;
    try {
      Object object = ReflectionUtils.newInstance(workerClass);
      worker = (IWorker) object;
      LOG.info("loaded worker class: " + workerClass);
    } catch (ClassNotFoundException | InstantiationException | IllegalAccessException e) {
      LOG.log(Level.SEVERE, String.format("failed to load the worker class %s",
          workerClass), e);
      throw new RuntimeException(e);
    }

    // TODO: need to provide all parameters
    worker.execute(workerStarter.config,
        workerStarter.job,
        null, null, null);

    // close the things, let others know that it is done
    workerStarter.close();
  }

  /**
   * create a AuroraWorkerStarter object by getting values from system property
   */
  public static AuroraWorkerStarter createAuroraWorker() {
    AuroraWorkerStarter workerStarter = new AuroraWorkerStarter();
    String hostname = System.getProperty("hostname");
    String portStr = System.getProperty("tcpPort");
    workerStarter.mesosTaskID = System.getProperty("taskID");
    try {
      workerStarter.workerAddress = InetAddress.getByName(hostname);
      workerStarter.workerPort = Integer.parseInt(portStr);
      LOG.log(Level.INFO, "worker IP: " + hostname + " workerPort: " + portStr);
      LOG.log(Level.INFO, "worker mesosTaskID: " + workerStarter.mesosTaskID);
    } catch (UnknownHostException e) {
      LOG.log(Level.SEVERE, "worker ip address is not valid: " + hostname, e);
      throw new RuntimeException(e);
    }

    // read job description file
    workerStarter.readJobDescFile();
    logJobInfo(workerStarter.job);

    // load config files
    workerStarter.loadConfig();
    LOG.fine("Config from files: \n" + workerStarter.config.toString());

    // override config files with values from job config if any
    workerStarter.overrideConfigsFromJob();

    // get unique workerID and let other workers know about this worker in the job
    workerStarter.initializeWithZooKeeper();

    return workerStarter;
  }

  /**
   * read job description file and construct job object
   */
  private void readJobDescFile() {
    String jobDescFile = System.getProperty(SchedulerContext.JOB_DESCRIPTION_FILE_CMD_VAR);
    jobDescFile = JOB_ARCHIVE_DIRECTORY + "/" + jobDescFile;
    job = JobUtils.readJobFile(jobDescFile);

    // printing for testing
    LOG.log(Level.INFO, "Job description file is read: " + jobDescFile);
  }

  /**
   * loadConfig from config files
   */
  public void loadConfig() {

    // first lets read the essential properties from java system properties
    String twister2Home = Paths.get("").toAbsolutePath().toString();
    String clusterType = System.getProperty(SchedulerContext.CLUSTER_TYPE);
    String configDir = twister2Home + "/" + JOB_ARCHIVE_DIRECTORY + "/" + clusterType;

    LOG.log(Level.INFO, String.format("Loading configuration with twister2_home: %s and "
        + "configuration: %s", twister2Home, configDir));
    Config conf = ConfigLoader.loadConfig(twister2Home, JOB_ARCHIVE_DIRECTORY, clusterType);
    config = Config.newBuilder().
        putAll(conf).
        put(Context.TWISTER2_HOME.getKey(), twister2Home).
        put(Context.TWISTER2_CONF.getKey(), configDir).
        put(Context.TWISTER2_CLUSTER_TYPE, clusterType).
        build();

    LOG.log(Level.INFO, "Config files are read from directory: " + configDir);
  }

  /**
   * configs from job object will override the ones in config from files if any
   */
  public void overrideConfigsFromJob() {

    Config.Builder builder = Config.newBuilder().putAll(config);

    JobAPI.Config conf = job.getConfig();
    LOG.log(Level.INFO, "Number of configs to override from job conf: " + conf.getKvsCount());

    for (JobAPI.Config.KeyValue kv : conf.getKvsList()) {
      builder.put(kv.getKey(), kv.getValue());
      LOG.log(Level.INFO, "Overriden conf key-value pair: " + kv.getKey() + ": " + kv.getValue());
    }

    config = builder.build();
  }

  public void initializeWithZooKeeper() {

    long startTime = System.currentTimeMillis();

    // TODO: need to put at least nodeIP to this NodeInfoUtils object
    JobMasterAPI.NodeInfo nodeInfo = NodeInfoUtils.createNodeInfo(null, null, null);
    //TODO: workerID has to be assigned properly to work it correctly
    int workerID = 0;
    JobMasterAPI.WorkerInfo workerInfo = WorkerInfoUtils.createWorkerInfo(
        workerID, workerAddress.getHostAddress(), workerPort, nodeInfo);
    zkWorkerController =
        new ZKWorkerController(config, job.getJobId(), job.getNumberOfWorkers(), workerInfo);
    try {
      int restartCount = 0;
      // startTime should come from job submission client
      zkWorkerController.initialize(restartCount, startTime);
    } catch (Exception e) {
      LOG.log(Level.SEVERE, e.getMessage(), e);
    }
    long duration = System.currentTimeMillis() - startTime;
    LOG.info("Initialization for the worker: " + zkWorkerController.getWorkerInfo()
        + " took: " + duration + "ms");
  }

  /**
   * needs to close down when finished computation
   */
  public void waitAndGetAllWorkers() {
    int numberOfWorkers = job.getNumberOfWorkers();
    LOG.info("Waiting for " + numberOfWorkers + " workers to join .........");

    long startTime = System.currentTimeMillis();
    List<JobMasterAPI.WorkerInfo> workerList = null;
    try {
      workerList = zkWorkerController.getAllWorkers();
    } catch (TimeoutException timeoutException) {
      LOG.log(Level.SEVERE, timeoutException.getMessage(), timeoutException);
      return;
    }
    long duration = System.currentTimeMillis() - startTime;

    if (workerList == null) {
      LOG.log(Level.SEVERE, "Could not get full worker list. timeout limit has been reached !!!!");
    } else {
      LOG.log(Level.INFO, "Waited " + duration + " ms for all workers to join.");

      LOG.info("list of all joined workers in the job: "
          + WorkerInfoUtils.workerListAsString(workerList));
    }
  }

  /**
   * needs to close down when finished computation
   */
  public void close() {
    zkWorkerController.close();
  }

  /**
   * a test method to print a job
   */
  public static void logJobInfo(JobAPI.Job job) {
    StringBuffer sb = new StringBuffer("Job Details:");
    sb.append("\nJob name: " + job.getJobName());
    sb.append("\nJob file: " + job.getJobFormat().getJobFile());
    sb.append("\nnumber of workers: " + job.getNumberOfWorkers());
    sb.append("\nCPUs: " + job.getComputeResource(0).getCpu());
    sb.append("\nRAM: " + job.getComputeResource(0).getRamMegaBytes());
    sb.append("\nDisk: " + job.getComputeResource(0).getDiskGigaBytes());

    JobAPI.Config conf = job.getConfig();
    sb.append("\nnumber of key-values in job conf: " + conf.getKvsCount());
    for (JobAPI.Config.KeyValue kv : conf.getKvsList()) {
      sb.append("\n" + kv.getKey() + ": " + kv.getValue());
    }

    LOG.info(sb.toString());
  }
}
