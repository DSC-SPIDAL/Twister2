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
package edu.iu.dsc.tws.rsched.schedulers.mesos;

import java.net.Inet4Address;
import java.nio.file.Paths;
import java.util.Map;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.resource.IPersistentVolume;
import edu.iu.dsc.tws.api.resource.IWorker;
import edu.iu.dsc.tws.api.resource.IWorkerController;
import edu.iu.dsc.tws.common.config.ConfigLoader;
import edu.iu.dsc.tws.common.util.ReflectionUtils;
import edu.iu.dsc.tws.common.zk.ZKJobMasterFinder;
import edu.iu.dsc.tws.master.worker.JMWorkerAgent;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI;
import edu.iu.dsc.tws.proto.system.job.JobAPI;
import edu.iu.dsc.tws.rsched.utils.JobUtils;

//import java.util.ArrayList;
//import java.util.List;
//import edu.iu.dsc.tws.rsched.bootstrap.ZKJobMasterFinder;

public class MesosDockerWorker {

  public static final Logger LOG = Logger.getLogger(MesosDockerWorker.class.getName());
  public static JMWorkerAgent jobMasterAgent;
  private static Config config;
  private static String jobID;
  private static int startingPort = 30000;
  private static int resourceIndex = 0;
  private static int workerId = 0;

  public static void main(String[] args) throws Exception {

    //gets the docker home directory
    //String homeDir = System.getenv("HOME");

    workerId = Integer.parseInt(System.getenv("WORKER_ID"));
    jobID = System.getenv("JOB_NAME");
    MesosDockerWorker worker = new MesosDockerWorker();

    String twister2Home = Paths.get("").toAbsolutePath().toString();
    String configDir = "twister2-job";
    config = ConfigLoader.loadConfig(twister2Home, configDir, "mesos");

    resourceIndex = Integer.parseInt(System.getenv("COMPUTE_RESOURCE_INDEX"));

    MesosWorkerLogger logger = new MesosWorkerLogger(config,
        "/persistent-volume/logs", "worker" + workerId);
    logger.initLogging();

    LOG.info("WORKER ID ..:" + workerId);

    Map<String, Integer> additionalPorts =
        MesosWorkerUtils.generateAdditionalPorts(config, startingPort);

    MesosWorkerController workerController = null;

    JobAPI.Job job = JobUtils.readJobFile(null, "twister2-job/"
        + jobID + ".job");
    try {

      JobAPI.ComputeResource computeResource = JobUtils.getComputeResource(job, resourceIndex);

      workerController = new MesosWorkerController(config, job,
          Inet4Address.getLocalHost().getHostAddress(), 2023, workerId, computeResource,
          additionalPorts);

    } catch (Exception e) {
      LOG.severe("Error " + e.getMessage());
    }
    //find the jobmaster
    ZKJobMasterFinder finder = new ZKJobMasterFinder(config, job.getJobId());
    finder.initialize();

    String jobMasterIPandPort = finder.getJobMasterIPandPort();
    if (jobMasterIPandPort == null) {
      LOG.info("Job Master has not joined yet. Will wait and try to get the address ...");
      jobMasterIPandPort = finder.waitAndGetJobMasterIPandPort(20000);
      LOG.info("Job Master address: " + jobMasterIPandPort);
    } else {
      LOG.info("Job Master address: " + jobMasterIPandPort);
    }

    finder.close();

    String jobMasterPortStr = jobMasterIPandPort.substring(jobMasterIPandPort.lastIndexOf(":") + 1);
    int jobMasterPort = Integer.parseInt(jobMasterPortStr);
    String jobMasterIP = jobMasterIPandPort.substring(0, jobMasterIPandPort.lastIndexOf(":"));
    //LOG.info("JobMaster IP..: " + jobMasterIP);
    //LOG.info("Worker ID..: " + workerId);
    //StringBuilder outputBuilder = new StringBuilder();
    //int workerCount = workerController.getNumberOfWorkers();
    int workerCount = job.getNumberOfWorkers();
    LOG.info("Worker Count..: " + workerCount);

    LOG.info(workerController.getWorkerInfo().toString());
    //start job master client

    worker.startJobMasterAgent(workerController.getWorkerInfo(), jobMasterIP, jobMasterPort,
        workerCount);

    config = JobUtils.overrideConfigs(job, config);
    config = JobUtils.updateConfigs(job, config);

    startWorker(workerController, null);

    try {
      Thread.sleep(3000);
    } catch (InterruptedException e) {
      LOG.info("sleep exception" + e.getMessage());
    }

    closeWorker();
  }

  public static void startWorker(IWorkerController workerController,
                                 IPersistentVolume pv) {

    JobAPI.Job job = JobUtils.readJobFile(null, "twister2-job/" + jobID + ".job");
    String workerClass = job.getWorkerClassName();
    LOG.info("Worker class---->>>" + workerClass);
    IWorker worker;
    try {
      Object object = ReflectionUtils.newInstance(workerClass);
      worker = (IWorker) object;
      LOG.info("Loaded worker class..: " + workerClass);
    } catch (ClassNotFoundException | InstantiationException | IllegalAccessException e) {
      LOG.severe(String.format("Failed to load the worker class %s", workerClass));
      throw new RuntimeException(e);
    }

    MesosVolatileVolume volatileVolume = null;
    //TODO method SchedulerContext.volatileDiskRequested deleted
    //volatileVolume needs to be checked from job object
//    if (SchedulerContext.volatileDiskRequested(config)) {
//      volatileVolume =
//          new MesosVolatileVolume(SchedulerContext.jobID(config), workerID);
//    }

    // lets create the resource plan
    //Map<Integer, String> processNames = MPIWorker.createResourcePlan(config);
    // now create the resource plan
    //AllocatedResources resourcePlan = MPIWorker.addContainers(config, processNames);
//    AllocatedResources resourcePlan = MesosWorkerUtils.createAllocatedResources("mesos",
//        workerID, job);
    //resourcePlan = new AllocatedResources(SchedulerContext.clusterType(config), workerID);
    worker.execute(config, workerId, jobMasterAgent.getJMWorkerController(),
        pv, volatileVolume);
  }

  /**
   * last method to call to close the worker
   */
  public static void closeWorker() {

    // send worker completed message to the Job Master and finish
    // Job master will delete the StatefulSet object
    jobMasterAgent.sendWorkerCompletedMessage();
    jobMasterAgent.close();
  }

  public void startJobMasterAgent(JobMasterAPI.WorkerInfo workerInfo, String jobMasterIP,
                                  int jobMasterPort, int numberOfWorkers) {

    LOG.info("JobMaster IP..: " + jobMasterIP);
    LOG.info("NETWORK INFO..: " + workerInfo.getWorkerIP());

    //TODO: should be either WorkerState.STARTED or WorkerState.RESTARTED
    JobMasterAPI.WorkerState initialState = JobMasterAPI.WorkerState.STARTED;

    jobMasterAgent = JMWorkerAgent.createJMWorkerAgent(config, workerInfo, jobMasterIP,
        jobMasterPort, numberOfWorkers, initialState);
    jobMasterAgent.startThreaded();

    // No need for sending workerStarting message anymore
    // that is called in startThreaded method
  }

}
