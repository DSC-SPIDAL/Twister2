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
package edu.iu.dsc.tws.rsched.schedulers.mesos.mpi;

import java.net.Inet4Address;
import java.nio.file.Paths;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.resource.IPersistentVolume;
import edu.iu.dsc.tws.api.resource.IWorker;
import edu.iu.dsc.tws.api.resource.IWorkerController;
import edu.iu.dsc.tws.common.config.ConfigLoader;
import edu.iu.dsc.tws.master.JobMasterContext;
import edu.iu.dsc.tws.master.worker.JMWorkerAgent;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI;
import edu.iu.dsc.tws.proto.system.job.JobAPI;
import edu.iu.dsc.tws.rsched.schedulers.mesos.MesosVolatileVolume;
import edu.iu.dsc.tws.rsched.schedulers.mesos.MesosWorkerController;
import edu.iu.dsc.tws.rsched.schedulers.mesos.MesosWorkerLogger;
import edu.iu.dsc.tws.rsched.schedulers.mesos.MesosWorkerUtils;
import edu.iu.dsc.tws.rsched.utils.JobUtils;

import mpi.MPI;
import mpi.MPIException;

public final class MesosMPIWorkerStarter {

  public static final Logger LOG = Logger.getLogger(MesosMPIWorkerStarter.class.getName());
  private static Config config;
  private static String jobName;
  private static JMWorkerAgent jobMasterAgent;
  private static int workerID;
  private static int numberOfWorkers;
  private static int resourceIndex = 0;
  private static int startingPort = 30000;

  private MesosMPIWorkerStarter() {
  }

  public static void main(String[] args) {

    try {
      MPI.Init(args);
      workerID = MPI.COMM_WORLD.getRank();
      numberOfWorkers = MPI.COMM_WORLD.getSize();
      System.out.println("Worker ranking..:" + workerID
          + " Number of workers..:" + numberOfWorkers);

    } catch (MPIException e) {
      LOG.log(Level.SEVERE, "Could not get rank or size from mpi.COMM_WORLD", e);
      throw new RuntimeException(e);
    }

    jobName = args[0];

    String twister2Home = Paths.get("").toAbsolutePath().toString();
    String configDir = "twister2-job";
    config = ConfigLoader.loadConfig(twister2Home, configDir, "mesos");

    MesosWorkerLogger logger = new MesosWorkerLogger(config,
        "/persistent-volume/logs", "worker" + workerID);
    logger.initLogging();

    MesosWorkerController workerController = null;
    //List<WorkerNetworkInfo> workerNetworkInfoList = new ArrayList<>();

    Map<String, Integer> additionalPorts =
        MesosWorkerUtils.generateAdditionalPorts(config, startingPort);

    try {
      JobAPI.Job job = JobUtils.readJobFile("twister2-job/" + jobName + ".job");

      // add any configuration from job file to the config object
      // if there are the same config parameters in both,
      // job file configurations will override
      config = JobUtils.overrideConfigs(job, config);
      config = JobUtils.updateConfigs(job, config);
      //this will change to get proper resource index.
      JobAPI.ComputeResource computeResource = JobUtils.getComputeResource(job, resourceIndex);
      LOG.info("in worker starter...... job worker count:" + job.getNumberOfWorkers());
      workerController = new MesosWorkerController(config, job,
          Inet4Address.getLocalHost().getHostAddress(), 2023, workerID, computeResource,
          additionalPorts);
      workerController.initializeWithZooKeeper();
    } catch (Exception e) {
      LOG.severe("Error " + e.getMessage());
    }

    //can not access docker env variable so it was passed as a parameter
    String jobMasterIP = args[1];
    LOG.info("JobMaster IP..: " + jobMasterIP);
    LOG.info("Worker ID..: " + workerID);
    int jobMasterPort = JobMasterContext.jobMasterPort(config);
    startJobMasterAgent(
        workerController.getWorkerInfo(), jobMasterIP, jobMasterPort);

    LOG.info("\nWorker Controller\nWorker ID..: "
        + workerController.getWorkerInfo().getWorkerID()
        + "\nIP address..: " + workerController.getWorkerInfo().getWorkerIP());

    startWorker(workerController, null);
    try {
      Thread.sleep(2000);
    } catch (InterruptedException e) {
      LOG.info("sleep exception" + e.getMessage());
    }
    try {
      MPI.Finalize();
    } catch (MPIException ignore) {
      LOG.info("MPI Finalize Exception" + ignore.getMessage());
    }

    closeWorker();
    //workerController.close();
  }

  public static void startJobMasterAgent(JobMasterAPI.WorkerInfo workerInfo, String jobMasterIP,
                                         int jobMasterPort) {

    LOG.info("JobMaster IP..: " + jobMasterIP);
    LOG.info("NETWORK INFO..: " + workerInfo.getWorkerIP().toString());

    //TODO: zero means starting for the first time
    int restartCount = 0;

    jobMasterAgent = JMWorkerAgent.createJMWorkerAgent(config, workerInfo, jobMasterIP,
        jobMasterPort, numberOfWorkers, restartCount);
    jobMasterAgent.startThreaded();
    // No need for sending workerStarting message anymore
    // that is called in startThreaded method
  }

  public static void startWorker(IWorkerController workerController,
                                 IPersistentVolume pv) {


    JobAPI.Job job = JobUtils.readJobFile("twister2-job/" + jobName + ".job");

    MesosVolatileVolume volatileVolume = null;
    //TODO method SchedulerContext.volatileDiskRequested deleted
    //volatileVolume needs to be checked from job object
//    if (SchedulerContext.volatileDiskRequested(config)) {
//      volatileVolume =
//          new MesosVolatileVolume(SchedulerContext.jobName(config), workerID);
//    }

    // lets create the resource plan
//    Map<Integer, JobMasterAPI.WorkerInfo> processNames =
//        MPIWorker.createResourcePlan(config, MPI.COMM_WORLD, null);
    // now create the resource plan
    //AllocatedResources resourcePlan = MPIWorker.addContainers(config, processNames);
//    AllocatedResources resourcePlan = MesosWorkerUtils.createAllocatedResources("mesos",
//        workerID, job);
    //resourcePlan = new AllocatedResources(SchedulerContext.clusterType(config), workerID);
    IWorker worker = JobUtils.initializeIWorker(job);
    worker.execute(config, job, workerController, pv, volatileVolume);
  }

  /**
   * last method to call to close the worker
   */
  public static void closeWorker() {

    // send worker completed message to the Job Master and finish
    // Job master will delete the StatefulSet object
    jobMasterAgent.sendWorkerCompletedMessage(JobMasterAPI.WorkerState.COMPLETED);
    jobMasterAgent.close();
  }

}
