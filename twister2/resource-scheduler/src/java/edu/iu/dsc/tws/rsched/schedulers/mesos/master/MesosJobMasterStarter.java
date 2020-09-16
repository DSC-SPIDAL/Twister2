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
package edu.iu.dsc.tws.rsched.schedulers.mesos.master;

import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.file.Paths;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.Context;
import edu.iu.dsc.tws.api.driver.IScalerPerCluster;
import edu.iu.dsc.tws.api.driver.NullScaler;
import edu.iu.dsc.tws.common.config.ConfigLoader;
import edu.iu.dsc.tws.common.zk.ZKJobMasterRegistrar;
import edu.iu.dsc.tws.master.JobMasterContext;
import edu.iu.dsc.tws.master.server.JobMaster;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI;
import edu.iu.dsc.tws.proto.system.job.JobAPI;
import edu.iu.dsc.tws.rsched.schedulers.mesos.MesosContext;
import edu.iu.dsc.tws.rsched.schedulers.mesos.MesosWorkerLogger;
import edu.iu.dsc.tws.rsched.schedulers.mesos.driver.MesosScaler;
import edu.iu.dsc.tws.rsched.utils.JobUtils;

//import edu.iu.dsc.tws.rsched.bootstrap.ZKJobMasterRegistrar;


public final class MesosJobMasterStarter {

  private static final Logger LOG = Logger.getLogger(MesosJobMasterStarter.class.getName());

  private MesosJobMasterStarter() {
  }

  public static void main(String[] args) {
    // we can not initialize the logger fully yet,
    // but we need to set the format as the first thing

    String homeDir = System.getenv("HOME");
    int workerId = Integer.parseInt(System.getenv("WORKER_ID"));
    String jobName = System.getenv("JOB_NAME");
    String jobId = System.getenv("JOB_ID");

    String twister2Home = Paths.get("").toAbsolutePath().toString();
    String configDir = "twister2-job";
    Config config = ConfigLoader.loadConfig(twister2Home, configDir, "mesos");
    Config.Builder builder = Config.newBuilder().putAll(config);
    builder.put(Context.JOB_ID, jobId);
    config = builder.build();
    JobTerminator terminator = new JobTerminator(config, System.getenv("FRAMEWORK_ID"));

    MesosWorkerLogger logger = new MesosWorkerLogger(config,
        "/persistent-volume/logs", "master");
    logger.initLogging();

    edu.iu.dsc.tws.rsched.schedulers.mesos.MesosController controller;
    controller = new edu.iu.dsc.tws.rsched.schedulers.mesos.MesosController(config);
    JobAPI.Job job = JobUtils.readJobFile("twister2-job/" + jobName + ".job");
//    try {
//      workerController = new MesosWorkerController(config, job,
//          Inet4Address.getLocalHost().getHostAddress(), 2023, workerId);
//      LOG.info("Initializing with zookeeper");
//      workerController.initializeWithZooKeeper();
//      LOG.info("Waiting for all workers to join");
//      workerController.getAllWorkers(
//          ZKContext.maxWaitTimeForAllWorkersToJoin(config));
//      LOG.info("Everyone has joined");
////      //container.execute(worker.config, id, null, workerController, null);
//
//
//    } catch (Exception e) {
//      LOG.severe("Error " + e.getMessage());
//    }

    //this block is for ZKjobmaster register
    ZKJobMasterRegistrar registrar = null;
    try {
      registrar = new ZKJobMasterRegistrar(config,
          Inet4Address.getLocalHost().getHostAddress(), 11011, job.getJobId());
      LOG.info("JobMaster REGISTERED..:" + Inet4Address.getLocalHost().getHostAddress());
    } catch (UnknownHostException e) {
      LOG.info("JobMaster CAN NOT BE REGISTERED:");
      e.printStackTrace();
    }
    boolean initialized = registrar.initialize();
    if (!initialized) {
      LOG.info("CAN NOT INITIALIZE");
    }
    if (!initialized && registrar.sameZNodeExist()) {
      registrar.deleteJobMasterZNode();
      registrar.initialize();
    }
    //end ZK job master register

    if (!JobMasterContext.jobMasterRunsInClient(config)) {
      JobMaster jobMaster;
      try {
        String workerIp = Inet4Address.getLocalHost().getHostAddress();
        JobMasterAPI.NodeInfo jobMasterNodeInfo = MesosContext.getNodeInfo(config, workerIp);
        IScalerPerCluster clusterScaler = new NullScaler();
        MesosScaler mesosScaler = new MesosScaler(config, job, controller);
        mesosScaler.setFrameWorkId(System.getenv("FRAMEWORK_ID"));
        JobMasterAPI.JobMasterState initialState = JobMasterAPI.JobMasterState.JM_STARTED;

        //JobMaster.jobID = jobId;
        jobMaster =
            new JobMaster(config, InetAddress.getLocalHost().getHostAddress(),
                terminator, job, jobMasterNodeInfo, clusterScaler, initialState);
        //jobMaster.jobId = jobId;
        LOG.info("JobMaster host address...:" + InetAddress.getLocalHost().getHostAddress());
        jobMaster.startJobMasterBlocking();
        //jobMaster.startJobMasterThreaded();
      } catch (Exception e) {
        LOG.log(Level.SEVERE, "Exception when getting local host address: ", e);
      }
    }

    waitIndefinitely();
    registrar.deleteJobMasterZNode();
    registrar.close();
  }

  /**
   * a method to make the job master wait indefinitely
   */
  public static void waitIndefinitely() {

    while (true) {
      try {
        LOG.info("JobMasterStarter thread waiting indefinitely. Sleeping 100sec. "
            + "Time: " + new java.util.Date());
        Thread.sleep(100000);
      } catch (InterruptedException e) {
        LOG.warning("Thread sleep interrupted.");
      }
    }
  }

}
