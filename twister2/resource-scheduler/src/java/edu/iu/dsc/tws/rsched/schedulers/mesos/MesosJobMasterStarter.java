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
import java.net.InetAddress;
import java.nio.file.Paths;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.common.config.Config;
import edu.iu.dsc.tws.common.config.ConfigLoader;
import edu.iu.dsc.tws.master.JobMaster;
import edu.iu.dsc.tws.master.JobMasterContext;
import edu.iu.dsc.tws.proto.system.job.JobAPI;
import edu.iu.dsc.tws.rsched.bootstrap.ZKContext;
import edu.iu.dsc.tws.rsched.utils.JobUtils;


public final class MesosJobMasterStarter {
  private static final Logger LOG = Logger.getLogger(MesosJobMasterStarter.class.getName());
  private Config config;
  private String jobName;

  private MesosJobMasterStarter() { }

  public static void main(String[] args) {
    // we can not initialize the logger fully yet,
    // but we need to set the format as the first thing
    String homeDir = System.getenv("HOME");
    int workerId = Integer.parseInt(System.getenv("WORKER_ID"));
    String jobName = System.getenv("JOB_NAME");
    int id = workerId;
    //MesosJobMasterStarter worker = new MesosJobMasterStarter();

    String twister2Home = Paths.get("").toAbsolutePath().toString();
    String configDir = "twister2-job/mesos/";
    Config config = ConfigLoader.loadConfig(twister2Home, configDir);


    MesosWorkerLogger logger = new MesosWorkerLogger(config,
        "/persistent-volume/logs", "master");
    logger.initLogging();



    MesosWorkerController workerController = null;
    try {
      JobAPI.Job job = JobUtils.readJobFile(null, "twister2-job/"
          + jobName + ".job");
      workerController = new MesosWorkerController(config, job,
          Inet4Address.getLocalHost().getHostAddress(), 22, id);
      LOG.info("Initializing with zookeeper");
      workerController.initializeWithZooKeeper();
      LOG.info("Waiting for all workers to join");
      workerController.waitForAllWorkersToJoin(
          ZKContext.maxWaitTimeForAllWorkersToJoin(config));
      LOG.info("Everyone has joined");
      //container.init(worker.config, id, null, workerController, null);

    } catch (Exception e) {
      e.printStackTrace();
    }



    if (JobMasterContext.jobMasterRunsInClient(config)) {
      JobMaster jobMaster = null;
      try {
        jobMaster =
            new JobMaster(config, InetAddress.getLocalHost().getHostAddress(),
                null, jobName);
        LOG.info("JobMaster host address:" + InetAddress.getLocalHost().getHostAddress());
        jobMaster.init();
      } catch (Exception e) {
        LOG.log(Level.SEVERE, "Exception when getting local host address: ", e);
      }
    }




    try {
      Thread.sleep(50000);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }

    waitIndefinitely();
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
