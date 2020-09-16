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

import java.util.logging.Logger;

import org.apache.mesos.MesosSchedulerDriver;
import org.apache.mesos.Protos;
import org.apache.mesos.Scheduler;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.scheduler.ILauncher;
import edu.iu.dsc.tws.api.scheduler.Twister2JobState;
import edu.iu.dsc.tws.proto.system.job.JobAPI;

//import org.apache.mesos.v1.scheduler.Scheduler;


/**
 * Launch a topology to mesos cluster
 */
public class MesosLauncher implements ILauncher {


  private MesosController controller;
  private MesosSchedulerDriver driver;

  private static final Logger LOG = Logger.getLogger(MesosLauncher.class.getName());
  private Config config;

  @Override
  public void initialize(Config mconfig) {
    this.config = mconfig;
    this.controller = new MesosController(config);
  }

  @Override
  public void close() {

  }

  @Override
  public boolean killJob(String jobID) {
    //Protos.Status status = driver.stop();
    boolean status;
    if (driver == null) {
      status = true;
      LOG.warning("Job already terminated!");
    } else {
      status = driver.stop() == Protos.Status.DRIVER_STOPPED ? false : true;
    }
    return status;
  }

  @Override
  public Twister2JobState launch(JobAPI.Job job) {
    Twister2JobState state = new Twister2JobState(false);

//    runFramework(MesosContext.getMesosMasterUri(config), job.getJobName());
    runFramework(MesosContext.getMesosMasterUri(config), job);
    //TODO when to return true?
    return state;
  }

  private void runFramework(String mesosMaster, JobAPI.Job job) {

    Scheduler scheduler = new MesosScheduler(controller, config, job);
    driver = new MesosSchedulerDriver(scheduler, controller.getFrameworkInfo(),
        mesosMaster);
    int status = driver.run() == Protos.Status.DRIVER_STOPPED ? 0 : 1;

    LOG.warning("Job already terminated!");
    //driver.stop();
    //System.exit(status);
  }

}
