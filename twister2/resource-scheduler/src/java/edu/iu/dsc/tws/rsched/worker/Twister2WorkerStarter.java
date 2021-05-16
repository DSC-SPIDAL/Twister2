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
package edu.iu.dsc.tws.rsched.worker;

import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.Context;
import edu.iu.dsc.tws.api.resource.IPersistentVolume;
import edu.iu.dsc.tws.api.resource.ISenderToDriver;
import edu.iu.dsc.tws.api.resource.IVolatileVolume;
import edu.iu.dsc.tws.api.resource.IWorker;
import edu.iu.dsc.tws.api.resource.IWorkerController;
import edu.iu.dsc.tws.api.resource.Twister2Worker;
import edu.iu.dsc.tws.api.resource.WorkerEnvironment;
import edu.iu.dsc.tws.common.util.ReflectionUtils;
import edu.iu.dsc.tws.master.JobMasterContext;
import edu.iu.dsc.tws.proto.system.JobExecutionState;
import edu.iu.dsc.tws.proto.system.job.JobAPI;
import edu.iu.dsc.tws.rsched.core.WorkerRuntime;

/**
 * This class starts Twister2Worker
 */
public class Twister2WorkerStarter implements IWorker {
  private static final Logger LOG = Logger.getLogger(Twister2WorkerStarter.class.getName());

  @Override
  public void execute(Config config, JobAPI.Job job, IWorkerController workerController,
                      IPersistentVolume persistentVolume, IVolatileVolume volatileVolume) {

    int workerID = workerController.getWorkerInfo().getWorkerID();
    WorkerEnvironment workerEnv = WorkerEnvironment.init(
        config, job, workerController, persistentVolume, volatileVolume);

    String workerClass = job.getWorkerClassName();
    Twister2Worker worker;
    try {
      Object object = ReflectionUtils.newInstance(workerClass);
      worker = (Twister2Worker) object;
      //LOG.info("loaded worker class: " + workerClass);
      worker.execute(workerEnv);
    } catch (ClassNotFoundException | InstantiationException | IllegalAccessException e) {
      LOG.severe(String.format("failed to load the worker class %s", workerClass));
      throw new RuntimeException(e);
    }

    //If the execute returns without any errors we assume that the job completed properly
    if (JobMasterContext.isJobMasterUsed(config)
        && !job.getDriverClassName().isEmpty()) {

      ISenderToDriver senderToDriver = WorkerRuntime.getSenderToDriver();
      JobExecutionState.WorkerJobState workerState =
          JobExecutionState.WorkerJobState.newBuilder()
              .setFailure(false)
              .setJobName(config.getStringValue(Context.JOB_ID))
              .setWorkerMessage("Worker Completed")
              .build();
      senderToDriver.sendToDriver(workerState);
    }
  }

}
