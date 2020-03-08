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
package edu.iu.dsc.tws.examples.internal.rsched;

import java.util.HashMap;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.JobConfig;
import edu.iu.dsc.tws.api.Twister2Job;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.SchedulerContext;
import edu.iu.dsc.tws.rsched.core.ResourceAllocator;
import edu.iu.dsc.tws.rsched.job.Twister2Submitter;

public final class BasicMesosJob {
  private static final Logger LOG = Logger.getLogger(BasicMesosJob.class.getName());

  private BasicMesosJob() {
  }

  public static void main(String[] args) {
    Config config = ResourceAllocator.loadConfig(new HashMap<>());
    System.out.println("read config values: " + config.size());
    System.out.println(config);

    String jobName = SchedulerContext.jobName(config);
    jobName += "-" + System.currentTimeMillis();
    System.out.println("job name is " + jobName);
    // build JobConfig
    HashMap<String, Object> configurations = new HashMap<>();
    configurations.put(SchedulerContext.THREADS_PER_WORKER, 8);

    JobConfig jobConfig = new JobConfig();
    jobConfig.putAll(configurations);

    // build the job
    Twister2Job twister2Job = Twister2Job.loadTwister2Job(config, jobConfig);
    twister2Job.setJobName(jobName);

    // now submit the job
    Twister2Submitter.submitJob(twister2Job, config);

    System.out.println("now terminating...");
    Twister2Submitter.terminateJob(twister2Job.getJobName(), config);
  }

}
