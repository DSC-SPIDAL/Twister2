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

import edu.iu.dsc.tws.api.JobConfig;
import edu.iu.dsc.tws.api.Twister2Job;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.SchedulerContext;
import edu.iu.dsc.tws.proto.system.job.JobAPI;
import edu.iu.dsc.tws.rsched.core.ResourceAllocator;
import edu.iu.dsc.tws.rsched.job.Twister2Submitter;
import edu.iu.dsc.tws.rsched.utils.JobUtils;

public final class BasicAuroraJob {
  private BasicAuroraJob() {
  }

  public static void main(String[] args) {

    // first load the configurations from command line and config files
    Config config = ResourceAllocator.loadConfig(new HashMap<>());
    System.out.println("read config values: " + config.size());
    System.out.println(config);

    // build JobConfig
    HashMap<String, Object> configurations = new HashMap<>();
    configurations.put(SchedulerContext.THREADS_PER_WORKER, 8);

    JobConfig jobConfig = new JobConfig();
    jobConfig.putAll(configurations);

    // build the job
    Twister2Job twister2Job = Twister2Job.loadTwister2Job(config, jobConfig);

    // now submit the job
    Twister2Submitter.submitJob(twister2Job, config);

    // now terminate the job
    terminateJob(config);
//    jobWriteTest(twister2Job);
//    jobReadTest();
  }

  /**
   * wait some time and terminate the job
   */
  public static void terminateJob(Config config) {

    long waitTime = 100000;
    try {
      System.out.println("Waiting " + waitTime + " ms. Will terminate the job afterward .... ");
      Thread.sleep(waitTime);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }

    String jobName = "basic-aurora";
    Twister2Submitter.terminateJob(jobName, config);
  }

  /**
   * test method for Twister2Job write to a file
   */
  public static void jobWriteTest(Twister2Job twister2Job) {
    String file = "testJobFile";
    JobUtils.writeJobFile(twister2Job.serialize(), file);
  }

  /**
   * test method to read Twister2Job file
   */
  public static void jobReadTest() {
    String fl = "/tmp/basic-aurora/basic-aurora3354891958097304472/twister2-core/basic-aurora.job";
    JobAPI.Job job = JobUtils.readJobFile(null, fl);
    System.out.println("job name: " + job.getJobName());
    System.out.println("job worker class name: " + job.getWorkerClassName());
    System.out.println("job workers: " + job.getNumberOfWorkers());
    System.out.println("CPUs: " + job.getComputeResource(0).getCpu());
    System.out.println("RAM: " + job.getComputeResource(0).getRamMegaBytes());
    System.out.println("Disk: " + job.getComputeResource(0).getDiskGigaBytes());
    JobAPI.Config conf = job.getConfig();
    System.out.println("number of key-values in job conf: " + conf.getKvsCount());

    for (JobAPI.Config.KeyValue kv : conf.getKvsList()) {
      System.out.println(kv.getKey() + ": " + kv.getValue());
    }

  }

}
